"""The two-stage fx-rates walk-forward (spec fx-rates/evaluation; design D7, D8).

    python -m src.fx_rates.walk_forward --stage lgbm    prior + LightGBM, every fold, both pairs
    python -m src.fx_rates.walk_forward --stage mlp     5-seed MLP for pairs whose LightGBM cell is KEEP

Model code and artifact storage are fx_session's, imported read-only and driven by THIS family's
config (artifacts under research_models/fx_rates/). Stage 1 records the digests of the fx_session
H1 caches and of the yield caches it read.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ..fx_fib.artifacts import sha256
from ..fx_session import artifacts as AR
from ..fx_session import model_lgbm as LG
from . import config_digest, feature_list_digest, load_config, repo_path
from .registry import assert_config_matches_verdicts, read_registry
from .splits import fold_arrays, make_folds

OOS = {'lgbm': 'oos_lgbm.csv.gz', 'mlp': 'oos_mlp.csv.gz'}
RUN = {'lgbm': 'run_lgbm.json', 'mlp': 'run_mlp.json'}
CARRY = ['y', 'entry_mid', 'exit_mid', 'exit_date', 'yield_stale_flag']


class StageOrderError(RuntimeError):
    pass


def results_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['results'])


def input_digests(cfg: dict) -> dict:
    from .rates_loader import SERIES, cache_path
    out = {f'yield:{s}': sha256(cache_path(s, cfg)) for s in SERIES}
    for p in cfg['pairs']:
        out[f'fx_h1:{p}'] = sha256(os.path.join(repo_path(cfg['paths']['fx_h1']), f'{p}_h1.csv'))
    return out


def load_inputs(cfg: dict) -> dict:
    from .align import load_fx_d1, load_yields
    from .labeling import dataset
    data = dataset(load_fx_d1(cfg), load_yields(cfg), cfg)
    return {p: d[d['reason'] == ''].sort_index() for p, d in data.items()}


def _arrays(rows: pd.DataFrame, cfg: dict):
    return rows[cfg['features']['columns']].to_numpy(dtype=float), rows['y'].to_numpy(dtype=float)


def _last_day(inputs: dict) -> pd.Timestamp:
    return max(d.index.max() for d in inputs.values())


def _write(cfg, stage, frames, run) -> None:
    os.makedirs(results_dir(cfg), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(results_dir(cfg), OOS[stage]), index=False,
                                                compression='gzip')
    with open(os.path.join(results_dir(cfg), RUN[stage]), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(run, fh, indent=1, default=str)


def load_oos(cfg: dict, stage: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), OOS[stage]))
    d['date'] = pd.to_datetime(d['date'])
    return d


def load_run(cfg: dict, stage: str) -> dict:
    with open(os.path.join(results_dir(cfg), RUN[stage]), encoding='utf-8') as fh:
        return json.load(fh)


def run_lgbm(cfg: dict | None = None, now_iso: str | None = None, inputs: dict | None = None,
             log=print, digests: dict | None = None) -> dict:
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if inputs is None:
        inputs = load_inputs(cfg)
        digests = input_digests(cfg)
    last = _last_day(inputs)
    run = {'stage': 'lgbm', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'device': 'cpu (lightgbm)',
           'input_digests': digests or {}, 'pairs': {}}
    out = []
    for pair in cfg['pairs']:
        rows = inputs[pair]
        X, y = _arrays(rows, cfg)
        info = []
        for f in make_folds(rows.index, cfg, last):
            a = fold_arrays(X, y, f)
            model = LG.train_lightgbm(a['X_fit'], a['y_fit'], a['X_val'], a['y_val'], cfg['lightgbm'])
            AR.save(cfg, model, pair, f.k, 'lightgbm', 'lightgbm', now_iso)
            ev = rows.iloc[f.eval]
            d = pd.DataFrame({'pair': pair, 'date': ev.index, 'fold': f.k})
            for c in CARRY:
                d[c] = ev[c].to_numpy()
            d['p_prior'] = LG.prior_up(a['y_fit'])
            d['p_lightgbm'] = LG.proba_up(model, a['X_eval'])
            out.append(d)
            info.append({'k': f.k, 'start': str(f.start.date()), 'end': str(f.end.date()), 'n_fit': len(f.fit),
                         'n_val': len(f.val), 'n_eval': len(f.eval), 'best_iteration': int(model.best_iteration_ or 0)})
            log(f"{pair} fold {f.k} {f.start.date()}..{f.end.date()}: fit {len(f.fit)} val {len(f.val)} "
                f"eval {len(f.eval)} | trees {model.best_iteration_}")
        run['pairs'][pair] = {'folds': info, 'n_eval': int(sum(i['n_eval'] for i in info))}
    _write(cfg, 'lgbm', out, run)
    return run


def keep_pairs(cfg: dict) -> list:
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    missing = [f'{p}_lightgbm_vs_prior' for p in cfg['pairs'] if f'{p}_lightgbm_vs_prior' not in reg]
    if missing:
        raise StageOrderError(f'the MLP stage needs both LightGBM verdicts first; missing {missing}')
    return [p for p in cfg['pairs'] if reg[f'{p}_lightgbm_vs_prior']['verdict'] == 'KEEP']


def run_mlp(cfg: dict | None = None, now_iso: str | None = None, inputs: dict | None = None, log=print) -> dict:
    from ..fx_session import model_mlp as MM
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    pairs = keep_pairs(cfg)
    run = {'stage': 'mlp', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'trained_pairs': pairs, 'pairs': {}}
    if not pairs:
        log('no LightGBM cell is KEEP: the MLP stage trains nothing')
        _write(cfg, 'mlp', [pd.DataFrame(columns=['pair', 'date', 'fold', 'y', 'p_mlp'])], run)
        return run
    MM.pick_device(cfg['mlp'].get('allow_cpu', False))
    inputs = inputs if inputs is not None else load_inputs(cfg)
    last = _last_day(inputs)
    out = []
    for pair in pairs:
        rows = inputs[pair]
        X, y = _arrays(rows, cfg)
        info = []
        for f in make_folds(rows.index, cfg, last):
            a = fold_arrays(X, y, f)
            AR.save(cfg, a['scaler'], pair, f.k, 'scaler', 'scaler', now_iso)
            members, metas = [], []
            for s in cfg['mlp']['seeds']:
                m, meta = MM.train_mlp(a['Xs_fit'], a['y_fit'], a['Xs_val'], a['y_val'], cfg['mlp'], s)
                MM.save_member(cfg, m, meta, pair, f.k, s, now_iso)
                members.append(m)
                metas.append(meta)
            ev = rows.iloc[f.eval]
            out.append(pd.DataFrame({'pair': pair, 'date': ev.index, 'fold': f.k, 'y': ev['y'].to_numpy(),
                                     'p_mlp': MM.ensemble_up(members, a['Xs_eval'], len(cfg['mlp']['seeds']))}))
            info.append({'k': f.k, 'best_epochs': [m['best_epoch'] for m in metas]})
            run.update(device=metas[0]['device'], torch=metas[0]['torch'], cuda=metas[0]['cuda'])
            log(f"{pair} fold {f.k}: mlp best epochs {[m['best_epoch'] for m in metas]}")
        run['pairs'][pair] = {'folds': info}
    _write(cfg, 'mlp', out, run)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description='run one fx-rates walk-forward stage')
    ap.add_argument('--stage', choices=['lgbm', 'mlp'], required=True)
    a = ap.parse_args(argv)
    (run_lgbm if a.stage == 'lgbm' else run_mlp)()


if __name__ == '__main__':
    main()
