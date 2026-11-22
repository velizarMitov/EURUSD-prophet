"""The two-stage fx-ny-volatility walk-forward (spec fx-ny-volatility/evaluation; design D5-D7).

    python -m src.fx_nyvol.walk_forward --stage lgbm    HAR (+ descriptive HAR-X) and LightGBM, every fold, both pairs
    python -m src.fx_nyvol.walk_forward --stage mlp     5-seed CUDA MLP for pairs whose LightGBM cell is KEEP

Stage 1 refuses before the label report; stage 2 refuses before both LightGBM verdicts. Artifacts go to
research_models/fx_nyvol/ through fx_session's artifact store (read-only code, this family's config);
the run files record the config and feature digests and the fx_session cache digests.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_fib.artifacts import sha256
from ..fx_session import artifacts as AR
from . import config_digest, feature_list_digest, load_config, repo_path
from . import models as M
from .registry import LABELS_JSON, RUNS, StageOrderError, guard, read_json, read_registry, results_dir
from .splits import fold_arrays, make_folds

OOS = {'lgbm': 'oos_lgbm.csv.gz', 'mlp': 'oos_mlp.csv.gz'}
CAL = ['is_nfp_day', 'is_fomc_day', 'dst_mismatch']


def input_digests(cfg: dict) -> dict:
    d = repo_path(cfg['paths']['fx_session_data'])
    return {f'fx_h1:{p}': sha256(os.path.join(d, f'{p}_h1.csv')) for p in cfg['factor_pairs']}


def load_inputs(cfg: dict) -> dict:
    from .target import dataset, load_frames
    return dataset(*load_frames(cfg), cfg)


def usable(data: dict) -> dict:
    return {p: d[d['reason'] == ''].sort_index() for p, d in data.items()}


def hot_threshold(y: pd.Series, cfg: dict) -> pd.Series:
    """Causal Q75: the quantile of y over the previous `hot_window_days` eligible days (never today)."""
    sc = cfg['scoring']
    w = int(sc['hot_window_days'])
    return y.rolling(w, min_periods=w).quantile(float(sc['hot_quantile'])).shift(1)


def _write(cfg, stage, frames, run) -> None:
    os.makedirs(results_dir(cfg), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(results_dir(cfg), OOS[stage]), index=False,
                                                compression='gzip')
    with open(os.path.join(results_dir(cfg), RUNS[stage]), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(run, fh, indent=1, default=str)


def load_oos(cfg: dict, stage: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), OOS[stage]))
    d['date'] = pd.to_datetime(d['date'], utc=True)
    return d


def run_lgbm(cfg: dict | None = None, now_iso: str | None = None, data: dict | None = None, log=print,
             digests: dict | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    if read_json(cfg, LABELS_JSON) is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_nyvol.labels')
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if data is None:
        data = load_inputs(cfg)
        digests = input_digests(cfg)
    rows_all = usable(data)
    last = max(d.index.max() for d in rows_all.values())
    cols, har_cols = cfg['features']['columns'], cfg['features']['har_columns']
    run = {'stage': 'lgbm', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'device': 'cpu (lightgbm, numpy OLS)',
           'input_digests': digests or {}, 'pairs': {}}
    out = []
    for pair in cfg['pairs']:
        rows = rows_all[pair]
        X, y = rows[cols].to_numpy(float), rows['y'].to_numpy(float)
        Xh = rows[har_cols].to_numpy(float)
        thr = hot_threshold(rows['y'], cfg).to_numpy()
        info = []
        for f in make_folds(rows.index, cfg, last):
            train = np.r_[f.fit, f.val]
            har = M.ols_fit(Xh[train], y[train])
            harx = M.ols_fit(X[train], y[train])
            model = M.train_lightgbm(X[f.fit], y[f.fit], X[f.val], y[f.val], cfg['lightgbm'])
            AR.save(cfg, model, pair, f.k, 'lightgbm', 'lightgbm', now_iso)
            ev = rows.iloc[f.eval]
            d = pd.DataFrame({'pair': pair, 'date': ev.index, 'fold': f.k, 'y': ev['y'].to_numpy(),
                              'y_bid': ev['y_bid'].to_numpy(), 'p_har': M.ols_predict(har, Xh[f.eval]),
                              'p_harx': M.ols_predict(harx, X[f.eval]), 'p_lightgbm': model.predict(X[f.eval]),
                              'hot_threshold': thr[f.eval]})
            for c in CAL:
                d[c] = ev[c].to_numpy()
            out.append(d)
            info.append({'k': f.k, 'start': str(f.start.date()), 'end': str(f.end.date()), 'n_fit': len(f.fit),
                         'n_val': len(f.val), 'n_eval': len(f.eval), 'last_train_day': str(rows.index[train].max().date()),
                         'har_coef': [float(c) for c in har], 'best_iteration': int(model.best_iteration_ or 0)})
            log(f"{pair} fold {f.k} {f.start.date()}..{f.end.date()}: fit {len(f.fit)} val {len(f.val)} "
                f"eval {len(f.eval)} | trees {model.best_iteration_}")
        run['pairs'][pair] = {'folds': info, 'n_eval': int(sum(i['n_eval'] for i in info))}
    _write(cfg, 'lgbm', out, run)
    return run


def keep_pairs(cfg: dict) -> list:
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    missing = [f'{p}_lightgbm_vs_har' for p in cfg['pairs'] if f'{p}_lightgbm_vs_har' not in reg]
    if missing:
        raise StageOrderError(f'the MLP stage needs both LightGBM verdicts first; missing {missing}')
    return [p for p in cfg['pairs'] if reg[f'{p}_lightgbm_vs_har']['verdict'] == 'KEEP']


def run_mlp(cfg: dict | None = None, now_iso: str | None = None, data: dict | None = None, log=print) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    pairs = keep_pairs(cfg)
    run = {'stage': 'mlp', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'trained_pairs': pairs, 'pairs': {}}
    if not pairs:
        log('no LightGBM cell is KEEP: the MLP stage trains nothing')
        _write(cfg, 'mlp', [pd.DataFrame(columns=['pair', 'date', 'fold', 'y', 'p_mlp'])], run)
        return run
    M.pick_device(cfg['mlp'].get('allow_cpu', False))
    data = data if data is not None else load_inputs(cfg)
    rows_all = usable(data)
    last = max(d.index.max() for d in rows_all.values())
    cols = cfg['features']['columns']
    seeds = cfg['mlp']['seeds']
    out = []
    for pair in pairs:
        rows = rows_all[pair]
        X, y = rows[cols].to_numpy(float), rows['y'].to_numpy(float)
        info = []
        for f in make_folds(rows.index, cfg, last):
            a = fold_arrays(X, y, f)
            AR.save(cfg, {'x_mean': a['x_scaler'].mean, 'x_sd': a['x_scaler'].sd, 'y_mean': a['y_scaler'].mean,
                          'y_sd': a['y_scaler'].sd}, pair, f.k, 'scalers', 'scaler', now_iso)
            members, metas = [], []
            for s in seeds:
                m, meta = M.train_mlp(a['Xs_fit'], a['ys_fit'], a['Xs_val'], a['ys_val'], cfg['mlp'], s)
                AR.save(cfg, {'state_dict': {k: v.cpu() for k, v in m.state_dict().items()}, 'meta': meta},
                        pair, f.k, f'mlp_seed{s}', 'mlp', now_iso)
                members.append(m)
                metas.append(meta)
            ev = rows.iloc[f.eval]
            out.append(pd.DataFrame({'pair': pair, 'date': ev.index, 'fold': f.k, 'y': ev['y'].to_numpy(),
                                     'p_mlp': M.ensemble_predict(members, a['Xs_eval'], len(seeds), a['y_scaler'])}))
            info.append({'k': f.k, 'best_epochs': [m['best_epoch'] for m in metas]})
            run.update(device=metas[0]['device'], torch=metas[0]['torch'], cuda=metas[0]['cuda'])
            log(f"{pair} fold {f.k}: mlp best epochs {[m['best_epoch'] for m in metas]}")
        run['pairs'][pair] = {'folds': info}
    _write(cfg, 'mlp', out, run)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description='run one fx-ny-volatility walk-forward stage')
    ap.add_argument('--stage', choices=['lgbm', 'mlp'], required=True)
    a = ap.parse_args(argv)
    (run_lgbm if a.stage == 'lgbm' else run_mlp)()


if __name__ == '__main__':
    main()
