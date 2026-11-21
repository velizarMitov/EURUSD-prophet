"""The fx-fibonacci walk-forward (spec fx-fibonacci/evaluation).

For every pair and fold, every model is trained on the SAME scaled arrays
(splits.fold_arrays) and scored on the fold's out-of-sample rows:
    mlp         the registered 5-seed ensemble (all-or-nothing)
    lightgbm    the registered baseline
    logistic    descriptive
    mlp_small   descriptive (64-32, first seed)
GROSS of spread and slippage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import artifacts as AR
from . import baselines as B
from . import config_digest, feature_list_digest, load_config, repo_path
from . import model_mlp_cuda as M
from . import mtf_aligner as AL
from . import splits as SP
from .metrics import prior_proba

MODELS = ('mlp', 'lightgbm', 'logistic', 'mlp_small')
OOS_FILE = 'oos_predictions.csv.gz'
RUN_FILE = 'walk_forward_run.json'


def default_trainers(cfg: dict) -> dict:
    return {
        'mlp': lambda a, seed: M.train_mlp(a['X_fit'], a['y_fit'], a['X_val'], a['y_val'], cfg['mlp'], seed),
        'lightgbm': lambda a: B.train_lightgbm(a['X_fit'], a['y_fit'], a['X_val'], a['y_val'], cfg['lightgbm']),
        'logistic': lambda a: B.train_logistic(a['X_fit'], a['y_fit'], cfg['logistic']),
        'mlp_small': lambda a: B.train_mlp_small(a['X_fit'], a['y_fit'], a['X_val'], a['y_val'], cfg),
    }


def train_fold_models(arr: dict, cfg: dict, pair: str, fold_k: int, now_iso: str,
                      save: bool = True, trainers: dict | None = None) -> dict:
    """Train every model on one fold's arrays; return eval-row probabilities and metadata."""
    t = trainers or default_trainers(cfg)
    seeds = cfg['mlp']['seeds']
    members, metas = [], []
    for seed in seeds:
        model, meta = t['mlp'](arr, seed)
        if save:
            AR.save_mlp(cfg, model, meta, pair, fold_k, f'seed{seed}', now_iso)
        members.append(model)
        metas.append(meta)
    proba = {'mlp': M.ensemble_proba(members, arr['X_eval'], len(seeds))}
    meta = {'mlp': metas}
    for name in ('lightgbm', 'logistic', 'mlp_small'):
        model, m = t[name](arr)
        proba[name] = B.proba(model, arr['X_eval'])
        meta[name] = m
        if save:
            if name == 'mlp_small':
                AR.save_mlp(cfg, model, m, pair, fold_k, 'small', now_iso)
            else:
                AR.save_object(cfg, model, pair, fold_k, name, now_iso)
    if save:
        AR.save_object(cfg, arr['scaler'], pair, fold_k, 'scaler', now_iso)
    for name, p in proba.items():
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f'{name} probabilities do not sum to 1')
    return {'proba': proba, 'meta': meta}


def pair_inputs(h1: pd.DataFrame, cfg: dict) -> dict:
    """Features, labels and the eligibility mask for one pair."""
    from .features import build_features, valid_rows
    from .labeling import build_labels
    F = build_features(h1, cfg)
    L = build_labels(h1, cfg)
    eligible = (valid_rows(F) & L['label'].notna()).to_numpy()
    td = AL.trading_date(h1.index)
    return {'X': F.to_numpy(dtype=np.float32), 'y': np.nan_to_num(L['label'].to_numpy()).astype(np.int64),
            'labels': L, 'eligible': eligible, 'trading_date': td}


def oos_frame(h1, inp, fold, arr, result, pair) -> pd.DataFrame:
    rows = fold.eval
    L = inp['labels'].iloc[rows]
    exit_pos = rows + L['exit_offset'].to_numpy().astype(int)
    df = pd.DataFrame({'pair': pair, 'time': h1.index[rows], 'pos': rows, 'fold': fold.k,
                       'close': h1['close'].to_numpy()[rows], 'label': arr['y_eval'],
                       'exit_offset': L['exit_offset'].to_numpy(), 'upper': L['upper'].to_numpy(),
                       'lower': L['lower'].to_numpy(), 'exit_close': L['exit_close'].to_numpy(),
                       'exit_day': inp['trading_date'][exit_pos]})
    probs = {'prior': prior_proba(arr['y_fit'], len(rows)), **result['proba']}
    for name, p in probs.items():
        for c in range(3):
            df[f'p_{name}_{c}'] = p[:, c]
    return df


def run_walk_forward(cfg: dict | None = None, now_iso: str | None = None, pairs=None,
                     max_folds: int | None = None, save: bool = True, log=print) -> dict:
    """Every fold of every pair; writes the pooled out-of-sample predictions and the run log."""
    from .data_loader import load_h1
    from .score import assert_config_matches_verdicts
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    H = cfg['labels']['horizon_bars']
    run = {'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'pairs': {}}
    frames = []
    for pair in pairs or cfg['pairs']:
        h1 = load_h1(pair, cfg)
        inp = pair_inputs(h1, cfg)
        folds = SP.make_folds(h1.index, inp['eligible'], cfg)[:max_folds]
        run['pairs'][pair] = {'folds': []}
        for f in folds:
            SP.check_fold(f, H)
            t0 = time.time()
            arr = SP.fold_arrays(inp['X'], inp['y'], f)
            res = train_fold_models(arr, cfg, pair, f.k, now_iso, save=save)
            frames.append(oos_frame(h1, inp, f, arr, res, pair))
            meta = res['meta']
            run.setdefault('device', meta['mlp'][0]['device'])
            run.setdefault('torch', meta['mlp'][0]['torch'])
            run.setdefault('cuda', meta['mlp'][0]['cuda'])
            info = {'k': f.k, 'start': str(f.start.date()), 'end': str(f.end.date()),
                    'n_fit': int(len(f.fit)), 'n_val': int(len(f.val)), 'n_eval': int(len(f.eval)),
                    'mlp_best_epochs': [m['best_epoch'] for m in meta['mlp']],
                    'lightgbm_best_iteration': meta['lightgbm']['best_iteration'],
                    'seconds': round(time.time() - t0, 1)}
            run['pairs'][pair]['folds'].append(info)
            log(f"{pair} fold {f.k} {info['start']}..{info['end']}: fit {info['n_fit']} val {info['n_val']} "
                f"eval {info['n_eval']} | mlp epochs {info['mlp_best_epochs']} | {info['seconds']}s")
    oos = pd.concat(frames, ignore_index=True)
    if save:
        out_dir = repo_path(cfg['paths']['results'])
        os.makedirs(out_dir, exist_ok=True)
        oos.to_csv(os.path.join(out_dir, OOS_FILE), index=False, compression='gzip')
        with open(os.path.join(out_dir, RUN_FILE), 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(run, fh, indent=1)
    return {'oos': oos, 'run': run}


def load_oos(cfg: dict) -> pd.DataFrame:
    path = os.path.join(repo_path(cfg['paths']['results']), OOS_FILE)
    return pd.read_csv(path, parse_dates=['time', 'exit_day'])


def main(argv=None):
    ap = argparse.ArgumentParser(description='run the fx-fibonacci walk-forward (trains on the GPU)')
    ap.add_argument('--max-folds', type=int, default=None, help='limit folds per pair (smoke runs)')
    a = ap.parse_args(argv)
    run_walk_forward(max_folds=a.max_folds)


if __name__ == '__main__':
    main()
