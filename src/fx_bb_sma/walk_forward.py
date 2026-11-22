"""The Cell 2 walk-forward: one pooled LightGBM vs the fit-slice win rate (spec fx-bb-sma-exit/evaluation; D9).

    python -m src.fx_bb_sma.walk_forward

Features are fx_bb's ten oriented features at the close of t (read-only); the target is net R > 0.
Model and artifact code are fx_session's / fx_bb's (read-only), driven by THIS family's config:
artifacts under research_models/fx_bb_sma/POOLED/, digests in results/fx_bb_sma/artifact_manifest.csv.
Writes oos_trades.csv.gz and run.json; runs the guard first and refuses before the label report.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ..fx_bb.features import assemble, build_features
from ..fx_bb.walk_forward import fit_fold
from ..fx_session import artifacts as AR
from . import bb_view, config_digest, feature_list_digest, load_config
from . import trades as T
from .registry import LABELS_JSON, RUN_JSON, StageOrderError, guard, read_json, results_dir
from .splits import make_folds

OOS = 'oos_trades.csv.gz'
POOLED = 'POOLED'
CARRY = ['pair', 'date', 'dir', 'side', 'close', 'tp', 'sl', 'dist', 'exit_reason', 'exit_date', 'exit_price',
         'gross_pips', 'net_pips', 'r_gross', 'r_net', 'spread_entry_pips', 'spread_exit_pips', 'y']


def cell2_rows(cfg: dict, inputs: tuple) -> pd.DataFrame:
    """Eligible trades with every feature, target y = net R > 0, sorted by entry date then pair."""
    ctxs, d1_all = inputs
    tr, _ = T.build_stream(ctxs, d1_all, cfg)
    tr = T.results(tr[tr['reason'] == ''].reset_index(drop=True), ctxs, cfg)
    feats = assemble(build_features(tr, d1_all, bb_view(cfg)), cfg)
    out = tr.join(feats)
    out = out[feats.notna().all(axis=1).to_numpy()].copy()
    out['y'] = (out['r_net'] > 0).astype(float)
    return out.sort_values(['date', 'pair'], kind='mergesort').reset_index(drop=True)


def load_oos(cfg: dict) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), OOS))
    for c in ('date', 'exit_date'):
        d[c] = pd.to_datetime(d[c])
    return d


def run(cfg: dict | None = None, now_iso: str | None = None, inputs: tuple | None = None, log=print,
        digests: dict | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    if read_json(cfg, LABELS_JSON) is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_bb_sma.labels')
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if inputs is None:
        inputs = T.load_inputs(cfg)
        digests = T.cache_digests(cfg)
    rows = cell2_rows(cfg, inputs)
    cols = cfg['features']['columns']
    X, y = rows[cols].to_numpy(dtype=float), rows['y'].to_numpy(dtype=float)
    folds = make_folds(rows['date'], rows['exit_date'], cfg)
    if not folds:
        raise StageOrderError('no walk-forward fold could be formed')
    rec = {'stage': 'walk_forward', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'device': 'cpu (lightgbm)', 'input_digests': digests or {},
           'n_cell2_trades': len(rows), 'folds': []}
    out = []
    for f in folds:
        model, prior, p = fit_fold(X, y, f, cfg)
        AR.save(cfg, model, POOLED, f.k, 'lightgbm', 'lightgbm', now_iso)
        d = rows.iloc[f.eval][CARRY + cols].copy()
        d['fold'] = f.k
        d['p_prior'] = prior
        d['p_lightgbm'] = p
        out.append(d)
        rec['folds'].append({'k': f.k, 'start': str(f.start.date()), 'end': str(f.end.date()), 'n_fit': len(f.fit),
                             'n_val': len(f.val), 'n_eval': len(f.eval),
                             'fit_last_exit': str(rows['exit_date'].iloc[f.fit].max().date()),
                             'val_first_date': str(rows['date'].iloc[f.val[0]].date()),
                             'val_last_exit': str(rows['exit_date'].iloc[f.val].max().date()), 'base_rate': prior,
                             'best_iteration': int(model.best_iteration_ or 0)})
        log(f"fold {f.k} {f.start.date()}..{f.end.date()}: fit {len(f.fit)} val {len(f.val)} test {len(f.eval)} | "
            f"base rate {prior:.3f} | trees {model.best_iteration_}")
    oos = pd.concat(out, ignore_index=True)
    rec['n_oos'] = len(oos)
    os.makedirs(results_dir(cfg), exist_ok=True)
    oos.to_csv(os.path.join(results_dir(cfg), OOS), index=False, compression='gzip')
    with open(os.path.join(results_dir(cfg), RUN_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def main(argv=None):
    argparse.ArgumentParser(description='run the fx-bb-sma Cell 2 walk-forward').parse_args(argv)
    run()


if __name__ == '__main__':
    main()
