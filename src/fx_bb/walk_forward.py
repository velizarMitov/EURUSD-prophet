"""The Cell 2 walk-forward: one pooled LightGBM vs the fit-slice base rate (spec fx-bb-reversion/evaluation;
design D6, D7).

    python -m src.fx_bb.walk_forward

Model code and artifact storage are fx_session's (model_lgbm, artifacts), imported read-only and
driven by THIS family's config: artifacts go to research_models/fx_bb/POOLED/fold_NN/, digests to
results/fx_bb/artifact_manifest.csv. Writes results/fx_bb/oos_events.csv.gz (every out-of-sample
event once, with p_prior and p_lightgbm) and results/fx_bb/run.json (config and feature digests,
the six cache digests, the folds). Runs the config guard first and refuses before the label report.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_session import artifacts as AR
from ..fx_session import model_lgbm as LG
from . import config_digest, feature_list_digest, load_config
from .registry import LABELS_JSON, RUN_JSON, StageOrderError, _json, guard, results_dir
from .score import mean_close_spread
from .splits import make_folds

OOS = 'oos_events.csv.gz'
POOLED = 'POOLED'
CARRY = ['pair', 'date', 'dir', 'side', 'close', 'close_t3', 'atr', 'date_t3', 'r', 'y']


def cell2_rows(d1_all: dict, cfg: dict) -> pd.DataFrame:
    """Every event usable by Cell 2 (eligible, full features, not a tie), sorted by date then pair."""
    from . import events as E
    from .features import with_features
    ev, _ = E.build_events(d1_all, cfg)
    ev = with_features(ev, d1_all, cfg)
    ev = E.outcomes(ev, d1_all)
    use = ev[(ev['reason2'] == '') & ev['y'].notna()]
    return use.sort_values(['date', 'pair'], kind='mergesort').reset_index(drop=True)


def fit_fold(X: np.ndarray, y: np.ndarray, fold, cfg: dict) -> tuple:
    """(model, base rate of the fit slice, P(win) of the test events)."""
    model = LG.train_lightgbm(X[fold.fit], y[fold.fit], X[fold.val], y[fold.val], cfg['lightgbm'])
    return model, LG.prior_up(y[fold.fit]), LG.proba_up(model, X[fold.eval])


def load_oos(cfg: dict) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), OOS))
    d['date'] = pd.to_datetime(d['date'])
    return d


def run(cfg: dict | None = None, now_iso: str | None = None, d1_all: dict | None = None, log=print,
        digests: dict | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    if _json(cfg, LABELS_JSON) is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_bb.labels')
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if d1_all is None:
        from .daily import build_all
        from .data_loader import cache_digests, load_all
        d1_all = build_all(*load_all(cfg), cfg)
        digests = cache_digests(cfg)
    rows = cell2_rows(d1_all, cfg)
    cols = cfg['features']['columns']
    X, y = rows[cols].to_numpy(dtype=float), rows['y'].to_numpy(dtype=float)
    folds = make_folds(rows['date'], rows['date_t3'], cfg)
    if not folds:
        raise StageOrderError('no walk-forward fold could be formed')
    rec = {'stage': 'walk_forward', 'run_at': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg), 'device': 'cpu (lightgbm)',
           'input_digests': digests or {}, 'n_cell2_events': len(rows), 'folds': []}
    out = []
    for f in folds:
        model, prior, p = fit_fold(X, y, f, cfg)
        AR.save(cfg, model, POOLED, f.k, 'lightgbm', 'lightgbm', now_iso)
        d = rows.iloc[f.eval][CARRY + cols].copy()
        d['fold'] = f.k
        d['p_prior'] = prior
        d['p_lightgbm'] = p
        out.append(d)
        info = {'k': f.k, 'start': str(f.start.date()), 'end': str(f.end.date()), 'n_fit': len(f.fit),
                'n_val': len(f.val), 'n_eval': len(f.eval), 'fit_last_date': str(rows['date'].iloc[f.fit[-1]].date()),
                'val_first_date': str(rows['date'].iloc[f.val[0]].date()),
                'val_last_t3': str(rows['date_t3'].iloc[f.val].max().date()), 'base_rate': prior,
                'best_iteration': int(model.best_iteration_ or 0)}
        rec['folds'].append(info)
        log(f"fold {f.k} {f.start.date()}..{f.end.date()}: fit {len(f.fit)} val {len(f.val)} test {len(f.eval)} | "
            f"base rate {prior:.3f} | trees {model.best_iteration_}")
    oos = pd.concat(out, ignore_index=True)
    oos['spread_pips'] = oos['pair'].map(mean_close_spread(d1_all, oos))     # descriptive use only
    rec['n_oos'] = len(oos)
    os.makedirs(results_dir(cfg), exist_ok=True)
    oos.to_csv(os.path.join(results_dir(cfg), OOS), index=False, compression='gzip')
    with open(os.path.join(results_dir(cfg), RUN_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def main(argv=None):
    argparse.ArgumentParser(description='run the fx-bb Cell 2 walk-forward').parse_args(argv)
    run()


if __name__ == '__main__':
    main()
