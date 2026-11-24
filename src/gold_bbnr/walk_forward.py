"""The gated walk-forwards (spec gold-bb-nyrange/bb-reversion, bb-sma-trade, ny-range; design D6-D8, D11).

    python -m src.gold_bbnr.walk_forward --part bb       Part 1a: outcomes + LightGBM vs the fit-slice base rate
    python -m src.gold_bbnr.walk_forward --part bb_sma   Part 1b: trades + LightGBM vs the fit-slice base rate
    python -m src.gold_bbnr.walk_forward --part nyvol    Part 2: HAR, HAR-X, LightGBM and the 5-seed CUDA MLP

Every part first runs the pre-registration gate (git, config digest, cache digests): no outcome, exit or
target is computed before it passes. Outputs, under results/gold_bbnr/:

  bb_events.csv.gz / bb_predictions.csv.gz          every Part 1a event with r; the model rows with p_prior, p_lightgbm
  bb_sma_trades.csv.gz / bb_sma_predictions.csv.gz  every opened trade with R; the model rows
  nyvol_predictions.csv.gz                          every scored day with y and the four forecasts
  run_<part>.json                                   digests, folds, sizes
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_nyvol import models as M
from ..fx_nyvol.splits import fold_arrays
from ..fx_session import model_lgbm as LG
from . import config_digest, load_config
from . import bb_events as E
from . import bb_features as F
from . import bb_trades as T
from . import ny_features as NF
from .inputs import Inputs, load
from .registry import StageOrderError, guard, results_dir
from .splits import day_folds, event_folds

FILES = {'bb': ('bb_events.csv.gz', 'bb_predictions.csv.gz'), 'bb_sma': ('bb_sma_trades.csv.gz', 'bb_sma_predictions.csv.gz'),
         'nyvol': (None, 'nyvol_predictions.csv.gz')}


def _gate(cfg):
    from .prereg import gate
    return gate(cfg)


def _write(cfg, name, df):
    os.makedirs(results_dir(cfg), exist_ok=True)
    df.to_csv(os.path.join(results_dir(cfg), name), index=False,
              compression={'method': 'gzip', 'mtime': 0})          # no write time in the header: same bytes


def _write_run(cfg, part, rec):
    with open(os.path.join(results_dir(cfg), f'run_{part}.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, default=str)


def load_csv(cfg: dict, name: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), name))
    for c in d.columns:
        if c == 'date' or c.startswith('date_') or c.endswith('_time'):
            d[c] = pd.to_datetime(d[c], format='ISO8601')
    return d


def lgbm_classifier(rows: pd.DataFrame, out_dates: pd.Series, cfg: dict, log) -> tuple:
    """Walk-forward binary LightGBM on rows sorted by date. Returns (predictions, fold info, fold reasons)."""
    cols = cfg['features_bb']['columns']
    X, y = rows[cols].to_numpy(float), rows['y'].to_numpy(float)
    folds, fr = event_folds(rows['date'], out_dates, cfg)
    preds, info = [], []
    for f in folds:
        model = LG.train_lightgbm(X[f.fit], y[f.fit], X[f.val], y[f.val], cfg['lightgbm_bb'])
        prior = LG.prior_up(y[f.fit])
        d = rows.iloc[f.eval].copy()
        d['fold'], d['p_prior'], d['p_lightgbm'] = f.k, prior, LG.proba_up(model, X[f.eval])
        preds.append(d)
        info.append({'k': f.k, 'start': str(f.start.date()), 'n_fit': len(f.fit), 'n_val': len(f.val),
                     'n_eval': len(f.eval), 'base_rate': prior, 'best_iteration': int(model.best_iteration_ or 0)})
        log(f'fold {f.k} {f.start.date()}: fit {len(f.fit)} val {len(f.val)} test {len(f.eval)} | '
            f'base rate {prior:.3f} | trees {model.best_iteration_}')
    out = pd.concat(preds, ignore_index=True) if preds else rows.iloc[:0].assign(fold=[], p_prior=[], p_lightgbm=[])
    return out, info, fr


def run_bb(cfg: dict, inp: Inputs, log=print) -> dict:
    ev, counts = E.detect(inp.d1, cfg)
    ev = F.with_features(ev, inp.d1, inp.dxy_d1, cfg)
    ev = E.outcomes(ev, inp.d1)
    rows = ev[ev['reason2'] == ''].sort_values('date', kind='mergesort').reset_index(drop=True)
    preds, info, fr = lgbm_classifier(rows, rows['date_t3'], cfg, log)
    ev['fold_reason'] = ''
    ev.loc[ev['reason2'] != '', 'fold_reason'] = ev['reason2']
    m = pd.Series(fr, index=rows['date']).to_dict()
    ev.loc[ev['reason2'] == '', 'fold_reason'] = ev.loc[ev['reason2'] == '', 'date'].map(m)
    _write(cfg, FILES['bb'][0], ev)
    _write(cfg, FILES['bb'][1], preds)
    return {'counts': counts, 'n_events': len(ev), 'n_rule': int((ev['reason'] == '').sum()), 'n_model': len(preds),
            'folds': info}


def run_bb_sma(cfg: dict, inp: Inputs, log=print) -> dict:
    ev = T.entries(inp.d1, cfg)
    x = T.context(inp.h1, inp.d1, cfg)
    tr = T.build_stream(ev, x, cfg)
    blocked = tr.attrs.get('blocked', 0)
    tr = F.with_features(tr.assign(date_t3=tr['date_t10']), inp.d1, inp.dxy_d1, cfg).drop(columns=['date_t3'])
    rows = tr[tr['reason2'] == ''].sort_values('date', kind='mergesort').reset_index(drop=True)
    preds, info, fr = lgbm_classifier(rows, rows['date_exit'], cfg, log)
    m = pd.Series(fr, index=rows['date']).to_dict()
    tr['fold_reason'] = np.where(tr['reason2'] != '', tr['reason2'], tr['date'].map(m).fillna(''))
    _write(cfg, FILES['bb_sma'][0], tr)
    _write(cfg, FILES['bb_sma'][1], preds)
    return {'candidates': len(ev), 'blocked': int(blocked), 'opened': len(tr), 'n_rule': int(tr['R'].notna().sum()),
            'n_model': len(preds), 'folds': info}


def run_nyvol(cfg: dict, inp: Inputs, log=print) -> dict:
    M.pick_device(cfg['mlp'].get('allow_cpu', False))               # refuses without CUDA, before any target
    ds = NF.dataset(inp.h1, inp.d1, inp.dxy_h1, cfg)
    rows = ds[ds['reason'] == ''].sort_index()
    cols, har_cols = cfg['features_ny']['columns'], cfg['features_ny']['har_columns']
    X, y, Xh = rows[cols].to_numpy(float), rows['y'].to_numpy(float), rows[har_cols].to_numpy(float)
    seeds = cfg['mlp']['seeds']
    out, info = [], []
    for f in day_folds(rows.index, cfg):
        train = np.r_[f.fit, f.val]
        har, harx = M.ols_fit(Xh[train], y[train]), M.ols_fit(X[train], y[train])
        lgbm = M.train_lightgbm(X[f.fit], y[f.fit], X[f.val], y[f.val], cfg['lightgbm_ny'])
        a = fold_arrays(X, y, f)
        members, epochs = [], []
        for s in seeds:
            mdl, meta = M.train_mlp(a['Xs_fit'], a['ys_fit'], a['Xs_val'], a['ys_val'], cfg['mlp'], s)
            members.append(mdl)
            epochs.append(meta['best_epoch'])
        ev = rows.iloc[f.eval]
        out.append(pd.DataFrame({'date': ev.index, 'fold': f.k, 'y': ev['y'].to_numpy(),
                                 'p_har': M.ols_predict(har, Xh[f.eval]), 'p_harx': M.ols_predict(harx, X[f.eval]),
                                 'p_lightgbm': lgbm.predict(X[f.eval]),
                                 'p_mlp': M.ensemble_predict(members, a['Xs_eval'], len(seeds), a['y_scaler'])}))
        info.append({'k': f.k, 'start': str(f.start.date()), 'n_fit': len(f.fit), 'n_val': len(f.val),
                     'n_eval': len(f.eval), 'last_train_day': str(rows.index[train].max().date()),
                     'trees': int(lgbm.best_iteration_ or 0), 'mlp_best_epochs': epochs})
        log(f'fold {f.k} {f.start.date()}: fit {len(f.fit)} val {len(f.val)} eval {len(f.eval)} | '
            f'trees {lgbm.best_iteration_} | mlp epochs {epochs}')
    _write(cfg, FILES['nyvol'][1], pd.concat(out, ignore_index=True))
    return {'usable_days': len(rows), 'scored_days': int(sum(i['n_eval'] for i in info)), 'folds': info}


RUNNERS = {'bb': run_bb, 'bb_sma': run_bb_sma, 'nyvol': run_nyvol}


def run(part: str, cfg: dict | None = None, inp: Inputs | None = None, gate=None, log=print) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    commit = (gate or _gate)(cfg)                                  # before any outcome
    inp = inp or load(cfg)
    rec = {'part': part, 'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
           'config_digest': config_digest(cfg), 'prereg_commit_date': commit}
    rec.update(RUNNERS[part](cfg, inp, log))
    _write_run(cfg, part, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description='run one gated gold-bb-nyrange walk-forward')
    ap.add_argument('--part', choices=list(RUNNERS), required=True)
    a = ap.parse_args(argv)
    try:
        run(a.part)
    except StageOrderError as e:
        raise SystemExit(str(e))


if __name__ == '__main__':
    main()
