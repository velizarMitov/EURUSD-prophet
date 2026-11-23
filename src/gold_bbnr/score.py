"""One-shot scoring of the seven cells (spec gold-bb-nyrange/evaluation).

    python -m src.gold_bbnr.score --family bb       gold_bb_rule_mean_reversion, gold_bb_lightgbm_vs_base_rate
    python -m src.gold_bbnr.score --family bb_sma   gold_bb_sma_rule_mean_r, gold_bb_sma_lightgbm_vs_base_rate
    python -m src.gold_bbnr.score --family nyvol    gold_harx_vs_har, gold_lgbm_vs_harx, gold_mlp_vs_harx

fx_bb's ratio day-block bootstrap (read-only): per business day from the first to the last scored date,
S = the sum of the row statistics and N = the row count; circular blocks drawn identically for S and N.
  "> 0" cells     KEEP when the alpha-quantile of sum(S*)/sum(N*) is above 0
  "below" cells   KEEP when the (1 - alpha)-quantile of the mean difference is below 0
Everything else is DROP. The gate runs first. Each cell is scored once; the family's rows are appended
together and a second attempt raises AlreadyScored. GROSS of cost, never a trading result.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_bb.score import day_sums, test_ratio
from ..fx_session.score import row_logloss
from . import config_digest, load_config, sha256_file
from .registry import CELLS, PREREG_JSON, AlreadyScored, StageOrderError, append, guard, path, scored
from .walk_forward import FILES, load_csv

GROSS = 'Gross of cost on bid prices, not a trading result.'
FACT = ('Earlier: fx-bb-reversion, fx-bb-sma-exit and fx-ny-orb-hot DROP; fx-ny-volatility LightGBM KEEP and its '
        'descriptive HAR-X best. Gold H1 history was used by the gold family (volatility CLEARED, ranges DROP).')


def _gate(cfg):
    from .prereg import gate
    return gate(cfg)


def _naive(dates) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(dates)
    return d.tz_localize(None) if d.tz is not None else d


def cell(cfg, family, hypothesis, metric, dates, values, side, model_mean=None, ref_mean=None, notes='') -> dict:
    sc, R = cfg['scoring'][family], cfg['scoring']['resamples']
    dates = _naive(dates)
    S, N = day_sums(dates, values)
    res = test_ratio(S, N, sc['alpha'], sc['block_days'], R, sc['seed'], side)
    pj = path(cfg, PREREG_JSON)
    return {'date': datetime.now(timezone.utc).date().isoformat(), 'hypothesis': hypothesis, 'family': family,
            'metric': metric, 'statistic': res['statistic'], 'model_mean': model_mean, 'reference_mean': ref_mean,
            'n_rows': len(values), 'n_dates': int(dates.nunique()), 'first_date': str(dates.min().date()),
            'last_date': str(dates.max().date()), 'bound': res['bound'], 'bound_quantile': res['bound_quantile'],
            'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': R, 'seed': sc['seed'],
            'boot_se': res['boot_se'], 'mde_80': res['mde_80'], 'verdict': res['verdict'],
            'config_digest': config_digest(cfg), 'prereg_sha256': sha256_file(pj) if os.path.exists(pj) else '',
            'notes': f'{notes} {GROSS} {FACT}'.strip()}


def score_bb(cfg: dict) -> list:
    ev, pr = load_csv(cfg, FILES['bb'][0]), load_csv(cfg, FILES['bb'][1])
    rule = ev[ev['reason'].fillna('') == '']
    y = pr['y'].to_numpy(float)
    lm, lp = row_logloss(y, pr['p_lightgbm'].to_numpy(float)), row_logloss(y, pr['p_prior'].to_numpy(float))
    c = CELLS['bb']
    return [cell(cfg, 'bb', c[0], 'mean r (ATR)', rule['date'], rule['r'].to_numpy(float), 'greater',
                 notes='Every Part 1a event with t+3; 3-day cooldown; alpha 0.05/2; 5-day blocks.'),
            cell(cfg, 'bb', c[1], 'mean log-loss difference (LightGBM - base rate)', pr['date'], lm - lp, 'less',
                 float(lm.mean()), float(lp.mean()),
                 notes=f"{pr['fold'].nunique()} 12-month folds from the first with >= 60 training events; alpha 0.05/2.")]


def score_bb_sma(cfg: dict) -> list:
    tr, pr = load_csv(cfg, FILES['bb_sma'][0]), load_csv(cfg, FILES['bb_sma'][1])
    rule = tr[tr['R'].notna()]
    y = pr['y'].to_numpy(float)
    lm, lp = row_logloss(y, pr['p_lightgbm'].to_numpy(float)), row_logloss(y, pr['p_prior'].to_numpy(float))
    c = CELLS['bb_sma']
    return [cell(cfg, 'bb_sma', c[0], 'mean R', rule['date'], rule['R'].to_numpy(float), 'greater',
                 notes='Every opened Part 1b trade with an exit; one open trade; alpha 0.05/2; 10-day blocks.'),
            cell(cfg, 'bb_sma', c[1], 'mean log-loss difference (LightGBM - base rate)', pr['date'], lm - lp, 'less',
                 float(lm.mean()), float(lp.mean()),
                 notes=f"{pr['fold'].nunique()} 12-month folds, purge from the exit date; alpha 0.05/2.")]


def score_nyvol(cfg: dict) -> list:
    d = load_csv(cfg, FILES['nyvol'][1])
    y = d['y'].to_numpy(float)
    se = {k: (y - d[f'p_{k}'].to_numpy(float)) ** 2 for k in ('har', 'harx', 'lightgbm', 'mlp')}
    c = CELLS['nyvol']
    note = f"{len(d)} scored days in {d['fold'].nunique()} 6-month folds; target log(NY range / close14); alpha 0.05/3."
    return [cell(cfg, 'nyvol', c[0], 'mean squared-error difference (HAR-X - HAR)', d['date'], se['harx'] - se['har'],
                 'less', float(se['harx'].mean()), float(se['har'].mean()), note),
            cell(cfg, 'nyvol', c[1], 'mean squared-error difference (LightGBM - HAR-X)', d['date'],
                 se['lightgbm'] - se['harx'], 'less', float(se['lightgbm'].mean()), float(se['harx'].mean()), note),
            cell(cfg, 'nyvol', c[2], 'mean squared-error difference (MLP - HAR-X)', d['date'], se['mlp'] - se['harx'],
                 'less', float(se['mlp'].mean()), float(se['harx'].mean()), note)]


SCORERS = {'bb': score_bb, 'bb_sma': score_bb_sma, 'nyvol': score_nyvol}


def score(family: str, cfg: dict | None = None, gate=None) -> list:
    cfg = cfg or load_config()
    guard(cfg)
    (gate or _gate)(cfg)
    have = scored(cfg, family)
    if any(c in have for c in CELLS[family]):
        raise AlreadyScored(f'{family} is already scored; each cell is scored once')
    for name in FILES[family]:
        if name and not os.path.exists(path(cfg, name)):
            raise StageOrderError(f'run the walk-forward first: python -m src.gold_bbnr.walk_forward --part {family}')
    rows = SCORERS[family](cfg)
    append(cfg, family, rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='score one gold-bb-nyrange family, once')
    ap.add_argument('--family', choices=list(SCORERS), required=True)
    a = ap.parse_args(argv)
    for r in score(a.family):
        print(f"{r['hypothesis']}: statistic {r['statistic']:+.5f}, bound {r['bound']:+.5f}, n {r['n_rows']} -> "
              f"{r['verdict']}")


if __name__ == '__main__':
    main()
