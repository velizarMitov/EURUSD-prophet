"""The day-block ratio bootstrap and the one-shot scoring of the two cells
(spec fx-bb-reversion/evaluation; design D8).

    python -m src.fx_bb.score --cell rule     Cell 1  bb_rule_mean_reversion   (mean r > 0)
    python -m src.fx_bb.score --cell model    Cell 2  bb_lightgbm_vs_base_rate (log-loss below the base rate)

Both cells resample whole BUSINESS DAYS (Monday to Friday, first to last scored event), so events
on one date and on neighbouring dates stay together. Per day d: S_d = the sum over the events dated
d (Cell 1: r; Cell 2: logloss_lgbm - logloss_prior) and N_d = their count. A circular block
bootstrap with 5-day blocks draws the SAME blocks for S and N and recomputes sum(S*) / sum(N*),
the event-weighted mean. With N == 1 every day it reproduces fx_fib.score.circular_block_means
exactly (pinned by a test).

    Cell 1 KEEP when the alpha-quantile of the bootstrap means is above 0
    Cell 2 KEEP when the (1 - alpha)-quantile is below 0          alpha = 0.05/2 each

Each cell is scored once. GROSS predictive results, never a trading edge.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..fx_fib.artifacts import sha256
from ..fx_session.artifacts import manifest_path
from ..fx_session.score import row_logloss
from . import config_digest, feature_list_digest, load_config
from .registry import (CELL_MODEL, CELL_RULE, LABELS_JSON, RUN_JSON, AlreadyScored, ConfigChanged,  # noqa: F401
                       StageOrderError, _json, append_row, guard, read_registry, results_dir)

GROSS = 'gross of spread and slippage, not a trading edge'
PRIOR_FACTS = ('Earlier fx families, recorded as facts and not blockers: fx-fibonacci dropped all four cells, '
               'fx-session dropped both LightGBM cells (MLP not triggered) and fx-rates dropped both LightGBM cells '
               '(MLP not triggered), all on 2026-11-21; each forecast every day and could see only a 54-56 % edge.')


# -- the ratio bootstrap ------------------------------------------------------------------------

def day_sums(dates, values, first=None, last=None) -> tuple:
    """(S, N) per business day from the first to the last date; days without events are 0, 0."""
    dates = pd.DatetimeIndex(dates)
    if (dates.dayofweek >= 5).any():
        raise ValueError('an event is dated on a weekend; D1 days are Monday to Friday')
    days = pd.bdate_range(first or dates.min(), last or dates.max())
    s = pd.Series(np.asarray(values, dtype=float), index=dates).groupby(level=0).sum()
    n = pd.Series(1.0, index=dates).groupby(level=0).sum()
    if not s.index.isin(days).all():
        raise ValueError('event dates fall outside the business-day range')
    return s.reindex(days, fill_value=0.0).to_numpy(), n.reindex(days, fill_value=0.0).to_numpy()


def ratio_block_means(S: np.ndarray, N: np.ndarray, block: int, resamples: int, seed: int) -> np.ndarray:
    """sum(S*) / sum(N*) under the circular block bootstrap, the same blocks for S and N.
    Mirrors fx_fib.score.circular_block_means draw for draw."""
    S, N = np.asarray(S, dtype=float), np.asarray(N, dtype=float)
    n = len(S)
    block = max(1, min(block, n))

    def sums(x):
        ext = np.r_[x, x[:block]]
        c = np.r_[0.0, np.cumsum(ext)]
        return c[np.arange(n) + block] - c[np.arange(n)], c[np.arange(n) + r] - c[np.arange(n)]

    k, r = divmod(n, block)
    fs, ps = sums(S)
    fn, pn = sums(N)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for i0 in range(0, resamples, 500):
        m = min(500, resamples - i0)
        idx = rng.integers(0, n, size=(m, k))
        s, c = fs[idx].sum(axis=1), fn[idx].sum(axis=1)
        if r:
            j = rng.integers(0, n, size=m)
            s += ps[j]
            c += pn[j]
        if (c == 0).any():
            raise ValueError('a bootstrap resample drew no event')
        out[i0:i0 + m] = s / c
    return out


def test_ratio(S, N, alpha: float, block: int, resamples: int, seed: int, side: str) -> dict:
    """side 'greater': KEEP when the alpha-quantile > 0. side 'less': KEEP when the (1-alpha)-quantile < 0."""
    boots = ratio_block_means(S, N, block, resamples, seed)
    q = alpha if side == 'greater' else 1 - alpha
    bound = float(np.quantile(boots, q))
    se = float(boots.std(ddof=1))
    z = NormalDist().inv_cdf
    keep = bound > 0 if side == 'greater' else bound < 0
    return {'statistic': float(np.sum(S) / np.sum(N)), 'bound': bound, 'bound_quantile': q, 'boot_se': se,
            'mde_80': float((z(1 - alpha) + z(0.80)) * se), 'verdict': 'KEEP' if keep else 'DROP'}


test_ratio.__test__ = False          # not a pytest test


# -- the cells ----------------------------------------------------------------------------------

def _base_row(cfg, n, now_iso, hypothesis, cell, metric, stat_name, dates, res) -> dict:
    sc = cfg['scoring']
    man = manifest_path(cfg)
    dates = pd.DatetimeIndex(dates)
    return {'n': n, 'date': now_iso[:10], 'hypothesis': hypothesis, 'cell': cell, 'metric': metric,
            'statistic_name': stat_name, 'n_events': len(dates), 'n_dates': int(dates.nunique()),
            'n_business_days': len(pd.bdate_range(dates.min(), dates.max())),
            'first_event': str(dates.min().date()), 'last_event': str(dates.max().date()),
            **{k: res[k] for k in ('statistic', 'bound', 'bound_quantile', 'boot_se', 'mde_80', 'verdict')},
            'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': sc['resamples'], 'seed': sc['seed'],
            'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
            'artifact_manifest_sha256': sha256(man) if os.path.exists(man) else ''}


CELL1_FILE = 'cell1_events.csv.gz'
CELL1_COLS = ['price', 'pair', 'date', 'dir', 'side', 'close', 'close_t3', 'atr', 'date_t3', 'r', 'spread_pips']


def mean_close_spread(d1_all: dict, ev: pd.DataFrame) -> dict:
    """Per pair: the mean spread (pips) of the valid D1 closing bars from its first to its last event."""
    from .daily import valid
    out = {}
    for pair, g in ev.groupby('pair'):
        v = valid(d1_all[pair])
        out[pair] = float(v.loc[g['date'].min():g['date'].max(), 'spread_close_pips'].mean())
    return out


def rule_inputs(cfg: dict, d1_all: dict | None = None) -> tuple:
    """(mid events with outcomes, bid events with outcomes), eligible only (reason ''), each with
    the pair's mean closing-bar spread in pips (descriptive use only)."""
    from . import events as E
    if d1_all is None:
        from .daily import build_all
        from .data_loader import load_all
        d1_all = build_all(*load_all(cfg), cfg)
    out = []
    for price in ('mid', 'bid'):
        ev, _ = E.build_events(d1_all, cfg, price)
        ev = E.outcomes(ev[ev['reason'] == ''].reset_index(drop=True), d1_all, price)
        ev['spread_pips'] = ev['pair'].map(mean_close_spread(d1_all, ev))
        ev['price'] = price
        out.append(ev)
    return tuple(out)


def load_cell1(cfg: dict) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), CELL1_FILE))
    d['date'] = pd.to_datetime(d['date'])
    return d


def score_rule(cfg: dict | None = None, now_iso: str | None = None, d1_all: dict | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if _json(cfg, LABELS_JSON) is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_bb.labels')
    if CELL_RULE in {r['hypothesis'] for r in read_registry(cfg)}:
        raise AlreadyScored(f'{CELL_RULE} is already scored; each cell is scored once')
    mid, bid = rule_inputs(cfg, d1_all)
    sc = cfg['scoring']
    S, N = day_sums(mid['date'], mid['r'])
    res = test_ratio(S, N, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'], 'greater')
    bid_stat = float(bid['r'].mean())
    gap = abs(res['statistic'] - bid_stat)
    flag = gap > float(sc['parity_tolerance_atr'])
    row = _base_row(cfg, 1, now_iso, CELL_RULE, 1, 'mean_outcome_atr', 'mean r (ATR20 units)', mid['date'], res)
    row.update(bid_statistic=bid_stat, bid_n_events=len(bid), parity_gap=gap, parity_flag='FLAG' if flag else '',
               notes=(f"Cell 1, the published rule with no fitted parameter: every eligible event "
                      f"{row['first_event']}..{row['last_event']} (window fixed by the data rules), mean "
                      f"counter-trend {cfg['horizon_days']}-day mid outcome in ATR{cfg['atr_period']} units, one-sided "
                      f"day-block bootstrap ({sc['block_days']}-day blocks), alpha 0.05/2. Bid parity: bid statistic "
                      f"{bid_stat:+.4f} on {len(bid)} bid events, gap {gap:.4f} ATR"
                      f"{' > tolerance: FLAGGED, verdict follows the mid test alone' if flag else ''}. "
                      f'Result is {GROSS}. {PRIOR_FACTS}'))
    os.makedirs(results_dir(cfg), exist_ok=True)
    pd.concat([mid[CELL1_COLS], bid[CELL1_COLS]], ignore_index=True).to_csv(
        os.path.join(results_dir(cfg), CELL1_FILE), index=False, compression='gzip')
    append_row(cfg, row)
    return row


def score_model(cfg: dict | None = None, now_iso: str | None = None) -> dict:
    from .walk_forward import load_oos
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    run = _json(cfg, RUN_JSON)
    if run is None:
        raise StageOrderError('run the walk-forward first: python -m src.fx_bb.walk_forward')
    if CELL_MODEL in {r['hypothesis'] for r in read_registry(cfg)}:
        raise AlreadyScored(f'{CELL_MODEL} is already scored; each cell is scored once')
    oos = load_oos(cfg).sort_values(['date', 'pair'], kind='mergesort')
    y = oos['y'].to_numpy(float)
    ll_m, ll_p = row_logloss(y, oos['p_lightgbm'].to_numpy(float)), row_logloss(y, oos['p_prior'].to_numpy(float))
    sc = cfg['scoring']
    S, N = day_sums(oos['date'], ll_m - ll_p)
    res = test_ratio(S, N, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'], 'less')
    row = _base_row(cfg, 2, now_iso, CELL_MODEL, 2, 'binary_log_loss', 'mean log-loss difference (LightGBM - base rate)',
                    oos['date'], res)
    n_folds = int(oos['fold'].nunique())
    row.update(n_folds=n_folds, model_mean=float(ll_m.mean()), reference_mean=float(ll_p.mean()),
               accuracy=float(np.mean((oos['p_lightgbm'].to_numpy(float) > 0.5) == (y == 1))),
               notes=(f"Cell 2: one LightGBM pooled over the six pairs vs the fit slice's up-rate, out-of-sample events "
                      f"{row['first_event']}..{row['last_event']} in {n_folds} expanding 12-month folds "
                      f"(10-business-day purge + embargo), day-block bootstrap ({sc['block_days']}-day blocks), "
                      f'alpha 0.05/2. Result is {GROSS}. {PRIOR_FACTS}'))
    append_row(cfg, row)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description='score one fx-bb cell, once')
    ap.add_argument('--cell', choices=['rule', 'model'], required=True)
    a = ap.parse_args(argv)
    r = score_rule() if a.cell == 'rule' else score_model()
    print(f"{r['hypothesis']}: {r['statistic_name']} {r['statistic']:+.5f} on {r['n_events']} events / {r['n_dates']} "
          f"dates, bound (q={r['bound_quantile']:.3f}) {r['bound']:+.5f}, mde_80 {r['mde_80']:.5f} -> {r['verdict']}")
    if a.cell == 'rule':
        print(f"bid statistic {r['bid_statistic']:+.5f} on {r['bid_n_events']} events, parity gap {r['parity_gap']:.4f}"
              f" {r['parity_flag']}")
    else:
        print(f"log-loss {r['model_mean']:.5f} vs {r['reference_mean']:.5f}, accuracy {r['accuracy']:.2%}")


if __name__ == '__main__':
    main()
