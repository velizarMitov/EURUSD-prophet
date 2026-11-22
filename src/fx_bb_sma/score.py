"""One-shot scoring of the two fx-bb-sma-exit cells (spec fx-bb-sma-exit/evaluation; design D8).

    python -m src.fx_bb_sma.score --cell rule     Cell 1  sma_exit_rule_net_r              (mean net R > 0)
    python -m src.fx_bb_sma.score --cell model    Cell 2  sma_exit_lightgbm_vs_base_rate   (log-loss below the base rate)

Both use fx_bb's day-block ratio bootstrap (read-only) keyed by ENTRY business day, with blocks of
10 business days (a trade lives up to 10), the same blocks for the daily sums and counts.
Cell 1 KEEP when the alpha-quantile > 0; Cell 2 KEEP when the (1 - alpha)-quantile < 0; alpha 0.05/2.
Results are net of the recorded bar spread only: never a trading result.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_bb.score import day_sums, ratio_block_means, test_ratio  # noqa: F401  (read-only reuse)
from ..fx_fib.artifacts import sha256
from ..fx_session.artifacts import manifest_path
from ..fx_session.score import row_logloss
from . import config_digest, feature_list_digest, load_config
from . import trades as T
from .registry import (CELL_MODEL, CELL_RULE, LABELS_JSON, RUN_JSON, AlreadyScored, ConfigChanged,  # noqa: F401
                       StageOrderError, append_row, guard, read_json, read_registry, results_dir)

COST = 'net of the recorded bar spread only (no slippage, swap or commission), not a trading result'
SEEN = ('These are the same first-breach events as fx-bb-reversion, whose 3-day hold was DROP on 2026-11-22 '
        '(mean +0.014 ATR): this family was designed after that result. Earlier fx families (fx-fibonacci, '
        'fx-session, fx-rates) were all DROP on 2026-11-21.')
CELL1_FILE = 'cell1_trades.csv.gz'
CELL1_COLS = ['pair', 'date', 'dir', 'side', 'close', 'sma', 'tp', 'sl', 'dist', 'entry_time', 'exit_reason',
              'exit_time', 'exit_date', 'exit_price', 'conflict', 'spread_entry_pips', 'spread_exit_pips', 'gross_pips',
              'net_pips', 'r_gross', 'r_net', 'r_ctrl', 'hold_hours']


def _base_row(cfg, n, now_iso, cell_name, cell, metric, stat_name, dates, res) -> dict:
    sc = cfg['scoring']
    man = manifest_path(cfg)
    dates = pd.DatetimeIndex(dates)
    return {'n': n, 'date': now_iso[:10], 'hypothesis': cell_name, 'cell': cell, 'metric': metric,
            'statistic_name': stat_name, 'n_trades': len(dates), 'n_dates': int(dates.nunique()),
            'n_business_days': len(pd.bdate_range(dates.min(), dates.max())), 'first_entry': str(dates.min().date()),
            'last_entry': str(dates.max().date()),
            **{k: res[k] for k in ('statistic', 'bound', 'bound_quantile', 'boot_se', 'mde_80', 'verdict')},
            'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': sc['resamples'], 'seed': sc['seed'],
            'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
            'artifact_manifest_sha256': sha256(man) if os.path.exists(man) else ''}


def scored_trades(cfg: dict, inputs: tuple | None = None) -> pd.DataFrame:
    """Every eligible trade with its result."""
    ctxs, d1_all = inputs if inputs is not None else T.load_inputs(cfg)
    tr, _ = T.build_stream(ctxs, d1_all, cfg)
    return T.results(tr[tr['reason'] == ''].reset_index(drop=True), ctxs, cfg)


def load_cell1(cfg: dict) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(results_dir(cfg), CELL1_FILE))
    for c in ('date', 'exit_date'):
        d[c] = pd.to_datetime(d[c])
    return d


def score_rule(cfg: dict | None = None, now_iso: str | None = None, inputs: tuple | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if read_json(cfg, LABELS_JSON) is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_bb_sma.labels')
    if CELL_RULE in {r['hypothesis'] for r in read_registry(cfg)}:
        raise AlreadyScored(f'{CELL_RULE} is already scored; each cell is scored once')
    tr = scored_trades(cfg, inputs)
    sc = cfg['scoring']
    S, N = day_sums(tr['date'], tr['r_net'])
    res = test_ratio(S, N, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'], 'greater')
    ctrl = float(tr['r_ctrl'].mean())
    gap = abs(res['statistic'] - ctrl)
    flag = gap > float(sc['parity_tolerance_r'])
    share = tr['exit_reason'].value_counts(normalize=True)
    row = _base_row(cfg, 1, now_iso, CELL_RULE, 1, 'mean_net_r', 'mean net R per trade', tr['date'], res)
    row.update(ctrl_statistic=ctrl, parity_gap=gap, parity_flag='FLAG' if flag else '',
               share_take_profit=float(share.get('take profit', 0.0)), share_stop_loss=float(share.get('stop loss', 0.0)),
               share_time_stop=float(share.get('time stop', 0.0)),
               notes=(f"Cell 1, the owner's trade with no fitted parameter: entry at the D1 mid close of a first Bollinger "
                      f"(20, 2) breach, take profit at SMA20[t], stop at the same distance, stop wins a same-bar conflict, "
                      f"time stop at the close of t+10; every eligible trade {row['first_entry']}..{row['last_entry']}; "
                      f"mean net R, day-block bootstrap ({sc['block_days']}-day blocks), alpha 0.05/2. Bid/ask control "
                      f"{ctrl:+.4f} R, gap {gap:.4f}{' > tolerance: FLAGGED, verdict follows the mid result' if flag else ''}. "
                      f'Result is {COST}. {SEEN}'))
    os.makedirs(results_dir(cfg), exist_ok=True)
    tr[CELL1_COLS].to_csv(os.path.join(results_dir(cfg), CELL1_FILE), index=False, compression='gzip')
    append_row(cfg, row)
    return row


def score_model(cfg: dict | None = None, now_iso: str | None = None) -> dict:
    from .walk_forward import load_oos
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if read_json(cfg, RUN_JSON) is None:
        raise StageOrderError('run the walk-forward first: python -m src.fx_bb_sma.walk_forward')
    if CELL_MODEL in {r['hypothesis'] for r in read_registry(cfg)}:
        raise AlreadyScored(f'{CELL_MODEL} is already scored; each cell is scored once')
    oos = load_oos(cfg).sort_values(['date', 'pair'], kind='mergesort')
    y = oos['y'].to_numpy(float)
    ll_m = row_logloss(y, oos['p_lightgbm'].to_numpy(float))
    ll_p = row_logloss(y, oos['p_prior'].to_numpy(float))
    sc = cfg['scoring']
    S, N = day_sums(oos['date'], ll_m - ll_p)
    res = test_ratio(S, N, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'], 'less')
    n_folds = int(oos['fold'].nunique())
    row = _base_row(cfg, 2, now_iso, CELL_MODEL, 2, 'binary_log_loss', 'mean log-loss difference (LightGBM - base rate)',
                    oos['date'], res)
    row.update(n_folds=n_folds, model_mean=float(ll_m.mean()), reference_mean=float(ll_p.mean()),
               accuracy=float(np.mean((oos['p_lightgbm'].to_numpy(float) > 0.5) == (y == 1))),
               notes=(f"Cell 2: one LightGBM pooled over the six pairs (ten fx-bb-reversion features) vs the fit slice's "
                      f"win rate; target = net result > 0; out-of-sample trades {row['first_entry']}..{row['last_entry']} in "
                      f"{n_folds} expanding 12-month folds, 10-business-day gap from each trade's exit; day-block "
                      f"bootstrap ({sc['block_days']}-day blocks), alpha 0.05/2. Result is {COST}. {SEEN}"))
    append_row(cfg, row)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description='score one fx-bb-sma cell, once')
    ap.add_argument('--cell', choices=['rule', 'model'], required=True)
    a = ap.parse_args(argv)
    r = score_rule() if a.cell == 'rule' else score_model()
    print(f"{r['hypothesis']}: {r['statistic_name']} {r['statistic']:+.5f} on {r['n_trades']} trades / {r['n_dates']} "
          f"dates, bound (q={r['bound_quantile']:.3f}) {r['bound']:+.5f}, mde_80 {r['mde_80']:.5f} -> {r['verdict']}")
    if a.cell == 'rule':
        print(f"exits: take profit {r['share_take_profit']:.1%}, stop loss {r['share_stop_loss']:.1%}, time stop "
              f"{r['share_time_stop']:.1%}; bid/ask control {r['ctrl_statistic']:+.4f} R (gap {r['parity_gap']:.4f}) "
              f"{r['parity_flag']}")
    else:
        print(f"log-loss {r['model_mean']:.5f} vs {r['reference_mean']:.5f}, accuracy {r['accuracy']:.2%}")


if __name__ == '__main__':
    main()
