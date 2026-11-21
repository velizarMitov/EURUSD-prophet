"""Triple-barrier labels on H1 (spec fx-fibonacci/barrier-labels).

For H1 bar t with close c_t and the ATR_D1 of the newest D1 bar closed by t's close:
    upper = c_t + m * ATR_D1        lower = c_t - m * ATR_D1
Touches are checked on bars t+1 .. t+H (high for the upper, low for the lower barrier):
    1  upper touched first          2  lower touched first
    0  neither within H bars, or both inside the same first-touch bar (a tie)
Horizons are counted in H1 bars, so weekends and missing hours add nothing. A bar whose
full horizon is not yet in the cache gets no label (NaN). GROSS of spread and slippage.

    python -m src.fx_fib.labeling      write results/fx_fib/labels.md
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import load_config, repo_path
from . import mtf_aligner as AL

GROSS_HEADER = 'Gross of spread and slippage — not a trading result.'
CLASS_NAMES = {0: 'neutral', 1: 'up', 2: 'down'}


def atr_d1_at(h1: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR_D1 of the newest D1 bar each H1 bar may see (NaN before the first)."""
    d1 = AL.build_d1(h1)
    d1['atr_d1'] = AL.atr(d1, period)
    return AL.attach_d1(h1.index, d1[['atr_d1', 'close_time']])['atr_d1'].to_numpy()


def triple_barrier(high, low, close, atr_d1, m: float, horizon: int) -> pd.DataFrame:
    """Columns: label (float, NaN when unlabelled), tie (bool), exit_offset (bars after t at
    which the label resolved; H for a time-out), upper, lower."""
    high, low, close, atr_d1 = (np.asarray(x, dtype=float) for x in (high, low, close, atr_d1))
    n = len(close)
    label = np.full(n, np.nan)
    tie = np.zeros(n, dtype=bool)
    exit_off = np.full(n, np.nan)
    upper = close + m * atr_d1
    lower = close - m * atr_d1
    k = n - horizon                                  # rows 0 .. k-1 have a complete horizon
    if k > 0:
        hw = sliding_window_view(high[1:], horizon)[:k]
        lw = sliding_window_view(low[1:], horizon)[:k]
        hit_up = hw >= upper[:k, None]
        hit_dn = lw <= lower[:k, None]
        big = horizon + 1
        first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), big)
        first_dn = np.where(hit_dn.any(axis=1), hit_dn.argmax(axis=1), big)
        lab = np.where(first_up < first_dn, 1.0, np.where(first_dn < first_up, 2.0, 0.0))
        is_tie = (first_up == first_dn) & (first_up < big)
        first = np.minimum(first_up, first_dn)
        off = np.where(first < big, first + 1, horizon).astype(float)
        ok = np.isfinite(atr_d1[:k])
        label[:k] = np.where(ok, lab, np.nan)
        tie[:k] = ok & is_tie
        exit_off[:k] = np.where(ok, off, np.nan)
    exit_close = np.full(n, np.nan)
    has = np.isfinite(exit_off)
    exit_close[has] = close[np.flatnonzero(has) + exit_off[has].astype(int)]
    return pd.DataFrame({'label': label, 'tie': tie, 'exit_offset': exit_off,
                         'upper': upper, 'lower': lower, 'exit_close': exit_close})


def build_labels(h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lab = cfg['labels']
    out = triple_barrier(h1['high'], h1['low'], h1['close'], atr_d1_at(h1, cfg['atr_period']),
                         lab['m_atr_d1'], lab['horizon_bars'])
    out.index = h1.index
    return out


def mean_uniqueness(labels: pd.DataFrame) -> float:
    """Average uniqueness (de Prado): for each labelled bar t, the mean over the bars its
    label spans (t+1 .. t+exit_offset) of 1 / (number of labels spanning that bar)."""
    ok = labels['label'].notna().to_numpy()
    starts = np.flatnonzero(ok) + 1
    ends = starts + labels['exit_offset'].to_numpy()[ok].astype(int)    # exclusive
    n = len(labels) + 1
    conc = np.zeros(n + 1)
    np.add.at(conc, starts, 1)
    np.add.at(conc, ends, -1)
    conc = np.cumsum(conc)[:n]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
    csum = np.r_[0.0, np.cumsum(inv)]
    u = (csum[ends] - csum[starts]) / (ends - starts)
    return float(u.mean()) if len(u) else float('nan')


def label_summary(labels: pd.DataFrame, train_end) -> dict:
    lab = labels['label']
    train = lab[(lab.index < train_end) & lab.notna()]
    full = lab.dropna()
    share = lambda s: {CLASS_NAMES[c]: float((s == c).mean()) for c in (0, 1, 2)}
    return {'n_labelled': int(len(full)), 'n_train': int(len(train)),
            'train_share': share(train), 'full_share': share(full),
            'ties': int(labels['tie'].sum()), 'tie_rate': float(labels['tie'].sum() / max(len(full), 1)),
            'mean_exit_bars': float(labels.loc[lab.notna(), 'exit_offset'].mean()),
            'mean_uniqueness': mean_uniqueness(labels)}


def write_report(cfg: dict | None = None, now_iso: str | None = None) -> str:
    from .data_loader import load_h1
    from .score import assert_config_matches_verdicts
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    lab = cfg['labels']
    train_end = pd.Timestamp(cfg['folds']['first_eval_start'], tz='UTC')
    lines = ['# fx-fibonacci triple-barrier labels', '', f'> **{GROSS_HEADER}**', '',
             f'Barriers ±{lab["m_atr_d1"]} × ATR_D1 from the H1 close, horizon {lab["horizon_bars"]} H1 bars, '
             'same-bar tie → 0. Train rows are those before the first evaluation fold '
             f'({cfg["folds"]["first_eval_start"]}).', '',
             '| Pair | Labelled | Train rows | Train neutral / up / down | All neutral / up / down '
             '| Ties | Mean exit (bars) | Mean uniqueness |',
             '|---|---|---|---|---|---|---|---|']
    flags = []
    for pair in cfg['pairs']:
        s = label_summary(build_labels(load_h1(pair, cfg), cfg), train_end)
        tr, fu = s['train_share'], s['full_share']
        lines.append(f"| {pair} | {s['n_labelled']:,} | {s['n_train']:,} | "
                     f"{tr['neutral']:.1%} / {tr['up']:.1%} / {tr['down']:.1%} | "
                     f"{fu['neutral']:.1%} / {fu['up']:.1%} / {fu['down']:.1%} | "
                     f"{s['ties']} ({s['tie_rate']:.2%}) | {s['mean_exit_bars']:.1f} | {s['mean_uniqueness']:.4f} |")
        low = [c for c, v in tr.items() if v < lab['min_class_share']]
        if low:
            flags.append(f'{pair}: train class(es) below {lab["min_class_share"]:.0%}: {low}')
    lines += ['', '## Class-balance check', '',
              ('All train classes are at or above ' f'{lab["min_class_share"]:.0%}.' if not flags
               else 'FAILED — amend the label config before any model trains: ' + '; '.join(flags)),
              '', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'labels.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-fibonacci label report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
