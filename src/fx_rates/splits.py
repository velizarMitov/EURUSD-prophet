"""Expanding walk-forward folds with a 5-day purge and a 5-day embargo (spec fx-rates/evaluation; D6).

Fold k evaluates the usable days in [start_k, start_k + fold_months), from first_eval_start, on
calendar boundaries shared by both pairs. Day d's target spans the closes of d+1 .. d+5, so
between consecutive slices purge_days + embargo_days usable days are dropped:

    ... fit ... | purge+embargo | ... val (last val_fraction) ... | purge+embargo | eval fold

The Fold dataclass and fold_arrays (fit-only RobustScaler) are fx_session's, imported read-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_session.splits import Fold, fit_scaler, fold_arrays  # noqa: F401  (re-exported)


def fold_bounds(cfg: dict, last_day: pd.Timestamp) -> list:
    f = cfg['folds']
    start = pd.Timestamp(f['first_eval_start'])
    out = []
    while start <= last_day:
        end = start + pd.DateOffset(months=int(f['fold_months']))
        out.append((start, end))
        start = end
    return out


def make_folds(days, cfg: dict, last_day: pd.Timestamp | None = None) -> list:
    days = pd.DatetimeIndex(days)
    if not days.is_monotonic_increasing:
        raise ValueError('days must be sorted')
    gap = int(cfg['folds']['purge_days']) + int(cfg['folds']['embargo_days'])
    frac = float(cfg['folds']['val_fraction'])
    pos = np.arange(len(days))
    out = []
    for k, (start, end) in enumerate(fold_bounds(cfg, last_day if last_day is not None else days.max())):
        ev = pos[(days >= start) & (days < end)]
        if not len(ev):
            continue
        before = pos[days < start][:-gap] if gap else pos[days < start]
        n_val = int(round(len(before) * frac))
        if n_val == 0 or n_val + gap >= len(before):
            continue
        val = before[-n_val:]
        fit = before[:-n_val - gap] if gap else before[:-n_val]
        out.append(Fold(k, start, end, fit, val, ev))
    return out
