"""Expanding 12-month walk-forward folds over the Cell 2 event dates (spec fx-bb-reversion/evaluation;
design D6).

Fold k tests the events dated in [start_k, start_k + 12 months), from 2015-01-01. Training uses
only events whose t+3 date is MORE than purge + embargo = 10 business days before the fold start;
the last 15 % of those (by date) is the early-stopping slice, and the fit slice keeps only events
whose t+3 date is more than 10 business days before the first validation date:

    ... fit ... | 10 bd after the last fit t+3 | ... val ... | 10 bd after the last val t+3 | test fold

Events of one date always fall in one slice. Nothing is shuffled. The Fold dataclass is
fx_session's, imported read-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_session.splits import Fold


def fold_bounds(cfg: dict, last_date: pd.Timestamp) -> list:
    f = cfg['folds']
    start = pd.Timestamp(f['first_eval_start'])
    out = []
    while start <= last_date:
        end = start + pd.DateOffset(months=int(f['fold_months']))
        out.append((start, end))
        start = end
    return out


def bdays_between(a, b) -> np.ndarray:
    """Business days from each date in `a` to `b` (np.busday_count, Monday to Friday)."""
    a = pd.DatetimeIndex(a).to_numpy().astype('datetime64[D]')
    return np.busday_count(a, np.datetime64(pd.Timestamp(b).date()))


def make_folds(dates, t3_dates, cfg: dict, last_date: pd.Timestamp | None = None) -> list:
    """Folds over date-sorted events (`dates`, their `t3_dates`), as positions into that order."""
    dates, t3 = pd.DatetimeIndex(dates), pd.DatetimeIndex(t3_dates)
    if not dates.is_monotonic_increasing:
        raise ValueError('events must be sorted by date')
    if t3.isna().any():
        raise ValueError('every event needs its t+3 date')
    gap = int(cfg['folds']['purge_days']) + int(cfg['folds']['embargo_days'])
    frac = float(cfg['folds']['val_fraction'])
    pos = np.arange(len(dates))
    out = []
    for k, (start, end) in enumerate(fold_bounds(cfg, last_date if last_date is not None else dates.max())):
        ev = pos[(dates >= start) & (dates < end)]
        if not len(ev):
            continue
        cand = pos[(dates < start) & (bdays_between(t3, start) > gap)]
        n_val = int(round(len(cand) * frac))
        if n_val == 0 or n_val >= len(cand):
            continue
        val = cand[-n_val:]
        v0 = dates[val[0]]
        val = cand[dates[cand] >= v0]                      # whole dates stay together
        head = cand[dates[cand] < v0]
        fit = head[bdays_between(t3[head], v0) > gap]
        if not len(fit):
            continue
        out.append(Fold(k, start, end, fit, val, ev))
    return out
