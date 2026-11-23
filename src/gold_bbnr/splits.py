"""Walk-forward folds (spec gold-bb-nyrange/bb-reversion, bb-sma-trade, ny-range; design D6, D8).

Part 1 (events)  fx_bb's expanding 12-month folds from 2018-01-01 (read-only): training only on rows whose
                 outcome date (t+3 for 1a, the exit date for 1b) is more than purge + embargo = 10 business
                 days before the fold start; the last 15 % is the early-stopping slice. A fold whose fit +
                 validation rows number fewer than `min_train` (60) predicts nothing: its rows are `thin fold`.
                 Rows dated before the first fold are `before first fold`.
Part 2 (days)    fx_session's expanding 6-month folds from 2018-01-01 with a 1-day embargo (read-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_bb.splits import make_folds as _event_folds
from ..fx_session.splits import make_folds as _day_folds


def event_folds(dates, outcome_dates, cfg: dict, last_date=None) -> tuple:
    """(kept folds, reason per row: '' scored, 'thin fold', 'before first fold')."""
    dates = pd.DatetimeIndex(dates)
    view = {'folds': cfg['folds_bb']}
    folds = _event_folds(dates, pd.DatetimeIndex(outcome_dates), view, last_date)
    m = int(cfg['folds_bb']['min_train'])
    kept = [f for f in folds if len(f.fit) + len(f.val) >= m]
    reason = np.where(dates < pd.Timestamp(cfg['folds_bb']['first_eval_start']), 'before first fold', 'thin fold')
    for f in kept:
        reason[f.eval] = ''
    return kept, reason


def day_folds(days, cfg: dict, last_day=None) -> list:
    return _day_folds(pd.DatetimeIndex(days), {'folds': cfg['folds_ny']}, last_day)
