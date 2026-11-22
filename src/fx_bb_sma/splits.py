"""Expanding 12-month folds over trade entry dates, the gap measured from each trade's EXIT date
(spec fx-bb-sma-exit/evaluation; design D9).

A fold trains only on trades whose exit date is more than purge + embargo = 10 business days before
the fold start; the last 15 % of those (by entry date) is the early-stopping slice, again with a
10-business-day gap after the fit trades' exits. This is fx_bb.splits.make_folds (read-only) with
the exit date in the role of the outcome date.
"""

from __future__ import annotations

from ..fx_bb.splits import Fold, bdays_between, fold_bounds  # noqa: F401  (re-exported)
from ..fx_bb.splits import make_folds as _make_folds


def make_folds(entry_dates, exit_dates, cfg: dict, last_date=None) -> list:
    return _make_folds(entry_dates, exit_dates, cfg, last_date)
