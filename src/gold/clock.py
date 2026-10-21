"""The gold bar-label clock, established from the weekly market open (design D3).

Gold (CME metals hours) opens Sunday 18:00 America/New_York -- one hour after
forex. Read on the broker's Europe/Berlin wall clock that instant is Monday
00:00, or Sunday 23:00 in the March / late-October weeks when New York and
Berlin are 5 h apart instead of 6. Measured on the 2013-2026 H1 history: 643 of
701 weekly opens at 00:00 and 54 at 23:00.

A run whose weekly-open labels stop landing on the predicted hour stops before
any feature is built: a broker clock change would otherwise shift every
eligibility rule by an hour without a trace.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.forecast_eval.m15_data import weekly_opens

LABEL_TZ = 'Europe/Berlin'
MARKET_OPEN_TZ = 'America/New_York'
MARKET_OPEN_HOUR = 18                      # metals open Sunday 18:00 New York


class GoldClockMismatch(RuntimeError):
    pass


def expected_open_hour(ts) -> int:
    """The label hour the weekly open carries on the Berlin clock, from the
    Sunday at or before `ts` (a holiday-delayed open keeps its week's hour)."""
    day = pd.Timestamp(ts).date()
    sunday = day - timedelta(days=(day.weekday() + 1) % 7)
    ny = datetime(sunday.year, sunday.month, sunday.day, MARKET_OPEN_HOUR, tzinfo=ZoneInfo(MARKET_OPEN_TZ))
    return ny.astimezone(ZoneInfo(LABEL_TZ)).hour


def evidence(index) -> dict:
    opens = weekly_opens(index)
    if len(opens) == 0:
        return {'n_weeks': 0, 'match_rate': float('nan')}
    hits = [int(ts.hour) == expected_open_hour(ts) for ts in opens]
    return {'n_weeks': len(opens), 'match_rate': float(sum(hits) / len(hits)),
            'label_tz': LABEL_TZ, 'market_open': f'Sunday {MARKET_OPEN_HOUR}:00 {MARKET_OPEN_TZ}',
            'first_open': str(opens[0]), 'last_open': str(opens[-1])}


def verify(index, threshold: float = 0.95) -> dict:
    """The clock evidence, or GoldClockMismatch below `threshold` (registered 0.95)."""
    ev = evidence(index)
    if not ev['n_weeks']:
        raise GoldClockMismatch('no weekend boundary in the frame: the label clock cannot be established')
    if not ev['match_rate'] >= threshold:
        raise GoldClockMismatch(
            f'gold bar labels do not match {LABEL_TZ}: only {ev["match_rate"]:.1%} of {ev["n_weeks"]} '
            f'weekly opens land on the predicted hour (threshold {threshold:.0%}). '
            'Establish the clock before any eligibility rule is applied.')
    return ev
