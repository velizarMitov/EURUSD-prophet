"""Mid prices and the trading-day calendar (spec fx-session/market-data).

One clock for the whole family: Berlin bar labels (Sofia = Berlin + 1 h all year).

    entry  = close of the bar labelled 14:00 Berlin on D  (16:00 Sofia)
    exit   = close of the bar labelled 21:00 Berlin on D  (23:00 Sofia)

The label hours never move, including the US/EU DST-mismatch weeks (design D3). A day is
eligible only when it is Monday-Friday and BOTH scored pairs have every bar labelled
14:00 through 21:00 (design D4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..forecast_eval.m15_data import expected_open_label


def mid(df: pd.DataFrame, point: float, col: str = 'close') -> pd.Series:
    """Bid price plus half the bar's spread (points converted with the symbol's point size)."""
    return df[col] + df['spread'] * point / 2.0


def session_hours(cfg: dict) -> list:
    s = cfg['session']
    return list(range(int(s['entry_hour']), int(s['exit_hour']) + 1))


WEEKEND_GAP = pd.Timedelta(hours=30)


def verified_weeks(index) -> pd.Series:
    """For each week (keyed by its Monday), True when the week's weekend open sits on the
    Berlin label hour the project's clock rule predicts for that Sunday (23:00, or 22:00 in
    the US/EU DST-mismatch weeks). A week whose open is elsewhere, e.g. Monday 00:00 or
    Sunday 23:00 in a mismatch week, is not on the Berlin clock. Mid-week holiday reopens
    are ignored; a week with no weekend open is absent (unverified)."""
    idx = pd.DatetimeIndex(index)
    gap = np.r_[True, np.diff(idx.asi8) > WEEKEND_GAP.value]
    out = {}
    for t in idx[gap]:
        if t.dayofweek == 6:
            sunday = t.normalize()
            ok = t.hour == expected_open_label(sunday.date())
        elif t.dayofweek == 0:
            sunday, ok = t.normalize() - pd.Timedelta(days=1), False
        else:
            continue
        out[sunday + pd.Timedelta(days=1)] = bool(ok)
    return pd.Series(out, dtype=bool)


def session_calendar(frames: dict, cfg: dict) -> pd.DataFrame:
    """Every Monday-Friday calendar date seen in the scored pairs' labels, with
    `eligible` and, when not eligible, a `reason`: `incomplete session` when a scored
    pair lacks a bar labelled 14:00-21:00, else `unverified clock week` when any loaded
    pair's weekly open for that week is off the Berlin clock (owner decision 2026-11-21)."""
    hours = set(session_hours(cfg))
    dates = None
    per_pair = {}
    for pair in cfg['pairs']:
        idx = pd.DatetimeIndex(frames[pair].index)
        day = idx.normalize()
        present = pd.Series(idx.hour, index=idx).groupby(day).agg(lambda h: hours.issubset(set(h)))
        per_pair[pair] = present
        dates = present.index if dates is None else dates.union(present.index)
    cal = pd.DataFrame(index=pd.DatetimeIndex(dates, name='date'))
    cal = cal[cal.index.dayofweek < 5]
    ok = pd.Series(True, index=cal.index)
    for pair, present in per_pair.items():
        ok &= present.reindex(cal.index).fillna(False).astype(bool)
    monday = cal.index - pd.to_timedelta(cal.index.dayofweek, unit='D')
    clock_ok = pd.Series(True, index=cal.index)
    for frame in frames.values():
        wk = verified_weeks(frame.index)
        clock_ok &= pd.Series(wk.reindex(monday).fillna(False).astype(bool).to_numpy(), index=cal.index)
    cal['eligible'] = ok & clock_ok
    cal['reason'] = ''
    cal.loc[~clock_ok, 'reason'] = 'unverified clock week'
    cal.loc[~ok, 'reason'] = 'incomplete session'
    return cal


def bar_at(df: pd.DataFrame, dates: pd.DatetimeIndex, hour: int, col: str) -> pd.Series:
    """`col` of the bar labelled `hour`:00 on each date (NaN where that bar is missing)."""
    stamps = pd.DatetimeIndex(dates) + pd.Timedelta(hours=hour)
    return pd.Series(df[col].reindex(stamps).to_numpy(), index=pd.DatetimeIndex(dates, name='date'))
