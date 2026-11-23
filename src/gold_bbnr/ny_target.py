"""Part 2: gold's New York session-range target (spec gold-bb-nyrange/ny-range; design D8).

Dates are the Monday-Friday Berlin label dates of the gold H1 bars (tz-aware UTC midnight, as in
fx_nyvol). A day is ELIGIBLE (reason '') when its NY-close D1 bar is valid and the bars labelled 14:00
through 21:00 all exist; else `invalid day` or `incomplete session`.

    y = log((max high - min low of the bars labelled 15:00-21:00) / close of the 14:00 bar)      (bid)

`target` is the only function that reads a price after 14:00; the label report never calls it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_session.features import _window, hour_grid

REASONS = ('invalid day', 'incomplete session', 'warm-up')


def calendar_dates(h1: pd.DataFrame) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(h1.index).normalize().unique()
    return pd.DatetimeIndex(d[d.dayofweek < 5].sort_values(), name='date')


def grid(h1: pd.DataFrame, dates) -> dict:
    return hour_grid(h1, pd.DatetimeIndex(dates))


def eligibility(g: dict, dates, d1: pd.DataFrame, cfg: dict) -> pd.Series:
    """Reason per date, from bar presence and D1 validity only (no price is read)."""
    s = cfg['session']
    dates = pd.DatetimeIndex(dates)
    ok_d1 = d1['valid'].reindex(dates.tz_localize(None)).fillna(False).to_numpy(bool)
    have = ~np.isnan(g['close'][:, s['data_hour']:s['target_hours'][1] + 1]).any(axis=1)
    reason = np.where(~ok_d1, 'invalid day', np.where(~have, 'incomplete session', ''))
    return pd.Series(reason, index=dates, name='reason')


def target(g: dict, dates, cfg: dict) -> pd.Series:
    s = cfg['session']
    t = _window(g, *s['target_hours'])
    c14 = _window(g, s['data_hour'], s['data_hour'])['close']
    with np.errstate(all='ignore'):
        y = np.log((t['high'] - t['low']) / c14)
    return pd.Series(y, index=pd.DatetimeIndex(dates), name='y')
