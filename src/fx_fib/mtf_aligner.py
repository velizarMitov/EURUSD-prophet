"""D1 bars built from H1, and the causal H1 <- D1 join (spec fx-fibonacci/market-data).

Labels are the broker's Berlin wall clock stored UTC-localised, so a label's calendar
date IS the broker trading date. An H1 bar labelled t closes at t + 1 h.

A D1 bar closes at the midnight that ends its date. That depends only on the date, so
a history cut off mid-day can never make a half-built day look closed. An H1 bar may
use a D1 bar only when the D1 close is at or before the H1 close:

    Wed 14:00 bar (closes 15:00 Wed)  -> newest usable D1 is Tuesday
    Wed 23:00 bar (closes 00:00 Thu)  -> Wednesday is complete and usable
    Fri 22:00 bar (closes 23:00 Fri)  -> Thursday; Friday becomes usable at the
                                         week's first bar, Sun 23:00 (closes Mon 00:00)

Reopen evenings (Sunday, and the holiday reopen bars of 25 December / 1 January) are
part of the NEXT trading date, so no one-hour "day" ever distorts ATR_D1 or a swing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..h1_features import wilder_atr

ONE_HOUR = pd.Timedelta(hours=1)
ONE_DAY = pd.Timedelta(days=1)


REOPEN_HOUR = 22


def trading_date(index) -> pd.DatetimeIndex:
    """Each H1 label's trading date. A calendar date whose FIRST bar is labelled
    22:00 or later holds only a market reopen (every Sunday, and the evenings of
    25 December and 1 January), so its bars belong to the next date. Decided from
    the date's first bar, so it is causal."""
    idx = pd.DatetimeIndex(index)
    if (idx.dayofweek == 5).any():
        raise ValueError('Saturday bars found: the label clock is not the broker clock')
    day = idx.normalize()
    first_hour = pd.Series(idx.hour, index=idx).groupby(day).transform('first').to_numpy()
    return day + pd.to_timedelta(np.where(first_hour >= REOPEN_HOUR, 1, 0), unit='D')


def build_d1(h1: pd.DataFrame) -> pd.DataFrame:
    """D1 OHLC by trading date, plus `close_time` (next midnight) and `n_h1`."""
    td = trading_date(h1.index)
    g = h1.groupby(td)
    d1 = pd.DataFrame({
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'n_h1': g['close'].size(),
    })
    d1.index.name = 'date'
    d1['close_time'] = d1.index + ONE_DAY
    return d1


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Causal Wilder ATR of an OHLC frame (shared with the production H1 features)."""
    return wilder_atr(df['high'], df['low'], df['close'], period)


def attach_d1(h1_index, d1_values: pd.DataFrame) -> pd.DataFrame:
    """For every H1 label, the row of `d1_values` (indexed like build_d1, with a
    `close_time` column) whose D1 bar is the newest one closed at or before the H1
    bar's close. Adds `d1_date`. Rows before any closed D1 bar are NaN."""
    idx = pd.DatetimeIndex(h1_index)
    left = pd.DataFrame({'_h1_close': idx + ONE_HOUR, '_h1_row': np.arange(len(idx))})
    right = d1_values.reset_index().rename(columns={'date': 'd1_date'}).sort_values('close_time')
    out = pd.merge_asof(left.sort_values('_h1_close'), right, left_on='_h1_close', right_on='close_time',
                        direction='backward', allow_exact_matches=True)
    out = out.sort_values('_h1_row').drop(columns=['_h1_close', '_h1_row', 'close_time'])
    out.index = idx
    return out
