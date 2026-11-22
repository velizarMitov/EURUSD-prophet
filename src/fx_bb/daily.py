"""D1 bars that close at 17:00 New York, built from the Berlin-labelled H1 caches
(spec fx-bb-reversion/market-data; design D3).

Each H1 label is the broker's Berlin wall clock (stored UTC-localised as-is). It is localised to
Europe/Berlin and converted to America/New_York; its trading date is (New York time + 7 h).date,
so a day runs 17:00 New York -> 17:00 New York:

  * the last bar of a day is labelled 22:00 Berlin (21:00 in the US/EU DST-mismatch weeks);
  * the 23:00 Berlin rollover bar (22:00 in mismatch weeks) opens the NEXT day, so the Sunday
    reopen belongs to Monday and no Sunday day exists.

Each row holds the bid open/high/low/close, the MID close (last bar's bid close plus half its
spread), n_h1 and the closing bar's spread. Days with fewer than `min_h1_per_d1` bars, and
Saturday stubs, are dropped. A closing bar whose spread exceeds `max_spread_pips` voids the mid
close (NaN) and the day is not valid; the bid prices stay. Nothing is filled or shifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_dates(index, cfg: dict) -> pd.DatetimeIndex:
    lab = pd.DatetimeIndex(index).tz_localize(None).tz_localize(cfg['data']['label_tz'], ambiguous='raise',
                                                                  nonexistent='raise')
    local = lab.tz_convert(cfg['data']['day_tz'])
    shift = pd.Timedelta(hours=24 - int(cfg['data']['day_close_hour']))
    return (local + shift).tz_localize(None).normalize()


def corrupt_spread(spread_points, point: float, pip: float, cfg: dict):
    return np.asarray(spread_points, dtype=float) * point / pip > float(cfg['data']['max_spread_pips'])


def d1_ny(h1: pd.DataFrame, point: float, pip: float, cfg: dict) -> pd.DataFrame:
    """D1 frame indexed by naive trading date. `close` is the mid close (NaN when corrupt)."""
    td = trading_dates(h1.index, cfg)
    g = h1.groupby(td.values)
    last_spread = g['spread'].last()
    d1 = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                       'close_bid': g['close'].last(), 'n_h1': g['close'].size(), 'spread_close': last_spread})
    d1['close'] = d1['close_bid'] + d1['spread_close'] * point / 2.0
    d1['corrupt_spread'] = corrupt_spread(d1['spread_close'], point, pip, cfg)
    d1.loc[d1['corrupt_spread'], 'close'] = np.nan
    d1['spread_close_pips'] = np.where(d1['corrupt_spread'], np.nan, d1['spread_close'] * point / pip)
    d1.index = pd.DatetimeIndex(d1.index, name='date')
    d1 = d1[(d1.index.dayofweek < 5) & (d1['n_h1'] >= int(cfg['data']['min_h1_per_d1']))]
    return d1[['open', 'high', 'low', 'close_bid', 'close', 'n_h1', 'spread_close', 'spread_close_pips',
               'corrupt_spread']]


def valid(d1: pd.DataFrame) -> pd.DataFrame:
    """The pair's valid days: a D1 bar with a usable mid close."""
    return d1[d1['close'].notna()]


def build_all(frames: dict, points: dict, cfg: dict) -> dict:
    return {p: d1_ny(frames[p], points[p], float(cfg['pip'][p]), cfg) for p in cfg['pairs']}
