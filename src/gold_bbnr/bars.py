"""Gold H1 bars and D1 bars that close at 17:00 New York (spec gold-bb-nyrange/gold-bars; design D3, D4).

The H1 cache of the gold family, results/gold/gold_h1.csv, is read-only: `time` plus bid OHLC, labels on
the broker's Berlin wall clock stored UTC-localised as-is. Bars outside the registered window
(2015-01-02 .. 2026-11-20 inclusive, by label date) are dropped. Every load runs the gold family's
weekly-open clock check and refuses below 95 %.

A D1 bar holds the H1 bars whose label, read as Berlin time and converted to New York time, plus 7 h,
falls on that date (fx_bb's convention): a day runs 17:00 -> 17:00 New York, so the 23:00 Berlin bar of a
DST-mismatch week opens the next day. A D1 bar is valid with >= 12 H1 bars on a weekday. Bid prices only;
gross of cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_bb.daily import trading_dates
from ..gold.clock import verify
from . import repo_path

OHLC = ['open', 'high', 'low', 'close']

# The last row of the gold H1 cache as pre-registered (commit 8d2f4bd). The gold logger keeps appending
# to that file, including the 22:00 bar of 2026-11-20 inside the window, so the study reads only the rows
# up to this label, and the digest check hashes only that byte prefix (see `pinned_bytes`).
PINNED_LAST = '2026-11-20T21:00:00+00:00'


def pinned_bytes(path: str, last: str = PINNED_LAST) -> bytes:
    """The file's bytes up to and including the line that starts with `last`. Without that line, the whole
    file, so its digest no longer matches and the gate refuses."""
    with open(path, 'rb') as fh:
        data = fh.read()
    i = data.find(b'\n' + last.encode())
    if i < 0:
        return data
    end = data.find(b'\n', i + 1)
    return data if end < 0 else data[:end + 1]


def _day_cfg(cfg: dict) -> dict:
    return {'data': {k: cfg['data'][k] for k in ('label_tz', 'day_tz', 'day_close_hour')}}


def clip_window(h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = pd.Timestamp(cfg['data']['window_start'], tz='UTC')
    b = pd.Timestamp(cfg['data']['window_end'], tz='UTC') + pd.Timedelta(days=1)
    return h1[(h1.index >= a) & (h1.index < b)]


def load_h1(cfg: dict, path: str | None = None) -> pd.DataFrame:
    """The window-clipped bid OHLC frame, after the clock check (GoldClockMismatch below the threshold)."""
    df = pd.read_csv(path or repo_path(cfg['paths_read']['gold_h1']), usecols=['time'] + OHLC)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index().astype(float)
    if not df.index.is_unique:
        raise ValueError('the gold H1 cache has duplicate labels')
    df = clip_window(df, cfg)
    df = df[df.index <= pd.Timestamp(PINNED_LAST)]
    verify(df.index, float(cfg['data']['clock_threshold']))
    return df


def d1_ny(h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """D1 frame indexed by the naive New York trading date: bid OHLC, n_h1, valid."""
    td = trading_dates(h1.index, _day_cfg(cfg))
    g = h1.groupby(td.values)
    d1 = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                       'close': g['close'].last(), 'n_h1': g['close'].size()})
    d1.index = pd.DatetimeIndex(d1.index, name='date')
    d1['valid'] = (d1.index.dayofweek < 5) & (d1['n_h1'] >= int(cfg['data']['min_h1_per_d1']))
    return d1


def valid(d1: pd.DataFrame) -> pd.DataFrame:
    """Valid days with the column names fx_bb's helpers expect (bid close twice: no mid on gold)."""
    v = d1[d1['valid']].copy()
    v['close_bid'] = v['close']
    return v


def h1_day_positions(h1: pd.DataFrame, cfg: dict) -> np.ndarray:
    """The NY trading date of every H1 bar (naive), aligned with h1's rows."""
    return trading_dates(h1.index, _day_cfg(cfg)).to_numpy()
