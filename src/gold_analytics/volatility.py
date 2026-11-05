"""Volatility and regime (design D7, spec gold-analytics/dashboard "Volatility and
regime"). All on D1 bars except intraday volatility (today's M5 bars).

  ATR(14)        Wilder: seeded with the mean of the first 14 true ranges
  realised vol   std of D1 log returns over the window x sqrt(252), in %
  intraday vol   sqrt of today's summed squared M5 log returns, so far and
                 scaled to a full 23-hour gold day
  regime         today's 20-day realised vol against the 33rd / 67th
                 percentiles of its trailing 250 values: low, normal or high
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANNUAL = 252
M5_PER_DAY = 23 * 12


def true_range(d1: pd.DataFrame) -> pd.Series:
    pc = d1['close'].shift(1)
    tr = pd.concat([d1['high'] - d1['low'], (d1['high'] - pc).abs(), (d1['low'] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = d1['high'].iloc[0] - d1['low'].iloc[0]
    return tr


def atr(d1: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = true_range(d1).to_numpy(float)
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, len(tr)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return pd.Series(out, index=d1.index)


def log_returns(d1: pd.DataFrame) -> pd.Series:
    return np.log(d1['close'] / d1['close'].shift(1))


def realised(d1: pd.DataFrame, window: int) -> pd.Series:
    """Annualised realised volatility, %, over a rolling `window` of D1 returns."""
    return log_returns(d1).rolling(window).std() * math.sqrt(ANNUAL) * 100


def intraday(m5_today: pd.DataFrame) -> dict:
    if len(m5_today) < 2:
        return {'n_bars': int(len(m5_today)), 'so_far_pct': None, 'day_pct': None, 'annualised_pct': None}
    r = np.log(m5_today['close'] / m5_today['close'].shift(1)).dropna()
    so_far = math.sqrt(float((r ** 2).sum())) * 100
    day = so_far * math.sqrt(M5_PER_DAY / len(r))
    return {'n_bars': int(len(m5_today)), 'so_far_pct': so_far, 'day_pct': day,
            'annualised_pct': day * math.sqrt(ANNUAL)}


def regime(d1: pd.DataFrame, window: int = 20, lookback: int = 250, low_q: float = 0.33,
           high_q: float = 0.67) -> dict:
    rv = realised(d1, window).dropna()
    if len(rv) < 2:
        return {'regime': None, 'value': None, 'low': None, 'high': None, 'n': int(len(rv))}
    tail = rv.iloc[-lookback:]
    lo, hi = float(tail.quantile(low_q)), float(tail.quantile(high_q))
    v = float(rv.iloc[-1])
    name = 'high' if v > hi else ('low' if v < lo else 'normal')
    return {'regime': name, 'value': v, 'low': lo, 'high': hi, 'n': int(len(tail)),
            'percentile': float((tail <= v).mean())}


def range_vs_average(d1: pd.DataFrame, window: int = 20) -> dict:
    rng = d1['high'] - d1['low']
    if len(rng) < 2:
        return {'today_usd': None, 'average_usd': None, 'ratio': None}
    today, avg = float(rng.iloc[-1]), float(rng.iloc[-window - 1:-1].mean())
    return {'today_usd': today, 'average_usd': avg, 'ratio': today / avg if avg else None,
            'today_pct': today / float(d1['open'].iloc[-1]) * 100}


def summary(d1: pd.DataFrame, cfg: dict) -> dict:
    a = atr(d1, cfg['atr_period'])
    r = cfg['regime']
    return {
        'atr': {'usd': None if a.dropna().empty else float(a.iloc[-1]),
                'pct': None if a.dropna().empty else float(a.iloc[-1] / d1['close'].iloc[-1] * 100),
                'period': cfg['atr_period']},
        'realised': {str(w): (None if realised(d1, w).dropna().empty else float(realised(d1, w).iloc[-1]))
                     for w in cfg['realised_windows']},
        'regime': regime(d1, r['window'], r['lookback'], r['low_q'], r['high_q']),
        'range': range_vs_average(d1, r['window']),
    }
