"""Historical periods (spec gold-analytics/dashboard "Historical periods").

A period is the last N D1 trading days, or the D1 days between two dates. For
it: the average daily return and range, ATR at its end, realised volatility,
maximum drawdown, trend persistence (share of days with the previous day's
sign) and a histogram of daily returns in 0.25 % bins.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .volatility import ANNUAL, atr


def select(d1: pd.DataFrame, days: int | None = None, start=None, end=None) -> pd.DataFrame:
    """The period's rows; each keeps the previous close for its first return."""
    d = d1.assign(prev_close=d1['close'].shift(1))
    if start is not None or end is not None:
        t = d['time'].dt.tz_localize(None) if d['time'].dt.tz is not None else d['time']
        mask = pd.Series(True, index=d.index)
        if start is not None:
            mask &= t >= pd.Timestamp(start)
        if end is not None:
            mask &= t <= pd.Timestamp(end)
        return d[mask].reset_index(drop=True)
    return d.iloc[-days:].reset_index(drop=True)


def max_drawdown(closes: pd.Series) -> float | None:
    if closes.empty:
        return None
    return float((closes / closes.cummax() - 1).min() * 100)


def histogram(ret_pct: pd.Series, bin_pct: float = 0.25) -> list[dict]:
    r = ret_pct.dropna()
    if r.empty:
        return []
    lo = math.floor(r.min() / bin_pct) * bin_pct
    hi = math.ceil(r.max() / bin_pct) * bin_pct
    if hi <= lo:
        hi = lo + bin_pct
    edges = np.round(np.arange(lo, hi + bin_pct / 2, bin_pct), 10)
    counts, _ = np.histogram(r, bins=edges)
    return [{'from': float(a), 'to': float(b), 'count': int(c)} for a, b, c in zip(edges[:-1], edges[1:], counts)]


def period(d1: pd.DataFrame, days: int | None = None, start=None, end=None, atr_period: int = 14,
           bin_pct: float = 0.25) -> dict:
    p = select(d1, days, start, end)
    if p.empty:
        return {'days': 0}
    ret = (p['close'] / p['prev_close'] - 1) * 100
    logr = np.log(p['close'] / p['prev_close'])
    sign = np.sign(ret)
    pair = (sign != 0) & (sign.shift(1) != 0) & sign.shift(1).notna()
    a = atr(d1, atr_period)
    end_idx = d1.index[d1['time'] == p['time'].iloc[-1]][0]
    return {
        'days': int(len(p)),
        'first': p['time'].iloc[0], 'last': p['time'].iloc[-1],
        'mean_ret_pct': float(ret.mean()),
        'mean_range_pct': float(((p['high'] - p['low']) / p['open'] * 100).mean()),
        'mean_range_usd': float((p['high'] - p['low']).mean()),
        'atr_usd': None if np.isnan(a.loc[end_idx]) else float(a.loc[end_idx]),
        'realised_pct': float(logr.std() * math.sqrt(ANNUAL) * 100) if logr.notna().sum() > 1 else None,
        'total_ret_pct': float((p['close'].iloc[-1] / p['prev_close'].iloc[0] - 1) * 100)
        if pd.notna(p['prev_close'].iloc[0]) else None,
        'max_drawdown_pct': max_drawdown(p['close']),
        'persistence': {'n': int(pair.sum()), 'share': float((sign == sign.shift(1))[pair].mean()) if pair.any() else None},
        'histogram': histogram(ret, bin_pct),
    }
