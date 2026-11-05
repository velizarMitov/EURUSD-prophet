"""Fills, costs and the daily series (spec window-strategy: "Results are net of
cost"; design D4).

Prices are bid closes. A buy fills at bid + that bar's spread and a sell at the
bid, so a long pays the spread of its entry bar and a short the spread of its
exit bar. A round trip is charged at least `floor` (a fraction of the entry
price): the broker's window median on 2026-11-04 is 0.0062 %. Every trade
opens and closes inside the window, so no swap exists to charge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_returns(trades: pd.DataFrame, floor_pct: float) -> pd.DataFrame:
    """Adds `gross`, `cost` and `net` (fractions of the entry price) to each trade."""
    t = trades.copy()
    entry = t['entry_price'].to_numpy(float)
    side = t['side'].to_numpy(float)
    gross = side * (t['exit_price'].to_numpy(float) - entry) / entry
    paid = np.where(side > 0, t['entry_spread'].to_numpy(float), t['exit_spread'].to_numpy(float)) / entry
    cost = np.maximum(paid, floor_pct / 100)
    t['gross'], t['cost'], t['net'] = gross, cost, gross - cost
    return t


def daily_series(trades: pd.DataFrame, days: pd.DatetimeIndex, col: str = 'net') -> pd.Series:
    """One value per eligible day: the sum of that day's trade returns, else 0."""
    days = pd.DatetimeIndex(days).sort_values()
    out = pd.Series(0.0, index=days)
    if len(trades):
        s = trades.groupby('day')[col].sum()
        stray = s.index.difference(days)
        if len(stray):
            raise ValueError(f'{len(stray)} trade day(s) are not eligible days, e.g. {stray[0]}')
        out.loc[s.index] = s.to_numpy()
    return out


def simulate(trades: pd.DataFrame, days: pd.DatetimeIndex, floor_pct: float, stress_pct: float,
             gate: bool = True) -> dict:
    """Net and stress daily series of `trades` (gate-filtered unless gate=False)."""
    t = trades[trades['gate_open'].astype(bool)] if gate else trades
    net = trade_returns(t, floor_pct)
    stress = trade_returns(t, stress_pct)
    return {'trades': net, 'net': daily_series(net, days), 'stress': daily_series(stress, days),
            'gross': daily_series(net, days, 'gross')}
