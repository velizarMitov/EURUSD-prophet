"""Time benchmarks without volume (spec gold-analytics/dashboard "Time benchmarks
without volume"). Hypothetical benchmarks for a chosen moment, never a trade.

  TWAP     the mean of the bar closes whose labels fall in [start, end)
  arrival  the close of the bar ending at the start (label = start - bar), and
           the move to the close of the bar ending at start + 15/30/60/120 min
"""

from __future__ import annotations

import pandas as pd


def twap(bars: pd.DataFrame, start, end) -> dict:
    sel = bars[(bars['time'] >= pd.Timestamp(start)) & (bars['time'] < pd.Timestamp(end))]
    if sel.empty:
        return {'twap': None, 'n_bars': 0}
    return {'twap': float(sel['close'].mean()), 'n_bars': int(len(sel)),
            'first_bar': sel['time'].iloc[0], 'last_bar': sel['time'].iloc[-1]}


def against(price: float | None, ref: float | None) -> dict:
    if price is None or ref is None:
        return {'usd': None, 'pct': None}
    return {'usd': price - ref, 'pct': (price / ref - 1) * 100}


def _close_ending_at(bars: pd.DataFrame, at: pd.Timestamp, bar_minutes: int):
    label = at - pd.Timedelta(minutes=bar_minutes)
    row = bars.loc[bars['time'] == label, 'close']
    return None if row.empty else float(row.iloc[0])


def arrival(bars: pd.DataFrame, start, bar_minutes: int, horizons) -> dict:
    s = pd.Timestamp(start)
    price = _close_ending_at(bars, s, bar_minutes)
    moves = []
    for h in horizons:
        p = _close_ending_at(bars, s + pd.Timedelta(minutes=h), bar_minutes)
        moves.append({'minutes': int(h), 'price': p, **against(p, price)})
    return {'start_label': s, 'arrival': price, 'bar_minutes': bar_minutes, 'moves': moves}
