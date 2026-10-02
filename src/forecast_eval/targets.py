"""Horizon targets and the cost breakeven (design D3, D6).

Every target is built from LOG returns over h bars, because log returns add
across steps: the h-bar target is exactly the sum of h one-bar targets, which
keeps the purge and uniqueness arithmetic in splits.py exact.

    direction    1 if log(close[t+h] / close[t]) > 0, 0 if < 0, NaN if == 0
    return_pct   100 * log(close[t+h] / close[t])
    volatility   |return_pct|

The target at t reads close[t+h] and nothing else from the future. The last h
rows have no target and are NaN.

Scaling to percent happens in exactly ONE place, `_to_pct`, and a test asserts
the package multiplies by 100 nowhere else. (The production invariant "the * 100
lives only in src/features.py" governs the production pipeline; this is a
separate system with its own single site.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_horizon(h: int) -> int:
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError(f'horizon must be a positive integer number of bars, got {h!r}')
    return int(h)


def _to_pct(x):
    """The package's single scaling site (see module docstring)."""
    return x * 100.0


def log_return(close: pd.Series, h: int) -> pd.Series:
    """log(close[t+h] / close[t]); NaN for the last h rows."""
    h = _check_horizon(h)
    lc = np.log(close.astype(float))
    return lc.shift(-h) - lc


def direction_target(close: pd.Series, h: int):
    """(target, n_zero_moves). Zero moves are NaN -- excluded from directional
    accuracy, never assigned to a class -- and counted so every report can say
    how many were dropped."""
    lr = log_return(close, h)
    target = pd.Series(np.where(lr > 0, 1.0, np.where(lr < 0, 0.0, np.nan)),
                       index=close.index, name=f'dir_h{h}')
    target[lr.isna()] = np.nan
    n_zero = int((lr == 0).sum())
    return target, n_zero


def return_pct_target(close: pd.Series, h: int) -> pd.Series:
    return _to_pct(log_return(close, h)).rename(f'ret_pct_h{h}')


def volatility_pct_target(close: pd.Series, h: int) -> pd.Series:
    return return_pct_target(close, h).abs().rename(f'vol_pct_h{h}')


def mean_abs_move(close: pd.Series, h: int) -> float:
    """E|close[t+h] - close[t]| in PRICE units, over the rows that have a
    target. The breakeven needs the cost and the move in the same units, and a
    spread is quoted in price units."""
    h = _check_horizon(h)
    move = (close.shift(-h) - close).dropna()
    if move.empty:
        raise ValueError(f'no complete {h}-bar windows in the series')
    return float(move.abs().mean())


def breakeven_accuracy(mean_abs_move_price: float, cost_price: float) -> float:
    """Directional accuracy at which a sign strategy nets zero after cost.

    With symmetric |move| m and round-trip cost c per trade, expected P&L per
    trade is (2*acc - 1) * m - c, which is zero at acc = 0.5 + c / (2m).
    """
    if mean_abs_move_price <= 0:
        raise ValueError('mean absolute move must be positive')
    if cost_price < 0:
        raise ValueError('cost cannot be negative')
    return 0.5 + cost_price / (2.0 * mean_abs_move_price)


def breakeven_for_horizon(close: pd.Series, h: int, cost_price: float) -> float:
    return breakeven_accuracy(mean_abs_move(close, h), cost_price)
