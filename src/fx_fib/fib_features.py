"""Fibonacci level, breakout, alignment and risk features (spec fx-fibonacci/structure-features).

Every distance is (H1 close - level) / ATR of the level's own timeframe. Pivot arrays are
the (n, 4) visible-pivot histories from swing_detector.pivot_history, oldest first, so
column 3 is the newest pivot P1 and column 2 is P0. Rows lacking the pivots a feature
needs are NaN; the assembler excludes them as warm-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leg_features(price: np.ndarray, idx: np.ndarray, close: np.ndarray, atr: np.ndarray,
                 cfg: dict, prefix: str) -> tuple[dict, dict]:
    """Features of the newest leg P0 -> P1 and of the last four pivots.

    Returns (features, levels). `levels` holds the raw 0.5 / 0.618 retracement and
    1.618 extension prices, which the alignment and risk features reuse."""
    f = cfg['fib']
    p0, p1 = price[:, 2], price[:, 3]
    span = p1 - p0
    with np.errstate(invalid='ignore', divide='ignore'):
        out = {
            f'{prefix}_swing_amp_atr': np.abs(span) / atr,
            f'{prefix}_swing_bars': np.where(np.isfinite(p0), (idx[:, 3] - idx[:, 2]).astype(float), np.nan),
            f'{prefix}_retracement_pct': (p1 - close) / span,
        }
        levels = {}
        for r in f['retracements']:
            level = p1 - r * span
            levels[r] = level
            out[f'{prefix}_dist_to_fib_{int(round(r * 1000)):03d}'] = (close - level) / atr
        ext = p1 + f['extension'] * span
        q0, q3 = price[:, 0], price[:, 3]
        target2 = q3 + f['target2'] * (q3 - q0)
        out[f'{prefix}_dist_to_ext_1618_w3'] = (close - ext) / atr
        out[f'{prefix}_dist_to_target2_0618'] = (close - target2) / atr
        out[f'{prefix}_price_band_width_atr'] = np.abs(ext - target2) / atr
    levels['ext'] = ext
    return out, levels


def breakouts(high: np.ndarray, low: np.ndarray, close: np.ndarray, ns=(1, 2, 4)) -> dict:
    """+1 when the close is above the highest high of the previous n bars, -1 below the
    lowest low, 0 otherwise; NaN until n previous bars exist."""
    h, l = pd.Series(high), pd.Series(low)
    out = {}
    for n in ns:
        prev_hi = h.shift(1).rolling(n, min_periods=n).max().to_numpy()
        prev_lo = l.shift(1).rolling(n, min_periods=n).min().to_numpy()
        val = np.where(close > prev_hi, 1.0, np.where(close < prev_lo, -1.0, 0.0))
        out[f'h1_breakout_{n}'] = np.where(np.isfinite(prev_hi), val, np.nan)
    return out


def close_vs_extreme_bars(kind: np.ndarray, idx: np.ndarray, high: np.ndarray, low: np.ndarray,
                          close: np.ndarray, atr: np.ndarray) -> dict:
    """The H1 close against the low of the newest leg's high bar and the high of its low
    bar, in ATR_H1. The last two pivots always hold one high and one low."""
    k3, k2 = kind[:, 3], kind[:, 2]
    ok = (k2 != 0) & (k3 != 0)
    hi_bar = np.where(k3 == 1, idx[:, 3], idx[:, 2])
    lo_bar = np.where(k3 == -1, idx[:, 3], idx[:, 2])
    hb = np.clip(hi_bar, 0, None)
    lb = np.clip(lo_bar, 0, None)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = (close - low[hb]) / atr
        b = (close - high[lb]) / atr
    return {'h1_close_vs_high_bar_low': np.where(ok, a, np.nan),
            'h1_close_vs_low_bar_high': np.where(ok, b, np.nan)}


def alignment_flag(breakout_1: np.ndarray, close: np.ndarray, d1_level_618: np.ndarray,
                   d1_ext: np.ndarray, atr_d1: np.ndarray, zone_atr: float) -> np.ndarray:
    """The 1-bar breakout direction when the H1 close sits inside the zone around the D1
    0.618 retracement or the D1 1.618 extension; 0 elsewhere."""
    with np.errstate(invalid='ignore'):
        zone = zone_atr * atr_d1
        inside = (np.abs(close - d1_level_618) <= zone) | (np.abs(close - d1_ext) <= zone)
    ok = np.isfinite(breakout_1) & np.isfinite(d1_level_618) & np.isfinite(d1_ext) & np.isfinite(atr_d1)
    return np.where(ok, np.where(inside, breakout_1, 0.0), np.nan)


def risk_features(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
                  h1_level_50: np.ndarray, cfg: dict) -> dict:
    """Distances to the 5-bar price square and the reward:risk to the H1 leg's 50 % level."""
    n = cfg['fib']['price_square_bars']
    floor = cfg['fib']['rr_divisor_floor_atr']
    lo5 = pd.Series(low).rolling(n, min_periods=n).min().to_numpy()
    hi5 = pd.Series(high).rolling(n, min_periods=n).max().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        reward = np.abs(h1_level_50 - close)
        risk = np.where(h1_level_50 >= close, close - lo5, hi5 - close)
        rr = reward / np.maximum(risk, floor * atr)
        return {'price_square_low_atr': (close - lo5) / atr,
                'price_square_high_atr': (hi5 - close) / atr,
                'rr_to_50pct_target': rr}
