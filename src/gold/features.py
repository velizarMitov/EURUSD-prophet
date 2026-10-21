"""Scale-free, price-only gold features (design D4; spec "Gold features are
scale-free and price-only").

Why scale-free: gold went from $1,312 to $4,131 over the history. The close at
the 70 % mark is $1,644 and the training maximum $2,062, so any raw level
(open/high/low/close, a moving-average level, an unnormalised ATR) would put the
whole test block outside every value the scaler saw. EUR/USD's production set
carries such levels and gets away with it because the pair is range-bound;
gold cannot. Every column below is invariant to multiplying the price by a
constant, and a test holds that.

  H1   src/pooled_h1_model.compute_pooled_features, unchanged (15 columns) --
       the set behind the only direction edge this project has confirmed.
  D1   gold's own set, below.

Every column at bar t reads bars <= t only.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

D1_FEATURE_COLUMNS = [
    'log_return', 'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_4', 'ret_lag_5',
    'range_pct', 'range_lag_1', 'range_lag_2', 'range_lag_3',
    'vol_20', 'ewma_vol_20', 'atr14_pct', 'bb_width',
    'dist_sma_21', 'dist_sma_50', 'dist_sma_200', 'rsi_14',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
]

_LEVEL = re.compile(r'^(open|high|low|close|sma_\d+|ema_\d+|atr_?\d+|bb_(upper|lower|mid))$', re.I)
_FORBIDDEN = re.compile(r'volume|spread', re.I)


class ScaleDependentFeature(ValueError):
    """A raw price level (or volume/spread) was offered as a model input."""


def assert_scale_free(columns) -> list:
    bad = [c for c in columns if _LEVEL.match(str(c)) or _FORBIDDEN.search(str(c))]
    if bad:
        raise ScaleDependentFeature(f'not a gold model input (raw level, volume or spread): {bad}')
    return list(columns)


def h1_feature_columns() -> list:
    from src.pooled_h1_model import FEATURE_COLUMNS
    return list(FEATURE_COLUMNS)


def h1_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.pooled_h1_model import compute_pooled_features
    return compute_pooled_features(df)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def d1_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].astype(float) for k in ('open', 'high', 'low', 'close'))
    out = pd.DataFrame(index=df.index)
    lr = np.log(c / c.shift(1))
    out['log_return'] = lr
    for k in range(1, 6):
        out[f'ret_lag_{k}'] = lr.shift(k)
    rng = (h - l) / o.replace(0, np.nan)
    out['range_pct'] = rng
    for k in range(1, 4):
        out[f'range_lag_{k}'] = rng.shift(k)
    out['vol_20'] = lr.rolling(20).std()
    out['ewma_vol_20'] = lr.ewm(span=20, adjust=False).std()
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out['atr14_pct'] = tr.ewm(com=13, adjust=False).mean() / c
    out['bb_width'] = 4 * c.rolling(20).std() / c.rolling(20).mean()
    for p in (21, 50, 200):
        out[f'dist_sma_{p}'] = c / c.rolling(p).mean() - 1
    out['rsi_14'] = _rsi(c, 14)
    dow, month = df.index.dayofweek.to_numpy(), df.index.month.to_numpy()
    out['dow_sin'], out['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    out['month_sin'], out['month_cos'] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    return out[D1_FEATURE_COLUMNS]


def build(df: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Feature frame for one cadence, guarded against scale-dependent columns."""
    out = d1_features(df) if cadence == 'D1' else h1_features(df)
    assert_scale_free(out.columns)
    return out
