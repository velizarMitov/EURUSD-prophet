"""The twelve fx-rates features (spec fx-rates/rates-features; design D5).

Yields are the lagged, limited-fill values from align.py (as of the previous FX trading day);
FX uses D1 mid closes up to and including day t (the decision is at t's close). Windows count
FX trading days (D1 rows). vol = std of daily log returns of the mid close.

    spread_2y_level        foreign 2Y - US 2Y (DE2Y for EURUSD, UK2Y for GBPUSD), pct points
    spread_2y_chg_{5,20,60}d   level(t) - level(t-h)
    spread_2y_zscore_60d   (level - mean60) / std60
    us_2s10s_slope         US10Y - US2Y
    fx_ret_{5,20}d         log(c_t / c_{t-h}) / (vol20 * sqrt(h))
    fx_vs_spread_div_20d   z(20d FX log return) - z(20d spread change); each z against the mean
                           and std of the PREVIOUS 252 days of the same quantity
    fx_ma20_vs_ma50        log(MA20 / MA50) / vol20
    fx_vol_20d_ratio       vol20 / vol60
    yield_stale_flag       1 when any series was forward-filled
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .align import align_yields


class FeatureColumnError(ValueError):
    pass


def _z_prev(x: pd.Series, n: int) -> pd.Series:
    prev = x.shift(1)
    return (x - prev.rolling(n, min_periods=n).mean()) / prev.rolling(n, min_periods=n).std()


def pair_features(d1: pd.DataFrame, aligned: pd.DataFrame, foreign: str, cfg: dict) -> pd.DataFrame:
    w = cfg['windows']
    spread = aligned[foreign] - aligned['us2y']
    c = d1['mid_close']
    lc = np.log(c)
    r = lc.diff()
    vol20 = r.rolling(w['vol_short'], min_periods=w['vol_short']).std()
    vol60 = r.rolling(w['vol_long'], min_periods=w['vol_long']).std()
    f = pd.DataFrame(index=d1.index)
    f['spread_2y_level'] = spread
    for h in w['spread_changes']:
        f[f'spread_2y_chg_{h}d'] = spread - spread.shift(h)
    zn = w['zscore']
    f[f'spread_2y_zscore_{zn}d'] = (spread - spread.rolling(zn, min_periods=zn).mean()) / \
        spread.rolling(zn, min_periods=zn).std()
    f['us_2s10s_slope'] = aligned['us10y'] - aligned['us2y']
    for h in w['fx_returns']:
        f[f'fx_ret_{h}d'] = (lc - lc.shift(h)) / (vol20 * np.sqrt(h))
    dh, dn = w['divergence_horizon'], w['divergence_norm']
    f[f'fx_vs_spread_div_{dh}d'] = _z_prev(lc - lc.shift(dh), dn) - _z_prev(spread - spread.shift(dh), dn)
    f['fx_ma20_vs_ma50'] = np.log(c.rolling(w['ma_fast'], min_periods=w['ma_fast']).mean()
                                  / c.rolling(w['ma_slow'], min_periods=w['ma_slow']).mean()) / vol20
    f['fx_vol_20d_ratio'] = vol20 / vol60
    f['yield_stale_flag'] = aligned['yield_stale_flag']
    return f.replace([np.inf, -np.inf], np.nan)


def assemble(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = cfg['features']['columns']
    extra = [c for c in df.columns if c not in cols]
    missing = [c for c in cols if c not in df.columns]
    if extra or missing:
        raise FeatureColumnError(f'feature columns differ from the config list: extra {extra}, missing {missing}')
    return df[cols]


def build_features(d1: dict, yields: dict, cfg: dict) -> dict:
    """{pair: rows by date with the config's columns, `reason` ('' = complete) and the aligned
    yield ages}."""
    out = {}
    for pair in cfg['pairs']:
        a = align_yields(d1[pair].index, yields, cfg)
        f = assemble(pair_features(d1[pair], a, cfg['foreign_2y'][pair], cfg), cfg)
        reason = a['yield_reason'].copy()
        reason[(reason == '') & f.isna().any(axis=1)] = 'warm-up'
        f = f.copy()
        f['reason'] = reason
        out[pair] = f
    return out
