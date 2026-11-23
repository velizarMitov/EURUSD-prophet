"""The eleven Part 1 features at the close of the event day t (spec gold-bb-nyrange/bb-reversion; design D2, D9).

Nine gold features are fx_bb's definitions (its `pair_frame`, read-only), oriented by the breach side
s = -dir; fx_bb's cross-pair `simultaneous_frac` does not exist for one instrument. Two dollar features
use only daily DX-Y.NYB closes dated STRICTLY BEFORE t, signed by the event direction dir:

  dxy_zscore_20d   dir * (close - SMA20) / SD20 (population SD) of the latest 20 closes
  dxy_ret_5d       dir * log(close / close five closes earlier)

An event with any missing feature is excluded from the model cells as `warm-up`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_bb.features import pair_frame
from .bars import valid


class FeatureColumnsError(ValueError):
    pass


def _view(cfg: dict) -> dict:
    return {'features': cfg['features_bb'], 'bands': cfg['bands'], 'atr_period': cfg['atr_period']}


def dxy_daily(dxy_d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg['features_bb']
    c = dxy_d1['close'].dropna().sort_index()
    w, n = int(f['dxy_window']), int(f['dxy_ret_days'])
    sma, sd = c.rolling(w).mean(), c.rolling(w).std(ddof=int(cfg['bands']['ddof']))
    return pd.DataFrame({'z': (c - sma) / sd, 'ret': np.log(c / c.shift(n))}, index=c.index)


def dxy_before(dates, dxy_d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """For each date: the DXY values of the latest DXY date strictly before it."""
    dd = dxy_daily(dxy_d1, cfg)
    left = pd.DataFrame({'date': pd.DatetimeIndex(dates), 'pos': np.arange(len(dates))})
    right = pd.DataFrame({'dxy_date': pd.DatetimeIndex(dd.index), 'z': dd['z'].to_numpy(), 'ret': dd['ret'].to_numpy()})
    m = pd.merge_asof(left.sort_values('date'), right, left_on='date', right_on='dxy_date',
                      direction='backward', allow_exact_matches=False).sort_values('pos')
    return m.reset_index(drop=True)[['dxy_date', 'z', 'ret']]


def build_features(ev: pd.DataFrame, d1: pd.DataFrame, dxy_d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = pair_frame(valid(d1), _view(cfg))
    cols = cfg['features_bb']['columns']
    blk = b.reindex(pd.DatetimeIndex(ev['date']))
    s = ev['side'].to_numpy(float)
    d = ev['dir'].to_numpy(float)
    dx = dxy_before(ev['date'], dxy_d1, cfg)
    out = pd.DataFrame({
        'bb_zscore_abs': blk['zabs'].to_numpy(),
        'bb_width_rank_60d': blk['width_rank'].to_numpy(),
        'rsi_14_dist': s * (blk['rsi'].to_numpy() - 50.0),
        'bar_range_vs_atr20': blk['range_atr'].to_numpy(),
        'close_pos_in_bar': np.where(s > 0, blk['pos_up'].to_numpy(), 1.0 - blk['pos_up'].to_numpy()),
        'ret_3d_atr': s * blk['ret_atr'].to_numpy(),
        'trend_ma50_alignment': s * blk['ma50_atr'].to_numpy(),
        'atr_ratio_20_60': blk['atr_ratio'].to_numpy(),
        'breach_side': s,
        'dxy_zscore_20d': d * dx['z'].to_numpy(),
        'dxy_ret_5d': d * dx['ret'].to_numpy(),
    }, index=ev.index)
    return assemble(out, cfg).astype(float)


def assemble(feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    want = list(cfg['features_bb']['columns'])
    extra = [c for c in feats.columns if c not in want]
    missing = [c for c in want if c not in feats.columns]
    if extra or missing:
        raise FeatureColumnsError(f'feature frame does not match the config: extra {extra}, missing {missing}')
    return feats[want]


def with_features(ev: pd.DataFrame, d1: pd.DataFrame, dxy_d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Events plus the features; `reason2` adds 'warm-up' (model cells only) to the rule `reason`."""
    feats = build_features(ev, d1, dxy_d1, cfg)
    out = ev.join(feats)
    warm = feats.isna().any(axis=1).to_numpy()
    out['reason2'] = np.where(out['reason'] != '', out['reason'], np.where(warm, 'warm-up', ''))
    return out
