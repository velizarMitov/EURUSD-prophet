"""FX D1 bars and the causal yield alignment (spec fx-rates/rates-data; design D2, D4).

FX D1 bars come READ-ONLY from the fx_session H1 caches: build_d1 by trading date, a D1 bar
needs >= min_h1_per_d1 H1 bars, and its mid close is the last H1 bar's bid close plus half
that bar's spread.

For FX trading date t the yields are as of the PREVIOUS FX trading date (lag 1): per series,
the latest observation dated on or before that lagged date (a backward as-of join; nothing
ever looks forward). An observation older than the lagged date by more than
`fill_limit_business_days` business days excludes the day ("stale yields"); any older-than-
lagged observation sets yield_stale_flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_fib.mtf_aligner import trading_date
from ..fx_session.data_loader import load_h1, load_points
from .rates_loader import SERIES, load_series


def _fx_cfg(cfg: dict) -> dict:
    return {'paths': {'data': cfg['paths']['fx_h1']}}


def d1_from_h1(h1: pd.DataFrame, point: float, cfg: dict) -> pd.DataFrame:
    """D1 OHLC (bid) plus `mid_close` and `n_h1`, indexed by naive trading date."""
    td = trading_date(h1.index)
    mid = h1['close'] + h1['spread'] * point / 2.0
    g = h1.assign(mid=mid.to_numpy()).groupby(td)
    d1 = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                       'close': g['close'].last(), 'mid_close': g['mid'].last(), 'n_h1': g['close'].size()})
    d1.index = pd.DatetimeIndex(d1.index).tz_localize(None)
    d1.index.name = 'date'
    d1 = d1[d1['n_h1'] >= int(cfg['data']['min_h1_per_d1'])]
    return d1[d1.index >= pd.Timestamp(cfg['data']['fx_history_start'])]


def load_fx_d1(cfg: dict) -> dict:
    fcfg = _fx_cfg(cfg)
    points = load_points(fcfg)
    return {p: d1_from_h1(load_h1(p, fcfg), points[p], cfg) for p in cfg['pairs']}


def load_yields(cfg: dict) -> dict:
    return {s: load_series(s, cfg) for s in SERIES}


def align_yields(fx_dates, yields: dict, cfg: dict) -> pd.DataFrame:
    """Per FX date: each series' value as of the previous FX date, its age in business days,
    `yield_stale_flag` and `yield_reason` ('' usable, 'no lagged date', 'stale yields')."""
    dates = pd.DatetimeIndex(fx_dates, name='date')
    lag = int(cfg['alignment']['lag_fx_days'])
    limit = int(cfg['alignment']['fill_limit_business_days'])
    lagged = pd.Series(dates).shift(lag)
    out = pd.DataFrame(index=dates)
    out['lag_date'] = lagged.to_numpy()
    ok = lagged.notna().to_numpy()
    stale = np.zeros(len(dates), dtype=bool)
    too_old = np.zeros(len(dates), dtype=bool)
    left = pd.DataFrame({'lag_date': lagged[ok].to_numpy(), 'row': np.flatnonzero(ok)})
    for name, s in yields.items():
        right = pd.DataFrame({'obs_date': s.index, name: s.to_numpy()}).sort_values('obs_date')
        m = pd.merge_asof(left.sort_values('lag_date'), right, left_on='lag_date', right_on='obs_date',
                          direction='backward').sort_values('row')
        vals = np.full(len(dates), np.nan)
        vals[m['row'].to_numpy()] = m[name].to_numpy()
        out[name] = vals
        age = np.full(len(dates), np.nan)
        have = m['obs_date'].notna().to_numpy()
        rows = m['row'].to_numpy()[have]
        age[rows] = np.busday_count(m['obs_date'].to_numpy()[have].astype('datetime64[D]'),
                                    m['lag_date'].to_numpy()[have].astype('datetime64[D]'))
        out[f'{name}_age'] = age
        stale |= np.nan_to_num(age, nan=0) > 0
        too_old |= np.isnan(age) & ok | (np.nan_to_num(age, nan=0) > limit)
    out['yield_stale_flag'] = stale.astype(float)
    out['yield_reason'] = np.where(~ok, 'no lagged date', np.where(too_old, 'stale yields', ''))
    return out
