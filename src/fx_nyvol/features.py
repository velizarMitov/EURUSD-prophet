"""The fourteen fixed, scale-free features at the close of the bar labelled 14:00 Berlin
(spec fx-ny-volatility/range-features; design D4).

Every value of day d uses only bars labelled <= 14:00 on d and earlier days. Ranges are bid
max(high) - min(low) over complete windows (fx_session's _window, read-only):

  log_ny_range_lag1/ma5/ma20   y of the previous target-available day; mean of the previous 5 / 20
  log_asia_range               log(range(00:00-08:00) / close14_bid)
  log_london_range             log(range(09:00-13:00) / close14_bid)
  log_data_hour_range          log(range(14:00) / close14_bid), the range floored at one point (counted)
  abs_data_hour_return         |close14 - open14| / close14 (bid)
  london_range_ratio           London range / its mean over the previous 20 days that had one
  data_hour_range_ratio        the same for the 14:00 bar
  cross_pre_range_ratio        mean over EURUSD, GBPUSD, AUDUSD of range(00:00-14:00) / its previous-20-day mean
  day_of_week, is_nfp_day, is_fomc_day, dst_mismatch   fx_session's calendar_block (read-only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_session.features import _ratio_to_mean, _window, calendar_block
from . import session_view


class FeatureColumnsError(ValueError):
    pass


def har_lags(y_avail: pd.Series, dates, windows=(1, 5, 20)) -> pd.DataFrame:
    """For each date: the mean of the targets of the previous 1 / 5 / 20 target-available days,
    strictly before the date (never the date itself)."""
    s = y_avail.dropna().sort_index()
    cols = {}
    for w in windows:
        r = s.rolling(w, min_periods=w).mean()
        left = pd.DataFrame({'date': pd.DatetimeIndex(dates)})
        right = pd.DataFrame({'avail_date': s.index, 'v': r.to_numpy()})
        m = pd.merge_asof(left, right, left_on='date', right_on='avail_date', direction='backward',
                          allow_exact_matches=False)
        cols[w] = m['v'].to_numpy()
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates))


def build_features(frames: dict, points: dict, cfg: dict, dates, y_avail: dict, gr: dict | None = None) -> dict:
    from .target import grids
    s = cfg['session']
    dates = pd.DatetimeIndex(dates)
    gr = gr if gr is not None else {p: grids(frames[p], dates) for p in cfg['factor_pairs']}
    w = int(s['range_window_days'])
    cross = []
    for p in cfg['factor_pairs']:
        pre = _window(gr[p], *s['pre_hours'])
        cross.append(_ratio_to_mean(pre['high'] - pre['low'], dates, w))
    cross = np.column_stack(cross)
    cross_ratio = np.where(np.isnan(cross).any(axis=1), np.nan, cross.mean(axis=1))
    cal = calendar_block(dates, session_view(cfg))
    out = {}
    for p in cfg['pairs']:
        g = gr[p]
        asia, london = _window(g, *s['asia_hours']), _window(g, *s['london_hours'])
        data = _window(g, s['data_hour'], s['data_hour'])
        c14 = data['close']
        raw_dr = data['high'] - data['low']
        dr = np.where(np.isnan(raw_dr), np.nan, np.maximum(raw_dr, points[p]))
        lr = london['high'] - london['low']
        har = har_lags(y_avail[p], dates, s['har_windows'])
        with np.errstate(all='ignore'):
            f = pd.DataFrame({
                'log_ny_range_lag1': har[s['har_windows'][0]].to_numpy(),
                'log_ny_range_ma5': har[s['har_windows'][1]].to_numpy(),
                'log_ny_range_ma20': har[s['har_windows'][2]].to_numpy(),
                'log_asia_range': np.log((asia['high'] - asia['low']) / c14),
                'log_london_range': np.log(lr / c14),
                'log_data_hour_range': np.log(dr / c14),
                'abs_data_hour_return': np.abs(c14 - data['open']) / c14,
                'london_range_ratio': _ratio_to_mean(lr, dates, w),
                'data_hour_range_ratio': _ratio_to_mean(dr, dates, w),
                'cross_pre_range_ratio': cross_ratio,
            }, index=dates)
        f = pd.concat([f, cal], axis=1)
        f = assemble(f, cfg)
        f.attrs['data_hour_floored'] = int(np.nansum(raw_dr < points[p]))
        out[p] = f
    return out


def assemble(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = list(cfg['features']['columns'])
    extra = [c for c in df.columns if c not in cols]
    missing = [c for c in cols if c not in df.columns]
    if extra or missing:
        raise FeatureColumnsError(f'feature columns differ from the config: extra {extra}, missing {missing}')
    return df[cols]
