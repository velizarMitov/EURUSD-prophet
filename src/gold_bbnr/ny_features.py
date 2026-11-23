"""Part 2: the fourteen features at the close of the gold bar labelled 14:00 Berlin
(spec gold-bb-nyrange/ny-range; design D9).

Nine gold features (fx_nyvol's definitions, bid prices, gold point 0.01):

  log_ny_range_lag1/ma5/ma20   mean target of the previous 1 / 5 / 20 eligible days (fx_nyvol's har_lags)
  log_asia_range               log(range(00:00-08:00) / close14)
  log_london_range             log(range(09:00-13:00) / close14)
  log_data_hour_range          log(range(14:00) / close14), the range floored at one point
  abs_data_hour_return         |close14 - open14| / close14
  london_range_ratio           London range / its mean over the previous 20 days that had one
  data_hour_range_ratio        the same for the 14:00 bar

The dollar and calendar features:

  dollar_pre_range_ratio       DXY range over the bars labelled 00:00-14:00 / its previous-20-day mean;
                               missing when that DXY day has fewer than 10 of those bars
  day_of_week, is_nfp_day, is_fomc_day, dst_mismatch   fx_session's calendar_block (read-only)

Every value of day d uses bars labelled <= 14:00 on d and earlier days' targets only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_nyvol.features import har_lags
from ..fx_session.features import _ratio_to_mean, _window, calendar_block, hour_grid


class FeatureColumnsError(ValueError):
    pass


def dollar_ratio(dxy_h1: pd.DataFrame, dates, cfg: dict) -> np.ndarray:
    s = cfg['session']
    dates = pd.DatetimeIndex(dates)
    g = hour_grid(dxy_h1, dates)
    a, b = s['pre_hours']
    hi, lo = g['high'][:, a:b + 1], g['low'][:, a:b + 1]
    n = (~np.isnan(hi)).sum(axis=1)
    with np.errstate(all='ignore'):
        rng = np.nanmax(np.where(np.isnan(hi), -np.inf, hi), axis=1) - np.nanmin(np.where(np.isnan(lo), np.inf, lo), axis=1)
    rng = np.where(n >= int(cfg['dxy']['min_pre_bars']), rng, np.nan)
    return _ratio_to_mean(rng, dates, int(s['range_window_days']))


def build_features(g: dict, dates, y_avail: pd.Series, dxy_h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg['session']
    dates = pd.DatetimeIndex(dates)
    w = int(s['range_window_days'])
    asia, london = _window(g, *s['asia_hours']), _window(g, *s['london_hours'])
    data = _window(g, s['data_hour'], s['data_hour'])
    c14 = data['close']
    raw_dr = data['high'] - data['low']
    dr = np.where(np.isnan(raw_dr), np.nan, np.maximum(raw_dr, float(s['point'])))
    lr = london['high'] - london['low']
    har = har_lags(y_avail, dates, s['har_windows'])
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
            'dollar_pre_range_ratio': dollar_ratio(dxy_h1, dates, cfg),
        }, index=dates)
    cal = calendar_block(dates, {'session': {'fomc_dates': cfg['paths_read']['fomc_dates']}})
    f = pd.concat([f, cal], axis=1)
    f.attrs['data_hour_floored'] = int(np.nansum(raw_dr < float(s['point'])))
    return assemble(f, cfg)


def dataset(h1: pd.DataFrame, d1: pd.DataFrame, dxy_h1: pd.DataFrame, cfg: dict, read_target: bool = True) -> pd.DataFrame:
    """One row per Mon-Fri date: the fourteen features, `y` and `reason` ('' = usable by every model).

    With read_target=False (the label report) no price after 14:00 is read: `y` is NaN and the HAR lags
    are computed from a placeholder of zeros on eligible days, which gives the same missing pattern."""
    from .ny_target import calendar_dates, eligibility, grid, target
    dates = calendar_dates(h1)
    g = grid(h1, dates)
    reason = eligibility(g, dates, d1, cfg)
    y = target(g, dates, cfg) if read_target else pd.Series(np.nan, index=dates, name='y')
    y_avail = (y if read_target else pd.Series(0.0, index=dates)).where(reason == '')
    f = build_features(g, dates, y_avail, dxy_h1, cfg)
    out = f.copy()
    out['y'] = y.to_numpy()
    warm = f.isna().any(axis=1).to_numpy()
    out['reason'] = np.where(reason != '', reason, np.where(warm, 'warm-up', ''))
    out.attrs.update(f.attrs)
    return out


def assemble(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = list(cfg['features_ny']['columns'])
    extra = [c for c in df.columns if c not in cols]
    missing = [c for c in cols if c not in df.columns]
    if extra or missing:
        raise FeatureColumnsError(f'feature columns differ from the config: extra {extra}, missing {missing}')
    out = df[cols]
    out.attrs.update(df.attrs)
    return out
