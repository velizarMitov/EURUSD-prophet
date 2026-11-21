"""Pre-session features (spec fx-session/session-features).

For day D every value is known at the close of the entry bar (labelled 14:00 Berlin on D):
it uses only bars labelled <= 14:00 on D and earlier days. Windows on D's Berlin labels:

    Asia 00:00-08:00   London 09:00-13:00   data hour 14:00 (the entry bar)

A window move is its last close minus its first open. Moves are divided by ATR(14) of the
last completed valid D1 bar before D; ranges by the mean of the same range over the previous
20 days that had one. A D1 bar built from fewer than `min_h1_per_d1` H1 bars counts as missing
(GBPUSD 2014-03 holds one bar per day). Every window needs all of its bars.

Dollar factor: the mean over EURUSD, GBPUSD and AUDUSD of each pair's normalised move from D's
00:00 open to the entry close (and the same for the data-hour move); the residual is the pair's
own move minus that mean. Calendar: day of week, NFP (first Friday, a declared approximation),
scheduled FOMC statement days from the project's date file, and the US/EU DST-mismatch flag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..fx_fib.mtf_aligner import atr as wilder_atr
from ..fx_fib.mtf_aligner import build_d1
from . import repo_path

NY, BERLIN = ZoneInfo('America/New_York'), ZoneInfo('Europe/Berlin')


class FeatureColumnError(ValueError):
    pass


# -- per-pair grids ------------------------------------------------------------------------------

def hour_grid(h1: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """{col: array (n_dates, 24)} of each Mon-Fri date's bar values by label hour (NaN if missing)."""
    idx = pd.DatetimeIndex(h1.index)
    day, hour = idx.normalize(), idx.hour
    pos = pd.Index(dates).get_indexer(day)
    keep = pos >= 0
    out = {}
    for col in ('open', 'high', 'low', 'close'):
        g = np.full((len(dates), 24), np.nan)
        g[pos[keep], hour[keep]] = h1[col].to_numpy()[keep]
        out[col] = g
    return out


def _window(g: dict, a: int, b: int) -> dict:
    sl = slice(a, b + 1)
    complete = ~np.isnan(g['close'][:, sl]).any(axis=1)
    with np.errstate(all='ignore'):
        res = {'open': g['open'][:, a], 'close': g['close'][:, b],
               'high': np.nanmax(np.where(np.isnan(g['high'][:, sl]), -np.inf, g['high'][:, sl]), axis=1),
               'low': np.nanmin(np.where(np.isnan(g['low'][:, sl]), np.inf, g['low'][:, sl]), axis=1)}
    return {k: np.where(complete, v, np.nan) for k, v in res.items()}


def prev_d1(h1: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """For each date D: the newest valid D1 bar dated before D, with its ATR(14)."""
    d1 = build_d1(h1)
    valid = d1[d1['n_h1'] >= int(cfg['data']['min_h1_per_d1'])].copy()
    valid['atr'] = wilder_atr(valid, int(cfg['session']['atr_period']))
    left = pd.DataFrame({'date': pd.DatetimeIndex(dates)})
    right = valid[['open', 'high', 'low', 'close', 'atr']].reset_index().rename(columns={'date': 'd1_date'})
    out = pd.merge_asof(left, right, left_on='date', right_on='d1_date', direction='backward',
                        allow_exact_matches=False)
    return out.set_index('date')


def _ratio_to_mean(x: np.ndarray, dates, window: int) -> np.ndarray:
    s = pd.Series(x, index=dates)
    mean = s.dropna().rolling(window, min_periods=window).mean().shift(1)
    return (s / mean.reindex(dates)).to_numpy()


def pair_block(h1: pd.DataFrame, dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    s = cfg['session']
    g = hour_grid(h1, dates)
    asia, london = _window(g, *s['asia_hours']), _window(g, *s['london_hours'])
    data = _window(g, s['data_hour'], s['data_hour'])
    day = _window(g, 0, s['entry_hour'])                      # D's bars so far, 00:00 .. entry
    p = prev_d1(h1, dates, cfg)
    a = p['atr'].to_numpy()
    close14 = data['close']
    lr = london['high'] - london['low']
    dr = data['high'] - data['low']
    pr = (p['high'] - p['low']).to_numpy()
    w = int(s['range_window_days'])
    with np.errstate(all='ignore'):
        f = pd.DataFrame({
            'asia_move': (asia['close'] - asia['open']) / a,
            'london_move': (london['close'] - london['open']) / a,
            'data_hour_move': (data['close'] - data['open']) / a,
            'london_range_ratio': _ratio_to_mean(lr, dates, w),
            'data_hour_range_ratio': _ratio_to_mean(dr, dates, w),
            'london_position': np.where(lr > 0, (close14 - london['low']) / lr, 0.5),
            'prev_day_move': (p['close'] - p['open']).to_numpy() / a,
            'prev_day_range': pr / a,
            'prev_close_location': np.where(pr > 0, (p['close'] - p['low']).to_numpy() / pr, 0.5),
            'move_since_prev_close': (close14 - p['close'].to_numpy()) / a,
            'broke_prev_high': (day['high'] > p['high'].to_numpy()).astype(float),
            'broke_prev_low': (day['low'] < p['low'].to_numpy()).astype(float),
            'atr_d1_frac': a / close14,
            '_pre_move': (close14 - day['open']) / a,
        }, index=dates)
    # carry NaN through the flags and the position when their inputs are missing
    for col, inputs in (('broke_prev_high', (day['high'], a)), ('broke_prev_low', (day['low'], a)),
                        ('london_position', (lr, close14))):
        f.loc[np.isnan(inputs[0]) | np.isnan(inputs[1]), col] = np.nan
    f.attrs['reason'] = np.select(
        [np.isnan(asia['close']) | np.isnan(london['close']) | np.isnan(data['close']) | np.isnan(day['close']),
         np.isnan(a),
         f[['london_range_ratio', 'data_hour_range_ratio']].isna().any(axis=1).to_numpy()],
        ['missing pre-session bar', 'no previous day', 'warm-up'], '')
    return f


# -- calendar ------------------------------------------------------------------------------------

def dst_mismatch(dates) -> np.ndarray:
    """1 when New York and Berlin are not 6 hours apart at noon UTC on D."""
    out = []
    for d in pd.DatetimeIndex(dates):
        t = datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)
        gap = (t.astimezone(BERLIN).utcoffset() - t.astimezone(NY).utcoffset()).total_seconds() / 3600
        out.append(float(gap != 6))
    return np.array(out)


def fomc_days(cfg: dict) -> set:
    d = pd.read_csv(repo_path(cfg['session']['fomc_dates']), parse_dates=['date'])
    return set(d['date'].dt.date)


def calendar_block(dates, cfg: dict) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    fomc = fomc_days(cfg)
    dow = dates.dayofweek.to_numpy()
    return pd.DataFrame({
        'day_of_week': dow.astype(float),
        'is_nfp_day': ((dow == 4) & (dates.day.to_numpy() <= 7)).astype(float),
        'is_fomc_day': np.array([d.date() in fomc for d in dates], dtype=float),
        'dst_mismatch': dst_mismatch(dates),
    }, index=dates)


# -- assembly ------------------------------------------------------------------------------------

def weekdays(frames: dict) -> pd.DatetimeIndex:
    days = None
    for df in frames.values():
        d = pd.DatetimeIndex(df.index).normalize().unique()
        days = d if days is None else days.union(d)
    days = days[days.dayofweek < 5]
    return pd.DatetimeIndex(days.sort_values(), name='date')


def assemble(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = cfg['features']['columns']
    extra = [c for c in df.columns if c not in cols]
    missing = [c for c in cols if c not in df.columns]
    if extra or missing:
        raise FeatureColumnError(f'feature columns differ from the config list: extra {extra}, missing {missing}')
    return df[cols]


def build_features(frames: dict, cfg: dict, dates=None) -> dict:
    """{pair: DataFrame indexed by date with the config's columns plus `reason`}.
    `reason` is '' for a complete row, else why the row is excluded."""
    dates = weekdays(frames) if dates is None else pd.DatetimeIndex(dates, name='date')
    blocks = {p: pair_block(frames[p], dates, cfg) for p in cfg['factor_pairs']}
    pre = np.column_stack([blocks[p]['_pre_move'].to_numpy() for p in cfg['factor_pairs']])
    dh = np.column_stack([blocks[p]['data_hour_move'].to_numpy() for p in cfg['factor_pairs']])
    usd = pre.mean(axis=1)
    usd_dh = dh.mean(axis=1)
    factor_ok = ~(np.isnan(pre).any(axis=1) | np.isnan(dh).any(axis=1))
    cal = calendar_block(dates, cfg)
    out = {}
    for pair in cfg['pairs']:
        b = blocks[pair]
        f = b.drop(columns=['_pre_move']).copy()
        f['usd_factor'] = usd
        f['usd_factor_data_hour'] = usd_dh
        f['pair_residual'] = b['_pre_move'].to_numpy() - usd
        f = pd.concat([f, cal], axis=1)
        f = assemble(f, cfg)
        reason = pd.Series(b.attrs['reason'], index=dates)
        reason[(reason == '') & ~factor_ok] = 'missing factor-pair input'
        reason[(reason == '') & f.isna().any(axis=1)] = 'incomplete input'
        f['reason'] = reason
        out[pair] = f
    return out
