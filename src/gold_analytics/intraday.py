"""Time-of-day statistics (spec gold-analytics/dashboard "Time-of-day statistics").

Slots are on the owner's clock (Sofia). For a chosen interval and period:
mean return, mean absolute move, mean range, acceleration (the change in mean
absolute move from the previous slot), momentum (the correlation of the first
and last hour of the owner's window across days), where the day's high and low
were set, and the median and 90th-percentile spread from the tick cache.

The finest bars are kept only as far back as MT5 holds them, so every answer
states the days it actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clock import display_day, display_minute_of_day


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


def resample(bars: pd.DataFrame, minutes: int, offset_hours: int = 0) -> pd.DataFrame:
    """OHLC bars on `minutes` bins of the label clock. `offset_hours` shifts the
    bin edges (4h bars use +1 so they start at Sofia midnight)."""
    if bars.empty:
        return bars.copy()
    shift = pd.Timedelta(hours=offset_hours)
    key = (bars['time'] + shift).dt.floor(f'{minutes}min') - shift
    g = bars.groupby(key)
    out = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                        'close': g['close'].last()})
    out.index.name = 'time'
    return out.reset_index()


def last_days(bars: pd.DataFrame, n: int) -> pd.DataFrame:
    """Bars of the last `n` label days that hold bars."""
    if bars.empty:
        return bars
    day = bars['time'].dt.normalize()
    keep = np.sort(day.unique())[-n:]
    return bars[day.isin(keep)].reset_index(drop=True)


def n_days(bars: pd.DataFrame) -> int:
    return int(bars['time'].dt.normalize().nunique()) if len(bars) else 0


def slot_label(minute: int) -> str:
    return f'{minute // 60:02d}:{minute % 60:02d}'


def weighted_median(values, weights) -> float | None:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = ~np.isnan(v) & (w > 0)
    if not ok.any():
        return None
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2)])


def spread_by_slot(spread: pd.DataFrame, minutes: int, days: int | None = None) -> pd.DataFrame:
    """The 5-minute spread rows aggregated to `minutes` Sofia slots (never finer
    than 5) with a tick-count weighted median of the slot medians and of the
    slot 90th percentiles."""
    cols = ['minute', 'spread_median', 'spread_p90', 'spread_days']
    if spread.empty:
        return pd.DataFrame(columns=cols)
    s = spread
    if days is not None:
        keep = sorted(s['day'].unique())[-days:]
        s = s[s['day'].isin(keep)]
    step = max(minutes, 5)
    minute = display_minute_of_day(s['slot']) // step * step
    s = s.assign(minute=minute)
    rows = []
    for m, g in s.groupby('minute'):
        rows.append({'minute': int(m), 'spread_median': weighted_median(g['median'], g['ticks']),
                     'spread_p90': weighted_median(g['p90'], g['ticks']), 'spread_days': int(g['day'].nunique())})
    return pd.DataFrame(rows, columns=cols)


def momentum(bars: pd.DataFrame, window: dict) -> dict:
    """Correlation, across days, of the return in the first hour of the owner's
    window with the return in its last hour."""
    if bars.empty:
        return {'n': 0, 'corr': None}
    a, b = _minutes(window['start']), _minutes(window['end'])
    m = display_minute_of_day(bars['time'])
    day = display_day(bars['time'])

    def hour_ret(lo, hi):
        mask = (m >= lo) & (m < hi)
        g = bars[mask].groupby(np.asarray(day)[mask])
        return g['close'].last() / g['open'].first() - 1

    first, last = hour_ret(a, a + 60), hour_ret(b - 60, b)
    both = pd.concat([first.rename('first'), last.rename('last')], axis=1).dropna()
    if len(both) < 3:
        return {'n': int(len(both)), 'corr': None}
    return {'n': int(len(both)), 'corr': float(both['first'].corr(both['last']))}


def extreme_times(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Per Sofia slot: the share of days whose high, and whose low, was set there."""
    cols = ['minute', 'high_share', 'low_share']
    if bars.empty:
        return pd.DataFrame(columns=cols)
    m = display_minute_of_day(bars['time']) // minutes * minutes
    t = bars.assign(minute=m, day=np.asarray(display_day(bars['time'])))
    hi = t.loc[t.groupby('day')['high'].idxmax(), 'minute'].value_counts(normalize=True)
    lo = t.loc[t.groupby('day')['low'].idxmin(), 'minute'].value_counts(normalize=True)
    out = pd.concat([hi.rename('high_share'), lo.rename('low_share')], axis=1).fillna(0.0)
    out.index.name = 'minute'
    return out.reset_index()[cols].sort_values('minute')


def time_of_day(bars: pd.DataFrame, minutes: int, window: dict, spread: pd.DataFrame | None = None,
                days_requested: int | None = None) -> dict:
    """The whole section for bars already at `minutes` resolution."""
    days_used = n_days(bars)
    out = {'interval': minutes, 'days_requested': days_requested, 'days_used': days_used,
           'limited': days_requested is not None and days_used < days_requested,
           'first_bar': None if bars.empty else bars['time'].min(), 'slots': [],
           'momentum': momentum(bars, window)}
    if bars.empty:
        return out
    m = display_minute_of_day(bars['time']) // minutes * minutes
    t = bars.assign(minute=m, ret=(bars['close'] / bars['open'] - 1) * 100,
                    rng=(bars['high'] - bars['low']) / bars['open'] * 100)
    t['abs'] = t['ret'].abs()
    g = t.groupby('minute')
    tab = pd.DataFrame({'ret_mean': g['ret'].mean(), 'abs_mean': g['abs'].mean(), 'range_mean': g['rng'].mean(),
                        'price': g['close'].mean(), 'n': g.size()})
    tab['accel'] = tab['abs_mean'].diff()
    tab = tab.reset_index()
    ext = extreme_times(bars, minutes)
    tab = tab.merge(ext, on='minute', how='left')
    if spread is not None and len(spread):
        sp = spread_by_slot(spread, minutes, days_requested)
        if minutes < 5:  # spread is only known per 5 minutes: each 1-minute slot shows its 5-minute value
            tab['m5'] = tab['minute'] // 5 * 5
            tab = tab.merge(sp.rename(columns={'minute': 'm5'}), on='m5', how='left').drop(columns='m5')
        else:
            tab = tab.merge(sp, on='minute', how='left')
        tab['spread_median_pct'] = tab['spread_median'] / tab['price'] * 100
        tab['spread_p90_pct'] = tab['spread_p90'] / tab['price'] * 100
        out['spread_days'] = int(spread['day'].nunique()) if days_requested is None \
            else min(days_requested, int(spread['day'].nunique()))
    a, b = _minutes(window['start']), _minutes(window['end'])
    tab['in_window'] = (tab['minute'] >= a) & (tab['minute'] < b)
    tab['slot'] = tab['minute'].map(slot_label)
    out['slots'] = tab.replace({np.nan: None}).to_dict('records')
    return out
