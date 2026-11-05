"""Sessions in their own time zones (design D4, spec gold-analytics/market-data
"Sessions are defined in their own time zones").

A session on day d runs from `start` to `end` wall time in its own zone on that
date, converted to labels, so Europe's and the US's daylight-saving changes each
land on their own dates. Bars whose labels fall in [start, end) belong to it.
The London-New York overlap is the intersection of the two on each day.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .clock import instant_to_label, ns


def _hhmm(s: str) -> time:
    h, m = s.split(':')
    return time(int(h), int(m))


def trading_days(start: date, end: date) -> list[date]:
    """Monday-Friday dates in [start, end]."""
    return [d.date() for d in pd.date_range(start, end, freq='D') if d.weekday() < 5]


def session_interval(day: date, session: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start label, end label) of `session` on `day`."""
    z = ZoneInfo(session['zone'])
    a = datetime.combine(day, _hhmm(session['start']), z)
    b = datetime.combine(day, _hhmm(session['end']), z)
    lab = instant_to_label([pd.Timestamp(a), pd.Timestamp(b)])
    return lab[0], lab[1]


def session_intervals(days, session: dict) -> pd.DataFrame:
    rows = [(d, *session_interval(d, session)) for d in days]
    return pd.DataFrame(rows, columns=['day', 'start', 'end'])


def overlap_intervals(days, a: dict, b: dict) -> pd.DataFrame:
    """The intersection of sessions `a` and `b` on each day; days where they do
    not meet are left out."""
    ia, ib = session_intervals(days, a), session_intervals(days, b)
    out = pd.DataFrame({'day': ia['day'],
                        'start': pd.concat([ia['start'], ib['start']], axis=1).max(axis=1),
                        'end': pd.concat([ia['end'], ib['end']], axis=1).min(axis=1)})
    return out[out['start'] < out['end']].reset_index(drop=True)


def all_sessions(cfg: dict) -> dict:
    """key -> session dict, including the configured overlap (as `of`)."""
    out = {s['key']: s for s in cfg['sessions']}
    if cfg.get('overlap'):
        out[cfg['overlap']['key']] = cfg['overlap']
    return out


def intervals_for(key: str, days, cfg: dict) -> pd.DataFrame:
    sessions = all_sessions(cfg)
    if key not in sessions:
        raise ValueError(f'session must be one of {list(sessions)}')
    s = sessions[key]
    if 'of' in s:
        a, b = (sessions[k] for k in s['of'])
        return overlap_intervals(days, a, b)
    return session_intervals(days, s)


# -- session statistics (task 3.1) -----------------------------------------

WILSON_Z = 1.959963984540054


def wilson(hits: int, n: int, z: float = WILSON_Z):
    """The Wilson score interval for a proportion; (None, None) when n = 0."""
    if n <= 0:
        return None, None
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def session_table(bars: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """One row per interval that holds bars: open (first bar's open), close (last
    bar's close), high, low, range, % return and realised volatility (sqrt of the
    summed squared in-session log returns, the first measured from the open)."""
    cols = ['day', 'start', 'end', 'open', 'close', 'high', 'low', 'range_usd', 'range_pct', 'ret_pct',
            'rv_pct', 'n_bars']
    if bars.empty or intervals.empty:
        return pd.DataFrame(columns=cols)
    t = ns(bars['time'])
    starts = ns(intervals['start'])
    pos = np.searchsorted(starts, t, side='right') - 1
    ok = pos >= 0
    ok[ok] = t[ok] < ns(intervals['end'])[pos[ok]]
    b = bars.loc[ok].assign(iv=pos[ok])
    if b.empty:
        return pd.DataFrame(columns=cols)
    prev = b.groupby('iv')['close'].shift(1).fillna(b['open'])
    b = b.assign(r2=np.log(b['close'] / prev) ** 2)
    g = b.groupby('iv')
    out = pd.DataFrame({'open': g['open'].first(), 'close': g['close'].last(), 'high': g['high'].max(),
                        'low': g['low'].min(), 'rv_pct': np.sqrt(g['r2'].sum()) * 100, 'n_bars': g.size()})
    out = intervals.iloc[out.index].reset_index(drop=True).join(out.reset_index(drop=True))
    out['range_usd'] = out['high'] - out['low']
    out['range_pct'] = out['range_usd'] / out['open'] * 100
    out['ret_pct'] = (out['close'] / out['open'] - 1) * 100
    return out[cols]


def _share(mask: pd.Series) -> dict:
    n, k = int(mask.size), int(mask.sum())
    lo, hi = wilson(k, n)
    return {'n': n, 'hits': k, 'share': (k / n) if n else None, 'ci_low': lo, 'ci_high': hi}


def session_summary(table: pd.DataFrame) -> dict:
    """Across a period: means, directional bias (share closing above the open, with
    its Wilson 95 % interval), breakout frequency (share trading beyond the
    previous same session's high or low) and trend persistence (share whose
    return has the previous same session's sign). Descriptive, never an edge."""
    n = len(table)
    if n == 0:
        return {'n': 0}
    t = table.reset_index(drop=True)
    has_prev = t.index > 0
    breakout = ((t['high'] > t['high'].shift(1)) | (t['low'] < t['low'].shift(1)))[has_prev]
    sign = np.sign(t['ret_pct'])
    pair = (sign != 0) & (sign.shift(1) != 0) & has_prev
    return {
        'n': n,
        'mean_range_usd': float(t['range_usd'].mean()), 'mean_range_pct': float(t['range_pct'].mean()),
        'mean_ret_pct': float(t['ret_pct'].mean()), 'median_ret_pct': float(t['ret_pct'].median()),
        'mean_rv_pct': float(t['rv_pct'].mean()),
        'bias_up': _share(t['ret_pct'] > 0),
        'breakout': _share(breakout),
        'persistence': _share((sign == sign.shift(1))[pair]),
    }


def overnight_day_intervals(days, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day = the configured day_start session's open to the day_end session's
    close; overnight = the previous trading day's day_end close to today's
    day_start open (Monday's overnight spans the weekend)."""
    od = cfg['overnight_day']
    sessions = all_sessions(cfg)
    a = session_intervals(days, sessions[od['day_start']])
    b = session_intervals(days, sessions[od['day_end']])
    day = pd.DataFrame({'day': a['day'], 'start': a['start'], 'end': b['end']})
    night = pd.DataFrame({'day': a['day'], 'start': b['end'].shift(1), 'end': a['start']}).iloc[1:]
    return night.reset_index(drop=True), day


def overnight_against_day(bars: pd.DataFrame, days, cfg: dict) -> dict:
    night_iv, day_iv = overnight_day_intervals(days, cfg)
    out = {}
    for key, iv in (('overnight', night_iv), ('day', day_iv)):
        t = session_table(bars, iv)
        s = {'n': len(t)}
        if len(t):
            s.update({'mean_ret_pct': float(t['ret_pct'].mean()),
                      'sum_log_ret_pct': float(np.log(t['close'] / t['open']).sum() * 100),
                      'bias_up': _share(t['ret_pct'] > 0),
                      'mean_range_pct': float(t['range_pct'].mean())})
        out[key] = s
    return out


def now_session(now_label, cfg: dict) -> dict:
    """The session(s) running at `now_label`, or the next one to start."""
    now = pd.Timestamp(now_label)
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    d = now.date()
    days = [x for x in pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=6)).date if x.weekday() < 5]
    current, upcoming = [], []
    for key in all_sessions(cfg):
        iv = intervals_for(key, days, cfg)
        if ((iv['start'] <= now) & (now < iv['end'])).any():
            current.append(key)
        nxt = iv.loc[iv['start'] > now, 'start']
        if len(nxt):
            upcoming.append((nxt.min(), key))
    nxt = min(upcoming) if upcoming else None
    return {'current': current, 'next': None if nxt is None else {'key': nxt[1], 'start_label': nxt[0]}}
