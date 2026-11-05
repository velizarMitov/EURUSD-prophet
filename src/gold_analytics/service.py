"""The section payloads behind GET /api/gold-analytics/{section} (design D8, spec
gold-analytics/dashboard).

Every payload carries `stale`, `as_of` (Sofia time), the descriptive label and
the BG/EN dictionary, and only label timestamps are ever turned into display
times (Sofia). Parameters are checked against the config; any other value is a
ValueError naming the allowed values, which api.py answers with 400.

Caching: the overview and the chart for 5 s; every other section until one of
its cache files changes on disk (at most 10 minutes, so live top-ups show).
Bars are held in memory for 30 s between sections. The page runs no model and
writes nothing but the analytics' own cache (the last live tick).
"""

from __future__ import annotations

import inspect
import math
import os
import threading
import time as _time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import benchmarks, history, intraday, load_config, repo_path, volatility
from . import sessions as S
from .clock import instant_to_label, sofia_iso, to_display
from .data import Frame, Source
from .events import calendar, fred_client, news_windows
from .refresh import read_spread
from .texts import LABEL, TEXTS

SECTIONS = ('overview', 'chart', 'sessions', 'intraday', 'events', 'volatility', 'history', 'benchmarks')
SHORT_TTL = {'overview': 5.0, 'chart': 5.0}
LONG_TTL = 600.0
FRAME_TTL = 30.0
MARKET_OPEN_TICK_AGE = pd.Timedelta(minutes=10)


# -- parameters -------------------------------------------------------------

def _choice(name: str, value, allowed, cast=str):
    allowed = list(allowed)
    try:
        v = cast(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be one of {allowed}') from None
    if v not in allowed:
        raise ValueError(f'{name} must be one of {allowed}')
    return v


def _date(name: str, value) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f'{name} must be a date YYYY-MM-DD') from None


def _sofia_moment(name: str, value) -> pd.Timestamp:
    """'YYYY-MM-DDTHH:MM' Sofia wall time -> label."""
    try:
        t = pd.Timestamp(str(value))
    except ValueError:
        raise ValueError(f'{name} must be a Sofia time YYYY-MM-DDTHH:MM') from None
    if t.tzinfo is not None:
        raise ValueError(f'{name} must be a Sofia time YYYY-MM-DDTHH:MM, without a zone')
    return instant_to_label([t.tz_localize('Europe/Sofia', ambiguous=True, nonexistent='shift_forward')])[0]


# -- JSON -------------------------------------------------------------------

def jsonable(o):
    """Timestamps (always labels) become Sofia ISO strings; numpy and NaN become
    plain JSON values."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return None if pd.isna(o) else sofia_iso(o)
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NaT:
        return None
    return o


def label_now(now: datetime | None = None) -> pd.Timestamp:
    """Now on the label clock: Berlin wall time stored as UTC."""
    t = (now or datetime.now(ZoneInfo('Europe/Berlin'))).astimezone(ZoneInfo('Europe/Berlin'))
    return pd.Timestamp(t.replace(tzinfo=None), tz='UTC')


# -- the service --------------------------------------------------------------

class Service:
    def __init__(self, cfg: dict | None = None, source: Source | None = None, clock=None, fred=None):
        self.cfg = cfg or load_config()
        self.source = source or Source(self.cfg)
        self.clock = clock or (lambda: datetime.now(ZoneInfo('Europe/Berlin')))
        self.fred = fred or fred_client
        self._lock = threading.Lock()
        self._frames: dict = {}
        self._results: dict = {}

    # -- inputs -------------------------------------------------------------
    def now(self) -> pd.Timestamp:
        return label_now(self.clock())

    def frame(self, tf: str) -> Frame:
        hit = self._frames.get(tf)
        if hit and _time.monotonic() - hit[0] < FRAME_TTL:
            return hit[1]
        f = self.source.bars(tf)
        self._frames[tf] = (_time.monotonic(), f)
        return f

    def spread(self) -> pd.DataFrame:
        return read_spread(self.cfg)

    def _signature(self) -> tuple:
        sig = []
        for key in ('bars_m1', 'bars_m15', 'spread_5m', 'events', 'h1', 'd1'):
            try:
                sig.append(os.stat(repo_path(self.cfg['paths'][key])).st_mtime_ns)
            except OSError:
                sig.append(None)
        return tuple(sig)

    def options(self) -> dict:
        c = self.cfg
        return {'timeframes': c['timeframes'], 'intervals': c['intervals'], 'periods': c['periods'],
                'events': list(c['events']), 'event_months': c['event_months'],
                'sessions': list(S.all_sessions(c)), 'owner_window': c['owner_window']}

    # -- entry --------------------------------------------------------------
    def build(self, section: str, **params) -> dict:
        section = _choice('section', section, SECTIONS)
        params = {k: v for k, v in params.items() if v is not None and v != ''}
        accepted = [p for p in inspect.signature(getattr(self, f'_{section}')).parameters]
        unknown = sorted(set(params) - set(accepted))
        if unknown:
            raise ValueError(f'{section} takes only {accepted}, not {unknown}')
        key = (section, tuple(sorted((k, str(v)) for k, v in params.items())))
        sig = None if section in SHORT_TTL else self._signature()
        ttl = SHORT_TTL.get(section, LONG_TTL)
        with self._lock:
            hit = self._results.get(key)
            if hit and _time.monotonic() - hit[0] < ttl and hit[1] == sig:
                return hit[2]
            data, stale, as_of = getattr(self, f'_{section}')(**params)
            payload = jsonable({'section': section, 'stale': bool(stale), 'as_of': as_of, 'params': params,
                                'label': LABEL, 'texts': TEXTS, 'options': self.options(),
                                'generated_at': self.now(), 'data': data})
            self._results[key] = (_time.monotonic(), sig, payload)
            return payload

    # -- helpers ------------------------------------------------------------
    def _finest(self, days: int, since=None) -> tuple[pd.DataFrame, str, bool]:
        """The finest bars that cover `days` label days (or reach back to
        `since`): M1, then M15, then H1. Returns (bars, resolution, stale)."""
        for tf, res in (('M1', '1m'), ('M15', '15m'), ('H1', '1h')):
            f = self.frame(tf)
            if f.df.empty:
                continue
            covers = (f.df['time'].min() <= pd.Timestamp(since)) if since is not None \
                else intraday.n_days(f.df) >= days
            if covers or tf == 'H1':
                return f.df, res, f.stale
        return self.frame('H1').df, '1h', True

    def _today_m5(self) -> pd.DataFrame:
        m1 = self.frame('M1').df
        today = self.now().normalize()
        return intraday.resample(m1[m1['time'] >= today], 5)

    def _session_block(self, bars: pd.DataFrame, days, rows_for: str | None = None) -> dict:
        out = {}
        for key in S.all_sessions(self.cfg):
            iv = S.intervals_for(key, days, self.cfg)
            tab = S.session_table(bars, iv)
            last = iv.iloc[-1] if len(iv) else None
            block = {'summary': S.session_summary(tab),
                     'hours_sofia': None if last is None else
                     [t.strftime('%H:%M') for t in to_display([last['start'], last['end']])]}
            if rows_for == key:
                block['rows'] = tab.tail(20).to_dict('records')
            out[key] = block
        return out

    # -- sections -----------------------------------------------------------
    def _overview(self):
        tick = self.source.tick()
        d1 = self.frame('D1')
        d = d1.df
        now = self.now()
        out = {'tick': {'bid': tick['bid'], 'ask': tick['ask'],
                        'spread': None if tick['bid'] is None or tick['ask'] is None else tick['ask'] - tick['bid'],
                        'time': None if tick['time'] is None else pd.Timestamp(tick['time'])}}
        if len(d):
            today = d[d['time'] == now.normalize()]
            row = today.iloc[-1] if len(today) else d.iloc[-1]
            prior = d[d['time'] < row['time']]
            prev_close = float(prior['close'].iloc[-1]) if len(prior) else None
            price = tick['bid'] if tick['bid'] is not None else float(row['close'])
            out['day'] = {'label_day': row['time'].date(), 'is_today': bool(len(today)),
                          'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']),
                          'prev_close': prev_close, **benchmarks.against(price, prev_close)}
            rv = volatility.realised(d, 20).dropna()
            out['hv20_pct'] = float(rv.iloc[-1]) if len(rv) else None
        sess = S.now_session(now, self.cfg)
        out['session'] = sess
        out['intraday_vol'] = volatility.intraday(self._today_m5())
        tick_time = out['tick']['time']
        out['market_open'] = bool(not tick['stale'] and tick_time is not None and now - tick_time < MARKET_OPEN_TICK_AGE)
        as_of = tick_time if tick_time is not None else d1.as_of
        return out, tick['stale'] or d1.stale, as_of

    def _chart(self, timeframe='1h', start=None):
        tf = _choice('timeframe', timeframe, self.cfg['timeframes'])
        s = None if start is None else pd.Timestamp(_date('start', start), tz='UTC')
        bars, stale = self._chart_bars(tf)
        first = bars['time'].min() if len(bars) else None
        cap = self.cfg['chart_max_bars']
        sel = (bars[bars['time'] >= s].head(cap) if s is not None else bars.tail(cap)).reset_index(drop=True)
        data = {'timeframe': tf, 'first_bar': first, 'n_bars': int(len(sel)), 'capped_at': cap,
                'more_before': bool(len(sel) and first is not None and sel['time'].iloc[0] > first),
                'bars': [[t, o, h, l, c] for t, o, h, l, c in
                         zip(sel['time'], sel['open'], sel['high'], sel['low'], sel['close'])]}
        return data, stale, sel['time'].iloc[-1] if len(sel) else None

    def _chart_bars(self, tf: str) -> tuple[pd.DataFrame, bool]:
        R = intraday.resample
        if tf in ('1m', '5m', '30m'):
            m1 = self.frame('M1')
            if tf == '1m':
                return m1.df, m1.stale
            if tf == '5m':
                return R(m1.df, 5), m1.stale
            m15 = self.frame('M15')
            first = m1.df['time'].min() if len(m1.df) else None
            older = m15.df if first is None else m15.df[m15.df['time'] < first]
            return pd.concat([R(older, 30), R(m1.df, 30)], ignore_index=True), m1.stale or m15.stale
        if tf == '15m':
            f = self.frame('M15')
            return f.df, f.stale
        if tf in ('1h', '4h'):
            f = self.frame('H1')
            return (f.df if tf == '1h' else R(f.df, 240, offset_hours=1)), f.stale
        f = self.frame('D1')
        if tf == 'D1':
            return f.df, f.stale
        d = f.df
        week = d['time'].dt.normalize() - pd.to_timedelta(d['time'].dt.weekday, unit='D')   # Monday
        g = d.groupby(week)
        w = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                          'close': g['close'].last()})
        w.index.name = 'time'
        return w.reset_index(), f.stale

    def _sessions(self, period=60, session=None):
        n = _choice('period', period, self.cfg['periods'], int)
        key = None if session is None else _choice('session', session, list(S.all_sessions(self.cfg)))
        bars, res, stale = self._finest(n)
        bars = intraday.last_days(bars, n)
        days = sorted({d.date() for d in bars['time'].dt.normalize() if d.weekday() < 5})
        data = {'period': n, 'resolution': res, 'days_used': len(days),
                'sessions': self._session_block(bars, days, key),
                'overnight_day': S.overnight_against_day(bars, days, self.cfg)}
        return data, stale, bars['time'].max() if len(bars) else None

    def _intraday(self, interval=60, period=60):
        m = _choice('interval', interval, self.cfg['intervals'], int)
        n = _choice('period', period, self.cfg['periods'], int)
        R = intraday.resample
        if m == 60:
            f = self.frame('H1')
            bars, stale, res = f.df, f.stale, '1h'
        elif m == 15:
            f = self.frame('M15')
            bars, stale, res = f.df, f.stale, '15m'
        elif m == 30:
            src, res, stale = self._finest(n)
            src = src if res != '1h' else self.frame('M15').df
            bars = R(src, 30)
            res = '1m' if res == '1m' else '15m'
        else:
            f = self.frame('M1')
            bars, stale, res = (f.df if m == 1 else R(f.df, 5)), f.stale, '1m'
        bars = intraday.last_days(bars, n)
        sp = self.spread()
        out = intraday.time_of_day(bars, m, self.cfg['owner_window'], sp, n)
        out['resolution'] = res
        out['spread_cached_days'] = int(sp['day'].nunique()) if len(sp) else 0
        return out, stale, bars['time'].max() if len(bars) else None

    def _events(self, event='nfp', months=24):
        ev = _choice('event', event, list(self.cfg['events']))
        mo = _choice('months', months, self.cfg['event_months'], int)
        cal = calendar(self.cfg, client=self.fred)
        now = self.now()
        since = now - pd.DateOffset(months=mo)
        allr = cal['events'][cal['events']['event'] == ev]
        rel = allr[(allr['label'] >= since) & (allr['label'] <= now)]
        f = self.frame('M15')
        bars = f.df[f.df['time'] >= since - pd.Timedelta(days=7)]
        nw = news_windows(bars, rel, self.cfg, self.spread())
        upcoming = allr[allr['label'] > now]
        data = {'event': ev, 'months': mo, 'fred': cal['fred'], 'cache_age_days': cal['cache_age_days'],
                'first_bar': f.df['time'].min() if len(f.df) else None,
                'limited': bool(len(f.df) and f.df['time'].min() > since),
                'next': None if upcoming.empty else upcoming['label'].min(),
                **nw}
        return data, f.stale, f.as_of

    def _volatility(self):
        f = self.frame('D1')
        d = f.df
        data = volatility.summary(d, self.cfg)
        data['intraday'] = volatility.intraday(self._today_m5())
        r = self.cfg['regime']
        rv = volatility.realised(d, r['window']).dropna().tail(r['lookback'])
        data['rv_series'] = [[t, v] for t, v in zip(d.loc[rv.index, 'time'], rv)]
        return data, f.stale, f.as_of

    def _history(self, period=None, start=None, end=None):
        f = self.frame('D1')
        d = f.df
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError('a custom range needs both start and end, YYYY-MM-DD')
            s, e = _date('start', start), _date('end', end)
            if e < s:
                raise ValueError('end must not be before start')
            p = history.period(d, start=s, end=e, atr_period=self.cfg['atr_period'],
                               bin_pct=self.cfg['histogram_bin_pct'])
            n = p['days']
            since = pd.Timestamp(s, tz='UTC')
            label = {'start': s, 'end': e}
        else:
            n = _choice('period', 20 if period is None else period, self.cfg['periods'], int)
            p = history.period(d, days=n, atr_period=self.cfg['atr_period'], bin_pct=self.cfg['histogram_bin_pct'])
            since = None
            label = {'period': n}
        data = {**label, **p}
        if p.get('days'):
            bars, res, stale = self._finest(n, since)
            lo, hi = p['first'], p['last'] + pd.Timedelta(days=1)
            bars = bars[(bars['time'] >= lo) & (bars['time'] < hi)]
            days = sorted({x.date() for x in bars['time'].dt.normalize() if x.weekday() < 5})
            data['resolution'] = res
            data['sessions'] = self._session_block(bars, days)
            last = d.iloc[-1]
            rv20 = volatility.realised(d, 20).dropna()
            data['compare'] = {
                'range_pct': {'now': float((last['high'] - last['low']) / last['open'] * 100),
                              'average': p['mean_range_pct']},
                'realised_pct': {'now': float(rv20.iloc[-1]) if len(rv20) else None, 'average': p['realised_pct']},
                'session': self._session_now_vs(bars, days),
            }
        return data, f.stale, f.as_of

    def _session_now_vs(self, bars: pd.DataFrame, days) -> dict | None:
        """The running session's return so far against its period average."""
        now = self.now()
        sess = S.now_session(now, self.cfg)['current']
        if not sess:
            return None
        key = sess[0]
        today = now.date()
        iv = S.intervals_for(key, [today], self.cfg)
        live = self.frame('M1').df
        live = live[live['time'] >= iv['start'].iloc[0]] if len(iv) else live.iloc[0:0]
        now_tab = S.session_table(live, iv)
        avg = S.session_summary(S.session_table(bars, S.intervals_for(key, days, self.cfg)))
        return {'key': key, 'now_ret_pct': float(now_tab['ret_pct'].iloc[0]) if len(now_tab) else None,
                'now_range_pct': float(now_tab['range_pct'].iloc[0]) if len(now_tab) else None,
                'average_ret_pct': avg.get('mean_ret_pct'), 'average_range_pct': avg.get('mean_range_pct')}

    def _benchmarks(self, start=None, end=None):
        now = self.now()
        if start is None:
            s = self._default_start(now)
        else:
            s = _sofia_moment('start', start)
        e = now if end is None else _sofia_moment('end', end)
        if e <= s:
            raise ValueError('end must be after start')
        m1 = self.frame('M1')
        if len(m1.df) and m1.df['time'].min() <= s:
            bars, minutes, stale = m1.df, 1, m1.stale
        else:
            f = self.frame('M15')
            bars, minutes, stale = f.df, 15, f.stale
        tick = self.source.tick()
        price = tick['bid']
        tw = benchmarks.twap(bars, s, e)
        data = {'start': s, 'end': e, 'bar_minutes': minutes, **tw,
                'price_now': price, 'vs_twap': benchmarks.against(price, tw['twap']),
                'arrival': benchmarks.arrival(bars, s, minutes, self.cfg['arrival_horizons_min'])}
        return data, stale or tick['stale'], bars['time'].max() if len(bars) else None

    def _default_start(self, now: pd.Timestamp) -> pd.Timestamp:
        """The latest window open (15:30 Sofia) on a weekday before now."""
        sofia = to_display([now])[0]
        h, m = (int(x) for x in self.cfg['owner_window']['start'].split(':'))
        d = sofia.date()
        for back in range(0, 8):
            day = d - timedelta(days=back)
            cand = _sofia_moment('start', f'{day.isoformat()}T{h:02d}:{m:02d}')
            if day.weekday() < 5 and cand < now:
                return cand
        return now - pd.Timedelta(days=1)


_SERVICE: Service | None = None
_SERVICE_LOCK = threading.Lock()


def build(section: str, **params) -> dict:
    """The module entry for api.py: one shared Service."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = Service()
    return _SERVICE.build(section, **params)
