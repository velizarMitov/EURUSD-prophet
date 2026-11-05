"""The read-only MT5 source with a cache fallback (design D2, spec
gold-analytics/market-data "MT5 is read only").

Every time here is a LABEL: the broker's Europe/Berlin wall time localised to
UTC as-is, exactly like every cache in the repo (clock.py converts). MT5 is
asked with label datetimes, so the epoch it compares is on the same clock.

Only time, OHLC, bid and ask cross the boundary: rates and ticks are
whitelisted field by field, so the volume fields never reach a frame.

Each call opens and closes its own MT5 session (as src/live_data.py does), under
one lock, because the terminal connection is process-global and the API serves
requests from a thread pool. When the terminal cannot be initialised, the call
returns the cached frame with stale=True and as_of = its newest label.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import load_config, repo_path

OHLC = ['open', 'high', 'low', 'close']
BAR_FIELDS = ['time'] + OHLC
TICK_FIELDS = ['time_msc', 'bid', 'ask']
TIMEFRAMES = ('M1', 'M15', 'H1', 'D1')
BAR_SECONDS = {'M1': 60, 'M15': 900, 'H1': 3600, 'D1': 86400}

_MT5_LOCK = threading.Lock()


class MT5Unavailable(RuntimeError):
    pass


@dataclass
class Frame:
    df: pd.DataFrame
    stale: bool
    as_of: pd.Timestamp | None


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({'time': pd.Series(dtype='datetime64[ns, UTC]'),
                         **{c: pd.Series(dtype=float) for c in OHLC}})


def _empty_ticks() -> pd.DataFrame:
    return pd.DataFrame({'time': pd.Series(dtype='datetime64[ns, UTC]'),
                         'bid': pd.Series(dtype=float), 'ask': pd.Series(dtype=float)})


def _as_label(ts) -> datetime:
    """A tz-aware UTC datetime carrying label wall time, for the MT5 range calls."""
    return utc_ts(ts).to_pydatetime()


def rates_frame(rates) -> pd.DataFrame:
    """MT5 rates -> time + OHLC. Fields are picked by name, nothing else is kept."""
    if rates is None or len(rates) == 0:
        return _empty_bars()
    df = pd.DataFrame({k: rates[k] for k in BAR_FIELDS})
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df[OHLC] = df[OHLC].astype(float)
    return df


def ticks_frame(ticks) -> pd.DataFrame:
    """MT5 ticks -> time (ms precision) + bid + ask."""
    if ticks is None or len(ticks) == 0:
        return _empty_ticks()
    df = pd.DataFrame({k: ticks[k] for k in TICK_FIELDS})
    df['time'] = pd.to_datetime(df.pop('time_msc'), unit='ms', utc=True)
    df[['bid', 'ask']] = df[['bid', 'ask']].astype(float)
    return df[['time', 'bid', 'ask']]


def year_chunks(start, end) -> list[tuple[datetime, datetime]]:
    """[start, end) split at each 1 January: one long copy_rates_range fails with
    "Invalid params"."""
    s, e = _as_label(start), _as_label(end)
    out = []
    while s < e:
        nxt = datetime(s.year + 1, 1, 1, tzinfo=timezone.utc)
        out.append((s, min(nxt, e)))
        s = nxt
    return out


def month_chunks(start, end) -> list[tuple[datetime, datetime]]:
    """[start, end) split at each month start. M1 needs it: a year of M1 is more
    than the terminal's "Max bars in chart" (100,000 here), and MT5 then returns
    nothing at all for the request."""
    s, e = _as_label(start), _as_label(end)
    out = []
    while s < e:
        nxt = datetime(s.year + (s.month == 12), s.month % 12 + 1, 1, tzinfo=timezone.utc)
        out.append((s, min(nxt, e)))
        s = nxt
    return out


def hour_chunks(start, end) -> list[tuple[datetime, datetime]]:
    s, e = _as_label(start), _as_label(end)
    out = []
    while s < e:
        out.append((s, min(s + timedelta(hours=1), e)))
        s += timedelta(hours=1)
    return out


def utc_ts(x) -> pd.Timestamp:
    """A label timestamp, tz-aware UTC (naive input is read as a label)."""
    t = pd.Timestamp(x)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _newest(df: pd.DataFrame):
    return df['time'].max() if len(df) else None


def read_bar_cache(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return _empty_bars()
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df[BAR_FIELDS].sort_values('time').reset_index(drop=True)


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union on label; a live bar replaces the cached one with the same label."""
    parts = [d for d in (old, new) if len(d)]
    if not parts:
        return _empty_bars()
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates('time', keep='last').sort_values('time').reset_index(drop=True)


class Source:
    """Gold from MT5, read only: tick(), bars(), ticks()."""

    def __init__(self, cfg: dict | None = None, mt5=None):
        self.cfg = cfg or load_config()
        self._mt5 = mt5

    # -- the MT5 session --------------------------------------------------
    def _module(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5  # Windows-only; a missing package is "unreachable"
            self._mt5 = mt5
        return self._mt5

    @contextlib.contextmanager
    def session(self):
        """(mt5, symbol) for one short session, or MT5Unavailable."""
        with _MT5_LOCK:
            try:
                mt5 = self._module()
            except ImportError as e:
                raise MT5Unavailable(f'MetaTrader5 package unavailable: {e}') from e
            if not mt5.initialize():
                raise MT5Unavailable('MT5 terminal not reachable')
            try:
                from src.gold.data import resolve_gold_symbol
                syms = mt5.symbols_get() or []
                sym = resolve_gold_symbol([s.name for s in syms], self.cfg['symbol_aliases'])
                yield mt5, sym
            finally:
                mt5.shutdown()

    def path(self, key: str) -> str:
        return repo_path(self.cfg['paths'][key])

    # -- live tick --------------------------------------------------------
    def tick(self) -> dict:
        """The live bid and ask. Falls back to the last tick seen, then to the
        newest cached H1 close, marked stale."""
        try:
            with self.session() as (mt5, sym):
                t = mt5.symbol_info_tick(sym)
                if t is None:
                    raise MT5Unavailable('no tick')
                out = {'bid': float(t.bid), 'ask': float(t.ask),
                       'time': pd.Timestamp(int(t.time_msc), unit='ms', tz='UTC').isoformat()}
            self._save_last_tick(out)
            return {**out, 'stale': False, 'as_of': out['time']}
        except MT5Unavailable:
            last = self._read_last_tick()
            if last:
                return {**last, 'stale': True, 'as_of': last['time']}
            h1 = read_bar_cache(self.path('h1'))
            if not len(h1):
                return {'bid': None, 'ask': None, 'time': None, 'stale': True, 'as_of': None}
            row = h1.iloc[-1]
            t = (row['time'] + pd.Timedelta(hours=1)).isoformat()
            return {'bid': float(row['close']), 'ask': None, 'time': t, 'stale': True, 'as_of': t}

    def _save_last_tick(self, tick: dict) -> None:
        path = self.path('last_tick')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(tick, fh)
        except OSError:
            pass  # a cache miss only costs the stale fallback

    def _read_last_tick(self) -> dict | None:
        try:
            with open(self.path('last_tick'), encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    # -- bars -------------------------------------------------------------
    def _cache_path(self, tf: str) -> str:
        return self.path({'M1': 'bars_m1', 'M15': 'bars_m15', 'H1': 'h1', 'D1': 'd1'}[tf])

    def fetch_bars(self, tf: str, start, end) -> pd.DataFrame:
        """Live bars in [start, end), chunked by year (M1 by month). Raises MT5Unavailable."""
        if tf not in TIMEFRAMES:
            raise ValueError(f'timeframe must be one of {list(TIMEFRAMES)}')
        chunks = month_chunks(start, end) if tf == 'M1' else year_chunks(start, end)
        with self.session() as (mt5, sym):
            code = getattr(mt5, f'TIMEFRAME_{tf}')
            parts = [rates_frame(mt5.copy_rates_range(sym, code, a, b)) for a, b in chunks]
        return merge_bars(_empty_bars(), pd.concat(parts, ignore_index=True) if parts else _empty_bars())

    def bars(self, tf: str, start=None, end=None) -> Frame:
        """Cached bars topped up live from the newest cached label. With MT5
        closed, the cache alone, stale."""
        if tf not in TIMEFRAMES:
            raise ValueError(f'timeframe must be one of {list(TIMEFRAMES)}')
        cached = read_bar_cache(self._cache_path(tf))
        end_ts = pd.Timestamp(end, tz='UTC') if end is not None and pd.Timestamp(end).tzinfo is None \
            else (pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=2))
        top_from = _newest(cached)
        if top_from is None:
            top_from = pd.Timestamp(start, tz='UTC') if start is not None else end_ts - pd.Timedelta(days=30)
        stale = False
        try:
            live = self.fetch_bars(tf, top_from, end_ts)
            df = merge_bars(cached, live)
        except MT5Unavailable:
            df, stale = cached, True
        if start is not None:
            df = df[df['time'] >= utc_ts(start)]
        if end is not None:
            df = df[df['time'] < end_ts]
        df = df.reset_index(drop=True)
        return Frame(df, stale, _newest(df))

    # -- ticks ------------------------------------------------------------
    def fetch_ticks(self, start, end) -> pd.DataFrame:
        """Live bid/ask ticks in [start, end), in hour chunks. Raises MT5Unavailable."""
        with self.session() as (mt5, sym):
            flag = getattr(mt5, 'COPY_TICKS_ALL', -1)
            parts = [ticks_frame(mt5.copy_ticks_range(sym, a, b, flag)) for a, b in hour_chunks(start, end)]
        parts = [p for p in parts if len(p)]
        if not parts:
            return _empty_ticks()
        return pd.concat(parts, ignore_index=True).sort_values('time', kind='stable').reset_index(drop=True)

    def ticks(self, start, end) -> Frame:
        try:
            df = self.fetch_ticks(start, end)
            return Frame(df, False, _newest(df))
        except MT5Unavailable:
            return Frame(_empty_ticks(), True, None)
