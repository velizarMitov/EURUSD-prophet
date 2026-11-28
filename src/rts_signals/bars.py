"""Read-only MT5 M1 bars with their spread, and the family's append-only caches (spec
rts-signals/trade-simulation; design D5).

    python -m src.rts_signals.bars       top up the six pairs' caches from the mt5app terminal

Market-data calls only: initialize, symbol_select, symbol_info, copy_rates_range and shutdown.
Labels are the broker's Berlin wall clock stored as UTC, like every cache in this repo. "Now" is real
time on that clock (forecast_eval.m15_data.label_now), never the last tick; the minute in progress is
never stored.

Caches: research_models/rts_signals/cache/<SYMBOL>_m1.csv with `time`, bid `open high low close` and
`spread` in the symbol's points. A top-up appends only bars after the cache's last label; an earlier bar
is never rewritten.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from . import load_config, sub

COLS = ['open', 'high', 'low', 'close', 'spread']


def label_now(now_utc=None) -> pd.Timestamp:
    from ..forecast_eval.m15_data import label_now as _ln
    return _ln(now_utc)


class Mt5M1Feed:
    def __init__(self, cfg: dict, mt5=None):
        self.cfg = cfg
        self._mt5 = mt5

    def _module(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        return self._mt5

    def bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        mt5 = self._module()
        if not mt5.initialize(path=os.path.normpath(self.cfg['mt5']['terminal_path']), timeout=30000):
            raise ConnectionError(f'MT5 terminal is not reachable: {mt5.last_error()}')
        try:
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start.to_pydatetime(), end.to_pydatetime())
        finally:
            mt5.shutdown()
        if rates is None or len(rates) == 0:
            return empty()
        df = pd.DataFrame({k: rates[k] for k in ['time'] + COLS})
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df.set_index('time').sort_index().astype(float)


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz='UTC', name='time'), dtype=float)


def cache_file(cfg: dict, symbol: str) -> str:
    return os.path.join(sub(cfg, 'cache'), f'{symbol}_m1.csv')


def load(cfg: dict, symbol: str) -> pd.DataFrame:
    p = cache_file(cfg, symbol)
    if not os.path.exists(p):
        return empty()
    df = pd.read_csv(p)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.set_index('time').sort_index()[COLS].astype(float)


def update(cfg: dict, symbol: str, fresh: pd.DataFrame, now: pd.Timestamp) -> int:
    """Append closed bars of `fresh` labelled after the cache's last label. Never rewrites."""
    fresh = fresh[~fresh.index.duplicated(keep='last')].sort_index()
    fresh = fresh[fresh.index + pd.Timedelta(minutes=1) <= now]
    p = cache_file(cfg, symbol)
    last = None
    if os.path.exists(p):
        with open(p, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 200))
            tail = fh.read().decode('utf-8').strip().splitlines()
        if len(tail) and not tail[-1].startswith('time'):
            last = pd.Timestamp(tail[-1].split(',')[0])
            last = last.tz_localize('UTC') if last.tzinfo is None else last.tz_convert('UTC')
    new = fresh if last is None else fresh[fresh.index > last]
    if len(new):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out = new[COLS].copy()
        out.index.name = 'time'
        out.to_csv(p, mode='a', header=not os.path.exists(p), lineterminator='\n')
    return int(len(new))


def top_up(cfg: dict | None = None, feed=None, now=None) -> dict:
    cfg = cfg or load_config()
    feed = feed or Mt5M1Feed(cfg)
    now = label_now(now)
    out = {}
    for p in cfg['pairs'].values():
        sym = p['symbol']
        c = load(cfg, sym)
        start = (c.index.max() + pd.Timedelta(minutes=1)) if len(c) else pd.Timestamp(cfg['mt5']['history_start'], tz='UTC')
        out[sym] = update(cfg, sym, feed.bars(sym, start, now + pd.Timedelta(minutes=1)), now)
    return out


def main(argv=None):
    argparse.ArgumentParser(description='top up the rts_signals M1 caches (read-only MT5)').parse_args(argv)
    print(top_up(), f'at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC')


if __name__ == '__main__':
    main()
