"""Read-only MetaTrader 5 access for the forward logger (task 9.1; design D11).

The terminal this attaches to is whatever account is logged in -- today a
REAL-money account. This module can only READ: bars (with each bar's own
spread), symbol metadata (point, digits, swap rates) and the broker SERVER
name. It never reads, stores or returns the account login, and it contains no
call that can place, change or inspect an order or a position; an AST test in
tests/test_forecast_eval_guards.py fails the build if one appears.

Timestamps follow the convention of the existing H1 cache (src/live_data.py):
MT5's raw epoch is localised to UTC as-is, so a bar's label is the broker
server's wall clock. The study's history and the forward rows therefore share
one clock.

The MT5 module is injected so tests run without a terminal.
"""

from __future__ import annotations

import os

import pandas as pd


class MT5Unavailable(RuntimeError):
    pass


TIMEFRAMES = {'H1': 'TIMEFRAME_H1', 'D1': 'TIMEFRAME_D1'}
SOURCE = 'MT5'


class MT5Reader:
    def __init__(self, mt5_module=None):
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module
            except ImportError as e:
                raise MT5Unavailable(f'MetaTrader5 package not installed: {e}')
        self.mt5 = mt5_module
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def connect(self) -> None:
        if not self.mt5.initialize():
            err = getattr(self.mt5, 'last_error', lambda: 'unknown')()
            raise MT5Unavailable(f'MT5 terminal not reachable: {err}')
        self._connected = True

    def close(self) -> None:
        if self._connected:
            self.mt5.shutdown()
            self._connected = False

    def server(self) -> str:
        """Broker server name only. The login is deliberately not read."""
        info = self.mt5.account_info()
        if info is None:
            raise MT5Unavailable('terminal is not logged in to any account')
        return str(info.server)

    def bars(self, symbol: str, timeframe: str, count: int, sync: bool = True) -> pd.DataFrame:
        """Most recent `count` bars, oldest first, INCLUDING the still-forming one
        (callers drop it). Columns: open high low close tick_volume spread."""
        if sync:
            from src.mt5_coverage import sync_symbol
            sync_symbol(self.mt5, symbol, attempts=1, sleep_seconds=0)
        tf = getattr(self.mt5, TIMEFRAMES[timeframe])
        rates = self.mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None or len(rates) == 0:
            raise MT5Unavailable(f'no {timeframe} bars for {symbol}')
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df['time'], unit='s', utc=True)
        df.index.name = 'time'
        return df[['open', 'high', 'low', 'close', 'tick_volume', 'spread']].astype(float).sort_index()

    def symbol_meta(self, symbol: str) -> dict:
        s = self.mt5.symbol_info(symbol)
        if s is None:
            raise MT5Unavailable(f'symbol {symbol} not found')
        return {'point': float(s.point), 'digits': int(s.digits),
                'swap_long': float(s.swap_long), 'swap_short': float(s.swap_short),
                'swap_mode': int(getattr(s, 'swap_mode', -1))}


OFFSET_STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'research_models', 'forward_eval', 'h1_offset_state.json')


def feed_now(h1_index, now_utc=None, state_path: str = OFFSET_STATE) -> pd.Timestamp:
    """The broker server's current time, inferred from the H1 series by the
    production helper. Its offset state goes to a sandbox file, never to the
    serving path's state."""
    from src.live_data import infer_h1_feed_now
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    return infer_h1_feed_now(h1_index, now_utc=now_utc, state_path=state_path)


def closed_bars(df: pd.DataFrame, timeframe: str, now_feed: pd.Timestamp) -> pd.DataFrame:
    """Only fully closed bars, by the production rules (src/live_data.py):
    the forming bar, Saturday bars and pre-open Sunday bars are dropped. On a
    weekend the last Friday bar stays, because its hour has elapsed."""
    from src.live_data import drop_incomplete_bars, drop_incomplete_h1_bars
    if timeframe == 'H1':
        return drop_incomplete_h1_bars(df, now=now_feed)
    return drop_incomplete_bars(df, now=now_feed)
