"""A fake MetaTrader5 module for the gold analytics tests. It carries the volume
and spread fields a real terminal returns, so a test can prove they are dropped,
and records every range call."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

RATE_DTYPE = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
              ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]
TICK_DTYPE = [('time', 'i8'), ('bid', 'f8'), ('ask', 'f8'), ('last', 'f8'), ('volume', 'i8'),
              ('time_msc', 'i8'), ('flags', 'i4'), ('volume_real', 'f8')]


def rates(times, price=2000.0):
    """One bar per label in `times`, each 1.0 above the last."""
    out = np.zeros(len(times), dtype=RATE_DTYPE)
    for i, t in enumerate(times):
        p = price + i
        out[i] = (int(pd.Timestamp(t).timestamp()), p, p + 2, p - 1, p + 1, 77, 25, 99)
    return out


def ticks(rows):
    """rows of (label, bid, ask)."""
    out = np.zeros(len(rows), dtype=TICK_DTYPE)
    for i, (t, bid, ask) in enumerate(rows):
        ms = int(pd.Timestamp(t).timestamp() * 1000)
        out[i] = (ms // 1000, bid, ask, 0.0, 5, ms, 6, 5.0)
    return out


class FakeMT5:
    TIMEFRAME_M1, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_D1 = 1, 15, 16385, 16408
    COPY_TICKS_ALL = -1

    def __init__(self, up=True, bars=None, tick_rows=None, tick=None):
        self.up = up
        self.bars = bars or {}          # timeframe code -> rates array
        self.tick_rows = tick_rows      # (label, bid, ask) rows, filtered per call
        self._tick = tick
        self.rate_calls = []
        self.tick_calls = []
        self.inits = 0

    def initialize(self):
        self.inits += 1
        return self.up

    def shutdown(self):
        pass

    def symbols_get(self):
        return [SimpleNamespace(name='GOLD'), SimpleNamespace(name='EURUSD')]

    def symbol_info_tick(self, sym):
        return self._tick

    def copy_rates_range(self, sym, tf, a, b):
        self.rate_calls.append((tf, a, b))
        arr = self.bars.get(tf)
        if arr is None:
            return None
        lo, hi = int(a.timestamp()), int(b.timestamp())
        return arr[(arr['time'] >= lo) & (arr['time'] < hi)]

    def copy_ticks_range(self, sym, a, b, flags):
        self.tick_calls.append((a, b))
        if self.tick_rows is None:
            return None
        arr = ticks(self.tick_rows)
        lo, hi = int(a.timestamp() * 1000), int(b.timestamp() * 1000)
        return arr[(arr['time_msc'] >= lo) & (arr['time_msc'] < hi)]
