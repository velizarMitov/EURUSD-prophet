"""Read-only MT5 access for the live passes (spec gold-nyrange-forward/live-data; design D3, D8).

Market-data calls only: initialize (the configured terminal), symbols_get, symbol_info, symbol_select,
copy_rates_from_pos and shutdown. Closed bars come from position 1, so the bar in progress is never
read; any bar whose label + 1 h is after `label_now()` is dropped as well.

"Now" is real time on the Berlin bar-label clock (forecast_eval.m15_data.label_now), never the last
tick: over a weekend the last tick is stale and would make Friday look like today.

`Mt5Feed(cfg, mt5=...)` takes an injected module for tests; tests normally inject a whole fake feed with
the same four methods instead.
"""

from __future__ import annotations

import pandas as pd

OHLC = ['open', 'high', 'low', 'close']


def label_now(now_utc=None) -> pd.Timestamp:
    from ..forecast_eval.m15_data import label_now as _ln
    return _ln(now_utc)


def expiry_date(expiration_time) -> pd.Timestamp | None:
    """MT5's expiration_time (epoch seconds, 0 = none) as a naive calendar date."""
    t = int(expiration_time or 0)
    return pd.Timestamp(t, unit='s').normalize() if t > 0 else None


class Mt5Feed:
    def __init__(self, cfg: dict, mt5=None):
        self.cfg = cfg
        self._mt5 = mt5

    def _module(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        return self._mt5

    def _open(self):
        mt5 = self._module()
        if not mt5.initialize(path=self.cfg['mt5']['terminal_path'], timeout=30000):
            raise ConnectionError(f'MT5 terminal is not reachable: {mt5.last_error()}')
        return mt5

    def symbols(self) -> dict:
        """{name: expiry date or None} for the gold symbol, the lag pair and every listed contract."""
        m = self.cfg['mt5']
        mt5 = self._open()
        try:
            out = {}
            for s in mt5.symbols_get() or []:
                if s.name in (m['gold_symbol'], m['lag_pair_symbol']) or s.name.startswith(m['contract_prefix']):
                    out[s.name] = expiry_date(getattr(s, 'expiration_time', 0))
            return out
        finally:
            mt5.shutdown()

    def closed_bars(self, symbol: str, n: int) -> pd.DataFrame:
        """The last `n` CLOSED H1 bars from position 1: label index and bid OHLC."""
        mt5 = self._open()
        try:
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, int(n))
        finally:
            mt5.shutdown()
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], tz='UTC', name='time'))
        df = pd.DataFrame({k: rates[k] for k in ['time'] + OHLC})
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('time').sort_index().astype(float)
        return df[df.index + pd.Timedelta(hours=1) <= self.label_now()]

    @staticmethod
    def label_now() -> pd.Timestamp:
        return label_now()
