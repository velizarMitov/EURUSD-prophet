"""M15 mid bars on the Berlin label clock (spec fx-ny-orb-hot/orb-rule; design D2).

Historical: forecast_eval's `aggregate_m15(load_m1(<PAIR>_M1.parquet))` (read-only), never its cached
`m15_bars`, which writes into another family's research directory. The clock is checked with
`verify_clock` (Europe/Berlin) and refuses on a mismatch.

Every frame here carries bid OHLC, `spread_price` and the mid OHLC = bid + spread_price / 2, plus
`corrupt` (spread above `max_spread_pips`), which voids the bar's day.

Live: `Mt5Live` reads MT5 read-only. Closed bars come from position 1, so the forming bar is never
read; `bar_in_progress_open` reads only the OPEN of the bar in progress, which is final from its
first tick (the entry price of a live signal). Live bars go to results/fx_ny_orb/live/, append-only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..forecast_eval.m15_data import aggregate_m15, load_m1, verify_clock
from . import repo_path

COLS = ['open', 'high', 'low', 'close', 'spread_price']


def with_mid(bars: pd.DataFrame, pip: float, cfg: dict) -> pd.DataFrame:
    out = bars[COLS].astype(float).copy()
    half = out['spread_price'] / 2.0
    for c in ('open', 'high', 'low', 'close'):
        out[f'm{c[0]}'] = out[c] + half
    out['corrupt'] = out['spread_price'] / pip > float(cfg['data']['max_spread_pips'])
    return out


def hist_bars(pair: str, cfg: dict, loader=load_m1, verify=verify_clock) -> pd.DataFrame:
    """Historical M15 bars of `pair` from its M1 parquet; nothing is cached or written."""
    m15 = aggregate_m15(loader(repo_path(cfg['paths']['m1'][pair])))
    verify(m15.index)
    return with_mid(m15, float(cfg['pip'][pair]), cfg)


def by_day(bars: pd.DataFrame) -> dict:
    """{label date (UTC-localised midnight): that date's bars}."""
    idx = pd.DatetimeIndex(bars.index)
    return {d: g for d, g in bars.groupby(idx.normalize())}


# -- live ---------------------------------------------------------------------------------------

class Mt5Live:
    """Read-only MT5 access. `mt5` may be injected for tests."""

    TF = {'H1': 'TIMEFRAME_H1', 'M15': 'TIMEFRAME_M15'}

    def __init__(self, mt5=None):
        self._mt5 = mt5

    def _module(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        return self._mt5

    def _open(self):
        mt5 = self._module()
        if not mt5.initialize():
            raise ConnectionError('MT5 terminal is not reachable')
        return mt5

    def symbol(self, pair: str) -> str:
        from ..pooled_h1_data import resolve_symbol
        mt5 = self._open()
        try:
            return resolve_symbol(pair, [s.name for s in (mt5.symbols_get() or [])])
        finally:
            mt5.shutdown()

    def point(self, symbol: str) -> float:
        mt5 = self._open()
        try:
            return float(mt5.symbol_info(symbol).point)
        finally:
            mt5.shutdown()

    def closed_bars(self, symbol: str, tf: str, n: int) -> pd.DataFrame:
        """The last `n` CLOSED bars (from position 1): labels, bid OHLC and spread in points."""
        mt5 = self._open()
        try:
            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, self.TF[tf]), 1, int(n))
        finally:
            mt5.shutdown()
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'spread'])
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df.set_index('time').sort_index()[['open', 'high', 'low', 'close', 'spread']].astype(float)

    def bar_in_progress_open(self, symbol: str, tf: str = 'M15') -> tuple:
        """(label, bid open, spread points) of the bar in progress: only its open, final at its first tick."""
        mt5 = self._open()
        try:
            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, self.TF[tf]), 0, 1)
        finally:
            mt5.shutdown()
        if rates is None or len(rates) == 0:
            return None
        r = rates[0]
        return pd.Timestamp(int(r['time']), unit='s', tz='UTC'), float(r['open']), float(r['spread'])

    @staticmethod
    def label_now() -> pd.Timestamp:
        """Real time on the bar-label (Berlin wall) clock. The live passes use this, never the last tick:
        over a weekend or an outage the last tick is stale and would make a past day look like today."""
        from ..forecast_eval.m15_data import label_now
        return label_now()

    def server_label_now(self, symbol: str) -> pd.Timestamp:
        """The broker server's wall clock (= the bar-label clock) from the latest tick (diagnostics only)."""
        mt5 = self._open()
        try:
            tick = mt5.symbol_info_tick(symbol)
        finally:
            mt5.shutdown()
        if tick is None:
            raise ConnectionError(f'no tick for {symbol}')
        return pd.Timestamp(int(tick.time), unit='s', tz='UTC')


def live_dir(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['results']), 'live')


def cache_file(pair: str, tf: str, cfg: dict) -> str:
    return os.path.join(live_dir(cfg), f'{pair}_{tf.lower()}.csv')


def load_cache(pair: str, tf: str, cfg: dict) -> pd.DataFrame:
    path = cache_file(pair, tf, cfg)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'spread'])
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.astype(float)


def update_cache(pair: str, tf: str, fresh: pd.DataFrame, cfg: dict) -> dict:
    """Append bars newer than the cache's last label; count overlap revisions; never rewrite."""
    old = load_cache(pair, tf, cfg)
    mism = 0
    if len(old):
        shared = old.index.intersection(fresh.index)
        if len(shared):
            d = (old.loc[shared, ['open', 'high', 'low', 'close']] - fresh.loc[shared, ['open', 'high', 'low', 'close']])
            mism = int((d.abs().to_numpy().max(axis=1) > 1e-9).sum())
        new = fresh[fresh.index > old.index.max()]
    else:
        new = fresh
    if len(new):
        os.makedirs(live_dir(cfg), exist_ok=True)
        out = new[['open', 'high', 'low', 'close', 'spread']].copy()
        out.index.name = 'time'
        path = cache_file(pair, tf, cfg)
        out.to_csv(path, mode='a', header=not os.path.exists(path), lineterminator='\n')
    return {'appended': int(len(new)), 'overlap_mismatches': mism}


def live_m15(pair: str, point: float, cfg: dict) -> pd.DataFrame:
    df = load_cache(pair, 'M15', cfg)
    bars = df[['open', 'high', 'low', 'close']].copy()
    bars['spread_price'] = df['spread'] * point
    return with_mid(bars, float(cfg['pip'][pair]), cfg)


def days_with_corrupt(bars: pd.DataFrame) -> set:
    idx = pd.DatetimeIndex(bars.index)
    return set(idx[bars['corrupt'].to_numpy(bool)].normalize())


__all__ = ['with_mid', 'hist_bars', 'by_day', 'Mt5Live', 'update_cache', 'load_cache', 'live_m15', 'np']
