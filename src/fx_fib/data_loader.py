"""MT5 H1 cache for the fx-fibonacci family (spec fx-fibonacci/market-data).

One append-only OHLC cache per configured pair at results/fx_fib/data/<PAIR>_h1.csv.
Labels are the broker server's wall clock (Europe/Berlin) stored UTC-localised as-is,
the convention of every bar cache in this repo; nothing here converts a timestamp.

    python -m src.fx_fib.data_loader      refresh every configured pair from MT5

Source is a live MT5 terminal only -- no other vendor is ever tried. Bars are read
from position 1, so the bar still forming at position 0 is never cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ..forecast_eval.m15_data import ClockMismatch, verify_clock
from ..mt5_coverage import CoverageError, assert_coverage, sync_symbol
from ..pooled_h1_data import SymbolResolutionError, resolve_symbol
from . import load_config, repo_path

OHLC = ['open', 'high', 'low', 'close']
REFRESH_LOG_FIELDS = ['run_at', 'pair', 'symbol', 'status', 'appended', 'rows', 'start', 'end',
                      'clock_match_rate', 'clock_weeks', 'overlap_mismatches', 'detail']


class Mt5Source:
    """Live terminal access. Every method returns None when MT5 is unreachable."""

    def __init__(self, mt5=None):
        self._mt5 = mt5

    def _module(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError:
                return None
            self._mt5 = mt5
        return self._mt5

    def symbol_names(self):
        mt5 = self._module()
        if mt5 is None or not mt5.initialize():
            return None
        try:
            return [s.name for s in (mt5.symbols_get() or [])]
        finally:
            mt5.shutdown()

    def fetch_h1(self, symbol: str, bars: int):
        mt5 = self._module()
        if mt5 is None or not mt5.initialize():
            return None
        try:
            sync_symbol(mt5, symbol)
            info = mt5.terminal_info()
            if info is not None and getattr(info, 'maxbars', 0):
                bars = min(bars, int(info.maxbars) - 1)   # start + count must fit "Max bars in chart"
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, bars)
        finally:
            mt5.shutdown()
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return df.set_index('time').sort_index()[OHLC].astype(float)


def cache_path(pair: str, cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['data']), f'{pair}_h1.csv')


def load_h1(pair: str, cfg: dict | None = None) -> pd.DataFrame:
    """The cached H1 frame: UTC-localised label index (broker wall clock), float OHLC."""
    cfg = cfg or load_config()
    df = pd.read_csv(cache_path(pair, cfg), parse_dates=['time'], index_col='time')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError(f'{pair} cache index is not unique and increasing')
    return df[OHLC].astype(float)


def _append(path: str, new: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    out = new[OHLC].copy()
    out.index.name = 'time'
    out.to_csv(path, mode='a', header=header, lineterminator='\n')


def _log(cfg: dict, rows: list) -> None:
    path = os.path.join(repo_path(cfg['paths']['data']), 'refresh_log.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=REFRESH_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def refresh(cfg: dict | None = None, source=None, now_iso: str | None = None) -> dict:
    """Append every configured pair's newly closed H1 bars to its cache.

    Symbols are resolved for ALL pairs before any bar is fetched; an unresolved or
    ambiguous pair raises SymbolResolutionError and nothing is written. A pair whose
    fetch fails, has a hole or fails the Berlin weekly-open check is reported and its
    cache is left unchanged. Existing rows are never rewritten."""
    cfg = cfg or load_config()
    source = source or Mt5Source()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    names = source.symbol_names()
    if names is None:
        report = {'status': 'mt5_unreachable', 'pairs': {}}
        _log(cfg, [{'run_at': now_iso, 'pair': p, 'status': 'mt5_unreachable', 'appended': 0}
                   for p in cfg['pairs']])
        return report
    resolved = {p: resolve_symbol(p, names) for p in cfg['pairs']}

    start = pd.Timestamp(cfg['data']['history_start'], tz='UTC')
    report = {'status': 'ok', 'pairs': {}}
    log_rows = []
    for pair, symbol in resolved.items():
        row = {'run_at': now_iso, 'pair': pair, 'symbol': symbol, 'appended': 0}
        df = source.fetch_h1(symbol, int(cfg['data']['max_bars']))
        try:
            if df is None or df.empty:
                raise LookupError('MT5 returned no bars')
            df = df[df.index >= start]
            assert_coverage(df, 'H1', label=f'MT5 {symbol} H1')
            ev = verify_clock(df.index, cfg['data']['label_tz'])
        except LookupError as e:
            row.update(status='mt5_unreachable', detail=str(e))
        except CoverageError as e:
            row.update(status='coverage_error', detail=str(e)[:300])
        except ClockMismatch as e:
            row.update(status='clock_mismatch', detail=str(e)[:300])
        else:
            path = cache_path(pair, cfg)
            mismatches = 0
            if os.path.exists(path):
                old = load_h1(pair, cfg)
                shared = old.index.intersection(df.index)
                if len(shared):
                    diff = (old.loc[shared, OHLC] - df.loc[shared, OHLC]).abs().to_numpy().max(axis=1)
                    mismatches = int((diff > 1e-9).sum())
                new = df[df.index > old.index.max()]
            else:
                new = df
            _append(path, new)
            rows = load_h1(pair, cfg)
            row.update(status='ok', appended=int(len(new)), rows=int(len(rows)),
                       start=str(rows.index.min()), end=str(rows.index.max()),
                       clock_match_rate=round(ev['match_rate'], 4), clock_weeks=ev['n_weeks'],
                       overlap_mismatches=mismatches)
        report['pairs'][pair] = {k: v for k, v in row.items() if k not in ('run_at', 'pair')}
        log_rows.append(row)
    if any(r['status'] != 'ok' for r in log_rows):
        report['status'] = 'partial'
    _log(cfg, log_rows)
    return report


def main(argv=None):
    argparse.ArgumentParser(description='refresh the fx-fibonacci H1 caches from MT5').parse_args(argv)
    try:
        print(json.dumps(refresh(), indent=1))
    except SymbolResolutionError as e:
        raise SystemExit(f'refresh stopped for every pair: {e}')


if __name__ == '__main__':
    main()
