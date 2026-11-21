"""MT5 H1 caches with the bar spread (spec fx-session/market-data).

One append-only cache per pair at results/fx_session/data/<PAIR>_h1.csv with the columns
time, open, high, low, close (bid) and spread (points). Labels are the broker's Berlin wall
clock stored UTC-localised as-is, like every cache in this repo; nothing converts a timestamp.
Each pair's point size is written to symbols.json and the refresh log.

    python -m src.fx_session.data_loader      refresh EURUSD, GBPUSD and AUDUSD from MT5

Bars are read from position 1, so the bar still forming is never cached. The refresh is
all-or-nothing: if any pair is unreachable, too shallow, holed or off the Berlin clock,
nothing is written for any pair. A month the config lists under data.known_gaps is a
declared broker gap (GBPUSD 2014-03 holds one bar per day): it is not refused and not filled;
its days fail the session eligibility rule on their own.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..forecast_eval.m15_data import CLOCK_MATCH_MIN, ClockMismatch, clock_evidence
from ..mt5_coverage import find_gaps, find_thin_months, sync_symbol
from ..pooled_h1_data import SymbolResolutionError, resolve_symbol
from . import load_config, repo_path

COLS = ['open', 'high', 'low', 'close', 'spread']
OHLC = COLS[:4]
REFRESH_LOG_FIELDS = ['run_at', 'pair', 'symbol', 'point', 'status', 'appended', 'rows', 'start', 'end',
                      'clock_match_rate', 'clock_weeks', 'overlap_mismatches', 'detail']


class RefreshRefused(RuntimeError):
    """A pair failed a check; nothing was written for any pair."""


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

    def symbol_point(self, symbol: str):
        mt5 = self._module()
        if mt5 is None or not mt5.initialize():
            return None
        try:
            info = mt5.symbol_info(symbol)
            return None if info is None else float(info.point)
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
        return df.set_index('time').sort_index()[COLS].astype(float)


def data_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['data'])


def cache_path(pair: str, cfg: dict) -> str:
    return os.path.join(data_dir(cfg), f'{pair}_h1.csv')


def load_h1(pair: str, cfg: dict | None = None) -> pd.DataFrame:
    """The cached frame: UTC-localised label index (broker wall clock), float OHLC + spread."""
    cfg = cfg or load_config()
    df = pd.read_csv(cache_path(pair, cfg), parse_dates=['time'], index_col='time')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError(f'{pair} cache index is not unique and increasing')
    return df[COLS].astype(float)


def load_points(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    with open(os.path.join(data_dir(cfg), 'symbols.json'), encoding='utf-8') as fh:
        return {p: float(v['point']) for p, v in json.load(fh).items()}


def _log(cfg: dict, rows: list) -> None:
    path = os.path.join(data_dir(cfg), 'refresh_log.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=REFRESH_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def coverage_holes(index, known_months=()) -> list:
    """Interior holes and thin months, except months the config declares as known broker
    gaps (design D2). Never fills anything: an undeclared hole refuses the refresh."""
    known = {str(m) for m in known_months}
    out = []
    for a, b, h in find_gaps(index):
        months = {str(p) for p in pd.period_range(a.tz_localize(None).to_period('M'),
                                                  b.tz_localize(None).to_period('M'), freq='M')}
        if not months <= known:
            out.append(f'missing {a} -> {b} ({h:.1f} h)')
    for period, got, expected in find_thin_months(index, 'H1'):
        if period.strftime('%Y-%m') not in known:
            out.append(f'thin month {period.strftime("%Y-%m")}: {got} bars vs ~{expected}')
    return out


def verify_clock_segments(index, tz: str, known_months=()) -> dict:
    """The project's weekly-open clock check, run on each stretch between declared gap
    months (whose one-bar-per-day data would otherwise read as dozens of weekly opens).
    The pooled match rate must reach the project threshold."""
    idx = pd.DatetimeIndex(index)
    month = idx.tz_localize(None).to_period('M').astype(str)
    gap = month.isin({str(m) for m in known_months})
    seg_id = np.cumsum(np.r_[0, np.diff(gap.astype(int)) != 0])
    matched = weeks = 0
    for k in np.unique(seg_id[~gap]):
        part = idx[(seg_id == k) & ~gap]
        ev = clock_evidence(part, tz)
        if ev['n_weeks']:
            matched += ev['match_rate'] * ev['n_weeks']
            weeks += ev['n_weeks']
    if not weeks:
        raise ClockMismatch('no weekend boundary in the frame: the label clock cannot be established')
    rate = matched / weeks
    if rate < CLOCK_MATCH_MIN:
        raise ClockMismatch(f'bar labels do not match {tz}: only {rate:.1%} of {weeks} weekly opens land on '
                            'the predicted hour. Establish the clock before any session rule is applied.')
    return {'match_rate': rate, 'n_weeks': int(weeks)}


def _check(pair: str, symbol: str, df, cfg: dict) -> tuple:
    """Return (frame to merge, clock evidence) or raise RefreshRefused with a reason."""
    if df is None or df.empty:
        raise RefreshRefused(f'{pair}: MT5 returned no bars')
    df = df[df.index >= pd.Timestamp(cfg['data']['history_start'], tz='UTC')]
    path = cache_path(pair, cfg)
    first = load_h1(pair, cfg).index.min() if os.path.exists(path) else df.index.min()
    if first > pd.Timestamp(cfg['data']['latest_allowed_start'], tz='UTC'):
        raise RefreshRefused(f'{pair}: earliest bar {first} is after {cfg["data"]["latest_allowed_start"]}')
    holes = coverage_holes(df.index, cfg['data'].get('known_gaps', {}).get(pair, []))
    if holes:
        raise RefreshRefused(f'{pair}: coverage: ' + '; '.join(holes)[:300])
    try:
        ev = verify_clock_segments(df.index, cfg['data']['label_tz'], cfg['data'].get('known_gaps', {}).get(pair, []))
    except ClockMismatch as e:
        raise RefreshRefused(f'{pair}: clock: {str(e)[:300]}') from e
    return df, ev


def refresh(cfg: dict | None = None, source=None, now_iso: str | None = None) -> dict:
    """Append every pair's newly closed H1 bars, or write nothing at all.

    Symbols are resolved for all pairs before any fetch (an unresolved symbol raises
    SymbolResolutionError). Every pair is fetched and checked before any file is written."""
    cfg = cfg or load_config()
    source = source or Mt5Source()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    pairs = cfg['factor_pairs']
    names = source.symbol_names()
    if names is None:
        _log(cfg, [{'run_at': now_iso, 'pair': p, 'status': 'mt5_unreachable', 'appended': 0} for p in pairs])
        return {'status': 'mt5_unreachable', 'pairs': {}}
    resolved = {p: resolve_symbol(p, names) for p in pairs}

    checked, failures = {}, {}
    for pair, symbol in resolved.items():
        try:
            point = source.symbol_point(symbol)
            if not point:
                raise RefreshRefused(f'{pair}: no point size for {symbol}')
            df, ev = _check(pair, symbol, source.fetch_h1(symbol, int(cfg['data']['max_bars'])), cfg)
            checked[pair] = (symbol, point, df, ev)
        except RefreshRefused as e:
            failures[pair] = str(e)
    if failures:
        _log(cfg, [{'run_at': now_iso, 'pair': p, 'symbol': resolved[p], 'status': 'refused', 'appended': 0,
                    'detail': failures.get(p, 'not written: another pair was refused')} for p in pairs])
        return {'status': 'refused', 'pairs': failures}

    os.makedirs(data_dir(cfg), exist_ok=True)
    report, rows = {'status': 'ok', 'pairs': {}}, []
    for pair, (symbol, point, df, ev) in checked.items():
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
        out = new[COLS].copy()
        out.index.name = 'time'
        out.to_csv(path, mode='a', header=not os.path.exists(path), lineterminator='\n')
        full = load_h1(pair, cfg)
        row = {'run_at': now_iso, 'pair': pair, 'symbol': symbol, 'point': point, 'status': 'ok',
               'appended': int(len(new)), 'rows': int(len(full)), 'start': str(full.index.min()),
               'end': str(full.index.max()), 'clock_match_rate': round(ev['match_rate'], 4),
               'clock_weeks': ev['n_weeks'], 'overlap_mismatches': mismatches}
        rows.append(row)
        report['pairs'][pair] = {k: v for k, v in row.items() if k not in ('run_at', 'pair')}
    with open(os.path.join(data_dir(cfg), 'symbols.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({p: {'symbol': s, 'point': pt} for p, (s, pt, _, _) in checked.items()}, fh, indent=1)
    _log(cfg, rows)
    return report


def main(argv=None):
    argparse.ArgumentParser(description='refresh the fx-session H1 caches from MT5').parse_args(argv)
    try:
        print(json.dumps(refresh(), indent=1))
    except SymbolResolutionError as e:
        raise SystemExit(f'refresh stopped for every pair: {e}')


if __name__ == '__main__':
    main()
