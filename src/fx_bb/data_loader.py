"""Six-pair H1 data on the verified Berlin label clock (spec fx-bb-reversion/market-data; design D2).

    python -m src.fx_bb.data_loader      refresh USDJPY, USDCAD and USDCHF from MT5

EURUSD, GBPUSD and AUDUSD are READ-ONLY from the fx_session H1 caches; this module never
refreshes or writes them. USDJPY, USDCAD and USDCHF live in this family's own append-only caches
(results/fx_bb/data/<PAIR>_h1.csv: time, bid OHLC, spread in points), written all-or-nothing.

Checks, on every refresh and again on every load:
  * coverage: interior holes and thin months refuse, unless the month is a declared broker gap;
  * clock: the project's weekly-open check, >= 90 % on the stretches between declared gaps;
  * lag: per calendar year from the pair's start, |corr| of the pair's hourly log returns with
    EURUSD's must peak at lag 0 among -1, 0, +1. This catches a one-hour shift that the
    weekly-open check passes (USDCHF 2012-01..2014-11 on this broker, about 96 % weekly match).

Each pair is used only from its configured start (USDCHF 2015-02-01). Bars are never shifted.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..forecast_eval.m15_data import ClockMismatch
from ..fx_fib.artifacts import sha256
from ..fx_session.data_loader import COLS, OHLC, Mt5Source, coverage_holes, load_h1, load_points, \
    verify_clock_segments  # noqa: F401  (Mt5Source re-exported for the CLI)
from ..pooled_h1_data import SymbolResolutionError, resolve_symbol
from . import load_config, repo_path

REFRESH_LOG_FIELDS = ['run_at', 'pair', 'symbol', 'point', 'status', 'appended', 'rows', 'start', 'end',
                      'clock_match_rate', 'clock_weeks', 'lag_years', 'overlap_mismatches', 'detail']


class RefreshRefused(RuntimeError):
    """A pair failed a check; nothing was written for any pair."""


class DataCheckFailed(RuntimeError):
    """A loaded pair failed the clock or lag check; the run is refused."""


def own_pairs(cfg: dict) -> list:
    return [p for p in cfg['pairs'] if cfg['sources'][p] == 'fx_bb']


def _src_cfg(cfg: dict, pair: str) -> dict:
    """A minimal config pointing fx_session's read helpers at the pair's cache directory."""
    key = 'data' if cfg['sources'][pair] == 'fx_bb' else 'fx_session_data'
    return {'paths': {'data': cfg['paths'][key]}}


def data_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['data'])


def cache_path(pair: str, cfg: dict) -> str:
    key = 'data' if cfg['sources'][pair] == 'fx_bb' else 'fx_session_data'
    return os.path.join(repo_path(cfg['paths'][key]), f'{pair}_h1.csv')


def start_of(pair: str, cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg['data']['start'][pair], tz='UTC')


def gaps_of(pair: str, cfg: dict) -> list:
    return cfg['data'].get('known_gaps', {}).get(pair, [])


# -- the lag check ----------------------------------------------------------------------------

def lag_table(pair_close: pd.Series, ref_close: pd.Series, lags=(-1, 0, 1)) -> pd.DataFrame:
    """Per calendar year: |corr(ref log return at t, pair log return at t - k)| for each lag k,
    and the lag where it peaks. Returns on consecutive cached bars of each series."""
    e = np.log(ref_close).diff()
    r = np.log(pair_close).diff()
    j = pd.concat([e, r], axis=1, keys=['e', 'r']).dropna()
    rows = []
    for yr, g in j.groupby(j.index.year):
        cs = {k: abs(g['e'].corr(g['r'].shift(k))) for k in lags}
        rows.append({'year': int(yr), **{f'lag_{k:+d}': cs[k] for k in lags},
                     'peak': max(cs, key=lambda k: -np.inf if np.isnan(cs[k]) else cs[k]), 'n': len(g)})
    return pd.DataFrame(rows).set_index('year') if rows else pd.DataFrame(columns=['peak', 'n'])


def lag_failures(table: pd.DataFrame) -> list:
    return [int(y) for y, p in table['peak'].items() if p != 0]


# -- loading ----------------------------------------------------------------------------------

def load_pair(pair: str, cfg: dict) -> pd.DataFrame:
    """The pair's cached H1 frame from its configured start; never earlier bars."""
    df = load_h1(pair, _src_cfg(cfg, pair))
    return df[df.index >= start_of(pair, cfg)]


def load_point(pair: str, cfg: dict) -> float:
    return float(load_points(_src_cfg(cfg, pair))[pair])


def check_pair(pair: str, df: pd.DataFrame, ref: pd.DataFrame, cfg: dict) -> dict:
    """Clock and lag evidence for one pair, or raise DataCheckFailed naming the pair and year."""
    try:
        ev = verify_clock_segments(df.index, cfg['data']['label_tz'], gaps_of(pair, cfg))
    except ClockMismatch as e:
        raise DataCheckFailed(f'{pair}: clock: {str(e)[:300]}') from e
    out = {'clock_match_rate': ev['match_rate'], 'clock_weeks': ev['n_weeks'], 'lag_years': ''}
    if pair != cfg['reference_pair']:
        tab = lag_table(df['close'], ref['close'], tuple(cfg['data']['lag_check_lags']))
        bad = lag_failures(tab)
        if bad:
            detail = ', '.join(f"{y} peaks at lag {int(tab.loc[y, 'peak']):+d}" for y in bad)
            raise DataCheckFailed(f'{pair}: lag check vs {cfg["reference_pair"]}: {detail} '
                                  '(a shifted label clock; bars are never shifted, start the pair later)')
        out['lag_years'] = f'{tab.index.min()}-{tab.index.max()}'
        out['lag_table'] = tab
    return out


def load_all(cfg: dict | None = None, check: bool = True) -> tuple:
    """({pair: H1 frame from its start}, {pair: point}) for all six pairs, after the checks."""
    cfg = cfg or load_config()
    frames = {p: load_pair(p, cfg) for p in cfg['pairs']}
    points = {p: load_point(p, cfg) for p in cfg['pairs']}
    if check:
        ref = frames[cfg['reference_pair']]
        for p, df in frames.items():
            check_pair(p, df, ref, cfg)
    return frames, points


def cache_digests(cfg: dict) -> dict:
    return {p: sha256(cache_path(p, cfg)) for p in cfg['pairs']}


# -- refresh of the family's own caches -------------------------------------------------------

def _log(cfg: dict, rows: list) -> None:
    path = os.path.join(data_dir(cfg), 'refresh_log.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=REFRESH_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _check(pair: str, df, ref: pd.DataFrame, cfg: dict) -> tuple:
    """Return (frame to merge, evidence) or raise RefreshRefused with a reason."""
    if df is None or df.empty:
        raise RefreshRefused(f'{pair}: MT5 returned no bars')
    df = df[df.index >= start_of(pair, cfg)]
    if df.empty:
        raise RefreshRefused(f'{pair}: no bars from {cfg["data"]["start"][pair]}')
    path = cache_path(pair, cfg)
    first = load_pair(pair, cfg).index.min() if os.path.exists(path) else df.index.min()
    if first > pd.Timestamp(cfg['data']['latest_allowed_start'][pair], tz='UTC'):
        raise RefreshRefused(f'{pair}: earliest bar {first} is after {cfg["data"]["latest_allowed_start"][pair]}')
    holes = coverage_holes(df.index, gaps_of(pair, cfg))
    if holes:
        raise RefreshRefused(f'{pair}: coverage: ' + '; '.join(holes)[:300])
    try:
        ev = check_pair(pair, df, ref, cfg)
    except DataCheckFailed as e:
        raise RefreshRefused(str(e)) from e
    return df, ev


def refresh(cfg: dict | None = None, source=None, now_iso: str | None = None) -> dict:
    """Append every own pair's newly closed H1 bars, or write nothing at all.

    Symbols are resolved for all pairs before any fetch (an unresolved symbol raises
    SymbolResolutionError). Every pair is fetched and checked before any file is written.
    The lag reference (EURUSD) is read from the fx_session cache and never written."""
    cfg = cfg or load_config()
    source = source or Mt5Source()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    pairs = own_pairs(cfg)
    names = source.symbol_names()
    if names is None:
        _log(cfg, [{'run_at': now_iso, 'pair': p, 'status': 'mt5_unreachable', 'appended': 0} for p in pairs])
        return {'status': 'mt5_unreachable', 'pairs': {}}
    resolved = {p: resolve_symbol(p, names) for p in pairs}
    ref = load_pair(cfg['reference_pair'], cfg)

    checked, failures = {}, {}
    for pair, symbol in resolved.items():
        try:
            point = source.symbol_point(symbol)
            if not point:
                raise RefreshRefused(f'{pair}: no point size for {symbol}')
            df, ev = _check(pair, source.fetch_h1(symbol, int(cfg['data']['max_bars'])), ref, cfg)
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
            old = load_pair(pair, cfg)
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
        full = load_pair(pair, cfg)
        row = {'run_at': now_iso, 'pair': pair, 'symbol': symbol, 'point': point, 'status': 'ok',
               'appended': int(len(new)), 'rows': int(len(full)), 'start': str(full.index.min()),
               'end': str(full.index.max()), 'clock_match_rate': round(ev['clock_match_rate'], 4),
               'clock_weeks': ev['clock_weeks'], 'lag_years': ev['lag_years'], 'overlap_mismatches': mismatches}
        rows.append(row)
        report['pairs'][pair] = {k: v for k, v in row.items() if k not in ('run_at', 'pair')}
    with open(os.path.join(data_dir(cfg), 'symbols.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({p: {'symbol': s, 'point': pt} for p, (s, pt, _, _) in checked.items()}, fh, indent=1)
    _log(cfg, rows)
    return report


def main(argv=None):
    argparse.ArgumentParser(description='refresh the fx-bb H1 caches (USDJPY, USDCAD, USDCHF) from MT5').parse_args(argv)
    try:
        print(json.dumps(refresh(), indent=1))
    except SymbolResolutionError as e:
        raise SystemExit(f'refresh stopped for every pair: {e}')


if __name__ == '__main__':
    main()
