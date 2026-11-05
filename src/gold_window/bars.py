"""Bars and spreads for the window backtest (design D1).

  H1   results/gold/gold_h1.csv (bid OHLC) + the MT5 per-bar spread, joined on label
  D1   results/gold/gold_d1.csv
  M15  MT5 bars with their spread (MT5 keeps gold M15 only from 2022-08-12)

MT5 is read-only and fetched year by year: one long copy_rates_range fails
with "Invalid params". The fetch is saved once to the gitignored
research_models/gold/window/mt5/ with a digest in artifact_manifest.csv, so a
backtest re-run reads the same inputs. Labels are stored exactly as MT5 gives
them: Berlin wall time localised to UTC as-is, like every cache in this repo.

Spread is kept here as a COST input only (in USD); no model or gate reads it.

    python -m src.gold_window.bars      # fetch from MT5 and write the manifest
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
from datetime import datetime, timezone

import pandas as pd

from . import load_config, repo_path

KEEP = ['time', 'open', 'high', 'low', 'close', 'spread']
FILES = {'H1': 'h1_spread.csv.gz', 'M15': 'm15.csv.gz'}
MANIFEST_FIELDS = ['path', 'sha256', 'rows', 'first', 'last', 'written_at']


def fetch(mt5, symbol: str, timeframe: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Year-chunked read of `timeframe` bars; keeps time, OHLC and spread (USD)."""
    tf = getattr(mt5, f'TIMEFRAME_{timeframe}')
    point = mt5.symbol_info(symbol).point
    parts = []
    for y in range(start_year, end_year + 1):
        r = mt5.copy_rates_range(symbol, tf, datetime(y, 1, 1), datetime(y + 1, 1, 1))
        if r is not None and len(r):
            parts.append(pd.DataFrame({k: r[k] for k in KEEP}))
    if not parts:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'spread_usd'])
    df = pd.concat(parts, ignore_index=True).drop_duplicates('time').sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['spread_usd'] = df.pop('spread').astype(float) * point
    return df.reset_index(drop=True)


def _gz_bytes(df: pd.DataFrame) -> bytes:
    """Deterministic gzip (mtime 0) so the same frame always has the same digest."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as gz:
        gz.write(df.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    return buf.getvalue()


def save(df: pd.DataFrame, path: str) -> str:
    data = _gz_bytes(df)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def update_manifest(manifest: str, rel: str, digest: str, rows: int, first, last) -> None:
    """Replace (or add) the row for `rel`; other rows are kept as they are."""
    rows_out = []
    if os.path.exists(manifest):
        with open(manifest, encoding='utf-8', newline='') as fh:
            rows_out = [r for r in csv.DictReader(fh) if r['path'] != rel]
    rows_out.append({'path': rel, 'sha256': digest, 'rows': rows, 'first': str(first), 'last': str(last),
                     'written_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator='\n')
        w.writeheader()
        w.writerows(sorted(rows_out, key=lambda r: r['path']))


def mt5_path(timeframe: str, cfg: dict | None = None) -> tuple[str, str]:
    cfg = cfg or load_config()
    rel = f"{cfg['paths']['mt5_dir']}/{FILES[timeframe]}"
    return rel, repo_path(rel)


def fetch_all(mt5=None, cfg: dict | None = None, now: datetime | None = None) -> dict:
    """Fetch H1 spreads (2014 on) and M15 bars (2022 on) and pin them in the manifest."""
    from src.gold.data import resolve_gold_symbol
    from src.gold import load_config as gold_config
    cfg = cfg or load_config()
    if mt5 is None:
        import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError('MT5 terminal not reachable')
    try:
        sym = resolve_gold_symbol([s.name for s in mt5.symbols_get()], gold_config()['symbol_aliases'])
        end = (now or datetime.now(timezone.utc)).year
        out = {}
        for tf, start in (('H1', 2014), ('M15', 2022)):
            df = fetch(mt5, sym, tf, start, end)
            rel, path = mt5_path(tf, cfg)
            digest = save(df, path)
            update_manifest(repo_path(cfg['paths']['artifact_manifest']), rel, digest, len(df),
                            df['time'].min() if len(df) else '', df['time'].max() if len(df) else '')
            out[tf] = df
        return out
    finally:
        mt5.shutdown()


def _read_mt5(timeframe: str, cfg: dict | None = None) -> pd.DataFrame:
    _, path = mt5_path(timeframe, cfg)
    return pd.read_csv(path, parse_dates=['time'])


def load_h1(cfg: dict | None = None, cache: pd.DataFrame | None = None,
            spreads: pd.DataFrame | None = None) -> pd.DataFrame:
    """The H1 cache with `spread_usd` joined on label; every cache row is kept."""
    if cache is None:
        cache = pd.read_csv(repo_path('results/gold/gold_h1.csv'), parse_dates=['time'])
    if spreads is None:
        spreads = _read_mt5('H1', cfg)
    out = cache.merge(spreads[['time', 'spread_usd']], on='time', how='left')
    out['spread_usd'] = out['spread_usd'].fillna(0.0)
    return out.sort_values('time').reset_index(drop=True)


def load_d1() -> pd.DataFrame:
    return pd.read_csv(repo_path('results/gold/gold_d1.csv'), parse_dates=['time'])


def load_m15(cfg: dict | None = None) -> pd.DataFrame:
    return _read_mt5('M15', cfg)


if __name__ == '__main__':
    res = fetch_all()
    for tf, df in res.items():
        zero = (df['spread_usd'] <= 0).groupby(df['time'].dt.year).mean().round(2).to_dict()
        print(tf, len(df), df['time'].min(), '..', df['time'].max(), '| share of zero spread by year:', zero)
