"""The per-day tick pull into the 5-minute spread cache (design D3, spec
gold-analytics/market-data "The spread comes from ticks, aggregated per day").

    python -m src.gold_analytics.refresh --days 250

For each finished Monday-Friday label day not yet cached it pulls that day's
ticks (hour chunks), buckets them into 5-minute slots on the label clock and
stores, per slot, the median and 90th-percentile spread (ask - bid, USD) and the
tick count. The tick count is there to judge coverage only and is never shown
as volume. Days already cached are skipped; days the terminal returned no ticks
for (holidays) are remembered in spread_days.json so they are not pulled again.
It also tops up the M1 and M15 bar caches and the NFP/CPI release calendar
from FRED. Page requests never run this.

Writes only under research_models/gold/analytics/.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time as _time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import load_config, repo_path
from .data import MT5Unavailable, Source, merge_bars, read_bar_cache
from .events import calendar

SPREAD_COLUMNS = ['day', 'slot', 'median', 'p90', 'ticks']


def _guard(path: str, cfg: dict) -> str:
    """Refuse any write outside the analytics cache folder."""
    root = os.path.normcase(os.path.abspath(repo_path(cfg['paths']['cache_dir'])))
    p = os.path.normcase(os.path.abspath(path))
    if os.path.commonpath([root, p]) != root:
        raise RuntimeError(f'refresh writes only under {root}, not {path}')
    return path


def _write_gz(df: pd.DataFrame, path: str) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0) as gz:
        gz.write(df.to_csv(index=False, lineterminator='\n').encode('utf-8'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(buf.getvalue())
    os.replace(tmp, path)


def label_today(now: datetime | None = None) -> date:
    """Today on the label clock (Berlin wall time)."""
    return (now or datetime.now(ZoneInfo('Europe/Berlin'))).astimezone(ZoneInfo('Europe/Berlin')).date()


def last_trading_days(n: int, today: date) -> list[date]:
    """The `n` Monday-Friday label days before `today` (today is still forming)."""
    out, d = [], today - timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def bucket(ticks: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Ticks (time, bid, ask) -> one row per slot: median and p90 spread, tick count."""
    if ticks.empty:
        return pd.DataFrame(columns=SPREAD_COLUMNS)
    t = ticks.assign(spread=ticks['ask'] - ticks['bid'], slot=ticks['time'].dt.floor(f'{minutes}min'))
    g = t.groupby('slot')['spread']
    out = pd.DataFrame({'median': g.median(), 'p90': g.quantile(0.9), 'ticks': g.size()}).reset_index()
    out.insert(0, 'day', out['slot'].dt.strftime('%Y-%m-%d'))
    return out[SPREAD_COLUMNS]


def read_spread(cfg: dict | None = None) -> pd.DataFrame:
    """The spread cache with `slot` as label timestamps; empty if never built."""
    cfg = cfg or load_config()
    path = repo_path(cfg['paths']['spread_5m'])
    if not os.path.exists(path):
        return pd.DataFrame({'day': pd.Series(dtype=str), 'slot': pd.Series(dtype='datetime64[ns, UTC]'),
                             'median': pd.Series(dtype=float), 'p90': pd.Series(dtype=float),
                             'ticks': pd.Series(dtype=int)})
    df = pd.read_csv(path)
    df['slot'] = pd.to_datetime(df['slot'], utc=True)
    return df


def _days_path(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['cache_dir']), 'spread_days.json')


def _read_days(cfg: dict) -> dict:
    try:
        with open(_days_path(cfg), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def refresh_spread(source: Source, cfg: dict, days: int, today: date, log=print) -> dict:
    cache = read_spread(cfg)
    done = _read_days(cfg)
    have = set(cache['day'].astype(str)) | set(done)
    wanted = last_trading_days(days, today)
    todo = [d for d in wanted if d.isoformat() not in have]
    log(f'spread cache: {len(wanted) - len(todo)} of {len(wanted)} days cached, pulling {len(todo)}')
    parts = [cache] if len(cache) else []
    for i, d in enumerate(todo, 1):
        t0 = _time.time()
        ticks = source.fetch_ticks(datetime(d.year, d.month, d.day), datetime(d.year, d.month, d.day) + timedelta(days=1))
        rows = bucket(ticks, cfg['spread_bucket_min'])
        if len(rows):
            parts.append(rows)
        done[d.isoformat()] = int(len(ticks))
        log(f'  [{i}/{len(todo)}] {d} {len(ticks):,} ticks -> {len(rows)} slots ({_time.time() - t0:.1f} s)')
        # Save as we go: a long first build can be interrupted and resumed.
        if i % 10 == 0 or i == len(todo):
            _save_spread(parts, done, cfg)
    return {'pulled': len(todo), 'cached': len(wanted) - len(todo), 'days': len(wanted)}


def _save_spread(parts, done, cfg) -> None:
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SPREAD_COLUMNS)
    out = out.drop_duplicates('slot', keep='last').sort_values('slot')
    _write_gz(out, _guard(repo_path(cfg['paths']['spread_5m']), cfg))
    path = _guard(_days_path(cfg), cfg)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(dict(sorted(done.items())), fh, indent=0)


def refresh_bars(source: Source, cfg: dict, now: datetime, log=print) -> dict:
    """Top up the M1 and M15 caches from their newest label (the whole MT5 history
    the first time; the terminal's "Max bars in chart" caps how far back it goes)."""
    out = {}
    for tf, key in (('M1', 'bars_m1'), ('M15', 'bars_m15')):
        path = _guard(repo_path(cfg['paths'][key]), cfg)
        old = read_bar_cache(path)
        start = old['time'].max() if len(old) else pd.Timestamp('2000-01-01', tz='UTC')
        live = source.fetch_bars(tf, start, pd.Timestamp(now) + pd.Timedelta(days=2))
        df = merge_bars(old, live)
        _write_gz(df, path)
        out[tf] = {'rows': len(df), 'first': str(df['time'].min()) if len(df) else None}
        log(f'{tf}: {len(df):,} bars from {out[tf]["first"]}')
    return out


def refresh(days: int = 250, source: Source | None = None, cfg: dict | None = None,
            now: datetime | None = None, log=print, events_client=None) -> dict:
    cfg = cfg or load_config()
    source = source or Source(cfg)
    now = now or datetime.now(ZoneInfo('Europe/Berlin'))
    spread = refresh_spread(source, cfg, days, label_today(now), log)
    bars = refresh_bars(source, cfg, now.replace(tzinfo=None), log)
    _guard(repo_path(cfg['paths']['events']), cfg)
    cal = calendar(cfg, client=events_client, force=True)
    log(f"release calendar: FRED {cal['fred']}, {len(cal['events'])} events")
    return {'spread': spread, 'bars': bars, 'events': {'fred': cal['fred'], 'rows': len(cal['events'])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build the gold analytics spread and bar caches from MT5 (read only).')
    ap.add_argument('--days', type=int, default=250, help='trading days of spread to keep cached (default 250)')
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(repo_path('.env'))  # FRED_API_KEY, as api.py loads it
    except ImportError:
        pass
    try:
        res = refresh(days=args.days)
    except MT5Unavailable as e:
        print(f'MT5 not reachable: {e}', file=sys.stderr)
        return 1
    print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
