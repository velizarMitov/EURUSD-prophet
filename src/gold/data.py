"""Gold bars from MT5, OHLC only (design D2; spec gold-forecasting/study).

Source is the live MT5 terminal; the on-disk cache under results/gold/ is the
only offline fallback. No other vendor is ever tried: yfinance GC=F is a futures
contract, not the broker's CFD, and mixing instruments contaminates the sample.

Columns are WHITELISTED, not blacklisted: only `time` and OHLC are read out of
the rate records, so volume and spread never reach memory beyond the raw
record, never reach disk, and no model can consume them.

Bar labels are the broker server's wall clock (Europe/Berlin), localised to UTC
as-is -- the same convention as every other cache in this repo (CLAUDE.md
invariant 6). They are NOT UTC.
"""

from __future__ import annotations

import os

import pandas as pd

from . import load_config, repo_path

OHLC = ['open', 'high', 'low', 'close']
BAR_SECONDS = {'D1': 86400, 'H1': 3600}

# Gold (CME metals hours) is CLOSED on Good Friday; FX is not. The shared 72 h
# coverage guard therefore reads every Easter as a hole: all 12 gaps > 72 h in
# the 2013-2026 history (D1 96 h, H1 74-96 h) span a Good Friday, and none
# other exists. Only a gap that spans Good Friday and is at most this long is
# exempt; every other gap is still refused.
GOOD_FRIDAY_MAX_GAP_HOURS = 100.0


class GoldDataUnavailable(RuntimeError):
    """No terminal, no usable live fetch, and no cache: the run stops."""


def resolve_gold_symbol(available_names, aliases) -> str:
    """First alias that resolves to exactly one broker symbol. A single base
    name is not enough: resolve_symbol('XAUUSD') does not match `GOLD`."""
    from src.pooled_h1_data import SymbolResolutionError, resolve_symbol
    names = list(available_names)
    errors = []
    for alias in aliases:
        try:
            return resolve_symbol(alias, names)
        except SymbolResolutionError as e:
            errors.append(str(e))
    raise GoldDataUnavailable('no gold symbol resolved: ' + ' | '.join(errors))


def _rates_to_frame(rates) -> pd.DataFrame:
    df = pd.DataFrame({k: rates[k] for k in ['time'] + OHLC})
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df.set_index('time').sort_index().astype(float)


def _closed_only(df: pd.DataFrame, timeframe: str, server_now) -> pd.DataFrame:
    """Drop the bar still forming. `server_now` is the last tick's epoch on the
    bar-label clock; without it the newest bar is dropped (never kept unsure)."""
    if df.empty:
        return df
    if server_now is None:
        return df.iloc[:-1]
    now = pd.Timestamp(int(server_now), unit='s', tz='UTC')
    closes = df.index + pd.Timedelta(seconds=BAR_SECONDS[timeframe])
    return df[closes <= now]


def _good_friday(year: int) -> pd.Timestamp:
    from dateutil.easter import easter
    return pd.Timestamp(easter(year)) - pd.Timedelta(days=2)


def _spans_good_friday(a, b, hours: float) -> bool:
    gf = _good_friday(pd.Timestamp(a).year)
    return hours <= GOOD_FRIDAY_MAX_GAP_HOURS and pd.Timestamp(a).normalize() < gf < pd.Timestamp(b)


def trim_leading_thin(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop the run of thin months at the START of the history. The broker's
    gold H1 before 2014-12 carries ~21 bars a month (one a day): daily bars
    labelled H1, not a hole to repair. A thin month after the first full one is
    left in place for check_coverage to refuse."""
    from src.mt5_coverage import find_thin_months
    thin = {p for p, _, _ in find_thin_months(df.index, timeframe)}
    if not thin:
        return df
    idx = df.index.tz_convert(None) if df.index.tz is not None else df.index
    months = pd.Series(idx.to_period('M').to_timestamp(), index=df.index)
    first_full = None
    for m in sorted(months.unique())[1:]:          # the first month is exempt (partial)
        if m not in thin:
            first_full = m
            break
    if first_full is None:
        return df
    leading = sorted(t for t in thin if t < first_full)
    return df[months >= first_full] if leading else df


def check_coverage(df: pd.DataFrame, timeframe: str, label: str) -> pd.DataFrame:
    """The shared coverage guard with the single gold exemption (Good Friday).
    Raises CoverageError naming every other gap or thin month."""
    from src.mt5_coverage import CoverageError, find_gaps, find_thin_months
    gaps = [(a, b, h) for a, b, h in find_gaps(df.index) if not _spans_good_friday(a, b, h)]
    thin = find_thin_months(df.index, timeframe)
    if gaps or thin:
        parts = [f'missing {a} -> {b} ({h:.1f} h)' for a, b, h in gaps]
        parts += [f'thin month {p:%Y-%m}: {got} bars vs ~{exp}' for p, got, exp in thin]
        raise CoverageError(f'{label} ({timeframe}) has a coverage hole and was REFUSED, '
                            f'not repaired: ' + '; '.join(parts))
    return df


def fetch_live(timeframe: str, count: int, mt5=None, aliases=None):
    """(frame, symbol) from a live terminal, or None when none is reachable.
    Raises CoverageError for a holed frame (refused, never repaired)."""
    from src.mt5_coverage import sync_symbol
    if mt5 is None:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            return None
    aliases = aliases or load_config()['symbol_aliases']
    if not mt5.initialize():
        return None
    try:
        symbol = resolve_gold_symbol([s.name for s in (mt5.symbols_get() or [])], aliases)
        sync_symbol(mt5, symbol)
        tf = {'D1': mt5.TIMEFRAME_D1, 'H1': mt5.TIMEFRAME_H1}[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        tick = mt5.symbol_info_tick(symbol)
    finally:
        mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    df = _closed_only(_rates_to_frame(rates), timeframe, getattr(tick, 'time', None))
    df = trim_leading_thin(df, timeframe)
    return check_coverage(df, timeframe, label=f'MT5 {symbol} {timeframe}'), symbol


def read_cache(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=['time'] + OHLC)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.set_index('time').sort_index().astype(float)


def append_cache(path: str, fresh: pd.DataFrame) -> pd.DataFrame:
    """Append bars newer than the cache's last bar. Cached bars are never
    rewritten, so a bar once used for a forecast keeps its recorded values."""
    old = read_cache(path)
    if old is not None and not old.empty:
        fresh = fresh[fresh.index > old.index[-1]]
        merged = pd.concat([old, fresh])
    else:
        merged = fresh
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if old is None or not fresh.empty:
        out = merged.reset_index()[['time'] + OHLC]
        out['time'] = out['time'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        tmp = path + '.tmp'
        out.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return merged


def _meta_path(path: str) -> str:
    return path[:-4] + '.meta.json' if path.endswith('.csv') else path + '.meta.json'


def _write_symbol(path: str, symbol: str) -> None:
    import json
    with open(_meta_path(path), 'w', encoding='utf-8') as fh:
        json.dump({'symbol': symbol}, fh)


def cached_symbol(path: str) -> str | None:
    """The broker symbol the cache was last fetched from."""
    import json
    try:
        with open(_meta_path(path), encoding='utf-8') as fh:
            return json.load(fh).get('symbol')
    except (OSError, ValueError):
        return None


def load(timeframe: str, mt5=None, allow_fetch: bool = True, cfg: dict | None = None):
    """(frame, record). `record` says where the bars came from, for run records."""
    from src.mt5_coverage import CoverageError
    cfg = cfg or load_config()
    path = repo_path(cfg['paths'][{'D1': 'd1_cache', 'H1': 'h1_cache'}[timeframe]])
    record = {'timeframe': timeframe, 'cache': os.path.relpath(path, repo_path('')).replace('\\', '/')}
    live = None
    if allow_fetch:
        try:
            live = fetch_live(timeframe, cfg['bars'][timeframe], mt5=mt5, aliases=cfg['symbol_aliases'])
        except CoverageError as e:
            record['live_refused'] = str(e)
    if live is not None:
        frame, symbol = live
        df = append_cache(path, frame)
        _write_symbol(path, symbol)
        record.update(source='mt5', symbol=symbol)
    else:
        df = read_cache(path)
        if df is None:
            raise GoldDataUnavailable(f'no MT5 terminal reachable and no cache at {record["cache"]}')
        record.update(source='cache', symbol=cached_symbol(path))
    record.update(rows=len(df), first=str(df.index[0]), last=str(df.index[-1]))
    return df, record
