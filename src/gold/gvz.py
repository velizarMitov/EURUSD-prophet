"""GVZ morning check (openspec change gold-gvz-morning-check, spec gold-forecasting/gvz-check).

GVZ is the CBOE Gold ETF Volatility Index: the 30-day volatility the options
market prices into GLD. It is a MARKET reading shown beside the model's forecast,
never a model claim, and nothing here is ever scored or registered.

Two halves (design D1):

  acquire  refresh(), observe(), settle() -- called only by the hourly gold
           logger. FRED GVZCLS comes through the shared src.macro_data chain
           (API -> public CSV -> cache) into results/gold/macro/gvzcls.csv;
           yfinance ^GVZ tops up the US sessions FRED has not published yet into
           gvz_yfinance.csv. yfinance is imported inside the fetch only.
  read     read_series(), reading() -- files only, no network, no writes; what
           src.gold.summary serves at /api/gold.

Dates are US session dates (YYYY-MM-DD), not gold bar labels. Only completed
sessions are stored: a value dated on or after today's America/New_York date is
dropped. On read, FRED wins on every date it has; a yfinance value is served only
for a later date. The shared FRED chain forward-fills a US holiday with the
previous close -- a past value, so the percentile is barely touched.

The observation log (one row per weekday at the first pass at or after 13:00
Europe/Sofia) and its settlement log (the realised 13:00-23:00 Sofia range) are
append-only and never scored.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import repo_path

NY = ZoneInfo('America/New_York')
SOFIA = ZoneInfo('Europe/Sofia')
LABEL_TZ = ZoneInfo('Europe/Berlin')
FRED_COL = 'gvz'

LOG_FIELDS = ['date', 'logged_at', 'gvz', 'gvz_date', 'gvz_source', 'percentile', 'implied_move_pct',
              'implied_move_usd', 'gold_close', 'gold_close_bar', 'lstm_vol_pct', 'lstm_as_of_bar']
SETTLE_FIELDS = ['date', 'realised_range_pct', 'range_bars', 'first_open', 'settled_at']


class GvzUnavailable(RuntimeError):
    """Neither GVZ source delivered fresh data (named in the message)."""


def settings(cfg2=None) -> dict:
    if cfg2 is None:
        from .prereg2 import load_config_v2
        cfg2 = load_config_v2()
    return cfg2['gvz']


def cache_paths(cfg2) -> tuple[str, str]:
    s, macro = settings(cfg2), repo_path(cfg2['paths']['macro_cache'])
    return os.path.join(macro, s['fred_cache']), os.path.join(macro, s['yahoo_cache'])


# ── clocks ──────────────────────────────────────────────────────────────────

def _utc_now(now=None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    now = pd.Timestamp(now)
    return (now.tz_localize('UTC') if now.tzinfo is None else now).to_pydatetime()


def ny_today(now=None) -> date:
    return _utc_now(now).astimezone(NY).date()


def last_completed_session(now=None) -> date:
    """The US weekday before today's New York date (holidays are not modelled)."""
    d = ny_today(now) - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ── read half: files only ───────────────────────────────────────────────────

def _empty() -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _read_fred(path) -> pd.Series:
    from src.macro_data import _read_cache
    frame = _read_cache(path)
    if frame is None or FRED_COL not in frame:
        return _empty()
    s = pd.to_numeric(frame[FRED_COL], errors='coerce').dropna()
    s.index = s.index.tz_localize(None).normalize()
    return s[~s.index.duplicated(keep='last')].sort_index()


def _read_yahoo(path) -> pd.Series:
    if not os.path.exists(path):
        return _empty()
    try:
        df = pd.read_csv(path)
        s = pd.Series(pd.to_numeric(df[FRED_COL], errors='coerce').values, index=pd.to_datetime(df['date']))
    except (OSError, ValueError, KeyError):
        return _empty()
    s = s.dropna()
    return s[~s.index.duplicated(keep='last')].sort_index()


def read_series(cfg2) -> pd.DataFrame:
    """GVZ by US session date with its source: FRED on every FRED date, yfinance
    only after FRED's newest date."""
    fred_path, yahoo_path = cache_paths(cfg2)
    fred, yahoo = _read_fred(fred_path), _read_yahoo(yahoo_path)
    if len(fred):
        yahoo = yahoo[yahoo.index > fred.index[-1]]
    parts = [pd.DataFrame({FRED_COL: s, 'source': src}) for s, src in ((fred, 'FRED'), (yahoo, 'yfinance')) if len(s)]
    if not parts:
        return pd.DataFrame({FRED_COL: _empty(), 'source': pd.Series(dtype=object)})
    return pd.concat(parts).sort_index()


def reading(cfg2, now=None, gold_close=None, close_bar=None, series=None) -> dict:
    """The /api/gold `gvz` block: value, date, source, percentile against the
    stored history, arithmetic implied daily move, stale flag."""
    s = settings(cfg2)
    ser = read_series(cfg2) if series is None else series
    if ser.empty:
        return {'state': 'none'}
    when, v = ser.index[-1], float(ser[FRED_COL].iloc[-1])
    hist = ser[(ser.index >= pd.Timestamp(s['history_start'])) & (ser.index <= when)][FRED_COL]
    move = v / math.sqrt(s['trading_days_per_year'])
    stale = (ny_today(now) - when.date()).days > s['stale_after_days']
    close = float(gold_close) if gold_close not in (None, '') else None
    return {'state': 'stale' if stale else 'ok', 'stale': stale, 'value': round(v, 2),
            'date': when.date().isoformat(), 'source': ser['source'].iloc[-1],
            'percentile': int(round(100.0 * float((hist < v).mean()))) if len(hist) else None,
            'history_n': int(len(hist)), 'history_start': s['history_start'],
            'implied_move_pct': round(move, 4),
            'implied_move_usd': round(close * move / 100.0, 2) if close is not None else None,
            'gold_close': close, 'gold_close_bar': close_bar}


def newest_date(cfg2):
    ser = read_series(cfg2)
    return ser.index[-1].date() if len(ser) else None


def needs_refresh(cfg2, now=None) -> bool:
    d = newest_date(cfg2)
    return d is None or d < last_completed_session(now)


# ── acquire half: logger only ───────────────────────────────────────────────

def _yahoo_history(symbol: str) -> pd.Series:
    import yfinance as yf
    return yf.Ticker(symbol).history(period='10d')['Close']


def refresh(cfg2, now=None, fred_fetch=None, yahoo_fetch=None) -> str:
    """Refresh both caches. Raises GvzUnavailable when neither source delivers
    fresh data; the caches are then left as they were."""
    from src.macro_data import fetch_fred_feature
    s = settings(cfg2)
    fred_path, yahoo_path = cache_paths(cfg2)
    today = pd.Timestamp(ny_today(now))
    fred_fetch = fred_fetch or fetch_fred_feature
    yahoo_fetch = yahoo_fetch or _yahoo_history
    fails = []

    have = _read_fred(fred_path)
    start = pd.Timestamp(s['history_start']) if have.empty else have.index[-1] - pd.Timedelta(days=30)

    def combine(raw):
        out = raw[[FRED_COL]].dropna()
        return out[out.index.tz_localize(None).normalize() < today]

    try:
        frame, src = fred_fetch({FRED_COL: s['fred_id']}, start, today, combine, fred_path)
        if frame is None or src == 'cache':
            fails.append('FRED unreachable')
    except Exception as e:  # noqa: BLE001 -- the other source may still answer
        fails.append(f'FRED {type(e).__name__}')

    try:
        raw = yahoo_fetch(s['yahoo_symbol'])
        idx = pd.DatetimeIndex(raw.index)
        days = idx.tz_convert(NY).tz_localize(None) if idx.tz is not None else idx
        new = pd.Series(pd.to_numeric(pd.Series(raw.values), errors='coerce').values, index=days.normalize()).dropna()
        new = new[new.index < today]
        merged = pd.concat([x for x in (_read_yahoo(yahoo_path), new) if len(x)] or [_empty()]).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        os.makedirs(os.path.dirname(yahoo_path), exist_ok=True)
        pd.DataFrame({'date': merged.index.strftime('%Y-%m-%d'), FRED_COL: merged.values}).to_csv(yahoo_path, index=False)
    except Exception as e:  # noqa: BLE001
        fails.append(f'yfinance {type(e).__name__}')

    if len(fails) == 2:
        raise GvzUnavailable(f'GVZ: {"; ".join(fails)}')
    ser = read_series(cfg2)
    note = f'fresh {ser.index[-1].date()} {ser["source"].iloc[-1]}' if len(ser) else 'empty'
    return note + (f' ({fails[0]})' if fails else '')


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _append(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerows(rows)


def _label_now(now) -> pd.Timestamp:
    """Now on the bar-label clock: Berlin wall clock, stored as UTC as-is."""
    return pd.Timestamp(_utc_now(now).astimezone(LABEL_TZ).replace(tzinfo=None)).tz_localize('UTC')


def current_lstm(preds_path, now=None) -> tuple[str, str]:
    """(vol_pct, as_of_bar) of the newest h1_lstm forecast whose target hour has
    not closed; blanks when there is none."""
    rows = [r for r in _read_rows(preds_path) if r.get('cell') == 'h1_lstm']
    if not rows:
        return '', ''
    r = rows[-1]
    if pd.Timestamp(r['as_of_bar']) + pd.Timedelta(hours=2) <= _label_now(now):
        return '', ''
    return r.get('vol_pct', ''), r['as_of_bar']


def observe(cfg2, h1: pd.DataFrame, preds_path, now=None, now_iso=None) -> dict | None:
    """One row per weekday, at the first pass at or after `log_after_sofia`."""
    s = settings(cfg2)
    local = _utc_now(now).astimezone(SOFIA)
    hh, mm = (int(x) for x in s['log_after_sofia'].split(':'))
    if local.weekday() >= 5 or (local.hour, local.minute) < (hh, mm):
        return None
    day = local.date().isoformat()
    path = repo_path(s['log'])
    if any(r['date'] == day for r in _read_rows(path)):
        return None
    close, bar = ('', '')
    if h1 is not None and len(h1):
        close, bar = float(h1['close'].iloc[-1]), str(h1.index[-1])
    rd = reading(cfg2, now, close if close != '' else None, bar or None)
    vol, as_of = current_lstm(preds_path, now)
    has = rd['state'] != 'none'
    row = {'date': day, 'logged_at': now_iso or _utc_now(now).isoformat(timespec='seconds'),
           'gvz': rd['value'] if has else '', 'gvz_date': rd['date'] if has else '',
           'gvz_source': rd['source'] if has else '', 'percentile': rd['percentile'] if has else '',
           'implied_move_pct': rd['implied_move_pct'] if has else '',
           'implied_move_usd': '' if not has or rd['implied_move_usd'] is None else rd['implied_move_usd'],
           'gold_close': close, 'gold_close_bar': bar, 'lstm_vol_pct': vol, 'lstm_as_of_bar': as_of}
    _append(path, [row], LOG_FIELDS)
    return row


def settle(cfg2, h1: pd.DataFrame, now=None, now_iso=None) -> list:
    """For each logged date not yet settled, once a bar labelled after that
    date's last session bar exists: the high-low range of the session bars as a
    percent of the first open, blank below `min_session_bars`."""
    s = settings(cfg2)
    first, last = s['session_labels']
    log, out_path = repo_path(s['log']), repo_path(s['settlements'])
    done = {r['date'] for r in _read_rows(out_path)}
    if h1 is None or not len(h1):
        return []
    newest = h1.index[-1]
    out = []
    for r in _read_rows(log):
        if r['date'] in done:
            continue
        day = pd.Timestamp(r['date'], tz='UTC')
        lo_lab, hi_lab = day + pd.Timedelta(hours=first), day + pd.Timedelta(hours=last)
        if newest <= hi_lab:
            continue
        bars = h1[(h1.index >= lo_lab) & (h1.index <= hi_lab)]
        n = len(bars)
        rng, op = '', ''
        if n:
            op = float(bars['open'].iloc[0])
            if n >= s['min_session_bars']:
                rng = round((float(bars['high'].max()) - float(bars['low'].min())) / op * 100.0, 4)
        out.append({'date': r['date'], 'realised_range_pct': rng, 'range_bars': n, 'first_open': op,
                    'settled_at': now_iso or _utc_now(now).isoformat(timespec='seconds')})
        done.add(r['date'])
    _append(out_path, out, SETTLE_FIELDS)
    return out
