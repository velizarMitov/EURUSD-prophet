"""Gold logger health (openspec change gold-forecasting-v2, spec
gold-forecasting/logger-health).

Every logger pass, including a failed one, appends one row to
results/gold/forward/health.csv. `status` turns those rows and the forward logs
into what the owner needs to know: when the logger last ran, when it last
succeeded, and how many market hours of the last 24 have no forecast. `stalled`
decides whether /gold warns.

Read-only apart from `record`, which only the logger calls. Imports no model
library, so api.py can serve /api/gold/health cheaply.

Market hours, on the New York clock where the metals' session is defined: open
from Sunday 18:00 to Friday 17:00, with a daily break 17:00-18:00. Bar labels are
Europe/Berlin wall clock stored as UTC (src/gold/summary.label_clock_now).
Exchange holidays are not modelled: a holiday can raise a false stall warning,
never hide a real one.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import load_config, repo_path
from .summary import LABEL_TZ, label_clock_now

HEALTH_FIELDS = ['started_at', 'ended_at', 'outcome', 'mt5', 'newest_h1', 'forecasts', 'settled',
                 'gaps', 'note']
NY = ZoneInfo('America/New_York')
STALL_HOURS = 2
MAX_MISSED_24H = 3
SETTLE_GRACE_MIN = 15                 # the logger runs at :03; give it time before calling a bar missed


def health_path(cfg) -> str:
    return os.path.join(repo_path(cfg['paths']['forward']), 'health.csv')


def record(row: dict, cfg=None) -> None:
    """Append one pass to health.csv (the logger's only call into this module)."""
    cfg = cfg or load_config()
    path = health_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=HEALTH_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in HEALTH_FIELDS})


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def label_to_ny(label: pd.Timestamp) -> datetime:
    """A bar label (Berlin wall clock stored as UTC) as a New York time."""
    wall = pd.Timestamp(label).tz_localize(None) if pd.Timestamp(label).tzinfo else pd.Timestamp(label)
    return wall.tz_localize(LABEL_TZ, ambiguous=True, nonexistent='shift_forward').tz_convert(NY).to_pydatetime()


def market_open(label: pd.Timestamp) -> bool:
    """Is the hour starting at this bar label inside the metals session?"""
    ny = label_to_ny(label)
    wd, h = ny.weekday(), ny.hour            # Monday = 0 ... Sunday = 6
    if h == 17:
        return False                          # daily break
    if wd == 5:
        return False                          # Saturday
    if wd == 4 and h >= 17:
        return False                          # Friday after the close
    if wd == 6 and h < 18:
        return False                          # Sunday before the open
    return True


def expected_h1_bars(now_label: pd.Timestamp, hours: int = 24) -> list:
    """H1 bar labels of the last `hours` that have closed (with a grace period)
    and fall inside market hours: the bars a healthy logger would have forecast."""
    newest_closed = (now_label - pd.Timedelta(minutes=SETTLE_GRACE_MIN)).floor('h') - pd.Timedelta(hours=1)
    bars = pd.date_range(end=newest_closed, periods=hours, freq='h')
    return [b for b in bars if market_open(b)]


def _ts(s):
    try:
        t = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None
    return t if t.tzinfo else t.tz_localize('UTC')


def status(cfg=None, now=None) -> dict:
    """Last pass, last success, missed market hours in the last 24, and whether
    the market is open now. `now` is a real-time instant (UTC-aware) or None."""
    cfg = cfg or load_config()
    now_utc = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize('UTC')
    now_label = label_clock_now(now_utc.to_pydatetime())
    rows = _rows(health_path(cfg))
    last = rows[-1] if rows else None
    ok = [r for r in rows if r.get('outcome') == 'ok' and r.get('mt5') == 'yes']
    last_ok = ok[-1] if ok else None

    fwd = repo_path(cfg['paths']['forward'])
    h1 = [r for r in _rows(os.path.join(fwd, 'predictions.csv')) if r.get('cadence') == 'H1']
    logged = {pd.Timestamp(r['as_of_bar']) for r in h1}
    expected = expected_h1_bars(now_label)
    missed = [str(b) for b in expected if b not in logged]

    # Before the first pass that writes health.csv, the newest logged H1 forecast
    # is the best evidence of the last success: a fresh install must not warn.
    success_at = _ts(last_ok['ended_at']) if last_ok else None
    if success_at is None and h1:
        success_at = max((t for t in (_ts(r.get('logged_at')) for r in h1) if t is not None), default=None)
    since_ok_h = None if success_at is None else (now_utc - success_at).total_seconds() / 3600.0
    out = {'generated_at': now_utc.isoformat(), 'market_open': market_open(now_label.floor('h')),
           'last_pass': last, 'last_success': last_ok,
           'hours_since_success': None if since_ok_h is None else round(since_ok_h, 2),
           'missed_24h': len(missed), 'missed_bars': missed[-24:],
           'expected_24h': len(expected), 'passes': len(rows)}
    out['stalled'] = stalled(out)
    return out


def stalled(st: dict) -> bool:
    """Warn only while the market is open: no successful pass for more than two
    hours, or more than three market hours missed in the last 24."""
    if not st.get('market_open'):
        return False
    h = st.get('hours_since_success')
    if h is None or h > STALL_HOURS:
        return True
    return st.get('missed_24h', 0) > MAX_MISSED_24H
