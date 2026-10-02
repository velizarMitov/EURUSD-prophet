"""M15 session data layer (tasks 14.1-14.4; spec horizon-study "Mid-price series
for sub-hourly cadences", "The bar-label clock is established by measurement",
"Session-restricted evaluation window for sub-hourly cells"; design D15, D16).

THE CLOCK. Every bar cache in this repository stores MT5's epoch localised to UTC
as-is (src/live_data.py), so a label is the broker SERVER's wall clock, not UTC.
Measured on the M1 file, the weekly market open label is `Sun 23:00` year-round
and exactly `22:00` in the March and late-October weeks when US and EU daylight
saving disagree. The forex week opens Sunday 17:00 America/New_York, and only
Europe/Berlin (CET/CEST) reproduces all four cases -- `verify_clock` holds that
against every weekend in the frame rather than assuming it.

Labels are LEFT as UTC-localised wall clock, because the H1 study and the forward
logger already share that convention. Nothing here converts a timestamp; the
session is a label time-of-day window:

    session   14:30 <= label < 22:00, Mon-Fri
              = 15:30-23:00 Europe/Sofia, the owner's own clock, on every day of
                the year, because Sofia is Berlin + 1 h under one EU DST rule
    eligible  a trade at horizon h needs its as-of bar AND its target bar inside
              that window on the same label date. Restricting only the as-of bar
              leaves the last session hour carrying a 3 pp bid-versus-mid gap,
              because its target lands in the widening-spread hour (design D17).

THE PRICE. `results/eurusd_m15.csv` is byte-pinned and carries no spread, so it
cannot support a sub-hourly study: with no spread there is no mid price and no
honest cost. Bars are therefore aggregated from the M1 parquet, which carries
`spread_points` per bar, and every target is built on the MID price. It is never
read for targets and never written.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .costs import MissingSpreadError

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M1_PATH = os.path.join(REPO, 'results', 'curl', 'raw', 'EURUSD_M1.parquet')
CACHE_DIR = os.path.join(REPO, 'research_models', 'horizon_study', 'm15')

BAR_MINUTES = 15
BARS_PER_SESSION = 30                      # 7.5 h / 15 min

LABEL_TZ = 'Europe/Berlin'                 # established by verify_clock, not assumed
OWNER_TZ = 'Europe/Sofia'
MARKET_OPEN_TZ = 'America/New_York'
MARKET_OPEN_HOUR = 17                      # the forex week opens Sunday 17:00 New York

SESSION_START_MIN = 14 * 60 + 30           # 15:30 Europe/Sofia
SESSION_END_MIN = 22 * 60                  # 23:00 Europe/Sofia
MIN_WEEKEND_GAP_H = 10
CLOCK_MATCH_MIN = 0.90


class PyarrowMissing(RuntimeError):
    pass


class ClockMismatch(RuntimeError):
    pass


class SourceChanged(RuntimeError):
    pass


# ── the clock, established from the weekly market boundary ──────────────────

def weekly_opens(index, min_gap_h: int = MIN_WEEKEND_GAP_H) -> pd.DatetimeIndex:
    """The first bar after each weekend gap."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return idx[:0]
    gap = np.diff(idx.asi8) / 1e9 / 3600.0
    return idx[1:][gap > min_gap_h]


def expected_open_label(day, tz: str = LABEL_TZ) -> int:
    """The hour the weekly open carries when the labels are `tz` wall clock.

    The market opens Sunday `MARKET_OPEN_HOUR` New York; that instant, read on
    `tz`'s clock, is the label a bar would be stamped with."""
    ny = datetime(day.year, day.month, day.day, MARKET_OPEN_HOUR, tzinfo=ZoneInfo(MARKET_OPEN_TZ))
    return ny.astimezone(ZoneInfo(tz)).hour


def dst_aligned(day) -> bool:
    """True when New York and the label zone keep their usual 6-hour distance.
    They do not in the March and late-October daylight-saving mismatch weeks."""
    t = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
    return (t.astimezone(ZoneInfo(LABEL_TZ)).utcoffset()
            - t.astimezone(ZoneInfo(MARKET_OPEN_TZ)).utcoffset()) == timedelta(hours=6)


def clock_evidence(index, tz: str = LABEL_TZ) -> dict:
    """Match every weekly open label in the frame against what `tz` predicts,
    reported separately for the aligned weeks and the DST-mismatch weeks."""
    opens = weekly_opens(index)
    rows = []
    for ts in opens:
        day = ts.date()
        rows.append({'label_hour': int(ts.hour), 'expected_hour': expected_open_label(day, tz),
                     'aligned': dst_aligned(day)})
    d = pd.DataFrame(rows)
    if d.empty:
        return {'label_tz': tz, 'n_weeks': 0, 'match_rate': float('nan')}
    d['match'] = d['label_hour'] == d['expected_hour']
    out = {'label_tz': tz, 'n_weeks': int(len(d)), 'match_rate': float(d['match'].mean()),
           'market_open': f'Sunday {MARKET_OPEN_HOUR}:00 {MARKET_OPEN_TZ}'}
    for key, sel in (('aligned', d['aligned']), ('dst_mismatch', ~d['aligned'])):
        part = d[sel]
        out[key] = {'n': int(len(part)),
                    'match_rate': float(part['match'].mean()) if len(part) else float('nan'),
                    'modal_label_hour': int(part['label_hour'].mode().iloc[0]) if len(part) else None,
                    'expected_hour': int(part['expected_hour'].mode().iloc[0]) if len(part) else None}
    return out


def verify_clock(index, tz: str = LABEL_TZ, min_match: float = CLOCK_MATCH_MIN) -> dict:
    """Refuse the frame unless its weekly open labels match `tz`. A broker that
    changed its server clock must stop the run, never silently shift every
    session rule by an hour."""
    ev = clock_evidence(index, tz)
    if not ev['n_weeks']:
        raise ClockMismatch('no weekend boundary in the frame: the label clock cannot be established')
    if not (ev['match_rate'] >= min_match):
        raise ClockMismatch(
            f'bar labels do not match {tz}: only {ev["match_rate"]:.1%} of {ev["n_weeks"]} weekly '
            f'opens land on the predicted hour (aligned weeks modal '
            f'{ev["aligned"]["modal_label_hour"]}, expected {ev["aligned"]["expected_hour"]}). '
            'Establish the clock before any session rule is applied.')
    return ev


# ── the session, in label time ──────────────────────────────────────────────

def minute_of_day(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return idx.hour.to_numpy() * 60 + idx.minute.to_numpy()


def session_mask(index) -> np.ndarray:
    """Bars the owner is at the screen for: 14:30 <= label < 22:00, Mon-Fri."""
    idx = pd.DatetimeIndex(index)
    m = minute_of_day(idx)
    return ((m >= SESSION_START_MIN) & (m < SESSION_END_MIN)
            & (idx.dayofweek.to_numpy() < 5))


def eligible_mask(index, h: int) -> np.ndarray:
    """As-of bars whose trade at horizon h both starts and ENDS in the session,
    on the same label date."""
    if int(h) < 1:
        raise ValueError('h must be >= 1')
    h = int(h)
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    ok = session_mask(idx)
    ok[max(0, n - h):] = False
    tgt = np.arange(n) + h
    inside = np.zeros(n, dtype=bool)
    live = np.flatnonzero(ok)
    if live.size:
        t = tgt[live]
        m = minute_of_day(idx)
        same_day = idx[t].normalize().asi8 == idx[live].normalize().asi8
        inside[live] = ((m[t] + BAR_MINUTES) <= SESSION_END_MIN) & same_day
    return ok & inside


def eligible_positions(index, h: int) -> np.ndarray:
    return np.flatnonzero(eligible_mask(index, h))


def session_position(index) -> np.ndarray:
    """Where the bar sits in the session, 0.0 at the open and 1.0 at the close.
    NaN outside the session. A feature, never a filter on the target."""
    m = minute_of_day(index).astype(float)
    span = SESSION_END_MIN - SESSION_START_MIN
    out = (m - SESSION_START_MIN) / span
    out[~session_mask(index)] = np.nan
    return out


def owner_clock(index) -> pd.DatetimeIndex:
    """The labels on the owner's clock, for reports only. Sofia is Berlin + 1 h
    all year, so this is a fixed shift and never a DST conversion."""
    idx = pd.DatetimeIndex(index)
    off = (datetime(2026, 1, 1, 12, tzinfo=ZoneInfo(OWNER_TZ)).utcoffset()
           - datetime(2026, 1, 1, 12, tzinfo=ZoneInfo(LABEL_TZ)).utcoffset())
    return idx + off


def label_now(now_utc=None) -> pd.Timestamp:
    """Real time expressed on the BAR-LABEL clock: the label a bar closing right
    now would carry.

    Needed because everything else here reads bar labels, which are Europe/Berlin
    wall clock stored as if UTC. Feeding a true UTC "now" straight into
    `session_mask` is wrong by the Berlin offset -- two hours in summer -- which
    would open and close the session at the wrong moment."""
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz='UTC')
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    berlin = now.tz_convert(LABEL_TZ)
    return pd.Timestamp(berlin.replace(tzinfo=None)).tz_localize('UTC')


# ── the source file ─────────────────────────────────────────────────────────

def source_fingerprint(path: str = M1_PATH) -> dict:
    """Enough to prove a regenerated copy is the same file. The parquet itself is
    excluded from git (DATA.md section 7), so this is its only record."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'M1 source not found: {path}. Regenerate it with '
            '`pip install pyarrow && python -m src.curl_mt5_fetch` on a machine with MT5.')
    sha = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            sha.update(chunk)
    m1 = load_m1(path, columns=['close'])
    return {'path': os.path.relpath(path, REPO).replace('\\', '/'), 'sha256': sha.hexdigest(),
            'rows': int(len(m1)), 'first': str(m1.index[0]), 'last': str(m1.index[-1]),
            'bytes': int(os.path.getsize(path))}


def verify_source(recorded: dict, path: str = M1_PATH) -> dict:
    """Raise unless the file on disk is the one the record was built from."""
    fp = source_fingerprint(path)
    diffs = [k for k in ('sha256', 'rows') if recorded.get(k) != fp[k]]
    if diffs:
        raise SourceChanged(
            f'{fp["path"]} differs from the study record on {diffs}: '
            f'recorded rows={recorded.get("rows")} sha256={str(recorded.get("sha256"))[:12]}, '
            f'on disk rows={fp["rows"]} sha256={fp["sha256"][:12]}')
    return fp


# ── M1 -> M15 on the mid price ──────────────────────────────────────────────

def load_m1(path: str = M1_PATH, columns=None) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise PyarrowMissing(
            f'reading {os.path.basename(path)} needs pyarrow, which this project does not '
            f'declare as a dependency: install it with `pip install pyarrow` ({e})')
    df = pd.read_parquet(path, columns=columns)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize('UTC') if idx.tz is None else idx
    return df.sort_index()


def aggregate_m15(m1: pd.DataFrame) -> pd.DataFrame:
    """15-minute bars with bid OHLC, the MID close, summed tick volume and the
    median spread of the minutes in the bar. Bars are stamped at their open, as
    MT5 stamps them."""
    missing = {'spread_points', 'point'} - set(m1.columns)
    if missing:
        raise MissingSpreadError(
            f'M1 source has no {sorted(missing)}: refusing to build sub-hourly bars without a '
            'per-bar spread, because a mid price and an honest cost cannot be derived without it')
    point = float(np.nanmedian(m1['point'].to_numpy(dtype=float)))
    df = m1.assign(spread_price=m1['spread_points'].astype(float) * point)
    df['mid'] = df['close'].astype(float) + df['spread_price'] / 2.0
    out = df.resample(f'{BAR_MINUTES}min').agg(
        open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
        close=('close', 'last'), mid=('mid', 'last'),
        tick_volume=('tick_volume', 'sum'), spread_price=('spread_price', 'median'),
    ).dropna(subset=['close'])
    out['point'] = point
    return out


def m15_bars(path: str = M1_PATH, cache: bool = True, verify: bool = True) -> pd.DataFrame:
    """Aggregated M15 mid bars, cached under the gitignored research directory
    because re-reading 3 million M1 rows for every cell is pure waste."""
    cache_path = os.path.join(CACHE_DIR, f'm15_{os.path.basename(path)}')
    if cache and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(path):
        out = load_m1(cache_path)
    else:
        out = aggregate_m15(load_m1(path))
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            out.to_parquet(cache_path)
    if verify:
        verify_clock(out.index)
    return out


def from_mt5_bars(bars: pd.DataFrame, point: float) -> pd.DataFrame:
    """Live M15 bars in the aggregation's own shape, so the forward logger feeds
    the challengers exactly what the study fitted them on. MT5 quotes the spread
    in points; here it becomes a price, and the mid follows from it."""
    if 'spread' not in bars.columns:
        raise MissingSpreadError('live M15 bars carry no spread: refusing to build a mid price')
    point = float(point)
    if not np.isfinite(point) or point <= 0:
        raise ValueError(f'invalid point size {point!r}')
    out = bars[['open', 'high', 'low', 'close', 'tick_volume']].astype(float).copy()
    out['spread_price'] = bars['spread'].astype(float) * point
    out['mid'] = out['close'] + out['spread_price'] / 2.0
    out['point'] = point
    return out


def mid_from(close, spread_points, point: float) -> float:
    """The mid a forward row is scored on, from what the row recorded."""
    return float(close) + float(spread_points) * float(point) / 2.0


def session_summary(index) -> dict:
    """What the session rule actually selects -- reported beside every result."""
    idx = pd.DatetimeIndex(index)
    sess = session_mask(idx)
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    out = {'bars': int(len(idx)), 'first': str(idx[0]), 'last': str(idx[-1]),
           'span_years': round(years, 2), 'in_session': int(sess.sum()),
           'session_days': int(len(set(idx[sess].normalize().asi8))),
           'session_label_window': f'{SESSION_START_MIN // 60:02d}:{SESSION_START_MIN % 60:02d}'
                                   f'-{SESSION_END_MIN // 60:02d}:{SESSION_END_MIN % 60:02d}',
           'owner_window': '15:30-23:00 Europe/Sofia', 'label_tz': LABEL_TZ}
    return out
