"""News-hour calendars (design D6, spec gold-analytics/market-data "News hours come
from release calendars").

NFP (FRED release 50, Employment Situation) and CPI (FRED release 10) are at
08:30 America/New_York; FOMC statements, from results/fomc_dates.csv, at 14:00
America/New_York. Each release is placed in its own zone and converted to a
label, so the US/EU daylight-saving mismatch weeks land at 14:30 Sofia instead
of 15:30.

FRED is asked at most once a day; its answer is cached in events.csv. When FRED
is unreachable or the key is missing, the cache is used and its age reported.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import load_config, repo_path
from .clock import instant_to_label, ns, to_display
from .intraday import weighted_median

FRED_URL = 'https://api.stlouisfed.org/fred/release/dates'
MAX_CACHE_AGE_DAYS = 1.0


def fred_client(release_id: int) -> list[str]:
    """Release dates (YYYY-MM-DD) from the FRED API. Raises on any failure."""
    key = os.getenv('FRED_API_KEY')
    if not key or key == 'YOUR_FRED_API_KEY_HERE':
        raise RuntimeError('FRED_API_KEY is not set')
    q = urllib.parse.urlencode({'release_id': release_id, 'api_key': key, 'file_type': 'json',
                                'include_release_dates_with_no_data': 'true', 'sort_order': 'asc',
                                'limit': 10000})
    with urllib.request.urlopen(f'{FRED_URL}?{q}', timeout=15) as r:
        body = json.load(r)
    return [row['date'] for row in body.get('release_dates', [])]


def release_label(day: date, at: str, zone: str) -> pd.Timestamp:
    """The label of `at` wall time in `zone` on `day`."""
    h, m = (int(x) for x in at.split(':'))
    inst = datetime.combine(day, time(h, m), ZoneInfo(zone))
    return instant_to_label([pd.Timestamp(inst)])[0]


def _fomc_dates(cfg: dict) -> list[str]:
    df = pd.read_csv(repo_path(cfg['paths']['fomc']))
    return sorted(df['date'].astype(str).unique())


def _read_cache(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype=str)


def calendar(cfg: dict | None = None, client=None, now: datetime | None = None, force: bool = False) -> dict:
    """{'events': frame(event, date, label, sofia), 'fred': ok|cache|missing,
    'cache_age_days': float|None}. FOMC is always present."""
    cfg = cfg or load_config()
    client = client or fred_client
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize('UTC') if now_ts.tzinfo is None else now_ts.tz_convert('UTC')
    path = repo_path(cfg['paths']['events'])
    cache = _read_cache(path)
    age = None
    if cache is not None and len(cache):
        fetched = pd.Timestamp(cache['fetched_at'].max())
        age = (now_ts - fetched).total_seconds() / 86400
    fred_rows, status = None, 'missing'
    if cache is not None and age is not None and age < MAX_CACHE_AGE_DAYS and not force:
        fred_rows, status = cache, 'ok'
    else:
        try:
            rows = []
            for key, ev in cfg['events'].items():
                if ev['source'] == 'fred':
                    rows += [{'event': key, 'date': d, 'fetched_at': now_ts.isoformat()} for d in client(ev['release_id'])]
            fred_rows, status, age = pd.DataFrame(rows, columns=['event', 'date', 'fetched_at']), 'ok', 0.0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fred_rows.to_csv(path, index=False, lineterminator='\n')
        except Exception:  # noqa: BLE001 -- any FRED failure falls back to the cache
            if cache is not None and len(cache):
                fred_rows, status = cache, 'cache'
    rows = []
    for key, ev in cfg['events'].items():
        if ev['source'] == 'fred':
            dates = [] if fred_rows is None else fred_rows.loc[fred_rows['event'] == key, 'date'].tolist()
        else:
            dates = _fomc_dates(cfg)
        for d in sorted(set(dates)):
            day = date.fromisoformat(d)
            if day.weekday() >= 5:
                continue
            rows.append({'event': key, 'date': d, 'label': release_label(day, ev['time'], ev['zone'])})
    df = pd.DataFrame(rows, columns=['event', 'date', 'label'])
    if len(df):
        df['sofia'] = [t.strftime('%H:%M') for t in to_display(df['label'])]
    else:
        df['sofia'] = pd.Series(dtype=str)
    return {'events': df, 'fred': status, 'cache_age_days': None if age is None else round(age, 2)}


# -- news-hour windows (task 3.4) ------------------------------------------

def _window_moves(times: np.ndarray, opens: np.ndarray, closes: np.ndarray, a: np.ndarray, b: np.ndarray):
    """|close of the last bar / open of the first bar - 1| in %, for bars with
    labels in each [a, b); NaN where the window holds no bar."""
    ia = np.searchsorted(times, a, side='left')
    ib = np.searchsorted(times, b, side='left') - 1
    ok = (ib >= ia) & (ia < len(times))
    out = np.full(len(a), np.nan)
    out[ok] = np.abs(closes[ib[ok]] / opens[ia[ok]] - 1) * 100
    return out


def _window_spread(slots: np.ndarray, med: np.ndarray, ticks: np.ndarray, a, b) -> float | None:
    """Tick-weighted median spread over the 5-minute slots in all windows [a, b)."""
    vals, wts = [], []
    for x, y in zip(np.searchsorted(slots, a, side='left'), np.searchsorted(slots, b, side='left')):
        vals.append(med[x:y])
        wts.append(ticks[x:y])
    if not vals:
        return None
    return weighted_median(np.concatenate(vals), np.concatenate(wts))


def news_windows(bars: pd.DataFrame, releases: pd.DataFrame, cfg: dict, spread: pd.DataFrame | None = None) -> dict:
    """For one event's releases inside the bars' span: per 30-minute window from
    -2 h to +3 h around the release, the mean absolute move and median spread on
    release days, against the same label clock slots on the same weekday on the
    other days of the period."""
    w = cfg['event_window']
    offsets = list(range(-w['before_min'], w['after_min'], w['step_min']))
    out = {'n_releases': 0, 'offsets': [], 'first_hour': None}
    if bars.empty or releases.empty:
        return out
    times = ns(bars['time'])
    opens, closes = bars['open'].to_numpy(float), bars['close'].to_numpy(float)
    lo, hi = bars['time'].min(), bars['time'].max()
    rel = releases[(releases['label'] >= lo) & (releases['label'] <= hi)].reset_index(drop=True)
    if rel.empty:
        return out
    days = pd.DatetimeIndex(bars['time'].dt.normalize().unique())
    release_days = set(rel['label'].dt.normalize())
    others = [d for d in days if d not in release_days]
    if spread is not None and len(spread):
        sp = spread.sort_values('slot')
        sp_slots, sp_med, sp_ticks = ns(sp['slot']), sp['median'].to_numpy(float), sp['ticks'].to_numpy(float)
    else:
        sp_slots = None

    # Comparison anchors: for each release, the same clock time on the other same-weekday days.
    base_anchor = []
    for L in rel['label']:
        tod = L - L.normalize()
        base_anchor += [d + tod for d in others if d.weekday() == L.weekday()]
    rel_anchor = ns(rel['label'])
    base_anchor = ns(base_anchor) if base_anchor else np.array([], dtype=np.int64)
    minute = 60 * 10 ** 9

    def block(start_min, length_min):
        da, dl = start_min * minute, length_min * minute
        r = _window_moves(times, opens, closes, rel_anchor + da, rel_anchor + da + dl)
        bmv = _window_moves(times, opens, closes, base_anchor + da, base_anchor + da + dl) if len(base_anchor) \
            else np.array([])
        row = {'offset_min': start_min, 'length_min': length_min,
               'release_abs_mean': float(np.nanmean(r)) if np.isfinite(r).any() else None,
               'n_release': int(np.isfinite(r).sum()),
               'baseline_abs_mean': float(np.nanmean(bmv)) if len(bmv) and np.isfinite(bmv).any() else None,
               'n_baseline': int(np.isfinite(bmv).sum()) if len(bmv) else 0}
        row['ratio'] = (row['release_abs_mean'] / row['baseline_abs_mean']
                        if row['release_abs_mean'] is not None and row['baseline_abs_mean'] else None)
        if sp_slots is not None:
            row['release_spread_median'] = _window_spread(sp_slots, sp_med, sp_ticks, rel_anchor + da, rel_anchor + da + dl)
            row['baseline_spread_median'] = _window_spread(sp_slots, sp_med, sp_ticks, base_anchor + da,
                                                           base_anchor + da + dl) if len(base_anchor) else None
        return row

    out['n_releases'] = int(len(rel))
    out['n_comparison_days'] = int(len(set(base_anchor // (86400 * 10 ** 9)))) if len(base_anchor) else 0
    out['offsets'] = [block(o, w['step_min']) for o in offsets]
    out['first_hour'] = block(0, 60)
    out['release_sofia'] = rel['sofia'].value_counts().to_dict() if 'sofia' in rel else {}
    out['first_release'], out['last_release'] = rel['date'].iloc[0], rel['date'].iloc[-1]
    return out
