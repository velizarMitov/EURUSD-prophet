"""Read-only data behind the plain-language home page (openspec change
`friendly-home-page`, capability `home-overview`).

Everything the page at `/` shows is assembled here from logs that already
exist. Three properties are load-bearing:

  * NOTHING HERE WRITES OR PREDICTS. The page must be safe to open any number of
    times. That rules out `paper_trading.build_all_ledgers` (it persists one CSV
    per variant) and `PredictionService.predict` (it runs every model and logs a
    row). The reliability block uses `paper_trading.build_ledger`, which only
    reads, and the refresh button on the page is what calls the existing
    `POST /api/predict`.
  * THE WORDING RULES LIVE HERE, NOT IN THE BROWSER. Each block emits a stable
    key (`no_signal`, `too_early`, ...) that the page translates. The thresholds
    can therefore be tested in Python, and Bulgarian and English cannot disagree
    about what a number means.
  * NO COST FIGURE LEAVES THIS MODULE. "Right" means the direction was right.
    The paper-trading `win` is net of a spread, so it is a cost verdict, and it
    stays on the advanced pages (owner decision, 2026-10-02).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER_TZ = 'Europe/Sofia'
LOG_PATH = os.path.join(BASE_DIR, 'results', 'prediction_log.csv')

# Below this many settled forecasts a hit rate is noise. It matches the
# forward-eval view's MIN_SETTLED_TO_SHOW so the two pages never disagree on
# when a number starts to mean anything.
MIN_SETTLED = 30
WILSON_Z = 1.959963984540054          # two-sided 95 %


def _threshold() -> float:
    """The serving consensus guard, imported rather than re-typed so the home
    page can never call "slightly up" what the API calls MIXED."""
    from src.inference import PredictionService
    return float(PredictionService.CONFIDENCE_THRESHOLD)


def _now(now=None) -> datetime:
    if now is None:
        return datetime.now(ZoneInfo(OWNER_TZ))
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert(OWNER_TZ).to_pydatetime()


def _num(v):
    """A finite float or None. Blank CSV cells arrive as NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _latest_row(log_path: str):
    if not os.path.exists(log_path) or not os.path.getsize(log_path):
        return None
    log = pd.read_csv(log_path)
    if log.empty:
        return None
    return log.sort_values('as_of_date').iloc[-1]


# ── Tomorrow ────────────────────────────────────────────────────────────────

def word_for(direction, confidence, threshold: float) -> str:
    """The only three things the headline may say. There is no "strong": no
    model here has a proven daily edge, so certainty words would be invented."""
    conf = _num(confidence)
    d = str(direction or '').upper()
    if conf is None or conf < threshold or not (d.startswith('UP') or d.startswith('DOWN')):
        return 'no_signal'
    return 'slightly_up' if d.startswith('UP') else 'slightly_down'


def tomorrow_block(log_path: str = LOG_PATH, now=None) -> dict:
    row = _latest_row(log_path)
    if row is None:
        return {'available': False, 'reason': 'no_forecast_logged'}
    # The price-only variant drives the headline: the macro features are
    # KEEP-provisional with no proven edge. Pre-dual rows have no baseline_*
    # values, so fall back to the with-macro committee and say so.
    direction, confidence = row.get('baseline_direction'), row.get('baseline_confidence')
    fallback = _num(confidence) is None and pd.isna(direction)
    if fallback:
        direction, confidence = row.get('pred_direction'), row.get('pred_confidence')
    today = _now(now).date().isoformat()
    agreement = row.get('variant_agreement')
    return {
        'available': True,
        'key': word_for(direction, confidence, _threshold()),
        'forecasting_date': str(row['forecasting_date']),
        'as_of_date': str(row['as_of_date']),
        'logged_at': _owner_stamp(row.get('logged_at')),
        'outdated': str(row['forecasting_date']) < today,
        'variants_disagree': str(agreement).strip().lower() == 'false',
        'variant': 'with_macro' if fallback else 'baseline',
    }


def _owner_stamp(stamp):
    if stamp is None or pd.isna(stamp):
        return None
    try:
        ts = pd.Timestamp(stamp)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert(OWNER_TZ).strftime('%Y-%m-%d %H:%M')


# ── Expected movement ───────────────────────────────────────────────────────

def movement_block(log_path: str = LOG_PATH) -> dict:
    row = _latest_row(log_path)
    vol = None if row is None else _num(row.get('vol_pred_pct'))
    if vol is None:
        # Never 0: a missing forecast is not a forecast of no movement.
        return {'available': False, 'reason': 'no_volatility_logged'}
    return {'available': True, 'vol_pct': round(vol, 2),
            'forecasting_date': str(row['forecasting_date'])}


# ── Today's session ─────────────────────────────────────────────────────────

def next_open(now=None) -> str:
    """When the owner's 15:30 Sofia session next opens, ISO on the Sofia clock.
    Weekends roll to Monday. Bank holidays are not modelled; the logger simply
    finds no new bar on those days."""
    t = _now(now)
    day = t.date()
    if t.weekday() < 5 and (t.hour, t.minute) < (15, 30):
        candidate = day
    else:
        candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return f'{candidate.isoformat()}T15:30'


def session_block(now=None, out: str | None = None, record_path: str | None = None) -> dict:
    """The registered session cells' newest calls, grouped by how far ahead
    they look. Outside the session nothing directional is returned at all: a
    call from yesterday evening would read as today's."""
    from src.forecast_eval import m15_data as MD
    from src.forecast_eval import report as R

    out = out or R.FWD
    record_path = record_path or R.RECORD
    state = R.session_state(now)
    if not state['in_session']:
        return {'available': True, 'open': False, 'next_open': next_open(now)}

    admitted, _alpha, _n = R.admitted_cells(record_path)
    direction_models = R.direction_models(record_path)
    latest = R.latest_forecasts(R._read('predictions', out))
    label_now = MD.label_now(now)
    # Keyed by the WINDOW, not just the distance ahead: an M15 cell and an H1
    # cell both looking 60 minutes ahead start at different bar closes, and
    # one row showing a single window for both would misstate one of them.
    groups: dict[tuple, dict] = {}
    if not latest.empty:
        for _, r in latest.iterrows():
            key = (r['model'], r['cadence'], int(r['horizon']))
            if key not in admitted or r['model'] not in direction_models:
                continue
            if r['until'] <= label_now or r['direction'] not in ('+', '−'):
                continue                       # window over, or no call
            mins = R.minutes_ahead(r['cadence'], int(r['horizon']))
            start, end = R.owner_time(r['from']), R.owner_time(r['until'])
            g = groups.setdefault((mins, start, end), {
                'minutes': mins,
                'from': start.strftime('%H:%M'),
                'until': end.strftime('%H:%M'),
                'calls': []})
            g['calls'].append({'model': str(r['model']),
                               'key': 'up' if r['direction'] == '+' else 'down'})
    rows = [groups[k] for k in sorted(groups)]
    return {'available': True, 'open': True, 'horizons': rows}


# ── Reliability ─────────────────────────────────────────────────────────────

def wilson_lower(hits: int, n: int, z: float = WILSON_Z) -> float | None:
    if n <= 0:
        return None
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def reliability_state(hits: int, n: int) -> str:
    """Descriptive orientation only. This is not a pre-registered test and it
    never decides anything: "above_coin" still reads "not yet proven"."""
    if n < MIN_SETTLED:
        return 'too_early'
    lo = wilson_lower(hits, n)
    return 'above_coin' if lo is not None and lo > 0.5 else 'coin_range'


def _config() -> dict:
    with open(os.path.join(BASE_DIR, 'config.json'), encoding='utf-8') as f:
        return json.load(f)


def reliability_block(log_path: str = LOG_PATH, ledger_builder=None, data_cfg=None) -> dict:
    """Settled daily price-only calls: how many were in the right direction.

    A day the committee called MIXED took no position and is not counted, so N
    is the number of real calls, not the number of days."""
    if ledger_builder is None:
        from src.paper_trading import build_ledger as ledger_builder
    if data_cfg is None:
        data_cfg = _config()['data']
    ledger = ledger_builder(log_path, data_cfg, direction_column='baseline_direction')
    taken = ledger[ledger['direction'] != 'FLAT'] if len(ledger) else ledger
    n = int(len(taken))
    # Direction right, before any cost. gross_pips is already signed by the
    # call, so a correct call is strictly positive.
    hits = int((pd.to_numeric(taken['gross_pips'], errors='coerce') > 0).sum()) if n else 0
    lo = wilson_lower(hits, n)
    return {
        'available': True,
        'hits': hits,
        'n': n,
        'rate': round(hits / n, 4) if n else None,
        'lower_95': None if lo is None else round(lo, 4),
        'min_settled': MIN_SETTLED,
        'key': reliability_state(hits, n),
    }


# ── Assembly ────────────────────────────────────────────────────────────────

def _guarded(fn, *args, **kwargs) -> dict:
    """One failing source degrades its own block and nothing else."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 -- a home tile must never 500 the page
        return {'available': False, 'reason': f'{type(e).__name__}: {e}'[:200]}


def build_home(part: str | None = None, now=None, log_path: str = LOG_PATH) -> dict:
    """The JSON behind `/api/home`. The reliability block needs realised prices
    (MT5 or Yahoo) and can be slow, so it is served only on its own call
    (`part="reliability"`) and the fast tiles never wait for it."""
    generated = _now(now).strftime('%Y-%m-%d %H:%M')
    if part == 'reliability':
        return {'generated_at': generated,
                'reliability': _guarded(reliability_block, log_path)}
    return {
        'generated_at': generated,
        'tomorrow': _guarded(tomorrow_block, log_path, now),
        'movement': _guarded(movement_block, log_path),
        'session': _guarded(session_block, now),
    }
