"""Read-only payload for GET /api/gold and the /gold page (design D11; spec
gold-forecasting/gold-view).

Reads only recorded files: the forward logger's predictions, the family
registries, the one-shot test report, the forward verdicts and the GVZ caches
the logger keeps (change gold-gvz-morning-check). Runs no model,
fetches no market data, writes nothing -- and imports no model library, so
api.py can serve it without loading TensorFlow or XGBoost.

  * The newest logged row per cell is taken WHOLE: a blank field is never
    filled from an older row.
  * A row whose target bar has already closed on the bar-label clock is not
    shown as current; the cell says when its last forecast ended instead.
  * Every cell carries its verdicts as recorded and the label
    "research — no proven edge" until a forward KEEP.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import load_config, repo_path

LABEL_TZ = 'Europe/Berlin'
NO_EDGE = 'research — no proven edge'
KEEP_LABEL = 'statistical skill, gross of cost; not a trading result'
CELLS = [('h1_gbm', 'H1'), ('h1_lstm', 'H1'), ('h1_garch_hod', 'H1'),
         ('d1_gbm', 'D1'), ('d1_lstm', 'D1'), ('d1_garch', 'D1')]


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def label_clock_now(now=None) -> pd.Timestamp:
    """Now on the bar-label clock: Berlin wall clock, stored as UTC as-is."""
    now = now or datetime.now(ZoneInfo(LABEL_TZ))
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo(LABEL_TZ))
    wall = now.astimezone(ZoneInfo(LABEL_TZ)).replace(tzinfo=None)
    return pd.Timestamp(wall).tz_localize('UTC')


def target_window_end(as_of: pd.Timestamp, cadence: str) -> pd.Timestamp:
    """When the forecast's target bar closes, on the label clock."""
    if cadence == 'H1':
        return as_of + pd.Timedelta(hours=2)
    nxt = as_of + pd.offsets.BDay(1)
    return nxt + pd.Timedelta(days=1)


def verdicts(cfg) -> dict:
    """{cell: {family: verdict}} -- the last recorded row per cell and family."""
    out = {}
    for fam, path in cfg['paths']['registries'].items():
        for r in _rows(repo_path(path)):
            out.setdefault(r['cell'], {})[fam] = r['verdict']
    return out


def forward_verdicts(cfg) -> dict:
    path = os.path.join(repo_path(cfg['paths']['forward']), 'forward_verdicts.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return {r['cell']: r for r in json.load(fh).get('verdicts', [])}


def range_map(cfg) -> dict:
    """{prediction key: {level: {lo, hi}}} from the round-two ranges log
    (change gold-forecasting-v2); empty before the first range is logged."""
    out = {}
    for r in _rows(os.path.join(repo_path(cfg['paths']['forward']), 'ranges.csv')):
        lo, hi = _num(r.get('lo')), _num(r.get('hi'))
        if lo is not None and hi is not None:
            out.setdefault(r['key'], {})[str(r['level'])] = {'lo': lo, 'hi': hi}
    return out


def current_forecast(row: dict, cadence: str, now: pd.Timestamp) -> tuple[dict | None, str | None]:
    as_of = pd.Timestamp(row['as_of_bar'])
    end = target_window_end(as_of, cadence)
    if end <= now:
        return None, str(end)
    p = _num(row.get('p_up'))
    ret = _num(row.get('ret_pct'))
    vol = _num(row.get('vol_pct'))
    close = _num(row.get('as_of_close'))
    cur = {'as_of_bar': str(as_of), 'window_end': str(end), 'as_of_close': close,
           'p_up': p, 'direction': None, 'probability': None,
           'ret_pct': ret, 'implied_price': (round(close * float(np.exp(ret / 100.0)), 2)
                                             if close is not None and ret is not None else None),
           'vol_pct': vol, 'logged_at': row.get('logged_at'), 'phase': row.get('phase')}
    if p is not None:
        cur['direction'] = 'up' if p > 0.5 else 'down'
        cur['probability'] = p if p > 0.5 else 1.0 - p
    return cur, None


def gvz_block(h1_row: dict | None, lstm: dict | None, now_label: pd.Timestamp, cfg2=None) -> dict:
    """The morning-check block (change gold-gvz-morning-check): GVZ from the
    caches the logger keeps, the dollar move at the newest H1 close the logger
    recorded, and the h1_lstm move only while that forecast is current. Files
    only; any error reads as no data."""
    try:
        from . import gvz as GV
        close = _num(h1_row.get('as_of_close')) if h1_row else None
        bar = h1_row.get('as_of_bar') if h1_row else None
        now_utc = pd.Timestamp(now_label.tz_localize(None)).tz_localize(LABEL_TZ)
        out = GV.reading(cfg2 if cfg2 is not None else _cfg2(), now_utc, close, bar)
        cur = (lstm or {}).get('current')
        if out['state'] != 'none':
            out['lstm_vol_pct'] = cur.get('vol_pct') if cur else None
            out['lstm_as_of_bar'] = cur.get('as_of_bar') if cur else None
        return out
    except Exception:  # noqa: BLE001 -- the card is optional; the rest of the payload stands
        return {'state': 'none'}


def _cfg2():
    from .prereg2 import load_config_v2
    return load_config_v2()


def build(cfg: dict | None = None, now=None, cfg2=None) -> dict:
    cfg = cfg or load_config()
    now = label_clock_now(now) if not isinstance(now, pd.Timestamp) else now
    preds = _rows(os.path.join(repo_path(cfg['paths']['forward']), 'predictions.csv'))
    newest = {}
    for r in preds:                                  # file order is logging order
        newest[r['cell']] = r
    recorded, fwd = verdicts(cfg), forward_verdicts(cfg)
    ranges = range_map(cfg)
    cells = []
    for cell, cad in CELLS:
        fv = fwd.get(cell, {})
        keep = str(fv.get('verdict', '')).startswith('KEEP')
        entry = {'cell': cell, 'cadence': cad, 'verdicts': recorded.get(cell, {}),
                 'forward_verdict': fv.get('verdict'), 'label': KEEP_LABEL if keep else NO_EDGE,
                 'current': None, 'last_window_end': None}
        if cell in newest:
            entry['current'], entry['last_window_end'] = current_forecast(newest[cell], cad, now)
            if entry['current'] is not None:
                entry['current']['ranges'] = ranges.get(newest[cell]['key'])
        cells.append(entry)
    test = None
    tp = repo_path(cfg['paths']['test_report'])
    if os.path.exists(tp):
        with open(tp, encoding='utf-8') as fh:
            test = [{k: r.get(k) for k in ('family', 'cell', 'verdict', 'n_rows')}
                    for r in json.load(fh).get('results', [])]
    try:
        from .alerts import open_alert
        alert = open_alert(cfg, now)
    except Exception:  # noqa: BLE001 -- the alert banner is optional
        alert = None
    return {'generated_at': str(now), 'clock': f'{LABEL_TZ} bar labels', 'alert': alert,
            'gvz': gvz_block(newest.get('h1_lstm'), next((c for c in cells if c['cell'] == 'h1_lstm'), None), now,
                             cfg2),
            'state': 'forecasts' if preds else 'no_forecast',
            'cost_basis': 'gross of cost — no spread, pips or P&L',
            'label': NO_EDGE, 'cells': cells, 'test_report': test}
