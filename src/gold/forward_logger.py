"""Gold forward logger (design D10; spec gold-forecasting/forward-arbiter).

Runs hourly (scheduled task EURUSDProphet-GoldLogger, :03). One pass:
  1. take the lock (one pass at a time);
  2. refresh the OHLC caches from MT5 (append-only; falls back to the cache);
  3. verify the clock: the whole history at the registered 0.95 AND the last 26
     weeks at 0.6 (design D3 -- broker clocks demonstrably change);
  4. verify every frozen artifact against results/gold/artifact_manifest.csv; a
     cell whose digest differs logs a failure and no forecast, the rest go on;
  5. per cadence, forecast the NEWEST closed bar (its target cannot have closed:
     it is the newest bar there is). Every as-of bar since the last logged one
     that was not forecast gets a GAP record -- its target has already closed,
     so a forecast now would be written after its outcome existed;
  6. settle matured rows with the study's own targets and eligibility rule.

Outputs (results/gold/forward/): predictions, settlements, gaps, failures CSVs.
Every row carries `phase`: `dry_run` until results/gold/forward/PRE_REGISTRATION.md
is committed and unmodified, then `registered`. Dry-run rows are never scored.

Independent of the EUR/USD logger: own lock, own files, own manifest.
GROSS of cost: no spread, no pips, no P&L anywhere.

    python -m src.gold.forward_logger            one pass
    python -m src.gold.forward_logger --report   forward scoring report (task 9.2)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.forecast_eval import prereg as FP

from . import REPO, load_config, repo_path
from . import challengers as C
from . import clock as K
from . import data as D
from . import targets as G

LIVE_CLOCK_WEEKS = 26
LIVE_CLOCK_MIN = 0.6
LOCK_STALE_S = 30 * 60
FORWARD_CELLS = {'D1': ('d1_gbm', 'd1_lstm', 'd1_garch'), 'H1': ('h1_gbm', 'h1_lstm', 'h1_garch_hod')}

PRED_FIELDS = ['key', 'cell', 'cadence', 'as_of_bar', 'as_of_close', 'p_up', 'ret_pct', 'vol_pct',
               'implied_price', 'model_digest', 'phase', 'logged_at']
SETTLE_FIELDS = ['key', 'cell', 'cadence', 'as_of_bar', 'exit_bar', 'exit_close', 'realised_dir',
                 'realised_ret_pct', 'realised_vol_pct', 'eligible', 'phase', 'settled_at']
GAP_FIELDS = ['cadence', 'as_of_bar', 'reason', 'logged_at']
FAIL_FIELDS = ['cell', 'cadence', 'as_of_bar', 'error', 'logged_at']


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def out_path(cfg, name):
    return os.path.join(repo_path(cfg['paths']['forward']), f'{name}.csv')


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def append_rows(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerows(rows)


# ── lock ────────────────────────────────────────────────────────────────────

class Locked(RuntimeError):
    pass


class RunLock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
            os.remove(self.path)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise Locked('another gold logger pass holds the lock') from e
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


# ── checks ──────────────────────────────────────────────────────────────────

def phase(cfg) -> str:
    md = repo_path(cfg['paths']['forward_prereg_md'])
    return 'registered' if FP.committed_and_clean(md, REPO) is not None else 'dry_run'


def verify_live_clock(h1: pd.DataFrame, cfg) -> dict:
    full = K.verify(h1.index, cfg['clock']['weekly_open_threshold'])
    recent = h1[h1.index >= h1.index[-1] - pd.Timedelta(weeks=LIVE_CLOCK_WEEKS)]
    live = K.verify(recent.index, LIVE_CLOCK_MIN)
    return {'full': full, 'recent': live}


def read_manifest(cfg) -> dict:
    """{cell: [(path, sha256), ...]}"""
    out = {}
    for r in read_rows(repo_path(cfg['paths']['manifest'])):
        out.setdefault(r['cell'], []).append((r['path'], r['sha256']))
    return out


def cell_digest(entries) -> str:
    return hashlib.sha256('|'.join(sorted(f'{p}:{d}' for p, d in entries)).encode()).hexdigest()[:16]


def verify_cell(cell, manifest) -> str:
    """The cell's version digest, or ValueError naming the first mismatching file."""
    entries = manifest.get(cell)
    if not entries:
        raise ValueError(f'{cell} is not in the artifact manifest')
    for rel, digest in entries:
        full = repo_path(rel)
        if not os.path.exists(full):
            raise ValueError(f'{rel} missing')
        with open(full, 'rb') as fh:
            if hashlib.sha256(fh.read()).hexdigest() != digest:
                raise ValueError(f'{rel} differs from the manifest digest')
    return cell_digest(entries)


def load_cell(cell, cfg):
    m = C.make(cell, cfg)
    return m.load(os.path.join(repo_path(cfg['paths']['artifacts']), cell))


# ── one pass ────────────────────────────────────────────────────────────────

def last_logged(cfg, cadence):
    seen = [pd.Timestamp(r['as_of_bar']) for r in read_rows(out_path(cfg, 'predictions')) if r['cadence'] == cadence]
    seen += [pd.Timestamp(r['as_of_bar']) for r in read_rows(out_path(cfg, 'gaps')) if r['cadence'] == cadence]
    return max(seen) if seen else None


def forecast_cadence(cad, df, cfg, models, digests, ph, now_iso):
    """Predictions for the newest bar plus gap records for skipped ones."""
    newest = df.index[-1]
    prev = last_logged(cfg, cad)
    if prev is not None and newest <= prev:
        return [], [], [], 'nothing new'
    gaps = []
    if prev is not None:
        skipped = df.index[(df.index > prev) & (df.index < newest)]
        gaps = [{'cadence': cad, 'as_of_bar': str(t), 'reason': 'target bar closed before this pass',
                 'logged_at': now_iso} for t in skipped]
    preds, fails = [], []
    inp = C.build_inputs(df, cad) if models else None
    close = float(df['close'].iloc[-1])
    for cell, model in models.items():
        try:
            out = model.predict(inp)
            row = {k: (float(v[-1]) if np.isfinite(v[-1]) else '') for k, v in out.items()}
            ret = row.get('ret_pct', '')
            preds.append({'key': f'{cell}|{cad}|{newest}', 'cell': cell, 'cadence': cad, 'as_of_bar': str(newest),
                          'as_of_close': close, 'p_up': row.get('p_up', ''), 'ret_pct': ret,
                          'vol_pct': row.get('vol_pct', ''),
                          'implied_price': G.implied_price(close, ret) if ret != '' else '',
                          'model_digest': digests[cell], 'phase': ph, 'logged_at': now_iso})
        except Exception as e:
            fails.append({'cell': cell, 'cadence': cad, 'as_of_bar': str(newest),
                          'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso})
    return preds, gaps, fails, f'forecast {len(preds)} cell(s), {len(gaps)} gap(s)'


def settle(cfg, frames, now_iso, preds_name='predictions', setts_name='settlements', skip=None):
    """Settle every unsettled prediction whose next bar now exists. The round-two
    shadow log (predictions_v2 / settlements_v2) is settled by the same rule."""
    done = {r['key'] for r in read_rows(out_path(cfg, setts_name))}
    out = []
    cache = {}
    for r in read_rows(out_path(cfg, preds_name)):
        if r['key'] in done or (skip is not None and skip(r)):
            continue
        cad = r['cadence']
        df = frames[cad]
        if cad not in cache:
            cache[cad] = (G.build_targets(df['close']), G.eligible(df.index, cad), {t: i for i, t in enumerate(df.index)})
        tg, elig, pos = cache[cad]
        i = pos.get(pd.Timestamp(r['as_of_bar']))
        if i is None or i + 1 >= len(df):
            continue
        out.append({'key': r['key'], 'cell': r['cell'], 'cadence': cad, 'as_of_bar': r['as_of_bar'],
                    'exit_bar': str(df.index[i + 1]), 'exit_close': float(df['close'].iloc[i + 1]),
                    'realised_dir': '' if np.isnan(tg['dir'][i]) else int(tg['dir'][i]),
                    'realised_ret_pct': float(tg['ret'][i]), 'realised_vol_pct': float(tg['vol'][i]),
                    'eligible': bool(elig[i]), 'phase': r['phase'], 'settled_at': now_iso})
    return out


def run_once(cfg=None, frames=None, loader=load_cell, ph=None) -> dict:
    cfg = cfg or load_config()
    now_iso = _now()
    status, new_preds = {}, []
    with RunLock(os.path.join(repo_path(cfg['paths']['artifacts']), 'forward.lock')):
        live = {'D1': True, 'H1': True}
        if frames is None:
            frames = {}
            for cad in ('D1', 'H1'):
                frames[cad], rec = D.load(cad, cfg=cfg)
                live[cad] = rec['source'] == 'mt5'
        ph = ph or phase(cfg)
        try:
            verify_live_clock(frames['H1'], cfg)
        except K.GoldClockMismatch as e:
            append_rows(out_path(cfg, 'failures'), [{'cell': '*', 'cadence': '*', 'as_of_bar': str(frames['H1'].index[-1]),
                                                     'error': f'GoldClockMismatch: {e}', 'logged_at': now_iso}], FAIL_FIELDS)
            return {'status': 'clock mismatch'}
        manifest = read_manifest(cfg)
        for cad, cells in FORWARD_CELLS.items():
            if not live[cad]:
                # The cache's newest bar may be hours old in reality: its target could
                # already have closed, so forecasting from it would be writing after the
                # outcome existed. Missed bars become gaps on the next live pass.
                append_rows(out_path(cfg, 'failures'), [{'cell': '*', 'cadence': cad, 'as_of_bar': '',
                                                         'error': 'no live MT5 data: nothing forecast',
                                                         'logged_at': now_iso}], FAIL_FIELDS)
                status[cad] = 'no live data'
                continue
            models, digests, fails = {}, {}, []
            for cell in cells:
                try:
                    digests[cell] = verify_cell(cell, manifest)
                    models[cell] = loader(cell, cfg)
                except Exception as e:
                    fails.append({'cell': cell, 'cadence': cad, 'as_of_bar': str(frames[cad].index[-1]),
                                  'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso})
            preds, gaps, pfails, msg = forecast_cadence(cad, frames[cad], cfg, models, digests, ph, now_iso)
            if msg == 'nothing new':
                fails = []                       # a failing cell is reported once per bar, not once per pass
            append_rows(out_path(cfg, 'predictions'), preds, PRED_FIELDS)
            new_preds += preds
            append_rows(out_path(cfg, 'gaps'), gaps, GAP_FIELDS)
            append_rows(out_path(cfg, 'failures'), fails + pfails, FAIL_FIELDS)
            status[cad] = msg
            status['forecasts'] = status.get('forecasts', 0) + len(preds)
            status['gaps'] = status.get('gaps', 0) + len(gaps)
        settled = settle(cfg, frames, now_iso)
        append_rows(out_path(cfg, 'settlements'), settled, SETTLE_FIELDS)
        status['settled'] = len(settled)
        status['ranges'] = log_ranges(cfg, now_iso, new_preds)
        status['alerts'] = log_alerts(cfg, now_iso, new_preds)
        status['shadow'] = log_shadow(cfg, frames, live, ph, now_iso)
        status['v2'] = log_v2(cfg, frames, live, ph, now_iso)
        status['v2_settled'] = settle_v2(cfg, frames, now_iso)
        status['phase'] = ph
        status['mt5'] = live['H1'] or live['D1']
        status['newest_h1'] = str(frames['H1'].index[-1]) if len(frames['H1']) else ''
    return status


def log_ranges(cfg, now_iso, new_preds, cfg2=None) -> str:
    """Round-two price ranges (change gold-forecasting-v2, spec gold-forecasting/price-ranges),
    written to their own files so round one's schema never changes: a range row
    for each LSTM forecast made in THIS pass -- never for an older forecast, whose
    outcome may already exist -- and a settlement for every range whose forecast
    has settled. A failure here is recorded and never touches the
    round-one rows of this pass."""
    try:
        from . import prereg2 as P2
        from . import ranges as R
        cfg2 = cfg2 or P2.load_config_v2()
        preds = read_rows(out_path(cfg, 'predictions'))
        setts = {r['key']: r for r in read_rows(out_path(cfg, 'settlements'))}
        have = {(r['key'], str(r['level'])) for r in read_rows(out_path(cfg, 'ranges'))}
        rows = [r for r in R.forecast_rows(cfg, cfg2, new_preds, preds, setts, now_iso)
                if (r['key'], str(r['level'])) not in have]
        append_rows(out_path(cfg, 'ranges'), rows, R.RANGE_FIELDS)
        done = {(r['key'], str(r['level'])) for r in read_rows(out_path(cfg, 'range_settlements'))}
        settled = R.settle_rows(read_rows(out_path(cfg, 'ranges')), done, setts, now_iso)
        append_rows(out_path(cfg, 'range_settlements'), settled, R.RANGE_SETTLE_FIELDS)
        return f'{len(rows)} range(s), {len(settled)} settled'
    except Exception as e:  # noqa: BLE001 -- ranges must never cost the round-one pass
        append_rows(out_path(cfg, 'failures'), [{'cell': 'ranges', 'cadence': '*', 'as_of_bar': '',
                                                 'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                    FAIL_FIELDS)
        return f'failed: {type(e).__name__}'


V2_PRED_FIELDS = PRED_FIELDS + ['version', 'shadow']


def log_shadow(cfg, frames, live, ph, now_iso, cfg2=None, loader=None) -> str:
    """Forecast with the active shadow version (change gold-forecasting-v2, spec
    gold-forecasting/model-versions) beside the registered models, into
    predictions_v2.csv with its own digest and shadow=1, then settle that log.
    Shadow rows never enter predictions.csv, so no registered forward verdict
    can count them. Only live MT5 data is forecast, as for the registered cells."""
    try:
        from . import prereg2 as P2
        from . import versions as V
        cfg2 = cfg2 or P2.load_config_v2()
        vid = V.active(cfg2)
        if vid is None:
            return 'no shadow version'
        vcfg = V.version_cfg(cfg, cfg2, vid)
        manifest = read_manifest(vcfg)
        have = {r['key'] for r in read_rows(out_path(cfg, 'predictions_v2'))}
        loader = loader or load_cell
        rows = []
        for cad, cells in FORWARD_CELLS.items():
            if not live.get(cad, False) or not len(frames[cad]):
                continue
            df = frames[cad]
            newest = df.index[-1]
            todo = [c for c in cells if f'{c}|{cad}|{newest}|{vid}' not in have]
            if not todo:
                continue
            inp = C.build_inputs(df, cad)
            close = float(df['close'].iloc[-1])
            for cell in todo:
                digest = verify_cell(cell, manifest)
                out = loader(cell, vcfg).predict(inp)
                row = {k: (float(v[-1]) if np.isfinite(v[-1]) else '') for k, v in out.items()}
                ret = row.get('ret_pct', '')
                rows.append({'key': f'{cell}|{cad}|{newest}|{vid}', 'cell': cell, 'cadence': cad,
                             'as_of_bar': str(newest), 'as_of_close': close, 'p_up': row.get('p_up', ''),
                             'ret_pct': ret, 'vol_pct': row.get('vol_pct', ''),
                             'implied_price': G.implied_price(close, ret) if ret != '' else '',
                             'model_digest': digest, 'phase': ph, 'logged_at': now_iso,
                             'version': vid, 'shadow': 1})
        append_rows(out_path(cfg, 'predictions_v2'), rows, V2_PRED_FIELDS)
        return f'{vid}: {len(rows)} forecast(s)'
    except Exception as e:  # noqa: BLE001 -- the shadow must never cost the registered pass
        append_rows(out_path(cfg, 'failures'), [{'cell': 'shadow', 'cadence': '*', 'as_of_bar': '',
                                                 'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                    FAIL_FIELDS)
        return f'failed: {type(e).__name__}'


def log_v2(cfg, frames, live, ph, now_iso, cfg2=None, loader=None, macro_fn=None) -> str:
    """The round-two cells, logged for OBSERVATION only (change gold-forecasting-v2,
    task 6.2: all were DROP on validation; the owner chose to watch them live).
    Same rules as the registered cells: newest closed bar, live MT5 data only,
    digest-verified artifacts, one row per bar, into predictions_v2.csv with
    version 'v2' and shadow 0. A macro source that cannot supply a series skips
    only the macro cells and is recorded in failures.csv with the series named."""
    try:
        from . import models_v2 as M2
        from . import prereg2 as P2
        cfg2 = cfg2 or P2.load_config_v2()
        man = M2.manifest(cfg2)
        if not man:
            return 'no v2 models'
        loader = loader or (lambda cell: M2.load(cell, cfg, cfg2))
        have = {r['key'] for r in read_rows(out_path(cfg, 'predictions_v2'))}
        rows, notes = [], []
        for cad in ('D1', 'H1'):
            if not live.get(cad, False) or not len(frames[cad]):
                continue
            df = frames[cad]
            newest = df.index[-1]
            todo = [c for c, spec in M2.CELLS.items() if spec['cadence'] == cad
                    and c in man and f'{c}|{cad}|{newest}|{M2.VERSION}' not in have]
            if not todo:
                continue
            inp = C.build_inputs(df, cad)
            inp_macro = None
            if cad == 'D1' and any(M2.CELLS[c]['macro'] for c in todo):
                from . import macro as M
                try:
                    mac = (macro_fn or (lambda d: M2.live_macro(cfg2, d)))(df)
                    inp_macro = M.with_macro(inp, mac)[1]
                except M.MacroUnavailable as e:
                    append_rows(out_path(cfg, 'failures'), [{'cell': 'macro', 'cadence': 'D1', 'as_of_bar': str(newest),
                                                             'error': f'gap: {e}', 'logged_at': now_iso}], FAIL_FIELDS)
                    notes.append(f'macro gap: {e}')
                    todo = [c for c in todo if not M2.CELLS[c]['macro']]
            close = float(df['close'].iloc[-1])
            for cell in todo:
                try:
                    digest = verify_cell(cell, man)
                    model, meta = loader(cell)
                    row = M2.forecast(cell, model, meta, inp_macro if M2.CELLS[cell]['macro'] else inp)
                    ret = row.get('ret_pct', '')
                    rows.append({'key': f'{cell}|{cad}|{newest}|{M2.VERSION}', 'cell': cell, 'cadence': cad,
                                 'as_of_bar': str(newest), 'as_of_close': close, 'p_up': row.get('p_up', ''),
                                 'ret_pct': ret, 'vol_pct': row.get('vol_pct', ''),
                                 'implied_price': G.implied_price(close, ret) if ret != '' else '',
                                 'model_digest': digest, 'phase': ph, 'logged_at': now_iso,
                                 'version': M2.VERSION, 'shadow': 0})
                except Exception as e:  # noqa: BLE001 -- one cell's failure is recorded, the rest go on
                    append_rows(out_path(cfg, 'failures'), [{'cell': cell, 'cadence': cad, 'as_of_bar': str(newest),
                                                             'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                                FAIL_FIELDS)
        append_rows(out_path(cfg, 'predictions_v2'), rows, V2_PRED_FIELDS)
        return '; '.join([f'{len(rows)} forecast(s)'] + notes)
    except Exception as e:  # noqa: BLE001 -- round two must never cost the registered pass
        append_rows(out_path(cfg, 'failures'), [{'cell': 'v2', 'cadence': '*', 'as_of_bar': '',
                                                 'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                    FAIL_FIELDS)
        return f'failed: {type(e).__name__}'


def settle_v2(cfg, frames, now_iso, cfg2=None) -> int:
    """Settle predictions_v2.csv: barrier rows by their barrier label once the
    k-hour window has closed (a time-out settles as not eligible); every other
    row (round-two and shadow) by the next bar, as the registered cells."""
    try:
        from . import direction2 as D2
        from . import prereg2 as P2
        cfg2 = cfg2 or P2.load_config_v2()
        is_barrier = lambda r: r['cell'] == 'h1_barrier'
        out = settle(cfg, frames, now_iso, 'predictions_v2', 'settlements_v2', skip=is_barrier)
        done = {r['key'] for r in read_rows(out_path(cfg, 'settlements_v2'))}
        pending = [r for r in read_rows(out_path(cfg, 'predictions_v2')) if is_barrier(r) and r['key'] not in done]
        if pending and 'H1' in frames and len(frames['H1']):
            df = frames['H1']
            b = cfg2['study']['direction2']['barrier']
            k = b['k_hours']
            lab = D2.barrier_labels(df, k, b['m'], b['ewma_span'])
            pos = {t: i for i, t in enumerate(df.index)}
            tg = G.build_targets(df['close'])
            for r in pending:
                i = pos.get(pd.Timestamp(r['as_of_bar']))
                if i is None or i + k >= len(df):
                    continue
                label = lab['label'].iloc[i]
                out.append({'key': r['key'], 'cell': r['cell'], 'cadence': 'H1', 'as_of_bar': r['as_of_bar'],
                            'exit_bar': str(df.index[i + k]), 'exit_close': float(df['close'].iloc[i + k]),
                            'realised_dir': '' if np.isnan(label) else int(label),
                            'realised_ret_pct': float(tg['ret'][i]), 'realised_vol_pct': float(tg['vol'][i]),
                            'eligible': bool(lab['eligible'].iloc[i] and not np.isnan(label)),
                            'phase': r['phase'], 'settled_at': now_iso})
        append_rows(out_path(cfg, 'settlements_v2'), out, SETTLE_FIELDS)
        return len(out)
    except Exception as e:  # noqa: BLE001
        append_rows(out_path(cfg, 'failures'), [{'cell': 'v2-settle', 'cadence': '*', 'as_of_bar': '',
                                                 'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                    FAIL_FIELDS)
        return 0


def log_alerts(cfg, now_iso, new_preds, alert_cfg=None, notifier=None) -> str:
    """Large-move alerts (change gold-forecasting-v2, spec gold-forecasting/move-alerts):
    one row and one notification per new h1_lstm forecast at or above the
    calibrated threshold, then the realised move of every alert that settled. A
    failed notification is reported in the returned status (it lands in the
    health row) and never stops the row being written."""
    try:
        from . import alerts as A
        alert_cfg = alert_cfg if alert_cfg is not None else A.settings()
        notifier = notifier or A.notify
        existing = read_rows(out_path(cfg, 'alerts'))
        rows = A.check(new_preds, {r['key'] for r in existing}, alert_cfg, now_iso)
        notes = []
        for r in rows:
            ok, err = notifier(r['vol_pct'])
            r['notified'] = 'yes' if ok else 'failed'
            if not ok:
                notes.append(f'notification failed: {err}')
        append_rows(out_path(cfg, 'alerts'), rows, A.ALERT_FIELDS)
        setts = {r['key']: r for r in read_rows(out_path(cfg, 'settlements'))}
        done = {r['key'] for r in read_rows(out_path(cfg, 'alert_settlements'))}
        settled = A.settle(read_rows(out_path(cfg, 'alerts')), done, setts, now_iso)
        append_rows(out_path(cfg, 'alert_settlements'), settled, A.ALERT_SETTLE_FIELDS)
        if alert_cfg.get('threshold_vol_pct') is None:
            return 'not calibrated'
        return '; '.join([f'{len(rows)} alert(s), {len(settled)} settled'] + notes)
    except Exception as e:  # noqa: BLE001 -- alerts must never cost the round-one pass
        append_rows(out_path(cfg, 'failures'), [{'cell': 'alerts', 'cadence': 'H1', 'as_of_bar': '',
                                                 'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso}],
                    FAIL_FIELDS)
        return f'failed: {type(e).__name__}'


def run_with_health(cfg=None, runner=None) -> dict:
    """One pass, always followed by one health row (spec gold-forecasting/logger-health):
    a pass that finds the lock taken, the clock wrong, MT5 closed or a crash is still
    recorded, so a stall is visible instead of silent."""
    from . import health as HL
    cfg = cfg or load_config()
    runner = runner or (lambda: run_once(cfg))
    started = _now()
    row = {'started_at': started}
    try:
        st = runner()
    except Locked as e:
        row.update(outcome='locked', mt5='', note=str(e)[:200], ended_at=_now())
        HL.record(row, cfg)
        return {'status': str(e)}
    except Exception as e:
        row.update(outcome='error', mt5='', note=f'{type(e).__name__}: {e}'[:200], ended_at=_now())
        HL.record(row, cfg)
        raise
    if st.get('status') == 'clock mismatch':
        outcome = 'clock mismatch'
    elif not st.get('mt5', True):
        outcome = 'no live data'
    else:
        outcome = 'ok'
    row.update(outcome=outcome, mt5='yes' if st.get('mt5') else 'no', newest_h1=st.get('newest_h1', ''),
               forecasts=st.get('forecasts', 0), settled=st.get('settled', 0), gaps=st.get('gaps', 0),
               note=(f"D1: {st.get('D1', '')}; H1: {st.get('H1', '')}; ranges: {st.get('ranges', '')}; "
                     f"alerts: {st.get('alerts', '')}; shadow: {st.get('shadow', '')}; v2: {st.get('v2', '')}")[:300], ended_at=_now())
    HL.record(row, cfg)
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold forward logger')
    ap.add_argument('--report', action='store_true', help='forward scoring report (registered rows only)')
    args = ap.parse_args(argv)
    if args.report:
        from .forward_report import main as report_main
        return report_main()
    print(json.dumps(run_with_health()))


if __name__ == '__main__':
    main()
