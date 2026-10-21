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


def settle(cfg, frames, now_iso):
    """Settle every unsettled prediction whose next bar now exists."""
    done = {r['key'] for r in read_rows(out_path(cfg, 'settlements'))}
    out = []
    cache = {}
    for r in read_rows(out_path(cfg, 'predictions')):
        if r['key'] in done:
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
    status = {}
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
            append_rows(out_path(cfg, 'gaps'), gaps, GAP_FIELDS)
            append_rows(out_path(cfg, 'failures'), fails + pfails, FAIL_FIELDS)
            status[cad] = msg
        settled = settle(cfg, frames, now_iso)
        append_rows(out_path(cfg, 'settlements'), settled, SETTLE_FIELDS)
        status['settled'] = len(settled)
        status['phase'] = ph
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold forward logger')
    ap.add_argument('--report', action='store_true', help='forward scoring report (registered rows only)')
    args = ap.parse_args(argv)
    if args.report:
        from .forward_report import main as report_main
        return report_main()
    try:
        print(json.dumps(run_once()))
    except Locked as e:
        print(json.dumps({'status': str(e)}))


if __name__ == '__main__':
    main()
