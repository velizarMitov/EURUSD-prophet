"""External forward logger (design D6; spec "The logger forecasts the newest
closed bar every hour", "Settlement follows each instrument's own rules", "A
failure here costs nothing elsewhere").

Runs hourly at :01 (scheduled task EURUSDProphet-ExternalLogger), before the
gold pass at :03, so the gold chart export already sees this hour's external
forecasts. One pass:
  1. take the lock (one pass at a time);
  2. read each instrument's H1 bars live from MT5 -- gold through the gold
     family's reader, EUR/USD through the forward arbiter's -- and check each
     against its registered weekly-open clock. A pass that sees MT5 still
     forming the hour polls for up to 60 s. No live bars, no forecast: a cached
     bar may be hours old, and its target could already have closed;
  3. per instrument, forecast the NEWEST closed bar with each model (one worker
     call per model, both instruments in it), in the frozen configuration.
     Bars missed since the last pass are GAPS, never forecast late;
  4. compute the frozen EUR/USD baseline for the same bar;
  5. settle every matured row with the instrument's own targets and the
     market-break rule, copying in the baseline and the in-house forecast;
  6. write one health row.

Outputs (results/external_tsfm/forward/): predictions, settlements, gaps,
failures, health. Append-only. Every row carries `phase`: `dry_run` until
results/external_tsfm/PRE_REGISTRATION.md and .json are committed unmodified,
then `registered`. Gross of cost: no spread, no pips, no P&L.

    python -m src.external.tsfm.forward_logger      one pass
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import (ARTIFACTS, FORWARD, INSTRUMENTS, MODELS, PREREG_JSON, PREREG_MD, REPLAY, REPO,
               gpu_busy, repo_path)
from . import baseline as B
from . import client as CL
from . import data as D
from . import finetune as FT
from . import forecast as Q
from . import sources as S
from . import vendor as V

LIVE_BARS = 20000                 # > TimesFM's 16,256-bar context
LOCK_STALE_S = 30 * 60
POLL_S, POLL_TRIES = 10, 6        # up to 60 s for MT5 to close the hour
LIVE_CLOCK_WEEKS = 26
LIVE_CLOCK_MIN = 0.6
GOLD_BASELINE_WAIT_H = 6          # settle a gold row unpaired only if gold never logged its baseline
DEFAULT_CONFIG = 'zero_shot:returns:2048'

DECILES = [f'q{i}' for i in range(10, 100, 10)]
PRED_FIELDS = (['key', 'model', 'instrument', 'as_of_bar', 'as_of_close', 'config_id'] + DECILES +
               ['size_pct', 'p_up', 'baseline_size_pct', 'model_digest', 'phase', 'spans_weekend_gap',
                'logged_at'])
SETTLE_FIELDS = ['key', 'model', 'instrument', 'as_of_bar', 'exit_bar', 'exit_close', 'realised_ret_pct',
                 'realised_abs_pct', 'eligible', 'size_pct', 'baseline_size_pct', 'inhouse_size_pct',
                 'inhouse_p_up', 'phase', 'settled_at']
GAP_FIELDS = ['instrument', 'as_of_bar', 'reason', 'logged_at']
FAIL_FIELDS = ['model', 'instrument', 'as_of_bar', 'error', 'logged_at']
HEALTH_FIELDS = ['run_at', 'status', 'mt5', 'phase', 'forecasts', 'gaps', 'settled', 'failures', 'note']


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def out_path(name: str, repo: str = REPO) -> str:
    return repo_path(f'{FORWARD}/{name}.csv', repo)


def read_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def append_rows(path: str, rows, fields) -> None:
    """Append-only: the file is only ever opened in 'a' mode."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerows(rows)


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
            raise Locked('another external logger pass holds the lock') from e
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


# ── registration state ──────────────────────────────────────────────────────

def phase(repo: str = REPO) -> str:
    from src.forecast_eval import prereg as FP
    ok = all(FP.committed_and_clean(repo_path(p, repo), repo) is not None for p in (PREREG_MD, PREREG_JSON))
    return 'registered' if ok else 'dry_run'


def frozen_configs(repo: str = REPO) -> dict:
    """{'<model>_<instrument>': config_id}: the registration's, else the
    replay's choice (dry run), else zero-shot returns at 2,048."""
    reg = repo_path(PREREG_JSON, repo)
    if os.path.exists(reg):
        with open(reg, encoding='utf-8') as fh:
            return {f'{c["model"]}_{c["instrument"]}': c['config_id'] for c in json.load(fh)['cells']}
    rep = repo_path(f'{REPLAY}/replay.json', repo)
    out = {f'{m}_{i}': f'{m}:{DEFAULT_CONFIG}' for m in MODELS for i in INSTRUMENTS}
    if os.path.exists(rep):
        with open(rep, encoding='utf-8') as fh:
            for k, c in json.load(fh).get('choice', {}).items():
                if c.get('chosen'):
                    out[k] = c['chosen']
    return out


# ── live bars ───────────────────────────────────────────────────────────────

def label_now() -> pd.Timestamp:
    """The broker label clock's current hour reading: Europe/Berlin wall time,
    stamped UTC as every cache in the repo is (CLAUDE.md invariant 6)."""
    return pd.Timestamp.now(tz='Europe/Berlin').tz_localize(None).tz_localize('UTC')


def fetch_gold(mt5=None) -> pd.DataFrame | None:
    from src.gold import clock as K
    from src.gold import data as GD
    live = GD.fetch_live('H1', LIVE_BARS, mt5=mt5)
    if live is None:
        return None
    frame = live[0][D.OHLC]
    K.verify(frame.index)
    recent = frame[frame.index >= frame.index[-1] - pd.Timedelta(weeks=LIVE_CLOCK_WEEKS)]
    K.verify(recent.index, LIVE_CLOCK_MIN)
    return frame


def fetch_eurusd(reader=None, repo: str = REPO) -> pd.DataFrame | None:
    from src.forecast_eval import m15_data as MD
    from src.forecast_eval.mt5_reader import MT5Reader, MT5Unavailable, closed_bars, feed_now
    try:
        with (reader or MT5Reader()) as r:
            raw = r.bars(D.EURUSD_SYMBOL, 'H1', LIVE_BARS)
    except MT5Unavailable:
        return None
    now = feed_now(raw.index, state_path=repo_path(f'{ARTIFACTS}/h1_offset_state.json', repo))
    frame = D._utc(closed_bars(raw, 'H1', now)[D.OHLC])
    MD.verify_clock(frame.index[frame.index >= frame.index[-1] - pd.Timedelta(weeks=LIVE_CLOCK_WEEKS)],
                    min_match=LIVE_CLOCK_MIN)
    return frame


def fetch_polling(fetch, now_fn=label_now, sleep=time.sleep):
    """Wait up to 60 s for the hour that just ended, while the market trades
    (the newest bar is less than 3 hours old). Weekends are not polled."""
    frame = fetch()
    for _ in range(POLL_TRIES):
        if frame is None or frame.empty:
            return frame
        expected = now_fn().floor('h') - pd.Timedelta(hours=1)
        lag = now_fn() - frame.index[-1]
        if frame.index[-1] >= expected or lag >= pd.Timedelta(hours=3):
            return frame
        sleep(POLL_S)
        frame = fetch()
    return frame


# ── one pass ────────────────────────────────────────────────────────────────

def last_logged(instrument: str, repo: str = REPO):
    seen = [pd.Timestamp(r['as_of_bar']) for r in read_rows(out_path('predictions', repo))
            if r['instrument'] == instrument]
    seen += [pd.Timestamp(r['as_of_bar']) for r in read_rows(out_path('gaps', repo))
             if r['instrument'] == instrument]
    return max(seen) if seen else None


def _job(model, instrument, cid, frame, repo):
    """(worker job, model digest, input mode) after verifying every file it loads."""
    _, weights, mode, context = cid.split(':')
    context = int(context)
    digest = V.verify(model, repo=repo)
    adapter = None
    if weights == 'fine_tuned':
        digest += '+' + V.verify(FT.adapter_artifact(model, instrument), repo=repo)
        adapter = FT.adapter_dir(model, instrument, repo)
    series = D.model_series(frame['close'].to_numpy(), mode)[-context:]
    if mode == 'returns' and not np.isfinite(series[0]):
        series = series[1:]
    return {'key': instrument, 'adapter_dir': adapter, 'context': context,
            'series': [float(x) for x in series]}, digest, mode


def forecast_pass(frames: dict, ph: str, repo: str, call, now_iso: str):
    """(prediction rows, gap rows, failure rows, per-instrument status)."""
    configs = frozen_configs(repo)
    preds, gaps, fails, status, todo = [], [], [], {}, {}
    for inst, frame in frames.items():
        newest = frame.index[-1]
        prev = last_logged(inst, repo)
        if prev is not None and newest <= prev:
            status[inst] = 'nothing new'
            continue
        if prev is not None:
            gaps += [{'instrument': inst, 'as_of_bar': str(t), 'reason': 'bar closed while the logger was not running',
                      'logged_at': now_iso} for t in frame.index[(frame.index > prev) & (frame.index < newest)]]
        todo[inst] = newest
        status[inst] = f'forecast {newest}'
    base = {}
    if 'eurusd' in todo:
        try:
            base['eurusd'] = float(B.size_pct(B.load(repo), frames['eurusd'].iloc[-B.LOOKBACK:])[-1])
        except Exception as e:                             # noqa: BLE001 -- the models still log
            fails.append({'model': B.ARTIFACT, 'instrument': 'eurusd', 'as_of_bar': str(todo['eurusd']),
                          'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso})
    for model in MODELS:
        jobs, meta = [], {}
        for inst, newest in todo.items():
            cid = configs[f'{model}_{inst}']
            try:
                job, digest, mode = _job(model, inst, cid, frames[inst], repo)
            except Exception as e:                         # noqa: BLE001 -- recorded per model
                fails.append({'model': model, 'instrument': inst, 'as_of_bar': str(newest),
                              'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso})
                continue
            jobs.append(job)
            meta[inst] = (cid, digest, mode)
        if not jobs:
            continue
        r = call({'op': 'forecast', 'model': model, 'weights_dir': V.weights_dir(model, repo), 'jobs': jobs,
                  'device': 'cpu' if gpu_busy(repo) else 'auto'})
        if not r.get('ok'):
            fails += [{'model': model, 'instrument': inst, 'as_of_bar': str(todo[inst]), 'error': r.get('error'),
                       'logged_at': now_iso} for inst in meta]
            continue
        for inst, (cid, digest, mode) in meta.items():
            frame, newest = frames[inst], todo[inst]
            close = float(frame['close'].iloc[-1])
            dec = Q.to_return_deciles(r['deciles'][inst], mode, close)
            s = Q.summarise(dec)
            prev_gap = len(frame) > 1 and (newest - frame.index[-2]) > pd.Timedelta(hours=1)
            preds.append({'key': f'{model}|{inst}|{newest}', 'model': model, 'instrument': inst,
                          'as_of_bar': str(newest), 'as_of_close': close, 'config_id': cid,
                          **{q: s[q] for q in DECILES}, 'size_pct': s['size_pct'], 'p_up': s['p_up'],
                          'baseline_size_pct': base.get(inst, ''), 'model_digest': digest, 'phase': ph,
                          'spans_weekend_gap': bool(prev_gap), 'logged_at': now_iso})
    return preds, gaps, fails, status


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else ''
    except (TypeError, ValueError):
        return ''


def settle(frames: dict, repo: str = REPO, now_iso: str | None = None, now=None) -> list[dict]:
    """Settle every unsettled prediction whose next bar exists in this pass's
    bars, with the instrument's own targets and the market-break rule. A gold
    row waits for the gold logger's baseline for the same as-of bar (up to 6 h)."""
    from src.forecast_eval import targets as T
    from src.gold import targets as G
    now_iso = now_iso or _now()
    now = now or pd.Timestamp.now(tz='UTC')
    done = {r['key'] for r in read_rows(out_path('settlements', repo))}
    gold_base = S.gold_rows('h1_garch_hod', repo)
    gold_lstm = S.gold_rows('h1_lstm', repo)
    eur_gbm = S.eurusd_h1_gbm(repo)
    cache, out = {}, []
    for r in read_rows(out_path('predictions', repo)):
        inst = r['instrument']
        if r['key'] in done or inst not in frames:
            continue
        frame = frames[inst]
        if inst not in cache:
            close = frame['close']
            if inst == 'gold':
                tg = G.build_targets(close)
                ret, vol = tg['ret'], tg['vol']
            else:
                ret = T.return_pct_target(close, 1).to_numpy(float)
                vol = T.volatility_pct_target(close, 1).to_numpy(float)
            cache[inst] = (ret, vol, D.eligible(frame.index), {t: i for i, t in enumerate(frame.index)})
        ret, vol, elig, pos = cache[inst]
        i = pos.get(pd.Timestamp(r['as_of_bar']))
        if i is None or i + 1 >= len(frame):
            continue
        a = r['as_of_bar']
        if inst == 'gold':
            base = _num(gold_base.get(a, {}).get('vol_pct'))
            logged = pd.Timestamp(r['logged_at'])
            if base == '' and now - logged < pd.Timedelta(hours=GOLD_BASELINE_WAIT_H):
                continue
            inhouse, inhouse_p = _num(gold_lstm.get(a, {}).get('vol_pct')), _num(gold_lstm.get(a, {}).get('p_up'))
        else:
            base = _num(r.get('baseline_size_pct'))
            inhouse, inhouse_p = '', _num(eur_gbm.get(a, {}).get('p_up'))
        out.append({'key': r['key'], 'model': r['model'], 'instrument': inst, 'as_of_bar': a,
                    'exit_bar': str(frame.index[i + 1]), 'exit_close': float(frame['close'].iloc[i + 1]),
                    'realised_ret_pct': float(ret[i]), 'realised_abs_pct': float(vol[i]),
                    'eligible': bool(elig[i]), 'size_pct': _num(r['size_pct']), 'baseline_size_pct': base,
                    'inhouse_size_pct': inhouse, 'inhouse_p_up': inhouse_p, 'phase': r['phase'],
                    'settled_at': now_iso})
    return out


def run_once(repo: str = REPO, frames: dict | None = None, call=CL.call, ph: str | None = None) -> dict:
    """One pass. `frames` injects bars (tests); otherwise MT5 is read live."""
    now_iso = _now()
    status = {'run_at': now_iso}
    with RunLock(repo_path(f'{ARTIFACTS}/forward.lock', repo)):
        fails = []
        if frames is None:
            frames = {}
            for inst, fetch in (('gold', fetch_gold), ('eurusd', lambda: fetch_eurusd(repo=repo))):
                try:
                    f = fetch_polling(fetch)
                except Exception as e:                     # noqa: BLE001 -- clock, coverage, terminal
                    fails.append({'model': '*', 'instrument': inst, 'as_of_bar': '',
                                  'error': f'{type(e).__name__}: {e}', 'logged_at': now_iso})
                    continue
                if f is None or f.empty:
                    fails.append({'model': '*', 'instrument': inst, 'as_of_bar': '',
                                  'error': 'MT5 unreachable: nothing forecast', 'logged_at': now_iso})
                    continue
                frames[inst] = f
        ph = ph or phase(repo)
        status.update(phase=ph, mt5=bool(frames))
        if not CL.available(repo) and call is CL.call:
            preds, gaps, pfails, st = [], [], [], {i: 'external environment unavailable' for i in frames}
            status['note'] = 'external environment unavailable'
        else:
            preds, gaps, pfails, st = forecast_pass(frames, ph, repo, call, now_iso)
        append_rows(out_path('predictions', repo), preds, PRED_FIELDS)
        append_rows(out_path('gaps', repo), gaps, GAP_FIELDS)
        settled = settle(frames, repo, now_iso)
        append_rows(out_path('settlements', repo), settled, SETTLE_FIELDS)
        fails += pfails
        append_rows(out_path('failures', repo), fails, FAIL_FIELDS)
        status.update(instruments=st, forecasts=len(preds), gaps=len(gaps), settled=len(settled),
                      failures=len(fails))
    return status


def health_row(status: dict) -> dict:
    if status.get('error'):
        state = 'failed'
    elif not status.get('mt5'):
        state = 'mt5 unreachable'
    elif status.get('note') == 'external environment unavailable':
        state = 'external environment unavailable'
    elif status.get('failures'):
        state = 'partial'
    else:
        state = 'ok'
    return {'run_at': status.get('run_at', _now()), 'status': state, 'mt5': status.get('mt5', ''),
            'phase': status.get('phase', ''), 'forecasts': status.get('forecasts', 0),
            'gaps': status.get('gaps', 0), 'settled': status.get('settled', 0),
            'failures': status.get('failures', 0),
            'note': json.dumps({k: v for k, v in status.items() if k in ('instruments', 'error', 'note')},
                               sort_keys=True)}


def run_with_health(repo: str = REPO, **kw) -> dict:
    """One pass and exactly one health row, whatever happens."""
    try:
        status = run_once(repo, **kw)
    except Locked as e:
        return {'status': 'skipped', 'reason': str(e)}
    except Exception as e:                                 # noqa: BLE001 -- the health row records it
        status = {'run_at': _now(), 'error': f'{type(e).__name__}: {e}'}
    append_rows(out_path('health', repo), [health_row(status)], HEALTH_FIELDS)
    return status


def main(argv=None):
    argparse.ArgumentParser(description='External forward logger: one pass.').parse_args(argv)
    print(json.dumps(run_with_health(), default=str))


if __name__ == '__main__':
    main()
