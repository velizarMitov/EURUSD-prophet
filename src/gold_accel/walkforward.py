"""Walk-forward direction predictions (design D3/D4, spec "Predictions are
walk-forward and complete").

For each model and prediction year Y, a fresh challenger is built with
src.gold.challengers.make(base, gold_cfg, {'columns': cols}), fit on rows whose
bar AND next-bar target are labelled before Y-01-01, and predicts every row
labelled in Y. Inputs are built once over the pinned caches: features read bars
<= t only. A neural year exists only as the full five-seed ensemble; a failed
year is recorded and never filled in.

The H1 and M15 caches are pinned as gzip copies at the first run, so a resumed
or repeated run reads the same rows. Writes only under research_models/gold/accel/
and results/gold/accel/.

    python -m src.gold_accel.walkforward [--models h1_gbm,h1_accel]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.gold_window import bars as WB
from src.gold_window.walkforward import _gpu_lock, year_positions

from . import load_config, repo_path
from .features import build_inputs, columns_for, feature_frame


def _log(msg: str) -> None:
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def _guard(path: str, cfg: dict) -> str:
    roots = [os.path.normcase(os.path.abspath(repo_path(cfg['paths'][k]))) for k in ('models_dir', 'results')]
    p = os.path.normcase(os.path.abspath(path))
    if not any(os.path.commonpath([r, p]) == r for r in roots):
        raise RuntimeError(f'gold_accel writes only under {roots}, not {path}')
    return path


def _rel(path: str) -> str:
    return os.path.relpath(path, repo_path('.')).replace('\\', '/')


def _read_h1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.set_index('time').sort_index()[['open', 'high', 'low', 'close']].astype(float)


def pin_inputs(cfg: dict) -> dict:
    """H1 and M15 frames from the pinned copies; pinned from the live caches the
    first time (digests in the manifest)."""
    inputs_dir = repo_path(cfg['paths']['inputs_dir'])
    manifest = _guard(repo_path(cfg['paths']['artifact_manifest']), cfg)
    out = {}
    for key, src in (('h1', 'h1'), ('m15', 'm15')):
        pinned = _guard(os.path.join(inputs_dir, f'{key}.csv.gz'), cfg)
        if not os.path.exists(pinned):
            df = _read_h1(repo_path(cfg['paths'][src]))
            digest = WB.save(df.reset_index(), pinned)
            WB.update_manifest(manifest, _rel(pinned), digest, len(df), df.index.min(), df.index.max())
            _log(f'pinned {key}: {len(df):,} rows {df.index.min()} .. {df.index.max()}')
        out[key] = _read_h1(pinned)
    return out


def window_lstm_reuse(cfg: dict) -> dict:
    """Design D3: the gold-window-strategy h1_lstm predictions are reused only when
    that run pinned the digest of the H1 input it read."""
    manifest = repo_path('results/gold/window/artifact_manifest.csv')
    rows = []
    if os.path.exists(manifest):
        with open(manifest, encoding='utf-8', newline='') as fh:
            rows = list(csv.DictReader(fh))
    if not any(r['path'].endswith('gold_h1.csv') or '/inputs/h1' in r['path'] for r in rows):
        return {'reused': False, 'reason': 'the gold-window-strategy run recorded no digest of its H1 input, '
                                           'so h1_lstm is refit here on the pinned input'}
    return {'reused': False, 'reason': 'input digest present but not compared; refit for a like-for-like pair'}


def model_years(model: str, index: pd.DatetimeIndex, cfg: dict) -> list[int]:
    first = cfg['models'][model]['first_year']
    floor = index[0].year + cfg['walkforward']['warmup_years']
    if first < floor:
        raise ValueError(f'{model}: first_year {first} is inside the {floor - index[0].year}-year warm-up')
    return list(range(first, index[-1].year + 1))


def run(cfg: dict | None = None, models: list | None = None, frames: dict | None = None, make=None,
        gold_cfg: dict | None = None, log=_log) -> dict:
    from src.gold import challengers as C
    from src.gold import load_config as gold_config
    from src.gold import targets as G
    cfg = cfg or load_config()
    make = make or C.make
    gold_cfg = gold_cfg or gold_config()
    frames = frames or pin_inputs(cfg)
    h1, m15 = frames['h1'], frames['m15']
    feats = feature_frame(h1, m15, cfg['features']['profile_days'])
    targets = G.build_targets(h1['close'])
    models_dir = repo_path(cfg['paths']['models_dir'])
    manifest = _guard(repo_path(cfg['paths']['artifact_manifest']), cfg)
    out = {}
    with _gpu_lock():
        for model in models or cfg['run_order']:
            spec = cfg['models'][model]
            cols = columns_for(model, cfg)
            inputs = build_inputs(h1, feats, cols)
            parts = []
            for year in model_years(model, h1.index, cfg):
                ydir = _guard(os.path.join(models_dir, model, str(year)), cfg)
                done = os.path.join(ydir, 'done.json')
                pred_file = os.path.join(ydir, 'predictions.csv.gz')
                if os.path.exists(done):
                    with open(done, encoding='utf-8') as fh:
                        status = json.load(fh)
                    if status['ok']:
                        parts.append(pd.read_csv(pred_file, parse_dates=['time']))
                    log(f'{model} {year}: resumed ({"ok" if status["ok"] else "no predictions"})')
                    continue
                fit_pos, pred_pos = year_positions(h1.index, year)
                t0 = time.time()
                status = {'model': model, 'base': spec['base'], 'year': year, 'columns': cols,
                          'fit_rows': int(len(fit_pos)),
                          'fit_last_label': str(h1.index[fit_pos[-1]]) if len(fit_pos) else '',
                          'pred_rows': int(len(pred_pos))}
                try:
                    m = make(spec['base'], gold_cfg, {'columns': cols})
                    m.fit(inputs, fit_pos, targets, seed=gold_cfg['seeds'][0])
                    res = m.predict(inputs)
                    frame = pd.DataFrame({'time': h1.index[pred_pos],
                                          'p_up': np.asarray(res['p_up'], float)[pred_pos]})
                    files = m.save(ydir)
                    WB.save(frame, pred_file)
                    files.append(pred_file)
                    for f in files:
                        WB.update_manifest(manifest, _rel(f), WB.sha256_file(f), 0, '', '')
                    parts.append(frame)
                    status['ok'] = True
                    status['config_hash'] = getattr(m, 'config_hash', None)
                except Exception as e:                       # recorded, never patched
                    status.update(ok=False, error=f'{type(e).__name__}: {e}')
                finally:
                    try:
                        import keras
                        keras.backend.clear_session()
                    except Exception:
                        pass
                status['seconds'] = round(time.time() - t0, 1)
                os.makedirs(ydir, exist_ok=True)
                with open(done, 'w', encoding='utf-8') as fh:
                    json.dump(status, fh, indent=1)
                log(f'{model} {year}: {"ok" if status["ok"] else "FAILED " + status["error"]} '
                    f'(fit {status["fit_rows"]} rows to {status["fit_last_label"]}, {status["seconds"]} s)')
            allp = (pd.concat(parts, ignore_index=True).sort_values('time') if parts
                    else pd.DataFrame(columns=['time', 'p_up']))
            path = _guard(os.path.join(models_dir, f'predictions_{model}.csv.gz'), cfg)
            WB.save(allp, path)
            WB.update_manifest(manifest, _rel(path), WB.sha256_file(path), len(allp),
                               allp['time'].min() if len(allp) else '', allp['time'].max() if len(allp) else '')
            out[model] = allp
    reuse = window_lstm_reuse(cfg)
    os.makedirs(models_dir, exist_ok=True)
    with open(_guard(os.path.join(models_dir, 'h1_lstm_reuse.json'), cfg), 'w', encoding='utf-8') as fh:
        json.dump(reuse, fh, indent=1)
    return out


def load_predictions(model: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(os.path.join(repo_path(cfg['paths']['models_dir']), f'predictions_{model}.csv.gz'))
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Walk-forward gold direction models with and without acceleration.')
    ap.add_argument('--models', default='', help='comma-separated subset, in run order by default')
    args = ap.parse_args(argv)
    res = run(models=[m for m in args.models.split(',') if m] or None)
    for k, v in res.items():
        print(k, len(v), v['time'].min() if len(v) else '-', '..', v['time'].max() if len(v) else '-')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
