"""Walk-forward model predictions (spec window-strategy: "Model predictions are
walk-forward and complete"; design D2).

For each model and each prediction year Y, a fresh challenger from
src.gold.challengers is fit on rows whose bar AND next-bar target are labelled
before Y-01-01 (expanding window), then predicts every row labelled in Y.
Inputs are built once over the whole cache: features read bars <= t only.
The first `warmup_years` calendar years of each cache are fit-only.

A neural year exists only as the full five-seed ensemble: a failed or partial
fit leaves that year without predictions, never patched.

Writes only under research_models/gold/window/ and to
results/gold/window/artifact_manifest.csv.

    python -m src.gold_window.walkforward
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load_config, repo_path
from . import bars as B

CADENCE = {'h1_lstm': 'H1', 'd1_lstm': 'D1', 'd1_garch': 'D1'}
OUTPUTS = ('p_up', 'vol_pct')


def _log(msg: str) -> None:
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


def year_positions(index: pd.DatetimeIndex, year: int) -> tuple[np.ndarray, np.ndarray]:
    """(fit positions, prediction positions) for prediction year `year`.
    The last row before the year is dropped from the fit: its target is in `year`."""
    start = pd.Timestamp(f'{year}-01-01', tz=index.tz)
    end = pd.Timestamp(f'{year + 1}-01-01', tz=index.tz)
    n_before = int(np.searchsorted(index, start, side='left'))
    pred = np.flatnonzero((index >= start) & (index < end))
    return np.arange(max(0, n_before - 1)), pred


def prediction_years(index: pd.DatetimeIndex, warmup_years: int) -> list[int]:
    first, last = index[0].year, index[-1].year
    return list(range(first + warmup_years, last + 1))


def _gpu_lock():
    """Hold the shared GPU lock only when TensorFlow actually sees a GPU; on a
    CPU-only TensorFlow the lock would just push the hourly external pass onto
    the same CPU."""
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            from src.external.tsfm import gpu_job
            return gpu_job()
    except Exception:
        pass
    return contextlib.nullcontext()


def _frames() -> dict:
    h1 = pd.read_csv(repo_path('results/gold/gold_h1.csv'), parse_dates=['time'])
    d1 = B.load_d1()
    return {'H1': h1.set_index('time').sort_index().astype(float),
            'D1': d1.set_index('time').sort_index().astype(float)}


def _year_dir(models_dir: str, model: str, year: int) -> str:
    return os.path.join(models_dir, model, str(year))


def run(cfg: dict | None = None, frames: dict | None = None, make=None, build_inputs=None,
        build_targets=None, gold_cfg: dict | None = None, models: list | None = None) -> dict:
    """Fit and predict every (model, year); resume from completed years.
    Returns {model: DataFrame(time, p_up, vol_pct)} over all prediction years."""
    from src.gold import challengers as C
    from src.gold import load_config as gold_config
    from src.gold import targets as G
    cfg = cfg or load_config()
    make = make or C.make
    build_inputs = build_inputs or C.build_inputs
    build_targets = build_targets or G.build_targets
    gold_cfg = gold_cfg or gold_config()
    frames = frames or _frames()
    models_dir = repo_path(cfg['paths']['models_dir'])
    manifest = repo_path(cfg['paths']['artifact_manifest'])
    warm = cfg['walkforward']['warmup_years']
    out = {}
    with _gpu_lock():
        for model in models or cfg['walkforward']['models']:
            df = frames[CADENCE[model]]
            inputs = targets = None
            parts = []
            for year in prediction_years(df.index, warm):
                ydir = _year_dir(models_dir, model, year)
                done = os.path.join(ydir, 'done.json')
                pred_file = os.path.join(ydir, 'predictions.csv.gz')
                if os.path.exists(done):
                    with open(done, encoding='utf-8') as fh:
                        status = json.load(fh)
                    if status['ok']:
                        parts.append(pd.read_csv(pred_file, parse_dates=['time']))
                    _log(f'{model} {year}: resumed ({"ok" if status["ok"] else "no predictions"})')
                    continue
                if inputs is None:
                    inputs = build_inputs(df, CADENCE[model])
                    targets = build_targets(df['close'])
                fit_pos, pred_pos = year_positions(df.index, year)
                t0 = time.time()
                status = {'model': model, 'year': year, 'fit_rows': int(len(fit_pos)),
                          'fit_last_label': str(df.index[fit_pos[-1]]) if len(fit_pos) else '',
                          'pred_rows': int(len(pred_pos))}
                try:
                    m = make(model, gold_cfg)
                    m.fit(inputs, fit_pos, targets, seed=gold_cfg['seeds'][0])
                    res = m.predict(inputs)
                    frame = pd.DataFrame({'time': df.index[pred_pos]})
                    for k in OUTPUTS:
                        frame[k] = np.asarray(res.get(k, np.full(len(df), np.nan)), float)[pred_pos]
                    files = m.save(ydir)
                    B.save(frame, pred_file)
                    files.append(pred_file)
                    for f in files:
                        rel = os.path.relpath(f, repo_path('.')).replace('\\', '/')
                        B.update_manifest(manifest, rel, B.sha256_file(f), 0, '', '')
                    parts.append(frame)
                    status['ok'] = True
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
                _log(f'{model} {year}: {"ok" if status["ok"] else "FAILED " + status["error"]} '
                     f'(fit {status["fit_rows"]} rows to {status["fit_last_label"]}, {status["seconds"]} s)')
            allp = (pd.concat(parts, ignore_index=True).sort_values('time') if parts
                    else pd.DataFrame(columns=['time', *OUTPUTS]))
            path = os.path.join(models_dir, f'predictions_{model}.csv.gz')
            B.save(allp, path)
            rel = os.path.relpath(path, repo_path('.')).replace('\\', '/')
            B.update_manifest(manifest, rel, B.sha256_file(path), len(allp),
                              allp['time'].min() if len(allp) else '', allp['time'].max() if len(allp) else '')
            out[model] = allp
    return out


def load_predictions(model: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = os.path.join(repo_path(cfg['paths']['models_dir']), f'predictions_{model}.csv.gz')
    return pd.read_csv(path, parse_dates=['time'])


if __name__ == '__main__':
    res = run()
    for k, v in res.items():
        print(k, len(v), v['time'].min() if len(v) else '-', '..', v['time'].max() if len(v) else '-')
