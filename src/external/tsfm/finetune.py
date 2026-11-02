"""Fine-tuning on the full pre-window history (design D10; spec "Fine-tuning uses
the whole history before the clean window"). Owner's request, 2026-11-02.

    python -m src.external.tsfm.finetune --freeze    freeze each instrument's H1 history
    python -m src.external.tsfm.finetune --train     LoRA, 2 models x 2 instruments (GPU)
    python -m src.external.tsfm.finetune --check     reload each adapter, compare its probe

--freeze writes results/external_tsfm/finetune/<instrument>_h1_history.csv:
every bar BEFORE 2025-11-01 (data.write_history refuses any later one). The
fine-tuning reads nothing else, so the boundary is checked in one place.

--train, per model and instrument, in the side environment on the GPU:
  stage 1  LoRA on the bars before 2024-11-01; the holdout year 2024-11-01 ..
           2025-10-31 is scored (mean pinball loss of the 9 deciles of the next
           return, in %) at step 0 -- the zero-shot model -- and every 100 steps
           up to 3,000. The step count with the lowest holdout loss is chosen.
  stage 2  from the base weights again, LoRA on EVERY bar before 2025-11-01 for
           exactly that many steps, seed 42. The adapter is frozen with its
           digests in the artifact manifest.
A fine-tune that fails, or whose holdout loss never goes below the zero-shot
loss, is recorded and dropped; the zero-shot model stays.

The learning rate is a constant schedule in both stages, so step k of stage 1
and a stage-2 run of k steps see the same rate. Chronos-2 uses its package's
own LoRA fine-tuning (`Chronos2Pipeline.fit`) at the package's recommended
LoRA rate 1e-5; TimesFM 2.5 has none, so the worker trains LoRA adapters on
its attention and feed-forward layers through the same first-step path its
forecasts use, at 1e-4.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import ARTIFACTS, CLEAN_START, FINETUNE, INSTRUMENTS, MODELS, REPO, RESULTS, gpu_job, repo_path
from . import client as CL
from . import data as D
from . import sources as S
from . import vendor as V

CONTEXT = 2048
BATCH = 32
SEED = 42
MAX_STEPS = 3000
EVAL_EVERY = 100
HOLDOUT_CAP = 3000
LEARNING_RATE = {'chronos2': 1e-5, 'timesfm25': 1e-4}
# TimesFM's training path runs the model twice (flip invariance): two micro-
# batches of 16 keep its peak under 4 GB of the 8 GB card; the batch stays 32.
MICRO_BATCH = {'chronos2': None, 'timesfm25': 16}
META = f'{FINETUNE}/finetune_meta.json'
ADAPTERS = f'{ARTIFACTS}/finetuned'
WORK = f'{ARTIFACTS}/finetune_work'
VERSIONED = ('chronos-forecasting', 'timesfm', 'peft', 'torch', 'transformers', 'numpy')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def adapter_dir(model: str, instrument: str, repo: str = REPO) -> str:
    return repo_path(f'{ADAPTERS}/{model}_{instrument}', repo)


def adapter_artifact(model: str, instrument: str) -> str:
    return f'{model}_{instrument}_lora'


def read_meta(repo: str = REPO) -> dict:
    path = repo_path(META, repo)
    if not os.path.exists(path):
        return {'pairs': {}}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def write_meta(meta: dict, repo: str = REPO) -> None:
    path = repo_path(META, repo)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def versions(repo: str = REPO) -> dict:
    path = repo_path(f'{RESULTS}/env_freeze.txt', repo)
    out = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                name, _, ver = line.strip().partition('==')
                if name.lower() in VERSIONED:
                    out[name.lower()] = ver
    return out


# ── freeze ──────────────────────────────────────────────────────────────────

def freeze(repo: str = REPO, gold=None, eurusd=None) -> dict:
    cs = D.clean_start()
    out = {}
    g = gold if gold is not None else D.gold_h1(repo)
    out['gold'] = D.write_history('gold', g[g.index < cs], f'{S.GOLD_H1_CACHE} (append-only cache)', repo)
    if eurusd is None:
        eurusd, n = D.eurusd_h1_deepest()
        src = f'MT5 {D.EURUSD_SYMBOL} H1, deepest read the terminal serves ({n} bars)'
    else:
        src = 'injected frame'
    out['eurusd'] = D.write_history('eurusd', eurusd[eurusd.index < cs], src, repo)
    meta = read_meta(repo)
    meta['history'] = {**out, 'frozen_at': _now(), 'clean_start': CLEAN_START}
    write_meta(meta, repo)
    return out


# ── train ───────────────────────────────────────────────────────────────────

def prepare(model: str, instrument: str, repo: str = REPO, reader=D.read_frozen) -> dict:
    """The npy inputs of both stages, built ONLY from the frozen history file
    (re-checked against the boundary as it is read). Series position j is bar
    j + 1's return, so an as-of bar p sits at j = p - 1 and its target at j + 1."""
    hist = reader(D.history_path(instrument, repo))
    if len(hist) and hist.index[-1] >= D.clean_start():
        raise D.BoundaryViolation(f'{instrument} history holds a bar on or after {CLEAN_START}')
    ret = D.returns_pct(hist['close'].to_numpy())
    k = int(np.searchsorted(hist.index, pd.Timestamp(D.HOLDOUT_START, tz='UTC')))
    hold = D.spaced(D.as_of_positions(hist.index, D.HOLDOUT_START, CLEAN_START), HOLDOUT_CAP)
    work = repo_path(f'{WORK}/{model}_{instrument}', repo)
    os.makedirs(work, exist_ok=True)
    paths = {'train1': os.path.join(work, 'train_stage1.npy'), 'train2': os.path.join(work, 'train_stage2.npy'),
             'hold_series': os.path.join(work, 'holdout_series.npy'),
             'hold_pos': os.path.join(work, 'holdout_positions.npy')}
    np.save(paths['train1'], ret[1:k])
    np.save(paths['train2'], ret[1:])
    np.save(paths['hold_series'], ret[1:])
    np.save(paths['hold_pos'], (hold - 1).astype(np.int64))
    return {**paths, 'work': work, 'n_train1': int(k - 1), 'n_train2': int(len(ret) - 1),
            'n_holdout': int(len(hold)), 'train1_last_bar': str(hist.index[k - 1]),
            'history_last_bar': str(hist.index[-1])}


def _request(model, prep, repo, steps, train, eval_every, out_dir=None, work_dir=None):
    return {'op': 'finetune', 'model': model, 'weights_dir': V.weights_dir(model, repo),
            'train_path': prep[train], 'holdout_series_path': prep['hold_series'],
            'holdout_positions_path': prep['hold_pos'], 'steps': int(steps), 'eval_every': eval_every,
            'learning_rate': LEARNING_RATE[model], 'batch_size': BATCH, 'micro_batch': MICRO_BATCH[model],
            'context': CONTEXT, 'seed': SEED, 'out_dir': out_dir, 'work_dir': work_dir}


def train_pair(model: str, instrument: str, repo: str = REPO, call=CL.call) -> dict:
    entry = {'model': model, 'instrument': instrument, 'learning_rate': LEARNING_RATE[model],
             'batch_size': BATCH, 'micro_batch': MICRO_BATCH[model], 'context': CONTEXT, 'seed': SEED,
             'max_steps': MAX_STEPS,
             'eval_every': EVAL_EVERY, 'lr_schedule': 'constant', 'started_at': _now()}
    try:
        entry['base_digest'] = V.verify(model, repo=repo)
        entry['history_digest'] = V.verify(f'history_{instrument}', repo=repo)
        prep = prepare(model, instrument, repo)
    except Exception as e:                                # noqa: BLE001 -- recorded, never fatal
        return {**entry, 'status': 'failed', 'reason': f'{type(e).__name__}: {e}'}
    entry['data'] = {k: v for k, v in prep.items() if k.startswith(('n_', 'train1_last', 'history_last'))}
    log = os.path.join(prep['work'], 'train.log')
    s1 = call(_request(model, prep, repo, MAX_STEPS, 'train1', EVAL_EVERY,
                       work_dir=os.path.join(prep['work'], 'stage1')), timeout=None, log_path=log)
    if not s1.get('ok'):
        return {**entry, 'status': 'failed', 'reason': f'stage 1: {s1.get("error")}'}
    curve = s1['curve']
    entry.update(stage1_curve=curve, stage1_seconds=s1.get('seconds'), gpu=s1.get('gpu'),
                 peak_mem_gb=s1.get('peak_mem_gb'))
    zero = [c[1] for c in curve if c[0] == 0]
    trained = [c for c in curve if c[0] > 0]
    if not zero or not trained:
        return {**entry, 'status': 'failed', 'reason': 'stage 1 returned no holdout curve'}
    best_step, best_loss = min(trained, key=lambda c: c[1])
    entry.update(zero_shot_loss=zero[0], best_loss=best_loss, chosen_steps=int(best_step))
    if not best_loss < zero[0]:
        return {**entry, 'status': 'failed', 'reason': 'holdout loss never went below the zero-shot loss'}
    out = adapter_dir(model, instrument, repo)
    shutil.rmtree(out, ignore_errors=True)
    s2 = call(_request(model, prep, repo, best_step, 'train2', 0, out_dir=out,
                       work_dir=os.path.join(prep['work'], 'stage2')), timeout=None, log_path=log)
    if not s2.get('ok'):
        return {**entry, 'status': 'failed', 'reason': f'stage 2: {s2.get("error")}'}
    files = [os.path.join(out, f) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))]
    rows = V.record(adapter_artifact(model, instrument), files,
                    f'LoRA stage 2: {best_step} steps on history_{instrument} {entry["history_digest"]}',
                    repo=repo)
    return {**entry, 'status': 'trained', 'stage2_seconds': s2.get('seconds'), 'probe': s2.get('probe'),
            'adapter': [r['path'] for r in rows], 'adapter_digest': V.verify(adapter_artifact(model, instrument),
                                                                            repo=repo),
            'finished_at': _now()}


def train(repo: str = REPO, pairs=None, call=CL.call) -> dict:
    meta = read_meta(repo)
    meta['versions'] = versions(repo)
    with gpu_job(repo):
        for model, inst in pairs or [(m, i) for m in MODELS for i in INSTRUMENTS]:
            entry = train_pair(model, inst, repo, call)
            meta.setdefault('pairs', {})[f'{model}_{inst}'] = entry
            write_meta(meta, repo)
            print(f'{model} {inst}: {entry["status"]}'
                  + (f' ({entry.get("chosen_steps")} steps)' if entry['status'] == 'trained'
                     else f' -- {entry["reason"]}'), flush=True)
    return meta


def check(repo: str = REPO, call=CL.call) -> dict:
    """Reload each frozen adapter from disk and require the SAME probe deciles
    as at training time (CPU, same code path)."""
    meta = read_meta(repo)
    out = {}
    for key, e in meta.get('pairs', {}).items():
        if e.get('status') != 'trained':
            continue
        try:
            V.verify(adapter_artifact(e['model'], e['instrument']), repo=repo)
            work = repo_path(f'{WORK}/{key}', repo)
            r = call({'op': 'probe', 'model': e['model'], 'weights_dir': V.weights_dir(e['model'], repo),
                      'adapter_dir': adapter_dir(e['model'], e['instrument'], repo), 'context': CONTEXT,
                      'holdout_series_path': os.path.join(work, 'holdout_series.npy'),
                      'holdout_positions_path': os.path.join(work, 'holdout_positions.npy')}, timeout=None)
            same = bool(r.get('ok') and np.array_equal(np.asarray(r['probe']), np.asarray(e['probe'])))
            out[key] = 'identical' if same else f'DIFFERENT: {r.get("error", "deciles differ")}'
        except Exception as ex:                           # noqa: BLE001
            out[key] = f'FAILED: {type(ex).__name__}: {ex}'
    meta['reload_check'] = {'at': _now(), 'result': out}
    write_meta(meta, repo)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Full-history LoRA fine-tuning of the external models.')
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--pair', action='append', help='model:instrument, e.g. chronos2:gold (default all four)')
    a = ap.parse_args(argv)
    if a.freeze:
        print(json.dumps(freeze(), indent=1))
    if a.train:
        pairs = [tuple(p.split(':')) for p in a.pair] if a.pair else None
        train(pairs=pairs)
    if a.check:
        print(json.dumps(check(), indent=1))


if __name__ == '__main__':
    main()
