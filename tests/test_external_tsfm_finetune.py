"""Fine-tuning reads only the frozen pre-window history (openspec change
tsfm-second-opinion, spec "Fine-tuning uses the whole history before the clean
window", design D10)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.external.tsfm import data as D
from src.external.tsfm import finetune as FT
from src.external.tsfm import vendor as V


def _bars(start, end):
    idx = pd.date_range(start, end, freq='h', tz='UTC')
    idx = idx[idx.dayofweek < 5]
    close = 2000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, len(idx))))
    return pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / 'results' / 'external_tsfm').mkdir(parents=True)
    return str(tmp_path)


def test_an_injected_clean_window_bar_makes_the_freeze_refuse(repo):
    df = _bars('2025-10-20', '2025-10-31 23:00')
    late = df.iloc[-1:].copy()
    late.index = pd.DatetimeIndex([pd.Timestamp('2025-11-01 00:00', tz='UTC')])
    with pytest.raises(D.BoundaryViolation, match='2025-11-01'):
        D.write_history('gold', pd.concat([df, late]), 'test', repo)
    assert not os.path.exists(D.history_path('gold', repo))


def test_freeze_keeps_every_bar_before_the_boundary_and_none_after(repo):
    gold = _bars('2024-06-03', '2025-11-14')
    eur = _bars('2024-01-02', '2025-11-14')
    out = FT.freeze(repo, gold=gold, eurusd=eur)
    for inst, src in (('gold', gold), ('eurusd', eur)):
        frozen = D.read_frozen(D.history_path(inst, repo))
        before = src[src.index < D.clean_start()]
        assert len(frozen) == len(before) == out[inst]['rows']
        assert frozen.index[-1] < D.clean_start()
        assert frozen.index[-1] == before.index[-1]
    m = V.read_manifest(repo=repo)
    assert {'history_gold', 'history_eurusd'} <= set(m)
    meta = FT.read_meta(repo)
    assert meta['history']['clean_start'] == '2025-11-01'


def test_training_inputs_come_only_from_the_frozen_file(repo):
    FT.freeze(repo, gold=_bars('2024-06-03', '2025-11-14'), eurusd=_bars('2024-06-03', '2025-11-14'))
    read = []

    def spy(path):
        read.append(path)
        return D.read_frozen(path)
    prep = FT.prepare('chronos2', 'gold', repo, reader=spy)
    assert read == [D.history_path('gold', repo)]
    hist = D.read_frozen(D.history_path('gold', repo))
    assert prep['n_train2'] == len(hist) - 1
    assert pd.Timestamp(prep['train1_last_bar']) < pd.Timestamp('2024-11-01', tz='UTC')
    hold = np.load(prep['hold_pos'])
    series = np.load(prep['hold_series'])
    assert hold.max() + 1 < len(series)                 # every holdout target is inside the history


def test_prepare_refuses_a_history_that_crosses_the_boundary(repo):
    bad = _bars('2025-10-01', '2025-11-03')
    with pytest.raises(D.BoundaryViolation):
        FT.prepare('chronos2', 'gold', repo, reader=lambda p: bad)


def _fake_call(curve, files=('adapter_model.safetensors', 'adapter_config.json')):
    calls = []

    def call(req, timeout=None, log_path=None):
        calls.append(req)
        if req['out_dir'] is None:
            return {'ok': True, 'curve': curve, 'seconds': 1.0, 'gpu': 'fake', 'peak_mem_gb': 0.1}
        os.makedirs(req['out_dir'], exist_ok=True)
        for f in files:
            with open(os.path.join(req['out_dir'], f), 'w') as fh:
                fh.write(f'{f}-{req["steps"]}')
        return {'ok': True, 'seconds': 1.0, 'probe': [[0.0] * 9]}
    return call, calls


def _ready(repo):
    FT.freeze(repo, gold=_bars('2024-06-03', '2025-11-14'), eurusd=_bars('2024-06-03', '2025-11-14'))
    wdir = V.weights_dir('chronos2', repo)
    os.makedirs(wdir)
    with open(os.path.join(wdir, 'model.safetensors'), 'w') as fh:
        fh.write('w')
    V.record('chronos2', [os.path.join(wdir, 'model.safetensors')], 'test', repo=repo)


def test_stage_two_uses_the_best_stage_one_step_and_all_history(repo):
    _ready(repo)
    call, calls = _fake_call([[0, 0.30], [100, 0.29], [200, 0.27], [300, 0.28]])
    e = FT.train_pair('chronos2', 'gold', repo, call)
    assert e['status'] == 'trained' and e['chosen_steps'] == 200
    assert calls[0]['steps'] == FT.MAX_STEPS and calls[0]['train_path'].endswith('train_stage1.npy')
    assert calls[1]['steps'] == 200 and calls[1]['train_path'].endswith('train_stage2.npy')
    assert calls[1]['seed'] == 42
    assert V.verify('chronos2_gold_lora', repo=repo) == e['adapter_digest']


def test_a_fine_tune_that_never_beats_zero_shot_is_dropped(repo):
    _ready(repo)
    call, calls = _fake_call([[0, 0.25], [100, 0.26], [200, 0.27]])
    e = FT.train_pair('chronos2', 'gold', repo, call)
    assert e['status'] == 'failed' and 'zero-shot' in e['reason']
    assert len(calls) == 1
    assert 'chronos2_gold_lora' not in V.read_manifest(repo=repo)


def test_a_worker_failure_is_recorded_not_raised(repo):
    _ready(repo)
    e = FT.train_pair('chronos2', 'gold', repo, lambda req, **kw: {'ok': False, 'error': 'CUDA OOM'})
    assert e['status'] == 'failed' and 'CUDA OOM' in e['reason']


def test_meta_lists_every_pair(repo):
    _ready(repo)
    call, _ = _fake_call([[0, 0.30], [100, 0.29]])
    meta = FT.train(repo, call=call)
    assert set(meta['pairs']) == {'chronos2_gold', 'chronos2_eurusd', 'timesfm25_gold', 'timesfm25_eurusd'}
    for e in meta['pairs'].values():
        assert e['status'] in ('trained', 'failed')
        assert e['status'] == 'trained' or e['reason']
    with open(os.path.join(repo, 'results', 'external_tsfm', 'finetune', 'finetune_meta.json')) as fh:
        assert json.load(fh)['pairs']
