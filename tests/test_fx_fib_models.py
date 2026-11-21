"""fx-fibonacci-mlp, group 7: splits, scaler, CUDA MLP, ensemble, artifacts and baselines."""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_fib import artifacts as AR
from src.fx_fib import baselines as B
from src.fx_fib import load_config
from src.fx_fib import model_mlp_cuda as M
from src.fx_fib import splits as SP
from src.fx_fib import walk_forward_backtest as WF

CFG = load_config()
H = CFG['labels']['horizon_bars']
FAST = dict(CFG['mlp'], max_epochs=25, patience=5)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def hourly_index(start='2018-01-01', end='2020-12-31 23:00'):
    idx = pd.date_range(start, end, freq='h', tz='UTC')
    return idx[idx.dayofweek < 5]


def toy(n=3000, d=42, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logits = np.c_[np.zeros(n), 1.5 * X[:, 0], -1.5 * X[:, 0]]
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    return X, y


# -- 7.1 splits ------------------------------------------------------------------

def _folds(index=None, eligible=None, cfg=CFG):
    index = hourly_index() if index is None else index
    eligible = np.ones(len(index), bool) if eligible is None else eligible
    return SP.make_folds(index, eligible, cfg)


def test_folds_are_chronological_with_purge_and_embargo():
    cfg = copy.deepcopy(CFG)
    cfg['folds']['first_eval_start'] = '2019-07-01'
    folds = _folds(cfg=cfg)
    assert [f.start.strftime('%Y-%m-%d') for f in folds] == ['2019-07-01', '2020-01-01', '2020-07-01']
    for f in folds:
        SP.check_fold(f, H)


def test_overlapping_fit_rows_are_removed():
    f = _folds()[0]
    # Any fit row whose horizon t+1..t+H contains the first validation bar must be gone.
    assert not np.any((f.fit + H >= f.val[0]))
    assert f.val[0] - f.fit.max() >= 2 * H


def test_gap_size_between_validation_and_evaluation():
    for f in _folds():
        assert f.eval.min() - f.val.max() >= 2 * H


def test_ineligible_rows_are_never_used():
    idx = hourly_index()
    elig = np.ones(len(idx), bool)
    elig[::7] = False
    for f in SP.make_folds(idx, elig, CFG):
        for rows in (f.fit, f.val, f.eval):
            assert elig[rows].all()


def test_two_pairs_share_fold_dates():
    a = _folds(hourly_index())
    b = _folds(hourly_index('2017-06-01'))
    assert [(f.start, f.end) for f in a] == [(f.start, f.end) for f in b]


# -- 7.2 scaler ----------------------------------------------------------------------

def test_scaler_statistics_come_from_fit_rows_only():
    X, y = toy(4000)
    f = SP.Fold(0, None, None, np.arange(0, 2000), np.arange(2100, 2400), np.arange(2500, 4000))
    X_shift = X.copy()
    X_shift[2000:] += 100.0                      # later rows must not move the scaler
    sc = SP.fit_scaler(X_shift, f)
    assert np.allclose(sc.center_, np.median(X[:2000], axis=0))
    arr = SP.fold_arrays(X_shift, y, f, sc)
    assert arr['X_fit'].dtype == np.float32 and np.array_equal(arr['rows_eval'], f.eval)


# -- 7.3 / 7.4 the CUDA MLP ----------------------------------------------------------

def test_architecture_is_exactly_the_registered_one():
    net = M.build_mlp(42, [128, 64, 32], [0.3, 0.2])
    kinds = [type(m).__name__ for m in net]
    assert kinds == ['Linear', 'LayerNorm', 'ReLU', 'Dropout', 'Linear', 'LayerNorm', 'ReLU', 'Dropout',
                     'Linear', 'ReLU', 'Linear']
    assert [m.out_features for m in net if isinstance(m, torch.nn.Linear)] == [128, 64, 32, 3]
    assert [m.p for m in net if isinstance(m, torch.nn.Dropout)] == [0.3, 0.2]


def test_cuda_missing_and_cpu_not_allowed_refuses(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    X, y = toy(200)
    with pytest.raises(M.CudaUnavailable):
        M.train_mlp(X, y, X, y, dict(FAST, allow_cpu=False), seed=1)


@needs_cuda
def test_training_runs_on_cuda_and_outputs_valid_probabilities():
    X, y = toy()
    model, meta = M.train_mlp(X[:2000], y[:2000], X[2000:2500], y[2000:2500], FAST, seed=11)
    p = M.predict_proba(model, X[2500:])
    assert p.shape == (500, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert 'NVIDIA' in meta['device'] or 'GeForce' in meta['device']
    assert meta['cuda'] and meta['seed'] == 11 and meta['epochs_trained'] >= meta['best_epoch'] + 1


@needs_cuda
def test_best_epoch_weights_are_restored():
    X, y = toy()
    model, meta = M.train_mlp(X[:2000], y[:2000], X[2000:2500], y[2000:2500], FAST, seed=22)
    p = M.predict_proba(model, X[2000:2500])
    ll = -np.mean(np.log(p[np.arange(500), y[2000:2500]]))
    assert ll == pytest.approx(meta['best_val_logloss'], abs=1e-4)


@needs_cuda
def test_rerun_with_the_same_seed_reproduces_probabilities():
    X, y = toy()
    a, _ = M.train_mlp(X[:2000], y[:2000], X[2000:2500], y[2000:2500], FAST, seed=33)
    b, _ = M.train_mlp(X[:2000], y[:2000], X[2000:2500], y[2000:2500], FAST, seed=33)
    diff = np.abs(M.predict_proba(a, X[2500:]) - M.predict_proba(b, X[2500:])).max()
    assert diff <= 1e-5


def test_global_determinism_switch_is_restored(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    before = torch.are_deterministic_algorithms_enabled()
    X, y = toy(300)
    M.train_mlp(X, y, X, y, dict(FAST, allow_cpu=True, max_epochs=2), seed=1)
    assert torch.are_deterministic_algorithms_enabled() == before


# -- 7.5 ensemble and artifacts ---------------------------------------------------------

def test_partial_ensemble_is_refused():
    net = M.build_mlp(4, [8, 4], [0.1])
    with pytest.raises(ValueError, match='partial ensemble'):
        M.ensemble_proba([net] * 4, np.zeros((2, 4), np.float32), 5)
    with pytest.raises(ValueError, match='partial ensemble'):
        M.ensemble_proba([net] * 4 + [None], np.zeros((2, 4), np.float32), 5)


@pytest.fixture
def art_cfg(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['artifacts'] = str(tmp_path / 'art')
    cfg['paths']['results'] = str(tmp_path / 'res')
    return cfg


def _save_five(cfg):
    for s in cfg['mlp']['seeds']:
        torch.manual_seed(s)
        net = M.build_mlp(42, [128, 64, 32], [0.3, 0.2])
        AR.save_mlp(cfg, net, {'hidden': [128, 64, 32], 'dropout': [0.3, 0.2], 'seed': s}, 'EURUSD', 3, f'seed{s}', 'T')


def test_ensemble_loads_all_five_verified_members(art_cfg):
    _save_five(art_cfg)
    members = AR.load_ensemble(art_cfg, 'EURUSD', 3, 42)
    assert len(members) == 5
    p = M.ensemble_proba(members, np.zeros((3, 42), np.float32), 5)
    assert np.allclose(p.sum(axis=1), 1.0)
    man = AR.read_manifest(art_cfg)
    assert len(man) == 5 and all(len(r['sha256']) == 64 for r in man.values())


def test_tampered_member_blocks_the_ensemble(art_cfg):
    _save_five(art_cfg)
    path = os.path.join(art_cfg['paths']['artifacts'], 'EURUSD', 'fold_03', 'mlp_seed33.pt')
    with open(path, 'ab') as fh:
        fh.write(b'x')
    with pytest.raises(AR.ArtifactError, match='digest mismatch'):
        AR.load_ensemble(art_cfg, 'EURUSD', 3, 42)


def test_missing_member_blocks_the_ensemble(art_cfg):
    _save_five(art_cfg)
    os.remove(os.path.join(art_cfg['paths']['artifacts'], 'EURUSD', 'fold_03', 'mlp_seed55.pt'))
    with pytest.raises(AR.ArtifactError, match='missing artifact'):
        AR.load_ensemble(art_cfg, 'EURUSD', 3, 42)


# -- 7.6 baselines on identical inputs --------------------------------------------------

def test_lightgbm_uses_fixed_parameters_and_early_stopping():
    X, y = toy()
    model, meta = B.train_lightgbm(X[:2000], y[:2000], X[2000:2500], y[2000:2500], CFG['lightgbm'])
    assert model.get_params()['num_leaves'] == 31 and model.get_params()['learning_rate'] == 0.03
    assert 1 <= meta['best_iteration'] <= 2000
    assert np.allclose(B.proba(model, X[2500:]).sum(axis=1), 1.0)


def test_logistic_baseline_trains():
    X, y = toy()
    model, _ = B.train_logistic(X[:2000], y[:2000], CFG['logistic'])
    assert B.proba(model, X[2500:]).shape == (500, 3)


def test_every_model_receives_identical_arrays(art_cfg):
    X, y = toy(1200)
    f = SP.Fold(0, None, None, np.arange(0, 800), np.arange(850, 950), np.arange(1000, 1200))
    arr = SP.fold_arrays(X, y, f)
    seen = {}

    def rec(name):
        def train(a, *seed):
            seen.setdefault(name, []).append((a['X_fit'], a['y_fit'], a['X_val'], a['X_eval']))
            net = M.build_mlp(42, [8, 4], [0.1])
            return (net, {'hidden': [8, 4], 'dropout': [0.1]}) if name.startswith('mlp') else (
                B.train_logistic(a['X_fit'], a['y_fit'], CFG['logistic']))
        return train

    trainers = {n: rec(n) for n in WF.MODELS}
    out = WF.train_fold_models(arr, art_cfg, 'EURUSD', 0, 'T', save=False, trainers=trainers)
    ref = seen['mlp'][0]
    for name, calls in seen.items():
        for call in calls:
            assert all(x is r for x, r in zip(call, ref)), name
    assert len(seen['mlp']) == 5 and set(out['proba']) == set(WF.MODELS)
