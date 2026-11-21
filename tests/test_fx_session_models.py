"""fx-session-ny, group 5: folds, LightGBM, the class prior and the conditional CUDA MLP."""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_session import artifacts as AR
from src.fx_session import load_config
from src.fx_session import model_lgbm as LG
from src.fx_session import model_mlp as MM
from src.fx_session import splits as SP

CFG = load_config()


def bdays(start='2012-01-02', end='2014-12-31', drop=()):
    d = pd.bdate_range(start, end, tz='UTC')
    return d[~d.isin(pd.DatetimeIndex(drop, tz='UTC'))]


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'd'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    return c


def toy(n=1200, k=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = (X[:, 0] + 0.8 * rng.normal(size=n) > 0).astype(float)
    return X, y


# -- 5.1 folds ----------------------------------------------------------------------------------

def test_both_pairs_share_fold_boundaries():
    e = bdays()
    g = bdays(drop=['2013-03-05', '2014-02-11'])
    last = max(e.max(), g.max())
    fe, fg = SP.make_folds(e, CFG, last), SP.make_folds(g, CFG, last)
    assert [(f.start, f.end) for f in fe] == [(f.start, f.end) for f in fg]
    assert fe[0].start == pd.Timestamp('2013-01-01', tz='UTC') and len(fe) == 4


def test_one_usable_day_separates_each_slice():
    days = bdays()
    for f in SP.make_folds(days, CFG):
        assert f.val[0] - f.fit[-1] == 2           # exactly one day dropped between fit and val
        assert f.eval[0] - f.val[-1] == 2          # and between val and eval
        assert days[f.fit[-1]] < days[f.val[0]] < days[f.eval[0]]
        assert abs(len(f.val) / (len(f.fit) + len(f.val) + 1) - 0.15) < 0.01


def test_scaler_sees_only_the_fit_slice():
    days = bdays()
    X, y = toy(len(days))
    f = SP.make_folds(days, CFG)[0]
    a = SP.fold_arrays(X, y, f)
    X2 = X.copy()
    X2[f.val] *= 100
    X2[f.eval] += 50
    b = SP.fold_arrays(X2, y, f)
    assert np.allclose(a['scaler'].center_, b['scaler'].center_) and np.allclose(a['scaler'].scale_, b['scaler'].scale_)
    assert a['Xs_fit'].dtype == np.float32


# -- 5.2 LightGBM and the prior -------------------------------------------------------------------

def test_lightgbm_same_seed_gives_identical_probabilities():
    X, y = toy()
    p1 = LG.proba_up(LG.train_lightgbm(X[:800], y[:800], X[800:1000], y[800:1000], CFG['lightgbm']), X[1000:])
    p2 = LG.proba_up(LG.train_lightgbm(X[:800], y[:800], X[800:1000], y[800:1000], CFG['lightgbm']), X[1000:])
    assert np.array_equal(p1, p2) and 0 < p1.min() and p1.max() < 1
    assert np.mean((p1 > 0.5) == (y[1000:] == 1)) > 0.6          # it learns the planted signal


def test_prior_is_the_fit_up_rate():
    assert LG.prior_up(np.array([1, 1, 0, 1, 0] * 10 + [1, 1])) == pytest.approx(32 / 52)


def test_lightgbm_settings_carry_no_class_weights():
    p = CFG['lightgbm']
    assert 'class_weight' not in p and 'is_unbalance' not in p and 'scale_pos_weight' not in p
    assert p['objective'] == 'binary' and p['deterministic'] is True


# -- 5.3 the MLP ------------------------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@gpu
def test_mlp_same_seed_rerun_matches_and_learns():
    X, y = toy()
    p = dict(CFG['mlp'], max_epochs=40)
    m1, meta = MM.train_mlp(X[:800], y[:800], X[800:1000], y[800:1000], p, 11)
    m2, _ = MM.train_mlp(X[:800], y[:800], X[800:1000], y[800:1000], p, 11)
    a, b = MM.predict_up(m1, X[1000:]), MM.predict_up(m2, X[1000:])
    assert np.abs(a - b).max() < 1e-5
    assert meta['device'] != 'cpu' and np.mean((a > 0.5) == (y[1000:] == 1)) > 0.6
    assert not torch.are_deterministic_algorithms_enabled()


def test_mlp_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    X, y = toy(200)
    with pytest.raises(MM.CudaUnavailable):
        MM.train_mlp(X[:150], y[:150], X[150:], y[150:], CFG['mlp'], 11)


@gpu
def test_saved_ensemble_loads_verified_and_refuses_tampering(cfg):
    X, y = toy(400)
    p = dict(cfg['mlp'], max_epochs=3)
    for s in cfg['mlp']['seeds']:
        m, meta = MM.train_mlp(X[:300], y[:300], X[300:], y[300:], p, s)
        MM.save_member(cfg, m, meta, 'EURUSD', 0, s, 'T')
    models = MM.load_ensemble(cfg, 'EURUSD', 0, X.shape[1])
    assert MM.ensemble_up(models, X[:5], 5).shape == (5,)
    with pytest.raises(ValueError, match='partial ensemble'):
        MM.ensemble_up(models[:4], X[:5], 5)
    rel = f"{AR.fold_dir(cfg, 'EURUSD', 0)}/mlp_seed33.joblib"
    with open(rel, 'ab') as fh:
        fh.write(b'x')
    with pytest.raises(AR.ArtifactError, match='digest mismatch'):
        MM.load_ensemble(cfg, 'EURUSD', 0, X.shape[1])
    os.remove(rel)
    with pytest.raises(AR.ArtifactError, match='missing'):
        MM.load_ensemble(cfg, 'EURUSD', 0, X.shape[1])
