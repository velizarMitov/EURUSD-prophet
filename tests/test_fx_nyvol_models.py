"""fx-ny-volatility, group 4: folds, HAR, the LightGBM regressor and the CUDA MLP."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_nyvol import models as M
from src.fx_nyvol import splits as S
from src.fx_session.splits import Fold
from tests.fx_nyvol_synth import CFG

DAYS = pd.bdate_range('2012-02-01', '2026-11-20', tz='UTC')


def test_folds_share_boundaries_and_predict_every_day_once():
    a = DAYS[::1]
    b = DAYS.delete(np.arange(5, len(DAYS), 37))                     # GBPUSD with a few different days
    last = max(a.max(), b.max())
    fa, fb = S.make_folds(a, CFG, last), S.make_folds(b, CFG, last)
    assert [(f.start, f.end) for f in fa] == [(f.start, f.end) for f in fb]
    assert fa[0].start == pd.Timestamp('2013-01-01', tz='UTC') and all(x.end == y.start for x, y in zip(fa, fa[1:]))
    tested = np.concatenate([f.eval for f in fa])
    assert len(tested) == len(set(tested)) == int((a >= pd.Timestamp('2013-01-01', tz='UTC')).sum())


def test_fold_starting_2019_07_01_embargoes_friday_28_june():
    f = [x for x in S.make_folds(DAYS, CFG) if x.start == pd.Timestamp('2019-07-01', tz='UTC')][0]
    train = np.r_[f.fit, f.val]
    assert DAYS[train].max() == pd.Timestamp('2019-06-27', tz='UTC')
    assert pd.Timestamp('2019-06-28', tz='UTC') not in set(DAYS[train])


def test_scalers_are_fit_on_the_fit_slice_only():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 3)), rng.normal(size=300)
    f = Fold(0, None, None, np.arange(200), np.arange(210, 250), np.arange(260, 300))
    a = S.fold_arrays(X, y, f)
    X2, y2 = X.copy(), y.copy()
    X2[200:] += 100
    y2[200:] -= 50
    b = S.fold_arrays(X2, y2, f)
    assert np.allclose(a['x_scaler'].mean, b['x_scaler'].mean) and a['y_scaler'].mean == b['y_scaler'].mean
    assert np.allclose(a['Xs_fit'].mean(axis=0), 0, atol=1e-6)


def test_har_recovers_known_coefficients_and_ignores_test_days():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = 0.5 + X @ np.array([0.3, 0.2, 0.4]) + rng.normal(0, 0.01, 2000)
    coef = M.ols_fit(X[:1500], y[:1500])
    np.testing.assert_allclose(coef, [0.5, 0.3, 0.2, 0.4], atol=0.01)
    y2 = y.copy()
    y2[1500:] = 99.0
    assert np.array_equal(M.ols_fit(X[:1500], y2[:1500]), coef)


def test_same_seed_lightgbm_regressor_is_identical():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 14))
    y = X[:, 0] * 0.5 + rng.normal(0, 1, 800)
    p = [M.train_lightgbm(X[:500], y[:500], X[500:600], y[500:600], CFG['lightgbm']).predict(X[600:]) for _ in range(2)]
    assert np.array_equal(p[0], p[1]) and CFG['lightgbm']['objective'] == 'l2'


def test_mlp_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(M.CudaUnavailable):
        M.train_mlp(np.zeros((10, 3), np.float32), np.zeros(10, np.float32), np.zeros((5, 3), np.float32),
                    np.zeros(5, np.float32), CFG['mlp'], 11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_mlp_on_cuda_beats_a_constant_on_a_learnable_toy_series():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 4)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, 1500)).astype(np.float32)
    f = Fold(0, None, None, np.arange(1000), np.arange(1000, 1200), np.arange(1200, 1500))
    a = S.fold_arrays(X, y, f)
    params = {**CFG['mlp'], 'max_epochs': 80}
    members = [M.train_mlp(a['Xs_fit'], a['ys_fit'], a['Xs_val'], a['ys_val'], params, s)[0] for s in params['seeds']]
    pred = M.ensemble_predict(members, a['Xs_eval'], len(params['seeds']), a['y_scaler'])
    mse = np.mean((pred - a['y_eval']) ** 2)
    assert mse < 0.5 * np.mean((a['y_eval'] - a['y_fit'].mean()) ** 2)
    with pytest.raises(ValueError, match='partial ensemble'):
        M.ensemble_predict(members[:4], a['Xs_eval'], 5, a['y_scaler'])
