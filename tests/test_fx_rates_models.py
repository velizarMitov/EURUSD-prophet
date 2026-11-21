"""fx-rates-2y, group 5: folds with purge and embargo, and the model wiring."""
import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_rates import load_config
from src.fx_rates import splits as SP
from src.fx_session import artifacts as AR
from src.fx_session import model_lgbm as LG
from src.fx_session import model_mlp as MM

CFG = load_config()


def days(drop=()):
    d = pd.bdate_range('2013-01-21', '2017-12-29')
    return d[~d.isin(pd.DatetimeIndex(drop))]


def toy(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    return X, (X[:, 0] + rng.normal(size=n) > 0).astype(float)


def test_both_pairs_share_boundaries_from_2015():
    e, g = days(), days(drop=['2015-03-03', '2016-05-10'])
    last = max(e.max(), g.max())
    fe, fg = SP.make_folds(e, CFG, last), SP.make_folds(g, CFG, last)
    assert [(f.start, f.end) for f in fe] == [(f.start, f.end) for f in fg]
    assert fe[0].start == pd.Timestamp('2015-01-01') and len(fe) == 6


def test_gaps_hold_purge_plus_embargo_and_no_fit_target_reaches_validation():
    d = days()
    h = CFG['target']['horizon_days']
    for f in SP.make_folds(d, CFG):
        assert f.val[0] - f.fit[-1] == 11 and f.eval[0] - f.val[-1] == 11       # 10 days dropped each time
        assert f.fit[-1] + h < f.val[0] and f.val[-1] + h < f.eval[0]           # target windows end before
        assert d[f.val[-1]] < f.start <= d[f.eval[0]]


def test_scaler_is_fit_on_the_fit_slice_only():
    d = days()
    X, y = toy(len(d))
    f = SP.make_folds(d, CFG)[0]
    a = SP.fold_arrays(X, y, f)
    X2 = X.copy()
    X2[f.val] += 30
    X2[f.eval] *= 9
    b = SP.fold_arrays(X2, y, f)
    assert np.allclose(a['scaler'].center_, b['scaler'].center_)


def test_lightgbm_with_this_family_config_is_deterministic_and_the_prior_is_the_fit_rate():
    X, y = toy()
    p1 = LG.proba_up(LG.train_lightgbm(X[:800], y[:800], X[800:1000], y[800:1000], CFG['lightgbm']), X[1000:])
    p2 = LG.proba_up(LG.train_lightgbm(X[:800], y[:800], X[800:1000], y[800:1000], CFG['lightgbm']), X[1000:])
    assert np.array_equal(p1, p2)
    assert LG.prior_up(y[:800]) == pytest.approx(y[:800].mean())


def test_mlp_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    X, y = toy(200)
    with pytest.raises(MM.CudaUnavailable):
        MM.train_mlp(X[:150], y[:150], X[150:], y[150:], CFG['mlp'], 11)


def test_artifacts_land_in_this_family_tree(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = dict(c['paths'], results=str(tmp_path / 'r'), artifacts=str(tmp_path / 'a'))
    X, y = toy(300)
    rel = AR.save(c, LG.train_lightgbm(X[:200], y[:200], X[200:], y[200:], c['lightgbm']), 'EURUSD', 0,
                  'lightgbm', 'lightgbm', 'T')
    assert rel.startswith(str(tmp_path / 'a')) and os.path.exists(os.path.join(str(tmp_path / 'r'), 'artifact_manifest.csv'))
    assert CFG['paths']['artifacts'] == 'research_models/fx_rates'
