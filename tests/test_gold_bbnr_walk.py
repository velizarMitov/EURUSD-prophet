"""gold-bb-nyrange, tasks 7.1-7.2: the gated walk-forwards on synthetic data."""
import copy
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_nyvol import models as M
from src.gold_bbnr import bars as B
from src.gold_bbnr import load_config
from src.gold_bbnr import ny_target as NT
from src.gold_bbnr import walk_forward as W
from src.gold_bbnr.inputs import Inputs
from src.gold_bbnr.splits import event_folds
from tests.test_gold_bbnr_ny import dxy_h1, gold_h1

CFG = load_config()


class Refused(RuntimeError):
    pass


def no_gate(cfg):
    return 'T'


def refuse(cfg):
    raise Refused('not committed')


def inputs(n_days=700, seed=7):
    h1 = gold_h1(n_days, seed=seed)
    dd = pd.bdate_range('2018-06-01', '2022-12-31')
    v = 95 + np.cumsum(np.random.default_rng(2).normal(0, 0.3, len(dd)))
    dxy_d1 = pd.DataFrame({'open': v, 'high': v, 'low': v, 'close': v}, index=pd.DatetimeIndex(dd, name='date'))
    return Inputs(h1, B.d1_ny(h1, CFG), dxy_d1, dxy_h1(h1))


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    c['folds_bb'].update(first_eval_start='2020-01-01', min_train=8)
    c['folds_ny'].update(first_eval_start='2019-10-01')
    c['mlp'].update(seeds=[11, 22], max_epochs=4, patience=2)
    return c


INP = inputs()


def test_every_part_refuses_before_the_gate(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('no outcome before the gate')
    monkeypatch.setattr(NT, 'target', boom)
    for part in ('bb', 'bb_sma', 'nyvol'):
        with pytest.raises(Refused):
            W.run(part, cfg, INP, gate=refuse, log=lambda *a: None)
    assert not os.path.exists(cfg['paths']['results']) or not os.listdir(cfg['paths']['results'])


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def test_bb_parts_run_purge_and_are_deterministic(cfg):
    for part in ('bb', 'bb_sma'):
        rec = W.run(part, cfg, INP, gate=no_gate, log=lambda *a: None)
        assert rec['n_model'] > 0 and rec['folds']
        files = [os.path.join(cfg['paths']['results'], f) for f in W.FILES[part]]
        first = [_sha(p) for p in files]
        W.run(part, cfg, INP, gate=no_gate, log=lambda *a: None)
        assert [_sha(p) for p in files] == first
    ev = W.load_csv(cfg, W.FILES['bb'][0])
    use = ev[ev['reason2'].fillna('') == ''].sort_values('date').reset_index(drop=True)
    folds, _ = event_folds(use['date'], use['date_t3'], cfg)
    for f in folds:
        gap = np.busday_count(use['date_t3'].iloc[np.r_[f.fit, f.val]].to_numpy().astype('datetime64[D]'),
                              np.datetime64(f.start.date()))
        assert (gap > 10).all()
    tr = W.load_csv(cfg, W.FILES['bb_sma'][0])
    assert set(tr['exit_reason'].dropna()) <= {'stop', 'take profit', 'time stop'}


def test_a_thin_fold_predicts_nothing(cfg):
    c = copy.deepcopy(cfg)
    c['folds_bb']['min_train'] = 10_000
    rec = W.run('bb', c, INP, gate=no_gate, log=lambda *a: None)
    assert rec['n_model'] == 0
    ev = W.load_csv(c, W.FILES['bb'][0])
    assert set(ev.loc[ev['date'] >= '2020-01-01', 'fold_reason'].dropna()) <= {'thin fold', 'warm-up', 'no outcome'}


def test_nyvol_refuses_without_cuda_before_any_target(cfg, monkeypatch):
    def no_cuda(allow_cpu=False):
        raise M.CudaUnavailable('no CUDA')

    def boom(*a, **k):
        raise AssertionError('target read')
    monkeypatch.setattr(M, 'pick_device', no_cuda)
    monkeypatch.setattr(NT, 'target', boom)
    with pytest.raises(M.CudaUnavailable):
        W.run('nyvol', cfg, INP, gate=no_gate, log=lambda *a: None)


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='CUDA required for the MLP')
def test_nyvol_folds_and_harx_is_ols_on_fourteen(cfg):
    rec = W.run('nyvol', cfg, INP, gate=no_gate, log=lambda *a: None)
    assert rec['scored_days'] > 0
    for f in rec['folds']:
        assert pd.Timestamp(f['last_train_day']) < pd.Timestamp(f['start'])
    d = W.load_csv(cfg, W.FILES['nyvol'][1])
    from src.gold_bbnr import ny_features as NF
    from src.gold_bbnr.splits import day_folds
    ds = NF.dataset(INP.h1, INP.d1, INP.dxy_h1, cfg)
    rows = ds[ds['reason'] == ''].sort_index()
    f0 = day_folds(rows.index, cfg)[0]
    tr = np.r_[f0.fit, f0.val]
    X = rows[cfg['features_ny']['columns']].to_numpy(float)
    coef = M.ols_fit(X[tr], rows['y'].to_numpy(float)[tr])
    exp = M.ols_predict(coef, X[f0.eval])
    got = d[d['fold'] == f0.k]['p_harx'].to_numpy()
    np.testing.assert_allclose(got, exp, rtol=1e-10)
