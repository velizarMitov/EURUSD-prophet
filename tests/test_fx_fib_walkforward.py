"""fx-fibonacci-mlp, group 8: metrics, the gross rule and the walk-forward run."""
import copy
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_fib import gross as G
from src.fx_fib import load_config
from src.fx_fib import metrics as MT
from src.fx_fib import walk_forward_backtest as WF
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()


# -- metrics ---------------------------------------------------------------------

def test_log_loss_brier_and_f1_by_hand():
    y = np.array([0, 1, 2, 0])
    p = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]])
    assert MT.log_loss(y, p) == pytest.approx(-np.mean(np.log([0.8, 0.7, 0.4, 0.5])))
    onehot = np.eye(3)[y]
    assert MT.brier(y, p) == pytest.approx(((p - onehot) ** 2).sum(axis=1).mean())
    assert MT.macro_f1(y, p) == pytest.approx(1.0)


def test_prior_baseline_uses_train_fold_frequencies():
    p = MT.prior_proba(np.array([0, 0, 0, 1, 2, 2]), 4)
    assert p.shape == (4, 3) and np.allclose(p[0], [0.5, 1 / 6, 1 / 3])


def test_ece_of_a_perfectly_calibrated_bin_is_zero():
    y = np.array([0, 0, 0, 1])                  # 75 % accuracy at 0.75 confidence
    p = np.tile([0.75, 0.125, 0.125], (4, 1))
    assert MT.ece(y, p) == pytest.approx(0.0)
    r = MT.reliability(y, p)
    assert r['n'].sum() == 4


# -- the gross rule -----------------------------------------------------------------

def _rows(n=10):
    t = pd.date_range('2024-03-04', periods=n, freq='h', tz='UTC')
    close = np.full(n, 100.0)
    return pd.DataFrame({'time': t, 'pos': np.arange(n), 'close': close,
                         'label': [1, 2, 0, 1, 0, 2, 0, 0, 1, 0], 'exit_offset': [2, 1, 3, 1, 1, 2, 1, 1, 1, 1],
                         'upper': close + 1, 'lower': close - 1,
                         'exit_close': [100, 100, 100.5, 100, 99.8, 100, 100, 100, 100, 100.2],
                         'exit_day': t.normalize()})


def test_gross_rule_by_hand_with_a_skipped_overlapping_signal():
    rows = _rows()
    pred = np.array([1, 1, 2, 0, 1, 2, 0, 0, 2, 0])
    tr, skipped = G.trades(rows, pred)
    # bar 0 long -> upper (+1 %), busy until pos 2; bar 1 is skipped; bar 2 short, time-out at
    # 100.5 (-0.5 %), busy until 5; bar 4 skipped; bar 5 short -> lower (+1 %); bar 8 short -> upper (-1 %).
    assert skipped == 2
    assert list(tr['ret_pct'].round(6)) == [1.0, -0.5, 1.0, -1.0]
    s = G.stats(tr, skipped, rows['exit_day'])
    assert s['trades'] == 4 and s['win_rate'] == 0.5
    assert s['profit_factor'] == pytest.approx(2.0 / 1.5)
    assert s['expectancy_pct'] == pytest.approx(0.125)
    assert s['max_drawdown_pct'] >= 0


def test_no_signal_means_no_trades():
    s = G.stats(*G.trades(_rows(), np.zeros(10, int)), _rows()['exit_day'])
    assert s['trades'] == 0


# -- the walk-forward run --------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_walk_forward_smoke_run_on_the_gpu(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg['paths'] = {'data': str(tmp_path / 'd'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    cfg['folds'].update(first_eval_start='2024-01-01', fold_months=3)
    cfg['mlp'].update(max_epochs=3, patience=2)
    cfg['lightgbm'].update(n_estimators=30, early_stopping_rounds=5)
    frames = {'EURUSD': synthetic_weeks(date(2023, 1, 1), 60, seed=31, vol=0.0015),
              'GBPUSD': synthetic_weeks(date(2023, 1, 1), 60, seed=32, vol=0.0015)}
    import src.fx_fib.data_loader as DL
    monkeypatch.setattr(DL, 'load_h1', lambda pair, cfg=None: frames[pair])
    out = WF.run_walk_forward(cfg, 'T', max_folds=1, log=lambda *a: None)
    oos, run = out['oos'], out['run']
    assert set(oos['pair']) == {'EURUSD', 'GBPUSD'} and (oos['time'] >= pd.Timestamp('2024-01-01', tz='UTC')).all()
    for m in ('prior',) + WF.MODELS:
        p = oos[[f'p_{m}_{c}' for c in range(3)]].to_numpy()
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5), m
    saved = json.load(open(os.path.join(cfg['paths']['results'], WF.RUN_FILE)))
    assert 'NVIDIA' in saved['device'] or 'GeForce' in saved['device']
    assert saved['config_digest'] == run['config_digest'] and len(saved['pairs']['EURUSD']['folds']) == 1
    back = WF.load_oos(cfg)
    assert len(back) == len(oos)
    assert os.path.exists(os.path.join(cfg['paths']['artifacts'], 'EURUSD', 'fold_00', 'mlp_seed11.pt'))
