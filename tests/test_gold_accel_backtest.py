"""Scoring and report of the acceleration study (openspec change gold-acceleration-direction, task 3.1)."""
import numpy as np
import pandas as pd
import pytest

from src.gold import targets as G
from src.gold_accel import load_config
from src.gold_accel.backtest import block_bootstrap_mean, evaluate, reading, render, scored_frame

CFG = load_config()


def _h1(n_days=60):
    idx = pd.date_range('2024-01-01', periods=24 * n_days, freq='h', tz='UTC')
    idx = idx[idx.weekday < 5]
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.002, len(idx))))
    return pd.DataFrame({'open': c, 'high': c, 'low': c, 'close': c}, index=idx)


def _pred(h1, p):
    return pd.DataFrame({'time': h1.index, 'p_up': p})


def _perfect(h1):
    d = G.build_targets(h1['close'])['dir']
    return np.where(np.isnan(d), 0.5, np.where(d > 0.5, 0.9, 0.1))


def _noise(h1, seed):
    return np.random.default_rng(seed).uniform(0.3, 0.7, len(h1))


SPREAD = pd.Series(dtype=float)


def test_identical_models_differ_by_exactly_zero():
    h1 = _h1()
    p = _noise(h1, 1)
    res = evaluate({'h1_gbm': _pred(h1, p), 'h1_gbm_accel': _pred(h1, p)}, h1, SPREAD, CFG)
    r = res['pairs'][('h1_gbm_accel', 'all')]
    assert r['diff'] == 0.0 and r['ci'] == (0.0, 0.0)
    assert reading(r['ci']) == 'not distinguishable from zero'


def test_a_perfect_model_hits_every_row():
    h1 = _h1()
    res = evaluate({'h1_gbm': _pred(h1, _perfect(h1))}, h1, SPREAD, CFG)
    m = res['metrics']['all']['h1_gbm']
    assert m['hit'] == 1.0 and m['auc'] == 1.0 and m['vs_coin'][0] == pytest.approx(0.5)
    # trading the perfect sign gross of a zero spread still pays the cost floor
    c = res['costs']['h1_gbm']['floor']
    assert c['trades'] > 0 and c['win_share'] > 0.9
    assert c['mean_cost_pct'] == pytest.approx(CFG['costs']['floor_round_trip_pct'])


def test_the_window_view_uses_only_as_of_labels_14_to_20():
    h1 = _h1()
    df = scored_frame(_pred(h1, _noise(h1, 2)), h1, CFG)
    assert set(df[df['window']].index.hour) == set(range(14, 21))
    assert not df['window'][df.index.hour == 21].any()


def test_an_interval_holding_zero_is_worded_as_not_distinguishable():
    assert reading((-0.01, 0.02)) == 'not distinguishable from zero'
    assert reading((0.001, 0.02)).startswith('interval above zero (descriptive')
    m, lo, hi = block_bootstrap_mean(np.r_[np.ones(50), np.zeros(50)], 10, 500, 0)
    assert m == 0.5 and lo < 0.5 < hi


def test_the_report_has_no_trade_or_edge_wording():
    h1 = _h1()
    preds = {'h1_gbm': _pred(h1, _noise(h1, 3)), 'h1_gbm_accel': _pred(h1, _noise(h1, 4)),
             'h1_accel': _pred(h1, _noise(h1, 5))}
    res = evaluate(preds, h1, SPREAD, CFG)
    meta = {'inputs': {'h1': {'rows': 1, 'first': 'a', 'last': 'b', 'sha256': '0' * 64}},
            'reuse': {'reason': 'refit'}, 'years': {m: '2024-2024 ok' for m in preds}}
    md = render(res, CFG, meta)
    low = md.lower()
    for word in ('buy', 'sell', 'edge', 'improve', 'купи', 'продай'):
        assert word not in low, word
    assert md.count(CFG['label']) >= 6           # the header and every section
