"""fx-bb-reversion, groups 5.2-5.3: the pooled LightGBM wiring and the day-block ratio bootstrap."""
import numpy as np
import pandas as pd
import pytest

from src.fx_bb import load_config
from src.fx_bb import score as SC
from src.fx_bb import walk_forward as WF
from src.fx_fib.score import circular_block_means
from src.fx_session.splits import Fold

CFG = load_config()


def test_one_event_every_day_reproduces_the_project_bootstrap_exactly():
    x = np.random.default_rng(1).normal(0.01, 1.0, 997)
    a = SC.ratio_block_means(x, np.ones_like(x), 5, 10000, 20261124)
    b = circular_block_means(x, 5, 10000, 20261124)
    assert np.array_equal(a, b)
    a2 = SC.ratio_block_means(x[:1000 - 5], np.ones(995), 5, 1234, 7)       # no partial block
    assert np.array_equal(a2, circular_block_means(x[:995], 5, 1234, 7))


def _events(mean, n_days=1500, seed=0, sd=1.1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2012-03-01', periods=n_days)
    k = rng.choice(n_days, size=n_days // 3, replace=True)        # ~1 event every 3 days, some days several
    dates = days[np.sort(k)]
    return dates, rng.normal(mean, sd, len(dates))


def test_planted_positive_mean_is_keep_and_zero_mean_is_drop_for_cell_1():
    sc = CFG['scoring']
    for mean, verdict in ((0.35, 'KEEP'), (0.0, 'DROP')):
        dates, r = _events(mean)
        S, N = SC.day_sums(dates, r)
        res = SC.test_ratio(S, N, sc['alpha'], sc['block_days'], 2000, sc['seed'], 'greater')
        assert res['verdict'] == verdict, (mean, res)
        assert res['statistic'] == pytest.approx(r.mean())


def test_rule_verdict_is_drop_when_the_bound_is_below_zero():
    dates, r = _events(0.0, n_days=600, seed=3)
    r = r - r.mean() + 0.02                                   # mean exactly +0.02 ATR, as in the spec scenario
    S, N = SC.day_sums(dates, r)
    res = SC.test_ratio(S, N, 0.025, 5, 2000, 1, 'greater')
    assert res['statistic'] == pytest.approx(0.02) and res['bound'] < 0
    assert res['verdict'] == 'DROP'


def test_cell_2_side_keeps_only_when_the_upper_bound_is_below_zero():
    dates, d = _events(-0.05, seed=5, sd=0.1)
    S, N = SC.day_sums(dates, d)
    assert SC.test_ratio(S, N, 0.025, 5, 2000, 1, 'less')['verdict'] == 'KEEP'
    S, N = SC.day_sums(dates, d + 0.06)
    assert SC.test_ratio(S, N, 0.025, 5, 2000, 1, 'less')['verdict'] == 'DROP'


def test_day_sums_counts_events_per_business_day_and_refuses_weekends():
    dates = pd.to_datetime(['2015-01-05', '2015-01-05', '2015-01-07'])
    S, N = SC.day_sums(dates, [1.0, 2.0, -1.0])
    assert list(N) == [2, 0, 1] and list(S) == [3, 0, -1]
    with pytest.raises(ValueError, match='weekend'):
        SC.day_sums(pd.to_datetime(['2015-01-04']), [1.0])


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(CFG['features']['columns'])))
    y = (X[:, 0] + rng.normal(0, 1.5, n) > 0).astype(float)
    return X, y


def test_same_seed_lightgbm_is_identical_and_base_rate_is_the_fit_up_rate():
    X, y = _toy()
    f = Fold(0, pd.Timestamp('2015-01-01'), pd.Timestamp('2016-01-01'), np.arange(0, 400), np.arange(410, 480),
             np.arange(500, 600))
    m1, prior1, p1 = WF.fit_fold(X, y, f, CFG)
    m2, prior2, p2 = WF.fit_fold(X, y, f, CFG)
    assert np.array_equal(p1, p2) and prior1 == prior2
    assert prior1 == pytest.approx(y[:400].mean()) and prior1 != pytest.approx(y[410:480].mean())
    assert m1.get_params()['num_leaves'] == 4 and 'class_weight' not in {k for k, v in m1.get_params().items() if v}
