"""Volatility, regime and periods (openspec change gold-market-analytics, task 3.3)."""
import math

import numpy as np
import pandas as pd
import pytest

from src.gold_analytics.history import histogram, max_drawdown, period
from src.gold_analytics.volatility import atr, intraday, range_vs_average, realised, regime, true_range


def _d1(closes, highs=None, lows=None, opens=None, start='2025-01-01'):
    n = len(closes)
    t = pd.bdate_range(start, periods=n, tz='UTC')
    c = np.asarray(closes, float)
    return pd.DataFrame({'time': t, 'open': opens if opens is not None else c,
                         'high': highs if highs is not None else c + 1, 'low': lows if lows is not None else c - 1,
                         'close': c})


def test_true_range_and_wilder_atr_by_hand():
    d = _d1([10, 12, 11, 15], highs=[11, 13, 12, 16], lows=[9, 10, 10, 14])
    # TR: 2 | max(3, 3, 0)=3 | max(2, 0, 2)=2 | max(2, 5, 3)=5
    assert true_range(d).tolist() == [2, 3, 2, 5]
    a = atr(d, 3)
    assert np.isnan(a.iloc[1])
    assert a.iloc[2] == pytest.approx(7 / 3)
    assert a.iloc[3] == pytest.approx((7 / 3 * 2 + 5) / 3)


def test_realised_volatility_by_hand():
    d = _d1([100, 101, 99, 100])
    r = np.log([101 / 100, 99 / 101, 100 / 99])
    assert realised(d, 3).iloc[-1] == pytest.approx(np.std(r, ddof=1) * math.sqrt(252) * 100)


def test_regime_above_the_67th_percentile_reads_high():
    rng = np.random.default_rng(1)
    calm = 2000 * np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
    wild = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.03, 20)))
    r = regime(_d1(np.concatenate([calm, wild])), 20, 250)
    assert r['regime'] == 'high' and r['value'] > r['high']
    r_calm = regime(_d1(calm), 20, 250)
    assert r_calm['regime'] in ('low', 'normal', 'high') and r_calm['n'] == 250


def test_regime_reads_low_below_the_33rd():
    rng = np.random.default_rng(2)
    wild = 2000 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    calm = wild[-1] * np.exp(np.cumsum(rng.normal(0, 0.001, 25)))
    assert regime(_d1(np.concatenate([wild, calm])), 20, 250)['regime'] == 'low'


def test_intraday_and_range_against_average():
    m5 = pd.DataFrame({'close': [100.0, 101.0, 100.0]})
    i = intraday(m5)
    so_far = math.sqrt(math.log(1.01) ** 2 + math.log(100 / 101) ** 2) * 100
    assert i['so_far_pct'] == pytest.approx(so_far) and i['day_pct'] == pytest.approx(so_far * math.sqrt(276 / 2))
    d = _d1([100] * 21 + [100], highs=[101] * 21 + [104], lows=[99] * 21 + [100])
    rv = range_vs_average(d, 20)
    assert (rv['today_usd'], rv['average_usd'], rv['ratio']) == (4, 2, 2)


def test_period_drawdown_histogram_and_persistence():
    d = _d1([100, 110, 99, 105, 104, 108])
    p = period(d, days=5, atr_period=3)
    assert p['days'] == 5
    rets = [10.0, -10.0, 105 / 99 * 100 - 100, 104 / 105 * 100 - 100, 108 / 104 * 100 - 100]
    assert p['mean_ret_pct'] == pytest.approx(np.mean(rets))
    assert p['max_drawdown_pct'] == pytest.approx(-10.0)   # 110 -> 99
    assert p['total_ret_pct'] == pytest.approx(8.0)
    # signs + - + - + : no day keeps the previous sign
    assert p['persistence'] == {'n': 4, 'share': 0.0}
    assert max_drawdown(pd.Series([1.0, 2.0, 1.5, 3.0])) == pytest.approx(-25.0)
    h = histogram(pd.Series([0.1, 0.2, -0.3, 0.6]), 0.25)
    assert [(b['from'], b['to'], b['count']) for b in h] == [(-0.5, -0.25, 1), (-0.25, 0.0, 0), (0.0, 0.25, 2),
                                                             (0.25, 0.5, 0), (0.5, 0.75, 1)]


def test_a_custom_range_uses_only_its_days():
    d = _d1(list(range(100, 400)), start='2024-10-01')
    p = period(d, start='2025-01-01', end='2025-06-30')
    inside = d[(d['time'] >= '2025-01-01') & (d['time'] <= '2025-06-30 23:59')]
    assert p['days'] == len(inside)
    assert p['first'] == inside['time'].iloc[0] and p['last'] == inside['time'].iloc[-1]
