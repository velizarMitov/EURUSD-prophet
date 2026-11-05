"""Acceleration features (openspec change gold-acceleration-direction, task 1.2)."""
import numpy as np
import pandas as pd
import pytest

from src.gold import features as GF
from src.gold_accel import load_config
from src.gold_accel.features import build_inputs, columns_for, feature_frame, h1_accel, m15_accel

CFG = load_config()
ACCEL = ['accel_1h', 'activity_rel', 'accel_expected_next', 'accel_m15']


def _h1(n=24 * 90, seed=0):
    idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({'open': o, 'high': np.maximum(o, c) * 1.001, 'low': np.minimum(o, c) * 0.999,
                         'close': c}, index=idx)


def _m15(h1, seed=1):
    idx = pd.date_range(h1.index[0], h1.index[-1] + pd.Timedelta(minutes=45), freq='15min', tz='UTC')
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.001, len(idx))))
    return pd.DataFrame({'open': c, 'high': c, 'low': c, 'close': c}, index=idx)


def test_no_look_ahead():
    h1 = _h1()
    m15 = _m15(h1)
    t = h1.index[24 * 70]
    before = feature_frame(h1, m15).loc[t, ACCEL]
    h1b, m15b = h1.copy(), m15.copy()
    h1b.loc[h1b.index > t, ['open', 'high', 'low', 'close']] *= 1.5
    m15b.loc[m15b.index >= t + pd.Timedelta(hours=1), 'close'] *= 0.7
    after = feature_frame(h1b, m15b).loc[t, ACCEL]
    assert np.isfinite(before.to_numpy()).all()
    pd.testing.assert_series_equal(before, after)


def test_scale_free():
    h1 = _h1()
    m15 = _m15(h1)
    a = feature_frame(h1, m15)[ACCEL]
    b = feature_frame(h1 * 3, m15 * 3)[ACCEL]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_activity_rel_by_hand():
    # Hour 10 moves 0.20 % on each of 60 days, then 0.40 % on day 61; every other hour 0.
    days = 61
    idx = pd.date_range('2024-01-01', periods=24 * days, freq='h', tz='UTC')
    r = np.zeros(len(idx))
    r[idx.hour == 10] = 0.20
    r[-14] = 0.40                     # day 61, 10:00
    close = 2000 * np.exp(np.cumsum(r) / 100)
    h1 = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)
    f = h1_accel(h1, 60)
    t = idx[-14]
    assert t.hour == 10
    assert f.loc[t, 'activity_rel'] == pytest.approx(2.0)
    assert f.loc[t, 'accel_1h'] == pytest.approx(0.40)
    # the next hour (11:00) averaged 0 and 10:00 averaged 0.20
    assert f.loc[t - pd.Timedelta(hours=1), 'accel_expected_next'] == pytest.approx(0.20)


def test_accel_m15_by_hand_and_nan_when_a_bar_is_missing():
    t0 = pd.Timestamp('2024-03-04 14:00', tz='UTC')
    idx = pd.date_range(t0 - pd.Timedelta(minutes=15), periods=5, freq='15min')
    moves = [0.0, 0.10, -0.10, 0.10, 0.40]           # % per bar; the first is the bar before t0
    close = 2000 * np.exp(np.cumsum(moves) / 100)
    m15 = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)
    got = m15_accel(pd.DatetimeIndex([t0]), m15)
    assert got.iloc[0] == pytest.approx(0.40 - 0.10)
    assert np.isnan(m15_accel(pd.DatetimeIndex([t0]), m15.drop(idx[2])).iloc[0])


def test_columns_are_scale_free_and_inputs_match():
    for model in CFG['models']:
        cols = columns_for(model, CFG)
        assert GF.assert_scale_free(cols) == cols
    assert columns_for('h1_gbm', CFG) == GF.h1_feature_columns()
    assert columns_for('h1_gbm_accel', CFG) == GF.h1_feature_columns() + CFG['features']['h1']
    assert columns_for('h1_accel_m15', CFG) == CFG['features']['h1'] + ['accel_m15'] + CFG['features']['calendar']
    h1 = _h1()
    frame = feature_frame(h1, None)
    inp = build_inputs(h1, frame, columns_for('h1_accel', CFG))
    assert inp.X.shape == (len(h1), 7)
    assert inp.extra['complete'].sum() > 0 and not inp.extra['complete'][:24 * 60].any()
