"""fx-fibonacci-mlp, Module 1b: D1 from H1 and the causal H1 <- D1 join."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_fib import mtf_aligner as AL
from tests.fx_fib_synth import synthetic_weeks

H1 = synthetic_weeks(date(2024, 1, 7), 6, seed=3)


def _ts(s):
    return pd.Timestamp(s, tz='UTC')


def _d1_date_for(label, h1=H1):
    d1 = AL.build_d1(h1)
    return AL.attach_d1(pd.DatetimeIndex([_ts(label)]), d1[['close', 'close_time']])['d1_date'].iloc[0]


# -- task 3.1 -------------------------------------------------------------------

def test_sunday_stub_is_merged_into_monday():
    d1 = AL.build_d1(H1)
    assert not (d1.index.dayofweek == 6).any()
    sunday_bar = H1.loc[_ts('2024-01-07 23:00')]
    monday = d1.loc[_ts('2024-01-08')]
    assert monday['open'] == sunday_bar['open']
    assert monday['high'] >= sunday_bar['high'] and monday['low'] <= sunday_bar['low']
    assert monday['n_h1'] == 25                        # Sun 23:00 + Mon 00:00..23:00
    assert monday['close'] == H1.loc[_ts('2024-01-08 23:00'), 'close']


def test_holiday_reopen_bar_joins_the_next_trading_day():
    # Market closed on Christmas Day; it reopens with one bar labelled 25 Dec 23:00.
    wk = synthetic_weeks(date(2024, 12, 22), 1, seed=4)
    h1 = wk[(wk.index < _ts('2024-12-24 22:00')) | (wk.index >= _ts('2024-12-25 23:00'))]
    d1 = AL.build_d1(h1)
    assert _ts('2024-12-25') not in d1.index
    assert d1.loc[_ts('2024-12-26'), 'open'] == h1.loc[_ts('2024-12-25 23:00'), 'open']
    assert d1.loc[_ts('2024-12-26'), 'n_h1'] == 25


def test_reopen_rule_is_causal():
    # The 23:00 Christmas reopen bar is assigned the same date whether or not later bars exist.
    wk = synthetic_weeks(date(2024, 12, 22), 1, seed=4)
    h1 = wk[(wk.index < _ts('2024-12-24 22:00')) | (wk.index >= _ts('2024-12-25 23:00'))]
    cut = h1.loc[:_ts('2024-12-25 23:00')]
    assert AL.trading_date(cut.index)[-1] == AL.trading_date(h1.index)[len(cut) - 1] == _ts('2024-12-26')


def test_d1_close_is_the_midnight_ending_its_date():
    d1 = AL.build_d1(H1)
    assert (d1['close_time'] == d1.index + pd.Timedelta(days=1)).all()
    assert d1.loc[_ts('2024-01-12'), 'close_time'] == _ts('2024-01-13')     # Friday -> Saturday 00:00


def test_d1_rebuild_is_identical():
    pd.testing.assert_frame_equal(AL.build_d1(H1), AL.build_d1(H1.copy()))


def test_saturday_bars_are_refused():
    bad = pd.concat([H1, H1.iloc[[0]].set_axis([_ts('2024-01-13 10:00')])])
    with pytest.raises(ValueError, match='Saturday'):
        AL.build_d1(bad)


# -- task 3.2 -------------------------------------------------------------------

@pytest.mark.parametrize('label,expected', [
    ('2024-01-10 14:00', '2024-01-09'),     # Wednesday midday -> Tuesday
    ('2024-01-10 22:00', '2024-01-09'),     # closes 23:00, Wednesday still open
    ('2024-01-10 23:00', '2024-01-10'),     # closes at midnight -> Wednesday complete
    ('2024-01-12 22:00', '2024-01-11'),     # Friday's last bar -> Thursday
    ('2024-01-14 23:00', '2024-01-12'),     # week's first bar -> Friday
])
def test_each_h1_bar_sees_only_closed_days(label, expected):
    assert _d1_date_for(label) == _ts(expected)


def test_first_day_has_no_d1_context():
    d1 = AL.build_d1(H1)
    out = AL.attach_d1(H1.index[:5], d1[['close', 'close_time']])
    assert out['close'].isna().all()


def test_altering_future_bars_leaves_past_rows_unchanged():
    t = _ts('2024-01-24 14:00')
    future = H1.copy()
    future.loc[future.index > t, ['open', 'high', 'low', 'close']] *= 1.5
    base = AL.attach_d1(H1.index, AL.build_d1(H1)[['high', 'low', 'close', 'close_time']])
    alt = AL.attach_d1(future.index, AL.build_d1(future)[['high', 'low', 'close', 'close_time']])
    pd.testing.assert_frame_equal(base.loc[:t], alt.loc[:t])
    assert not base.loc[t + pd.Timedelta(days=2):].equals(alt.loc[t + pd.Timedelta(days=2):])


@pytest.mark.parametrize('seed', range(10))
def test_truncated_history_gives_identical_rows(seed):
    t = H1.index[np.random.default_rng(seed).integers(30, len(H1))]
    cut = H1.loc[:t]
    full = AL.attach_d1(H1.index, AL.build_d1(H1)[['open', 'high', 'low', 'close', 'close_time']])
    part = AL.attach_d1(cut.index, AL.build_d1(cut)[['open', 'high', 'low', 'close', 'close_time']])
    pd.testing.assert_frame_equal(full.loc[:t], part)


# -- task 3.3 -------------------------------------------------------------------

@pytest.mark.parametrize('seed', range(5))
def test_atr_is_unchanged_by_truncation_on_both_timeframes(seed):
    t = H1.index[np.random.default_rng(seed).integers(400, len(H1))]
    cut = H1.loc[:t]
    pd.testing.assert_series_equal(AL.atr(H1).loc[:t], AL.atr(cut))
    d1_full, d1_cut = AL.build_d1(H1), AL.build_d1(cut)
    d1_full['atr'], d1_cut['atr'] = AL.atr(d1_full), AL.atr(d1_cut)
    full = AL.attach_d1(H1.index, d1_full[['atr', 'close_time']])
    part = AL.attach_d1(cut.index, d1_cut[['atr', 'close_time']])
    pd.testing.assert_frame_equal(full.loc[:t], part)


def test_atr_has_a_warm_up_of_its_period():
    a = AL.atr(H1, 14)
    assert a.iloc[:13].isna().all() and a.iloc[13:].notna().all()
