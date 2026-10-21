"""Gold targets and H1 eligibility (task 3.2; spec "Targets are next-bar return,
direction and volatility", "H1 eligibility excludes bars that border a market gap")."""
import numpy as np
import pandas as pd
import pytest

from src.gold import targets as G


def _week(monday='2024-01-15', break_hour=23):
    """Mon 00:00 .. Fri 22:00 H1 labels with a daily break at `break_hour`."""
    m = pd.Timestamp(monday, tz='UTC')
    idx = pd.date_range(m, m + pd.Timedelta(days=4, hours=22), freq='h')
    return idx[idx.hour != break_hour]


def _pos(idx, ts):
    return idx.get_loc(pd.Timestamp(ts, tz='UTC'))


def test_last_bar_before_the_daily_break_is_ineligible():
    idx = _week()
    ok = G.h1_eligible(idx)
    assert not ok[_pos(idx, '2024-01-16 22:00')]           # target would be 00:00 after the break
    assert not ok[_pos(idx, '2024-01-17 00:00')]           # first bar after the break as as-of


def test_first_bar_of_the_week_is_ineligible_as_target_and_as_of():
    idx = _week().append(_week('2024-01-22'))
    ok = G.h1_eligible(idx)
    assert not ok[_pos(idx, '2024-01-19 22:00')]           # Fri 22:00 -> Mon 00:00
    assert not ok[_pos(idx, '2024-01-22 00:00')]


def test_dst_mismatch_week_break_moves_and_the_rule_follows():
    idx = _week('2024-03-11', break_hour=22)
    ok = G.h1_eligible(idx)
    assert not ok[_pos(idx, '2024-03-12 21:00')]
    assert not ok[_pos(idx, '2024-03-12 23:00')]
    assert ok[_pos(idx, '2024-03-12 20:00')]                # an ordinary hour stays in


def test_ordinary_mid_session_hour_is_eligible():
    idx = _week()
    assert G.h1_eligible(idx)[_pos(idx, '2024-01-17 10:00')]


def test_targets_read_only_the_next_bar():
    close = pd.Series([100.0, 101.0, 100.5, 100.5, 102.0])
    t = G.build_targets(close)
    assert t['ret'][0] == pytest.approx(100 * np.log(1.01))
    assert t['dir'][0] == 1 and t['dir'][1] == 0
    assert np.isnan(t['dir'][2])                             # zero move: no class
    assert t['vol'][1] == pytest.approx(abs(t['ret'][1]))
    assert np.isnan(t['ret'][-1])
    altered = close.copy()
    altered.iloc[3:] = 999.0
    assert G.build_targets(altered)['ret'][0] == t['ret'][0]


def test_price_is_derived_from_the_return():
    assert G.implied_price(4000.0, 0.10) == pytest.approx(4000.0 * np.exp(0.001))
    assert round(G.implied_price(4131.0, 0.04), 2) == 4132.65
