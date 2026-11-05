"""Time-of-day statistics (openspec change gold-market-analytics, task 3.2)."""
import numpy as np
import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.intraday import (extreme_times, momentum, resample, spread_by_slot, time_of_day,
                                         weighted_median)

WINDOW = load_config()['owner_window']


def _m1(day, n=600, start='08:00', price=2000.0):
    t = pd.date_range(f'{day} {start}', periods=n, freq='min', tz='UTC')
    c = price + np.arange(n) * 0.1
    return pd.DataFrame({'time': t, 'open': c - 0.05, 'high': c + 0.2, 'low': c - 0.2, 'close': c})


def test_the_60_minute_spread_aggregates_the_5_minute_rows():
    slots = pd.date_range('2026-11-03 14:00', periods=12, freq='5min', tz='UTC')  # 15:00-16:00 Sofia
    med = [0.30, 0.30, 0.30, 0.40, 0.40, 0.40, 0.50, 0.50, 0.50, 0.60, 0.60, 0.90]
    ticks = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 200]
    sp = pd.DataFrame({'day': '2026-11-03', 'slot': slots, 'median': med, 'p90': np.array(med) + 0.2,
                       'ticks': ticks})
    out = spread_by_slot(sp, 60)
    assert out['minute'].tolist() == [15 * 60]
    # tick weights: 310 in total, half = 155: the 200-tick slot at 0.90 holds the middle
    assert out['spread_median'].item() == pytest.approx(0.90)
    assert out['spread_p90'].item() == pytest.approx(1.10)
    assert weighted_median([1, 2, 3], [1, 1, 1]) == 2
    five = spread_by_slot(sp, 5)
    assert len(five) == 12 and five['spread_median'].iloc[0] == pytest.approx(0.30)


def test_a_1_minute_request_beyond_m1_states_the_days_used():
    bars = pd.concat([_m1('2026-11-02'), _m1('2026-11-03'), _m1('2026-11-04')], ignore_index=True)
    out = time_of_day(bars, 1, WINDOW, days_requested=250)
    assert (out['days_used'], out['days_requested'], out['limited']) == (3, 250, True)
    assert out['first_bar'] == pd.Timestamp('2026-11-02 08:00', tz='UTC')
    assert out['slots'][0]['slot'] == '09:00'   # 08:00 label = 09:00 Sofia


def test_slot_means_acceleration_and_the_window_flag():
    bars = resample(_m1('2026-11-03', n=900), 60)  # 08:00-22:59 label, 09:00-23:59 Sofia
    out = time_of_day(bars, 60, WINDOW)
    rows = {r['slot']: r for r in out['slots']}
    r = rows['16:00']
    assert r['n'] == 1 and r['ret_mean'] == pytest.approx((5.9 + 0.05) / (bars['open'].iloc[7]) * 100, rel=1e-6)
    assert r['accel'] == pytest.approx(rows['16:00']['abs_mean'] - rows['15:00']['abs_mean'])
    assert not rows['15:00']['in_window']   # the window opens at 15:30
    assert rows['16:00']['in_window'] and rows['22:00']['in_window'] and not rows['23:00']['in_window']


def test_extreme_times_and_momentum():
    up = _m1('2026-11-03', n=900)                       # rising all day: high last, low first
    out = extreme_times(resample(up, 60), 60)
    assert out.loc[out['high_share'] > 0, 'minute'].tolist() == [23 * 60]
    assert out.loc[out['low_share'] > 0, 'minute'].tolist() == [9 * 60]
    days = [resample(_m1(f'2026-11-0{d}', n=900, price=2000 + 50 * d), 15) for d in (2, 3, 4, 5)]
    mo = momentum(pd.concat(days, ignore_index=True), WINDOW)
    assert mo['n'] == 4


def test_4h_bins_start_at_sofia_midnight():
    h1 = resample(_m1('2026-11-03', n=24 * 60, start='00:00'), 60)
    h4 = resample(h1, 240, offset_hours=1)
    # label 23:00 = 00:00 Sofia, so the bins are 03:00, 07:00, ... on the label clock
    assert [t.hour for t in h4['time']] == [23, 3, 7, 11, 15, 19, 23]  # the last label 23:00 opens 4 Nov Sofia
    first = h4.iloc[1]
    src = h1[(h1['time'] >= '2026-11-03 03:00') & (h1['time'] < '2026-11-03 07:00')]
    assert (first['open'], first['high'], first['low'], first['close']) == (
        src['open'].iloc[0], src['high'].max(), src['low'].min(), src['close'].iloc[-1])
