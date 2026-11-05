"""Session statistics (openspec change gold-market-analytics, task 3.1)."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.sessions import (intervals_for, now_session, overnight_against_day, session_summary,
                                         session_table, trading_days, wilson)

CFG = load_config()


def _bars(rows):
    """rows of (label, open, high, low, close)."""
    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close'])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df


def test_session_table_hand_values():
    iv = pd.DataFrame({'day': [date(2026, 1, 14)],
                       'start': [pd.Timestamp('2026-01-14 09:00', tz='UTC')],
                       'end': [pd.Timestamp('2026-01-14 11:00', tz='UTC')]})
    bars = _bars([('2026-01-14 08:00', 1, 1, 1, 1),          # before the session
                  ('2026-01-14 09:00', 100, 103, 99, 102),
                  ('2026-01-14 10:00', 102, 104, 98, 101),
                  ('2026-01-14 11:00', 101, 120, 90, 110)])  # the end is exclusive
    t = session_table(bars, iv).iloc[0]
    assert (t['open'], t['close'], t['high'], t['low'], t['n_bars']) == (100, 101, 104, 98, 2)
    assert t['range_usd'] == 6 and t['range_pct'] == pytest.approx(6.0)
    assert t['ret_pct'] == pytest.approx(1.0)
    assert t['rv_pct'] == pytest.approx(math.sqrt(math.log(1.02) ** 2 + math.log(101 / 102) ** 2) * 100)


def _table(rets, highs=None, lows=None):
    n = len(rets)
    return pd.DataFrame({'ret_pct': rets, 'range_usd': 1.0, 'range_pct': 0.1, 'rv_pct': 0.2,
                         'high': highs if highs is not None else np.full(n, 10.0),
                         'low': lows if lows is not None else np.full(n, 5.0)})


def test_33_of_60_up_reads_55_percent_with_wilson_bounds():
    s = session_summary(_table([0.1] * 33 + [-0.1] * 27))
    b = s['bias_up']
    assert (b['n'], b['hits']) == (60, 33) and b['share'] == pytest.approx(0.55)
    z = 1.959963984540054
    p, n = 0.55, 60
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert (b['ci_low'], b['ci_high']) == (pytest.approx(centre - half), pytest.approx(centre + half))
    assert b['ci_low'] == pytest.approx(0.4249, abs=1e-4) and b['ci_high'] == pytest.approx(0.6691, abs=1e-4)
    assert b['ci_low'] < 0.5 < b['ci_high']   # the interval holds a coin: no edge


def test_breakout_and_persistence():
    t = _table([0.1, 0.2, -0.1, -0.3, 0.0, 0.4],
               highs=[10, 11, 10.5, 10.5, 12, 11], lows=[5, 5, 5, 4, 4, 4])
    s = session_summary(t)
    # breakouts vs the previous row: 11>10 yes; 10.5 no/5 no; 10.5 no but 4<5 yes; 12>10.5 yes; no -> 3 of 5
    assert (s['breakout']['hits'], s['breakout']['n']) == (3, 5)
    # sign pairs with no zero: (+,+) same, (+,-) diff, (-,-) same; pairs touching the 0.0 are skipped
    assert (s['persistence']['hits'], s['persistence']['n']) == (2, 3)
    assert wilson(0, 0) == (None, None)


def test_overnight_and_day_split_the_clock():
    days = trading_days(date(2026, 1, 12), date(2026, 1, 16))
    labels = pd.date_range('2026-01-12 00:00', '2026-01-17 00:00', freq='h', tz='UTC')
    bars = _bars([(t, 100 + i, 100.5 + i, 99.5 + i, 101 + i) for i, t in enumerate(labels)])
    od = overnight_against_day(bars, days, CFG)
    assert od['day']['n'] == 5 and od['overnight']['n'] == 4
    # day = London 08:00 (09:00 label) to NY 13:30 (19:30 label); hourly bars give 09:00..19:00
    assert od['day']['mean_ret_pct'] > 0 and od['overnight']['bias_up']['share'] == 1.0


def test_now_session_and_the_next_one():
    # 2026-01-14 12:00 label = 13:00 Sofia: London is open, New York opens at 14:20 label.
    s = now_session('2026-01-14 12:00', CFG)
    assert s['current'] == ['london'] and s['next']['key'] == 'new_york'
    assert s['next']['start_label'] == pd.Timestamp('2026-01-14 14:20', tz='UTC')
    # Saturday: nothing runs; the next is Asia on Monday (09:00 Tokyo = 01:00 label).
    s = now_session('2026-01-17 12:00', CFG)
    assert s['current'] == [] and s['next'] == {'key': 'asia', 'start_label': pd.Timestamp('2026-01-19 01:00', tz='UTC')}
    assert len(intervals_for('overlap', [date(2026, 1, 14)], CFG)) == 1
