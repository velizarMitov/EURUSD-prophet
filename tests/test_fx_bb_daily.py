"""fx-bb-reversion, group 2.2: D1 bars closing at 17:00 New York, from the Berlin labels."""
from datetime import date

import numpy as np
import pandas as pd

from src.fx_bb import daily as D
from src.fx_bb import load_config
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
PT, PIP = 0.00001, 0.0001


def h1(first=date(2012, 1, 1), n_weeks=2):
    f = synthetic_weeks(first, n_weeks, seed=3)
    f['spread'] = 8.0
    return f


def test_sunday_reopen_belongs_to_monday_and_no_sunday_day_exists():
    f = h1()
    td = D.trading_dates(f.index, CFG)
    sunday_open = pd.Timestamp('2012-01-01 23:00', tz='UTC')
    assert td[f.index.get_loc(sunday_open)] == pd.Timestamp('2012-01-02')
    d1 = D.d1_ny(f, PT, PIP, CFG)
    assert (d1.index.dayofweek < 5).all() and pd.Timestamp('2012-01-01') not in d1.index
    assert d1.loc['2012-01-02', 'open'] == f.loc[sunday_open, 'open']


def test_wednesday_closes_on_the_22_bar_and_the_23_rollover_opens_thursday():
    f = h1()
    d1 = D.d1_ny(f, PT, PIP, CFG)
    wed = pd.Timestamp('2012-01-04')
    bar22 = pd.Timestamp('2012-01-04 22:00', tz='UTC')
    bar23 = pd.Timestamp('2012-01-04 23:00', tz='UTC')
    assert d1.loc[wed, 'close_bid'] == f.loc[bar22, 'close']
    assert d1.loc[wed, 'close'] == f.loc[bar22, 'close'] + 8.0 * PT / 2
    assert d1.loc[pd.Timestamp('2012-01-05'), 'open'] == f.loc[bar23, 'open']
    assert d1.loc[wed, 'n_h1'] == 24


def test_dst_mismatch_week_closes_on_the_21_bar():
    f = h1(first=date(2026, 3, 8), n_weeks=1)              # US on summer time, Berlin not yet
    d1 = D.d1_ny(f, PT, PIP, CFG)
    wed = pd.Timestamp('2026-03-11')
    assert d1.loc[wed, 'close_bid'] == f.loc[pd.Timestamp('2026-03-11 21:00', tz='UTC'), 'close']
    assert d1.loc[pd.Timestamp('2026-03-12'), 'open'] == f.loc[pd.Timestamp('2026-03-11 22:00', tz='UTC'), 'open']
    assert (d1.index.dayofweek < 5).all()                   # Friday's 22:00 bar is a dropped Saturday stub


def test_short_day_is_dropped():
    f = h1()
    thu = (f.index >= pd.Timestamp('2012-01-05 00:00', tz='UTC')) & (f.index < pd.Timestamp('2012-01-05 15:00', tz='UTC'))
    f = f[~thu]                                            # Thursday keeps 23:00 Wed + 15:00..22:00 = 9 bars
    d1 = D.d1_ny(f, PT, PIP, CFG)
    assert pd.Timestamp('2012-01-05') not in d1.index and pd.Timestamp('2012-01-04') in d1.index


def test_corrupt_spread_voids_the_mid_close_but_not_the_bid_prices():
    f = h1()
    day = D.trading_dates(f.index, CFG) == pd.Timestamp('2012-01-04')
    f.loc[day, 'spread'] = 300000.0
    d1 = D.d1_ny(f, PT, PIP, CFG)
    row = d1.loc[pd.Timestamp('2012-01-04')]
    assert np.isnan(row['close']) and row['corrupt_spread'] and np.isnan(row['spread_close_pips'])
    assert row['close_bid'] == f.loc[pd.Timestamp('2012-01-04 22:00', tz='UTC'), 'close']
    assert row['high'] == f.loc[day, 'high'].max()
    assert pd.Timestamp('2012-01-04') not in D.valid(d1).index
    assert pd.Timestamp('2012-01-03') in D.valid(d1).index


def test_usdjpy_spread_limit_is_50_pips():
    assert not D.corrupt_spread([500], 0.001, 0.01, CFG)[0]      # 0.5 yen = 50 pips: not above the limit
    assert D.corrupt_spread([501], 0.001, 0.01, CFG)[0]
    assert D.corrupt_spread([5001], 0.00001, 0.0001, CFG)[0] and not D.corrupt_spread([90], 0.00001, 0.0001, CFG)[0]
