"""Fills, cost floor and the daily series (openspec change gold-window-strategy, task 3.2)."""
import pandas as pd
import pytest

from src.gold_window import simulate as S

FLOOR, STRESS = 0.0062, 0.0122
DAY = pd.Timestamp('2021-03-03', tz='UTC')       # a Wednesday: triple swap night at the broker


def _trade(side, entry=2000.0, exit_=2010.0, es=0.25, xs=0.40, day=DAY, gate=True):
    return {'rule': 'r', 'day': day, 'entry_time': day + pd.Timedelta(hours=15), 'entry_price': entry,
            'entry_spread': es, 'side': side, 'exit_time': day + pd.Timedelta(hours=22), 'exit_price': exit_,
            'exit_spread': xs, 'gate_value': 1.0, 'threshold': 0.5, 'gate_open': gate}


def test_zero_spread_trade_is_charged_the_floor():
    t = S.trade_returns(pd.DataFrame([_trade(1, es=0.0, xs=0.0)]), FLOOR)
    assert t.cost.iloc[0] == pytest.approx(FLOOR / 100)


def test_buy_pays_its_bars_spread_and_a_sell_does_not():
    # spreads wide enough to beat the floor: 2.0 / 2000 = 0.1 %
    long_ = S.trade_returns(pd.DataFrame([_trade(1, es=2.0, xs=4.0)]), FLOOR).iloc[0]
    short = S.trade_returns(pd.DataFrame([_trade(-1, es=2.0, xs=4.0)]), FLOOR).iloc[0]
    assert long_.cost == pytest.approx(2.0 / 2000)       # long buys at entry: entry bar's spread
    assert short.cost == pytest.approx(4.0 / 2000)       # short buys back at exit: exit bar's spread
    assert long_.gross == pytest.approx(10 / 2000) and short.gross == pytest.approx(-10 / 2000)


def test_stress_floor_is_higher():
    tr = pd.DataFrame([_trade(1, es=0.0, xs=0.0)])
    days = pd.DatetimeIndex([DAY])
    out = S.simulate(tr, days, FLOOR, STRESS)
    assert out['net'].iloc[0] - out['stress'].iloc[0] == pytest.approx((STRESS - FLOOR) / 100)


def test_seven_trade_day_sums_into_one_value_and_other_days_are_zero():
    other = DAY + pd.Timedelta(days=1)
    tr = pd.DataFrame([_trade(1, exit_=2000.0 + i) for i in range(7)])
    t = S.trade_returns(tr, FLOOR)
    s = S.daily_series(t, pd.DatetimeIndex([DAY, other]))
    assert s.loc[DAY] == pytest.approx(t.net.sum()) and s.loc[other] == 0.0
    assert len(s) == 2


def test_no_swap_is_ever_charged():
    """The same trade costs the same on a Wednesday (broker triple swap) and a Monday."""
    mon = DAY - pd.Timedelta(days=2)
    wed = S.trade_returns(pd.DataFrame([_trade(1)]), FLOOR).net.iloc[0]
    monday = S.trade_returns(pd.DataFrame([_trade(1, day=mon)]), FLOOR).net.iloc[0]
    assert wed == monday


def test_closed_gate_trades_are_dropped_unless_gate_is_off():
    days = pd.DatetimeIndex([DAY])
    tr = pd.DataFrame([_trade(1, gate=False)])
    assert S.simulate(tr, days, FLOOR, STRESS)['net'].iloc[0] == 0.0
    assert S.simulate(tr, days, FLOOR, STRESS, gate=False)['net'].iloc[0] != 0.0


def test_trade_on_an_ineligible_day_is_refused():
    with pytest.raises(ValueError):
        S.daily_series(S.trade_returns(pd.DataFrame([_trade(1)]), FLOOR),
                       pd.DatetimeIndex([DAY + pd.Timedelta(days=1)]))
