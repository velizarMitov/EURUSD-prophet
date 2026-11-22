"""fx-bb-sma-exit, group 2: the simulated trade on synthetic Berlin-labelled H1 bars."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.fx_bb.daily import d1_ny
from src.fx_bb_sma import bb_view, load_config
from src.fx_bb_sma import trades as T

CFG = copy.deepcopy(load_config())
CFG['pairs'] = ['EURUSD']                      # one pair: no reference dates, so no pair gaps
PT, PIP, WIG = 1e-5, 1e-4, 0.0001
START = '2013-01-07'                           # a Monday in January: US and EU clocks aligned


def levels(noise_seed=0):
    rng = np.random.default_rng(noise_seed)
    lv = list(1.1000 + rng.normal(0, 0.0003, 30))
    lv += [1.0900] + [1.0960] * 9 + [1.0870] + [1.0960] * 5      # breach day 30, inside 31-39, breach day 40
    return lv


DAYS = pd.bdate_range(START, periods=len(levels()))


def h1_from_days(lv, spread=10.0):
    stamps, vals = [], []
    for d, x in zip(DAYS, lv):
        for lab in [d - pd.Timedelta(hours=1)] + [d + pd.Timedelta(hours=h) for h in range(23)]:
            stamps.append(lab)
            vals.append(x)
    v = np.asarray(vals)
    idx = pd.DatetimeIndex(stamps).tz_localize('UTC')
    return pd.DataFrame({'open': v, 'high': v + WIG, 'low': v - WIG, 'close': v,
                         'spread': np.full(len(v), float(spread))}, index=idx)


def bar(day: int, hour: int):
    return pd.Timestamp(DAYS[day]) + pd.Timedelta(hours=hour) if hour < 23 else \
        pd.Timestamp(DAYS[day]) - pd.Timedelta(hours=1)


def run(h1, cfg=CFG):
    d1 = d1_ny(h1, PT, PIP, bb_view(cfg))
    ctx = T.contexts({'EURUSD': h1}, {'EURUSD': PT}, {'EURUSD': d1}, cfg)
    tr, counts = T.build_stream(ctx, {'EURUSD': d1}, cfg)
    return T.results(tr, ctx, cfg), counts, ctx


def setbar(h1, day, hour, **kw):
    lab = pd.Timestamp(bar(day, hour)).tz_localize('UTC')
    for k, v in kw.items():
        h1.loc[lab, k] = v
    return h1


def first(h1):
    res, counts, ctx = run(h1)
    t = res[res['k'] == 30]
    assert len(t) == 1
    return t.iloc[0], res, counts, ctx


def test_construction_breaches_on_day_30_and_40_only():
    res, counts, ctx = run(h1_from_days(levels()))
    fb = ctx['EURUSD'].ind['first_breach'].to_numpy()
    assert list(np.flatnonzero(fb)) == [30, 40]
    t = res.iloc[0]
    assert t['dir'] == 1 and t['side'] == -1 and t['tp'] > t['close'] > t['sl']
    assert t['tp'] == pytest.approx(t['sma']) and t['dist'] == pytest.approx(t['close'] - t['sl'])


def test_take_profit_on_day_t_plus_3():
    h1 = h1_from_days(levels())
    t0, *_ = first(h1)
    setbar(h1, 33, 10, high=t0['tp'] + 0.0002)
    t, *_ = first(h1)
    assert t['exit_reason'] == 'take profit' and t['exit_price'] == pytest.approx(t['tp'])
    assert t['exit_time'] == pd.Timestamp(bar(33, 10)).tz_localize('UTC')
    gross = t['tp'] - t['close']
    assert t['gross_pips'] == pytest.approx(gross / PIP) and t['r_gross'] == pytest.approx(gross / t['dist'])
    assert t['r_net'] == pytest.approx((gross - 10 * PT) / t['dist'])            # half of 10 points twice


def test_stop_loss_and_same_bar_conflict_is_a_stop():
    h1 = h1_from_days(levels())
    t0, *_ = first(h1)
    setbar(h1, 34, 5, low=t0['sl'] - 0.0002)
    t, *_ = first(h1)
    assert t['exit_reason'] == 'stop loss' and t['exit_price'] == pytest.approx(t0['sl']) and not t['conflict']
    h1 = h1_from_days(levels())
    setbar(h1, 34, 5, low=t0['sl'] - 0.0002, high=t0['tp'] + 0.0002)
    t, *_ = first(h1)
    assert t['exit_reason'] == 'stop loss' and t['conflict'] and t['r_gross'] == pytest.approx(-1.0)


def test_gap_through_the_stop_fills_at_the_mid_open():
    h1 = h1_from_days(levels())
    t0, *_ = first(h1)
    gap = t0['sl'] - 0.0020
    setbar(h1, 32, 23, open=gap, low=gap - 0.0001, high=gap + 0.0001, close=gap)
    t, *_ = first(h1)
    assert t['exit_reason'] == 'stop loss' and t['exit_price'] == pytest.approx(gap + 10 * PT / 2)
    assert t['r_gross'] < -1.0


def test_time_stop_at_the_close_of_t_plus_10():
    t, res, counts, ctx = first(h1_from_days(levels()))
    assert t['exit_reason'] == 'time stop'
    assert t['exit_time'] == pd.Timestamp(bar(40, 22)).tz_localize('UTC')
    assert t['exit_price'] == pytest.approx(1.0870 + 10 * PT / 2)
    assert t['hold_hours'] == pytest.approx(14 * 24)                  # 10 trading days span two weekends


def test_entry_bar_is_not_scanned():
    h1 = h1_from_days(levels())
    t0, *_ = first(h1)
    setbar(h1, 30, 22, high=t0['tp'] + 0.001)                 # the entry bar itself reaches TP
    t, *_ = first(h1)
    assert t['exit_reason'] == 'time stop'


def test_time_stop_exit_and_new_entry_at_the_same_close():
    res, counts, _ = run(h1_from_days(levels()))
    assert list(res['k']) == [30, 40] and counts['EURUSD']['skipped_open'] == 0
    assert res.iloc[1]['entry_time'] == res.iloc[0]['exit_time']


def test_breach_during_an_open_trade_opens_nothing():
    lv = levels()
    lv[33] = 1.0870                                           # a first breach while the day-30 long is open
    res, counts, ctx = run(h1_from_days(lv))
    assert ctx['EURUSD'].ind['first_breach'].iloc[33]
    assert 33 not in set(res['k']) and counts['EURUSD']['skipped_open'] >= 1


def test_spec_example_net_r_of_plus_0_98():
    gross, cost, dist = 0.0050, (10 + 10) * PT / 2, 0.0050    # +50 pips, 1.0 pip spread at entry and exit
    assert gross / PIP == pytest.approx(50) and (gross - cost) / PIP == pytest.approx(49)
    assert (gross - cost) / dist == pytest.approx(0.98)


def test_corrupt_spread_excludes_the_trade_and_still_blocks_the_pair():
    lv = levels()
    lv[33] = 1.0870
    h1 = h1_from_days(lv)
    setbar(h1, 31, 4, spread=600000.0)
    res, counts, _ = run(h1)
    t = res[res['k'] == 30].iloc[0]
    assert t['reason'] == 'corrupt spread'
    assert 33 not in set(res['k'])                            # still blocked to t+10
    assert 40 in set(res['k'])                                # free again at the closing bar of t+10


def test_open_at_data_end():
    h1 = h1_from_days(levels())
    cut = pd.Timestamp(bar(36, 22)).tz_localize('UTC')
    res, *_ = run(h1[h1.index <= cut])
    t = res[res['k'] == 30].iloc[0]
    assert t['reason'] == 'open at data end' and np.isnan(t['r_net'])


def test_short_mirrors_long_with_zero_spread():
    lv = levels()
    h_long = h1_from_days(lv, spread=0.0)
    t0, *_ = first(h_long)
    setbar(h_long, 33, 10, high=t0['tp'] + 0.0002)
    a, *_ = first(h_long)
    mirror = [2.2 - x for x in lv]
    h_short = h1_from_days(mirror, spread=0.0)
    setbar(h_short, 33, 10, low=2.2 - (t0['tp'] + 0.0002))
    b, *_ = first(h_short)
    assert b['dir'] == -1 and b['exit_reason'] == a['exit_reason'] == 'take profit'
    assert b['r_gross'] == pytest.approx(a['r_gross']) and b['gross_pips'] == pytest.approx(a['gross_pips'])


# -- task 2.2: the bid/ask control ----------------------------------------------------------------

def test_control_equals_gross_mid_with_zero_spread():
    h1 = h1_from_days(levels(), spread=0.0)
    t0, *_ = first(h1)
    setbar(h1, 33, 10, high=t0['tp'] + 0.0002)
    t, *_ = first(h1)
    assert t['r_ctrl'] == pytest.approx(t['r_gross'])
    tt, *_ = first(h1_from_days(levels(), spread=0.0))
    assert tt['r_ctrl'] == pytest.approx(tt['r_gross'])                       # time stop


def test_control_long_enters_at_the_ask_and_exits_on_the_bid():
    h1 = h1_from_days(levels(), spread=20.0)
    t0, *_ = first(h1)
    setbar(h1, 33, 10, high=t0['tp'] + 0.0001)                # bid high reaches TP
    t, *_ = first(h1)
    entry_ask = 1.0900 + 20 * PT
    assert t['r_ctrl'] == pytest.approx((t['tp'] - entry_ask) / t['dist'])


def test_control_short_enters_at_the_bid_and_exits_on_the_ask():
    lv = [2.2 - x for x in levels()]
    h1 = h1_from_days(lv, spread=20.0)
    t0, *_ = first(h1)
    assert t0['dir'] == -1
    setbar(h1, 33, 10, low=t0['tp'] - 20 * PT - 0.0001)       # ask low = bid low + spread reaches TP
    t, *_ = first(h1)
    assert t['r_ctrl'] == pytest.approx(-(t['tp'] - (2.2 - 1.0900)) / t['dist'])
