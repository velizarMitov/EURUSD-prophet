"""rts-signals task 4.2: the base trade on synthetic minute bars."""

import numpy as np
import pandas as pd
import pytest

from src.rts_signals import load_config
from src.rts_signals.rules import session
from src.rts_signals.simulate import base_trade

CFG = load_config()
H = 0.00002          # half the synthetic bar range


def bars(report_date, path, c4, spread=12, hi=None, lo=None, gap=None):
    """Minute bars from the session open, one per price in `path` (then flat to the session end), plus
    30 minutes before the 4 pm label at `c4`. `hi`/`lo` override single bars {i: price}."""
    s = session(report_date)
    n = int((s['end'] - s['open']) / pd.Timedelta(minutes=1))
    px = np.r_[path, np.full(max(0, n - len(path)), path[-1])][:n]
    idx = pd.date_range(s['open'], periods=n, freq='1min', tz='UTC', name='time')
    d = pd.DataFrame({'open': px, 'high': px + H, 'low': px - H, 'close': px, 'spread': float(spread)}, index=idx)
    for i, v in (hi or {}).items():
        d.iloc[i, d.columns.get_loc('high')] = v
    for i, v in (lo or {}).items():
        d.iloc[i, d.columns.get_loc('low')] = v
    if gap:
        d = d.drop(d.index[gap[0]:gap[1]])
    pre = pd.date_range(s['close_4pm'] - pd.Timedelta(minutes=30), periods=30, freq='1min', tz='UTC', name='time')
    p = pd.DataFrame({'open': c4, 'high': c4, 'low': c4, 'close': c4, 'spread': float(spread)}, index=pre)
    return pd.concat([p, d]).sort_index()


NOW = pd.Timestamp('2026-12-31', tz='UTC')


def rec(direction, entry, stop, target, close, pair='EURO', date='2026-11-24'):
    return {'report_date': date, 'pair': pair, 'direction': direction, 'entry': entry, 'stop': stop,
            'target': target, 'close': close, 'npr': 'false', 'entry_type': 'rule0' if entry == close else 'guerilla'}


def test_target_first():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)
    path = np.linspace(1.0900, 1.0860, 200)
    t = base_trade(CFG, r, bars('2026-11-24', path, 1.0900), NOW)
    assert (t['status'], t['exit_reason'], t['exit']) == ('filled', 'target', 1.0866)
    assert t['gross_pips'] == pytest.approx(34) and t['gross_r'] == pytest.approx(34 / 30)


def test_stop_first():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)
    t = base_trade(CFG, r, bars('2026-11-24', np.linspace(1.0900, 1.0940, 100), 1.0900), NOW)
    assert (t['exit_reason'], t['exit']) == ('stop', 1.0930) and t['gross_r'] == pytest.approx(-1)


def test_tie_counts_as_stop():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)
    t = base_trade(CFG, r, bars('2026-11-24', [1.0900] * 10, 1.0900, hi={5: 1.0935}, lo={5: 1.0860}), NOW)
    assert t['exit_reason'] == 'stop'


def test_proportionality_gap_of_20_points():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)
    t = base_trade(CFG, r, bars('2026-11-24', [1.0920] * 5, 1.0900), NOW)
    assert t['proportional'] is True and t['entry'] == 1.0920
    assert t['stop'] == pytest.approx(1.0950) and t['target'] == pytest.approx(1.0886)
    small = base_trade(CFG, r, bars('2026-11-24', [1.0910] * 5, 1.0900), NOW)
    assert small['proportional'] is False and small['stop'] == 1.0930 and small['entry'] == 1.0910


def test_untouched_guerilla():
    r = rec('SELL', 1.1009, 1.1044, 1.0968, 1.0975)
    t = base_trade(CFG, r, bars('2026-11-24', np.linspace(1.0975, 1.0990, 300), 1.0975), NOW)
    assert t['status'] == 'entry not reached' and t['net_r'] == ''


def test_guerilla_filled_then_target():
    r = rec('BUY', 158.98, 158.58, 159.42, 159.20, pair='YEN')
    path = np.r_[np.linspace(159.20, 158.98, 60), np.linspace(158.98, 159.50, 200)]
    t = base_trade(CFG, r, bars('2026-11-24', path, 159.20, spread=14), NOW)
    assert (t['status'], t['entry'], t['exit_reason'], t['exit']) == ('filled', 158.98, 'target', 159.42)
    assert t['gross_pips'] == pytest.approx(44) and t['net_pips'] == pytest.approx(44 - 1.4)


def test_friday_noon_exit():
    s = session('2026-11-19')                                       # Thursday report -> Friday session
    assert s['friday_close'] and s['end'] == pd.Timestamp('2026-11-20 18:00', tz='UTC')
    r = rec('BUY', 1.0900, 1.0860, 1.0950, 1.0900, date='2026-11-19')
    t = base_trade(CFG, r, bars('2026-11-19', [1.0905] * 5, 1.0900), NOW)
    assert t['exit_reason'] == 'time' and t['exit_time'].startswith('2026-11-20 17:59')


def test_net_below_gross_by_exactly_the_spread():
    r = rec('BUY', 1.0900, 1.0870, 1.0940, 1.0900)
    t = base_trade(CFG, r, bars('2026-11-24', np.linspace(1.0900, 1.0950, 100), 1.0900, spread=12), NOW)
    assert t['exit_reason'] == 'target'
    assert t['gross_pips'] - t['net_pips'] == pytest.approx(1.2)
    assert t['spread_cost'] == pytest.approx(12 * 0.00001)
    s = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)              # a sell pays the spread at its exit bar
    u = base_trade(CFG, s, bars('2026-11-24', np.linspace(1.0900, 1.0860, 200), 1.0900, spread=9), NOW)
    assert u['gross_pips'] - u['net_pips'] == pytest.approx(0.9)


def test_gap_in_bars_is_no_price_data():
    r = rec('BUY', 1.0900, 1.0870, 1.0940, 1.0900)
    t = base_trade(CFG, r, bars('2026-11-24', [1.0900] * 5, 1.0900, gap=(0, 90)), NOW)
    assert t['status'] == 'no price data' and t['gross_r'] == ''


def test_pending_npr_and_mismatch():
    r = rec('BUY', 1.0900, 1.0870, 1.0940, 1.0900)
    b = bars('2026-11-24', [1.0900] * 5, 1.0900)
    assert base_trade(CFG, r, b, pd.Timestamp('2026-11-25 12:00', tz='UTC'))['status'] == 'pending'
    assert base_trade(CFG, dict(r, npr='true'), b, NOW)['status'] == 'npr'
    assert base_trade(CFG, dict(r, close=1.0800, entry=1.0800, stop=1.0770, target=1.0840), b, NOW)['status'] == \
        'price mismatch'
