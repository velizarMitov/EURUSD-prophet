"""gold-bb-nyrange, task 4.1: Part 1b levels, exit scan, one open trade and R."""
import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import bb_trades as T
from src.gold_bbnr import load_config

CFG = load_config()


def ctx_from(bars, days):
    """bars: list of (open, high, low, close); `days` gives each valid day's last bar position."""
    a = np.array(bars, dtype=float)
    times = pd.date_range('2020-01-01', periods=len(a), freq='h', tz='UTC')
    return T.Ctx(pd.bdate_range('2020-01-01', periods=len(days)), np.array(days), a[:, 0], a[:, 1], a[:, 2], a[:, 3],
                 times)


def test_levels_on_the_spec_example():
    c = list(1900 + np.sin(np.arange(40)) * 2)
    c[-1] = 1850.0
    idx = pd.bdate_range('2019-01-01', periods=len(c))
    d1 = pd.DataFrame({'open': c, 'high': np.array(c) + 1, 'low': np.array(c) - 1, 'close': c, 'n_h1': 22,
                       'valid': True}, index=pd.DatetimeIndex(idx, name='date'))
    ev = T.entries(d1, CFG)
    r = ev.iloc[-1]
    assert r['dir'] == 1 and r['tp'] == pytest.approx(r['sma'])
    assert r['sl'] == pytest.approx(r['entry'] - (r['sma'] - r['entry']))
    assert r['reason'] == 'no exit data'


def test_scan_tie_is_a_stop_and_gap_fills_at_the_open():
    x = ctx_from([(1900, 1901, 1899, 1900), (1900, 1945, 1855, 1900), (1900, 1901, 1899, 1900)], [0, 2])
    i, px, why = T.scan(x, 1, 2, 1, 1940.0, 1860.0)
    assert (i, px, why) == (1, 1860.0, 'stop')
    x = ctx_from([(1900, 1901, 1899, 1900), (1850, 1855, 1845, 1850)], [0, 1])
    assert T.scan(x, 1, 1, 1, 1940.0, 1860.0) == (1, 1850.0, 'stop')
    x = ctx_from([(1900, 1901, 1899, 1900), (1950, 1955, 1945, 1950)], [0, 1])
    assert T.scan(x, 1, 1, 1, 1940.0, 1860.0) == (1, 1940.0, 'take profit')
    x = ctx_from([(1900, 1901, 1899, 1900), (1900, 1905, 1895, 1903)], [0, 1])
    assert T.scan(x, 1, 1, 1, 1940.0, 1860.0) == (1, 1903.0, 'time stop')


def _ev(rows):
    return pd.DataFrame([{'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=k), 'k': k, 'dir': d, 'entry': e,
                          'tp': tp, 'sl': sl, 'reason': ''} for k, d, e, tp, sl in rows])


def test_r_is_minus_one_at_the_stop_and_blocking():
    cfg = {**CFG, 'trade': {**CFG['trade'], 'time_stop_days': 3}}
    # one bar per day; day 0 entry; stop hit on day 2; a breach on day 1 is blocked, one on day 2 is allowed
    bars = [(1900, 1901, 1899, 1900), (1900, 1901, 1899, 1900), (1870, 1871, 1855, 1860), (1860, 1861, 1859, 1860),
            (1860, 1861, 1859, 1860), (1860, 1861, 1859, 1860), (1860, 1861, 1859, 1860)]
    x = ctx_from(bars, list(range(7)))
    ev = _ev([(0, 1, 1900.0, 1940.0, 1860.0), (1, 1, 1900.0, 1940.0, 1860.0), (2, -1, 1860.0, 1830.0, 1890.0)])
    tr = T.build_stream(ev, x, cfg)
    assert list(tr['k']) == [0, 2] and tr.attrs['blocked'] == 1
    assert tr['R'].iloc[0] == pytest.approx(-1.0) and tr['exit_reason'].iloc[0] == 'stop'


def test_time_stop_exit_and_new_entry_at_the_same_close():
    cfg = {**CFG, 'trade': {**CFG['trade'], 'time_stop_days': 2}}
    bars = [(1900, 1901, 1899, 1900)] * 6
    x = ctx_from(bars, list(range(6)))
    ev = _ev([(0, 1, 1900.0, 1940.0, 1860.0), (2, 1, 1900.0, 1940.0, 1860.0)])
    tr = T.build_stream(ev, x, cfg)
    assert list(tr['k']) == [0, 2] and tr['exit_reason'].iloc[0] == 'time stop'
    assert tr['exit_time'].iloc[0] == x.times[2] and tr['R'].iloc[0] == pytest.approx(0.0)


def test_excluded_trade_blocks_to_the_end():
    cfg = {**CFG, 'trade': {**CFG['trade'], 'time_stop_days': 2}}
    x = ctx_from([(1900, 1901, 1899, 1900)] * 4, list(range(4)))
    ev = _ev([(2, 1, 1900.0, 1940.0, 1860.0), (3, 1, 1900.0, 1940.0, 1860.0)])
    ev.loc[0, 'reason'] = 'no exit data'
    ev.loc[1, 'reason'] = 'no exit data'
    tr = T.build_stream(ev, x, cfg)
    assert list(tr['k']) == [2] and tr.attrs['blocked'] == 1 and np.isnan(tr['R'].iloc[0])
