"""fx-ny-orb-hot, group 3.1: the opening-range breakout on hand-built M15 days."""
import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import load_config
from src.fx_ny_orb import rule as R

CFG = load_config()
DAY = pd.Timestamp('2024-01-10', tz='UTC')
F = 0.0040                     # stop 0.0010, target 0.0024, width max 0.0020, breakout max 0.0010
PIP = 0.0001


def day(level=1.1000, wig=0.0002, spread=0.0, until='21:00'):
    labels = pd.date_range(DAY + pd.Timedelta(hours=14), DAY + pd.Timedelta(until + ':00') - pd.Timedelta(minutes=15),
                           freq='15min')
    d = pd.DataFrame({'mo': level, 'mh': level + wig, 'ml': level - wig, 'mc': level, 'spread_price': spread},
                     index=labels)
    return d


def at(d, hhmm, **kw):
    lab = DAY + pd.Timedelta(hhmm + ':00')
    for k, v in kw.items():
        d.loc[lab, k] = v
    return d


def shift_from(d, hhmm, level, wig=0.0001):
    """Every bar from `hhmm` on sits flat at `level` (+/- wig)."""
    sel = d.index >= DAY + pd.Timedelta(hhmm + ':00')
    d.loc[sel, ['mo', 'mc']] = level
    d.loc[sel, 'mh'], d.loc[sel, 'ml'] = level + wig, level - wig
    return d


def long_breakout(d):
    shift_from(d, '15:30', 1.1008)                                    # the 15:30 close breaks the high 1.1002
    return d


def test_missing_1430_bar_means_no_range():
    d = day().drop(DAY + pd.Timedelta('14:30:00'))
    assert R.entries(d, F, CFG)['reason'] == 'no range'


def test_range_too_wide_breakout_too_far_and_late_breakout():
    d = at(day(), '14:15', mh=1.1025)                                 # width 0.0027 > 0.0020
    assert R.entries(d, F, CFG)['reason'] == 'range too wide'
    d = at(day(), '15:30', mc=1.1013, mh=1.1014)                      # 0.0011 beyond the high 1.1002
    assert R.entries(d, F, CFG)['reason'] == 'breakout too far'
    d = at(day(), '18:45', mc=1.1008, mh=1.1009)
    e = R.entries(d, F, CFG)
    assert e['reason'] == 'too late' and e['breakout_time'] == DAY + pd.Timedelta('18:45:00')
    assert R.entries(day(), F, CFG)['reason'] == 'no breakout'


def test_only_the_first_breakout_counts():
    d = at(day(), '15:15', mc=1.1013, mh=1.1014)                      # first breakout too far
    at(d, '16:00', mc=1.1006, mh=1.1007)                              # a later valid one is ignored
    assert R.entries(d, F, CFG)['reason'] == 'breakout too far'


def test_long_reaches_its_target():
    d = long_breakout(day())
    at(d, '16:30', mh=1.1040, mc=1.1035)
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['direction'] == 1 and t['entry'] == pytest.approx(1.1008) and t['entry_time'] == DAY + pd.Timedelta('15:45:00')
    assert t['stop'] == pytest.approx(1.0998) and t['target'] == pytest.approx(1.1032)
    assert t['exit_reason'] == 'take profit' and t['exit'] == pytest.approx(1.1032) and t['r_gross'] == pytest.approx(2.4)


def test_short_reaches_its_stop_and_never_re_enters():
    d = at(day(), '15:30', mc=1.0992, ml=1.0991)
    at(d, '15:45', mo=1.0992, mh=1.0993, ml=1.0991, mc=1.0992)
    at(d, '16:00', mh=1.1003, mc=1.1001)                              # stop = 1.0992 + 0.0010
    at(d, '17:00', mc=1.1009, mh=1.1010)                              # an opposite breakout later
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['direction'] == -1 and t['exit_reason'] == 'stop loss' and t['r_gross'] == pytest.approx(-1.0)


def test_same_bar_conflict_is_a_stop():
    d = long_breakout(day())
    at(d, '16:00', mh=1.1040, ml=1.0990)
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['exit_reason'] == 'stop loss' and t['conflict']


def test_stop_fills_at_the_gap_open():
    d = long_breakout(day())
    at(d, '16:15', mo=1.0990, mh=1.0991, ml=1.0985, mc=1.0988)
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['exit'] == pytest.approx(1.0990) and t['r_gross'] == pytest.approx(-1.8)


def test_forced_exit_at_the_2045_close():
    d = long_breakout(day())
    at(d, '20:45', mc=1.1015)
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['exit_reason'] == 'forced exit' and t['exit_time'] == DAY + pd.Timedelta('20:45:00')
    assert t['exit'] == pytest.approx(1.1015)


def test_hand_computed_net_of_plus_2306_r():
    d = day(level=1.0990, wig=0.0002)
    at(d, '15:30', mc=1.0997, mh=1.0998)                              # 0.0005 beyond the high 1.0992
    shift_from(d, '15:45', 1.1000)
    at(d, '15:45', spread_price=0.00005)
    at(d, '16:30', mh=1.1030, spread_price=0.00005)
    t = R.simulate_day(d, F, True, PIP, CFG)
    assert t['entry'] == pytest.approx(1.1000) and t['exit_reason'] == 'take profit'
    assert t['r_net'] == pytest.approx((0.0024 - 0.00005 - 0.000044) / 0.0010)
    assert t['r_net'] == pytest.approx(2.306)


def test_non_hot_control_is_labelled():
    t = R.simulate_day(long_breakout(day()), F, False, PIP, CFG)
    assert t['hot'] is False and t['reason'] == ''


def test_long_and_short_mirror_on_a_reflected_day():
    d = long_breakout(day())
    at(d, '16:30', mh=1.1040, mc=1.1035)
    L = 2.2
    m = d.copy()
    m['mo'], m['mc'] = L - d['mo'], L - d['mc']
    m['mh'], m['ml'] = L - d['ml'], L - d['mh']
    a = R.simulate_day(d, F, True, PIP, CFG)
    b = R.simulate_day(m, F, True, PIP, CFG)
    assert b['direction'] == -a['direction'] and b['exit_reason'] == a['exit_reason']
    assert b['r_gross'] == pytest.approx(a['r_gross']) and b['entry_time'] == a['entry_time']


def test_entries_never_reads_after_the_entry_open():
    d = long_breakout(day())
    cut = d[d.index <= DAY + pd.Timedelta('15:45:00')].copy()
    cut.loc[cut.index[-1], ['mh', 'ml', 'mc']] = np.nan              # only the entry bar's open is known
    a, b = R.entries(d, F, CFG), R.entries(cut, F, CFG)
    assert {k: a[k] for k in ('entry', 'stop', 'target', 'entry_time')} == \
        {k: b[k] for k in ('entry', 'stop', 'target', 'entry_time')}
