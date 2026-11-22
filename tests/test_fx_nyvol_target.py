"""fx-ny-volatility, groups 2.1 and 3.1: eligibility, the New York range target and the features."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.fx_nyvol import features as F
from src.fx_nyvol import target as TG
from tests.fx_nyvol_synth import CFG, POINTS, flat_frames, stamp, walk_frames

DAY = '2012-01-25'           # a Wednesday of the third week


def _set(f, day, hours, **kw):
    for h in hours:
        for k, v in kw.items():
            f.loc[stamp(day, h), k] = v


def test_hand_computed_target_and_the_rollover_bar_is_outside():
    fr = flat_frames()
    e = fr['EURUSD']
    _set(e, DAY, [14], close=1.1000, open=1.1000, high=1.1002, low=1.0998)
    _set(e, DAY, range(15, 22), high=1.1040, low=1.1010)
    _set(e, DAY, [17], high=1.1060)
    _set(e, DAY, [19], low=1.1005)
    d = pd.DatetimeIndex([pd.Timestamp(DAY, tz='UTC')])
    y = TG.targets(fr, POINTS, CFG, d)['EURUSD']['y'].iloc[0]
    assert y == pytest.approx(np.log(0.0055 / 1.1000)) and y == pytest.approx(np.log(0.005))
    _set(e, DAY, [22], high=1.2000, low=1.0000)
    assert TG.targets(fr, POINTS, CFG, d)['EURUSD']['y'].iloc[0] == pytest.approx(y)


def test_mid_target_adds_half_of_each_bars_own_spread():
    fr = flat_frames()
    e = fr['EURUSD']
    _set(e, DAY, [17], high=1.1060, spread=20.0)
    d = pd.DatetimeIndex([pd.Timestamp(DAY, tz='UTC')])
    t = TG.targets(fr, POINTS, CFG, d)['EURUSD'].iloc[0]
    assert t['y'] == pytest.approx(np.log((1.1060 + 20 * 1e-5 / 2 - 1.0995) / 1.1))
    assert t['y_bid'] == pytest.approx(np.log((1.1060 - 1.0995) / 1.1))


def test_missing_gbpusd_18_bar_excludes_the_day_for_both_pairs():
    fr = walk_frames()
    fr['GBPUSD'] = fr['GBPUSD'].drop(stamp(DAY, 18))
    r = TG.eligibility(fr, POINTS, CFG)
    assert r.loc[pd.Timestamp(DAY, tz='UTC')] == 'incomplete session'
    ds = TG.dataset(fr, POINTS, CFG)
    assert ds['EURUSD'].loc[pd.Timestamp(DAY, tz='UTC'), 'reason'] == 'incomplete session'


def test_corrupt_audusd_spread_excludes_the_day_for_both_scored_pairs():
    fr = walk_frames()
    _set(fr['AUDUSD'], DAY, range(5, 15), spread=300000.0)
    ds = TG.dataset(fr, POINTS, CFG)
    for p in CFG['pairs']:
        assert ds[p].loc[pd.Timestamp(DAY, tz='UTC'), 'reason'] == 'corrupt spread'
    other = pd.Timestamp('2012-01-24', tz='UTC')
    assert TG.eligibility(fr, POINTS, CFG).loc[other] == ''


def test_unverified_clock_week_is_excluded():
    fr = walk_frames()
    fr['EURUSD'] = fr['EURUSD'].drop(stamp('2012-01-22', 23))          # the week opens Monday 00:00
    r = TG.eligibility(fr, POINTS, CFG)
    week = r.loc[pd.Timestamp('2012-01-23', tz='UTC'):pd.Timestamp('2012-01-27', tz='UTC')]
    assert set(week) == {'unverified clock week'}


def test_missing_pre_session_bar_of_audusd_excludes_the_day():
    fr = walk_frames()
    fr['AUDUSD'] = fr['AUDUSD'].drop(stamp(DAY, 3))
    assert TG.eligibility(fr, POINTS, CFG).loc[pd.Timestamp(DAY, tz='UTC')] == 'missing pre-session bar'


# -- 3.1 features ----------------------------------------------------------------------------------

def _feat(fr):
    cal = TG.calendar(fr, CFG)
    reason = TG.eligibility(fr, POINTS, CFG, cal)
    tg = TG.targets(fr, POINTS, CFG, cal.index)
    avail = (reason == '').to_numpy()
    return F.build_features(fr, POINTS, CFG, cal.index, {p: tg[p].loc[avail, 'y'] for p in CFG['pairs']}), tg


def test_hand_computed_london_range():
    fr = flat_frames()
    e = fr['EURUSD']
    _set(e, DAY, range(9, 14), high=1.1020, low=1.0990)
    _set(e, DAY, [11], high=1.1030)
    _set(e, DAY, [12], low=1.0975)
    _set(e, DAY, [14], close=1.1000)
    f, _ = _feat(fr)
    assert f['EURUSD'].loc[pd.Timestamp(DAY, tz='UTC'), 'log_london_range'] == pytest.approx(np.log(0.005))


def test_cross_pre_range_ratio_is_half_on_a_quiet_morning():
    fr = flat_frames(rng_=0.0010)
    for p in fr:
        _set(fr[p], '2012-02-15', range(0, 15), high=1.1 + 0.00025, low=1.1 - 0.00025)
    f, _ = _feat(fr)
    assert f['EURUSD'].loc[pd.Timestamp('2012-02-15', tz='UTC'), 'cross_pre_range_ratio'] == pytest.approx(0.5)


def test_har_lags_are_unchanged_when_todays_range_doubles():
    fr = walk_frames()
    d = pd.Timestamp('2012-03-07', tz='UTC')
    a, tga = _feat(fr)
    e = fr['EURUSD']
    for h in range(15, 22):
        lab = stamp('2012-03-07', h)
        mid = (e.loc[lab, 'high'] + e.loc[lab, 'low']) / 2
        span = e.loc[lab, 'high'] - e.loc[lab, 'low']
        e.loc[lab, 'high'], e.loc[lab, 'low'] = mid + span * 3, mid - span * 3
    b, tgb = _feat(fr)
    assert tgb['EURUSD'].loc[d, 'y'] != tga['EURUSD'].loc[d, 'y']
    for c in CFG['features']['har_columns']:
        assert b['EURUSD'].loc[d, c] == a['EURUSD'].loc[d, c] and np.isfinite(a['EURUSD'].loc[d, c])


def test_scale_test_prices_times_100():
    fr = walk_frames()
    big = {p: f.assign(**{c: f[c] * 100 for c in ('open', 'high', 'low', 'close')}) for p, f in fr.items()}
    pts = {p: v * 100 for p, v in POINTS.items()}
    cfg = copy.deepcopy(CFG)
    cfg['data']['pip'] = {p: v * 100 for p, v in CFG['data']['pip'].items()}     # a unit change scales the pip too
    a, _ = _feat(fr)
    cal = TG.calendar(big, cfg)
    reason = TG.eligibility(big, pts, cfg, cal)
    tg = TG.targets(big, pts, cfg, cal.index)
    avail = (reason == '').to_numpy()
    b = F.build_features(big, pts, cfg, cal.index, {p: tg[p].loc[avail, 'y'] for p in CFG['pairs']})
    for p in CFG['pairs']:
        np.testing.assert_allclose(a[p].to_numpy(), b[p].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_missing_column_error_names_the_column():
    f, _ = _feat(walk_frames())
    with pytest.raises(F.FeatureColumnsError, match='cross_pre_range_ratio'):
        F.assemble(f['EURUSD'].drop(columns='cross_pre_range_ratio'), CFG)


def test_warm_up_days_are_excluded_and_later_days_are_usable():
    ds = TG.dataset(walk_frames(n_weeks=14), POINTS, CFG)
    e = ds['EURUSD']
    first_ok = e.index[e['reason'] == ''][0]
    assert (e.loc[:first_ok, 'reason'].iloc[:-1].isin(['warm-up', 'unverified clock week', 'incomplete session'])).all()
    assert (e['reason'] == 'warm-up').sum() >= 20
    assert e.loc[e['reason'] == '', CFG['features']['columns'] + ['y']].notna().all().all()
