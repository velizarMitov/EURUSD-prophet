"""gold-nyrange-forward, group 2: live data, the append-only caches, the clock and the contract roll."""
import numpy as np
import pandas as pd
import pytest

from src.gold_nyfwd import cache as C
from src.gold_nyfwd import contracts as K
from src.gold_nyfwd import live as LV
from src.gold_nyfwd import prereg as PR
from tests._gold_nyfwd_fakes import CFG, GOLD, FakeFeed, cfg_in, common_files_in_tmp, listed, market, no_gate  # noqa: F401


@pytest.fixture
def cfg(tmp_path):
    return cfg_in(tmp_path)


def test_repeated_top_up_stores_each_closed_bar_once_and_never_rewrites(cfg):
    bars = market()
    now = pd.Timestamp('2026-07-15 15:08', tz='UTC')
    feed = FakeFeed(bars, listed(), now)
    LV.top_up(cfg, feed, now)
    first = C.load(cfg, GOLD)
    assert first.index.is_unique and first.index.max() == pd.Timestamp('2026-07-15 14:00', tz='UTC')
    assert len(first) == min(4000, int((bars[GOLD].index + pd.Timedelta(hours=1) <= now).sum()))
    # the broker revises an old bar; a second top-up in the same hour appends nothing and changes nothing
    changed = {k: v.copy() for k, v in bars.items()}
    changed[GOLD].loc[pd.Timestamp('2026-07-01 10:00', tz='UTC'), 'close'] += 5.0
    out = LV.top_up(cfg, FakeFeed(changed, listed(), now + pd.Timedelta(minutes=20)), now)
    assert out[GOLD] == {'appended': 0, 'revised_not_applied': 1}
    pd.testing.assert_frame_equal(C.load(cfg, GOLD), first)
    # an hour later exactly one new bar is appended
    LV.top_up(cfg, FakeFeed(bars, listed(), now + pd.Timedelta(hours=1)), now)
    assert len(C.load(cfg, GOLD)) == len(first) + 1


def test_a_saturday_on_the_label_clock_makes_every_pass_report_weekend(cfg):
    PR.write(cfg)
    sat = pd.Timestamp('2026-07-18 16:00', tz='UTC')            # the last tick would still be Friday 22:59
    feed = FakeFeed(market(), listed(), sat)
    assert LV.run_forecast(cfg, feed, gate=no_gate) == {'status': 'weekend'}
    assert LV.run_settle(cfg, feed, gate=no_gate) == {'status': 'weekend'}
    assert not LV.read_log(cfg, 'forecasts').shape[0]


def test_feed_drops_a_bar_whose_hour_has_not_ended():
    from src.gold_nyfwd.feed import Mt5Feed

    class M:
        TIMEFRAME_H1 = 16385

        def initialize(self, **k):
            return True

        def shutdown(self):
            pass

        def symbol_select(self, *a):
            return True

        def copy_rates_from_pos(self, sym, tf, pos, n):
            assert pos == 1
            t = np.array([pd.Timestamp('2026-07-15 13:00', tz='UTC').timestamp(),
                          pd.Timestamp('2026-07-15 14:00', tz='UTC').timestamp()], dtype='int64')
            return np.array([(t[0], 1., 2., 0.5, 1.5), (t[1], 1., 2., 0.5, 1.5)],
                            dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8')])

    f = Mt5Feed(CFG, mt5=M())
    f.label_now = lambda: pd.Timestamp('2026-07-15 14:30', tz='UTC')
    assert list(f.closed_bars(GOLD, 10).index) == [pd.Timestamp('2026-07-15 13:00', tz='UTC')]


# -- contracts ------------------------------------------------------------------------------------

EXP = {'USDIndDec26': pd.Timestamp('2026-12-11'), 'USDIndMar27': pd.Timestamp('2027-03-12')}


def test_roll_rule_takes_dec26_on_the_9th_and_mar27_on_the_10th():
    assert K.choose(pd.Timestamp('2026-12-09', tz='UTC'), EXP, 2) == 'USDIndDec26'
    assert K.choose(pd.Timestamp('2026-12-10', tz='UTC'), EXP, 2) == 'USDIndMar27'
    assert K.choose(pd.Timestamp('2026-12-01', tz='UTC'), EXP, 2) == 'USDIndDec26'


def test_without_a_successor_the_day_has_no_dollar_data(cfg):
    only = {'USDIndDec26': EXP['USDIndDec26']}
    assert K.choose(pd.Timestamp('2026-12-14', tz='UTC'), only, 2) is None
    assert K.choose(pd.Timestamp('2026-12-10', tz='UTC'), only, 2) is None
    rec = {'contracts': {'USDIndDec26': {'expiry': '2026-12-11', 'lag_check': {'status': 'pass'}}}}
    days = pd.DatetimeIndex(['2026-12-09', '2026-12-14'], tz='UTC')
    assert K.day_map(days, rec, 2) == {days[0]: 'USDIndDec26', days[1]: None}


def test_a_contract_one_hour_off_is_refused_and_a_true_one_passes(cfg):
    good = market(contract='USDIndSep26')
    bad = market(contract='USDIndSep26', dxy_lag=1)
    ok = K.lag_check(good['USDIndSep26'], good['EURUSD'], cfg)
    off = K.lag_check(bad['USDIndSep26'], bad['EURUSD'], cfg)
    assert ok['status'] == 'pass' and ok['best_lag'] == 0
    assert off['status'] == 'fail' and off['best_lag'] != 0
    assert K.lag_check(good['USDIndSep26'].iloc[:50], good['EURUSD'], cfg)['status'] == 'insufficient'


def test_failed_contract_is_recorded_and_its_days_have_no_dollar_data(cfg):
    bad = market(contract='USDIndSep26', dxy_lag=1)
    now = pd.Timestamp('2026-07-15 15:08', tz='UTC')
    LV.top_up(cfg, FakeFeed(bad, listed(), now), now)
    rec = K.read(cfg)
    assert rec['contracts']['USDIndSep26']['lag_check']['status'] == 'fail'
    assert K.day_map(pd.DatetimeIndex(['2026-07-15'], tz='UTC'), rec, 2) == {pd.Timestamp('2026-07-15', tz='UTC'): None}


def test_the_twenty_day_mean_across_a_roll_mixes_both_contracts_ranges():
    from src.gold_bbnr import load_config as gbc
    from src.gold_bbnr.ny_features import dollar_ratio
    gb = gbc()
    days = pd.DatetimeIndex(pd.bdate_range('2026-11-02', '2026-12-31'), tz='UTC')
    a = market('2026-11-02', '2026-12-31', contract='USDIndDec26', seed=5)['USDIndDec26']
    b = market('2026-11-02', '2026-12-31', contract='USDIndMar27', seed=9)['USDIndMar27']
    b[['open', 'high', 'low', 'close']] *= 1.3                    # a different level: no adjustment is made
    rec = {'contracts': {n: {'expiry': str(e.date()), 'lag_check': {'status': 'pass'}} for n, e in EXP.items()}}
    mp = K.day_map(days, rec, 2)
    assert mp[pd.Timestamp('2026-12-09', tz='UTC')] == 'USDIndDec26' and mp[pd.Timestamp('2026-12-10', tz='UTC')] == 'USDIndMar27'
    frame = K.dollar_frame(days, mp, {'USDIndDec26': a, 'USDIndMar27': b}.__getitem__)
    got = dollar_ratio(frame, days, gb)
    # by hand: each day's 00:00-14:00 range from its own contract, divided by the previous 20 days' mean
    rng = []
    for d in days:
        src = a if mp[d] == 'USDIndDec26' else b
        w = src[(src.index >= d) & (src.index <= d + pd.Timedelta(hours=14))]
        rng.append(w['high'].max() - w['low'].min())
    rng = pd.Series(rng, index=days)
    i = days.get_loc(pd.Timestamp('2026-12-21', tz='UTC'))
    want = rng.iloc[i] / rng.iloc[i - 20:i].mean()
    window = [mp[d] for d in days[i - 20:i]]
    assert 'USDIndDec26' in window and 'USDIndMar27' in window
    assert got[i] == pytest.approx(want, rel=1e-12)


def test_known_contracts_are_remembered_after_delisting(cfg):
    now = pd.Timestamp('2026-12-01 15:08', tz='UTC')
    K.remember(cfg, {'USDIndDec26': pd.Timestamp('2026-12-11'), GOLD: None}, now)
    K.remember(cfg, {'USDIndMar27': pd.Timestamp('2027-03-12')}, now)
    assert set(K.read(cfg)['contracts']) == {'USDIndDec26', 'USDIndMar27'}
    assert K.choose(pd.Timestamp('2026-12-08', tz='UTC'), K.expiries(K.read(cfg)), 2) == 'USDIndDec26'
