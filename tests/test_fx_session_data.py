"""fx-session-ny, group 2: MT5 caches with spread, mid prices and the session calendar."""
import copy
import os
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fx_session import data_loader as DL
from src.fx_session import load_config
from src.pooled_h1_data import SymbolResolutionError
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
NAMES = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY']


def with_spread(df, seed=0):
    out = df.copy()
    out['spread'] = np.random.default_rng(seed).integers(3, 12, len(out)).astype(float)
    return out


def frames(first=date(2010, 1, 3), n_weeks=8, shift=None):
    shift = shift or {}
    return {p: with_spread(synthetic_weeks(first, n_weeks, shift_h=shift.get(p, 0), seed=i), i)
            for i, p in enumerate(('EURUSD', 'GBPUSD', 'AUDUSD'))}


class FakeSource:
    def __init__(self, names, frames, points=None):
        self.names, self.frames, self.fetched = names, frames, []
        self.points = points or {}

    def symbol_names(self):
        return self.names

    def symbol_point(self, symbol):
        return self.points.get(symbol, 0.00001)

    def fetch_h1(self, symbol, bars):
        self.fetched.append(symbol)
        f = self.frames.get(symbol)
        return None if f is None else f.copy()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'data'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    c['data']['history_start'], c['data']['latest_allowed_start'] = '2010-01-01', '2010-02-01'
    return c


def _exists(cfg, pair):
    return os.path.exists(DL.cache_path(pair, cfg))


def test_first_refresh_writes_three_caches_with_spread_and_points(cfg):
    rep = DL.refresh(cfg, FakeSource(NAMES, frames()), 'now')
    assert rep['status'] == 'ok'
    for pair in CFG['factor_pairs']:
        head = open(DL.cache_path(pair, cfg), encoding='utf-8').readline().strip()
        assert head == 'time,open,high,low,close,spread'
        assert rep['pairs'][pair]['clock_match_rate'] == 1.0
    assert DL.load_points(cfg) == {'EURUSD': 1e-5, 'GBPUSD': 1e-5, 'AUDUSD': 1e-5}
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert list(log['pair']) == CFG['factor_pairs'] and (log['point'] == 1e-5).all()


def test_refresh_appends_only_new_bars_and_counts_revisions(cfg):
    full = frames(n_weeks=10)
    DL.refresh(cfg, FakeSource(NAMES, {k: v.iloc[:-30] for k, v in full.items()}), 'first')
    before = open(DL.cache_path('EURUSD', cfg), 'rb').read()
    revised = {k: v.copy() for k, v in full.items()}
    revised['EURUSD'].iloc[5, 3] += 0.01
    rep = DL.refresh(cfg, FakeSource(NAMES, revised), 'second')
    after = open(DL.cache_path('EURUSD', cfg), 'rb').read()
    assert after.startswith(before) and rep['pairs']['EURUSD']['appended'] == 30
    assert rep['pairs']['EURUSD']['overlap_mismatches'] == 1


def test_shallow_history_stops_every_pair_and_names_the_start(cfg):
    f = frames()
    f['GBPUSD'] = with_spread(synthetic_weeks(date(2012, 3, 4), 8, seed=9))
    rep = DL.refresh(cfg, FakeSource(NAMES, f), 'now')
    assert rep['status'] == 'refused' and '2012-03-04' in rep['pairs']['GBPUSD']
    assert not any(_exists(cfg, p) for p in CFG['factor_pairs'])


def test_shifted_clock_on_one_pair_writes_nothing_for_any_pair(cfg):
    rep = DL.refresh(cfg, FakeSource(NAMES, frames(shift={'AUDUSD': 1})), 'now')
    assert rep['status'] == 'refused' and 'clock' in rep['pairs']['AUDUSD']
    assert not any(_exists(cfg, p) for p in CFG['factor_pairs'])
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert set(log['status']) == {'refused'}


def test_unreachable_pair_writes_nothing(cfg):
    f = frames()
    f.pop('EURUSD')
    rep = DL.refresh(cfg, FakeSource(NAMES, f), 'now')
    assert rep['status'] == 'refused' and not any(_exists(cfg, p) for p in CFG['factor_pairs'])


def test_mt5_down_fetches_nothing(cfg):
    class Down(FakeSource):
        def symbol_names(self):
            return None
    src = Down([], {})
    assert DL.refresh(cfg, src, 'now')['status'] == 'mt5_unreachable' and src.fetched == []


def test_unresolved_symbol_stops_before_any_fetch(cfg):
    src = FakeSource(['EURUSD', 'GBPUSD'], frames())
    with pytest.raises(SymbolResolutionError, match='AUDUSD'):
        DL.refresh(cfg, src, 'now')
    assert src.fetched == []


def test_live_source_skips_the_forming_bar_keeps_the_spread_and_respects_maxbars(monkeypatch):
    calls = []

    class FakeMt5:
        TIMEFRAME_H1 = 16385

        def initialize(self):
            return True

        def shutdown(self):
            pass

        def terminal_info(self):
            return SimpleNamespace(maxbars=100000)

        def symbol_info(self, symbol):
            return SimpleNamespace(point=0.00001)

        def copy_rates_from_pos(self, symbol, tf, start, count):
            calls.append((start, count))
            return np.array([(1_700_000_000, 1.1, 1.2, 1.0, 1.15, 100, 7, 0)],
                            dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
                                   ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')])

    monkeypatch.setattr(DL, 'sync_symbol', lambda mt5, symbol: True)
    src = DL.Mt5Source(FakeMt5())
    df = src.fetch_h1('EURUSD', 120000)
    assert calls[-1] == (1, 99999)                 # position 0 (the forming bar) is never read
    assert list(df.columns) == DL.COLS and df['spread'].iloc[0] == 7
    assert src.symbol_point('EURUSD') == 0.00001


# -- task 2.2: mid prices and the session calendar ------------------------------------------

from src.fx_session import session as SE  # noqa: E402


def test_mid_close_is_bid_plus_half_the_spread():
    bar = pd.DataFrame({'close': [1.08000], 'spread': [6.0]})
    assert SE.mid(bar, 0.00001).iloc[0] == pytest.approx(1.08003)


def test_dst_mismatch_day_keeps_the_14_and_21_label_hours():
    f = frames(first=date(2026, 3, 8), n_weeks=2)          # 2026-03-10: US on summer time, Berlin not
    cal = SE.session_calendar(f, CFG)
    d = pd.Timestamp('2026-03-10', tz='UTC')
    assert cal.loc[d, 'eligible']
    entry = SE.bar_at(f['EURUSD'], pd.DatetimeIndex([d]), 14, 'close')
    assert entry.iloc[0] == f['EURUSD'].loc[d + pd.Timedelta(hours=14), 'close']
    assert SE.session_hours(CFG) == list(range(14, 22))


def test_early_close_day_is_excluded_with_a_reason():
    f = frames()
    d = pd.Timestamp('2010-01-07', tz='UTC')
    f['EURUSD'] = f['EURUSD'][~((f['EURUSD'].index.normalize() == d) & (f['EURUSD'].index.hour >= 20))]
    cal = SE.session_calendar(f, CFG)
    assert not cal.loc[d, 'eligible'] and cal.loc[d, 'reason'] == 'incomplete session'
    assert cal.loc[pd.Timestamp('2010-01-06', tz='UTC'), 'eligible']


def test_eligibility_is_joint_across_both_scored_pairs():
    f = frames()
    d = pd.Timestamp('2010-01-12', tz='UTC')
    f['GBPUSD'] = f['GBPUSD'].drop(d + pd.Timedelta(hours=16))
    cal = SE.session_calendar(f, CFG)
    assert not cal.loc[d, 'eligible']
    assert (cal.index.dayofweek < 5).all()


def _thin_march(f):
    g = f['GBPUSD']
    march = (g.index.month == 3)
    f['GBPUSD'] = g[~march | (g.index.hour == 0) | (g.index.dayofweek == 6)]   # weekly opens kept
    return f


def test_undeclared_thin_month_is_refused(cfg):
    cfg['data']['known_gaps'] = {}
    rep = DL.refresh(cfg, FakeSource(NAMES, _thin_march(frames(n_weeks=20))), 'now')
    assert rep['status'] == 'refused' and 'thin month 2010-03' in rep['pairs']['GBPUSD']
    assert not any(_exists(cfg, p) for p in CFG['factor_pairs'])


def test_declared_broker_gap_is_kept_as_is_and_not_filled(cfg):
    cfg['data']['known_gaps'] = {'GBPUSD': ['2010-03']}
    f = _thin_march(frames(n_weeks=20))
    rep = DL.refresh(cfg, FakeSource(NAMES, f), 'now')
    assert rep['status'] == 'ok'
    assert len(DL.load_h1('GBPUSD', cfg)) == len(f['GBPUSD'])          # nothing invented
    assert CFG['data']['known_gaps'] == {'GBPUSD': ['2014-03']}


def test_week_whose_open_is_monday_midnight_is_unverified():
    f = frames()
    sunday_bar = pd.Timestamp('2010-01-10 23:00', tz='UTC')
    f['GBPUSD'] = f['GBPUSD'].drop(sunday_bar)          # the week opens at Monday 00:00
    cal = SE.session_calendar(f, CFG)
    week = cal.loc['2010-01-11':'2010-01-15']
    assert (~week['eligible']).all() and set(week['reason']) == {'unverified clock week'}
    assert cal.loc['2010-01-18':'2010-01-22', 'eligible'].all()


def test_factor_pair_off_the_berlin_clock_also_unverifies_the_week():
    cal = SE.session_calendar(frames(shift={'AUDUSD': 1}), CFG)
    assert not cal['eligible'].any() and set(cal['reason']) == {'unverified clock week'}


def test_mismatch_week_opening_at_23_is_unverified():
    f = frames(first=date(2026, 3, 8), n_weeks=2)                     # mismatch: expected open 22:00
    assert SE.verified_weeks(f['EURUSD'].index).loc[pd.Timestamp('2026-03-09', tz='UTC')]
    late = f['EURUSD'].drop(pd.Timestamp('2026-03-08 22:00', tz='UTC'))   # opens 23:00 instead
    assert not SE.verified_weeks(late.index).loc[pd.Timestamp('2026-03-09', tz='UTC')]
