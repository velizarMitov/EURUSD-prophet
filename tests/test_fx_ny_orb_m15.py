"""fx-ny-orb-hot, group 2: M15 mid bars, the clock check and the live read-only feed."""
import copy
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval.m15_data import ClockMismatch
from src.fx_ny_orb import REPO, load_config
from src.fx_ny_orb import m15 as M
from tests.test_fx_bb_protected import _digest

CFG = load_config()


def _m1(minutes=600, start='2024-01-08 13:00', spread=10, point=1e-5, seed=0):
    idx = pd.date_range(start, periods=minutes, freq='1min', tz='UTC')
    c = 1.1 + np.cumsum(np.random.default_rng(seed).normal(0, 1e-5, minutes))
    return pd.DataFrame({'open': c, 'high': c + 2e-5, 'low': c - 2e-5, 'close': c, 'tick_volume': 1,
                         'spread_points': float(spread), 'point': point}, index=idx)


def test_mid_high_is_bid_high_plus_half_the_spread():
    bars = pd.DataFrame({'open': [1.1], 'high': [1.105], 'low': [1.099], 'close': [1.1], 'spread_price': [0.0001]},
                        index=pd.DatetimeIndex(['2024-01-08 14:00'], tz='UTC'))
    m = M.with_mid(bars, 0.0001, CFG)
    assert m['mh'].iloc[0] == pytest.approx(1.10505) and not m['corrupt'].iloc[0]


def test_historical_bars_from_m1_and_a_corrupt_spread_voids_the_day():
    m1 = _m1()
    m1.iloc[90:120, m1.columns.get_loc('spread_points')] = 60000.0          # 600 pips over two whole bars
    bars = M.hist_bars('EURUSD', CFG, loader=lambda p: m1, verify=lambda idx: None)
    assert (bars.index.minute % 15 == 0).all() and bars['corrupt'].any()
    assert M.days_with_corrupt(bars) == {pd.Timestamp('2024-01-08', tz='UTC')}


def test_shifted_clock_is_refused():
    def refuse(idx):
        raise ClockMismatch('bar labels do not match Europe/Berlin')
    with pytest.raises(ClockMismatch):
        M.hist_bars('EURUSD', CFG, loader=lambda p: _m1(), verify=refuse)


class FakeMt5:
    TIMEFRAME_H1, TIMEFRAME_M15 = 16385, 15

    def __init__(self):
        self.calls = []

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def copy_rates_from_pos(self, symbol, tf, start, count):
        self.calls.append((start, count))
        t0 = 1_700_000_100 // 900 * 900
        rows = [(t0 + 900 * i, 1.1, 1.2, 1.0, 1.15, 10, 7, 0) for i in range(count if start else 1)]
        return np.array(rows, dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
                                     ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')])

    def symbol_info_tick(self, s):
        return SimpleNamespace(time=1_700_000_500)


def test_live_feed_never_reads_the_forming_bar_except_its_open():
    fake = FakeMt5()
    live = M.Mt5Live(fake)
    df = live.closed_bars('EURUSD', 'M15', 5)
    assert fake.calls[-1] == (1, 5) and len(df) == 5 and list(df.columns) == ['open', 'high', 'low', 'close', 'spread']
    t, o, sp = live.bar_in_progress_open('EURUSD')
    assert fake.calls[-1] == (0, 1) and o == 1.1 and sp == 7
    assert live.server_label_now('EURUSD') == pd.Timestamp(1_700_000_500, unit='s', tz='UTC')


def test_live_cache_is_append_only_and_counts_revisions(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['results'] = str(tmp_path)
    idx = pd.date_range('2026-11-23 14:00', periods=8, freq='15min', tz='UTC')
    df = pd.DataFrame({'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15, 'spread': 7.0}, index=idx)
    assert M.update_cache('EURUSD', 'M15', df.iloc[:5], cfg)['appended'] == 5
    before = open(M.cache_file('EURUSD', 'M15', cfg), 'rb').read()
    rev = df.copy()
    rev.iloc[2, 0] = 1.3
    r = M.update_cache('EURUSD', 'M15', rev, cfg)
    assert r == {'appended': 3, 'overlap_mismatches': 1}
    assert open(M.cache_file('EURUSD', 'M15', cfg), 'rb').read().startswith(before)
    live = M.live_m15('EURUSD', 1e-5, cfg)
    assert len(live) == 8 and live['mh'].iloc[0] == pytest.approx(1.2 + 7e-5 / 2)


REAL = os.path.exists(os.path.join(REPO, 'results', 'curl', 'raw', 'EURUSD_M1.parquet'))


@pytest.mark.skipif(not REAL, reason='M1 parquet not present')
def test_real_history_leaves_the_horizon_study_cache_untouched():
    before = _digest('research_models/horizon_study')
    bars = M.hist_bars('GBPUSD', CFG)
    assert _digest('research_models/horizon_study') == before
    assert bars.index.min() < pd.Timestamp('2018-08-10', tz='UTC') and bars.index.max() > pd.Timestamp('2026-08-06', tz='UTC')
