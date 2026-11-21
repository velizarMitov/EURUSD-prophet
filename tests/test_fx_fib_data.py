"""fx-fibonacci-mlp, Module 1a: the MT5 H1 cache (spec fx-fibonacci/market-data)."""
import copy
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval.m15_data import ClockMismatch, expected_open_label, verify_clock
from src.fx_fib import data_loader as DL
from src.fx_fib import load_config
from src.pooled_h1_data import SymbolResolutionError
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()


class FakeSource:
    def __init__(self, names, frames):
        self.names, self.frames, self.fetched = names, frames, []

    def symbol_names(self):
        return self.names

    def fetch_h1(self, symbol, bars):
        self.fetched.append(symbol)
        f = self.frames.get(symbol)
        return None if f is None else f.copy()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['data'] = str(tmp_path / 'data')
    c['data']['history_start'] = '2023-01-01'
    return c


def _frames(n_weeks=8):
    return {'EURUSD': synthetic_weeks(date(2024, 1, 7), n_weeks, seed=1),
            'GBPUSD': synthetic_weeks(date(2024, 1, 7), n_weeks, seed=2)}


def test_first_refresh_writes_both_caches_with_ohlc_only(cfg):
    rep = DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD', 'USDJPY'], _frames()), 'now')
    assert rep['status'] == 'ok'
    for pair in ('EURUSD', 'GBPUSD'):
        head = open(DL.cache_path(pair, cfg), encoding='utf-8').readline().strip()
        assert head == 'time,open,high,low,close'
        df = DL.load_h1(pair, cfg)
        assert df.index.is_unique and len(df) == rep['pairs'][pair]['rows']
        assert rep['pairs'][pair]['clock_match_rate'] == 1.0


def test_refresh_appends_only_new_bars_and_never_rewrites_rows(cfg):
    full = _frames(10)
    early = {k: v.iloc[:-30] for k, v in full.items()}
    DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], early), 'first')
    before = open(DL.cache_path('EURUSD', cfg), 'rb').read()
    rep = DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], full), 'second')
    after = open(DL.cache_path('EURUSD', cfg), 'rb').read()
    assert after.startswith(before) and rep['pairs']['EURUSD']['appended'] == 30
    assert len(DL.load_h1('EURUSD', cfg)) == len(full['EURUSD'])


def test_overlap_revisions_are_reported_not_rewritten(cfg):
    frames = _frames()
    DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], frames), 'first')
    revised = {k: v.copy() for k, v in frames.items()}
    revised['EURUSD'].iloc[5, 3] += 0.01
    rep = DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], revised), 'second')
    assert rep['pairs']['EURUSD']['overlap_mismatches'] == 1
    assert DL.load_h1('EURUSD', cfg).iloc[5, 3] == pytest.approx(frames['EURUSD'].iloc[5, 3])


def test_mt5_unreachable_leaves_caches_untouched_and_tries_no_other_vendor(cfg):
    class Down(FakeSource):
        def symbol_names(self):
            return None
    src = Down([], {})
    rep = DL.refresh(cfg, src, 'now')
    assert rep['status'] == 'mt5_unreachable' and src.fetched == []
    assert not os.path.exists(DL.cache_path('EURUSD', cfg))


def test_empty_fetch_for_one_pair_leaves_that_cache_unchanged(cfg):
    frames = _frames()
    frames.pop('GBPUSD')
    rep = DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], frames), 'now')
    assert rep['status'] == 'partial' and rep['pairs']['GBPUSD']['status'] == 'mt5_unreachable'
    assert os.path.exists(DL.cache_path('EURUSD', cfg))
    assert not os.path.exists(DL.cache_path('GBPUSD', cfg))


@pytest.mark.parametrize('names', [['EURUSD'], ['EURUSD', 'GBPUSD.a', 'GBPUSD.b']])
def test_unresolved_or_ambiguous_pair_stops_every_pair_before_any_fetch(cfg, names):
    src = FakeSource(names, _frames())
    with pytest.raises(SymbolResolutionError, match='GBPUSD'):
        DL.refresh(cfg, src, 'now')
    assert src.fetched == [] and not os.path.exists(DL.cache_path('EURUSD', cfg))


def test_suffixed_broker_symbol_is_resolved_and_reported(cfg):
    frames = _frames()
    src = FakeSource(['EURUSD.a', 'GBPUSD.a'], {'EURUSD.a': frames['EURUSD'], 'GBPUSD.a': frames['GBPUSD']})
    rep = DL.refresh(cfg, src, 'now')
    assert rep['pairs']['EURUSD']['symbol'] == 'EURUSD.a'


# -- task 2.2: the Berlin weekly-open check --------------------------------------

def test_normal_weeks_pass_the_clock_check():
    ev = verify_clock(synthetic_weeks(date(2024, 1, 7), 6).index, 'Europe/Berlin')
    assert ev['match_rate'] == 1.0 and ev['aligned']['expected_hour'] == 23


def test_march_dst_mismatch_weeks_are_judged_on_their_own_hour():
    # 2024-03-10 .. 2024-03-24: the US has moved its clock, the EU has not.
    weeks = synthetic_weeks(date(2024, 3, 10), 3)
    assert expected_open_label(date(2024, 3, 10)) == 22
    assert verify_clock(weeks.index, 'Europe/Berlin')['dst_mismatch']['match_rate'] == 1.0
    wrong = synthetic_weeks(date(2024, 3, 10), 3, shift_h=1)
    with pytest.raises(ClockMismatch):
        verify_clock(wrong.index, 'Europe/Berlin')


def test_shifted_clock_series_is_refused_and_nothing_is_written(cfg):
    frames = {'EURUSD': synthetic_weeks(date(2024, 1, 7), 8, shift_h=1),
              'GBPUSD': synthetic_weeks(date(2024, 1, 7), 8)}
    rep = DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], frames), 'now')
    assert rep['pairs']['EURUSD']['status'] == 'clock_mismatch'
    assert not os.path.exists(DL.cache_path('EURUSD', cfg))
    assert os.path.exists(DL.cache_path('GBPUSD', cfg))


def test_live_source_skips_the_forming_bar_and_respects_the_terminal_bar_limit(monkeypatch):
    from types import SimpleNamespace
    calls = []

    class FakeMt5:
        TIMEFRAME_H1 = 16385

        def initialize(self):
            return True

        def shutdown(self):
            pass

        def symbol_select(self, *a):
            return True

        def terminal_info(self):
            return SimpleNamespace(maxbars=100000)

        def copy_rates_from_pos(self, symbol, tf, start, count):
            calls.append((start, count))
            return np.array([(1_700_000_000, 1.1, 1.2, 1.0, 1.15)],
                            dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8')])

    monkeypatch.setattr(DL, 'sync_symbol', lambda mt5, symbol: True)
    df = DL.Mt5Source(FakeMt5()).fetch_h1('EURUSD', 120000)
    assert calls[-1] == (1, 99999)          # position 0 (the forming bar) is never read
    assert list(df.columns) == DL.OHLC


def test_every_run_is_logged_per_pair(cfg):
    DL.refresh(cfg, FakeSource(['EURUSD', 'GBPUSD'], _frames()), 'run-1')
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert list(log['pair']) == ['EURUSD', 'GBPUSD'] and set(log['status']) == {'ok'}
    assert (log['clock_match_rate'] == 1.0).all()
