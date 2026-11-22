"""fx-ny-orb-hot, groups 6.2-6.3: the live forecast, signal and settle passes on a fake MT5 feed."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import live as LV
from src.fx_ny_orb import load_config
from src.fx_ny_orb import rule as R
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
TODAY = pd.Timestamp('2026-11-04', tz='UTC')            # a Wednesday in the last synthetic week
L0 = 1.1000


def h1_frames():
    out = {}
    for i, p in enumerate(('EURUSD', 'GBPUSD', 'AUDUSD')):
        f = synthetic_weeks(date(2026, 5, 31), 23, seed=i)
        f['spread'] = 8.0
        out[p] = f
    return out


def m15_from(h1):
    rows = []
    for t, r in h1.iterrows():
        for q in range(4):
            rows.append((t + pd.Timedelta(minutes=15 * q), r['open'], r['high'], r['low'], r['close'], r['spread']))
    return pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'spread']).set_index('time')


def plant(m15):
    """Today: a tight 14:00-14:45 range at L0, a long breakout closing at 15:30, then flat above."""
    m = m15.copy()
    day = (m.index >= TODAY + pd.Timedelta(hours=14)) & (m.index < TODAY + pd.Timedelta(hours=22))
    m.loc[day, ['open', 'close']] = L0
    m.loc[day, 'high'], m.loc[day, 'low'] = L0 + 0.0002, L0 - 0.0002
    up = m.index >= TODAY + pd.Timedelta(hours=15, minutes=30)
    m.loc[day & up, ['open', 'close']] = L0 + 0.0008
    m.loc[day & up, 'high'], m.loc[day & up, 'low'] = L0 + 0.0009, L0 + 0.0007
    return m


class FakeFeed:
    def __init__(self, now):
        self.now = pd.Timestamp(now)
        h1 = h1_frames()
        self.h1 = h1
        self.m15 = {p: plant(m15_from(h1[p])) for p in ('EURUSD', 'GBPUSD')}

    def symbol(self, p):
        return p

    def point(self, s):
        return 1e-5

    def label_now(self):
        return self.now

    def server_label_now(self, s):
        return pd.Timestamp('2026-10-30 22:59', tz='UTC')        # a stale tick: must never drive the passes

    def closed_bars(self, s, tf, n):
        f, dur = (self.h1[s], 60) if tf == 'H1' else (self.m15[s], 15)
        return f[f.index + pd.Timedelta(minutes=dur) <= self.now].iloc[-n:][['open', 'high', 'low', 'close', 'spread']]

    def bar_in_progress_open(self, s, tf='M15'):
        f = self.m15[s]
        cur = f[(f.index <= self.now) & (f.index + pd.Timedelta(minutes=15) > self.now)]
        return None if not len(cur) else (cur.index[0], float(cur['open'].iloc[0]), float(cur['spread'].iloc[0]))


MODELS = {p: {'coef_harx': [-5.0] + [0.0] * 14, 'coef_har': [-5.4, 0.0, 0.0, 0.0]} for p in ('EURUSD', 'GBPUSD')}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    return c


def at(hhmm):
    return TODAY + pd.Timedelta(hhmm + ':00')


def test_forecast_refuses_before_the_14_bar_closes_and_logs_once(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    assert LV.run_forecast(cfg, FakeFeed(at('14:50')), MODELS)['status'].startswith('too early')
    assert not len(LV.read_log(cfg, 'forecasts'))
    out = LV.run_forecast(cfg, FakeFeed(at('15:05')), MODELS)
    fc = LV.read_log(cfg, 'forecasts')
    assert out['status'] == 'logged' and len(fc) == 2 and set(fc['status']) == {'ok'}
    r = fc.iloc[0]
    assert r['p_harx'] == pytest.approx(-5.0) and r['hot'] and r['or_high'] == pytest.approx(L0 + 0.0002 + 4e-5)
    assert r['F'] == pytest.approx(np.exp(-5.0) * r['close14'])
    before = open(LV.log_path(cfg, 'forecasts'), 'rb').read()
    assert LV.run_forecast(cfg, FakeFeed(at('16:30')), MODELS)['status'] == 'already logged'
    assert open(LV.log_path(cfg, 'forecasts'), 'rb').read() == before


def test_forecast_uses_only_bars_labelled_14_or_earlier(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    feed = FakeFeed(at('21:30'))
    pts = LV.top_up(cfg, feed, {p: p for p in cfg['factor_pairs']})
    full = LV.forecast_rows(cfg, LV.h1_frames(cfg), pts, TODAY, MODELS)
    cut = {p: f[f.index <= at('14:00')] for p, f in LV.h1_frames(cfg).items()}
    trunc = LV.forecast_rows(cfg, cut, pts, TODAY, MODELS)
    for a, b in zip(full, trunc):
        assert {k: a[k] for k in ('status', 'p_harx', 'p_har', 'hot_threshold', 'F')} == \
            {k: b[k] for k in ('status', 'p_harx', 'p_har', 'hot_threshold', 'F')}


def test_signal_pass_logs_once_and_never_uses_the_forming_bar(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    LV.run_forecast(cfg, FakeFeed(at('15:05')), MODELS)
    assert LV.run_signal(cfg, FakeFeed(at('15:40')))['signals'] == []       # 15:30 bar not closed yet
    out = LV.run_signal(cfg, FakeFeed(at('15:52')))
    s = LV.read_log(cfg, 'signals')
    assert len(out['signals']) == 2 and len(s) == 2
    r = s.iloc[0]
    assert r['direction'] == 1 and r['entry_time'] == at('15:45') and r['entry'] == pytest.approx(L0 + 0.0008 + 4e-5)
    assert r['breakout_time'] == at('15:30')
    before = open(LV.log_path(cfg, 'signals'), 'rb').read()
    LV.run_signal(cfg, FakeFeed(at('16:31')))
    assert open(LV.log_path(cfg, 'signals'), 'rb').read() == before
    assert LV.run_signal(cfg, FakeFeed(at('19:20')))['status'] == 'after the entry cut-off'


def test_signal_is_logged_whatever_the_hot_flag(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    cold = {p: {'coef_harx': [-5.9] + [0.0] * 14, 'coef_har': [-5.9, 0, 0, 0]} for p in cfg['pairs']}
    LV.run_forecast(cfg, FakeFeed(at('15:05')), cold)
    fc = LV.read_log(cfg, 'forecasts')
    assert set(fc['status']) == {'ok'} and not fc['hot'].astype(bool).any()
    out = LV.run_signal(cfg, FakeFeed(at('15:52')))
    s = LV.read_log(cfg, 'signals')
    assert len(out['signals']) == 2 and not s['hot'].astype(bool).any()


def test_settlement_once_and_ledger_matches_the_rule(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    LV.run_forecast(cfg, FakeFeed(at('15:05')), MODELS)
    LV.run_signal(cfg, FakeFeed(at('16:01')))
    out = LV.run_settle(cfg, FakeFeed(at('22:10')))
    st, lg = LV.read_log(cfg, 'settlements'), LV.read_log(cfg, 'signal_ledger')
    assert out['status'] == 'settled' and len(st) == 2 and len(lg) == 2
    assert set(lg['status']) == {'settled'} and set(st['status']) == {'ok'}
    assert st['trade_reason'].fillna('').eq('').all() and st['real_high'].gt(st['real_low']).all()
    assert st['se_harx'].notna().all()
    day = LV.m15_day(cfg, 'EURUSD', 1e-5, TODAY)
    f = LV.read_log(cfg, 'forecasts').iloc[0]
    t = R.simulate_day(day, float(f['F']), True, 1e-4, cfg)
    assert lg.iloc[0]['r_net'] == pytest.approx(t['r_net']) and lg.iloc[0]['exit_reason'] == t['exit_reason']
    before = open(LV.log_path(cfg, 'settlements'), 'rb').read()
    assert LV.run_settle(cfg, FakeFeed(at('23:30')))['status'] == 'nothing to settle'
    assert open(LV.log_path(cfg, 'settlements'), 'rb').read() == before


def test_a_weekend_never_logs_a_forecast_for_friday(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    sunday = TODAY + pd.Timedelta(days=4, hours=17)                     # Sunday 17:00; the last tick is Friday's
    assert LV.run_forecast(cfg, FakeFeed(sunday), MODELS)['status'] == 'weekend'
    assert not len(LV.read_log(cfg, 'forecasts'))
    late = TODAY + pd.Timedelta(days=1, hours=9)                        # Thursday 09:00: yesterday is never forecast
    assert LV.run_forecast(cfg, FakeFeed(late), MODELS)['status'].startswith('too early')
    assert not len(LV.read_log(cfg, 'forecasts'))


def test_missed_signal_pass_is_marked_and_never_backfilled(cfg, monkeypatch):
    monkeypatch.setattr(LV, '_rebuild_export', lambda c, n: 'skipped')
    LV.run_forecast(cfg, FakeFeed(at('15:05')), MODELS)
    LV.run_settle(cfg, FakeFeed(at('22:10')))                                # no signal pass ran today
    lg = LV.read_log(cfg, 'signal_ledger')
    assert set(lg['status']) == {'missed'} and not len(LV.read_log(cfg, 'signals'))
    assert LV.run_signal(cfg, FakeFeed(at('22:20')))['status'] == 'after the entry cut-off'
