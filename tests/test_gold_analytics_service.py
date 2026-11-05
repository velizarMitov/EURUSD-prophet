"""Section payloads (openspec change gold-market-analytics, task 4.1)."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.data import Frame
from src.gold_analytics.service import SECTIONS, Service

NOW = datetime(2026, 11, 4, 16, 0, tzinfo=ZoneInfo('Europe/Berlin'))   # a Wednesday, 17:00 Sofia
BANNED = ('volume', 'vwap', 'poc', 'profile', 'buy', 'sell', 'купи', 'продай')


def _walk(times, seed):
    rng = np.random.default_rng(seed)
    c = 4000 * np.exp(np.cumsum(rng.normal(0, 0.001, len(times))))
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({'time': times, 'open': o, 'high': np.maximum(o, c) + 0.5,
                         'low': np.minimum(o, c) - 0.5, 'close': c})


def _intraday(minutes, days):
    t = pd.date_range(end='2026-11-04 15:59', periods=days * 24 * 60 // minutes, freq=f'{minutes}min', tz='UTC')
    t = t.floor(f'{minutes}min')
    return t[t.weekday < 5]


class FakeSource:
    def __init__(self, stale=False):
        self.stale = stale
        self.frames = {
            'D1': _walk(pd.bdate_range(end='2026-11-04', periods=400, tz='UTC'), 1),
            'H1': _walk(_intraday(60, 400), 2),
            'M15': _walk(_intraday(15, 120), 3),
            'M1': _walk(_intraday(1, 8), 4),
        }

    def bars(self, tf, start=None, end=None):
        df = self.frames[tf]
        return Frame(df, self.stale, df['time'].max())

    def tick(self):
        return {'bid': 4010.0, 'ask': 4010.4, 'time': '2026-11-04T15:59:30+00:00',
                'stale': self.stale, 'as_of': '2026-11-04T15:59:30+00:00'}


def _fred(release_id):
    return {50: ['2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06'],
            10: ['2026-08-12', '2026-09-11', '2026-10-15', '2026-11-13']}[release_id]


@pytest.fixture
def svc(tmp_path):
    cfg = load_config()
    cfg['paths'] = {**cfg['paths'], **{k: str(tmp_path / cfg['paths'][k].split('/')[-1])
                                       for k in ('cache_dir', 'bars_m1', 'bars_m15', 'spread_5m', 'events',
                                                 'last_tick')}}
    return Service(cfg, FakeSource(), clock=lambda: NOW, fred=_fred)


def test_bad_timeframe_lists_the_allowed_values(svc):
    with pytest.raises(ValueError) as e:
        svc.build('chart', timeframe='2h')
    for tf in ('1m', '5m', '15m', '30m', '1h', '4h', 'D1', 'W1'):
        assert f"'{tf}'" in str(e.value)


@pytest.mark.parametrize('section,params', [
    ('nope', {}), ('intraday', {'interval': 7}), ('sessions', {'session': 'tokyo'}), ('sessions', {'period': 7}),
    ('history', {'start': '2025-13-01', 'end': '2025-06-30'}), ('history', {'start': '2025-01-01'}),
    ('events', {'event': 'gdp'}), ('events', {'months': 5}), ('benchmarks', {'start': 'yesterday'}),
    ('chart', {'start': '04/11/2026'}), ('volatility', {'period': 20})])
def test_every_bad_parameter_is_refused(svc, section, params):
    with pytest.raises(ValueError):
        svc.build(section, **params)


@pytest.mark.parametrize('section', SECTIONS)
def test_every_payload_is_labelled_and_wordless(svc, section):
    p = svc.build(section)
    assert p['label']['bg'].startswith('Описателно') and p['label']['en'].startswith('Descriptive')
    assert p['stale'] is False and p['as_of']
    assert all(v['bg'] and v['en'] for v in p['texts'].values())
    text = json.dumps(p, ensure_ascii=False).lower()
    for word in BANNED:
        assert word not in text, (section, word)


def test_overview_reads_the_live_state(svc):
    d = svc.build('overview')['data']
    assert d['tick']['bid'] == 4010.0 and d['tick']['spread'] == pytest.approx(0.4)
    assert d['session']['current'] == ['london', 'new_york', 'overlap']
    assert d['market_open'] is True and d['hv20_pct'] > 0
    assert d['tick']['time'].startswith('2026-11-04T16:59:30')   # shown in Sofia time


def test_stale_source_marks_the_payload(tmp_path):
    cfg = load_config()
    cfg['paths'] = {**cfg['paths'], 'events': str(tmp_path / 'e.csv'), 'spread_5m': str(tmp_path / 's.gz')}
    p = Service(cfg, FakeSource(stale=True), clock=lambda: NOW, fred=_fred).build('overview')
    assert p['stale'] is True and p['data']['market_open'] is False


def test_chart_caps_and_states_the_first_bar(svc):
    d = svc.build('chart', timeframe='1m')['data']
    assert d['n_bars'] == 2000 and d['more_before'] is True and d['first_bar']
    w = svc.build('chart', timeframe='W1')['data']
    assert all(pd.Timestamp(b[0]).tz_convert('Europe/Berlin').weekday() == 0 for b in w['bars'])
    h4 = svc.build('chart', timeframe='4h')['data']
    assert {pd.Timestamp(b[0]).hour for b in h4['bars']} <= {0, 4, 8, 12, 16, 20}


def test_sections_compute_on_the_fake_history(svc):
    s = svc.build('sessions', period=20)['data']
    assert s['resolution'] == '15m' and s['sessions']['london']['summary']['n'] == 20
    i = svc.build('intraday', interval=1, period=250)['data']
    assert i['limited'] is True and i['days_used'] < 250
    e = svc.build('events', event='nfp', months=6)['data']
    assert e['n_releases'] == 3 and e['next'].startswith('2026-11-06T15:30')
    h = svc.build('history', start='2026-01-01', end='2026-06-30')['data']
    assert h['days'] > 100 and 'sessions' in h
    b = svc.build('benchmarks', start='2026-11-04T15:30')['data']
    assert b['bar_minutes'] == 1 and b['arrival']['arrival'] is not None
