"""Gold logger health (openspec change gold-forecasting-v2, spec gold-forecasting/logger-health)."""
import copy
import csv
import os
import sys

import pandas as pd
import pytest

from src.gold import forward_logger as FL
from src.gold import health as HL
from src.gold import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gold_forward_logger import FakeModel, _frames, env  # noqa: E402,F401  (fixture reused)

CFG = load_config()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    (tmp_path / 'forward').mkdir()
    c['paths']['forward'] = str(tmp_path / 'forward')
    return c


def _label(s):
    return pd.Timestamp(s, tz='UTC')            # a Berlin wall-clock label, stored as UTC


# ── market hours ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('label, is_open', [
    ('2026-10-21 14:00', True),                 # Wednesday afternoon in Berlin
    ('2026-10-21 23:00', False),                # 17:00 New York: the daily break
    ('2026-10-23 23:00', False),                # Friday after the close
    ('2026-10-24 12:00', False),                # Saturday
    ('2026-10-25 22:00', False),                # Sunday 17:00 New York (Berlin already on CET)
    ('2026-10-25 23:00', True),                 # Sunday 18:00 New York: the weekly open
])
def test_market_hours_follow_the_new_york_session(label, is_open):
    assert HL.market_open(_label(label)) is is_open


# ── status and the stall rule ────────────────────────────────────────────────

def _write(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def _logged_bars(cfg, bars):
    _write(os.path.join(cfg['paths']['forward'], 'predictions.csv'), FL.PRED_FIELDS,
           [{'key': f'h1_lstm|H1|{b}', 'cell': 'h1_lstm', 'cadence': 'H1', 'as_of_bar': str(b),
             'logged_at': '2026-10-21T10:03:00+00:00'} for b in bars])


NOW = pd.Timestamp('2026-10-21 14:30', tz='UTC')     # 16:30 in Berlin, a Wednesday


def test_healthy_logger_is_not_stalled(cfg):
    _logged_bars(cfg, HL.expected_h1_bars(HL.label_clock_now(NOW.to_pydatetime())))
    HL.record({'started_at': '2026-10-21T14:03:00+00:00', 'ended_at': '2026-10-21T14:03:40+00:00',
               'outcome': 'ok', 'mt5': 'yes'}, cfg)
    st = HL.status(cfg, now=NOW)
    assert st['market_open'] and st['missed_24h'] == 0 and st['hours_since_success'] < 1
    assert st['stalled'] is False


def test_weekday_stall_is_flagged(cfg):
    _logged_bars(cfg, [])
    HL.record({'started_at': '2026-10-21T09:03:00+00:00', 'ended_at': '2026-10-21T09:03:40+00:00',
               'outcome': 'ok', 'mt5': 'yes'}, cfg)
    st = HL.status(cfg, now=NOW)
    assert st['hours_since_success'] > 5 and st['stalled'] is True


def test_missed_hours_alone_raise_a_stall(cfg):
    expected = HL.expected_h1_bars(HL.label_clock_now(NOW.to_pydatetime()))
    _logged_bars(cfg, expected[:-4])                 # the newest four hours are missing
    HL.record({'started_at': '2026-10-21T14:03:00+00:00', 'ended_at': '2026-10-21T14:03:40+00:00',
               'outcome': 'ok', 'mt5': 'yes'}, cfg)
    st = HL.status(cfg, now=NOW)
    assert st['missed_24h'] == 4 and st['stalled'] is True


def test_weekend_is_quiet(cfg):
    _logged_bars(cfg, [])
    HL.record({'started_at': '2026-10-23T14:03:00+00:00', 'ended_at': '2026-10-23T14:03:40+00:00',
               'outcome': 'ok', 'mt5': 'yes'}, cfg)
    st = HL.status(cfg, now=pd.Timestamp('2026-10-24 12:00', tz='UTC'))
    assert st['market_open'] is False and st['stalled'] is False


def test_a_fresh_install_uses_the_newest_forecast_not_a_false_alarm(cfg):
    _logged_bars(cfg, HL.expected_h1_bars(HL.label_clock_now(NOW.to_pydatetime())))
    path = os.path.join(cfg['paths']['forward'], 'predictions.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    rows[-1]['logged_at'] = '2026-10-21T14:03:10+00:00'
    _write(path, FL.PRED_FIELDS, rows)
    st = HL.status(cfg, now=NOW)
    assert st['last_success'] is None and st['hours_since_success'] < 1 and st['stalled'] is False


# ── one health row per pass, whatever happens ───────────────────────────────

def _health(cfg):
    return list(csv.DictReader(open(HL.health_path(cfg), encoding='utf-8')))


def test_every_pass_writes_one_row_and_the_round_one_headers_stay(env):
    cfg, _, h1 = env
    run = lambda upto: FL.run_with_health(cfg, runner=lambda: FL.run_once(
        cfg, frames=_frames(h1, upto), loader=lambda cell, c: FakeModel(), ph='dry_run'))
    run(400)
    run(401)
    rows = _health(cfg)
    assert [r['outcome'] for r in rows] == ['ok', 'ok'] and rows[0]['mt5'] == 'yes'
    assert int(rows[0]['forecasts']) > 0
    for name, fields in (('predictions', FL.PRED_FIELDS), ('settlements', FL.SETTLE_FIELDS)):
        with open(FL.out_path(cfg, name), encoding='utf-8') as fh:
            assert next(csv.reader(fh)) == fields


def test_mt5_unreachable_is_recorded(cfg):
    FL.run_with_health(cfg, runner=lambda: {'D1': 'no live data', 'H1': 'no live data', 'settled': 0,
                                            'mt5': False, 'newest_h1': ''})
    [row] = _health(cfg)
    assert row['outcome'] == 'no live data' and row['mt5'] == 'no' and row['forecasts'] == '0'


def test_a_locked_or_crashing_pass_still_leaves_a_row(cfg):
    def locked():
        raise FL.Locked('another pass holds the lock')

    def crash():
        raise RuntimeError('boom')

    assert FL.run_with_health(cfg, runner=locked) == {'status': 'another pass holds the lock'}
    with pytest.raises(RuntimeError):
        FL.run_with_health(cfg, runner=crash)
    assert [r['outcome'] for r in _health(cfg)] == ['locked', 'error']


# ── endpoint and page ───────────────────────────────────────────────────────

def test_health_endpoint_and_page_banner():
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    d = c.get('/api/gold/health').json()
    assert {'last_pass', 'last_success', 'missed_24h', 'stalled', 'market_open'} <= set(d)
    page = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'gold.html'),
                encoding='utf-8').read()
    assert "fetch('/api/gold/health'" in page and 'id="stall"' in page
    for key in ('stall', 'stall_since', 'stall_missed', 'stall_fix'):
        assert f'  {key}: {{bg: ' in page, key


def test_api_py_still_only_gains_lines():
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(['git', '-C', repo, 'diff', '--numstat', 'HEAD', '--', 'api.py'],
                         capture_output=True, text=True).stdout.strip()
    for line in out.splitlines():
        assert line.split('\t')[1] == '0', line
