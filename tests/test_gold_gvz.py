"""GVZ morning check (openspec change gold-gvz-morning-check, spec gold-forecasting/gvz-check)."""
import json
import os

from src.gold import CONFIG_PATH
from src.gold import prereg as P
from src.gold import prereg2 as P2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_gvz_settings_leave_the_pinned_settings_unchanged():
    cfg2 = P2.load_config_v2()
    assert 'gvz' in cfg2
    with open(os.path.join(REPO, 'results', 'gold', 'PRE_REGISTRATION_V2.json'), encoding='utf-8') as fh:
        reg2 = json.load(fh)
    assert P2.study_digest(cfg2) == reg2['config_v2_study_sha256']
    with open(os.path.join(REPO, 'results', 'gold', 'PRE_REGISTRATION.json'), encoding='utf-8') as fh:
        reg1 = json.load(fh)
    assert P.sha256_file(CONFIG_PATH) == reg1['config_sha256']


# ── read half (task 2.1) and acquisition (task 2.2) ─────────────────────────

import copy
import math
import subprocess
import sys

import pandas as pd
import pytest

from src import macro_data as MD
from src.gold import gvz as GV

CFG2 = P2.load_config_v2()


@pytest.fixture
def cfg2(tmp_path):
    c = copy.deepcopy(CFG2)
    c['paths']['macro_cache'] = str(tmp_path / 'macro')
    c['gvz']['log'] = str(tmp_path / 'gvz_log.csv')
    c['gvz']['settlements'] = str(tmp_path / 'gvz_settlements.csv')
    return c


def _fred_cache(cfg2, values: dict):
    frame = pd.DataFrame({'gvz': list(values.values())}, index=pd.to_datetime(list(values)).tz_localize('UTC'))
    frame.index.name = 'DATE'
    MD._merge_write_cache(frame, GV.cache_paths(cfg2)[0])


def _yahoo_cache(cfg2, values: dict):
    path = GV.cache_paths(cfg2)[1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({'date': list(values), 'gvz': list(values.values())}).to_csv(path, index=False)


def _series(values):
    idx = pd.to_datetime([f'2026-{1 + i // 28:02d}-{1 + i % 28:02d}' for i in range(len(values))])
    return pd.DataFrame({'gvz': values, 'source': 'FRED'}, index=idx)


def test_fred_wins_on_a_shared_date_and_yfinance_fills_only_later_dates(cfg2):
    _fred_cache(cfg2, {'2026-10-22': 24.15, '2026-10-23': 24.34})
    _yahoo_cache(cfg2, {'2026-10-23': 99.0, '2026-10-26': 23.55})
    ser = GV.read_series(cfg2)
    assert ser.loc['2026-10-23', 'gvz'] == 24.34 and ser.loc['2026-10-23', 'source'] == 'FRED'
    assert ser.loc['2026-10-26', 'gvz'] == 23.55 and ser.loc['2026-10-26', 'source'] == 'yfinance'
    rd = GV.reading(cfg2, pd.Timestamp('2026-10-27 11:00', tz='UTC'))
    assert (rd['value'], rd['date'], rd['source']) == (23.55, '2026-10-26', 'yfinance')
    _fred_cache(cfg2, {'2026-10-26': 23.80})                  # FRED catches up
    rd = GV.reading(cfg2, pd.Timestamp('2026-10-27 11:00', tz='UTC'))
    assert (rd['value'], rd['source']) == (23.8, 'FRED')


def test_implied_move_is_gvz_over_root_252(cfg2):
    rd = GV.reading(cfg2, pd.Timestamp('2026-01-02 12:00', tz='UTC'), 4000.0, 'bar', series=_series([25.2]))
    assert round(rd['implied_move_pct'], 2) == 1.59
    assert rd['implied_move_usd'] == 63.50
    assert math.isclose(rd['implied_move_pct'], 25.2 / math.sqrt(252), rel_tol=1e-4)


def test_no_gold_close_keeps_the_percent_and_drops_the_dollars(cfg2):
    rd = GV.reading(cfg2, pd.Timestamp('2026-01-02 12:00', tz='UTC'), None, series=_series([25.2]))
    assert rd['implied_move_pct'] > 0 and rd['implied_move_usd'] is None


def test_percentile_is_the_share_strictly_below(cfg2):
    rd = GV.reading(cfg2, pd.Timestamp('2026-05-01 12:00', tz='UTC'), series=_series([10.0] * 91 + [30.0] * 8 + [25.2]))
    assert rd['percentile'] == 91 and rd['history_n'] == 100


def test_stale_after_five_calendar_days(cfg2):
    _fred_cache(cfg2, {'2026-10-23': 24.34})
    assert GV.reading(cfg2, pd.Timestamp('2026-10-26 12:00', tz='UTC'))['state'] == 'ok'
    rd = GV.reading(cfg2, pd.Timestamp('2026-10-29 12:00', tz='UTC'))
    assert rd['state'] == 'stale' and rd['stale'] is True and rd['date'] == '2026-10-23'


def test_nothing_stored_reads_none(cfg2):
    assert GV.reading(cfg2) == {'state': 'none'}


def test_importing_gvz_does_not_import_yfinance():
    code = 'import sys, src.gold.gvz; print("yfinance" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def _fred_stub(values):
    def fetch(ids, start, end, combine, path):
        raw = pd.DataFrame({'gvz': list(values.values())}, index=pd.to_datetime(list(values)).tz_localize('UTC'))
        return MD._merge_write_cache(combine(raw), path), 'FRED_api'
    return fetch


def _yahoo_stub(values):
    idx = pd.DatetimeIndex(pd.to_datetime(list(values))).tz_localize('America/New_York')
    return lambda symbol: pd.Series(list(values.values()), index=idx)


def _boom(*a, **k):
    raise ConnectionError('offline')


AFTERNOON = pd.Timestamp('2026-10-27 15:00', tz='UTC')      # 17:00 Sofia, 11:00 New York


def test_refresh_stores_only_completed_sessions(cfg2):
    note = GV.refresh(cfg2, AFTERNOON, fred_fetch=_fred_stub({'2026-10-23': 24.34, '2026-10-27': 30.0}),
                      yahoo_fetch=_yahoo_stub({'2026-10-23': 24.34, '2026-10-26': 23.55, '2026-10-27': 25.0}))
    ser = GV.read_series(cfg2)
    assert str(ser.index[-1].date()) == '2026-10-26' and '2026-10-27' not in ser.index
    assert note.startswith('fresh 2026-10-26 yfinance')
    assert not GV.needs_refresh(cfg2, AFTERNOON)


def test_fred_unreachable_keeps_its_cache_and_reports_it(cfg2):
    _fred_cache(cfg2, {'2026-10-22': 24.15, '2026-10-23': 24.34})
    before = open(GV.cache_paths(cfg2)[0], encoding='utf-8').read()
    cached = lambda ids, start, end, combine, path: (MD._read_cache(path), 'cache')   # noqa: E731
    note = GV.refresh(cfg2, AFTERNOON, fred_fetch=cached, yahoo_fetch=_yahoo_stub({'2026-10-26': 23.55}))
    assert 'FRED unreachable' in note
    assert open(GV.cache_paths(cfg2)[0], encoding='utf-8').read() == before


def test_both_sources_failing_raises_and_names_gvz(cfg2):
    _fred_cache(cfg2, {'2026-10-23': 24.34})
    before = open(GV.cache_paths(cfg2)[0], encoding='utf-8').read()
    with pytest.raises(GV.GvzUnavailable, match='GVZ'):
        GV.refresh(cfg2, AFTERNOON, fred_fetch=_boom, yahoo_fetch=_boom)
    assert open(GV.cache_paths(cfg2)[0], encoding='utf-8').read() == before
    assert not os.path.exists(GV.cache_paths(cfg2)[1])


# ── observation and settlement (task 3.2) ───────────────────────────────────

import csv

from src.gold import forward_logger as FL


def _sofia(local):          # Sofia is UTC+2 after 2026-10-25
    return pd.Timestamp(local, tz='Europe/Sofia').tz_convert('UTC')


def _h1(day='2026-10-27', labels=range(0, 24), base=4000.0):
    idx = pd.DatetimeIndex([pd.Timestamp(f'{day} {h:02d}:00', tz='UTC') for h in labels])
    c = pd.Series(base, index=idx)
    return pd.DataFrame({'open': c, 'high': c + 5, 'low': c - 5, 'close': c})


def _preds(tmp_path, rows):
    path = str(tmp_path / 'predictions.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FL.PRED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FL.PRED_FIELDS})
    return path


def _log(cfg2):
    with open(cfg2['gvz']['log'], newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def test_one_observation_at_the_first_pass_after_13_sofia(cfg2, tmp_path):
    _fred_cache(cfg2, {'2026-10-26': 23.55})
    lstm = {'cell': 'h1_lstm', 'key': 'k', 'as_of_bar': '2026-10-27 11:00:00+00:00', 'vol_pct': '0.145'}
    preds = _preds(tmp_path, [lstm])
    h1 = _h1(labels=range(0, 12))
    for local in ('2026-10-27 12:03', '2026-10-27 13:03', '2026-10-27 14:03'):
        GV.observe(cfg2, h1, preds, _sofia(local), local)
    rows = _log(cfg2)
    assert len(rows) == 1 and rows[0]['logged_at'] == '2026-10-27 13:03'
    r = rows[0]
    assert (r['date'], r['gvz'], r['gvz_date'], r['gvz_source']) == ('2026-10-27', '23.55', '2026-10-26', 'FRED')
    assert (r['lstm_vol_pct'], r['lstm_as_of_bar']) == ('0.145', '2026-10-27 11:00:00+00:00')
    assert float(r['gold_close']) == 4000.0 and r['gold_close_bar'] == '2026-10-27 11:00:00+00:00'


def test_no_observation_on_saturday(cfg2, tmp_path):
    assert GV.observe(cfg2, _h1(), _preds(tmp_path, []), _sofia('2026-10-31 14:03')) is None
    assert not os.path.exists(cfg2['gvz']['log'])


def test_missing_forecast_stays_blank(cfg2, tmp_path):
    _fred_cache(cfg2, {'2026-10-26': 23.55})
    old = {'cell': 'h1_lstm', 'key': 'k', 'as_of_bar': '2026-10-26 20:00:00+00:00', 'vol_pct': '0.2'}
    GV.observe(cfg2, _h1(labels=range(0, 12)), _preds(tmp_path, [old]), _sofia('2026-10-27 13:03'))
    fresh = {'cell': 'h1_lstm', 'key': 'k2', 'as_of_bar': '2026-10-27 12:00:00+00:00', 'vol_pct': '0.15'}
    GV.observe(cfg2, _h1(labels=range(0, 13)), _preds(tmp_path, [old, fresh]), _sofia('2026-10-27 14:03'))
    rows = _log(cfg2)
    assert len(rows) == 1 and rows[0]['lstm_vol_pct'] == '' and rows[0]['lstm_as_of_bar'] == ''


def _settled(cfg2):
    path = cfg2['gvz']['settlements']
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def test_full_session_settles_with_its_range_and_leaves_the_log_unchanged(cfg2, tmp_path):
    GV.observe(cfg2, _h1(labels=range(0, 12)), _preds(tmp_path, []), _sofia('2026-10-27 13:03'))
    before = open(cfg2['gvz']['log'], 'rb').read()
    h1 = _h1(labels=range(0, 23))
    h1.loc['2026-10-27 15:00', 'high'] = 4040.0
    h1.loc['2026-10-27 18:00', 'low'] = 3980.0
    out = GV.settle(cfg2, h1, _sofia('2026-10-28 00:03'))
    assert len(out) == 1
    s = _settled(cfg2)[0]
    assert (s['date'], s['range_bars'], float(s['realised_range_pct'])) == ('2026-10-27', '10', 1.5)
    assert open(cfg2['gvz']['log'], 'rb').read() == before
    assert GV.settle(cfg2, h1, _sofia('2026-10-28 01:03')) == []       # settled once


def test_thin_session_settles_blank_with_its_bar_count(cfg2, tmp_path):
    GV.observe(cfg2, _h1(labels=range(0, 12)), _preds(tmp_path, []), _sofia('2026-10-27 13:03'))
    GV.settle(cfg2, _h1(labels=[12, 13, 14, 15, 16, 17, 18, 22]), _sofia('2026-10-28 00:03'))
    s = _settled(cfg2)[0]
    assert (s['realised_range_pct'], s['range_bars']) == ('', '7')


def test_session_not_over_is_not_settled(cfg2, tmp_path):
    GV.observe(cfg2, _h1(labels=range(0, 12)), _preds(tmp_path, []), _sofia('2026-10-27 13:03'))
    assert GV.settle(cfg2, _h1(labels=range(0, 20)), _sofia('2026-10-27 20:03')) == []
    assert _settled(cfg2) == []


# ── never scored (task 3.3) ─────────────────────────────────────────────────

import glob
import hashlib

from src.gold import leaderboard as LB
from src.gold import load_config


def _digests(paths):
    return {p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}


def test_a_month_of_gvz_rows_changes_no_registry_registration_manifest_or_leaderboard(cfg2, tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg['paths']['forward'] = str(tmp_path / 'forward')
    os.makedirs(cfg['paths']['forward'])
    gold = os.path.join(REPO, 'results', 'gold')
    pinned = ([os.path.join(REPO, *p.split('/')) for p in cfg['paths']['registries'].values()]
              + [os.path.join(REPO, *p.split('/')) for p in cfg2['paths']['registries'].values()]
              + glob.glob(os.path.join(gold, 'PRE_REGISTRATION*.json'))
              + glob.glob(os.path.join(gold, 'artifact_manifest*.csv')))
    before, board = _digests(pinned), LB.build(cfg)
    _fred_cache(cfg2, {'2026-09-30': 20.0})
    for day in pd.bdate_range('2026-10-01', '2026-10-30'):
        h1 = _h1(day.strftime('%Y-%m-%d'), labels=range(0, 23))
        for local in ('13:03', '23:03'):
            FL.log_gvz(cfg, local, {'H1': h1}, cfg2=cfg2, now=_sofia(f'{day.date()} {local}'),
                       fred_fetch=_fred_stub({str(day.date() - pd.Timedelta(days=1)): 21.0}),
                       yahoo_fetch=_yahoo_stub({str(day.date() - pd.Timedelta(days=1)): 21.0}))
    assert len(_log(cfg2)) == 22 and len(_settled(cfg2)) == 22
    assert _digests(pinned) == before
    assert LB.build(cfg) == board


def test_scoring_and_leaderboard_never_read_gvz():
    for rel in ('src/gold/forward_report.py', 'src/gold/leaderboard.py'):
        with open(os.path.join(REPO, *rel.split('/')), encoding='utf-8') as fh:
            assert 'gvz' not in fh.read().lower(), rel
