"""Live gold leaderboard (openspec change gold-forecasting-v2, spec gold-forecasting/leaderboard)."""
import copy
import csv
import os

import pytest

from src.gold import forward_logger as FL
from src.gold import history as H
from src.gold import leaderboard as LB
from src.gold import load_config

CFG = load_config()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['forward'] = str(tmp_path)
    H._CACHE.clear()
    return c


def _bars(n, day0=1):
    return [f'2026-09-{day0 + i // 24:02d} {i % 24:02d}:00:00+00:00' for i in range(n)]


def _log(cfg, cell, bars, digest, v2=False, shadow=False, vol=0.1):
    preds, setts = [], []
    for i, b in enumerate(bars):
        key = f'{cell}|H1|{b}'
        preds.append({'key': key, 'cell': cell, 'cadence': 'H1', 'as_of_bar': b, 'as_of_close': 4000,
                      'p_up': '' if 'garch' in cell else 0.6, 'ret_pct': '' if 'garch' in cell else 0.01,
                      'vol_pct': vol, 'model_digest': digest, 'phase': 'dry_run', 'logged_at': 'x',
                      'version': '20261201' if shadow else 'v2', 'shadow': '1' if shadow else '0'})
        setts.append({'key': key, 'cell': cell, 'cadence': 'H1', 'as_of_bar': b, 'exit_bar': 'x',
                      'exit_close': 4001, 'realised_dir': i % 2, 'realised_ret_pct': 0.05,
                      'realised_vol_pct': 0.05, 'eligible': 'True'})
    if v2:
        fields = FL.PRED_FIELDS + ['version', 'shadow']
        FL.append_rows(FL.out_path(cfg, 'predictions_v2'), preds, fields)
        FL.append_rows(FL.out_path(cfg, 'settlements_v2'), setts, FL.SETTLE_FIELDS)
    else:
        FL.append_rows(FL.out_path(cfg, 'predictions'), preds, FL.PRED_FIELDS)
        FL.append_rows(FL.out_path(cfg, 'settlements'), setts, FL.SETTLE_FIELDS)


def test_two_digests_make_two_rows_and_a_shadow_version_is_flagged(cfg):
    _log(cfg, 'h1_lstm', _bars(40), 'A')
    _log(cfg, 'h1_lstm', _bars(40, day0=5), 'S', v2=True, shadow=True)
    rows = LB.build(cfg)['rows']
    by = {r['model_digest']: r for r in rows}
    assert set(by) == {'A', 'S'} and by['S']['shadow'] is True and by['A']['shadow'] is False
    assert all(r['kind'] == 'descriptive — not a verdict' for r in rows)


def test_rows_under_30_say_too_few_and_sort_last(cfg):
    _log(cfg, 'h1_lstm', _bars(40), 'A')
    _log(cfg, 'h1_gbm', _bars(10), 'G')
    rows = LB.build(cfg, sort='hit_rate')['rows']
    assert rows[-1]['cell'] == 'h1_gbm' and rows[-1]['state'] == 'too_few' and 'hit_rate' not in rows[-1]
    assert 'KEEP' not in str(rows) and 'DROP' not in str(rows)


def test_garch_baseline_and_alerts_are_compared(cfg):
    _log(cfg, 'h1_lstm', _bars(40), 'A', vol=0.06)
    _log(cfg, 'h1_garch_hod', _bars(40), 'G', vol=0.2)
    FL.append_rows(FL.out_path(cfg, 'alerts'), [{'key': f'h1_lstm|H1|{_bars(1)[0]}'}], ['key'])
    out = LB.build(cfg, sort='vol_error_gain')
    lstm = [r for r in out['rows'] if r['cell'] == 'h1_lstm'][0]
    assert lstm['vol_baseline'] == 'h1_garch_hod' and lstm['vol_error_gain'] > 0
    assert out['alerts']['alerted_n'] == 1 and out['alerts']['other_n'] == 39


def test_unknown_sort_is_refused(cfg):
    with pytest.raises(ValueError, match='allowed'):
        LB.build(cfg, sort='profit')


def test_building_the_board_writes_nothing(cfg):
    _log(cfg, 'h1_lstm', _bars(40), 'A')
    before = {f: os.path.getmtime(os.path.join(cfg['paths']['forward'], f)) for f in os.listdir(cfg['paths']['forward'])}
    LB.build(cfg)
    after = {f: os.path.getmtime(os.path.join(cfg['paths']['forward'], f)) for f in os.listdir(cfg['paths']['forward'])}
    assert before == after


def test_endpoint_answers_and_refuses_bad_sort():
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get('/api/gold/leaderboard').status_code == 200
    assert c.get('/api/gold/leaderboard', params={'sort': 'profit'}).status_code == 400


def test_leaderboard_tab_is_bilingual_and_reads_the_endpoint():
    src = open(os.path.join(REPO, 'static', 'gold.html'), encoding='utf-8').read()
    assert "'/api/gold/leaderboard?sort='" in src and 'id="panel-board"' in src
    for key in ('tab_board', 'lb_note', 'lb_model', 'lb_version', 'lb_hit', 'lb_vol', 'lb_cov', 'lb_shadow',
                'lb_registered', 'lb_alerts', 'lb_alerts_text', 'lb_empty'):
        assert f'{key}: {{bg: ' in src, key
