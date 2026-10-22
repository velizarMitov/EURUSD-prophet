"""Gold history (openspec change app-shell-and-gold-history, spec gold-forecasting/gold-history)."""
import copy
import csv
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.gold import history as H
from src.gold import load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_WORDS = ('spread', 'pips', 'p&l', 'pnl', 'profit', 'equity')

PRED = ['key', 'cell', 'cadence', 'as_of_bar', 'as_of_close', 'p_up', 'ret_pct', 'vol_pct',
        'implied_price', 'model_digest', 'phase', 'logged_at']
SETT = ['key', 'cell', 'cadence', 'as_of_bar', 'exit_bar', 'exit_close', 'realised_dir',
        'realised_ret_pct', 'realised_vol_pct', 'eligible', 'phase', 'settled_at']
GAPS = ['cadence', 'as_of_bar', 'reason', 'logged_at']


def _write(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def _key(cell, cad, bar):
    return f'{cell}|{cad}|{bar}'


@pytest.fixture()
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    (tmp_path / 'forward').mkdir()
    c['paths']['forward'] = str(tmp_path / 'forward')
    c['paths']['results'] = str(tmp_path)
    H._CACHE.clear()
    return c


def _pred(cell, bar, close, p_up='', ret='', vol='', digest='A'):
    return {'key': _key(cell, 'H1', bar), 'cell': cell, 'cadence': 'H1', 'as_of_bar': bar,
            'as_of_close': close, 'p_up': p_up, 'ret_pct': ret, 'vol_pct': vol,
            'model_digest': digest, 'phase': 'dry_run', 'logged_at': '2026-10-21T16:03:04+00:00'}


def _sett(cell, bar, exit_bar, exit_close, rdir, rret, eligible='True'):
    return {'key': _key(cell, 'H1', bar), 'cell': cell, 'cadence': 'H1', 'as_of_bar': bar,
            'exit_bar': exit_bar, 'exit_close': exit_close, 'realised_dir': rdir,
            'realised_ret_pct': rret, 'realised_vol_pct': abs(float(rret)), 'eligible': eligible}


B16, B17, B18 = ('2026-10-21 16:00:00+00:00', '2026-10-21 17:00:00+00:00', '2026-10-21 18:00:00+00:00')


def _live_fixture(cfg, preds, setts=(), gaps=()):
    fwd = cfg['paths']['forward']
    _write(os.path.join(fwd, 'predictions.csv'), PRED, preds)
    _write(os.path.join(fwd, 'settlements.csv'), SETT, setts)
    _write(os.path.join(fwd, 'gaps.csv'), GAPS, gaps)


def _row(rows, cell, bar):
    hit = rows[(rows['cell'] == cell) & (rows['as_of_bar'] == bar)]
    assert len(hit) == 1, hit
    return hit.iloc[0]


# ── 3.1 live history ────────────────────────────────────────────────────────

def test_settled_forecast_is_paired_with_its_own_outcome(cfg):
    _live_fixture(cfg, [_pred('h1_lstm', B16, 4120.00, p_up=0.47, ret=-0.01, vol=0.11)],
                  [_sett('h1_lstm', B16, B17, 4131.48, 1, 0.2783)])
    r = _row(H.live(cfg), 'h1_lstm', B16)
    assert r['predicted_dir'] == 'down' and round(r['probability'], 2) == 0.53
    assert r['realised_dir'] == 1 and r['actual_price'] == 4131.48
    assert r['status'] == 'settled' and r['hit'] == False  # noqa: E712  a miss


def test_no_settlement_is_joined_to_another_key(cfg):
    _live_fixture(cfg, [_pred('h1_lstm', B16, 4120.0, p_up=0.6), _pred('h1_gbm', B16, 4120.0, p_up=0.6)],
                  [_sett('h1_gbm', B16, B17, 4131.48, 1, 0.28)])
    rows = H.live(cfg)
    assert _row(rows, 'h1_gbm', B16)['status'] == 'settled'
    lstm = _row(rows, 'h1_lstm', B16)
    assert lstm['status'] == 'pending' and lstm['actual_price'] != lstm['actual_price']   # NaN


def test_a_blank_volatility_stays_blank(cfg):
    _live_fixture(cfg, [_pred('h1_gbm', B16, 4120.0, p_up=0.6, ret=0.01),
                        _pred('h1_lstm', B16, 4120.0, p_up=0.6, ret=0.01, vol=0.11)])
    rec = [r for r in H.records(H.live(cfg)) if r['cell'] == 'h1_gbm'][0]
    assert rec['vol_pct'] is None and rec['move_error'] is None


def test_newest_hour_is_pending_and_left_out_of_summaries(cfg):
    preds = [_pred('h1_lstm', f'2026-10-{d:02d} {h:02d}:00:00+00:00', 4000.0, p_up=0.6, ret=0.01, vol=0.1)
             for d in (19, 20) for h in range(0, 24)]
    setts = [_sett('h1_lstm', p['as_of_bar'], 'x', 4001.0, 1, 0.025) for p in preds[:-1]]
    _live_fixture(cfg, preds, setts)
    rows = H.live(cfg)
    assert (rows['status'] == 'pending').sum() == 1
    [s] = H.summaries(rows)
    assert s['n'] == len(setts) and s['kind'] == 'descriptive — not a verdict'


def test_a_gap_is_listed_as_missed(cfg):
    _live_fixture(cfg, [_pred('h1_lstm', B16, 4120.0, p_up=0.6)],
                  gaps=[{'cadence': 'H1', 'as_of_bar': B17, 'reason': 'no bar', 'logged_at': 'x'}])
    rows = H.live(cfg)
    missed = rows[rows['status'] == 'missed']
    assert set(missed['cell']) == {'h1_gbm', 'h1_lstm', 'h1_garch_hod'}
    assert set(missed['as_of_bar']) == {B17}


def test_ineligible_settlement_is_not_scored_but_shows_its_outcome(cfg):
    _live_fixture(cfg, [_pred('h1_lstm', B16, 4120.0, p_up=0.6)],
                  [_sett('h1_lstm', B16, B17, 4131.48, 1, 0.28, eligible='False')])
    r = _row(H.live(cfg), 'h1_lstm', B16)
    assert r['status'] == 'not_scored' and r['actual_price'] == 4131.48 and pd.isna(r['hit'])


# ── 3.2 study history ───────────────────────────────────────────────────────

STUDY_FIELDS = ['time', 'close', 'eligible', 'y_dir', 'y_ret', 'y_vol', 'h1_gbm__p_up', 'h1_gbm__ret_pct',
                'h1_lstm__p_up', 'h1_lstm__ret_pct', 'h1_lstm__vol_pct', 'h1_garch_hod__p_up',
                'h1_garch_hod__vol_pct']


def test_study_row_shows_predicted_and_actual_price(cfg):
    _write(os.path.join(cfg['paths']['results'], 'test_predictions_h1.csv'), STUDY_FIELDS, [{
        'time': '2024-06-10 14:00:00+00:00', 'close': 2306.03, 'eligible': 'True', 'y_dir': 0,
        'y_ret': -0.0216846, 'y_vol': 0.0216846, 'h1_gbm__p_up': 0.4246, 'h1_gbm__ret_pct': -0.0006,
        'h1_lstm__p_up': 0.4559, 'h1_lstm__ret_pct': -0.0117, 'h1_lstm__vol_pct': 0.1297,
        'h1_garch_hod__vol_pct': 0.3537}])
    r = _row(H.study(cfg, 'test', 'H1'), 'h1_lstm', '2024-06-10 14:00:00+00:00')
    assert r['implied_price'] == 2305.76 and r['actual_price'] == 2305.53
    assert r['hit'] == True and r['status'] == 'settled'  # noqa: E712  predicted down, went down


def test_real_test_block_round_trips_its_row_count():
    path = os.path.join(REPO, 'results', 'gold', 'test_predictions_h1.csv')
    if not os.path.exists(path):
        pytest.skip('no saved test predictions')
    with open(path, encoding='utf-8') as fh:
        n = sum(1 for _ in fh) - 1
    rows = H.study(load_config(), 'test', 'H1')
    assert len(rows) == 3 * n
    assert set(rows.groupby('cell').size()) == {n}


def test_study_notices_say_what_they_are():
    assert 'not used to choose' in H.NOTICES['test']['en']
    assert 'first 80 %' in H.NOTICES['test']['en'] and 'first 70 %' in H.NOTICES['validation']['en']
    for src in H.SOURCES:
        assert H.NOTICES[src]['bg'].strip() and H.NOTICES[src]['en'].strip()


# ── 3.3 summaries and the side-by-side view ─────────────────────────────────

def test_wilson_interval_matches_the_textbook_value():
    lo, hi = H.wilson(55, 100)
    assert round(lo, 4) == 0.4524 and round(hi, 4) == 0.6439


def _many(cell, n, digest, start_day=1, vol=0.1):
    preds, setts = [], []
    for i in range(n):
        bar = f'2026-09-{start_day + i // 24:02d} {i % 24:02d}:00:00+00:00'
        preds.append(_pred(cell, bar, 4000.0, p_up=0.6, ret=0.01, vol=vol, digest=digest))
        setts.append(_sett(cell, bar, 'x', 4001.0, 1 if i % 2 else 0, 0.05))
    return preds, setts


def test_two_digests_give_two_summaries(cfg):
    a, sa = _many('h1_lstm', 40, 'A', start_day=1)
    b, sb = _many('h1_lstm', 40, 'B', start_day=5)
    _live_fixture(cfg, a + b, sa + sb)
    got = {(s['cell'], s['model_digest']): s['n'] for s in H.summaries(H.live(cfg))}
    assert got == {('h1_lstm', 'A'): 40, ('h1_lstm', 'B'): 40}


def test_29_rows_are_too_few(cfg):
    p, s = _many('h1_lstm', 29, 'A')
    _live_fixture(cfg, p, s)
    [summ] = H.summaries(H.live(cfg))
    assert summ['state'] == 'too_few' and 'hit_rate' not in summ


def test_garch_pairing_uses_only_shared_bars(cfg):
    lstm, sl = _many('h1_lstm', 40, 'A', vol=0.1)
    garch, sg = _many('h1_garch_hod', 60, 'G', vol=0.3)
    for p in garch:
        p['p_up'] = p['ret_pct'] = ''
    _live_fixture(cfg, lstm + garch, sl + sg)
    s = [x for x in H.summaries(H.live(cfg)) if x['cell'] == 'h1_lstm'][0]
    assert s['n_shared'] == 40 and s['vol_baseline'] == 'h1_garch_hod'
    assert round(s['vol_mae_shared'], 6) == 0.05 and round(s['vol_mae_baseline'], 6) == 0.25


def test_wide_puts_every_model_of_a_bar_on_one_line(cfg):
    _live_fixture(cfg, [_pred('h1_gbm', B17, 4131.0, p_up=0.48, ret=0.0),
                        _pred('h1_lstm', B17, 4131.0, p_up=0.49, ret=0.0, vol=0.11),
                        _pred('h1_garch_hod', B17, 4131.0, vol=0.14)],
                  [_sett(c, B17, B18, 4135.0, 1, 0.0968) for c in ('h1_gbm', 'h1_lstm', 'h1_garch_hod')])
    [line] = H.wide(H.live(cfg))
    assert set(line['forecasts']) == {'h1_gbm', 'h1_lstm', 'h1_garch_hod'}
    assert line['realised']['actual_price'] == 4135.0


# ── 3.4 read-only, isolated, fresh ──────────────────────────────────────────

def _tree_state():
    out = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', '__pycache__', 'research_models', 'mlruns')]
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            out[p] = (st.st_mtime_ns, st.st_size)
    return out


def test_building_every_source_writes_nothing():
    before = _tree_state()
    for src in H.SOURCES:
        for cad in H.CADENCES:
            H.query(src, cad, limit=5)
            H.query(src, cad, layout='wide', limit=5)
    after = _tree_state()
    # The two scheduled loggers append to their own folders on the clock; that
    # is them, not this read.
    scheduled = tuple(os.path.join(REPO, *p) for p in (('results', 'gold', 'forward'), ('results', 'forward_eval')))
    changed = [p for p in after if (p not in before or after[p] != before[p]) and not p.startswith(scheduled)]
    assert not changed, changed


def test_history_imports_no_model_library():
    code = ('import sys; from src.gold import history; history.query("live"); history.query("test", "D1"); '
            'bad = [m for m in ("tensorflow", "keras", "xgboost", "arch") if m in sys.modules]; '
            'print(bad); sys.exit(1 if bad else 0)')
    r = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_an_appended_row_shows_up_on_the_next_call(cfg):
    _live_fixture(cfg, [_pred('h1_lstm', B16, 4120.0, p_up=0.6)])
    assert len(H.live(cfg)) == 1
    with open(os.path.join(cfg['paths']['forward'], 'predictions.csv'), 'a', newline='', encoding='utf-8') as fh:
        csv.DictWriter(fh, fieldnames=PRED).writerow({k: v for k, v in _pred('h1_lstm', B17, 4121.0, p_up=0.4).items()})
    assert len(H.live(cfg)) == 2


# ── 4.1 the endpoint ────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


def test_endpoint_pages_the_h1_test_block(client):
    path = os.path.join(REPO, 'results', 'gold', 'test_predictions_h1.csv')
    if not os.path.exists(path):
        pytest.skip('no saved test predictions')
    first = client.get('/api/gold/history', params={'source': 'test', 'cadence': 'H1', 'limit': 200}).json()
    assert len(first['rows']) == 200 and first['total'] > 10000 and first['next_offset'] == 200
    second = client.get('/api/gold/history', params={'source': 'test', 'cadence': 'H1', 'limit': 200,
                                                     'offset': first['next_offset']}).json()
    assert second['rows'][0] != first['rows'][-1]
    assert second['rows'][0]['as_of_bar'] <= first['rows'][-1]['as_of_bar']    # newest first, continued
    assert 'not used to choose' in first['source_notice']['en']
    assert all(s['kind'] == 'descriptive — not a verdict' for s in first['summaries'])


@pytest.mark.parametrize('params', [{'source': 'backtest'}, {'cadence': 'M15'}, {'cell': 'd1_gbm'},
                                    {'layout': 'grid'}, {'limit': 0}, {'limit': 5000}, {'start': 'yesterday-ish'}])
def test_bad_parameters_answer_400_with_the_allowed_values(client, params):
    r = client.get('/api/gold/history', params=params)
    assert r.status_code == 400, (params, r.status_code)
    assert 'allowed' in r.json()['detail']


def test_payload_carries_no_cost_words(client):
    for src in H.SOURCES:
        for layout in H.LAYOUTS:
            text = client.get('/api/gold/history', params={'source': src, 'layout': layout, 'limit': 50}).text.lower()
            for w in COST_WORDS:
                assert w not in text, (src, layout, w)


# ── 5. the /gold page source ────────────────────────────────────────────────

def _gold_src():
    with open(os.path.join(REPO, 'static', 'gold.html'), encoding='utf-8') as fh:
        return fh.read()


def _gold_dictionary():
    import re
    block = _gold_src().split('const I18N = {', 1)[1].split('\n};', 1)[0]
    # A JS single-quoted string, escapes allowed (one value is "a coin\'s range").
    lit = r"'((?:[^'\\]|\\.)*)'"
    entry = re.compile(r"(\w+|'[^']+'):\s*\{\s*bg:\s*" + lit + r",\s*en:\s*" + lit + r"\s*\}", re.S)
    return {m.group(1).strip("'"): {'bg': m.group(2), 'en': m.group(3)} for m in entry.finditer(block)}


def test_every_gold_label_exists_in_both_languages():
    import re
    d = _gold_dictionary()
    assert len(d) > 80
    for k, v in d.items():
        assert v['bg'].strip() and v['en'].strip(), k
    src = _gold_src()
    for key in set(re.findall(r'data-i18n="([^"]+)"', src)) | set(re.findall(r"\bt\('([a-z_0-9]+)'\)", src)):
        assert key in d, f'{key} is used but not in the dictionary'


def test_gold_page_loads_nothing_external_and_names_no_cost():
    import re
    src = _gold_src()
    assert not re.search(r'''(?:src|href)\s*=\s*["']https?://|@import''', src, re.I)
    lower = src.lower()
    for w in COST_WORDS:
        assert w not in lower, w


def test_gold_page_reads_the_history_and_offers_the_side_by_side_layout():
    src = _gold_src()
    assert "fetch('/api/gold'" in src and "'/api/gold/history?'" in src
    assert 'value="wide"' in src and 'downloadCSV' in src
    assert 'descriptive' in _gold_dictionary() and _gold_dictionary()['descriptive']['en'] == 'descriptive — not a verdict'
