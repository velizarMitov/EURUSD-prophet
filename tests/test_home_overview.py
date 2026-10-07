"""The plain-language home page (openspec change `friendly-home-page`).

Group 1 covers the read-only summary behind it: the wording rules, the session
tile, the reliability states, and the promise that opening the page writes
nothing. Every case builds its own logs in tmp_path; none touches the live ones.
"""
import os

import pandas as pd
import pytest

from src import home_summary as H

LOG_COLUMNS = ['as_of_date', 'forecasting_date', 'as_of_close', 'pred_direction',
               'pred_return_pct', 'pred_confidence', 'gbm_direction', 'lstm_direction',
               'logged_at', 'h1_direction', 'h1_return_pct', 'h1_agreement',
               'baseline_direction', 'baseline_return_pct', 'baseline_confidence',
               'variant_agreement', 'ti_h1_direction', 'ti_h1_return_pct', 'vol_pred_pct']

# Sofia is UTC+3 in October. The bar label is Berlin, one hour behind Sofia.
IN_SESSION_UTC = '2026-10-02 13:00'      # Fri 16:00 Sofia, label 15:00
FRIDAY_LATE_UTC = '2026-10-02 20:30'     # Fri 23:30 Sofia -> session closed
SATURDAY_UTC = '2026-10-03 12:00'
TUESDAY_MORNING_UTC = '2026-10-06 06:00'  # Tue 09:00 Sofia


def _daily_log(tmp_path, **overrides):
    row = {c: '' for c in LOG_COLUMNS}
    row.update({'as_of_date': '2026-10-05', 'forecasting_date': '2026-10-06',
                'as_of_close': 1.1, 'pred_direction': 'UP', 'pred_confidence': 0.53,
                'logged_at': '2026-10-05T19:00:00+00:00', 'baseline_direction': 'UP',
                'baseline_confidence': 0.53, 'variant_agreement': True, 'vol_pred_pct': 0.42})
    row.update(overrides)
    path = tmp_path / 'prediction_log.csv'
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(path, index=False)
    return str(path)


# ── 1.1 Tomorrow ────────────────────────────────────────────────────────────

def test_near_chance_reads_no_clear_signal(tmp_path):
    log = _daily_log(tmp_path, baseline_confidence=0.515)
    assert H.tomorrow_block(log, now='2026-10-05 20:00')['key'] == 'no_signal'


def test_directional_call_reads_slightly_up(tmp_path):
    b = H.tomorrow_block(_daily_log(tmp_path), now='2026-10-05 20:00')
    assert b['key'] == 'slightly_up' and b['forecasting_date'] == '2026-10-06'
    assert b['outdated'] is False and b['variant'] == 'baseline'


def test_down_call_and_mixed_label():
    t = H._threshold()
    assert H.word_for('DOWN', 0.6, t) == 'slightly_down'
    assert H.word_for('MIXED / LOW CONFIDENCE', 0.9, t) == 'no_signal'
    assert H.word_for('UP', None, t) == 'no_signal'


def test_threshold_is_the_serving_guard_not_a_copy():
    from src.inference import PredictionService
    assert H._threshold() == PredictionService.CONFIDENCE_THRESHOLD == 0.52


def test_a_past_forecast_date_is_outdated(tmp_path):
    b = H.tomorrow_block(_daily_log(tmp_path), now='2026-10-07 09:00')
    assert b['outdated'] is True


def test_variant_disagreement_is_flagged_without_changing_the_word(tmp_path):
    b = H.tomorrow_block(_daily_log(tmp_path, variant_agreement=False), now='2026-10-05 20:00')
    assert b['variants_disagree'] is True and b['key'] == 'slightly_up'


def test_pre_dual_row_falls_back_to_the_macro_committee_and_says_so(tmp_path):
    log = _daily_log(tmp_path, baseline_direction='', baseline_confidence='',
                     pred_direction='DOWN', pred_confidence=0.55)
    b = H.tomorrow_block(log, now='2026-10-05 20:00')
    assert b['key'] == 'slightly_down' and b['variant'] == 'with_macro'


def test_missing_or_empty_log_is_unavailable(tmp_path):
    assert H.tomorrow_block(str(tmp_path / 'nope.csv'))['available'] is False
    empty = tmp_path / 'empty.csv'
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(empty, index=False)
    assert H.tomorrow_block(str(empty))['available'] is False


# ── 1.2 Expected movement ───────────────────────────────────────────────────

def test_movement_shows_the_logged_volatility(tmp_path):
    b = H.movement_block(_daily_log(tmp_path))
    assert b == {'available': True, 'vol_pct': 0.42, 'forecasting_date': '2026-10-06'}


def test_missing_volatility_is_unavailable_never_zero(tmp_path):
    b = H.movement_block(_daily_log(tmp_path, vol_pred_pct=''))
    assert b['available'] is False and 'vol_pct' not in b


# ── 1.3 Today's session ─────────────────────────────────────────────────────

def _fwd(tmp_path, preds):
    out = tmp_path / 'fwd'
    out.mkdir(exist_ok=True)
    pd.DataFrame(preds).to_csv(out / 'predictions.csv', index=False)
    return str(out)


def _pred(model='m15_session_gbm', cadence='M15', horizon=1,
          as_of='2026-10-02 14:45:00+00:00', p_up=0.62):
    return {'key': f'{model}|h{horizon}|{as_of}', 'model': model, 'cadence': cadence,
            'horizon': horizon, 'as_of_bar': as_of, 'p_up': p_up, 'phase': 'dry_run'}


def test_open_session_shows_registered_calls_on_the_owner_clock(tmp_path):
    """As-of label 14:45 -> the move runs label 15:00-15:15, which is
    16:00-16:15 in Sofia."""
    out = _fwd(tmp_path, [_pred(), _pred(model='m15_session_lstm', p_up=0.40)])
    b = H.session_block(now=IN_SESSION_UTC, out=out)
    assert b['open'] is True
    (row,) = b['horizons']
    assert (row['minutes'], row['from'], row['until']) == (15, '16:00', '16:15')
    assert {c['model']: c['key'] for c in row['calls']} == {
        'm15_session_gbm': 'up', 'm15_session_lstm': 'down'}


@pytest.mark.parametrize('pred', [
    _pred(as_of='2026-10-02 12:00:00+00:00'),     # window long over
    _pred(horizon=3),                             # 45 min: not in the registered family
    _pred(p_up=''),                               # no call
], ids=['finished', 'unregistered', 'no-call'])
def test_finished_unregistered_and_empty_cells_are_left_out(tmp_path, pred):
    out = _fwd(tmp_path, [pred])
    assert H.session_block(now=IN_SESSION_UTC, out=out)['horizons'] == []


def test_cells_with_the_same_distance_but_different_windows_stay_apart(tmp_path):
    out = _fwd(tmp_path, [
        _pred(horizon=4, as_of='2026-10-02 14:30:00+00:00'),     # M15: 15:45-16:45 Sofia
        _pred(model='h1_gbm', cadence='H1', horizon=1,
              as_of='2026-10-02 14:00:00+00:00'),               # H1:  16:00-17:00 Sofia
    ])
    rows = H.session_block(now=IN_SESSION_UTC, out=out)['horizons']
    assert len(rows) == 2 and {r['minutes'] for r in rows} == {60}


@pytest.mark.parametrize('now, expected', [
    (FRIDAY_LATE_UTC, '2026-10-05T15:30'),
    (SATURDAY_UTC, '2026-10-05T15:30'),
    (TUESDAY_MORNING_UTC, '2026-10-06T15:30'),
])
def test_closed_session_says_when_it_opens_and_shows_no_direction(tmp_path, now, expected):
    out = _fwd(tmp_path, [_pred()])
    b = H.session_block(now=now, out=out)
    assert b == {'available': True, 'open': False, 'next_open': expected}


# ── 1.4 Reliability ─────────────────────────────────────────────────────────

def _ledger(hits, n, flat=0):
    rows = ([{'direction': 'LONG', 'gross_pips': 10.0}] * hits
            + [{'direction': 'SHORT', 'gross_pips': -4.0}] * (n - hits)
            + [{'direction': 'FLAT', 'gross_pips': 0.0}] * flat)
    return lambda *a, **k: pd.DataFrame(rows, columns=['direction', 'gross_pips'])


@pytest.mark.parametrize('hits, n, key', [(7, 12, 'too_early'), (33, 60, 'coin_range'),
                                          (80, 120, 'above_coin')])
def test_reliability_states(hits, n, key):
    b = H.reliability_block('x.csv', ledger_builder=_ledger(hits, n, flat=5), data_cfg={})
    assert (b['hits'], b['n'], b['key']) == (hits, n, key)


def test_flat_days_are_not_calls():
    b = H.reliability_block('x.csv', ledger_builder=_ledger(3, 4, flat=9), data_cfg={})
    assert b['n'] == 4


def test_right_means_direction_right_before_any_cost():
    """A correct call that moved less than the spread is a paper-trading 'loss'
    but a right call here: the home page judges direction, not cost."""
    small = lambda *a, **k: pd.DataFrame([{'direction': 'LONG', 'gross_pips': 0.8}])
    assert H.reliability_block('x.csv', ledger_builder=small, data_cfg={})['hits'] == 1


def test_wilson_bound_matches_the_textbook_value():
    # p = 0.55, n = 60, z = 1.96: (0.58201 - 0.12989) / 1.06403
    assert H.wilson_lower(33, 60) == pytest.approx(0.4249, abs=1e-3)
    assert H.wilson_lower(0, 0) is None


def test_reliability_exposes_no_cost_field():
    b = H.reliability_block('x.csv', ledger_builder=_ledger(5, 6), data_cfg={})
    assert set(b) == {'available', 'hits', 'n', 'rate', 'lower_95', 'min_settled', 'key'}


# ── 1.5 Assembly ────────────────────────────────────────────────────────────

def _snapshot(*roots, skip=()):
    seen = {}
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if any(os.path.abspath(d).startswith(os.path.abspath(s)) for s in skip):
                continue
            for f in files:
                p = os.path.join(d, f)
                seen[p] = os.stat(p).st_mtime_ns
    return seen


def test_building_the_home_page_writes_nothing():
    """results/forward_eval is skipped: the scheduled logger writes there every
    15 minutes on its own, which would make this test flaky without telling us
    anything about the home page."""
    roots = [os.path.join(H.BASE_DIR, 'results'), os.path.join(H.BASE_DIR, 'models')]
    skip = [os.path.join(H.BASE_DIR, 'results', 'forward_eval')]
    before = _snapshot(*roots, skip=skip)
    H.build_home()
    H.build_home(part='reliability')
    assert _snapshot(*roots, skip=skip) == before


def test_a_failing_block_degrades_only_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(H, 'session_block', lambda *a, **k: 1 / 0)
    d = H.build_home(log_path=_daily_log(tmp_path), now='2026-10-05 20:00')
    assert d['session']['available'] is False and 'ZeroDivisionError' in d['session']['reason']
    assert d['tomorrow']['available'] is True and d['movement']['available'] is True


def test_an_unreachable_price_source_makes_reliability_unavailable(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError('no MT5, no Yahoo')
    monkeypatch.setattr('src.paper_trading.build_ledger', boom)
    d = H.build_home(part='reliability')
    assert d['reliability']['available'] is False and 'ConnectionError' in d['reliability']['reason']


def test_reliability_is_only_served_on_its_own_call(tmp_path):
    fast = H.build_home(log_path=_daily_log(tmp_path))
    assert 'reliability' not in fast and {'tomorrow', 'movement', 'session'} <= set(fast)


# ── 2. Routes ───────────────────────────────────────────────────────────────

REPO = H.BASE_DIR
HOME_HTML = os.path.join(REPO, 'static', 'home.html')
INDEX_HTML = os.path.join(REPO, 'static', 'index.html')


@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


def test_root_serves_the_home_page(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'id="tTomorrow"' in r.text and 'data-i18n="notice"' in r.text
    assert 'Fetch Live Market Data' not in r.text, 'root still serves the research console'


def test_advanced_is_the_old_dashboard_byte_for_byte(client):
    r = client.get('/advanced')
    assert r.status_code == 200
    assert r.content == open(INDEX_HTML, 'rb').read()


def test_api_home_returns_the_fast_blocks(client):
    d = client.get('/api/home').json()
    assert {'generated_at', 'tomorrow', 'movement', 'session'} <= set(d)
    assert 'reliability' not in d


def test_api_home_reliability_part(client):
    d = client.get('/api/home', params={'part': 'reliability'}).json()
    assert set(d) == {'generated_at', 'reliability'}


def test_other_pages_still_answer(client):
    for path in ('/forecasts', '/h1-direction', '/kronos-direction'):
        assert client.get(path).status_code == 200, path


# ── 3. The page ─────────────────────────────────────────────────────────────

def _dictionary():
    """The I18N object from home.html, parsed without a JS engine: one entry per
    line, `key: {bg: '...', en: '...'}` (an entry may wrap onto a second line)."""
    import re
    src = open(HOME_HTML, encoding='utf-8').read()
    block = src.split('const I18N = {', 1)[1].split('\n};', 1)[0]
    entry = re.compile(r"(\w+):\s*\{bg:\s*(['\"])(.*?)\2,\s*en:\s*(['\"])(.*?)\4\}", re.S)
    return {m.group(1): {'bg': m.group(3), 'en': m.group(5)} for m in entry.finditer(block)}


STATE_KEYS = {'no_signal', 'slightly_up', 'slightly_down', 'too_early', 'coin_range',
              'above_coin', 'up', 'down'}


def test_every_label_exists_in_both_languages():
    d = _dictionary()
    assert len(d) >= 40
    blank = [k for k, v in d.items() if not v['bg'].strip() or not v['en'].strip()]
    assert not blank, blank


def test_every_key_the_page_uses_is_in_the_dictionary():
    import re
    src = open(HOME_HTML, encoding='utf-8').read()
    used = set(re.findall(r'data-i18n="(\w+)"', src)) | set(re.findall(r"t\('(\w+)'\)", src))
    missing = (used | STATE_KEYS) - set(_dictionary())
    assert not missing, missing


def test_every_state_the_summary_can_emit_is_translated():
    """Drives the real wording functions, so a new key in home_summary.py that the
    page cannot translate fails here rather than showing a raw key on screen."""
    t = H._threshold()
    emitted = {H.word_for(d, c, t) for d in ('UP', 'DOWN', 'MIXED') for c in (0.4, 0.6)}
    emitted |= {H.reliability_state(h, n) for h, n in ((1, 2), (30, 60), (90, 120))}
    assert emitted <= set(_dictionary())


def test_default_language_is_bulgarian_and_storage_is_guarded():
    src = open(HOME_HTML, encoding='utf-8').read()
    assert '<html lang="bg">' in src and "let lang = 'bg';" in src
    script = src.split('<script>', 1)[1]
    for call in ('localStorage.getItem', 'localStorage.setItem'):
        at = script.index(call)
        assert 'try {' in script[max(0, at - 80):at], f'{call} is not inside try/catch'


def test_page_loads_nothing_external():
    import re
    src = open(HOME_HTML, encoding='utf-8').read()
    assert not re.search(r'<script[^>]+src=', src)
    assert not re.search(r'<link[^>]+href=["\']https?:', src)


BANNED = ('buy', 'sell', 'strong', 'guaranteed', 'profit', 'печалба', 'гарантира')


def _home_text_without_notice():
    d = _dictionary()
    src = open(HOME_HTML, encoding='utf-8').read().lower()
    for lang in ('bg', 'en'):
        src = src.replace(d['notice'][lang].lower(), '')
    return src


def test_page_carries_no_cost_profit_or_advice_words():
    from src.forecast_eval.report import COST_WORDS
    text = _home_text_without_notice()
    for word in COST_WORDS + BANNED:
        assert word not in text, word
    assert 'proven edge' not in text, 'only the negated notice may say "proven edge"'


def test_notice_says_no_edge_and_not_advice_in_both_languages():
    n = _dictionary()['notice']
    assert 'no proven edge' in n['en'].lower() and 'not financial advice' in n['en'].lower()
    assert 'няма доказано предимство' in n['bg'].lower() and 'не е финансов съвет' in n['bg'].lower()
    src = open(HOME_HTML, encoding='utf-8').read()
    notice_tag = src.split('data-i18n="notice"')[0].rsplit('<', 1)[1]
    assert 'notice' in notice_tag and 'button' not in src.split('data-i18n="notice"')[1][:40]


def test_api_home_json_carries_no_cost_words(client):
    import json as _json
    from src.forecast_eval.report import COST_WORDS
    blob = _json.dumps(client.get('/api/home').json()).lower()
    blob += _json.dumps(client.get('/api/home', params={'part': 'reliability'}).json()).lower()
    for word in COST_WORDS + BANNED:
        assert word not in blob, word
