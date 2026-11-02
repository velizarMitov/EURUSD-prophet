"""The second-opinion view (openspec change tsfm-second-opinion, spec
external-forecasts/second-opinion, design D8)."""
import csv
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.external.tsfm import forward_logger as L
from src.external.tsfm import view as VW
from tests.test_gold_mt5_export import TRADE_BG, TRADE_EN

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS_OF = '2026-11-02 10:00:00+00:00'
NOW = pd.Timestamp('2026-11-02 11:20', tz='UTC')            # label clock: the 11:00 bar is open


def _csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _pred(model, inst, size, p_up=0.55, close=3897.67, as_of=AS_OF, base=''):
    return {'key': f'{model}|{inst}|{as_of}', 'model': model, 'instrument': inst, 'as_of_bar': as_of,
            'as_of_close': close, 'config_id': f'{model}:zero_shot:returns:2048', 'size_pct': size, 'p_up': p_up,
            'baseline_size_pct': base, 'phase': 'dry_run', 'logged_at': '2026-11-02T09:01:05+00:00'}


@pytest.fixture
def repo(tmp_path):
    r = str(tmp_path)
    os.makedirs(os.path.join(r, 'src', 'gold'))
    with open(os.path.join(r, 'src', 'gold', 'gold_config_v2.json'), 'w') as fh:
        json.dump({'alerts': {'threshold_vol_pct': 0.151}}, fh)
    L.append_rows(L.out_path('predictions', r), [
        _pred('chronos2', 'gold', 0.18, 0.56), _pred('timesfm25', 'gold', 0.15, 0.44),
        _pred('chronos2', 'eurusd', 0.07, 0.52, 1.16, base=0.061), _pred('timesfm25', 'eurusd', 0.05, 0.48, 1.16, base=0.061)],
        L.PRED_FIELDS)
    _csv(os.path.join(r, 'results', 'gold', 'forward', 'predictions.csv'), [
        {'key': 'k', 'cell': 'h1_lstm', 'cadence': 'H1', 'as_of_bar': AS_OF, 'vol_pct': 0.14, 'p_up': 0.53}])
    _csv(os.path.join(r, 'results', 'forward_eval', 'predictions.csv'), [
        {'key': 'k', 'model': 'h1_gbm', 'cadence': 'H1', 'horizon': 1, 'as_of_bar': AS_OF, 'p_up': 0.49}])
    return r


def _model(view, inst, model):
    return next(m for m in view['instruments'][inst]['models'] if m['model'] == model)


def test_current_gold_forecast_in_sofia_time_and_dollars(repo):
    c = _model(VW.build(repo, NOW), 'gold', 'chronos2')['current']
    assert c['hour_sofia'] == '12:00–13:00'
    assert c['size_pct'] == pytest.approx(0.18)
    assert round(c['size_usd'], 2) == 7.02
    assert c['p_up_status'] == 'not_registered'


def test_gold_size_disagreement_scenario(repo):
    v = VW.build(repo, NOW)
    assert v['threshold_pct'] == pytest.approx(0.151)
    assert _model(v, 'gold', 'chronos2')['agreement']['size'] == 'disagree'     # 0.18 vs h1_lstm 0.14
    assert _model(v, 'gold', 'timesfm25')['agreement']['size'] == 'agree'       # 0.15 and h1_lstm 0.14: both below 0.151
    cur = _model(v, 'gold', 'chronos2')['current']
    assert cur['size_pct'] == pytest.approx(0.18) and cur['inhouse']['size_pct'] == pytest.approx(0.14)


def test_direction_and_eurusd_rules(repo):
    v = VW.build(repo, NOW)
    assert _model(v, 'gold', 'chronos2')['agreement']['direction'] == 'agree'      # 0.56 vs 0.53
    assert _model(v, 'gold', 'timesfm25')['agreement']['direction'] == 'disagree'  # 0.44 vs 0.53
    assert _model(v, 'eurusd', 'chronos2')['agreement'] == {'size': 'above', 'direction': 'disagree'}
    assert _model(v, 'eurusd', 'timesfm25')['agreement'] == {'size': 'below', 'direction': 'agree'}


def test_no_inhouse_forecast_means_nothing_to_compare(repo):
    os.remove(os.path.join(repo, 'results', 'gold', 'forward', 'predictions.csv'))
    assert _model(VW.build(repo, NOW), 'gold', 'chronos2')['agreement'] == {'size': 'nothing', 'direction': 'nothing'}


def test_logger_stopped_shows_nothing_current_and_names_the_last(repo):
    later = NOW + pd.Timedelta(hours=2)
    m = _model(VW.build(repo, later), 'gold', 'chronos2')
    assert m['current'] is None and m['last_as_of'] == AS_OF and m['last_logged_at']


def _settled(n, model='chronos2', inst='gold'):
    t = pd.date_range('2026-11-03', periods=n, freq='h', tz='UTC')
    return [{'key': f'{model}|{inst}|{x}', 'model': model, 'instrument': inst, 'as_of_bar': str(x),
             'realised_abs_pct': 0.1, 'eligible': True, 'size_pct': 0.12, 'baseline_size_pct': 0.13,
             'inhouse_size_pct': 0.11, 'phase': 'dry_run'} for x in t]


def test_twelve_rows_are_too_few_to_say(repo):
    L.append_rows(L.out_path('settlements', repo), _settled(12), L.SETTLE_FIELDS)
    rec = _model(VW.build(repo, NOW), 'gold', 'chronos2')['record']
    assert rec['n'] == 12 and rec['state'] == 'too_few' and 'mae' not in rec
    assert VW.TEXTS['too_few']['en'] == 'too few to say'


def test_a_descriptive_record_with_paired_differences(repo):
    L.append_rows(L.out_path('settlements', repo), _settled(40), L.SETTLE_FIELDS)
    rec = _model(VW.build(repo, NOW), 'gold', 'chronos2')['record']
    assert rec['state'] == 'descriptive' and rec['mae'] == pytest.approx(0.02)
    assert rec['vs_baseline']['difference'] == pytest.approx(0.01)
    assert rec['vs_inhouse']['difference'] == pytest.approx(-0.01)


def test_the_recorded_verdict_is_shown_unchanged(repo):
    verdict = {'cell': 'external_tsfm:chronos2_gold', 'verdict': 'DROP', 'n': 1500, 'ci_low': -0.003,
               'ci_high': 0.001, 'alpha': 0.0125}
    with open(os.path.join(repo, 'results', 'external_tsfm', 'verdicts.json'), 'w') as fh:
        json.dump([verdict], fh)
    assert _model(VW.build(repo, NOW), 'gold', 'chronos2')['record']['verdict'] == verdict


def _all_text():
    out = []
    for d in (VW.TEXTS, VW.LINE_WORDS):
        for v in d.values():
            out += [v['bg'], v['en']]
    return out


def test_every_text_carries_no_trade_words(repo):
    lines = [VW.gold_line('bg', repo, NOW), VW.gold_line('en', repo, NOW)]
    for s in _all_text() + lines:
        assert not TRADE_EN.search(s) and not TRADE_BG.search(s), s


def test_the_honesty_label():
    assert VW.TEXTS['label']['en'] == 'external model — research, no proven edge'
    assert VW.TEXTS['label']['bg']


def test_the_view_writes_nothing(repo):
    def snap():
        return {os.path.join(d, f): os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(repo) for f in fs}
    before = snap()
    VW.build(repo, NOW)
    assert snap() == before


def test_the_view_imports_no_model_library():
    code = ('import sys, src.external.tsfm.view; '
            'print(sorted({m.split(".")[0] for m in sys.modules} & '
            '{"torch", "chronos", "timesfm", "peft", "transformers", "tensorflow", "keras", "xgboost", "arch"}))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]'


# ── the endpoint and the page (task 7.2) ────────────────────────────────────

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


def _mtimes():
    out = {}
    for rel in ('results/external_tsfm', 'results/gold', 'results/forward_eval', 'models'):
        for d, _, fs in os.walk(os.path.join(REPO, rel)):
            for f in fs:
                out[os.path.join(d, f)] = os.path.getmtime(os.path.join(d, f))
    return out


def test_the_endpoint_returns_both_instruments_and_writes_nothing(client):
    before = _mtimes()
    r = client.get('/api/second-opinion')
    assert r.status_code == 200
    body = r.json()
    assert set(body['instruments']) == {'gold', 'eurusd'}
    assert all(len(b['models']) == 2 for b in body['instruments'].values())
    assert body['label']['en'] == 'external model — research, no proven edge'
    assert _mtimes() == before


def test_the_page_is_served_with_the_shell_in_bulgarian(client):
    from src import app_shell as SH
    r = client.get('/second-opinion')
    assert r.status_code == 200
    assert r.text.count(SH.MARKER) == 1 and '<html lang="bg">' in r.text
    assert "fetch('/api/second-opinion'" in r.text


def test_the_more_menu_lists_it_and_marks_it_current():
    import re
    from src import app_shell as SH
    nav = SH.render_nav('/second-opinion')
    assert re.search(r'href="/second-opinion" aria-current="page"', nav)
    assert 'ps-more-current' in nav
    assert '/second-opinion' in SH.render_nav('/gold')


# ── the /gold card (task 7.3) ───────────────────────────────────────────────

def _gold_page():
    with open(os.path.join(REPO, 'static', 'gold.html'), encoding='utf-8') as fh:
        return fh.read()


def test_the_gold_card_texts_exist_in_both_languages_without_trade_words():
    import re
    keys = re.findall(r"^\s+(so_\w+): \{bg: '([^']*)',\s*en: '([^']*)'\}", _gold_page(), re.M)
    pairs = re.findall(r"(so_\w+): \{bg: '([^']*)',\s*\n?\s*en: '([^']*)'\}", _gold_page())
    found = {k for k, _, _ in keys} | {k for k, _, _ in pairs}
    assert found >= {'so_title', 'so_lead', 'so_size', 'so_dir', 'so_agree', 'so_disagree', 'so_nothing',
                     'so_none', 'so_link', 'so_label'}
    for _, bg, en in keys + pairs:
        assert bg.strip() and en.strip()
        assert not TRADE_EN.search(en) and not TRADE_BG.search(bg), (bg, en)


def test_the_gold_card_reads_the_second_opinion_endpoint():
    src = _gold_page()
    assert 'id="second"' in src and "fetch('/api/second-opinion'" in src and 'href="/second-opinion"' in src
    assert 'setInterval(() => { load(); loadHealth(); }, 60e3)' in src        # gold's own refresh is unchanged


def test_the_gold_payload_keys_are_unchanged(client):
    assert sorted(client.get('/api/gold').json()) == ['alert', 'cells', 'clock', 'cost_basis', 'generated_at',
                                                       'gvz', 'label', 'state', 'test_report']


def test_mt5_line_scenario(repo):
    assert VW.gold_line('bg', repo, NOW) == ('Второ мнение: Chronos ±0,18 %, TimesFM ±0,15 % · размер: '
                                             'Chronos не е съгласен, TimesFM е съгласен с h1_lstm')
    assert VW.gold_line('en', repo, NOW) == ('Second opinion: Chronos ±0.18 %, TimesFM ±0.15 % · size: '
                                             'Chronos disagrees, TimesFM agrees with h1_lstm')
    assert VW.gold_line('bg', repo, NOW + pd.Timedelta(hours=2)) == ''
