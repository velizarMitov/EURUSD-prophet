"""The operator view (tasks 15.1, 15.2).

Every case builds its own forward log in tmp_path, so nothing touches the live
one.
"""
import os
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import forward_logger as FL
from src.forecast_eval import m15_data as MD
from src.forecast_eval import report as RP

# Sofia is UTC+3 in October (EEST) and UTC+2 in January (EET); the bar label is
# Berlin, one hour behind Sofia in both. 15:30 Sofia is label 14:30 either way.
SUMMER_OPEN_UTC = '2026-10-02 12:30'          # -> 15:30 Sofia, label 14:30
SUMMER_BEFORE_UTC = '2026-10-02 12:20'        # -> 15:20 Sofia, label 14:20
WINTER_OPEN_UTC = '2026-01-07 13:30'          # -> 15:30 Sofia, label 14:30
# Five minutes after the default as-of bar (label 21:45) closed: its forecasts are
# still live. Headline tests use this; a later 'now' would rightly hide them.
FRESH_UTC = '2026-10-01 19:50'                # -> 22:50 Sofia, label 21:50
NEXT_MORNING_UTC = '2026-10-02 06:34'         # -> 09:34 Sofia, label 08:34


def _log(tmp_path, preds=None, settles=None, gaps=None, fails=None):
    out = tmp_path / 'fwd'
    out.mkdir(exist_ok=True)
    for name, frame in (('predictions', preds), ('settlements', settles),
                        ('gaps', gaps), ('failures', fails)):
        if frame is not None:
            pd.DataFrame(frame).to_csv(out / f'{name}.csv', index=False)
    return str(out)


def _head(page: str) -> str:
    """Only the headline region: no CSS from <head>, no folded sections."""
    return page.split('<body>', 1)[1].split('<h2>Колко да вярваш</h2>')[0]


def _pred(model='m15_session_gbm', cadence='M15', horizon=1, as_of='2026-10-01 21:45:00+00:00',
          p_up=0.62, phase='dry_run', key=None):
    return {'key': key or f'{model}|h{horizon}|{as_of}', 'model': model, 'cadence': cadence,
            'horizon': horizon, 'version_id': 'v1', 'variant': 'frozen', 'as_of_bar': as_of,
            'target_offset_bars': horizon, 'p_up': p_up, 'ret_pct': '', 'vol_pct': '',
            'price_source': 'MT5', 'server': 'ActivTradesEU-Server', 'entry_close': 1.1,
            'spread_points': 5.0, 'swap_long': -7.1, 'swap_short': 2.3, 'phase': phase,
            'feature_digest': 'd', 'logged_at': '2026-10-01T19:46:00+00:00'}


# ── 15.1 it renders in every state ─────────────────────────────────────────

def test_renders_with_no_forward_rows(tmp_path):
    path = RP.write_dashboard(_log(tmp_path), now=SUMMER_BEFORE_UTC)
    s = open(path, encoding='utf-8').read()
    assert 'Още няма записана прогноза' in s
    assert 'Още никоя прогноза не е приключила' in s
    assert 'Няма пропуснати барове' in s


def test_renders_dry_run_rows_and_says_they_do_not_count(tmp_path):
    out = _log(tmp_path, preds=[_pred(), _pred(horizon=4, p_up=0.41)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert 'пробен период — нищо не се зачита още' in s
    assert '15 мин' in s and '60 мин' in s
    assert s.count('class="cell"') == 2


def test_direction_arrow_follows_the_probability(tmp_path):
    out = _log(tmp_path, preds=[_pred(p_up=0.62), _pred(horizon=2, p_up=0.37)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '<div class="arrow up">+</div>' in s
    assert '<div class="arrow dn">−</div>' in s
    assert '62.0% нагоре' in s and '37.0% нагоре' in s


def test_a_missing_probability_is_shown_as_unknown_not_as_a_direction(tmp_path):
    out = _log(tmp_path, preds=[_pred(p_up='')])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '<div class="arrow off">—</div>' in s and '— нагоре' in s


# ── the owner's clock ──────────────────────────────────────────────────────

def test_times_are_shown_in_the_owners_clock(tmp_path):
    """The as-of bar is LABEL 21:45, so the move runs from that bar's close,
    label 22:00, to label 22:15 -- which on the owner's clock is 23:00 -> 23:15.
    Showing the label times would be an hour early every day of the year."""
    out = _log(tmp_path, preds=[_pred(as_of='2026-10-01 21:45:00+00:00', horizon=1)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '01.10 23:00 → 01.10 23:15' in s
    assert '01.10 22:00 → 01.10 22:15' not in s, 'label times must not reach the screen'


def test_the_window_spans_the_whole_horizon(tmp_path):
    out = _log(tmp_path, preds=[_pred(cadence='H1', horizon=4, model='h1_gbm',
                                      as_of='2026-10-01 18:00:00+00:00')])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '01.10 20:00 → 02.10 00:00' in s, 'four hours from the as-of bar close'


def test_generated_time_is_the_owners_real_clock(tmp_path):
    s = open(RP.write_dashboard(_log(tmp_path), now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '02.10.2026 15:30 твое време' in s


@pytest.mark.parametrize('now_utc,expect_in', [(SUMMER_BEFORE_UTC, False),
                                               (SUMMER_OPEN_UTC, True),
                                               (WINTER_OPEN_UTC, True)])
def test_the_session_badge_opens_at_1530_sofia_in_both_halves_of_the_year(tmp_path, now_utc,
                                                                          expect_in):
    st = RP.session_state(now_utc)
    assert st['in_session'] is expect_in
    assert st['owner_now'].endswith('15:30') or st['owner_now'].endswith('15:20')
    s = open(RP.write_dashboard(_log(tmp_path), now=now_utc), encoding='utf-8').read()
    assert ('сесията е отворена' in s) is expect_in
    assert ('сесията е затворена' in s) is not expect_in


def test_label_now_is_the_bar_label_not_utc():
    """The bug this guards: feeding a true UTC now into the session window opens
    the session two hours late in summer."""
    label = MD.label_now('2026-10-02 12:30')
    assert label.strftime('%H:%M') == '14:30'
    assert MD.label_now('2026-01-07 13:30').strftime('%H:%M') == '14:30'
    assert MD.session_mask(pd.DatetimeIndex([MD.label_now(SUMMER_OPEN_UTC)]))[0]
    assert not MD.session_mask(pd.DatetimeIndex([pd.Timestamp(SUMMER_OPEN_UTC, tz='UTC')]))[0]


# ── running accuracy is never a verdict ────────────────────────────────────

def _settled(n, model='m15_session_gbm', cadence='M15', horizon=1, hit=lambda i: i % 2):
    preds, settles = [], []
    for i in range(n):
        key = f'{model}-{horizon}-{i}'
        preds.append(_pred(model=model, cadence=cadence, horizon=horizon, key=key,
                           as_of=f'2026-09-{1 + i // 24:02d} {i % 24:02d}:00:00+00:00'))
        settles.append({'key': key, 'settled_at': 'x', 'exit_bar': 'y', 'exit_close': 1.1,
                        'realised_dir': 1.0, 'realised_ret_pct': 0.0, 'realised_vol_pct': 0.0,
                        'correct': float(hit(i)), 'scorable': False,
                        'exclusion_reason': 'phase dry_run'})
    return preds, settles


def test_running_accuracy_is_labelled_interim_with_the_remaining_count(tmp_path):
    preds, settles = _settled(40)
    s = open(RP.write_dashboard(_log(tmp_path, preds=preds, settles=settles),
                                now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert 'междинно — остават 3 777' in s, 'the remaining count must be shown'
    assert '50.0%' in s
    assert 'За решение трябват <b>3 817</b>' in s
    assert 'KEEP' not in s and 'DROP' not in s
    # the remaining count and the stated total must be consistent
    assert 3777 + 40 == 3817


def test_an_accuracy_from_too_few_forecasts_is_not_shown_at_all(tmp_path):
    """One settled forecast reads 100 % or 0 %. Printing that invites exactly the
    wrong conclusion, so the number is withheld until there are enough."""
    preds, settles = _settled(1, hit=lambda i: 1)
    s = open(RP.write_dashboard(_log(tmp_path, preds=preds, settles=settles),
                                now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert f'твърде малко прогнози (под {RP.MIN_SETTLED_TO_SHOW})' in s
    assert '100.0%' not in s


def test_cells_outside_the_family_are_marked_as_never_deciding(tmp_path):
    preds, settles = _settled(40, model='daily_gbm_price', cadence='D1', horizon=1)
    s = open(RP.write_dashboard(_log(tmp_path, preds=preds, settles=settles),
                                now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert 'без решение (извън семейството)' in s
    assert 'междинно — остават' not in s


def test_admitted_cells_are_starred_and_the_others_are_not(tmp_path):
    admitted, alpha, n_req = RP.admitted_cells()
    assert ('m15_session_gbm', 'M15', 1) in admitted
    assert ('m15_session_gbm', 'M15', 8) not in admitted
    assert ('daily_gbm_price', 'D1', 1) not in admitted
    assert alpha == pytest.approx(0.05 / 12) and round(n_req) == 3817
    out = _log(tmp_path, preds=[_pred(horizon=1), _pred(horizon=8)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert s.count('class="star"') == 1, 'only the admitted cell is starred'


# ── the simplified headline ────────────────────────────────────────────────

def test_the_headline_groups_by_horizon_not_by_model(tmp_path):
    """"What happens in the next hour" is the question; a model name is not an
    answer. The owner asked for exactly this (2026-10-02)."""
    out = _log(tmp_path, preds=[
        _pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.44),
        _pred(model='m15_session_lstm', cadence='M15', horizon=1, p_up=0.47),
        _pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.49),
        _pred(model='h1_gbm', cadence='H1', horizon=4, p_up=0.52)])
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    head = _head(s)
    assert 'след 15 минути' in head and 'след 1 час' in head and 'след 4 часа' in head
    assert 'НАДОЛУ' in head and 'НАГОРЕ' in head
    assert 'm15_session_gbm' not in head, 'no model names in the headline'
    assert 'h1_gbm' not in head


def test_models_without_historical_evidence_stay_out_of_the_headline(tmp_path):
    """m15 at 1 hour never cleared a coin on history, so showing it as a call
    would present noise as a forecast."""
    out = _log(tmp_path, preds=[
        _pred(model='m15_session_gbm', cadence='M15', horizon=4, p_up=0.70),
        _pred(model='kronos_direction', cadence='H1', horizon=1, p_up=1.0)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    head = _head(s)
    assert 'няма прогноза' in head
    assert '70%' not in head and '100%' not in head
    assert 'kronos_direction' in s, 'but it is still logged and shown in the details'


def test_long_horizons_are_not_in_the_headline(tmp_path):
    out = _log(tmp_path, preds=[_pred(model='h1_gbm', cadence='H1', horizon=12, p_up=0.55)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    head = _head(s)
    assert 'няма прогноза' in head, '12 hours cannot be traded in one sitting'
    assert RP.minutes_ahead('H1', 12) > RP.HEADLINE_MAX_MINUTES


def test_the_headline_never_combines_models_into_one_call(tmp_path):
    """No ensemble of these models was ever fitted or validated, so a majority
    vote here would be a forecast nobody measured."""
    out = _log(tmp_path, preds=[
        _pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.44),
        _pred(model='m15_session_lstm', cadence='M15', horizon=1, p_up=0.47)])
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    assert s.count('НАДОЛУ') >= 2, 'both calls are listed side by side'
    assert 'и двата модела казват едно и също' in s
    assert 'съгласието не е допълнително доказателство' in s


def test_disagreement_is_said_plainly(tmp_path):
    out = _log(tmp_path, preds=[
        _pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.44),
        _pred(model='m15_session_lstm', cadence='M15', horizon=1, p_up=0.56)])
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    assert 'моделите не са съгласни' in s
    assert 'НАДОЛУ' in s and 'НАГОРЕ' in s


def test_the_trust_block_quotes_a_range_not_the_best_cell(tmp_path):
    """The headline mixes a 51 % cell with a 53 % one; quoting only the best
    would overstate the weaker model."""
    out = _log(tmp_path, preds=[
        _pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.48),
        _pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.49)])
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    assert 'между 51 и 53 от 100' in s
    assert 'Монета познава 50 от 100' in s
    assert 'почти хвърляне на монета' in s


def test_everything_else_is_folded_away(tmp_path):
    out = _log(tmp_path, preds=[_pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.49)])
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    for summary in ('Всички останали модели и хоризонти', 'Точност по клетки',
                    'Състояние на записа', 'Как се чете тази страница'):
        assert f'<summary>{summary}' in s or f'>{summary}<' in s, summary
    assert s.count('<details>') >= 4
    # the headline comes before every folded block
    assert s.index('след 1 час') < s.index('<details>')


# ── volatility models and the historical-evidence tag ─────────────────────

def test_volatility_models_are_not_shown_as_a_direction_call(tmp_path):
    """Their multi-task architecture produces a `p_up` as a by-product, but the
    validated target is the SIZE of the move. Showing it would invent a signal."""
    assert 'vol_ensemble' not in RP.direction_models()
    assert 'kronos_volatility' not in RP.direction_models()
    assert {'h1_gbm', 'm15_session_gbm', 'kronos_direction'} <= RP.direction_models()
    out = _log(tmp_path, preds=[_pred(model='vol_ensemble', cadence='D1', horizon=1,
                                      as_of='2026-10-01 00:00:00+00:00', p_up=0.51),
                                _pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.53)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert 'vol_ensemble' not in s and 'h1_gbm' in s
    assert 'Моделите за волатилност не са тук' in s


def test_the_historical_evidence_tag_defuses_a_confident_looking_number(tmp_path):
    """Kronos reports 90-100 % because it counts sampled paths; the study found no
    interval of it excluding 50 %. The tag is what keeps the two apart."""
    ev = RP.history_evidence()
    assert ev[('h1_gbm', 1)] == 'над монета на историята'
    assert ev[('kronos_direction', 1)] == 'монета на историята'
    assert ev[('m15_session_gbm', 1)] == 'над монета на историята'
    assert ev[('m15_session_gbm', 4)] == 'монета на историята'
    out = _log(tmp_path, preds=[_pred(model='kronos_direction', cadence='H1', horizon=1,
                                      p_up=1.0)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '100.0% нагоре' in s
    assert '<div class="ev no">монета на историята</div>' in s
    assert 'Високият процент не е сила на сигнала' in s


def test_evidence_is_absent_rather_than_guessed_when_no_study_exists(tmp_path):
    assert RP.history_evidence(str(tmp_path / 'nothing.csv')) == {}


def test_admitted_cells_come_first_inside_a_cadence(tmp_path):
    out = _log(tmp_path, preds=[_pred(model='h1_gbm', cadence='H1', horizon=24),
                                _pred(model='h1_gbm', cadence='H1', horizon=1)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert s.index('1 час') < s.index('24 часа'), 'the decidable cell is shown first'


def test_horizon_labels_are_readable_bulgarian():
    assert RP.HORIZON_LABEL['D1'](1) == '1 ден' and RP.HORIZON_LABEL['D1'](5) == '5 дни'
    assert RP.HORIZON_LABEL['H1'](1) == '1 час' and RP.HORIZON_LABEL['H1'](4) == '4 часа'
    assert RP.HORIZON_LABEL['M15'](4) == '60 мин'


# ── no cost arithmetic on the screen ───────────────────────────────────────

def test_no_cost_figure_reaches_the_screen(tmp_path):
    """The owner accounts for trading cost themselves (2026-10-02)."""
    out = _log(tmp_path, preds=[_pred(), _pred(cadence='H1', model='h1_gbm', horizon=2)])
    s = open(RP.write_dashboard(out, now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    for word in RP.COST_WORDS:
        assert word not in s.lower(), word
    assert 'Разходите не са тук' in s, 'and it says so plainly'


def test_the_guard_refuses_to_write_a_page_carrying_a_cost_column(tmp_path, monkeypatch):
    monkeypatch.setattr(RP, 'render', lambda *a, **k: '<html>breakeven 53.0%</html>')
    with pytest.raises(AssertionError, match='cost arithmetic'):
        RP.write_dashboard(_log(tmp_path), now=SUMMER_OPEN_UTC)


def test_the_cost_columns_stay_in_the_log(tmp_path):
    """Kept in the data, kept off the screen -- the methodology needs a measured
    cost to exist even when it decides nothing."""
    out = _log(tmp_path, preds=[_pred()])
    RP.write_dashboard(out, now=SUMMER_OPEN_UTC)
    cols = pd.read_csv(os.path.join(out, 'predictions.csv')).columns
    assert 'spread_points' in cols and 'swap_long' in cols


# ── 15.2 it never breaks logging, and serving is untouched ─────────────────

def test_a_view_failure_does_not_break_a_logging_run(tmp_path, monkeypatch):
    out = _log(tmp_path, preds=[_pred()])
    monkeypatch.setattr(RP, 'write_dashboard',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    assert FL.refresh_view(out) is None
    f = pd.read_csv(os.path.join(out, 'failures.csv'))
    assert (f['model'] == 'dashboard').any() and f['error'].str.contains('boom').any()


def test_the_view_is_rewritten_by_a_run(tmp_path):
    out = _log(tmp_path, preds=[_pred()])
    assert FL.refresh_view(out) == os.path.join(out, 'dashboard.html')
    assert os.path.exists(os.path.join(out, 'dashboard.html'))


def test_serving_changes_are_additive_and_leave_the_models_alone():
    """The owner lifted the api.py restriction on 2026-10-05 so the view can live
    on :8000. What stays true: src/inference.py and src/paper_trading.py are
    byte-identical, and api.py / static/index.html only GAIN lines -- the
    additive-only contract of test_external_kronos holds."""
    for rel in ('src/inference.py', 'src/paper_trading.py'):
        r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel],
                           cwd=RP.REPO, capture_output=True)
        assert r.returncode == 0, f'{rel} changed'
    out = subprocess.run(['git', 'diff', '--numstat', 'HEAD', '--', 'api.py', 'static/index.html'],
                         cwd=RP.REPO, capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        added, removed, path = line.split('\t')
        assert removed == '0', f'{path} deleted {removed} lines; this change is additive only'


def test_the_forecasts_route_lives_in_the_application():
    import api
    routes = {r.path for r in api.app.routes}
    assert '/forecasts' in routes
    html_ = api.forward_forecasts_page().body.decode('utf-8')
    assert '<h1>EUR/USD</h1>' in html_
    assert 'href="/">← към таблото</a>' in html_, 'a way back to the dashboard'
    for word in RP.COST_WORDS:
        assert word not in html_.lower(), word


def test_the_dashboard_links_to_the_forecasts_page():
    index = open(os.path.join(RP.REPO, 'static', 'index.html'), encoding='utf-8').read()
    assert 'href="/forecasts"' in index


def test_a_forecast_view_failure_is_a_500_and_does_not_break_other_routes(monkeypatch):
    import api
    from fastapi import HTTPException
    monkeypatch.setattr(RP, 'build_page',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(HTTPException) as e:
        api.forward_forecasts_page()
    assert e.value.status_code == 500 and 'boom' in e.value.detail
    monkeypatch.undo()
    assert api.provenance_status()['n_pinned'] >= 40, 'the rest of the app is unaffected'


def test_the_standalone_page_has_no_link_home():
    """A file opened from disk, or the :8001 viewer, has nowhere to link back to."""
    s = open(RP.write_dashboard(now=SUMMER_OPEN_UTC), encoding='utf-8').read()
    assert '← към таблото' not in s


def test_the_report_module_does_not_import_the_serving_path():
    """Checked on the parsed imports, not on the text: the docstring names those
    modules precisely to say it does not touch them."""
    import ast
    tree = ast.parse(open(RP.__file__, encoding='utf-8').read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or '')
    assert names, 'the module does import something'
    for n in names:
        assert not re.search(r'(^|\.)(api|inference|paper_trading)(\.|$)', n), n


# ── 15.3 the local viewer ──────────────────────────────────────────────────

def _get(url, timeout=10):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode('utf-8'), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf-8', 'replace'), dict(e.headers)


@pytest.fixture
def viewer(tmp_path):
    import threading
    out = _log(tmp_path, preds=[_pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.53)])
    httpd, url = RP.serve(0, out, serve_forever=False)       # port 0 = any free port
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield url, out, httpd
    httpd.shutdown()
    httpd.server_close()


def test_the_viewer_serves_the_page(viewer):
    url, _out, _httpd = viewer
    status, body, headers = _get(url)
    assert status == 200
    assert headers['Content-Type'] == 'text/html; charset=utf-8'
    assert headers['Cache-Control'] == 'no-store'
    assert '<h1>EUR/USD</h1>' in body and 'h1_gbm' in body


def test_the_viewer_re_renders_so_a_reload_shows_new_rows(viewer):
    """It renders per request rather than serving the last written file."""
    url, out, _httpd = viewer
    assert 'm15_session_gbm' not in _get(url)[1]
    pd.DataFrame([_pred(model='h1_gbm', cadence='H1', horizon=1, p_up=0.53),
                  _pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.58)]
                 ).to_csv(os.path.join(out, 'predictions.csv'), index=False)
    assert 'm15_session_gbm' in _get(url)[1]


def test_the_viewer_is_read_only_and_has_no_other_routes(viewer):
    import urllib.error
    import urllib.request
    url, _out, _httpd = viewer
    assert _get(url + 'anything-else')[0] == 404
    req = urllib.request.Request(url, data=b'x', method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            got = r.status
    except urllib.error.HTTPError as e:
        got = e.code
    assert got == 405


def test_the_viewer_binds_only_to_loopback(viewer):
    """These are the owner's private forecasts; nothing here is for the network."""
    _url, _out, httpd = viewer
    assert httpd.server_address[0] == '127.0.0.1'


def test_a_render_failure_becomes_a_500_not_a_crash(viewer, monkeypatch):
    url, _out, _httpd = viewer
    monkeypatch.setattr(RP, 'build_page',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    status, body, _h = _get(url)
    assert status == 500 and 'boom' in body
    monkeypatch.undo()
    assert _get(url)[0] == 200, 'and it keeps serving afterwards'


def test_the_viewer_does_not_import_the_served_application():
    import ast
    src = open(RP.__file__, encoding='utf-8').read()
    tree = ast.parse(src)
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or '')
    assert not any(re.search(r'(^|\.)(api|fastapi|uvicorn|inference|paper_trading)(\.|$)', m)
                   for m in mods), mods


def test_scheduled_tasks_run_without_a_console_window():
    """Regression: the tasks called run_module.cmd directly, and cmd.exe is a
    console program, so Windows flashed a window every 15 minutes. They must go
    through wscript.exe //B and a hidden (style 0) launcher."""
    d = os.path.join(RP.REPO, 'scripts', 'forecast_eval')
    install = open(os.path.join(d, 'install_tasks.ps1'), encoding='utf-8').read()
    assert 'wscript.exe' in install and '//B' in install and 'run_hidden.vbs' in install
    assert "'run_module.cmd'" not in install, 'the task must not call the cmd runner directly'
    vbs = open(os.path.join(d, 'run_hidden.vbs'), encoding='utf-8').read()
    assert re.search(r'\.Run\(cmd, 0, True\)', vbs), 'window style 0 = hidden, wait for exit'
    assert 'run_module.cmd' in vbs
    # schtasks allows 261 characters in /TR
    cmd = f'"C:\\Windows\\System32\\wscript.exe" //B //Nologo "{os.path.join(d, "run_hidden.vbs")}" ' \
          'src.forecast_eval.forward_logger EURUSDProphet-ForwardLogger'
    assert len(cmd) <= 261, len(cmd)


def test_the_page_is_self_contained():
    s = open(RP.write_dashboard(now=SUMMER_OPEN_UTC,
                                path=os.path.join(RP.FWD, 'dashboard.html')),
             encoding='utf-8').read()
    assert '<style>' in s and 'src="http' not in s and 'href="http' not in s
    assert re.search(r'<meta charset="utf-8">', s)


def test_a_window_that_has_ended_is_not_shown_as_a_current_forecast(tmp_path):
    """Owner report, 2026-10-07 09:34: the headline still showed the previous
    evening's 23:00-23:15 call as "след 15 минути". A finished window is history."""
    out = _log(tmp_path, preds=[_pred(model='m15_session_gbm', cadence='M15', horizon=1, p_up=0.41)])
    fresh = _head(open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read())
    assert 'след 15 минути' in fresh and '23:00 → 01.10 23:15' in fresh
    stale = _head(open(RP.write_dashboard(out, now=NEXT_MORNING_UTC), encoding='utf-8').read())
    assert 'след 15 минути' not in stale and '23:15' not in stale
    assert 'няма прогноза' in stale


def test_newest_forecast_with_no_probability_is_not_filled_from_an_older_one(tmp_path):
    """groupby().last() takes the last NON-NULL value per column, so a newest
    row with a blank p_up borrowed the previous row's 0.70 and showed it as the
    current call. The newest row must be taken whole."""
    out = _log(tmp_path, preds=[
        _pred(as_of='2026-10-01 21:30:00+00:00', p_up=0.70),
        _pred(as_of='2026-10-01 21:45:00+00:00', p_up='')])
    latest = RP.latest_forecasts(RP._read('predictions', out))
    (row,) = latest.itertuples()
    assert str(row.as_of).startswith('2026-10-01 21:45') and row.direction == '—'
    s = open(RP.write_dashboard(out, now=FRESH_UTC), encoding='utf-8').read()
    assert '70.0%' not in s and '70%' not in _head(s)
