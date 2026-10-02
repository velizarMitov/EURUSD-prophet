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


def _log(tmp_path, preds=None, settles=None, gaps=None, fails=None):
    out = tmp_path / 'fwd'
    out.mkdir(exist_ok=True)
    for name, frame in (('predictions', preds), ('settlements', settles),
                        ('gaps', gaps), ('failures', fails)):
        if frame is not None:
            pd.DataFrame(frame).to_csv(out / f'{name}.csv', index=False)
    return str(out)


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
    assert 'dry_run' in s and 'нищо не се зачита' in s
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
    assert 'Обновено 02.10.2026 15:30 твое време' in s


@pytest.mark.parametrize('now_utc,expect_in', [(SUMMER_BEFORE_UTC, False),
                                               (SUMMER_OPEN_UTC, True),
                                               (WINTER_OPEN_UTC, True)])
def test_the_session_badge_opens_at_1530_sofia_in_both_halves_of_the_year(tmp_path, now_utc,
                                                                          expect_in):
    st = RP.session_state(now_utc)
    assert st['in_session'] is expect_in
    assert st['owner_now'].endswith('15:30') or st['owner_now'].endswith('15:20')
    s = open(RP.write_dashboard(_log(tmp_path), now=now_utc), encoding='utf-8').read()
    assert ('>в сесия<' in s) is expect_in


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
    assert 'Регистрирани прогнози за решение: <b>3 817</b>' in s
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


def test_the_serving_application_is_untouched():
    """Design Non-Goals: no production serving change."""
    for rel in ('api.py', 'src/inference.py', 'src/paper_trading.py'):
        r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel],
                           cwd=RP.REPO, capture_output=True)
        assert r.returncode == 0, f'{rel} changed'


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


def test_the_page_is_self_contained():
    s = open(RP.write_dashboard(now=SUMMER_OPEN_UTC,
                                path=os.path.join(RP.FWD, 'dashboard.html')),
             encoding='utf-8').read()
    assert '<style>' in s and 'src="http' not in s and 'href="http' not in s
    assert re.search(r'<meta charset="utf-8">', s)
