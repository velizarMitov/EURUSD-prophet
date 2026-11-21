"""Large-move alerts (openspec change gold-forecasting-v2, spec gold-forecasting/move-alerts)."""
import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

from src.gold import alerts as A
from src.gold import forward_logger as FL
from src.gold import load_config

CFG = load_config()
BANNED = ('buy', 'sell', 'entry', 'stop', 'target', 'купи', 'продай', 'вход', 'стоп', 'цел')


def _pred(bar, vol, cell='h1_lstm'):
    return {'key': f'{cell}|H1|{bar}', 'cell': cell, 'cadence': 'H1', 'as_of_bar': bar, 'vol_pct': vol}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['forward'] = str(tmp_path)
    return c


ALERTS = {'percentile': 90, 'threshold_vol_pct': 0.2}


def test_calibration_stores_the_percentile_of_the_models_own_forecasts(tmp_path, monkeypatch):
    from src.gold import prereg as P
    from src.gold import study as ST
    path = tmp_path / 'gold_config_v2.json'
    json.dump({'study': {}, 'alerts': {'percentile': 90, 'threshold_vol_pct': None}}, open(path, 'w'))
    monkeypatch.setattr(A, 'config_v2_path', lambda: str(path))
    idx = pd.date_range('2024-01-01', periods=500, freq='h', tz='UTC')
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, 500)))
    h1 = pd.DataFrame({'open': c, 'high': c, 'low': c, 'close': c}, index=idx)
    vol = np.random.default_rng(1).uniform(0.05, 0.3, 500)

    class Fake:
        def predict(self, inp):
            return {'vol_pct': vol}

    monkeypatch.setattr(P, 'registration', lambda cfg: {})
    monkeypatch.setattr(ST, 'load_pinned', lambda cfg, reg: {'H1': h1})
    monkeypatch.setattr(FL, 'read_manifest', lambda cfg: {})
    monkeypatch.setattr(FL, 'verify_cell', lambda cell, man: 'digest123')
    monkeypatch.setattr(FL, 'load_cell', lambda cell, cfg: Fake())
    out = A.calibrate(CFG)
    assert out['threshold_vol_pct'] == pytest.approx(np.percentile(vol, 90))
    assert out['percentile'] == 90 and out['model_digest'] == 'digest123'
    assert json.load(open(path))['alerts']['threshold_vol_pct'] == pytest.approx(out['threshold_vol_pct'])


def test_check_fires_at_or_above_the_threshold_once_per_bar():
    preds = [_pred('2026-10-22 14:00:00+00:00', 0.25), _pred('2026-10-22 15:00:00+00:00', 0.19),
             _pred('2026-10-22 14:00:00+00:00', 0.30, cell='h1_gbm')]
    rows = A.check(preds, set(), ALERTS, 'now')
    assert [r['as_of_bar'] for r in rows] == ['2026-10-22 14:00:00+00:00']
    assert A.check(preds, {rows[0]['key']}, ALERTS, 'now') == []
    assert A.check(preds, set(), {'threshold_vol_pct': None}, 'now') == []


def test_a_second_pass_on_the_same_bar_makes_no_second_alert(cfg):
    calls = []
    note = lambda v: (calls.append(v) or (True, ''))
    p = [_pred('2026-10-22 14:00:00+00:00', 0.25)]
    FL.log_alerts(cfg, 'now', p, ALERTS, note)
    FL.log_alerts(cfg, 'later', p, ALERTS, note)
    assert len(FL.read_rows(FL.out_path(cfg, 'alerts'))) == 1 and calls == [0.25]


def test_a_failed_notification_still_writes_the_row_and_is_reported(cfg):
    status = FL.log_alerts(cfg, 'now', [_pred('2026-10-22 14:00:00+00:00', 0.25)], ALERTS,
                           lambda v: (False, 'toast API missing'))
    [row] = FL.read_rows(FL.out_path(cfg, 'alerts'))
    assert row['notified'] == 'failed' and 'notification failed: toast API missing' in status


def test_notify_survives_a_broken_powershell():
    def boom(*a, **k):
        raise FileNotFoundError('powershell.exe')
    ok, err = A.notify(0.3, runner=boom)
    assert ok is False and 'powershell' in err


def test_push_phone_is_skipped_silently_when_the_topic_env_is_unset(monkeypatch):
    monkeypatch.delenv(A.NTFY_TOPIC_ENV, raising=False)
    calls = []
    ok, err = A.push_phone(0.3, http_post=lambda r: calls.append(r) or _Resp(200))
    assert ok is True and err == 'skipped' and calls == []


def test_push_phone_posts_bilingual_body_with_an_ascii_title(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 'secret-topic')
    seen = {}

    def poster(req):
        seen['url'] = req.full_url
        seen['body'] = req.data.decode('utf-8')
        seen['headers'] = {k.title(): v for k, v in req.headers.items()}
        return _Resp(200)

    ok, err = A.push_phone(0.35, http_post=poster)
    assert ok is True and err == ''
    assert seen['url'] == 'https://ntfy.sh/secret-topic'
    assert '0.35' in seen['body']
    assert 'Неизвестна' in seen['body'] or 'неизвестна' in seen['body']
    assert 'unknown' in seen['body'].lower()
    # ntfy header values must be ASCII; Cyrillic belongs in the body only.
    seen['headers']['Title'].encode('ascii')
    assert seen['headers']['Priority'] == 'urgent'
    for w in BANNED:
        assert w not in seen['body'].lower() and w not in seen['headers']['Title'].lower()


def test_push_phone_honours_a_self_hosted_server(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    monkeypatch.setenv(A.NTFY_SERVER_ENV, 'https://ntfy.example.com/')
    seen = {}
    A.push_phone(0.2, http_post=lambda r: seen.setdefault('url', r.full_url) or _Resp(200))
    assert seen['url'] == 'https://ntfy.example.com/t'


def test_push_phone_reports_http_error(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    ok, err = A.push_phone(0.2, http_post=lambda r: _Resp(503))
    assert ok is False and '503' in err


def test_push_phone_survives_a_network_error(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    def boom(req):
        raise ConnectionError('no route')
    ok, err = A.push_phone(0.2, http_post=boom)
    assert ok is False and 'ConnectionError' in err and 'no route' in err


def test_log_alerts_carries_a_push_failure_into_the_status(cfg):
    status = FL.log_alerts(cfg, 'now', [_pred('2026-10-22 14:00:00+00:00', 0.25)], ALERTS,
                           notifier=lambda v: (True, ''),
                           pusher=lambda v, d=None, p=None: (False, 'HTTP 500'))
    [row] = FL.read_rows(FL.out_path(cfg, 'alerts'))
    assert row['notified'] == 'yes'  # toast succeeded -> the column stays bound to the toast
    assert 'phone push failed: HTTP 500' in status


def test_log_alerts_passes_the_same_bar_direction_to_the_push(cfg):
    pred = _pred('2026-10-22 14:00:00+00:00', 0.25)
    pred['p_up'] = 0.72
    seen = {}
    FL.log_alerts(cfg, 'now', [pred], ALERTS,
                  notifier=lambda v: (True, ''),
                  pusher=lambda v, d=None, p=None: (seen.update({'v': v, 'd': d, 'p': p}) or (True, '')))
    assert seen == {'v': 0.25, 'd': 'up', 'p': pytest.approx(72.0)}


def test_log_alerts_turns_sub_half_p_up_into_a_down_direction(cfg):
    pred = _pred('2026-10-22 14:00:00+00:00', 0.3)
    pred['p_up'] = 0.4
    seen = {}
    FL.log_alerts(cfg, 'now', [pred], ALERTS,
                  notifier=lambda v: (True, ''),
                  pusher=lambda v, d=None, p=None: (seen.update({'d': d, 'p': p}) or (True, '')))
    assert seen == {'d': 'down', 'p': pytest.approx(60.0)}


def test_log_alerts_omits_direction_when_p_up_is_missing(cfg):
    pred = _pred('2026-10-22 14:00:00+00:00', 0.3)  # no p_up key at all
    seen = {}
    FL.log_alerts(cfg, 'now', [pred], ALERTS,
                  notifier=lambda v: (True, ''),
                  pusher=lambda v, d=None, p=None: (seen.update({'d': d, 'p': p}) or (True, '')))
    assert seen == {'d': None, 'p': None}


def test_push_body_adds_the_research_direction_line_when_known(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    seen = {}
    A.push_phone(0.35, direction='up', prob_pct=72.0,
                 http_post=lambda r: seen.setdefault('body', r.data.decode('utf-8')) or _Resp(200))
    body_lower = seen['body'].lower()
    # Canonical size-only sentence stays (both languages, with "direction unknown").
    assert 'неизвестна' in seen['body'] and 'unknown' in body_lower
    # Research line present, both languages, marked as unvalidated, with prob and direction.
    assert 'research' in body_lower and 'edge' in body_lower
    assert 'нагоре' in seen['body'] and ' up ' in f' {body_lower} '
    assert '72' in seen['body']
    # Still never names a trade.
    for w in BANNED:
        assert w not in body_lower


def test_push_body_stays_size_only_when_direction_is_absent(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    seen = {}
    A.push_phone(0.35, http_post=lambda r: seen.setdefault('body', r.data.decode('utf-8')) or _Resp(200))
    assert 'research' not in seen['body'].lower()
    assert 'нагоре' not in seen['body'] and 'надолу' not in seen['body']


def test_canonical_alert_text_is_still_size_only():
    """The research line is a push-only addition; the canonical text (used by the
    toast, the CSV row and the open-alert banner) must still carry 'size only,
    direction unknown', so every size-only invariant stays intact."""
    t = A.texts(0.35)
    full = ' '.join(t[l]['title'] + ' ' + t[l]['body'] for l in ('bg', 'en')).lower()
    assert 'research' not in full and 'edge' not in full
    assert 'нагоре' not in full and 'надолу' not in full


class _Resp:
    def __init__(self, status):
        self.status = status
    def getcode(self):
        return self.status


def test_push_forecasts_fires_on_every_new_h1_lstm_d1_lstm_d1_gbm_bar():
    preds = [
        {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm', 'as_of_bar': '2026-10-22 14:00:00+00:00',
         'p_up': 0.62, 'vol_pct': 0.11},
        {'key': 'd1_lstm|D1|2026-10-22 00:00:00+00:00', 'cell': 'd1_lstm', 'as_of_bar': '2026-10-22 00:00:00+00:00',
         'p_up': 0.47, 'vol_pct': 0.9},
        {'key': 'd1_gbm|D1|2026-10-22 00:00:00+00:00', 'cell': 'd1_gbm', 'as_of_bar': '2026-10-22 00:00:00+00:00',
         'p_up': 0.55, 'vol_pct': 1.1},
        {'key': 'd1_garch|D1|2026-10-22 00:00:00+00:00', 'cell': 'd1_garch', 'as_of_bar': '2026-10-22 00:00:00+00:00',
         'vol_pct': 0.8},                      # GARCH is a baseline -- no arrow in MT5, no push
    ]
    calls = []
    log, notes = A.push_forecasts(preds, set(), lambda *a: (calls.append(a) or (True, '')), 'now')
    assert [r['key'] for r in log] == [preds[0]['key'], preds[1]['key'], preds[2]['key']]
    assert [c[4] for c in calls] == ['h1_lstm', 'd1_lstm', 'd1_gbm']   # flavor='forecast', cell= arg
    assert all(c[3] == 'forecast' for c in calls) and notes == []
    # Direction derivation: p_up >= 0.5 -> up, else down; prob is the confident-side %.
    assert calls[0][1] == 'up' and calls[0][2] == pytest.approx(62.0)
    assert calls[1][1] == 'down' and calls[1][2] == pytest.approx(53.0)
    assert calls[2][1] == 'up' and calls[2][2] == pytest.approx(55.0)


def test_push_forecasts_dedups_against_the_log_so_a_bar_never_double_sends():
    pred = {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm',
            'as_of_bar': '2026-10-22 14:00:00+00:00', 'p_up': 0.6, 'vol_pct': 0.1}
    calls = []
    _, _ = A.push_forecasts([pred], {pred['key']}, lambda *a: (calls.append(a) or (True, '')), 'now')
    assert calls == []


def test_push_forecasts_still_sends_when_p_up_is_missing():
    pred = {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm',
            'as_of_bar': '2026-10-22 14:00:00+00:00', 'vol_pct': 0.1}   # no p_up
    calls = []
    log, _ = A.push_forecasts([pred], set(), lambda *a: (calls.append(a) or (True, '')), 'now')
    assert [c[1] for c in calls] == [None] and [c[2] for c in calls] == [None]
    assert [r['key'] for r in log] == [pred['key']]


def test_push_forecasts_carries_a_failure_into_notes_and_still_logs_the_key():
    pred = {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm',
            'as_of_bar': '2026-10-22 14:00:00+00:00', 'p_up': 0.6, 'vol_pct': 0.1}
    log, notes = A.push_forecasts([pred], set(), lambda *a: (False, 'HTTP 503'), 'now')
    assert len(log) == 1 and 'HTTP 503' in notes[0] and 'h1_lstm' in notes[0]


def test_forecast_push_body_is_research_tagged_and_never_a_trade(monkeypatch):
    monkeypatch.setenv(A.NTFY_TOPIC_ENV, 't')
    seen = {}
    A.push_phone(0.12, direction='up', prob_pct=62.0, flavor='forecast', cell='h1_lstm',
                 http_post=lambda r: seen.setdefault('req', r) or _Resp(200))
    body = seen['req'].data.decode('utf-8')
    body_lower = body.lower()
    assert 'h1 lstm' in body_lower and 'research' in body_lower and 'edge' in body_lower
    assert 'нагоре' in body and ('62' in body)
    # Expected-size line is present but without the large-move "unknown" sentence.
    assert '0.12' in body and 'неизвестна' not in body and 'direction unknown' not in body_lower
    # No alert wording at the top.
    assert 'очаква се голямо движение' not in body and 'large move' not in body_lower
    # ntfy header: ASCII, no priority=urgent (quiet-friendly cadence).
    headers = {k.title(): v for k, v in seen['req'].headers.items()}
    headers['Title'].encode('ascii')
    assert headers['Priority'] == 'default'
    for w in BANNED:
        assert w not in body_lower


def test_logger_hooks_forecast_pushes_after_alerts(cfg):
    from src.gold import forward_logger as FL2
    preds = [
        {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm', 'as_of_bar': '2026-10-22 14:00:00+00:00',
         'p_up': 0.6, 'vol_pct': 0.05},                                           # below threshold => no alert
        {'key': 'd1_lstm|D1|2026-10-22 00:00:00+00:00', 'cell': 'd1_lstm', 'as_of_bar': '2026-10-22 00:00:00+00:00',
         'p_up': 0.45, 'vol_pct': 0.9},
    ]
    calls = []
    status = FL2.log_forecast_pushes(cfg, 'now', preds,
                                     pusher=lambda *a: (calls.append(a) or (True, '')))
    assert [c[4] for c in calls] == ['h1_lstm', 'd1_lstm']   # both pushed, neither crossed the alert threshold
    assert '2 pushed' in status
    # A second pass on the same bars pushes nothing.
    status2 = FL2.log_forecast_pushes(cfg, 'later', preds, pusher=lambda *a: (True, ''))
    assert '0 pushed' in status2


def test_logger_forecast_push_failures_land_in_the_health_status(cfg):
    from src.gold import forward_logger as FL2
    pred = {'key': 'h1_lstm|H1|2026-10-22 14:00:00+00:00', 'cell': 'h1_lstm',
            'as_of_bar': '2026-10-22 14:00:00+00:00', 'p_up': 0.6, 'vol_pct': 0.05}
    status = FL2.log_forecast_pushes(cfg, 'now', [pred], pusher=lambda *a: (False, 'HTTP 500'))
    assert 'forecast push failed' in status and 'HTTP 500' in status


def test_alert_text_names_size_and_never_a_trade():
    t = A.texts(0.35)
    for lang in ('bg', 'en'):
        text = (t[lang]['title'] + ' ' + t[lang]['body']).lower()
        assert '0.35' in text
        for w in BANNED:
            assert w not in text, (lang, w)
    assert 'неизвестна' in t['bg']['body'] and 'unknown' in t['en']['body'].lower()


def test_alerts_settle_with_the_realised_move(cfg):
    p = [_pred('2026-10-22 14:00:00+00:00', 0.25)]
    FL.log_alerts(cfg, 'now', p, ALERTS, lambda v: (True, ''))
    FL.append_rows(FL.out_path(cfg, 'settlements'), [{'key': p[0]['key'], 'realised_vol_pct': 0.42}], FL.SETTLE_FIELDS)
    FL.log_alerts(cfg, 'later', [], ALERTS, lambda v: (True, ''))
    [s] = FL.read_rows(FL.out_path(cfg, 'alert_settlements'))
    assert float(s['realised_vol_pct']) == 0.42


def test_open_alert_is_shown_only_while_its_hour_is_open(cfg):
    FL.log_alerts(cfg, 'now', [_pred('2026-10-22 14:00:00+00:00', 0.25)], ALERTS, lambda v: (True, ''))
    assert A.open_alert(cfg, pd.Timestamp('2026-10-22 15:30', tz='UTC'))['vol_pct'] == 0.25
    assert A.open_alert(cfg, pd.Timestamp('2026-10-22 16:00', tz='UTC')) is None


def test_page_banner_is_bilingual():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'gold.html'),
               encoding='utf-8').read()
    assert 'id="bigmove"' in src
    for key in ('bigmove', 'bigmove_dir', 'bigmove_until', 'bigmove_hour', 'bigmove_sofia', 'bigmove_tab'):
        assert f'  {key}: {{bg: ' in src


def test_page_rereads_alerts_while_open():
    # An alert written at :03 must reach a page that was opened earlier (missed on 2026-10-27).
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'gold.html'),
               encoding='utf-8').read()
    assert 'setInterval(() => { load(); loadHealth(); }, 60e3)' in src
    assert "addEventListener('visibilitychange'" in src


def test_notification_stays_until_dismissed():
    assert 'scenario="reminder"' in A._TOAST_PS and 'arguments="dismiss"' in A._TOAST_PS


def _page():
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'gold.html'),
                encoding='utf-8').read()


def test_banner_texts_name_size_and_never_a_trade():
    keys = re.findall(r"^  (bigmove\w*): \{bg: '([^']*)', en: '([^']*)'\}", _page(), re.M)
    assert {k for k, _, _ in keys} >= {'bigmove', 'bigmove_dir', 'bigmove_hour', 'bigmove_sofia', 'bigmove_tab'}
    for key, bg, en in keys:
        for w in BANNED:
            assert w not in bg.lower() and w not in en.lower(), (key, w)


def test_tab_title_carries_the_alert_and_is_restored():
    src = _page()
    body = src[src.index('function renderAlert()'):]
    body = body[:body.index('\n}\n')]
    assert "if (!a) { el.hidden = true; document.title = baseTitle; return; }" in body
    assert "document.title = `⚠ ${t('bigmove_tab')} ±${num(a.vol_pct, 2)} % · ${baseTitle}`;" in body
    assert 'const baseTitle = document.title;' in src
