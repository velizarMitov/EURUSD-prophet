"""Large-move alerts (openspec change gold-forecasting-v2, spec gold-forecasting/move-alerts)."""
import copy
import json
import os

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
    for key in ('bigmove', 'bigmove_dir', 'bigmove_until'):
        assert f'  {key}: {{bg: ' in src
