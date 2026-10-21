"""Read-only gold view payload (task 10.1; spec gold-forecasting/gold-view)."""
import copy
import csv
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.gold import forward_logger as FL
from src.gold import load_config
from src.gold import summary as SM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['forward'] = str(tmp_path / 'forward')
    c['paths']['test_report'] = str(tmp_path / 'test_report.json')
    c['paths']['registries'] = {k: str(tmp_path / f'{k}.csv') for k in CFG['paths']['registries']}
    os.makedirs(c['paths']['forward'])
    return c


def _write_preds(cfg, rows):
    path = os.path.join(cfg['paths']['forward'], 'predictions.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FL.PRED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FL.PRED_FIELDS})


def _row(cell, cad, as_of, **kw):
    return {'key': f'{cell}|{cad}|{as_of}', 'cell': cell, 'cadence': cad, 'as_of_bar': as_of,
            'as_of_close': 4131.0, 'phase': 'dry_run', **kw}


NOW = pd.Timestamp('2026-10-21 10:30', tz='UTC')       # label clock


def test_newest_row_is_taken_whole(cfg):
    _write_preds(cfg, [_row('h1_gbm', 'H1', '2026-10-21 08:00:00+00:00', p_up=0.62, ret_pct=0.01),
                       _row('h1_gbm', 'H1', '2026-10-21 09:00:00+00:00', p_up='', ret_pct=0.02)])
    out = SM.build(cfg, now=NOW)
    cur = next(c for c in out['cells'] if c['cell'] == 'h1_gbm')['current']
    assert cur['p_up'] is None and cur['direction'] is None
    assert '0.62' not in json.dumps(out)


def test_finished_window_is_not_current(cfg):
    _write_preds(cfg, [_row('h1_gbm', 'H1', '2026-10-21 06:00:00+00:00', p_up=0.55, ret_pct=0.01)])
    cell = next(c for c in SM.build(cfg, now=NOW)['cells'] if c['cell'] == 'h1_gbm')
    assert cell['current'] is None and cell['last_window_end'] == '2026-10-21 08:00:00+00:00'


def test_complete_forecast_shows_direction_return_price_and_move(cfg):
    _write_preds(cfg, [_row('d1_lstm', 'D1', '2026-10-21 00:00:00+00:00', p_up=0.53, ret_pct=0.04, vol_pct=0.81)])
    cur = next(c for c in SM.build(cfg, now=NOW)['cells'] if c['cell'] == 'd1_lstm')['current']
    assert cur['direction'] == 'up' and cur['probability'] == pytest.approx(0.53)
    assert cur['implied_price'] == 4132.65 and cur['vol_pct'] == 0.81


def test_nothing_logged_yet(cfg):
    out = SM.build(cfg, now=NOW)
    assert out['state'] == 'no_forecast' and all(c['current'] is None for c in out['cells'])


def test_labels_follow_the_forward_verdict_only(cfg):
    with open(cfg['paths']['registries']['direction'], 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['cell', 'verdict'])
        w.writeheader()
        w.writerow({'cell': 'h1_gbm', 'verdict': 'CLEARED (beats the training majority and a coin)'})
    out = SM.build(cfg, now=NOW)
    cell = next(c for c in out['cells'] if c['cell'] == 'h1_gbm')
    assert cell['label'] == SM.NO_EDGE and cell['verdicts']['direction'].startswith('CLEARED')
    with open(os.path.join(cfg['paths']['forward'], 'forward_verdicts.json'), 'w') as fh:
        json.dump({'verdicts': [{'cell': 'h1_gbm', 'verdict': 'KEEP — statistical skill, gross of cost; '
                                 'not a trading result'}]}, fh)
    cell = next(c for c in SM.build(cfg, now=NOW)['cells'] if c['cell'] == 'h1_gbm')
    assert cell['label'] == SM.KEEP_LABEL
    assert 'edge' not in cell['label'] and 'profit' not in cell['label']


def test_build_writes_nothing(cfg, tmp_path):
    _write_preds(cfg, [_row('h1_gbm', 'H1', '2026-10-21 09:00:00+00:00', p_up=0.55)])
    before = {p: os.path.getmtime(p) for p in map(str, tmp_path.rglob('*'))}
    SM.build(cfg, now=NOW)
    assert {p: os.path.getmtime(p) for p in map(str, tmp_path.rglob('*'))} == before


def test_summary_imports_no_model_library():
    code = ('import sys; from src.gold import summary; summary.build(); '
            'bad = [m for m in ("tensorflow", "keras", "xgboost", "arch") if m in sys.modules]; '
            'print(bad); sys.exit(1 if bad else 0)')
    r = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
