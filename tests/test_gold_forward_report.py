"""Gold forward registration and report (tasks 9.1-9.2; spec "Forward verdicts
measure skill, not profit", "Forward pre-registration precedes forward scoring")."""
import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.gold import forward_logger as FL
from src.gold import forward_report as FR
from src.gold import load_config

CFG = load_config()
COST_WORDS = ('profit', 'pips', 'p&l', 'pnl', 'sharpe', 'ledger', 'spread')


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['forward'] = str(tmp_path)
    cfg['paths']['forward_prereg_md'] = str(tmp_path / 'PRE_REGISTRATION.md')
    reg = FR.build_registration({'D1': 0.98, 'H1': 0.97}, {'D1': 255.0, 'H1': 5450.0},
                                {'h1_gbm': 1, 'h1_lstm': 1, 'd1_gbm': 1, 'd1_lstm': 1},
                                {'h1_gbm': 'abc'}, cfg)
    (tmp_path / 'PRE_REGISTRATION.json').write_text(json.dumps(reg))
    (tmp_path / 'PRE_REGISTRATION.md').write_text(FR.render_registration(reg), encoding='utf-8')
    monkeypatch.setattr(FR, 'scoring_allowed', lambda cfg: True)
    return cfg, tmp_path, reg


def _write(tmp, n, phase_of):
    rng = np.random.default_rng(2)
    t = pd.date_range('2027-01-04 01:00', periods=n, freq='h', tz='UTC')
    preds, sets = [], []
    for i, ts in enumerate(t):
        ret = rng.normal(0, 0.1)
        for cell in ('h1_gbm', 'h1_lstm', 'h1_garch_hod'):
            key = f'{cell}|H1|{ts}'
            preds.append({'key': key, 'cell': cell, 'cadence': 'H1', 'as_of_bar': str(ts), 'as_of_close': 4000,
                          'p_up': '' if cell == 'h1_garch_hod' else (0.6 if ret > 0 else 0.4),
                          'ret_pct': '' if cell == 'h1_garch_hod' else ret / 2,
                          'vol_pct': 0.08, 'phase': phase_of(i), 'logged_at': ''})
            sets.append({'key': key, 'realised_dir': int(ret > 0), 'realised_ret_pct': ret,
                         'realised_vol_pct': abs(ret), 'eligible': True})
    pd.DataFrame(preds, columns=FL.PRED_FIELDS).to_csv(tmp / 'predictions.csv', index=False)
    pd.DataFrame(sets).to_csv(tmp / 'settlements.csv', index=False)


def test_registration_admits_h1_and_records_gross_basis(env):
    _, _, reg = env
    d = reg['families']['direction']
    assert d['admitted'] == ['h1_gbm', 'h1_lstm']
    assert 'GROSS of cost' in reg['cost_basis'] and reg['min_window_months'] == 6


def test_dry_run_rows_are_never_scored(env):
    cfg, tmp, _ = env
    _write(tmp, 500, lambda i: 'dry_run')
    res = FR.score(cfg)
    assert res['n_scored_rows'] == 0
    assert all(r['n'] == 0 for r in res['verdicts'])


def test_interim_before_the_registered_n_and_window(env):
    cfg, tmp, _ = env
    _write(tmp, 500, lambda i: 'registered')
    res = FR.score(cfg)
    h1 = next(r for r in res['verdicts'] if r['cell'] == 'h1_gbm' and r['family'] == 'direction')
    assert h1['n'] == 500 and h1['verdict'] == FR.INTERIM and h1['accuracy'] == 1.0


def test_report_never_contains_profit_or_cost_figures(env):
    cfg, tmp, _ = env
    _write(tmp, 300, lambda i: 'registered' if i % 2 else 'dry_run')
    text = (FR.render_report(FR.score(cfg)) + json.dumps(FR.score(cfg))).lower()
    for w in COST_WORDS:
        assert w not in text, w


def test_scoring_refuses_without_a_committed_registration(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['forward'] = str(tmp_path)
    cfg['paths']['forward_prereg_md'] = str(tmp_path / 'PRE_REGISTRATION.md')
    with pytest.raises(PermissionError):
        FR.score(cfg)
