"""Pre-registration arithmetic, scoring gate and verdicts (tasks 12.1, 12.2, 12.4, 12.5)."""
import os
import subprocess

import numpy as np
import pytest

from src.forecast_eval import prereg as P
from src.forecast_eval import record as R


# ── 12.1 fixed-point family reproduces the design table ───────────────────

def test_fixed_point_reproduces_design_table():
    cells = P.direction_cells(R.new_record('t'))
    assert len(cells) == 34
    admitted, alpha, table, history = P.fixed_point_family(cells)
    assert [h[0] for h in history] == [34, 6]
    assert alpha == pytest.approx(0.05 / 6) and round(alpha, 5) == 0.00833
    assert sorted(admitted) == sorted((m, 'H1', h) for m in ('h1_gbm', 'kronos_direction') for h in (1, 2, 4))
    yrs = {(r['model'], r['horizon']): r['years'] for r in table}
    assert round(yrs[('h1_gbm', 1)], 1) == 0.5
    assert round(yrs[('h1_gbm', 2)], 1) == 1.1
    assert round(yrs[('h1_gbm', 4)], 1) == 2.2
    assert all(r['status'] == P.UNDERPOWERED for r in table if r['cadence'] == 'D1')


def test_lower_coverage_lengthens_and_can_shrink_the_family():
    cells = P.direction_cells(R.new_record('t'))
    full = P.fixed_point_family(cells)[0]
    half = P.fixed_point_family(cells, coverage=0.5)[0]
    assert set(half) <= set(full) and len(half) < len(full)


# ── 12.2 volatility power ─────────────────────────────────────────────────

def test_volatility_power_from_development_ci():
    from scipy.stats import norm
    r = P.volatility_power(n_scored=1000, ci_low=-0.02, ci_high=0.0, mae_baseline=0.20,
                           cadence='D1', h=1, alpha=0.0125)
    se = 0.02 / (2 * norm.ppf(0.975))
    sd = se * np.sqrt(1000)
    expect = ((norm.ppf(1 - 0.0125 / 2) + norm.ppf(0.8)) * sd / 0.01) ** 2
    assert r['n_required'] == pytest.approx(expect)
    assert r['years'] == pytest.approx(expect / 260)


# ── 12.4 scoring gate ─────────────────────────────────────────────────────

def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo)] + list(args), check=True, capture_output=True)


def test_scoring_gate(tmp_path):
    repo = tmp_path / 'r'
    (repo / 'results' / 'forward_eval').mkdir(parents=True)
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')
    _git(repo, 'commit', '-q', '--allow-empty', '-m', 'root')
    md = repo / 'results' / 'forward_eval' / 'PRE_REGISTRATION.md'
    js = repo / 'results' / 'forward_eval' / 'registration.json'
    gate = lambda: P.scoring_allowed(str(repo), str(md), str(js))
    assert not gate()                                   # nothing registered
    md.write_text('# protocol\n', encoding='utf-8')
    js.write_text('{"server": "ActivTradesEU-Server"}', encoding='utf-8')
    assert not gate()                                   # written but not committed
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'register')
    assert gate()                                       # committed: scoring allowed
    assert P.registered_server(str(js)) == 'ActivTradesEU-Server'
    md.write_text('# protocol, edited after the fact\n', encoding='utf-8')
    assert not gate()                                   # edited after registration: refused


def test_logger_refuses_to_score_without_registration(monkeypatch, tmp_path):
    from src.forecast_eval import forward_logger as FL
    monkeypatch.setattr(P, 'scoring_allowed', lambda *a, **k: False)
    seen = {}
    monkeypatch.setattr(FL, 'predict_cadence', lambda snap, c, phase, *a, **k: seen.setdefault('phase', phase) or {})
    monkeypatch.setattr(FL, 'settle', lambda *a, **k: 0)
    snap = FL.Snapshot(None, None, 's', {})
    FL.run_once(snapshot=snap, out=str(tmp_path),
                manifest_path='absent.csv', record_path='absent.json')
    assert seen['phase'] == 'dry_run'
    monkeypatch.setattr(P, 'scoring_allowed', lambda *a, **k: True)
    monkeypatch.setattr(P, 'registered_server', lambda *a, **k: 'S')
    seen.clear()
    FL.run_once(snapshot=snap, out=str(tmp_path),
                manifest_path='absent.csv', record_path='absent.json')
    assert seen['phase'] == 'scoring'


# ── 12.5 verdicts ─────────────────────────────────────────────────────────

def test_no_verdict_before_the_registered_n():
    x = np.ones(500)                                    # looks decisive
    v = P.direction_verdict(x, n_required_=3364, breakeven=0.5332, alpha=0.00833, h=1, cadence='H1')
    assert v['verdict'] is None and v['label'] == P.INTERIM


def test_predictive_but_not_cost_viable_is_dropped_with_its_label():
    rng = np.random.default_rng(0)
    x = (rng.random(6000) < 0.53).astype(float)         # above 50 %, below a 56 % breakeven
    v = P.direction_verdict(x, n_required_=3364, breakeven=0.56, alpha=0.00833, h=1, cadence='H1')
    assert v['verdict'] == 'DROP' and v['label'] == 'predictive, not cost-viable'


def test_keep_requires_the_lower_bound_above_breakeven():
    rng = np.random.default_rng(1)
    x = (rng.random(6000) < 0.60).astype(float)
    assert P.direction_verdict(x, 3364, 0.5332, 0.00833, 1, 'H1')['verdict'] == 'KEEP'


def test_underpowered_cells_never_adjudicate():
    v = P.direction_verdict(np.ones(10 ** 5), 3364, 0.5, 0.00833, 24, 'D1', admitted=False)
    assert v['verdict'] == P.UNDERPOWERED


def test_volatility_verdicts():
    rng = np.random.default_rng(2)
    base = np.abs(rng.normal(0, 1, 2000))
    better = base - 0.2
    assert P.volatility_verdict(better, base, 1000, 0.0125, 1, 'D1')['verdict'] == 'KEEP'
    assert P.volatility_verdict(base + 0.2, base, 1000, 0.0125, 1, 'D1')['verdict'] == 'DROP'
    assert P.volatility_verdict(better[:100], base[:100], 1000, 0.0125, 1, 'D1')['label'] == P.INTERIM
