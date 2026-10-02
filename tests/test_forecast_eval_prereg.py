"""Pre-registration arithmetic, scoring gate and verdicts (tasks 12.1, 12.2, 12.4, 12.5)."""
import os
import subprocess

import numpy as np
import pytest

from src.forecast_eval import prereg as P
from src.forecast_eval import record as R


# ── 12.1 fixed-point family reproduces the design table ───────────────────

H1_MODELS = ('h1_gbm', 'kronos_direction')
M15_MODELS = ('m15_session_gbm', 'm15_session_lstm')


def test_fixed_point_reproduces_design_table():
    """Design D13: 46 candidates -> 10 -> 12 cells, alpha 0.05/12, n = 3,817."""
    cells = P.direction_cells(R.new_record('t'))
    assert len(cells) == 46
    admitted, alpha, table, history = P.fixed_point_family(cells)
    assert [h[0] for h in history] == [46, 10, 12], 'the iteration is not monotone'
    assert alpha == pytest.approx(0.05 / 12) and round(alpha, 6) == 0.004167
    assert round(P.n_required(alpha, 0.03)) == 3817
    assert sorted(admitted) == sorted(
        [(m, 'H1', h) for m in H1_MODELS for h in (1, 2, 4)]
        + [(m, 'M15', h) for m in M15_MODELS for h in (1, 2, 4)])
    yrs = {(r['model'], r['horizon']): round(r['years'], 2) for r in table}
    assert [yrs[('h1_gbm', h)] for h in (1, 2, 4)] == [0.62, 1.24, 2.47]
    assert [yrs[('m15_session_gbm', h)] for h in (1, 2, 4)] == [0.51, 1.05, 2.10]
    assert all(r['status'] == P.UNDERPOWERED for r in table if r['cadence'] == 'D1')


def test_the_longer_m15_cells_are_underpowered_by_arithmetic():
    cells = P.direction_cells(R.new_record('t'))
    admitted, alpha, table, _ = P.fixed_point_family(cells)
    longer = {r['horizon']: round(r['years'], 1) for r in table
              if r['model'] == 'm15_session_gbm' and r['horizon'] in (8, 16, 26)}
    assert longer == {8: 4.9, 16: 14.7, 26: 14.7}
    assert all(('m15_session_gbm', 'M15', h) not in admitted for h in (8, 16, 26))


def test_m15_rate_is_measured_per_horizon_not_bars_over_h():
    """The target-inside-the-session cutoff costs more at longer horizons, so
    rate/h would overstate every cell. Optimism in a pre-registration is a bug."""
    assert P.trades_per_year('M15', 1) == 7523.0
    for h in (2, 4, 8):
        assert P.trades_per_year('M15', h) < P.trades_per_year('M15', 1) / h
    assert P.trades_per_year('H1', 2) == pytest.approx(P.BARS_PER_YEAR['H1'] / 2)
    with pytest.raises(KeyError, match='measured M15 trade rate'):
        P.trades_per_year('M15', 3)


def test_adding_the_m15_models_costs_the_h1_cells_little():
    """Design D13: the 1-hour H1 cell moves 0.5 -> 0.62 years. Stated in advance
    so the cost of the 'all models' decision is visible, not discovered."""
    rec = R.new_record('t')
    h1_only = [c for c in P.direction_cells(rec) if c[1] != 'M15']
    _, alpha_before, t_before, _ = P.fixed_point_family(h1_only)
    _, alpha_after, t_after, _ = P.fixed_point_family(P.direction_cells(rec))
    before = next(r for r in t_before if r['model'] == 'h1_gbm' and r['horizon'] == 1)
    after = next(r for r in t_after if r['model'] == 'h1_gbm' and r['horizon'] == 1)
    assert round(before['years'], 1) == 0.5 and round(after['years'], 2) == 0.62
    assert alpha_before == pytest.approx(0.05 / 6) and alpha_after == pytest.approx(0.05 / 12)


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


def test_better_than_a_coin_is_keep_even_below_the_spread_breakeven():
    """Owner's criterion: direction vs a coin. The spread only labels it."""
    rng = np.random.default_rng(0)
    x = (rng.random(6000) < 0.56).astype(float)
    v = P.direction_verdict(x, n_required_=3364, breakeven=0.60, alpha=0.00833, h=1, cadence='H1')
    assert v['verdict'] == 'KEEP' and v['cost_label'] == 'predictive, not cost-viable'


def test_not_better_than_a_coin_is_drop_at_the_registered_n():
    rng = np.random.default_rng(1)
    x = (rng.random(6000) < 0.50).astype(float)
    v = P.direction_verdict(x, 3364, 0.5332, 0.00833, 1, 'H1')
    assert v['verdict'] == 'DROP' and v['label'] == 'not shown better than a coin'


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
