"""forecast_eval benchmarks and statistical tests (tasks 4.1-4.4)."""
import numpy as np
import pytest

from src.forecast_eval import benchmarks as B
from src.forecast_eval import stats as S


# ── 4.1 benchmarks ───────────────────────────────────────────────────────

def test_coin_and_majority():
    assert B.COIN_FLIP_ACCURACY == 0.5
    assert B.train_majority_class([1, 1, 0, 1]) == 1
    assert B.majority_accuracy([0, 0, 1], [0, 1, 0, 0]) == 0.75


def test_zero_return_forecast():
    assert np.all(B.zero_return_forecast(5) == 0)


def test_random_sign_turnover_matches_the_model_exactly():
    rng = np.random.default_rng(1)
    n = 1000
    flips = rng.random(n - 1) < 0.30
    model = np.concatenate([[1.0], np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)])
    k = B.n_changes(model)
    sims = B.random_sign_strategies(n, k, n_sims=50, seed=3)
    assert all(B.n_changes(s) == k for s in sims)
    assert all(set(np.unique(s)) <= {-1.0, 1.0} for s in sims)
    assert B.turnover(sims[0]) == pytest.approx(B.turnover(model))


def test_random_sign_benchmark_reports_percentile():
    rng = np.random.default_rng(2)
    moves = rng.normal(0, 1, 500)
    perfect = np.sign(moves)
    r = B.random_sign_benchmark(perfect, moves, cost_price=0.0, n_sims=200)
    assert r['percentile'] == 1.0 and r['model_changes'] == B.n_changes(perfect)


# ── 4.2 PT test and block bootstrap ──────────────────────────────────────

def test_pt_perfect_skill_is_significant():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.5).astype(float)
    r = S.pesaran_timmermann(y, y)
    assert r['accuracy'] == 1.0 and r['p_value'] < 1e-10


def test_pt_no_skill_is_not_significant():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.5).astype(float)
    x = (rng.random(2000) < 0.5).astype(float)
    assert S.pesaran_timmermann(y, x)['p_value'] > 0.05


def test_pt_refuses_constant_prediction():
    assert np.isnan(S.pesaran_timmermann([1, 0, 1, 0], [1, 1, 1, 1])['p_value'])


def test_block_length_is_horizon_aware():
    assert S.block_length(12, 'H1') == 24 and S.block_length(48, 'H1') == 48
    assert S.block_length(1, 'D1') == 5 and S.block_length(12, 'D1') == 12


def test_bootstrap_refuses_when_sample_not_longer_than_a_block():
    lo, hi, pt = S.block_bootstrap_ci(np.ones(24), block_len=24, alpha=0.05, n_boot=50)
    assert np.isnan(lo) and np.isnan(hi) and pt == 1.0
    lo, hi, _ = S.block_bootstrap_delta(np.ones(10), np.zeros(10), 12, 0.05, 50)
    assert np.isnan(lo) and np.isnan(hi)


def test_block_bootstrap_parity_with_the_existing_helper():
    from src.h1_direction_model import _block_bootstrap_delta
    rng = np.random.default_rng(7)
    cc = (rng.random(800) < 0.53).astype(float)
    cr = (rng.random(800) < 0.50).astype(float)
    ours = S.block_bootstrap_delta(cc, cr, 24, 0.05, n_boot=300, seed=42)
    theirs = _block_bootstrap_delta(cc, cr, 24, 300, 0.05, seed=42)
    assert ours == theirs


def test_one_sided_bounds_bracket_the_mean():
    x = np.random.default_rng(3).normal(1.0, 1.0, 500)
    lo = S.one_sided_lower_bound(x, 10, 0.05, n_boot=300)
    hi = S.one_sided_upper_bound(x, 10, 0.05, n_boot=300)
    assert lo < x.mean() < hi


# ── 4.3 Clark-West and Diebold-Mariano ───────────────────────────────────

def test_zero_forecast_against_itself_is_not_significant():
    y = np.random.default_rng(0).normal(0, 1, 500)
    r = S.clark_west(y, np.zeros(500))
    assert np.isnan(r['p_value']) or r['p_value'] > 0.05


def test_clark_west_detects_real_skill():
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1, 1000)
    y = 0.5 * signal + rng.normal(0, 1, 1000)
    assert S.clark_west(y, 0.5 * signal)['p_value'] < 0.01


def test_nested_comparison_routes_to_clark_west():
    y = np.random.default_rng(0).normal(0, 1, 100)
    assert S.compare_return_forecasts(y, y * 0.1, None, nested=True)['test'] == 'clark_west'
    with pytest.raises(ValueError, match='clark_west'):
        S.diebold_mariano_hln(y, y, nested=True)
    r = S.compare_return_forecasts(y, y * 0.1, y * 0.2, nested=False)
    assert r['test'] == 'diebold_mariano_hln' and 0 <= r['p_value'] <= 1


# ── 4.4 FWER ─────────────────────────────────────────────────────────────

def test_bonferroni():
    assert list(S.bonferroni([0.01, 0.2, 0.5])) == pytest.approx([0.03, 0.6, 1.0])


def test_romano_wolf_not_above_bonferroni_on_correlated_cells():
    rng = np.random.default_rng(4)
    common = rng.normal(0, 1, (600, 1))
    X = 0.9 * common + 0.3 * rng.normal(0, 1, (600, 5)) + np.array([0.12, 0.10, 0.08, 0.02, 0.0])
    raw, rw = S.romano_wolf_mean_test(X, block_len=5, n_boot=500)
    assert np.all(rw <= S.bonferroni(raw) + 1e-12)
    assert np.all(np.diff(rw[np.argsort(-X.mean(axis=0))]) >= -1e-12)   # monotone step-down


def test_romano_wolf_controls_familywise_error_under_the_global_null():
    rng = np.random.default_rng(11)
    rejections = 0
    sims = 150
    for _ in range(sims):
        common = rng.normal(0, 1, (300, 1))
        X = 0.6 * common + 0.8 * rng.normal(0, 1, (300, 4))
        _, rw = S.romano_wolf_mean_test(X, block_len=5, n_boot=200, seed=int(rng.integers(1e9)))
        rejections += bool((rw < 0.05).any())
    assert rejections / sims <= 0.10
