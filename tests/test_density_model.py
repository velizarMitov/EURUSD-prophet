"""Unit tests for the pre-registered density family (H_den.1 / H_den.2).

Protocol: results/density/PRE_REGISTRATION.md, committed at
7524cab0237d325a144b7eef1f85ca759643b942 before src/density_model.py existed.

These cover the four things the registered verdict actually rests on:
  1. the CRPS closed forms (mixture, Student-t, empirical) against independent
     references — the metric IS the decision rule, so a wrong CRPS is a wrong
     verdict;
  2. the sigma floor guard, including that it aborts loudly on divergence;
  3. PIT uniformity on draws from a KNOWN density (diagnostic correctness);
  4. that no selection step indexes test rows.

Nothing here imports keras except the tests that genuinely need a model, so the
metric layer stays cheap to test.
"""
import math

import numpy as np
import pytest

from src.density_model import (
    ALPHA, CI_LEVEL, FAMILY_SIZE, K_GRID, SEEDS, SIGMA_FLOOR, BLOCK_LEN, N_BOOT,
    crps_empirical, crps_gaussian_mixture, crps_student_t, ensemble_mixture,
    paired_block_bootstrap, pit_empirical, pit_gaussian_mixture, pit_student_t,
    rank_histogram, select_k, split_masks,
)
from src.ltc_data import crps_gaussian_np


# ---------------------------------------------------------------------------
# 1. CRPS closed forms
# ---------------------------------------------------------------------------

def test_mixture_crps_matches_project_gaussian_reference_at_k1():
    """The pre-registration commits to this cross-check by name: a
    one-component mixture must equal src/ltc_data.py::crps_gaussian_np."""
    rng = np.random.default_rng(0)
    n = 200
    mu = rng.normal(0, 1, n)
    sigma = rng.uniform(0.1, 3.0, n)
    y = rng.normal(0, 2, n)

    got = crps_gaussian_mixture(np.ones((n, 1)), mu[:, None], sigma[:, None], y)
    ref = crps_gaussian_np(mu, sigma, y)
    assert np.max(np.abs(got - ref)) < 1e-12


def test_gaussian_crps_known_closed_form_value():
    """Standard normal at y = mu: CRPS = sigma (2 phi(0) - 1/sqrt(pi))."""
    expected = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    got = crps_gaussian_mixture(np.ones((1, 1)), np.zeros((1, 1)),
                                np.ones((1, 1)), np.zeros(1))
    assert got[0] == pytest.approx(expected, abs=1e-12)
    # and it scales linearly in sigma
    got3 = crps_gaussian_mixture(np.ones((1, 1)), np.zeros((1, 1)),
                                 np.full((1, 1), 3.0), np.zeros(1))
    assert got3[0] == pytest.approx(3.0 * expected, abs=1e-12)


def _crps_by_integration(cdf, y, lo=-200.0, hi=200.0, n=200001):
    """CRPS = int (F(x) - 1{x >= y})^2 dx, on a fine grid. Independent of every
    closed form under test.

    The integrand is DISCONTINUOUS at x = y (it jumps from F(y)^2 to
    (F(y)-1)^2), so the two sides are integrated separately with y as an exact
    endpoint. Straddling the jump with a single grid costs O(dx * jump) — about
    9e-5 here, which is large enough to fail a 2e-5 tolerance and would look
    like a bug in the closed form rather than in this reference.
    """
    left = np.linspace(lo, y, n)
    right = np.linspace(y, hi, n)
    return (np.trapezoid(cdf(left) ** 2, left)
            + np.trapezoid((cdf(right) - 1.0) ** 2, right))


@pytest.mark.parametrize('nu', [2.5, 4.0, 8.0, 30.0])
@pytest.mark.parametrize('y', [-1.7, 0.0, 0.9])
def test_student_t_crps_matches_numerical_integration(nu, y):
    """The pre-registration says the t-CRPS is a closed form and states it will
    be verified against integration. This is that verification."""
    from scipy.stats import t as st
    ref = _crps_by_integration(lambda x: st.cdf(x, nu), y)
    got = crps_student_t(np.array([y]), np.array([1.0]), nu)[0]
    assert got == pytest.approx(ref, rel=2e-5, abs=2e-5)


def test_student_t_crps_scales_with_sigma():
    y, nu, s = 0.8, 5.0, 2.5
    a = crps_student_t(np.array([y]), np.array([s]), nu)[0]
    b = s * crps_student_t(np.array([y / s]), np.array([1.0]), nu)[0]
    assert a == pytest.approx(b, rel=1e-12)


def test_student_t_crps_rejects_nu_at_or_below_one():
    with pytest.raises(ValueError):
        crps_student_t(np.array([0.0]), np.array([1.0]), 1.0)


def test_mixture_crps_matches_integration_for_a_real_mixture():
    """A genuinely multi-component, skewed mixture — the case the closed form
    exists for and the K=1 check cannot exercise."""
    from scipy.stats import norm
    w = np.array([[0.5, 0.3, 0.2]])
    mu = np.array([[-1.0, 0.4, 2.0]])
    sg = np.array([[0.7, 1.3, 0.4]])
    y = 0.55

    def cdf(x):
        return sum(w[0, i] * norm.cdf(x, mu[0, i], sg[0, i]) for i in range(3))

    ref = _crps_by_integration(cdf, y)
    got = crps_gaussian_mixture(w, mu, sg, np.array([y]))[0]
    assert got == pytest.approx(ref, rel=2e-5, abs=2e-5)


def test_empirical_crps_matches_brute_force_definition():
    """CRPS = E|X - y| - 0.5 E|X - X'|, computed the slow obvious way."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.2, 1.4, 300)
    for y in (-2.0, 0.0, 0.37, 3.1):
        brute = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
        got = crps_empirical(x, np.array([y]))[0]
        assert got == pytest.approx(brute, rel=1e-12, abs=1e-12)


def test_empirical_crps_rejects_empty_sample():
    with pytest.raises(ValueError):
        crps_empirical(np.array([]), np.array([0.0]))


def test_crps_is_minimised_by_the_true_distribution():
    """Propriety spot-check: the generating density must beat a misspecified
    one on average. If this fails, the metric cannot arbitrate anything."""
    rng = np.random.default_rng(11)
    y = rng.normal(0.0, 1.0, 4000)
    one = np.ones((y.size, 1))
    truth = crps_gaussian_mixture(one, np.zeros((y.size, 1)), one, y).mean()
    too_wide = crps_gaussian_mixture(one, np.zeros((y.size, 1)), one * 2.0, y).mean()
    too_narrow = crps_gaussian_mixture(one, np.zeros((y.size, 1)), one * 0.5, y).mean()
    biased = crps_gaussian_mixture(one, np.full((y.size, 1), 1.0), one, y).mean()
    assert truth < too_wide and truth < too_narrow and truth < biased


# ---------------------------------------------------------------------------
# 2. PIT — uniform on draws from a KNOWN density
# ---------------------------------------------------------------------------

def test_pit_is_uniform_for_draws_from_the_predictive_gaussian_mixture():
    """Sample from a known 2-component mixture, then PIT under that same
    mixture. By the probability integral transform the result must be U(0,1)."""
    rng = np.random.default_rng(3)
    n = 40000
    w = np.array([0.65, 0.35])
    mu = np.array([-0.5, 1.2])
    sg = np.array([0.8, 0.5])
    comp = rng.choice(2, size=n, p=w)
    y = rng.normal(mu[comp], sg[comp])

    pit = pit_gaussian_mixture(np.tile(w, (n, 1)), np.tile(mu, (n, 1)),
                               np.tile(sg, (n, 1)), y)
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    assert pit.mean() == pytest.approx(0.5, abs=0.01)
    assert pit.var() == pytest.approx(1 / 12, abs=0.005)

    _, _, freq = rank_histogram(pit, bins=20)
    assert np.max(np.abs(freq - 0.05)) < 0.006      # flat to within MC noise


def test_pit_is_uniform_for_draws_from_the_predictive_student_t():
    from scipy.stats import t as st
    rng = np.random.default_rng(4)
    nu, sigma, n = 5.0, 1.7, 40000
    y = st.rvs(nu, size=n, random_state=rng) * sigma
    pit = pit_student_t(y, np.full(n, sigma), nu)
    assert pit.mean() == pytest.approx(0.5, abs=0.01)
    assert pit.var() == pytest.approx(1 / 12, abs=0.005)


def test_pit_detects_a_miscalibrated_forecast():
    """Non-vacuity: a too-narrow predictive density must produce a visibly
    U-shaped (over-dispersed-observation) PIT, not a flat one."""
    rng = np.random.default_rng(6)
    n = 20000
    y = rng.normal(0, 1, n)
    one = np.ones((n, 1))
    pit_bad = pit_gaussian_mixture(one, np.zeros((n, 1)), one * 0.4, y)
    _, _, freq = rank_histogram(pit_bad, bins=20)
    assert freq[0] + freq[-1] > 0.30                # mass piles into the tails
    pit_ok = pit_gaussian_mixture(one, np.zeros((n, 1)), one, y)
    _, _, freq_ok = rank_histogram(pit_ok, bins=20)
    assert freq_ok[0] + freq_ok[-1] < 0.15


def test_pit_empirical_is_uniform_on_its_own_sample_draws():
    rng = np.random.default_rng(8)
    sample = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 20000)
    pit = pit_empirical(sample, y)
    assert pit.mean() == pytest.approx(0.5, abs=0.02)


def test_rank_histogram_bins_and_normalisation():
    counts, edges, freq = rank_histogram(np.linspace(0, 1, 1000), bins=20)
    assert counts.sum() == 1000
    assert len(edges) == 21
    assert freq.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. The sigma floor guard
# ---------------------------------------------------------------------------

def test_sigma_parameterisation_is_structurally_above_the_floor():
    """sigma = FLOOR + softplus(raw) can never reach the floor for finite raw,
    including at raw = -1e3 where softplus underflows to exactly 0."""
    from src.density_model import SIGMA_FLOOR as floor
    raw = np.array([-1e3, -50.0, -1.0, 0.0, 1.0, 50.0, 1e3])
    sigma = floor + np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)
    assert np.all(sigma >= floor)
    assert np.all(np.isfinite(sigma))
    assert sigma[0] == pytest.approx(floor)          # the binding edge


def test_sigma_guard_aborts_loudly_on_a_diverged_model():
    """The guard exists because a silent NaN would look like a result. Feed it a
    model whose parameters have gone non-finite and require it to raise."""
    pytest.importorskip('keras')
    from src.density_model import SigmaFloorViolation, _make_sigma_guard, build_mdn

    k = 2
    model = build_mdn(4, k, seed=0)
    x = np.zeros((8, 4), dtype=float)
    guard = _make_sigma_guard(model, x, k)

    guard.on_epoch_end(0)                            # healthy model: silent

    weights = model.get_weights()
    weights[-1] = np.full_like(weights[-1], np.nan)  # poison the sigma head bias
    model.set_weights(weights)
    with pytest.raises(SigmaFloorViolation):
        guard.on_epoch_end(1)


def test_sigma_guard_aborts_on_a_non_finite_loss():
    pytest.importorskip('keras')
    from src.density_model import SigmaFloorViolation, _make_sigma_guard, build_mdn
    model = build_mdn(4, 2, seed=0)
    guard = _make_sigma_guard(model, np.zeros((8, 4)), 2)
    with pytest.raises(SigmaFloorViolation):
        guard.on_epoch_end(0, logs={'loss': float('nan')})


# ---------------------------------------------------------------------------
# 4. No selection step may index test rows
# ---------------------------------------------------------------------------

class _TestBlockTripwire:
    """Stands in for the test mask. Any read at all raises."""

    def __getattr__(self, name):
        raise AssertionError(f"a selection step touched the test block (.{name})")

    def __getitem__(self, key):
        raise AssertionError(f"a selection step touched the test block ([{key!r}])")

    def __len__(self):
        raise AssertionError("a selection step touched the test block (len)")

    def __iter__(self):
        raise AssertionError("a selection step touched the test block (iter)")


def test_split_masks_are_disjoint_and_chronological():
    d = {'n': 1000, 'train_end': 700, 'val_end': 800}
    tr, va, te = split_masks(d)
    assert tr.sum() == 700 and va.sum() == 100 and te.sum() == 200
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    # chronological, no interleaving
    assert np.flatnonzero(tr).max() < np.flatnonzero(va).min()
    assert np.flatnonzero(va).max() < np.flatnonzero(te).min()


def test_select_k_cannot_reach_the_test_block():
    """Structural assertion of the pre-registered promise: `select_k` takes only
    train and validation masks, so the test rows are not reachable from it. The
    tripwire in `d` fires if anything indexes them."""
    pytest.importorskip('keras')
    rng = np.random.default_rng(1)
    n, p = 400, 5
    d = {'X': rng.normal(size=(n, p)), 'y': rng.normal(0, 0.5, n),
         'n': n, 'train_end': 280, 'val_end': 320,
         'test_rows': _TestBlockTripwire()}
    tr, va, te = split_masks(d)
    best, scores = select_k(d, tr, va, seeds=(42, 43), k_grid=(2,))
    assert best in (2,)
    assert set(scores) == {2}
    # and the signature genuinely has no test argument
    import inspect
    assert 'te' not in inspect.signature(select_k).parameters
    assert 'test' not in inspect.signature(select_k).parameters


def test_selection_helpers_take_no_test_argument():
    """A weaker but broader guard: nothing in the selection path accepts a test
    mask, so the promise cannot be broken by a future edit without also changing
    a signature this test pins."""
    import inspect
    from src.density_model import fit_mdn_ensemble, train_mdn
    for fn in (select_k, train_mdn):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {'te', 'test', 'test_mask'}), fn.__name__
    # fit_mdn_ensemble takes eval_masks (it must, to score the one-shot report),
    # but its FITTING arguments are train/validation only.
    sig = inspect.signature(fit_mdn_ensemble).parameters
    assert 'tr' in sig and 'va' in sig and 'eval_masks' in sig


# ---------------------------------------------------------------------------
# 5. Ensemble, bootstrap, and the pre-registered constants
# ---------------------------------------------------------------------------

def test_ensemble_is_an_equal_weight_mixture_over_seeds():
    """The validated object: 5 seeds x K components, weights summing to 1."""
    n, k, s = 7, 3, 5
    rng = np.random.default_rng(2)
    per_seed = []
    for _ in range(s):
        w = rng.random((n, k))
        w /= w.sum(axis=1, keepdims=True)
        per_seed.append((w, rng.normal(size=(n, k)), rng.uniform(0.2, 1.0, (n, k))))
    w, mu, sg = ensemble_mixture(per_seed)
    assert w.shape == (n, s * k) and mu.shape == (n, s * k) and sg.shape == (n, s * k)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(w[:, :k], per_seed[0][0] / s, atol=1e-12)


def test_ensemble_rejects_an_empty_seed_set():
    with pytest.raises(ValueError):
        ensemble_mixture([])


def test_ensemble_crps_is_a_proper_mixture_not_an_average_of_scores():
    """CRPS is not linear in the distribution, so scoring the mixture must NOT
    equal averaging the per-seed CRPS. If these ever coincide, the ensemble has
    been assembled the wrong way round."""
    rng = np.random.default_rng(9)
    n, k = 50, 2
    y = rng.normal(0, 1, n)
    per_seed = []
    for shift in (-0.8, 0.9, 0.1):
        w = np.full((n, k), 0.5)
        mu = np.column_stack([np.full(n, shift), np.full(n, -shift)])
        sg = np.full((n, k), 0.9)
        per_seed.append((w, mu, sg))
    mixture = crps_gaussian_mixture(*ensemble_mixture(per_seed), y).mean()
    avg_of_scores = np.mean([crps_gaussian_mixture(*p, y).mean() for p in per_seed])
    assert mixture != pytest.approx(avg_of_scores, rel=1e-6)
    assert mixture < avg_of_scores          # mixing is at least as good here


def test_paired_bootstrap_recovers_a_known_positive_gap():
    rng = np.random.default_rng(12)
    n = 900
    challenger = rng.normal(1.00, 0.10, n)
    rival = challenger + 0.05                      # challenger better by 0.05
    out = paired_block_bootstrap(rival, challenger)
    assert out['point_delta'] == pytest.approx(0.05, abs=1e-9)
    assert out['ci_low'] > 0 and out['cleared'] is True
    assert out['block_len'] == BLOCK_LEN and out['n_boot'] == N_BOOT
    assert out['ci_level'] == pytest.approx(CI_LEVEL)


def test_paired_bootstrap_does_not_clear_on_pure_noise():
    rng = np.random.default_rng(13)
    n = 900
    challenger = rng.normal(1.0, 0.2, n)
    rival = rng.normal(1.0, 0.2, n)
    out = paired_block_bootstrap(rival, challenger)
    assert out['ci_low'] < 0 < out['ci_high']
    assert out['cleared'] is False


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(14)
    a, b = rng.normal(1, .2, 400), rng.normal(1, .2, 400)
    assert paired_block_bootstrap(a, b) == paired_block_bootstrap(a, b)


def test_preregistered_constants_are_exactly_what_was_registered():
    """These numbers were fixed by commit 7524cab before any model existed.
    Changing one silently would invalidate the family's verdict, so they are
    pinned here rather than merely documented."""
    assert FAMILY_SIZE == 2
    assert ALPHA == pytest.approx(0.05 / FAMILY_SIZE) == pytest.approx(0.025)
    assert CI_LEVEL == pytest.approx(0.975)
    assert BLOCK_LEN == 5
    assert N_BOOT == 2000
    assert SIGMA_FLOOR == 0.05
    assert tuple(K_GRID) == (2, 3, 5)
    assert tuple(SEEDS) == (42, 43, 44, 45, 46)
    assert len(SEEDS) >= 5


def test_prereg_commit_hash_is_recorded_in_the_module():
    from src.density_model import PREREG_COMMIT
    assert PREREG_COMMIT == '7524cab0237d325a144b7eef1f85ca759643b942'
    assert len(PREREG_COMMIT) == 40
