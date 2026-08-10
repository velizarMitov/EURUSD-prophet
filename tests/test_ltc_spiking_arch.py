"""Invariant tests for ``src/ltc_spiking_arch.py`` (Idea-3 LTC + spiking abstention).

Same contract as the curl family's tests: these pin the properties that make the
architecture mean what its docstring says. Several correspond to bugs or degenerate
states hit while building it -- the dead readout at init, and the rate/tau gauge freedom.

RESEARCH ONLY. Nothing here asserts predictive power; that requires a pre-registered test
on ``validation[70:80]`` scored against ``results/ltc_hypothesis_log.csv``, which does not
exist yet.
"""

from __future__ import annotations

import numpy as np
import pytest

# JAX is an OPTIONAL research dependency (requirements-ltc.txt). The core app must install
# and `python -m pytest -q` must stay green without it, exactly as for the Kronos track.
eqx = pytest.importorskip("equinox", reason="requires requirements-ltc.txt")
jax = pytest.importorskip("jax", reason="requires requirements-ltc.txt")
pytest.importorskip("diffrax", reason="requires requirements-ltc.txt")
pytest.importorskip("optax", reason="requires requirements-ltc.txt")
import jax.numpy as jnp  # noqa: E402

from src import ltc_spiking_arch as arch  # noqa: E402

T, IN, COV, HID = 64, 5, 3, 16


@pytest.fixture(scope="module")
def model() -> arch.LTCSpikingModel:
    return arch.LTCSpikingModel(IN, COV, hidden_size=HID, key=jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def batch():
    k = jax.random.PRNGKey(1)
    ku, kd, kc, ky = jax.random.split(k, 4)
    u = jax.random.normal(ku, (T, IN))
    dt = jnp.abs(jax.random.normal(kd, (T,))) * 0.3 + 1.0
    cov = jax.random.normal(kc, (T, COV))
    y = jax.random.normal(ky, (T,)) * 0.3
    return u, dt, cov, y


# ======================================================================================
# The liquid cell: continuous-time correctness
# ======================================================================================


def test_zero_dt_is_exactly_the_identity(model: arch.LTCSpikingModel) -> None:
    """Two bars at the same timestamp cannot move the state. The CfC paper's sigmoid gate
    fails this (it applies g(x,u) on a zero-length interval); the exact exponential does
    not. This is the property that makes irregular sampling safe."""
    x = jax.random.normal(jax.random.PRNGKey(7), (HID,))
    u = jax.random.normal(jax.random.PRNGKey(8), (IN,))
    x_next, _ = model.cell(x, u, jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(x_next), np.asarray(x), atol=1e-6)


def test_closed_form_matches_a_real_ode_solver(model: arch.LTCSpikingModel) -> None:
    """The "closed-form" claim, checked against diffrax rather than asserted. A sign error,
    a reciprocal, or a dropped dt would all survive a shape test and none survive this."""
    x = jax.random.normal(jax.random.PRNGKey(7), (HID,))
    u = jax.random.normal(jax.random.PRNGKey(8), (IN,))
    for dt in (0.1, 1.0, 5.0, 20.0):
        closed, _ = model.cell(x, u, jnp.asarray(dt))
        integrated = arch.ode_reference_step(model.cell, x, u, jnp.asarray(dt))
        np.testing.assert_allclose(
            np.asarray(closed), np.asarray(integrated), rtol=1e-4, atol=1e-5
        )


def test_dt_composes_when_the_dynamics_are_frozen(model: arch.LTCSpikingModel) -> None:
    """Stepping dt1 then dt2 must equal one step of dt1+dt2 under frozen (h, tau).

    If this failed, the model's answer would depend on how the calendar happens to be
    bucketed -- resampling the same history at a different grid would change the forecast.
    Composition is checked on the frozen dynamics because that is what the closed form
    solves; with (h, tau) re-derived each step the two differ by the genuine curvature of
    the ODE, which is a modelling choice and not an error.
    """
    x = jax.random.normal(jax.random.PRNGKey(7), (HID,))
    u = jax.random.normal(jax.random.PRNGKey(8), (IN,))
    z = jnp.concatenate([x, u])
    cell = model.cell
    tau = cell.tau_min + (cell.tau_max - cell.tau_min) * jax.nn.sigmoid(cell.tau_net(z))
    h = cell.target_net(z)

    def frozen(state, dt):
        return h + (state - h) * jnp.exp(-dt / tau)

    two_steps = frozen(frozen(x, jnp.asarray(1.5)), jnp.asarray(2.5))
    one_step = frozen(x, jnp.asarray(4.0))
    np.testing.assert_allclose(np.asarray(two_steps), np.asarray(one_step), atol=1e-6)


def test_large_dt_relaxes_to_the_target(model: arch.LTCSpikingModel) -> None:
    """dt -> infinity must forget the state entirely and land on h(x,u)."""
    x = jax.random.normal(jax.random.PRNGKey(7), (HID,))
    u = jax.random.normal(jax.random.PRNGKey(8), (IN,))
    far, _ = model.cell(x, u, jnp.asarray(1e4))
    h = model.cell.target_net(jnp.concatenate([x, u]))
    np.testing.assert_allclose(np.asarray(far), np.asarray(h), atol=1e-6)


def test_tau_is_input_dependent_and_bounded(model: arch.LTCSpikingModel) -> None:
    """THE "liquid" claim. A tau that does not move with the input is a plain gated RNN
    wearing the name, and the loss alone cannot tell you which one you trained."""
    x = jnp.zeros((HID,))
    k = jax.random.split(jax.random.PRNGKey(11), 16)
    taus = jnp.stack([model.cell.tau_eff(x, jax.random.normal(kk, (IN,)) * 3.0) for kk in k])
    assert float(taus.std()) > 1e-4, "tau_eff is constant in the input"
    assert float(taus.min()) >= arch.TAU_MIN
    assert float(taus.max()) <= arch.TAU_MAX


# ======================================================================================
# Business time
# ======================================================================================


def test_business_time_is_strictly_monotone_in_calendar_time(
    model: arch.LTCSpikingModel,
) -> None:
    """A warp that could invert would let the model reorder history -- look-ahead wearing
    a nonlinearity. Positivity of the rate is what forbids it."""
    cov = jax.random.normal(jax.random.PRNGKey(3), (COV,))
    dts = jnp.array([0.0, 0.1, 1.0, 3.0, 10.0])
    warped = jnp.array([model.warp(d, cov)[0] for d in dts])
    assert bool(jnp.all(jnp.diff(warped) > 0))
    assert float(warped[0]) == 0.0


def test_warp_starts_at_exactly_calendar_time() -> None:
    """rate == 1 is the gauge-neutral centre, and the identity init must hit it EXACTLY,
    for every covariate value and every seed -- not approximately for a lucky key.

    Before the final layer was zeroed this returned 0.557 on key=5: the untrained warp was
    already a nontrivial clock, which makes tau uninterpretable (see the gauge test) and
    spends the first phase of training undoing a random initial time-distortion.
    """
    for seed in range(5):
        warp = arch.BusinessTimeWarp(COV, key=jax.random.PRNGKey(seed))
        for cov_value in (0.0, 1.0, -3.0):
            _, rate = warp(jnp.asarray(1.0), jnp.full((COV,), cov_value))
            assert float(rate) == pytest.approx(1.0, abs=1e-6)


def test_rate_and_tau_share_a_gauge(model: arch.LTCSpikingModel) -> None:
    """DOCUMENTS a real unidentifiability rather than hiding it: only dt/tau enters the
    dynamics, so scaling both by k changes nothing. Any claim of the form "the model
    learned a 12-day memory" is therefore not a statement until the clock scale is pinned.
    """
    x = jax.random.normal(jax.random.PRNGKey(7), (HID,))
    u = jax.random.normal(jax.random.PRNGKey(8), (IN,))
    z = jnp.concatenate([x, u])
    cell = model.cell
    tau = cell.tau_min + (cell.tau_max - cell.tau_min) * jax.nn.sigmoid(cell.tau_net(z))
    h = cell.target_net(z)
    k = 3.7
    a = h + (x - h) * jnp.exp(-jnp.asarray(2.0) / tau)
    b = h + (x - h) * jnp.exp(-(k * jnp.asarray(2.0)) / (k * tau))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6)


# ======================================================================================
# The spiking readout
# ======================================================================================


def test_spike_forward_is_hard_and_backward_is_not() -> None:
    """The entire reason abstention is trainable. The true derivative of a step is a Dirac
    delta -- identically zero wherever the optimiser evaluates it -- so without the
    surrogate the readout receives EXACTLY zero gradient and can never learn."""
    v = jnp.array([-2.0, -0.05, 0.0, 0.05, 2.0])
    out = arch.spike(v, arch.SURROGATE_ALPHA)
    assert set(np.asarray(out).tolist()) <= {0.0, 1.0}
    grad = jax.grad(lambda z: arch.spike(z, arch.SURROGATE_ALPHA).sum())(v)
    assert float(jnp.abs(grad).min()) > 0.0, "surrogate gradient vanished"
    # peaked at threshold, decaying away from it
    g = np.asarray(grad)
    assert g[2] > g[1] and g[2] > g[3]
    assert g[1] > g[0] and g[3] > g[4]


def test_lif_steady_state_is_invariant_to_sampling_rate() -> None:
    """A constant input must drive the same membrane fixed point whether it arrives as one
    long interval or many short ones. With a fixed per-step leak it would not, and the
    abstention decision would silently depend on the calendar bucketing -- the pathology
    the liquid cell avoids upstream would sneak back in through the readout."""
    readout = arch.LIFReadout(HID, key=jax.random.PRNGKey(4), bias_init_scale=0.0)
    x = jnp.ones((HID,)) * 0.1

    def run(dt, steps):
        state = readout.initial_state()
        for _ in range(steps):
            state, _, v = readout(state, x, jnp.asarray(dt))
        return float(v[0])

    np.testing.assert_allclose(run(0.05, 400), run(2.0, 10), rtol=1e-3)


def test_lif_reset_is_subtractive_not_to_zero() -> None:
    """Zeroing discards the overshoot, which IS the "how strong was the evidence" signal."""
    readout = arch.LIFReadout(HID, key=jax.random.PRNGKey(4), bias_init_scale=5.0)
    state = readout.initial_state()
    next_state, s, v = readout(state, jnp.zeros((HID,)), jnp.asarray(10.0))
    assert float(s[0]) == 1.0
    assert float(v[0]) > readout.v_th
    np.testing.assert_allclose(
        float(next_state.v[0]), float(v[0]) - readout.v_th, rtol=1e-5
    )


def test_readout_is_alive_at_init_across_seeds() -> None:
    """The dead-readout failure: with the bias at default init the neuron never reaches
    threshold and fires on 0% of bars, leaving the run to crawl out on a surrogate gradient
    two orders of magnitude down. Pins the measured fix (see the table in LIFReadout)."""
    rates = []
    for seed in range(8):
        key = jax.random.PRNGKey(seed)
        m = arch.LTCSpikingModel(IN, COV, hidden_size=HID, key=key)
        ku, kd, kc = jax.random.split(key, 3)
        u = jax.random.normal(ku, (200, IN))
        dt = jnp.abs(jax.random.normal(kd, (200,))) * 0.3 + 1.0
        cov = jax.random.normal(kc, (200, COV))
        rates.append(float(m(u, dt, cov).spike.mean()))
    assert min(rates) > 0.05, f"a seed started with a near-dead readout: {rates}"


# ======================================================================================
# CRPS and the loss
# ======================================================================================


def test_crps_matches_the_known_closed_form_value() -> None:
    """CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = 0.2336949...; a wrong constant or a
    missing sigma factor would still produce plausible-looking training curves."""
    v = arch.crps_gaussian(jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))
    assert float(v) == pytest.approx(0.2336949, abs=1e-6)


def test_crps_matches_monte_carlo() -> None:
    """Independent check of the closed form against the definition
    E|X - y| - 0.5 E|X - X'| for X, X' ~ N(mu, sigma)."""
    mu, sigma, y = 0.4, 1.3, -0.2
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a = mu + sigma * jax.random.normal(k1, (400_000,))
    b = mu + sigma * jax.random.normal(k2, (400_000,))
    mc = float(jnp.abs(a - y).mean() - 0.5 * jnp.abs(a - b).mean())
    closed = float(arch.crps_gaussian(jnp.array(mu), jnp.array(sigma), jnp.array(y)))
    assert closed == pytest.approx(mc, abs=5e-3)


def test_crps_is_proper_in_sigma() -> None:
    """A strictly proper rule is minimised by the TRUE distribution. This is why CRPS and
    not MAE: the model cannot buy a better score by misstating its confidence."""
    key = jax.random.PRNGKey(2)
    y = jax.random.normal(key, (20_000,)) * 1.0
    mu = jnp.zeros_like(y)
    honest = float(arch.crps_gaussian(mu, jnp.ones_like(y), y).mean())
    overconfident = float(arch.crps_gaussian(mu, jnp.full_like(y, 0.4), y).mean())
    underconfident = float(arch.crps_gaussian(mu, jnp.full_like(y, 2.5), y).mean())
    assert honest < overconfident
    assert honest < underconfident


def test_the_floor_term_pushes_the_fire_rate_up_not_down(model, batch) -> None:
    """THE SIGN TRAP. A `lambda * spike_rate` penalty does not prevent the degenerate
    silent optimum -- it causes it. Only a floor on the rate opposes collapse. This test
    fails if the two terms are ever swapped."""
    u, dt, cov, y = batch
    out = model(u, dt, cov)
    lo, _ = arch.selective_crps_loss(out, y, lam=jnp.asarray(0.0), rho_min=0.9)
    hi, _ = arch.selective_crps_loss(out, y, lam=jnp.asarray(50.0), rho_min=0.9)
    assert float(hi) > float(lo), "the floor multiplier did not penalise under-firing"

    # and the optional sparsity knob must push the other way
    a, _ = arch.selective_crps_loss(out, y, lam=jnp.asarray(0.0), rho_min=0.0, lam_rate=0.0)
    b, _ = arch.selective_crps_loss(out, y, lam=jnp.asarray(0.0), rho_min=0.0, lam_rate=10.0)
    assert float(b) > float(a)


def test_silence_is_not_free(model, batch) -> None:
    """The degenerate optimum, stated as a test: a model that abstains on everything gets a
    vacuous accuracy term, so the floor must make total silence strictly worse than
    complying."""
    u, dt, cov, y = batch
    out = model(u, dt, cov)
    silent = out._replace(spike=jnp.zeros_like(out.spike))
    compliant = out._replace(
        spike=(jnp.arange(len(y)) < int(0.5 * len(y))).astype(jnp.float32)
    )
    s_loss, s_aux = arch.selective_crps_loss(silent, y, lam=jnp.asarray(20.0), rho_min=0.3)
    c_loss, _ = arch.selective_crps_loss(compliant, y, lam=jnp.asarray(20.0), rho_min=0.3)
    assert float(s_aux.fire_rate) == 0.0
    assert float(s_loss) > float(c_loss)


def test_accuracy_gradient_stays_bounded_at_zero_coverage(model, batch) -> None:
    """THE COLLAPSE BUG, pinned.

    Written first as ``(s*crps).sum() / (s.sum() + eps)``, this loss drove the fire rate to
    0.000 within 100 steps on a task where abstention was exactly the wrong answer, and no
    amount of dual ascent recovered it. The cause was the epsilon: at zero coverage the
    denominator is 1e-6, so the accuracy gradient is scaled by 1e6 and -- since CRPS is
    positive everywhere -- points at "fire less", against a restoring force of order
    ``lam/T``.

    Flooring the denominator at ``rho_min * T`` bounds it. This test compares the gradient
    at zero coverage against the gradient at healthy coverage; with the epsilon form the
    ratio is ~1e6, with the floor it is order 1.
    """
    u, dt, cov, y = batch
    out = model(u, dt, cov)

    def acc_only(spikes):
        o = out._replace(spike=spikes)
        return arch.selective_crps_loss(o, y, lam=jnp.asarray(0.0), rho_min=0.3)[0]

    # a hair above zero coverage, so the surrogate has something to differentiate
    near_zero = jnp.zeros_like(out.spike).at[0].set(1.0)
    healthy = (jnp.arange(T) < int(0.6 * T)).astype(jnp.float32)
    g_zero = float(jnp.abs(jax.grad(acc_only)(near_zero)).max())
    g_healthy = float(jnp.abs(jax.grad(acc_only)(healthy)).max())
    assert g_zero < 100.0 * g_healthy, (
        f"accuracy gradient exploded at low coverage ({g_zero:.3e} vs {g_healthy:.3e}) "
        "— the denominator is eps-guarded again and the model will collapse to silence"
    )


def test_coverage_floor_counts_the_whole_batch_not_just_its_first_axis(model, batch) -> None:
    """The floor must scale with EVERY element, because training applies this loss under
    vmap to a (B, T) batch, not to a single (T,) sequence.

    Written as ``rho_min * s.shape[0]`` the floor counted the BATCH dimension only: at
    B=64, T=64 the guard engaged at 6.4 spikes instead of 409.6, i.e. below 0.15% coverage.
    That silently disabled the anti-collapse guard in the exact configuration the training
    loop ships in -- the one regime where the eps-collapse this floor exists to prevent
    comes back.
    """
    u, dt, cov, y = batch
    out = model(u, dt, cov)
    b, t, rho = 8, 16, 0.5
    tile = lambda v: jnp.broadcast_to(v[:t], (b, t))  # noqa: E731
    batched = arch.StepOutput(
        mu=tile(out.mu), sigma=tile(out.sigma),
        spike=jnp.zeros((b, t)).at[0, :10].set(1.0),  # 10 spikes out of 128
        membrane=tile(out.membrane), tau_mean=tile(out.tau_mean),
        business_rate=tile(out.business_rate), dt_business=tile(out.dt_business),
    )
    yb = jnp.broadcast_to(y[:t], (b, t))
    _, aux = arch.selective_crps_loss(batched, yb, lam=jnp.asarray(0.0), rho_min=rho)

    crps = arch.crps_gaussian(batched.mu, batched.sigma, yb)
    numerator = float((batched.spike * crps).sum())
    floored = numerator / (rho * batched.spike.size)          # correct: /64
    by_first_axis = numerator / max(float(batched.spike.sum()), rho * b)  # buggy: /10
    assert float(aux.crps_on_fired) == pytest.approx(floored, rel=1e-5)
    assert float(aux.crps_on_fired) != pytest.approx(by_first_axis, rel=1e-3)


def test_dual_ascent_raises_lambda_only_while_violated() -> None:
    """lambda is a Lagrange multiplier, not a hyperparameter: it climbs while the floor is
    breached and stops when it is met. It must never go negative -- a negative multiplier
    flips the floor into the collapse-inducing penalty."""
    lam = jnp.asarray(0.0)
    for _ in range(5):
        lam = arch.dual_ascent(lam, jnp.asarray(0.1), step=0.5)
    assert float(lam) > 0.0
    held = arch.dual_ascent(lam, jnp.asarray(0.0), step=0.5)
    assert float(held) == pytest.approx(float(lam))
    assert float(arch.dual_ascent(jnp.asarray(0.0), jnp.asarray(-5.0), step=1.0)) >= 0.0


# ======================================================================================
# End to end
# ======================================================================================


def test_forward_shapes_and_finiteness(model, batch) -> None:
    u, dt, cov, y = batch
    out = model(u, dt, cov)
    for name, value in out._asdict().items():
        assert value.shape == (T,), name
        assert bool(jnp.isfinite(value).all()), name
    assert bool(jnp.all(out.sigma > 0))


def test_gradients_reach_every_component(model, batch) -> None:
    """Including the readout and the warp -- both sit behind the non-differentiable spike
    and would silently receive nothing if the surrogate were mis-wired."""
    u, dt, cov, y = batch

    def loss_fn(m):
        return arch.selective_crps_loss(
            m(u, dt, cov), y, lam=jnp.asarray(1.0), rho_min=0.3
        )[0]

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    for name, g in (
        ("readout", grads.readout.proj.weight),
        ("warp head", grads.warp.net.layers[-1].weight),
        ("tau_net", grads.cell.tau_net.layers[0].weight),
        ("target_net", grads.cell.target_net.layers[0].weight),
        ("head", grads.head.net.layers[0].weight),
    ):
        assert float(jnp.abs(g).sum()) > 0.0, f"no gradient reached {name}"


def test_identity_init_stalls_earlier_warp_layers_for_exactly_one_step(model, batch) -> None:
    """The cost of the identity init, pinned so it cannot silently become permanent.

    Zeroing the warp's final layer makes ``rate == 1`` exactly, but backprop to every
    EARLIER layer is proportional to that zeroed weight, so they get no gradient on step
    one. That is acceptable only because it self-heals: the final layer moves immediately
    and the rest come alive on the next step. If a future change ever zeroed more than the
    head -- or froze it -- this test turns the silent freeze into a failure.
    """
    import optax

    u, dt, cov, y = batch

    def loss_fn(m):
        return arch.selective_crps_loss(
            m(u, dt, cov), y, lam=jnp.asarray(1.0), rho_min=0.3
        )[0]

    def first_layer_grad(m):
        return float(jnp.abs(eqx.filter_grad(loss_fn)(m).warp.net.layers[0].weight).sum())

    grads = eqx.filter_grad(loss_fn)(model)
    assert first_layer_grad(model) == 0.0, "expected the documented one-step stall"
    assert float(jnp.abs(grads.warp.net.layers[-1].weight).sum()) > 0.0, "no escape path"

    opt = optax.adam(1e-2)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    updates, _ = opt.update(eqx.filter(grads, eqx.is_inexact_array), state)
    stepped = eqx.apply_updates(model, updates)
    assert first_layer_grad(stepped) > 0.0, "the stall did not clear after one step"


def test_no_lookahead_in_the_recurrence(model, batch) -> None:
    """The project's cardinal rule, applied to a scan: poisoning bar t must not change any
    output before t. A reversed scan or an accidental centred window dies here."""
    u, dt, cov, _ = batch
    base = model(u, dt, cov)
    cut = T // 2
    u2 = u.at[cut:].set(u[cut:] * 100.0)
    poisoned = model(u2, dt, cov)
    np.testing.assert_allclose(
        np.asarray(base.mu[:cut]), np.asarray(poisoned.mu[:cut]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(base.spike[:cut]), np.asarray(poisoned.spike[:cut]), atol=0.0
    )


def test_risk_coverage_curve_is_ordered(model, batch) -> None:
    """The honest reporting object for a selective predictor. A single (accuracy, coverage)
    pair is unfalsifiable -- anything looks good at low enough coverage."""
    u, dt, cov, y = batch
    coverage, risk = arch.risk_coverage_curve(model(u, dt, cov), y)
    assert bool(jnp.all(jnp.diff(coverage) > 0))
    assert float(coverage[-1]) == pytest.approx(1.0)
    assert float(risk[-1]) == pytest.approx(
        float(arch.crps_gaussian(model(u, dt, cov).mu, model(u, dt, cov).sigma, y).mean()),
        rel=1e-5,
    )
