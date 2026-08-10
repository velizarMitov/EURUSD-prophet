"""Idea 3 — liquid time-constant dynamics in learned business time, with a spiking
abstention readout. RESEARCH ONLY: raw architecture, no data, no training loop, no claim.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path. This is a new
hypothesis family; its log is ``results/ltc_hypothesis_log.csv`` and NOTHING has been
registered in it yet. No number produced by this module is a result until a pre-registered
test has been scored on ``validation[70:80]``.

--------------------------------------------------------------------------------------
WHAT THIS IS, IN ONE PARAGRAPH
--------------------------------------------------------------------------------------
Three pieces stack:

  1. ``BusinessTimeWarp``  calendar Delta-t  ->  *business* Delta-t. A weekend is three
     calendar days but roughly one day of information flow; an FOMC hour is the opposite.
     The warp is learned, strictly positive, and monotone in calendar time by construction.
  2. ``LiquidCell``        a closed-form continuous-time (CfC) recurrent cell whose
     relaxation rate ``tau_eff(x, u)`` is INPUT-DEPENDENT -- the "liquid" part -- and which
     consumes Delta-t explicitly, so irregular calendar sampling is native rather than a
     preprocessing hack.
  3. ``LIFReadout``        a leaky integrate-and-fire layer whose spike means "I commit to
     a forecast on this bar". Silence is ABSTENTION. Surrogate gradients make that
     decision trainable end-to-end instead of thresholded after the fact.

The loss is CRPS on the bars the model chose to speak on, plus a Lagrangian floor that
stops the model from buying a perfect score by never speaking at all.

--------------------------------------------------------------------------------------
WHY THE CELL IS AN EXACT EXPONENTIAL AND NOT THE PAPER'S SIGMOID GATE
--------------------------------------------------------------------------------------
Hasani et al.'s CfC writes the time gate as ``sigmoid(-f(x,u) * t)``. That is an
approximation to the LTC ODE's solution and it is fine for regularly sampled data, but it
fails two properties this project needs and can test:

  * **Delta-t = 0 must be the identity.** Two bars at the same timestamp cannot move the
    state. The sigmoid form gives ``x <- g(x,u)`` at t=0, i.e. it moves the state on a
    zero-length interval.
  * **Delta-t must compose.** Stepping ``dt1`` then ``dt2`` with the input held must equal
    one step of ``dt1 + dt2``. Otherwise the model's answer depends on how the calendar
    happens to be bucketed, which is exactly the pathology irregular sampling introduces.

``x(t+dt) = h + (x - h) * exp(-dt / tau_eff)`` is the EXACT solution of
``dx/dt = -(x - h)/tau_eff`` for piecewise-constant ``(h, tau_eff)``, so it satisfies both
identically, not approximately. It is still closed-form and still continuous-time -- the
"closed-form" in CfC means "no ODE solver in the loop", which this keeps. The optional
``ode_reference_step`` integrates the same ODE with diffrax so the closed form can be
checked against a real solver rather than asserted.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# --------------------------------------------------------------------------------------
# Numerical guards. Every one of these bounds a documented failure mode, not a taste.
# --------------------------------------------------------------------------------------

#: Relaxation-rate bounds, in units of BUSINESS time (see BusinessTimeWarp).
#: tau -> 0 collapses the cell to x <- h (memoryless, and d/ddt blows up); tau -> inf
#: freezes the state and kills the gradient. Both are attractors during early training,
#: so tau is squashed into a band rather than left free and clipped after the fact --
#: clipping would zero the gradient exactly where the optimiser needs to escape.
TAU_MIN = 0.05
TAU_MAX = 50.0

#: Business-clock rate bound, in LOG space and symmetric: rate lives in
#: ``[exp(-MAX_LOG_RATE), exp(+MAX_LOG_RATE)]``, i.e. the clock can run at most ~20x fast
#: or ~20x slow. Symmetric-in-log matters for two reasons -- speeding up and slowing down
#: are equally reachable, and ``rate == 1`` (plain calendar time) is the exact centre, so
#: an untrained warp starts as the identity instead of at some arbitrary band midpoint.
MAX_LOG_RATE = 3.0

#: Fast-sigmoid surrogate width (Zenke & Ganguli 2018). Larger = sharper = closer to the
#: true derivative of the step, but the gradient vanishes further from threshold.
SURROGATE_ALPHA = 10.0

#: Readout bias init, as a multiple of the spike threshold. Measured, not guessed -- see
#: the table in ``LIFReadout``. Below 1.25 some seeds start with a permanently silent
#: readout, which costs the run its budget before abstention learning even begins.
BIAS_INIT_SCALE = 1.25

#: Predictive sd floor. CRPS is minimised at sigma -> 0 only when mu is exactly right;
#: with a floor the model cannot chase a degenerate spike at a lucky point.
SIGMA_MIN = 1e-3


# ======================================================================================
# 1. Learned business time
# ======================================================================================


class BusinessTimeWarp(eqx.Module):
    """Map calendar ``dt`` to business ``dt`` conditioned on activity covariates.

    ``dt_business = dt_calendar * rate(covariates)``, with
    ``rate = exp(MAX_LOG_RATE * tanh(net(covariates)))``.

    Monotone in ``dt_calendar`` BY CONSTRUCTION because ``rate > 0``. That matters: a warp
    that could invert would let the model reorder history, which is a look-ahead channel
    wearing a nonlinearity. The rate deliberately does NOT see ``dt_calendar`` itself --
    only exogenous activity (tick counts, spreads, session dummies) -- so the warp cannot
    learn a non-monotone function of time by routing through its own argument.

    This is the "learned business time" of the idea title: the model discovers its own
    clock instead of being handed volume-time or tick-time as a fixed preprocessing choice.

    --------------------------------------------------------------------------------------
    THE GAUGE PROBLEM, AND WHY THE PARAMETERISATION IS LOG-SYMMETRIC
    --------------------------------------------------------------------------------------
    The dynamics depend on ``dt_business / tau_eff`` and on nothing else. Multiplying every
    ``rate`` by ``k`` and every ``tau_eff`` by ``k`` leaves the model's output EXACTLY
    unchanged, so ``rate`` and ``tau`` are jointly unidentifiable up to a common scale.
    Two consequences, both real:

      * A learned ``tau`` reported on its own is meaningless -- "the model discovered a
        12-day memory" is not a statement until the clock's scale is pinned. Any such claim
        must quote ``tau`` together with the mean ``rate``, or quote only the ratio.
      * The optimiser can drift along that flat direction indefinitely, burning steps
        without changing the loss.

    Centring the rate at 1 in log space fixes the gauge softly: ``rate == 1`` is calendar
    time, it is where the model starts, and the symmetric bound gives it no reason to walk
    off in either direction. The alternative -- normalising the rate to mean 1 across the
    sequence -- would fix the gauge *hard*, but it computes a statistic over the whole
    sequence including future bars, which is a look-ahead channel this project bans. If a
    hard gauge is ever wanted, the normaliser must be a constant fitted on ``[0:70%]``.
    """

    net: eqx.nn.MLP
    log_rate_offset: Float[Array, ""]
    max_log_rate: float = eqx.field(static=True)

    def __init__(
        self,
        covariate_size: int,
        *,
        width: int = 32,
        depth: int = 2,
        max_log_rate: float = MAX_LOG_RATE,
        identity_init: bool = True,
        key: PRNGKeyArray,
    ) -> None:
        net = eqx.nn.MLP(covariate_size, 1, width, depth, activation=jax.nn.tanh, key=key)
        if identity_init:
            # Zero the FINAL layer so net(.) == 0 and rate == exp(0) == 1 exactly: the
            # untrained warp is the identity, i.e. plain calendar time. Without this the
            # random output bias puts the initial rate at an arbitrary point in the band
            # (measured: 0.557 on key=5), so "the model started from calendar time" would
            # be a claim the code does not honour, and tau at init would be uninterpretable
            # for the gauge reason above.
            #
            # THE COST, stated because it is a real zero-init trap and not free: with the
            # last layer's weight at zero, every EARLIER layer of this MLP receives exactly
            # zero gradient on the first step -- backprop to them is proportional to that
            # weight. Measured at init: |dW| is 0.0 for layers 0 and 1, and 3.6e-2 for the
            # final layer. So the escape path exists (the last layer moves immediately) and
            # the stall lasts exactly one optimiser step: layer 0 is back to |dW| = 2.2e-3
            # after step 1. It is self-healing, not frozen -- but a reader debugging a flat
            # warp on step 1 should know this is expected. Pinned by
            # test_identity_init_stalls_earlier_warp_layers_for_exactly_one_step.
            net = eqx.tree_at(
                lambda m: [m.layers[-1].weight, m.layers[-1].bias],
                net,
                [jnp.zeros_like(net.layers[-1].weight), jnp.zeros_like(net.layers[-1].bias)],
            )
        self.net = net
        self.max_log_rate = max_log_rate
        self.log_rate_offset = jnp.asarray(0.0)

    def __call__(
        self, dt_calendar: Float[Array, ""], covariates: Float[Array, " c"]
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Returns ``(dt_business, rate)``. ``rate`` is returned for diagnostics -- a warp
        that has collapsed to a constant means the learned clock earned nothing over
        calendar time, and that is invisible in the loss.

        ``log_rate_offset`` PINS THE GAUGE. Only ``dt * rate / tau`` enters the dynamics, so
        ``(rate, tau) -> (c*rate, c*tau)`` is an exact symmetry and the optimiser drifts
        freely along it until it hits whichever bound exists -- measured in Stage 1, the
        rate pinned at ``exp(MAX_LOG_RATE) = 20.09`` after two epochs and normal-bar state
        retention fell 0.955 -> 0.462 as a result. Raising the bound cannot fix that; it
        just moves the wall and makes retention worse.

        Subtracting a batch-derived offset INSIDE the tanh removes the symmetry instead of
        bounding it: the median rate is held at 1, so only the SHAPE of the clock is
        learned. Centring before the squash (not after) also keeps tanh operating near 0
        where it has full dynamic range -- centring the output would let tanh saturate and
        destroy the shape while still reporting a median of 1.

        ``stop_gradient`` because the offset is a gauge-fixing statistic, not a parameter:
        it is set by ``recentre_warp`` from the data, and letting the optimiser also push
        on it would reintroduce the flat direction it exists to remove.
        """
        raw = self.net(covariates)[0] - jax.lax.stop_gradient(self.log_rate_offset)
        rate = jnp.exp(self.max_log_rate * jnp.tanh(raw))
        return dt_calendar * rate, rate

    def raw_preactivation(self, covariates: Float[Array, " c"]) -> Float[Array, ""]:
        """Pre-squash, pre-offset output. Used by ``recentre_warp`` to fix the gauge."""
        return self.net(covariates)[0]


# ======================================================================================
# 2. The liquid (CfC) substrate
# ======================================================================================


def recentre_warp(model, covariates: Float[Array, "n c"]):
    """Fix the warp gauge so the MEDIAN business-clock rate is 1 over ``covariates``.

    Sets ``log_rate_offset`` to the median pre-activation, which makes
    ``median(tanh(raw - offset)) = 0`` and hence ``median(rate) = 1``. Only the rate's
    SHAPE across bars then carries information; its overall level is no longer a free
    parameter the optimiser can drift along.

    Call it every step during training (the drift is fast -- Stage 1 saw the rate travel
    from 1.0 to the ceiling at 20.09 within two epochs) and once at the end against the
    FITTING slice, so evaluation uses a frozen, data-independent gauge rather than a
    statistic of whatever batch a row happened to land in.

    Median, not mean: the rate distribution is heavily right-skewed once a weekend
    compresses to ~0.05 while ordinary bars sit near 1, and a mean would let the tail set
    the gauge.
    """
    raws = jax.vmap(model.warp.raw_preactivation)(covariates)
    return eqx.tree_at(
        lambda m: m.warp.log_rate_offset, model, jnp.median(raws)
    )


class LiquidCell(eqx.Module):
    """Closed-form continuous-time cell with an input-dependent relaxation rate.

        tau_eff = tau_min + (tau_max - tau_min) * sigmoid(W_tau [x, u])
        h       = tanh(W_h [x, u])                      <- the relaxation TARGET
        x(t+dt) = h + (x - h) * exp(-dt / tau_eff)      <- exact for constant (h, tau)

    ``tau_eff`` depends on BOTH the state and the input, which is what makes the dynamics
    liquid: the same architecture relaxes fast in a fast market and slowly in a quiet one,
    with no regime label supplied.

    The ``sigmoid`` squash on tau is a bounded, everywhere-differentiable map into
    ``[TAU_MIN, TAU_MAX]``. A ``softplus`` + ``clip`` would be the obvious alternative and
    is worse: ``clip`` zeroes the gradient on the saturated side, which pins a cell that
    has drifted out of range instead of pulling it back.
    """

    tau_net: eqx.nn.MLP
    target_net: eqx.nn.MLP
    hidden_size: int = eqx.field(static=True)
    tau_min: float = eqx.field(static=True)
    tau_max: float = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        width: int = 64,
        depth: int = 1,
        tau_min: float = TAU_MIN,
        tau_max: float = TAU_MAX,
        key: PRNGKeyArray,
    ) -> None:
        k_tau, k_target = jax.random.split(key)
        joint = input_size + hidden_size
        self.tau_net = eqx.nn.MLP(
            joint, hidden_size, width, depth, activation=jax.nn.tanh, key=k_tau
        )
        self.target_net = eqx.nn.MLP(
            joint, hidden_size, width, depth, activation=jax.nn.tanh,
            final_activation=jax.nn.tanh, key=k_target,
        )
        self.hidden_size = hidden_size
        self.tau_min = tau_min
        self.tau_max = tau_max

    def tau_eff(
        self, x: Float[Array, " h"], u: Float[Array, " i"]
    ) -> Float[Array, " h"]:
        """Per-unit relaxation rate. Exposed separately because it is the object the whole
        idea is about -- a diagnostic that collapses to a constant means the "liquid" claim
        has not been earned and the model is a plain gated RNN in disguise."""
        z = jnp.concatenate([x, u])
        raw = self.tau_net(z)
        return self.tau_min + (self.tau_max - self.tau_min) * jax.nn.sigmoid(raw)

    def __call__(
        self, x: Float[Array, " h"], u: Float[Array, " i"], dt: Float[Array, ""]
    ) -> tuple[Float[Array, " h"], Float[Array, " h"]]:
        """One closed-form step over an interval of length ``dt``. Returns ``(x_next, tau)``."""
        z = jnp.concatenate([x, u])
        tau = self.tau_min + (self.tau_max - self.tau_min) * jax.nn.sigmoid(self.tau_net(z))
        h = self.target_net(z)
        decay = jnp.exp(-dt / tau)
        return h + (x - h) * decay, tau


def ode_reference_step(
    cell: LiquidCell,
    x: Float[Array, " h"],
    u: Float[Array, " i"],
    dt: Float[Array, ""],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Float[Array, " h"]:
    """Integrate the SAME ODE numerically with diffrax, as a check on the closed form.

    ``LiquidCell.__call__`` claims to be the exact solution of ``dx/dt = -(x - h)/tau``
    with ``(h, tau)`` frozen at their values at the start of the interval. That claim is
    cheap to assert and easy to get subtly wrong (a sign, a reciprocal, a missing dt), so
    it is checked against a real solver in the tests rather than trusted.

    NOTE this is a VALIDATION path, not the training path. The point of a closed-form
    continuous-time cell is that no solver runs in the loop; putting diffrax in the forward
    pass would give up the entire speed argument for CfC over a neural ODE.
    """
    import diffrax

    z = jnp.concatenate([x, u])
    tau = cell.tau_min + (cell.tau_max - cell.tau_min) * jax.nn.sigmoid(cell.tau_net(z))
    h = cell.target_net(z)

    def vector_field(t, y, args):  # noqa: ARG001
        return -(y - h) / tau

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=float(dt),
        dt0=float(dt) / 100.0,
        y0=x,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
    )
    return solution.ys[-1]


# ======================================================================================
# 3. The spiking abstention readout
# ======================================================================================


@jax.custom_jvp
def spike(v: Float[Array, "..."], alpha: float) -> Float[Array, "..."]:
    """Heaviside spike with a fast-sigmoid surrogate derivative.

    Forward: a hard 0/1 threshold, so abstention is a real discrete decision and not a
    soft weight that quietly lets every bar contribute a little.

    Backward: ``d/dv ~ 1 / (1 + alpha |v|)^2`` (Zenke & Ganguli 2018). The true derivative
    is a Dirac delta -- zero everywhere the optimiser will ever evaluate it -- so without a
    surrogate the readout receives EXACTLY zero gradient and abstention can never be
    learned end-to-end. This function is the whole reason the decision is trainable.
    """
    return (v > 0).astype(v.dtype)


@spike.defjvp
def _spike_jvp(primals, tangents):
    v, alpha = primals
    dv, _ = tangents
    out = (v > 0).astype(v.dtype)
    surrogate = 1.0 / (1.0 + alpha * jnp.abs(v)) ** 2
    return out, surrogate * dv


class LIFState(NamedTuple):
    """Membrane potential of the readout neuron(s)."""

    v: Float[Array, " n"]


class LIFReadout(eqx.Module):
    """Leaky integrate-and-fire readout. A spike means "commit"; silence means "abstain".

        I      = W x + b                                  <- evidence current
        v      <- v * exp(-dt / tau_mem) + I * (1 - exp(-dt / tau_mem))
        s      = Theta(v - v_th)                          <- surrogate-differentiable
        v      <- v - s * v_th                            <- SUBTRACTIVE reset

    Two choices worth defending:

    **Delta-t-aware leak.** The membrane decays by ``exp(-dt / tau_mem)``, not by a fixed
    per-step constant. With a fixed constant the neuron leaks the same amount across a
    weekend as across an hour, so the abstention decision would silently depend on the
    calendar bucketing -- the identical pathology the liquid cell avoids upstream. The
    ``(1 - decay)`` factor on the current keeps the fixed point at ``I`` regardless of
    ``dt``, so a constant input drives the same steady state no matter the sampling.

    **Subtractive reset, not reset-to-zero.** Zeroing discards the overshoot above
    threshold, which is exactly the "how strong was the evidence" signal, and it puts a
    hard discontinuity in the backward path. Subtracting keeps the residue and integrates
    it into the next interval.

    **The bias is initialised ABOVE threshold**, at ``BIAS_INIT_SCALE * v_th``. This is not
    cosmetic, and the exact factor was measured rather than guessed. With a default Linear
    init the current is O(0.1) against ``v_th = 1``, the membrane never reaches threshold,
    and the readout fires on exactly 0% of bars from step one. The surrogate derivative
    there is ``1/(1 + alpha*1)^2 ~ 0.008`` -- nonzero, so the model is not formally stuck,
    but it is two orders of magnitude down and the run spends its budget crawling out of a
    dead readout instead of learning when to abstain.

    Setting the bias exactly AT threshold is not enough either, which is the
    counter-intuitive part: the subtractive reset drops the membrane by ``v_th`` after each
    spike, so a neuron whose steady state sits exactly at threshold approaches it from
    below and mostly fails to re-cross. Mean initial fire rate over 8 seeds, T=200:

        bias / v_th   mean rate   (min, max across seeds)
            1.00        0.107      (0.000, 0.495)   <- some seeds still born dead
            1.25        0.532      (0.365, 0.985)   <- every seed alive
            1.50        0.714      (0.515, 1.000)
            2.00        0.915      (0.780, 1.000)

    1.25 is the default because it is the smallest factor at which NO seed starts dead,
    while leaving the rate low enough that the Lagrangian still has room to work downward.
    """

    proj: eqx.nn.Linear
    log_tau_mem: Float[Array, " n"]
    v_th: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    n_units: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        *,
        n_units: int = 1,
        tau_mem_init: float = 1.0,
        v_th: float = 1.0,
        alpha: float = SURROGATE_ALPHA,
        bias_init_scale: float = BIAS_INIT_SCALE,
        key: PRNGKeyArray,
    ) -> None:
        proj = eqx.nn.Linear(hidden_size, n_units, key=key)
        if bias_init_scale:
            proj = eqx.tree_at(
                lambda m: m.bias,
                proj,
                jnp.full((n_units,), bias_init_scale * v_th, dtype=jnp.float32),
            )
        self.proj = proj
        # learned in log space so tau_mem stays strictly positive without a clamp
        self.log_tau_mem = jnp.full((n_units,), jnp.log(tau_mem_init))
        self.v_th = v_th
        self.alpha = alpha
        self.n_units = n_units

    def initial_state(self) -> LIFState:
        return LIFState(v=jnp.zeros((self.n_units,)))

    def __call__(
        self, state: LIFState, x: Float[Array, " h"], dt: Float[Array, ""]
    ) -> tuple[LIFState, Float[Array, " n"], Float[Array, " n"]]:
        """Returns ``(next_state, spikes, membrane_before_reset)``."""
        current = self.proj(x)
        tau_mem = jnp.exp(self.log_tau_mem)
        decay = jnp.exp(-dt / tau_mem)
        v = state.v * decay + current * (1.0 - decay)
        s = spike(v - self.v_th, self.alpha)
        return LIFState(v=v - s * self.v_th), s, v


# ======================================================================================
# 4. The full model
# ======================================================================================


class GaussianHead(eqx.Module):
    """Predictive ``N(mu, sigma)`` for the next-bar target.

    A distribution, not a point: CRPS needs one, and the abstention decision is only
    meaningful against a stated uncertainty. ``sigma`` is softplus-ed with a floor so the
    optimiser cannot manufacture a zero-width forecast at a lucky point.
    """

    net: eqx.nn.MLP
    sigma_min: float = eqx.field(static=True)

    def __init__(
        self, hidden_size: int, *, width: int = 64, depth: int = 1,
        sigma_min: float = SIGMA_MIN, key: PRNGKeyArray,
    ) -> None:
        self.net = eqx.nn.MLP(hidden_size, 2, width, depth, activation=jax.nn.tanh, key=key)
        self.sigma_min = sigma_min

    def __call__(self, x: Float[Array, " h"]) -> tuple[Float[Array, ""], Float[Array, ""]]:
        raw = self.net(x)
        return raw[0], self.sigma_min + jax.nn.softplus(raw[1])


class StepOutput(NamedTuple):
    """Everything one bar emits. The diagnostics are not optional decoration -- a model
    whose ``tau`` or ``rate`` has collapsed to a constant is a failed version of this idea
    that would otherwise be indistinguishable from a working one by loss alone."""

    mu: Float[Array, ""]
    sigma: Float[Array, ""]
    spike: Float[Array, ""]
    membrane: Float[Array, ""]
    tau_mean: Float[Array, ""]
    business_rate: Float[Array, ""]
    dt_business: Float[Array, ""]


class LTCSpikingModel(eqx.Module):
    """warp -> liquid cell -> (Gaussian head, LIF abstention gate), scanned over a sequence.

    ``__call__`` takes the whole sequence and returns per-bar ``StepOutput``s via
    ``jax.lax.scan``, so the recurrence compiles to a single XLA loop.

    The input contract, deliberately explicit because getting it wrong is silent:
      ``u``            (T, input_size)  features for bar t
      ``dt``           (T,)             calendar time SINCE THE PREVIOUS BAR, > 0
      ``covariates``   (T, cov_size)    activity features driving the business clock
    ``dt[0]`` is the interval before the first bar; pass the median spacing, not 0.
    """

    warp: BusinessTimeWarp
    cell: LiquidCell
    head: GaussianHead
    readout: LIFReadout
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        covariate_size: int,
        *,
        hidden_size: int = 32,
        width: int = 64,
        depth: int = 1,
        key: PRNGKeyArray,
    ) -> None:
        k_warp, k_cell, k_head, k_read = jax.random.split(key, 4)
        self.warp = BusinessTimeWarp(covariate_size, key=k_warp)
        self.cell = LiquidCell(input_size, hidden_size, width=width, depth=depth, key=k_cell)
        self.head = GaussianHead(hidden_size, width=width, depth=depth, key=k_head)
        self.readout = LIFReadout(hidden_size, key=k_read)
        self.hidden_size = hidden_size

    def __call__(
        self,
        u: Float[Array, "t i"],
        dt: Float[Array, " t"],
        covariates: Float[Array, "t c"],
    ) -> StepOutput:
        def step(carry, inputs):
            x, lif = carry
            u_t, dt_t, cov_t = inputs
            dt_bus, rate = self.warp(dt_t, cov_t)
            x_next, tau = self.cell(x, u_t, dt_bus)
            mu, sigma = self.head(x_next)
            lif_next, s, v = self.readout(lif, x_next, dt_bus)
            out = StepOutput(
                mu=mu, sigma=sigma, spike=s[0], membrane=v[0],
                tau_mean=tau.mean(), business_rate=rate, dt_business=dt_bus,
            )
            return (x_next, lif_next), out

        init = (jnp.zeros((self.hidden_size,)), self.readout.initial_state())
        _, outputs = jax.lax.scan(step, init, (u, dt, covariates))
        return outputs


# ======================================================================================
# 5. Loss
# ======================================================================================


def crps_gaussian(
    mu: Float[Array, "..."], sigma: Float[Array, "..."], y: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Closed-form CRPS of ``N(mu, sigma)`` against observation ``y`` (Gneiting & Raftery).

        CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],   z = (y - mu)/sigma

    STRICTLY PROPER, which is the entire reason it is the accuracy term here: it is
    minimised only by the true predictive distribution, so the model cannot buy a better
    score by inflating or shrinking ``sigma`` away from its honest value. Given that this
    project's documented prior is a near-zero-mean, near-unpredictable daily target, a
    scoring rule that punishes dishonest confidence is the one that matters.

    Units are those of ``y``. Lower is better; the "predict the unconditional
    distribution" baseline is the number any result must be read against.
    """
    z = (y - mu) / sigma
    cdf = jax.scipy.stats.norm.cdf(z)
    pdf = jax.scipy.stats.norm.pdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / jnp.sqrt(jnp.pi))


class LossAux(NamedTuple):
    crps_on_fired: Float[Array, ""]
    crps_all: Float[Array, ""]
    fire_rate: Float[Array, ""]
    floor_violation: Float[Array, ""]
    n_fired: Float[Array, ""]


def selective_crps_loss(
    out: StepOutput,
    y: Float[Array, " t"],
    *,
    lam: Float[Array, ""],
    rho_min: float = 0.20,
    lam_rate: float = 0.0,
) -> tuple[Float[Array, ""], LossAux]:
    """CRPS on the bars the model chose to speak on + a Lagrangian floor on the fire rate.

        L = sum_t s_t CRPS_t / sum_t s_t          <- accuracy, conditional on committing
          + lam * relu(rho_min - rate)            <- FLOOR: the anti-collapse term
          + lam_rate * rate                       <- optional sparsity pressure

    --------------------------------------------------------------------------------------
    THE SIGN TRAP, STATED LOUDLY BECAUSE IT INVERTS THE WHOLE DESIGN
    --------------------------------------------------------------------------------------
    A penalty of the form ``lam * spike_rate`` -- penalising firing -- does NOT prevent the
    degenerate optimum. It CAUSES it. The degenerate solution here is silence: abstain on
    every bar, commit to nothing, and the conditional accuracy term is vacuous. Adding a
    term that pushes the rate DOWN rewards exactly that collapse.

    What prevents it is a FLOOR: a penalty that activates when the rate falls BELOW
    ``rho_min``. That is the ``relu(rho_min - rate)`` term, and ``lam`` is its Lagrange
    multiplier, raised by ``dual_ascent`` until the constraint is satisfied.

    ``lam_rate`` is kept as a separate, independently-signed knob for the different job of
    discouraging over-firing once the floor is comfortably met. It defaults to 0.0, and the
    measurements say leave it there. If you ever set ``lam_rate > lam``, you have re-created
    the collapse.

    Selectivity does NOT need ``lam_rate``: the conditional mean induces it on its own,
    because dropping an above-average bar lowers the average by construction. Synthetic
    task, half the bars predictable and half pure noise, 800 steps, floor ``rho_min=0.30``:

        lam_rate   coverage   CRPS|fired   fire|predictable   fire|noise
          0.00       0.518      0.0714          1.000            0.016   <- correct
          0.25       0.316      0.2666          0.521            0.104
          0.50       0.311      0.9538          0.437            0.179
          2.00       0.299      1.0131          0.406            0.187

    At ``lam_rate = 0`` the model recovers the right answer almost exactly: it speaks on
    every predictable bar and on 1.6% of the noise ones, at a coverage that matches the
    true predictable fraction. Turning ``lam_rate`` up pins coverage to the floor but
    DEGRADES the selection -- a flat price per spike is paid regardless of whether the bar
    was worth speaking on, so the model starts discarding predictable bars too
    (fire|predictable falls 1.000 -> 0.406). Use ``lam_rate`` only if a hard coverage
    ceiling is required for external reasons, and read the risk-coverage curve, not the
    headline CRPS, when you do.

    One practical note: selectivity emerged between step 600 and 800 in these runs -- at
    600 steps coverage was still 0.998. A run stopped early looks like a model that refuses
    to abstain, which is a convergence artefact and not a property of the architecture.

    --------------------------------------------------------------------------------------
    WHY THE DENOMINATOR IS FLOORED AND NOT eps-GUARDED. THIS IS THE BUG THAT BIT.
    --------------------------------------------------------------------------------------
    The obvious way to write the conditional mean is ``(s*crps).sum() / (s.sum() + eps)``.
    It collapses, and the floor above cannot save it. Measured, on a synthetic task where
    half the bars are predictable and abstention is exactly the right answer: the fire rate
    went 0.500 -> 0.000 within 100 steps and never recovered, while ``lam`` climbed
    uselessly to 85.

    The reason is the epsilon, not the sign. As ``fired -> 0`` the denominator goes to
    ``eps = 1e-6``, so the derivative of the accuracy term is multiplied by ``1/eps = 1e6``.
    Since ``crps > 0`` everywhere, that enormous gradient points one way: fire LESS. The
    restoring force from the floor is only ``lam / T`` -- order 0.1. A 1e6 collapse force
    against a 0.1 restoring force is not a contest, and no amount of dual ascent closes a
    seven-order-of-magnitude gap.

    Flooring the denominator at ``rho_min * T`` -- the count the constraint demands anyway
    -- fixes it exactly. Below the floor the denominator is CONSTANT, so the accuracy term
    degenerates to a plain sum ``(1/rho_min*T) * sum_t s_t crps_t`` whose gradient is
    bounded and comparable to the floor's; above it, it is the true conditional mean and
    the intended semantics are untouched. After the fix the same run holds coverage at the
    floor and separates the regimes (see the module tests).

    --------------------------------------------------------------------------------------
    The second degenerate optimum, which the floor alone does NOT close
    --------------------------------------------------------------------------------------
    Dividing by ``sum_t s_t`` means the model improves its score by firing only on the
    easiest bars. At exactly ``rho_min`` that is the INTENDED behaviour -- it is what
    selective prediction is -- but it also means ``crps_on_fired`` is NOT comparable to a
    full-coverage baseline. ``crps_all`` rides along in the aux for that reason: any claim
    of an edge has to be made at a stated coverage, against a baseline at the SAME
    coverage. A risk-coverage curve, not a single number.
    """
    crps = crps_gaussian(out.mu, out.sigma, y)
    s = out.spike
    fired = s.sum()
    # NOT `fired + eps` -- see the derivation above; that form collapses to silence.
    #
    # `s.size`, NOT `s.shape[0]`. The loss is written for a single (T,) sequence but is
    # applied under vmap to a (B, T) batch, where shape[0] is the BATCH size. At B=64,
    # T=64 that made the floor count 0.1*64 = 6.4 instead of 0.1*4096 = 409.6, so the
    # denominator only engaged below 0.15% coverage -- i.e. the anti-collapse guard was
    # effectively disabled in exactly the batched setting it ships in, which is the one
    # regime where the eps-collapse documented above reappears. `size` is correct for
    # both the (T,) and (B, T) cases.
    denom = jnp.maximum(fired, rho_min * s.size)
    crps_on_fired = (s * crps).sum() / denom
    rate = s.mean()
    floor_violation = jax.nn.relu(rho_min - rate)
    loss = crps_on_fired + lam * floor_violation + lam_rate * rate
    return loss, LossAux(
        crps_on_fired=crps_on_fired,
        crps_all=crps.mean(),
        fire_rate=rate,
        floor_violation=floor_violation,
        n_fired=fired,
    )


def dual_ascent(
    lam: Float[Array, ""], floor_violation: Float[Array, ""], *, step: float = 0.1,
    lam_max: float = 1e3,
) -> Float[Array, ""]:
    """One dual step on the firing-rate constraint: ``lam <- clip(lam + step * violation)``.

    This is what makes ``lam`` a Lagrange multiplier rather than a hyperparameter you tune
    by hand. While the model fires below ``rho_min`` the violation is positive and ``lam``
    climbs until obeying the floor is cheaper than the accuracy it buys; once satisfied,
    the violation is 0 and ``lam`` stops moving. Clamped at ``lam_max`` so a model that
    physically cannot meet the floor (a dead readout) fails loudly instead of driving the
    multiplier to infinity and NaN-ing the run.

    ``lam`` is held at ``>= 0``: it is the multiplier on an INEQUALITY constraint, and a
    negative multiplier would flip the floor into the collapse-inducing penalty above.
    """
    return jnp.clip(lam + step * floor_violation, 0.0, lam_max)


def risk_coverage_curve(
    out: StepOutput, y: Float[Array, " t"], *, n_points: int = 20
) -> tuple[Float[Array, " p"], Float[Array, " p"]]:
    """CRPS as a function of coverage, by sweeping the membrane threshold post hoc.

    THE honest way to report a selective predictor. A single (accuracy, coverage) pair is
    unfalsifiable -- any model looks good at low enough coverage. The curve is what shows
    whether the abstention decision carries information: if CRPS is FLAT as coverage falls,
    the model is abstaining at random and the spiking readout has earned nothing over a
    coin flip, however good the headline number looks.
    """
    crps = crps_gaussian(out.mu, out.sigma, y)
    order = jnp.argsort(-out.membrane)  # most confident first
    crps_sorted = crps[order]
    ks = jnp.linspace(1, len(y), n_points).astype(int)
    cum = jnp.cumsum(crps_sorted)
    risk = cum[ks - 1] / ks
    coverage = ks / len(y)
    return coverage, risk
