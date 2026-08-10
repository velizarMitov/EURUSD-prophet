"""Idea 3 — training loop and pre-registered evaluation. RESEARCH ONLY.

    python -m src.ltc_experiment --timeframe H1 --epochs 30

Fits ``LTCSpikingModel`` on ``[0:70%]``, scores H_ltc.1 (AURC vs GARCH x day-of-week) and
H_ltc.2 (learned clock vs tick rate) on ``validation[70:80]``, and writes
``results/ltc/``. The test block is never indexed.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.

--------------------------------------------------------------------------------------
STATUS: WRITTEN BUT NOT EXECUTED
--------------------------------------------------------------------------------------
Unlike ``src/ltc_data.py`` (pure numpy, run end-to-end on the real H1 file) and
``src/curl_stress.py`` (validated by Monte Carlo), the JAX path in this module has NOT been
run. JAX is not installed in the environment it was written in. Treat every number it
produces as unverified until the smoke run in ``--dry-run`` mode passes on your machine.
Run that first; it is cheap and it exercises the whole loop on 2000 rows.

--------------------------------------------------------------------------------------
THREE THINGS THAT WILL BITE ON REAL DATA
--------------------------------------------------------------------------------------
1. **The weekend.** On real H1, dt_p999 = 49 and dt_max = 74 (units of median spacing);
   0.8% of bars follow a weekend. The cell composes ``exp(-dt/tau)``, so an unwarped
   weekend annihilates the hidden state -- for any tau of order a few bars,
   ``exp(-65/tau) ~ 0``. ``BusinessTimeWarp`` exists to compress that, and it must be
   initialised near-compressive or training starts from a state-wiping regime it may never
   escape. ``warp_sanity_check`` asserts a weekend maps to a business dt of order a few
   units, not 65. Run it before training; it is the difference between testing the
   subordination hypothesis and testing whether Adam can dig out of a dead initialisation.

2. **Sequence length vs CPU.** JAX on Windows is CPU-only. 42k training bars at H1 with
   ``seq_len=128`` and a scan is minutes per epoch, not seconds. Start at ``seq_len=64``
   and ``--epochs 5`` to time one epoch before committing to a full run. M15 is 350k rows
   -- roughly 6x H1, not cheaper.

3. **The degenerate optimum is silence.** With any firing penalty the global optimum is to
   never spike: zero coverage, zero loss. ``dual_ascent`` raises the multiplier when
   coverage falls below the floor, but the floor has to be a hard constraint the optimiser
   cannot buy its way out of. ``--coverage-floor`` defaults to 0.10 and the loop aborts
   with a clear error if realised coverage collapses below half of it for three consecutive
   epochs -- a silent model that "converged nicely" is the failure mode to fear here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import ltc_data as L

TIMEFRAME_FILES = {"H1": "results/eurusd_h1.csv", "M15": "results/eurusd_m15.csv"}
OUTDIR = Path("results/ltc")


# ======================================================================================
# Windowing
# ======================================================================================


def make_windows(
    x: np.ndarray, y: np.ndarray, dt: np.ndarray, mask: np.ndarray, *, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows whose LAST bar lies inside ``mask``.

    Context may reach back across the split boundary -- that is not leakage, because the
    context is strictly past relative to the predicted bar. What would be leakage is a
    window whose *target* sits in another split, and that cannot happen here.
    """
    idx = np.nonzero(mask)[0]
    idx = idx[idx >= seq_len - 1]
    xw = np.stack([x[i - seq_len + 1 : i + 1] for i in idx])
    dw = np.stack([dt[i - seq_len + 1 : i + 1] for i in idx])
    return xw, y[idx], dw, idx


# ======================================================================================
# Training
# ======================================================================================


def build_model(*, input_size: int, hidden: int, seed: int):
    """Construct the model exactly as ``train_model`` will.

    Split out so the warp can be probed at INITIALISATION, before any optimiser step. The
    spec is explicit that this ordering matters: a warp that starts in the state-annihilating
    regime means the run measures Adam's escape from a dead init rather than the
    subordination hypothesis.
    """
    import jax  # noqa: PLC0415

    from src.ltc_spiking_arch import LTCSpikingModel  # noqa: PLC0415

    key = jax.random.PRNGKey(seed)
    _, model_key = jax.random.split(key)
    return LTCSpikingModel(
        input_size=input_size, covariate_size=input_size, hidden_size=hidden, key=model_key
    )


#: Model B is TICK-BLIND. tick_volume is the held-out observable for H_ltc.2, so neither
#: the covariates nor the warp warm start may see it -- a Clark tick warm start would leak
#: the answer back into the model that is supposed to discover it.
TICK_BLIND_COVARIATES: tuple[str, ...] = (
    "log_return_pct",
    "abs_return_pct",
    "parkinson_pct",
)


def initialise_warp_as_clark_clock(
    model, x: np.ndarray, log_rate_proxy: np.ndarray, *, train_mask: np.ndarray
):
    """Warm-start ``BusinessTimeWarp`` so business time counts TICKS, not hours.

    --------------------------------------------------------------------------------------
    WHY THIS IS NEEDED AT ALL
    --------------------------------------------------------------------------------------
    ``BusinessTimeWarp`` is identity-initialised: ``rate == 1`` exactly, everywhere. That is
    the right gauge-neutral default in the abstract, but on this data it means a weekend
    arrives at the cell completely unwarped -- measured at init, ``compression_ratio =
    49.0`` and weekend state retention ``exp(-dt/tau) = 0.132``. Training from there is what
    the spec warns against: the optimiser has to discover time compression before it can
    learn anything else, and any result then confounds "did the subordination hypothesis
    hold" with "did Adam escape a dead initialisation".

    --------------------------------------------------------------------------------------
    THE TARGET, AND WHY IT IS THE PRINCIPLED ONE RATHER THAN A FUDGE
    --------------------------------------------------------------------------------------
    Clark (1973): the subordinator for a price process is cumulative TRANSACTION COUNT, not
    clock time. Here ``tick_rate = ticks / dt``, so setting ``rate = tick_rate / median
    tick_rate`` gives

        dt_business = dt * rate = ticks / median_tick_rate

    i.e. business time literally counts ticks. That is the hypothesis this family exists to
    test, expressed as a starting point rather than left for the optimiser to rediscover.
    A weekend bar has ordinary tick counts spread over ~49 hours, so its tick rate is far
    below median and it compresses hard -- which is exactly the desired behaviour, arrived
    at from theory rather than by tuning a constant until the check went green.

    Fitted by least squares on the warp head, using TRAIN ROWS ONLY, per the spec's standing
    rule that warp initialisation is a fitted quantity.

    DISCLOSURE: this is a hypothesis-aligned warm start. It biases the warp toward the
    Clark clock. It does NOT directly set ``tau_eff``, which is what H_ltc.2 scores, but the
    two interact through the dynamics, so the initialisation must be reported alongside any
    H_ltc.2 verdict rather than treated as a neutral implementation detail.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    warp = model.warp
    net = warp.net
    tr = np.asarray(train_mask, dtype=bool)

    # target rate = proxy / median(proxy) on train rows, in log space
    centre = float(np.median(np.asarray(log_rate_proxy, dtype=float)[tr]))
    target_log_rate = np.asarray(log_rate_proxy, dtype=float) - centre
    # rate = exp(max_log_rate * tanh(raw)) -> invert, keeping tanh's argument finite
    ratio = np.clip(target_log_rate / warp.max_log_rate, -0.995, 0.995)
    raw_target = np.arctanh(ratio)

    # penultimate activations of the warp MLP, so the head can be solved in closed form
    def penultimate(cov):
        h = cov
        for layer in net.layers[:-1]:
            h = net.activation(layer(h))
        return h

    import jax  # noqa: PLC0415

    feats = np.asarray(jax.vmap(penultimate)(jnp.asarray(x)))
    design = np.column_stack([feats, np.ones(len(feats))])
    coef, *_ = np.linalg.lstsq(design[tr], raw_target[tr], rcond=None)
    weight = jnp.asarray(coef[:-1].reshape(1, -1), dtype=jnp.float32)
    bias = jnp.asarray(coef[-1:], dtype=jnp.float32)

    model = eqx.tree_at(
        lambda m: [m.warp.net.layers[-1].weight, m.warp.net.layers[-1].bias],
        model,
        [weight, bias],
    )
    # pin the gauge immediately, so the warm start's median rate is 1 by construction
    from src.ltc_spiking_arch import recentre_warp  # noqa: PLC0415

    return recentre_warp(model, jnp.asarray(x[tr]))


def price_only_log_rate_proxy(data: pd.DataFrame) -> np.ndarray:
    """Information-rate proxy for MODEL B that never touches ``tick_volume``.

    ``log(parkinson_pct / dt)`` -- realised volatility per unit calendar time, the
    price-only analogue of ``log(ticks / dt)``. It carries the same structural feature that
    makes the warm start work (a weekend spreads its price movement over ~49 hours, so its
    rate is far below median and it compresses), while leaving the tick channel genuinely
    held out.
    """
    park = np.maximum(data["parkinson_pct"].to_numpy(dtype=float), 1e-6)
    return np.log(park / data["dt"].to_numpy(dtype=float))


def train_model(
    xw: np.ndarray,
    yw: np.ndarray,
    dw: np.ndarray,
    *,
    hidden: int = 32,
    epochs: int = 30,
    batch: int = 64,
    lr: float = 3e-3,
    coverage_floor: float = 0.10,
    seed: int = 20260810,
    verbose: bool = True,
    model=None,
    stop_windows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    """Fit on the INNER training windows only. Returns (best_model, history, best_epoch).

    ``model`` may be supplied pre-initialised (e.g. by
    ``initialise_warp_as_clark_clock``); otherwise a fresh one is built.

    ``stop_windows`` is the inner early-stopping slice ``[63:70]``. The epoch is chosen by
    AURC on THAT slice -- never on ``validation[70:80]``, which is the arbiter. Using the
    arbiter to pick an epoch count is model selection on the arbiter, and it would convert
    it into a training set exactly as the spec warns.

    AURC is the selection criterion rather than raw CRPS because AURC is H_ltc.1's primary
    quantity; selecting on a different loss than the one being scored would optimise for
    the wrong thing.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from src.ltc_spiking_arch import (  # noqa: PLC0415
        LTCSpikingModel,
        dual_ascent,
        recentre_warp,
        selective_crps_loss,
    )

    key = jax.random.PRNGKey(seed)
    # `hidden` was accepted and then dropped on the floor here -- the model was always
    # built at the default width, so --hidden silently did nothing.
    if model is None:
        model = build_model(input_size=xw.shape[-1], hidden=hidden, seed=seed)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    lam = jnp.array(1.0)

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(m, xb, yb, db, lam_):
        out = jax.vmap(m)(xb, db, xb)
        # The Lagrangian floor lives INSIDE selective_crps_loss as
        # `lam * relu(rho_min - rate)`. The draft called the loss without `lam`/`rho_min`
        # and then added a second, hand-rolled copy of the same penalty on top, which
        # double-counts the constraint and leaves the loss's own floor running at the
        # default rho_min=0.20 regardless of --coverage-floor.
        loss, aux = selective_crps_loss(out, yb, lam=lam_, rho_min=coverage_floor)
        return loss, aux

    @eqx.filter_jit
    def step(m, st, xb, yb, db, lam_):
        (loss, aux), grads = loss_fn(m, xb, yb, db, lam_)
        updates, st = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), st, eqx.filter(m, eqx.is_inexact_array)
        )
        m = eqx.apply_updates(m, updates)
        # Re-pin the gauge after EVERY update. The drift along the (rate, tau) symmetry is
        # fast -- Stage 1 measured the rate travelling from 1.0 to the exp(3) ceiling in
        # two epochs -- so recentring per epoch would leave it saturated most of the time.
        return recentre_warp(m, xb.reshape(-1, xb.shape[-1])), st, loss, aux

    def aux_rate_probe(m, n_probe: int = 4000):
        """Median warp rate over a fixed slice of fitting covariates -- the gauge monitor.
        A value pinned at exp(MAX_LOG_RATE) means the gauge fix has failed."""
        flat = xw.reshape(-1, xw.shape[-1])[:n_probe]
        _, rate = jax.vmap(m.warp)(jnp.ones(len(flat)), jnp.asarray(flat))
        return rate

    n = len(xw)
    history: list[dict[str, float]] = []
    starved = 0
    best_stop, best_epoch, best_model = float("inf"), -1, model
    for ep in range(epochs):
        t0 = time.perf_counter()
        key, sub = jax.random.split(key)
        order = np.asarray(jax.random.permutation(sub, n))
        losses, covs = [], []
        for i in range(0, n - batch + 1, batch):
            b = order[i : i + batch]
            model, opt_state, loss, aux = step(
                model, opt_state, jnp.asarray(xw[b]), jnp.asarray(yw[b]),
                jnp.asarray(dw[b]), lam,
            )
            losses.append(float(loss))
            # LossAux exposes `fire_rate`, not `coverage`.
            covs.append(float(aux.fire_rate))
            lam = dual_ascent(lam, jnp.maximum(coverage_floor - aux.fire_rate, 0.0))
        cov = float(np.mean(covs))
        # First epoch includes JAX tracing + XLA compilation, which is a one-off cost of
        # tens of seconds on CPU. Report both so a full-run estimate uses the steady-state
        # number, not the compile-inflated first epoch.
        elapsed = time.perf_counter() - t0

        stop_aurc = float("nan")
        if stop_windows is not None:
            xs, ys, ds = stop_windows
            sp = evaluate(model, xs, ys, ds)
            stop_aurc = L.aurc(sp["crps"].to_numpy(), sp["membrane"].to_numpy())
            if np.isfinite(stop_aurc) and stop_aurc < best_stop:
                best_stop, best_epoch = stop_aurc, ep
                best_model = model

        median_rate = float(np.median(np.abs(np.asarray(aux_rate_probe(model)))))
        history.append(
            {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "coverage": cov,
                "lambda": float(lam),
                "stop_aurc": stop_aurc,
                "median_warp_rate": median_rate,
                "seconds": elapsed,
            }
        )
        if verbose:
            print(
                f"  epoch {ep:>3d}  loss {history[-1]['loss']:.5f}  "
                f"coverage {cov:.3f}  lambda {float(lam):.3f}  "
                f"stop_AURC {stop_aurc:.5f}  median_rate {median_rate:.3f}  "
                f"{elapsed:.1f}s{'  (incl. compile)' if ep == 0 else ''}"
            )
        starved = starved + 1 if cov < coverage_floor / 2 else 0
        if starved >= 3:
            raise RuntimeError(
                f"coverage collapsed to {cov:.4f} for 3 epochs (floor {coverage_floor}). "
                "The model bought a good score by refusing to speak -- the degenerate "
                "optimum. Raise --coverage-floor or the dual-ascent rate; do NOT accept "
                "this run because the loss looked good."
            )

    # Freeze the gauge against the FITTING slice so evaluation does not depend on whichever
    # batch a row happened to land in.
    best_model = recentre_warp(best_model, jnp.asarray(xw.reshape(-1, xw.shape[-1])[:20000]))
    return best_model, pd.DataFrame(history), best_epoch


# ======================================================================================
# Evaluation
# ======================================================================================


def evaluate(model, xw, yw, dw) -> pd.DataFrame:
    """Per-row mu, sigma, membrane potential, spike and tau_eff on the given windows."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    out = jax.vmap(model)(jnp.asarray(xw), jnp.asarray(dw), jnp.asarray(xw))

    def get(name: str) -> np.ndarray:
        """Last bar of each window. NO hasattr fallback -- see below."""
        if not hasattr(out, name):
            raise AttributeError(
                f"StepOutput has no field {name!r}; available: {list(out._fields)}. "
                "Refusing to substitute a placeholder for a pre-registered quantity."
            )
        return np.asarray(getattr(out, name))[:, -1]

    # The draft guarded these with `hasattr(out, 'tau_eff') else np.nan`. StepOutput's field
    # is `tau_mean`, so that guard was always False and H_ltc.2 silently scored NaN --
    # a pre-registered hypothesis quietly evaluating to nothing while the run reported
    # success. A missing field is now a loud failure, which is the only safe behaviour for
    # a quantity that goes into the registry.
    frame = pd.DataFrame(
        {
            "mu": get("mu"),
            "sigma": np.abs(get("sigma")),
            "membrane": get("membrane"),
            "spike": get("spike"),
            # per-bar mean of tau_eff across hidden units -- the H_ltc.2 observable
            "tau_eff": get("tau_mean"),
            "business_rate": get("business_rate"),
            "y": np.asarray(yw),
        }
    )
    frame["crps"] = L.crps_gaussian_np(
        frame["mu"].to_numpy(), frame["sigma"].to_numpy(), frame["y"].to_numpy()
    )
    return frame


def score_hypotheses(
    pred: pd.DataFrame, data_val: pd.DataFrame, bench_sigma: np.ndarray
) -> dict[str, object]:
    """H_ltc.1 (AURC) and H_ltc.2 (clock), both on validation[70:80]."""
    y = pred["y"].to_numpy()
    bench_crps = L.crps_gaussian_np(np.zeros(len(y)), bench_sigma, y)

    boot = L.paired_block_bootstrap_daurc(
        pred["crps"].to_numpy(), pred["membrane"].to_numpy(), bench_crps, -bench_sigma
    )
    h1 = {
        "hypothesis": "H_ltc.1_selective_cfc_lif_vs_garch_dow",
        "model_aurc": L.aurc(pred["crps"].to_numpy(), pred["membrane"].to_numpy()),
        "benchmark_aurc": L.aurc(bench_crps, -bench_sigma),
        "coverage": float((pred["spike"].to_numpy() > 0.5).mean()),
        **boot,
        "alpha_bonferroni": 0.025,
    }
    h1["cleared_bar"] = bool(boot["ci_low"] > 0)

    return h1


def clock_statistic(pred: pd.DataFrame, data_val: pd.DataFrame) -> tuple[float, float]:
    """(partial, raw) Spearman of tau_eff against the HELD-OUT log tick rate."""
    tau = pred["tau_eff"].to_numpy()
    ltr = data_val["log_tick_rate"].to_numpy()
    ctrl = data_val[["abs_return_pct", "parkinson_pct"]].to_numpy()
    return L.partial_spearman(tau, ltr, ctrl), L.spearman(tau, ltr)


def score_clock_against_null(
    model, xw, yw, dw, data_val: pd.DataFrame, *, x_fit, log_rate_proxy, train_mask,
    hidden: int, n_null: int = 30, base_seed: int = 90210,
) -> dict[str, object]:
    """H_ltc.2 against an UNTRAINED-INITIALISATION null.

    Stage 1 established that the original zero-null was invalid: ``tau_net`` is a
    near-linear map of its inputs at init, so untrained models already produced partial rho
    of -0.71 / -0.42 / -0.61, and the sign flipped between data slices. The null is
    therefore not "no correlation" but "whatever an identically-constructed, UNTRAINED model
    produces", and only a trained statistic far outside that distribution is evidence of
    learning.

    The null models are built exactly as the trained one -- same architecture, same
    tick-blind covariates, same price-only warp warm start -- and differ ONLY in seed and in
    having received no gradient step. That isolates the effect of training rather than the
    effect of the architecture or the initialisation.
    """
    partial, raw = clock_statistic(evaluate(model, xw, yw, dw), data_val)

    null_partial, null_raw = [], []
    for i in range(n_null):
        m = build_model(input_size=xw.shape[-1], hidden=hidden, seed=base_seed + i)
        m = initialise_warp_as_clark_clock(
            m, x_fit, log_rate_proxy, train_mask=train_mask
        )
        p, r = clock_statistic(evaluate(m, xw, yw, dw), data_val)
        null_partial.append(p)
        null_raw.append(r)

    arr = np.asarray(null_partial, dtype=float)
    mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))
    z = (partial - mu) / sd if sd > 0 else float("nan")
    return {
        "hypothesis": "H_ltc.2_learned_clock_tracks_tick_rate",
        "partial_spearman": float(partial),
        "raw_spearman": float(raw),
        "null_n": int(n_null),
        "null_partial_mean": mu,
        "null_partial_sd": sd,
        "null_partial_min": float(np.nanmin(arr)),
        "null_partial_max": float(np.nanmax(arr)),
        "z_vs_untrained_null": float(z),
        "alpha_bonferroni": 0.025,
        # pre-declared: NEGATIVE direction AND z < -3 against the untrained null
        "cleared_bar": bool(np.isfinite(z) and z < -3.0 and partial < 0),
        "note": (
            "sign pre-declared NEGATIVE; cleared only at z < -3 vs the untrained-seed null. "
            "Model trained TICK-BLIND; tick_volume genuinely held out."
        ),
    }


def warp_sanity_check(
    model,
    x: np.ndarray,
    dt: np.ndarray,
    *,
    weekend_dt: float = 40.0,
    normal_dt: float = 1.5,
    max_rows: int = 4000,
) -> dict[str, float]:
    """Does a weekend annihilate the hidden state? Measured on REAL rows.

    --------------------------------------------------------------------------------------
    WHY THIS TAKES DATA, WHEN THE DRAFT TOOK ONLY THE MODEL
    --------------------------------------------------------------------------------------
    The draft evaluated the warp at ``dt=1`` and ``dt=65`` with the SAME (zero) covariates
    and called the ratio the compression. That number is ``65.0`` for every possible
    parameter setting, so the check could never pass and never fail informatively.

    The reason is in ``BusinessTimeWarp``: ``dt_business = dt_calendar * rate(covariates)``,
    and ``rate`` deliberately does NOT see ``dt_calendar`` -- that independence is what
    makes the warp monotone in time and forbids it learning a non-monotone clock by routing
    through its own argument. So at FIXED covariates the map is exactly linear in dt and
    the ratio is identically ``weekend_dt / normal_dt``.

    Compression is therefore only available through the covariates, and it is genuinely
    there: ``log_tick_rate = log(ticks / dt)`` is strongly negative on a post-weekend bar
    precisely because dt is large. So the honest check evaluates the warp on the covariate
    rows that ACTUALLY accompany weekend gaps, against those of ordinary bars.

    ``state_retention`` is the quantity that actually matters and the one the spec is really
    asking about: ``exp(-dt_business / tau_eff)``, the fraction of the hidden state that
    survives the gap. A compression ratio can look respectable while retention is still
    ~0 if tau happens to be short, so the ratio alone is not sufficient.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    dt = np.asarray(dt, dtype=float)
    normal_idx = np.nonzero(dt <= normal_dt)[0][:max_rows]
    weekend_idx = np.nonzero(dt >= weekend_dt)[0][:max_rows]
    if len(weekend_idx) == 0:
        raise ValueError(
            f"no rows with dt >= {weekend_dt}; cannot measure the weekend regime. "
            "Lower --weekend-dt or check add_delta_t."
        )

    hidden = jnp.zeros((model.hidden_size,))

    def probe(dt_i, x_i):
        dt_bus, rate = model.warp(dt_i, x_i)
        tau = model.cell.tau_eff(hidden, x_i).mean()
        return dt_bus, rate, tau, jnp.exp(-dt_bus / tau)

    def run(idx):
        d, r, t, keep = jax.vmap(probe)(jnp.asarray(dt[idx]), jnp.asarray(x[idx]))
        return (float(np.median(np.asarray(d))), float(np.median(np.asarray(r))),
                float(np.median(np.asarray(t))), float(np.median(np.asarray(keep))))

    n_dtb, n_rate, n_tau, n_keep = run(normal_idx)
    w_dtb, w_rate, w_tau, w_keep = run(weekend_idx)
    ratio = w_dtb / max(n_dtb, 1e-9)
    return {
        "n_normal_rows": int(len(normal_idx)),
        "n_weekend_rows": int(len(weekend_idx)),
        "median_calendar_dt_weekend": float(np.median(dt[weekend_idx])),
        "business_dt_normal_bar": n_dtb,
        "business_dt_weekend": w_dtb,
        "warp_rate_normal": n_rate,
        "warp_rate_weekend": w_rate,
        "tau_eff_normal": n_tau,
        "tau_eff_weekend": w_tau,
        "state_retention_normal": n_keep,
        "state_retention_weekend": w_keep,
        "compression_ratio": ratio,
        "healthy": bool(ratio < 20.0 and w_keep > 1e-3),
    }


# ======================================================================================
# Entry point
# ======================================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Idea 3 — CfC + LIF selective forecaster")
    ap.add_argument("--timeframe", choices=list(TIMEFRAME_FILES), default="H1")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--coverage-floor", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--n-null", type=int, default=30,
                    help="untrained seeds forming the H_ltc.2 null")
    ap.add_argument("--dry-run", action="store_true", help="2000 rows, 2 epochs — smoke test")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    L.init_hypothesis_log()

    data = L.build_dataset(L.load_bars(TIMEFRAME_FILES[args.timeframe]))
    if args.dry_run:
        data = data.iloc[:2000].copy()
        args.epochs = 2
    n = len(data)

    # 63 / 7 / 10 / 20. The inner [63:70] slice chooses the epoch count; validation[70:80]
    # is the arbiter and is used ONLY for the final score. The test block is never indexed.
    lo_fit = int(0.63 * n)
    lo_val = int(L.TRAIN_FRACTION * n)
    hi_val = int((L.TRAIN_FRACTION + L.VAL_FRACTION) * n)
    fit = np.zeros(n, dtype=bool); fit[:lo_fit] = True
    stop = np.zeros(n, dtype=bool); stop[lo_fit:lo_val] = True
    train = np.zeros(n, dtype=bool); train[:lo_val] = True
    val = np.zeros(n, dtype=bool); val[lo_val:hi_val] = True
    print(f"{args.timeframe}: {n:,} rows | fit[0:63] {fit.sum():,} | "
          f"stop[63:70] {stop.sum():,} | val[70:80] {val.sum():,}")
    print(L.gap_profile(data).to_string(index=False))

    y = data["target_return_pct"].to_numpy(dtype=float)
    dt = data["dt"].to_numpy(dtype=float)

    # Benchmark stays fitted on [0:70] exactly as registered -- it uses raw returns and no
    # scaler, so it is unaffected by the inner split.
    bench_sigma_all, bench_par = L.garch_dow_sigma(y, data["dow"].to_numpy(), train_mask=train)
    print(f"GARCH a={bench_par['alpha']:.4f} b={bench_par['beta']:.4f}")

    results: dict[str, object] = {}
    preds: dict[str, pd.DataFrame] = {}

    for tag, columns, proxy_name in (
        ("A_full", L.COVARIATES, "clark_tick_rate"),
        ("B_tick_blind", TICK_BLIND_COVARIATES, "price_only_parkinson_per_dt"),
    ):
        print(f"\n{'=' * 78}\nMODEL {tag}  covariates={list(columns)}\n{'=' * 78}")
        # Scaler fitted on the FITTING slice only, so the early-stopping slice does not
        # leak into standardisation.
        x = L.covariate_matrix(data, train_mask=fit, columns=columns)
        proxy = (
            data["log_tick_rate"].to_numpy(dtype=float)
            if proxy_name == "clark_tick_rate"
            else price_only_log_rate_proxy(data)
        )

        xw_fit, yw_fit, dw_fit, _ = make_windows(x, y, dt, fit, seq_len=args.seq_len)
        xw_st, yw_st, dw_st, _ = make_windows(x, y, dt, stop, seq_len=args.seq_len)
        xw_va, yw_va, dw_va, idx_va = make_windows(x, y, dt, val, seq_len=args.seq_len)
        print(f"windows: fit {len(xw_fit):,} | stop {len(xw_st):,} | val {len(xw_va):,}")

        m0 = build_model(input_size=x.shape[-1], hidden=args.hidden, seed=args.seed)
        m0 = initialise_warp_as_clark_clock(m0, x, proxy, train_mask=fit)
        warp_init = warp_sanity_check(m0, x, dt)
        print(f"warp at init ({proxy_name}): ratio {warp_init['compression_ratio']:.3f}  "
              f"retention_weekend {warp_init['state_retention_weekend']:.4f}  "
              f"healthy {warp_init['healthy']}")

        model, history, best_epoch = train_model(
            xw_fit, yw_fit, dw_fit,
            hidden=args.hidden, epochs=args.epochs, batch=args.batch, lr=args.lr,
            coverage_floor=args.coverage_floor, seed=args.seed, model=m0,
            stop_windows=(xw_st, yw_st, dw_st),
        )
        print(f"selected epoch {best_epoch} by inner-stop AURC "
              f"({history['stop_aurc'].min():.6f})")
        warp_post = warp_sanity_check(model, x, dt)
        print("warp post-training:", json.dumps(warp_post, default=str))

        pred = evaluate(model, xw_va, yw_va, dw_va)
        preds[tag] = pred
        history.to_csv(OUTDIR / f"training_history_{tag}.csv", index=False)
        pred.to_csv(OUTDIR / f"validation_predictions_{tag}.csv", index=False)

        if tag == "A_full":
            h1 = score_hypotheses(pred, data.iloc[idx_va], bench_sigma_all[idx_va])
            h1.update({"selected_epoch": int(best_epoch), "model": tag})
            results["H_ltc.1"] = h1
            results["warp_A"] = warp_post
        else:
            h2 = score_clock_against_null(
                model, xw_va, yw_va, dw_va, data.iloc[idx_va],
                x_fit=x, log_rate_proxy=proxy, train_mask=fit,
                hidden=args.hidden, n_null=args.n_null,
            )
            h2.update({"selected_epoch": int(best_epoch), "model": tag})
            results["H_ltc.2"] = h2
            results["warp_B"] = warp_post

    (OUTDIR / "hypothesis_scores.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print("\n" + json.dumps(results, indent=2, default=str))
    print(f"\nwrote -> {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
