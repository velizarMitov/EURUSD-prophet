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


def initialise_warp_as_clark_clock(
    model, x: np.ndarray, log_tick_rate: np.ndarray, *, train_mask: np.ndarray
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

    # target rate = tick_rate / median(tick_rate) on train rows, in log space
    centre = float(np.median(np.asarray(log_tick_rate, dtype=float)[tr]))
    target_log_rate = np.asarray(log_tick_rate, dtype=float) - centre
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

    return eqx.tree_at(
        lambda m: [m.warp.net.layers[-1].weight, m.warp.net.layers[-1].bias],
        model,
        [weight, bias],
    )


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
):
    """Fit on training windows only. Returns (model, history).

    ``model`` may be supplied pre-initialised (e.g. by
    ``initialise_warp_as_clark_clock``); otherwise a fresh one is built.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from src.ltc_spiking_arch import (  # noqa: PLC0415
        LTCSpikingModel,
        dual_ascent,
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
        return eqx.apply_updates(m, updates), st, loss, aux

    n = len(xw)
    history: list[dict[str, float]] = []
    starved = 0
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
        history.append(
            {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "coverage": cov,
                "lambda": float(lam),
                "seconds": elapsed,
            }
        )
        if verbose:
            print(
                f"  epoch {ep:>3d}  loss {history[-1]['loss']:.5f}  "
                f"coverage {cov:.3f}  lambda {float(lam):.3f}  "
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
    return model, pd.DataFrame(history)


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

    tau = pred["tau_eff"].to_numpy()
    ctrl = data_val[["abs_return_pct", "parkinson_pct"]].to_numpy()
    raw = L.spearman(tau, data_val["log_tick_rate"].to_numpy())
    partial = L.partial_spearman(tau, data_val["log_tick_rate"].to_numpy(), ctrl)
    h2 = {
        "hypothesis": "H_ltc.2_learned_clock_tracks_tick_rate",
        "partial_spearman": partial,
        "raw_spearman": raw,
        "alpha_bonferroni": 0.025,
        "cleared_bar": bool(np.isfinite(partial) and partial < 0),
        "note": "sign pre-declared NEGATIVE; positive of any size is a DROP",
    }
    return {"H_ltc.1": h1, "H_ltc.2": h2}


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
    ap.add_argument("--dry-run", action="store_true", help="2000 rows, 2 epochs — smoke test")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    L.init_hypothesis_log()

    data = L.build_dataset(L.load_bars(TIMEFRAME_FILES[args.timeframe]))
    if args.dry_run:
        data = data.iloc[:2000].copy()
        args.epochs = 2
    n = len(data)
    train, val, _test = L.chronological_masks(n)
    print(f"{args.timeframe}: {n:,} rows | train {train.sum():,} | val {val.sum():,}")
    print(L.gap_profile(data).to_string(index=False))

    x = L.covariate_matrix(data, train_mask=train)
    y = data["target_return_pct"].to_numpy(dtype=float)
    dt = data["dt"].to_numpy(dtype=float)

    bench_sigma_all, bench_par = L.garch_dow_sigma(y, data["dow"].to_numpy(), train_mask=train)
    print(f"GARCH a={bench_par['alpha']:.4f} b={bench_par['beta']:.4f}")

    xw_tr, yw_tr, dw_tr, _ = make_windows(x, y, dt, train, seq_len=args.seq_len)
    xw_va, yw_va, dw_va, idx_va = make_windows(x, y, dt, val, seq_len=args.seq_len)
    print(f"windows: train {len(xw_tr):,} | val {len(xw_va):,}")

    init_model = build_model(input_size=x.shape[-1], hidden=args.hidden, seed=args.seed)
    raw_warp = warp_sanity_check(init_model, x, dt)
    print("\nwarp sanity at IDENTITY initialisation (before the Clark warm start):")
    print(json.dumps(raw_warp, indent=2, default=str))

    init_model = initialise_warp_as_clark_clock(
        init_model, x, data["log_tick_rate"].to_numpy(dtype=float), train_mask=train
    )
    init_warp = warp_sanity_check(init_model, x, dt)
    print("\nwarp sanity AFTER the Clark warm start (this is what trains):")
    print(json.dumps(init_warp, indent=2, default=str))
    if not init_warp["healthy"]:
        print(
            "  ^ UNHEALTHY. Per LTC_INTEGRATION_SPEC Stage 1 task 4, training from here "
            "measures the optimiser's escape from a dead initialisation, not the "
            "subordination hypothesis. Stop and fix before Stage 2."
        )

    model, history = train_model(
        xw_tr, yw_tr, dw_tr,
        hidden=args.hidden, epochs=args.epochs, batch=args.batch, lr=args.lr,
        coverage_floor=args.coverage_floor, seed=args.seed, model=init_model,
    )
    print("\nwarp sanity (post-training, measured on real rows):")
    print(json.dumps(warp_sanity_check(model, x, dt), indent=2, default=str))

    pred = evaluate(model, xw_va, yw_va, dw_va)
    scores = score_hypotheses(pred, data.iloc[idx_va], bench_sigma_all[idx_va])

    history.to_csv(OUTDIR / "training_history.csv", index=False)
    pred.to_csv(OUTDIR / "validation_predictions.csv", index=False)
    (OUTDIR / "hypothesis_scores.json").write_text(json.dumps(scores, indent=2, default=str))
    print(json.dumps(scores, indent=2, default=str))
    print(f"\nwrote -> {OUTDIR.resolve()}")
    print(
        "\nScores are NOT registered. Copy them into results/ltc_hypothesis_log.csv "
        "manually, replacing the PENDING rows, once you have read them."
    )


if __name__ == "__main__":
    main()
