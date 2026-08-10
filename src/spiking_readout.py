"""Does a trained spiking readout beat a trivial sigma threshold? RESEARCH ONLY.

NEW FAMILY: ``results/spiking_readout_hypothesis_log.csv``. It does NOT reuse the LTC
family's alpha ladder -- this asks a different question of a different substrate on
different instruments.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.

--------------------------------------------------------------------------------------
THE QUESTION, AND WHY THE SUBSTRATE IS DELIBERATELY BORING
--------------------------------------------------------------------------------------
The LTC family (``results/ltc_hypothesis_log.csv``) ended with H_ltc.1 clearing a marginal
AURC bar while its learned business-time mechanism had collapsed -- weekend state retention
2.8e-09, tau and warp rate both pinned on their bounds. So the continuous-time machinery
cannot be what produced the edge.

That leaves an obvious suspect: the SPIKING READOUT. This module isolates it by holding the
substrate fixed and boring -- a plain GRU, no continuous-time dynamics, no learned warp, no
gauge freedom. ``Delta-t`` enters as an ordinary input feature (``log_dt`` plus a gap
indicator), not as ODE time. If a readout effect survives on that substrate, it belongs to
the readout.

    arm   substrate            readout                       isolates
    S0    GRU (trained once)   none, always forecasts        baseline quality
    S1    THE SAME weights     fixed sigma threshold         does abstention help at all
    S2    THE SAME weights     trained LIF                   does LIF beat the trivial rule

S1 and S2 share S0's frozen weights EXACTLY -- S2 trains only the readout on top of a
substrate that receives no gradient. That is a stronger control than retraining each arm,
because any AURC difference is then attributable to the readout and to nothing else.

--------------------------------------------------------------------------------------
WHY NOT EURUSD, AND WHY TWO INSTRUMENTS
--------------------------------------------------------------------------------------
The motivating observation was made on EURUSD ``validation[70:80]`` after that slice had
already been used to score H_ltc.1, and was then examined from three further angles. It is
post-hoc and that slice is spent for this question. It generates the hypothesis; it cannot
evidence it.

The arbiter is therefore GBPUSD H1 and AUDUSD H1, each with its own ``[0:70]/[70:80]``
split, and the effect must hold on BOTH. One instrument is an anecdote. This mirrors the
replication requirement the ``h1_direction`` family already uses (H_dir.3-5).

NOTE these files carry OHLC only -- no ``tick_volume`` -- so ``ltc_data.load_bars`` cannot
read them. That is fine and even convenient: this hypothesis is about the readout, so a
price-only covariate set is the honest input, and ``build_price_dataset`` below is
deliberately tick-free.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src import ltc_data as L

INSTRUMENTS: dict[str, str] = {
    "GBPUSD": "results/pooled_h1/GBPUSD_h1.csv",
    "AUDUSD": "results/pooled_h1/AUDUSD_h1.csv",
}
OUTDIR = Path("results/spiking_readout")
SPK_LOG = "results/spiking_readout_hypothesis_log.csv"

#: Price-only. No tick features exist on these instruments, and none are needed: the
#: hypothesis is about the readout, not about what the substrate is fed.
COVARIATES: tuple[str, ...] = (
    "log_return_pct",
    "abs_return_pct",
    "parkinson_pct",
    "log_dt",
    "is_gap",
)

#: A bar whose predecessor is more than this many median spacings away. On H1 the gap
#: distribution is bimodal -- ~1 or ~49 -- so any threshold between them selects the same
#: set (share_dt_gt_4 = 0.852%, share_dt_gt_40 = 0.838%); 4 is used so holidays are caught
#: alongside weekends.
GAP_THRESHOLD = 4.0

#: Pre-registered before any training. Frozen for the whole family.
PREREG = {
    "seq_len": 64,
    "hidden": 32,
    "lr": 3e-3,
    "batch": 64,
    "epochs_substrate": 25,
    "epochs_readout": 25,
    "coverage_floor": 0.10,
    "s1_target_coverage": 0.50,
    "block_len": 24,
    "n_boot": 2000,
    "seed": 20260810,
}


# ======================================================================================
# Data
# ======================================================================================


def build_price_dataset(path: str | Path) -> pd.DataFrame:
    """OHLC-only dataset with an explicit Delta-t, no tick columns anywhere.

    Same conventions as ``ltc_data.build_dataset``: percent log returns, ``shift(-1)``
    target, every covariate known at bar close, final row dropped.
    """
    df = pd.read_csv(path)
    time_col = "time" if "time" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[time_col], utc=True)
    df = df.drop(columns=[time_col]).set_index(idx).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    out = L.add_delta_t(df)
    close = out["close"].to_numpy(dtype=float)
    logret = np.concatenate([[np.nan], np.diff(np.log(close))]) * 100.0
    out["log_return_pct"] = logret
    hl = np.log(out["high"].to_numpy(dtype=float) / out["low"].to_numpy(dtype=float))
    out["parkinson_pct"] = np.sqrt(np.maximum(hl, 0) ** 2 / (4 * np.log(2))) * 100.0
    out["abs_return_pct"] = np.abs(logret)
    # Delta-t as an ORDINARY FEATURE, not as ODE time -- that is the point of this arm.
    out["log_dt"] = np.log(np.maximum(out["dt"].to_numpy(dtype=float), 1e-9))
    out["is_gap"] = (out["dt"].to_numpy(dtype=float) > GAP_THRESHOLD).astype(float)
    out["dow"] = out.index.dayofweek
    out["target_return_pct"] = np.concatenate([logret[1:], [np.nan]])
    return out.dropna(subset=["log_return_pct", "target_return_pct", "dt"]).copy()


def make_windows(x, y, mask, *, seq_len: int):
    """Sliding windows whose LAST bar lies inside ``mask``; context may reach back across
    a split boundary, which is strictly past relative to the predicted bar."""
    idx = np.nonzero(mask)[0]
    idx = idx[idx >= seq_len - 1]
    xw = np.stack([x[i - seq_len + 1 : i + 1] for i in idx])
    return xw, y[idx], idx


# ======================================================================================
# Substrate
# ======================================================================================


def build_substrate(*, input_size: int, hidden: int, seed: int):
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    from src.ltc_spiking_arch import GaussianHead  # noqa: PLC0415

    class GRUSubstrate(eqx.Module):
        """Plain GRU + Gaussian head. No warp, no tau, no gauge freedom -- by design."""

        cell: eqx.nn.GRUCell
        head: GaussianHead
        hidden_size: int = eqx.field(static=True)

        def __init__(self, input_size: int, hidden: int, *, key):
            k1, k2 = jax.random.split(key)
            self.cell = eqx.nn.GRUCell(input_size, hidden, key=k1)
            self.head = GaussianHead(hidden, key=k2)
            self.hidden_size = hidden

        def __call__(self, u):
            import jax.numpy as jnp  # noqa: PLC0415

            def step(h, u_t):
                h = self.cell(u_t, h)
                mu, sigma = self.head(h)
                return h, (mu, sigma, h)

            _, (mu, sigma, hs) = jax.lax.scan(
                step, jnp.zeros((self.hidden_size,)), u
            )
            return mu, sigma, hs

    return GRUSubstrate(input_size, hidden, key=jax.random.PRNGKey(seed))


def train_substrate(xw, yw, *, hidden, epochs, batch, lr, seed, stop_windows, verbose=True):
    """S0: fit the GRU with PLAIN mean CRPS. No abstention, no coverage term.

    Epoch chosen by mean CRPS on the inner stop slice ``[63:70]`` -- never on the arbiter.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from src.ltc_spiking_arch import crps_gaussian  # noqa: PLC0415

    model = build_substrate(input_size=xw.shape[-1], hidden=hidden, seed=seed)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(m, xb, yb):
        mu, sigma, _ = jax.vmap(m)(xb)
        return crps_gaussian(mu, sigma, yb).mean()

    @eqx.filter_jit
    def step(m, st, xb, yb):
        loss, grads = loss_fn(m, xb, yb)
        updates, st = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), st, eqx.filter(m, eqx.is_inexact_array)
        )
        return eqx.apply_updates(m, updates), st, loss

    @eqx.filter_jit
    def stop_crps(m, xb, yb):
        mu, sigma, _ = jax.vmap(m)(xb)
        return crps_gaussian(mu[:, -1], sigma[:, -1], yb).mean()

    n = len(xw)
    key = jax.random.PRNGKey(seed)
    best, best_epoch, best_model = float("inf"), -1, model
    history = []
    xs, ys = stop_windows
    for ep in range(epochs):
        t0 = time.perf_counter()
        key, sub = jax.random.split(key)
        order = np.asarray(jax.random.permutation(sub, n))
        losses = []
        for i in range(0, n - batch + 1, batch):
            b = order[i : i + batch]
            model, opt_state, loss = step(
                model, opt_state, jnp.asarray(xw[b]), jnp.asarray(yw[b])
            )
            losses.append(float(loss))
        sc = float(stop_crps(model, jnp.asarray(xs), jnp.asarray(ys)))
        if sc < best:
            best, best_epoch, best_model = sc, ep, model
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "stop_crps": sc,
                        "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"    S0 epoch {ep:>3d}  loss {history[-1]['loss']:.6f}  "
                  f"stop_CRPS {sc:.6f}  {history[-1]['seconds']:.1f}s")
    return best_model, pd.DataFrame(history), best_epoch


# ======================================================================================
# Readout (S2) on the FROZEN substrate
# ======================================================================================


def train_readout(substrate, xw, yw, dtw, *, hidden, epochs, batch, lr, coverage_floor,
                  seed, stop_windows, verbose=True):
    """S2: fit ONLY a LIF readout on top of a substrate that receives no gradient.

    ``eqx.filter_grad`` is given a spec that marks every substrate leaf non-differentiable,
    so S1 and S2 are guaranteed to sit on numerically identical mu/sigma. Any AURC
    difference between them is the readout and nothing else.

    ``selective_crps_loss`` and ``dual_ascent`` are imported UNCHANGED from
    ``ltc_spiking_arch`` -- they are already tested there, including the collapse-to-silence
    regression and the batched coverage floor.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    from src.ltc_spiking_arch import (  # noqa: PLC0415
        LIFReadout,
        StepOutput,
        dual_ascent,
        selective_crps_loss,
    )

    readout = LIFReadout(hidden, key=jax.random.PRNGKey(seed + 1))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(eqx.filter(readout, eqx.is_inexact_array))
    lam = jnp.array(1.0)

    def forward(rd, xb, db):
        mu, sigma, hs = jax.vmap(substrate)(xb)

        def one(h_seq, dt_seq):
            def st(state, inp):
                h_t, dt_t = inp
                state, s, v = rd(state, h_t, dt_t)
                return state, (s[0], v[0])

            _, (s, v) = jax.lax.scan(st, rd.initial_state(), (h_seq, dt_seq))
            return s, v

        spikes, memb = jax.vmap(one)(hs, db)
        return mu, sigma, spikes, memb

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(rd, xb, yb, db, lam_):
        mu, sigma, spikes, memb = forward(rd, xb, db)
        out = StepOutput(mu=mu, sigma=sigma, spike=spikes, membrane=memb,
                         tau_mean=memb, business_rate=memb, dt_business=db)
        return selective_crps_loss(out, yb, lam=lam_, rho_min=coverage_floor)

    @eqx.filter_jit
    def step(rd, st, xb, yb, db, lam_):
        (loss, aux), grads = loss_fn(rd, xb, yb, db, lam_)
        updates, st = opt.update(
            eqx.filter(grads, eqx.is_inexact_array), st, eqx.filter(rd, eqx.is_inexact_array)
        )
        return eqx.apply_updates(rd, updates), st, loss, aux

    @eqx.filter_jit
    def stop_eval(rd, xb, yb, db):
        from src.ltc_spiking_arch import crps_gaussian  # noqa: PLC0415

        mu, sigma, _, memb = forward(rd, xb, db)
        return crps_gaussian(mu[:, -1], sigma[:, -1], yb), memb[:, -1]

    n = len(xw)
    key = jax.random.PRNGKey(seed)
    best, best_epoch, best_rd = float("inf"), -1, readout
    history = []
    xs, ys, ds = stop_windows
    for ep in range(epochs):
        t0 = time.perf_counter()
        key, sub = jax.random.split(key)
        order = np.asarray(jax.random.permutation(sub, n))
        losses, covs = [], []
        for i in range(0, n - batch + 1, batch):
            b = order[i : i + batch]
            readout, opt_state, loss, aux = step(
                readout, opt_state, jnp.asarray(xw[b]), jnp.asarray(yw[b]),
                jnp.asarray(dtw[b]), lam,
            )
            losses.append(float(loss))
            covs.append(float(aux.fire_rate))
            lam = dual_ascent(lam, jnp.maximum(coverage_floor - aux.fire_rate, 0.0))
        c, m = stop_eval(readout, jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(ds))
        sa = L.aurc(np.asarray(c), np.asarray(m))
        if np.isfinite(sa) and sa < best:
            best, best_epoch, best_rd = sa, ep, readout
        history.append({"epoch": ep, "loss": float(np.mean(losses)),
                        "coverage": float(np.mean(covs)), "stop_aurc": sa,
                        "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"    S2 epoch {ep:>3d}  loss {history[-1]['loss']:.6f}  "
                  f"coverage {history[-1]['coverage']:.3f}  stop_AURC {sa:.6f}  "
                  f"{history[-1]['seconds']:.1f}s")
    return best_rd, pd.DataFrame(history), best_epoch


# ======================================================================================
# Registry — written PENDING, before anything is trained
# ======================================================================================


def init_hypothesis_log(path: str | Path = SPK_LOG) -> Path:
    """Pre-registration rows for the spiking-readout family.

    ONE testable hypothesis, so ``alpha = 0.05``. The two-instrument requirement is a
    CONJUNCTION, not two independent chances: both GBPUSD and AUDUSD must exclude zero in
    S2's favour. Requiring both is far stricter than a Bonferroni split (two independent
    tests each at 0.05 give a joint false-positive rate of 0.0025), so no further
    correction is applied -- and one instrument clearing on its own is a DROP.

    Row 2 is an OBSERVATION, not a hypothesis. It spends no alpha and does not enter the
    ladder; it exists so a real defect found in the LTC architecture is not lost.
    """
    p = Path(path)
    if p.exists():
        return p
    rows = [
        {
            "n": 1,
            "date": "",
            "hypothesis": "H_spk.1_trained_lif_beats_fixed_sigma_threshold",
            "arbiter": "GBPUSD_h1 validation[70:80] AND AUDUSD_h1 validation[70:80]",
            "primary_quantity": "delta_AURC (S1 - S2); positive favours the trained LIF",
            "point_estimate": "",
            "ci_low": "",
            "ci_high": "",
            "benchmark": (
                "S1 = the SAME frozen GRU substrate with a fixed sigma threshold "
                "calibrated on [0:63]; ranked by -sigma"
            ),
            "alpha_bonferroni": 0.05,
            "cleared_bar": "",
            "verdict": "PENDING",
            "notes": (
                "PRE-REGISTERED 2026-08-10 BEFORE ANY TRAINING. Substrate is a plain GRU "
                "(no continuous-time dynamics, no learned warp, no gauge freedom); Delta-t "
                "enters as ordinary features log_dt + is_gap, NOT as ODE time. S0/S1/S2 "
                "share NUMERICALLY IDENTICAL substrate weights: S0 is trained once with "
                "plain mean CRPS, then frozen; S1 adds a fixed sigma threshold; S2 trains "
                "ONLY a LIF readout on top, with the substrate receiving no gradient. So "
                "any delta is attributable to the readout alone. "
                "Hyperparameters frozen at pre-registration: seq_len=64, hidden=32, "
                "lr=0.003, batch=64, epochs 25/25, coverage_floor=0.10, S1 target coverage "
                "0.50, block_len=24, n_boot=2000, seed=20260810. Epoch selected on an inner "
                "[63:70] slice; the arbiter never chooses an epoch. "
                "BLOCK-LENGTH SENSITIVITY IS PRE-DECLARED, not post-hoc: ci_low will be "
                "reported at block_len 24 (registered), 96 and 168 for BOTH instruments, "
                "because the LTC family cleared at 24 and failed at 96 and 168 and that "
                "fragility must be visible from the start. "
                "CLEARED ONLY IF BOTH instruments exclude zero in S2's favour. One "
                "instrument is an anecdote. "
                "MOTIVATION IS POST-HOC AND DOES NOT EVIDENCE THIS: on EURUSD "
                "validation[70:80] -- already spent scoring H_ltc.1 and then examined from "
                "three further angles -- ranking by LIF membrane gave AURC 0.029627 vs "
                "0.045060 for -sigma, 0.045298 for -|mu| and 0.038540 for random (200 "
                "draws, sd 0.00063); corr(membrane, -sigma) = -0.2052, Spearman -0.4065; "
                "LIF top decile risk 0.023684, 38.6% below full coverage. That slice is "
                "exhausted for this question, which is why the arbiter is two other "
                "instruments."
            ),
        },
        {
            "n": 2,
            "date": "2026-08-10",
            "hypothesis": "OBS_ltc_model_A_sigma_head_is_anticalibrated",
            "arbiter": "EURUSD validation[70:80] (descriptive, post-hoc, no alpha spent)",
            "primary_quantity": "Spearman(predicted sigma, realised |y|); AURC of -sigma vs random",
            "point_estimate": -0.1834,
            "ci_low": "",
            "ci_high": "",
            "benchmark": "random ranking, AURC 0.038540 (mean of 200 draws, sd 0.00063)",
            "alpha_bonferroni": "n/a — OBSERVATION, spends no alpha and is not in the ladder",
            "cleared_bar": "n/a",
            "verdict": "RECORDED DEFECT (not a hypothesis test)",
            "notes": (
                "A real defect in the LTC architecture (src/ltc_spiking_arch.py "
                "GaussianHead as trained by src/ltc_experiment.py Model A), recorded so it "
                "is not lost. The sigma head is BOTH nearly constant AND inverted. "
                "Predicted sigma spans only 0.068789 to 0.071364 across its own deciles (a "
                "3.7% range) while realised mean |y| over those same deciles runs 0.070239 "
                "down to 0.036603 -- i.e. the bars the model calls MOST uncertain are the "
                "ones it actually predicts BEST. Spearman(sigma, |y|) = -0.1834. "
                "Consequences: ranking by -sigma gives AURC 0.045060, WORSE than random "
                "(0.038540); ranking by +sigma, i.e. deliberately inverting the model's own "
                "confidence, gives 0.033317, better than random. Mean CRPS by sigma decile "
                "falls monotonically 0.053002 -> 0.029655 in the wrong direction. "
                "This is why H_spk.1 uses a sigma threshold as the S1 control rather than "
                "assuming sigma is a sane confidence signal, and it is an independent "
                "reason to distrust the H_ltc.1 clearance: the architecture's own "
                "uncertainty estimate is not merely weak, it is backwards. "
                "Descriptive only -- measured on a slice already spent, so it is a defect "
                "report, not a test."
            ),
        },
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ======================================================================================
# Scoring
# ======================================================================================


def score_instrument(name: str, path: str, *, args) -> dict[str, object]:
    """Train S0, derive S1, train S2, and score all three on this instrument's arbiter."""
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from src.ltc_spiking_arch import crps_gaussian  # noqa: PLC0415

    data = build_price_dataset(path)
    n = len(data)
    lo_fit, lo_val = int(0.63 * n), int(0.70 * n)
    hi_val = int(0.80 * n)
    fit = np.zeros(n, bool); fit[:lo_fit] = True
    stop = np.zeros(n, bool); stop[lo_fit:lo_val] = True
    val = np.zeros(n, bool); val[lo_val:hi_val] = True
    print(f"\n{'=' * 78}\n{name}: {n:,} rows | fit {fit.sum():,} | stop {stop.sum():,} "
          f"| val {val.sum():,}\n{'=' * 78}")
    print(L.gap_profile(data).to_string(index=False))

    x = L.covariate_matrix(data, train_mask=fit, columns=COVARIATES)
    y = data["target_return_pct"].to_numpy(dtype=float)
    dt = data["dt"].to_numpy(dtype=float)

    xw_f, yw_f, _ = make_windows(x, y, fit, seq_len=args.seq_len)
    xw_s, yw_s, _ = make_windows(x, y, stop, seq_len=args.seq_len)
    xw_v, yw_v, idx_v = make_windows(x, y, val, seq_len=args.seq_len)
    dw_f, _, _ = make_windows(dt.reshape(-1, 1), y, fit, seq_len=args.seq_len)
    dw_s, _, _ = make_windows(dt.reshape(-1, 1), y, stop, seq_len=args.seq_len)
    dw_v, _, _ = make_windows(dt.reshape(-1, 1), y, val, seq_len=args.seq_len)
    dw_f, dw_s, dw_v = dw_f[..., 0], dw_s[..., 0], dw_v[..., 0]
    print(f"windows: fit {len(xw_f):,} | stop {len(xw_s):,} | val {len(xw_v):,}")

    # ---- S0 ----------------------------------------------------------------
    substrate, hist0, ep0 = train_substrate(
        xw_f, yw_f, hidden=args.hidden, epochs=args.epochs_substrate, batch=args.batch,
        lr=args.lr, seed=args.seed, stop_windows=(xw_s, yw_s),
    )
    print(f"  S0 selected epoch {ep0} (inner-stop CRPS {hist0['stop_crps'].min():.6f})")

    mu_v, sig_v, _ = jax.vmap(substrate)(jnp.asarray(xw_v))
    mu_v = np.asarray(mu_v)[:, -1]
    sig_v = np.asarray(sig_v)[:, -1]
    crps_v = np.asarray(crps_gaussian(jnp.asarray(mu_v), jnp.asarray(sig_v),
                                      jnp.asarray(yw_v)))

    # ---- S1: fixed sigma threshold calibrated on [0:63] ---------------------
    mu_f, sig_f, _ = jax.vmap(substrate)(jnp.asarray(xw_f))
    thr = float(np.quantile(np.asarray(sig_f)[:, -1], args.s1_target_coverage))
    s1_coverage = float((sig_v <= thr).mean())
    print(f"  S1 sigma threshold {thr:.6f} (fit-slice q{args.s1_target_coverage:.2f}) "
          f"-> arbiter coverage {s1_coverage:.3f}")

    # ---- S2: LIF readout on the FROZEN substrate ---------------------------
    readout, hist2, ep2 = train_readout(
        substrate, xw_f, yw_f, dw_f, hidden=args.hidden, epochs=args.epochs_readout,
        batch=args.batch, lr=args.lr, coverage_floor=args.coverage_floor, seed=args.seed,
        stop_windows=(xw_s, yw_s, dw_s),
    )
    print(f"  S2 selected epoch {ep2} (inner-stop AURC {hist2['stop_aurc'].min():.6f})")

    def lif_forward(xb, db):
        _, _, hs = jax.vmap(substrate)(xb)

        def one(h_seq, dt_seq):
            def st(state, inp):
                h_t, dt_t = inp
                state, s, v = readout(state, h_t, dt_t)
                return state, (s[0], v[0])

            _, (s, v) = jax.lax.scan(st, readout.initial_state(), (h_seq, dt_seq))
            return s, v

        return jax.vmap(one)(hs, db)

    spikes_v, memb_v = lif_forward(jnp.asarray(xw_v), jnp.asarray(dw_v))
    spikes_v = np.asarray(spikes_v)[:, -1]
    memb_v = np.asarray(memb_v)[:, -1]

    # Substrate identity check: S1 and S2 must sit on the SAME mu/sigma.
    aurc_s1 = L.aurc(crps_v, -sig_v)
    aurc_s2 = L.aurc(crps_v, memb_v)
    rng = np.random.default_rng(args.seed)
    aurc_rand = float(np.mean([L.aurc(crps_v, rng.normal(size=len(crps_v)))
                               for _ in range(200)]))

    boots = {}
    for bl in (args.block_len, 96, 168):
        boots[bl] = L.paired_block_bootstrap_daurc(
            crps_v, memb_v, crps_v, -sig_v,
            block_len=bl, n_boot=args.n_boot, seed=args.seed,
        )

    out = {
        "instrument": name,
        "n_rows": int(n),
        "n_val_windows": int(len(xw_v)),
        "S0_mean_crps": float(crps_v.mean()),
        "S0_selected_epoch": int(ep0),
        "S1_aurc": float(aurc_s1),
        "S1_sigma_threshold": thr,
        "S1_coverage": s1_coverage,
        "S2_aurc": float(aurc_s2),
        "S2_selected_epoch": int(ep2),
        "S2_coverage": float((spikes_v > 0.5).mean()),
        "random_aurc": aurc_rand,
        "spearman_sigma_absy": L.spearman(sig_v, np.abs(yw_v)),
        "corr_membrane_neg_sigma": float(np.corrcoef(memb_v, -sig_v)[0, 1]),
        "delta_aurc": float(aurc_s1 - aurc_s2),
        "block_len_sensitivity": {
            str(bl): {"delta_aurc": b["delta_aurc"], "ci_low": b["ci_low"],
                      "ci_high": b["ci_high"], "excludes_zero": bool(b["ci_low"] > 0)}
            for bl, b in boots.items()
        },
        "registered_block_len": int(args.block_len),
        "ci_low": float(boots[args.block_len]["ci_low"]),
        "ci_high": float(boots[args.block_len]["ci_high"]),
        "cleared_this_instrument": bool(boots[args.block_len]["ci_low"] > 0),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hist0.to_csv(OUTDIR / f"history_S0_{name}.csv", index=False)
    hist2.to_csv(OUTDIR / f"history_S2_{name}.csv", index=False)
    pd.DataFrame({"y": yw_v, "mu": mu_v, "sigma": sig_v, "crps": crps_v,
                  "membrane": memb_v, "spike": spikes_v}).to_csv(
        OUTDIR / f"validation_predictions_{name}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="H_spk.1 — does a trained LIF beat a sigma rule")
    ap.add_argument("--seq-len", type=int, default=PREREG["seq_len"])
    ap.add_argument("--hidden", type=int, default=PREREG["hidden"])
    ap.add_argument("--lr", type=float, default=PREREG["lr"])
    ap.add_argument("--batch", type=int, default=PREREG["batch"])
    ap.add_argument("--epochs-substrate", type=int, default=PREREG["epochs_substrate"])
    ap.add_argument("--epochs-readout", type=int, default=PREREG["epochs_readout"])
    ap.add_argument("--coverage-floor", type=float, default=PREREG["coverage_floor"])
    ap.add_argument("--s1-target-coverage", type=float, default=PREREG["s1_target_coverage"])
    ap.add_argument("--block-len", type=int, default=PREREG["block_len"])
    ap.add_argument("--n-boot", type=int, default=PREREG["n_boot"])
    ap.add_argument("--seed", type=int, default=PREREG["seed"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs_substrate = args.epochs_readout = 2
        args.n_boot = 200

    init_hypothesis_log()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in INSTRUMENTS.items():
        if args.dry_run:
            d = build_price_dataset(path).iloc[:3000]
            tmp = OUTDIR / f"_dry_{name}.csv"
            d.reset_index().rename(columns={"index": "time"}).to_csv(tmp, index=False)
            path = str(tmp)
        results[name] = score_instrument(name, path, args=args)

    cleared = all(r["cleared_this_instrument"] for r in results.values())
    summary = {
        "hypothesis": "H_spk.1_trained_lif_beats_fixed_sigma_threshold",
        "instruments": results,
        "cleared_bar": bool(cleared),
        "rule": "CLEARED only if BOTH instruments exclude zero in S2's favour",
    }
    (OUTDIR / "hypothesis_scores.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + json.dumps(summary, indent=2, default=str))
    print(f"\nwrote -> {OUTDIR.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
