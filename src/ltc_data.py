"""Idea 3 — data pipeline, benchmark and selective-prediction metrics. RESEARCH ONLY.

Deliberately JAX-free. Everything in this module is pure numpy/pandas so it can be tested
and run without a GPU, without JAX, and without the model. `src/ltc_experiment.py` holds
the JAX training loop and imports from here.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.

--------------------------------------------------------------------------------------
THE TARGET AND THE BENCHMARK MUST MATCH, OR AURC MEANS NOTHING
--------------------------------------------------------------------------------------
AURC needs a per-row loss. That fixes the target, and the target fixes which benchmark is
legitimate:

  target     next-bar log return in PERCENT (same convention as src/features.py --
             the ``* 100`` lives in exactly one place project-wide)
  loss       CRPS of the predictive distribution against the realised return
  benchmark  N(0, sigma_hat) where sigma_hat = GARCH(1,1) x day-of-week multiplier,
             both fitted on [0:70%] only

The zero mean in the benchmark is not laziness. The conditional mean of daily EUR/USD
return is indistinguishable from zero (ARCHITECTURE_DOCS 4.2.1), so N(0, sigma) is a
*strong* distributional benchmark, and the day-of-week multiplier is there because
``results/volatility_hypothesis_log.csv`` #3 showed a six-number weekday table beat our own
neural volatility ensemble outright. That table is the thing to beat.

A GARCH model does not abstain, so its risk-coverage curve is built by ranking rows by its
OWN predicted sigma (most confident = smallest sigma). This must be pre-declared, and it is:
it is the only ranking available to a model without a selective head, and using anything
else would hand the benchmark information the LTC does not get.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10

_ERF = np.vectorize(math.erf)


# ======================================================================================
# Loading and Delta-t
# ======================================================================================


def load_bars(path: str | Path, *, tick_col: str = "tick_volume") -> pd.DataFrame:
    """Load a single-instrument OHLCV CSV into a clean, strictly increasing UTC frame."""
    df = pd.read_csv(path)
    time_col = "time" if "time" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[time_col], utc=True)
    df = df.drop(columns=[time_col]).set_index(idx).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if tick_col not in df.columns:
        raise ValueError(
            f"{path}: no '{tick_col}'. The clock hypothesis needs it -- it is the "
            "observable that tau_eff is supposed to reproduce."
        )
    return df


def add_delta_t(df: pd.DataFrame, *, unit_hours: float | None = None) -> pd.DataFrame:
    """Explicit calendar Delta-t per row, in units of the MEDIAN bar spacing.

    Why median-normalised rather than raw hours: the cell composes ``exp(-dt / tau)``, so
    the numerical scale of ``dt`` sets the scale ``tau`` must learn. Feeding raw hours at
    H1 makes a weekend ``dt = 65`` while a normal bar is ``1``; feeding raw seconds makes
    the same ratio with numbers 3600x larger and the optimiser has to discover that.
    Normalising to median spacing puts a normal bar at 1.0 and a weekend at ~65, which is
    the ratio that actually matters and is scale-free across timeframes.

    The weekend is left LARGE on purpose. Compressing it is ``BusinessTimeWarp``'s job --
    that is the learned subordinator and the entire scientific claim of Idea 3. Clipping
    ``dt`` here would silently do the model's job for it and make the result unfalsifiable.
    """
    out = df.copy()
    delta = out.index.to_series().diff().dt.total_seconds() / 3600.0
    med = float(delta.median()) if unit_hours is None else unit_hours
    if not np.isfinite(med) or med <= 0:
        raise ValueError("could not infer a positive median bar spacing")
    out["dt_hours"] = delta.to_numpy(dtype=float)
    out["dt"] = out["dt_hours"] / med
    out.loc[out.index[0], ["dt_hours", "dt"]] = [med, 1.0]
    out.attrs["median_spacing_hours"] = med
    return out


def gap_profile(df: pd.DataFrame) -> pd.DataFrame:
    """How bad is the irregular sampling? Run this before trusting any Delta-t handling."""
    dt = df["dt"].to_numpy(dtype=float)
    q = np.nanquantile(dt, [0.5, 0.9, 0.99, 0.999, 1.0])
    return pd.DataFrame(
        [
            {
                "median_spacing_hours": df.attrs.get("median_spacing_hours", np.nan),
                "dt_p50": q[0],
                "dt_p90": q[1],
                "dt_p99": q[2],
                "dt_p999": q[3],
                "dt_max": q[4],
                "share_dt_gt_4": float((dt > 4).mean()),
                "share_dt_gt_40": float((dt > 40).mean()),
            }
        ]
    )


# ======================================================================================
# Dataset
# ======================================================================================


def build_dataset(df: pd.DataFrame, *, tick_col: str = "tick_volume") -> pd.DataFrame:
    """Covariates, target and clock observable, with no look-ahead.

    Target is ``shift(-1)`` of the percent log return, matching the project convention.
    Every covariate is known at bar close; the final row has no target and is dropped.
    """
    out = add_delta_t(df)
    close = out["close"].to_numpy(dtype=float)
    logret = np.concatenate([[np.nan], np.diff(np.log(close))]) * 100.0
    out["log_return_pct"] = logret

    hl = np.log(out["high"].to_numpy(dtype=float) / out["low"].to_numpy(dtype=float))
    out["parkinson_pct"] = np.sqrt(np.maximum(hl, 0) ** 2 / (4 * np.log(2))) * 100.0
    out["abs_return_pct"] = np.abs(logret)
    out["ticks"] = out[tick_col].astype(float).clip(lower=1.0)
    out["log_ticks"] = np.log(out["ticks"].to_numpy(dtype=float))
    # activity per unit of elapsed time -- the natural "information rate" observable
    out["tick_rate"] = out["ticks"].to_numpy(dtype=float) / out["dt"].to_numpy(dtype=float)
    out["log_tick_rate"] = np.log(np.maximum(out["tick_rate"].to_numpy(dtype=float), 1e-9))
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek

    out["target_return_pct"] = np.concatenate([logret[1:], [np.nan]])
    return out.dropna(subset=["log_return_pct", "target_return_pct", "dt"]).copy()


COVARIATES: tuple[str, ...] = (
    "log_return_pct",
    "abs_return_pct",
    "parkinson_pct",
    "log_ticks",
    "log_tick_rate",
)


def covariate_matrix(
    data: pd.DataFrame, *, train_mask: np.ndarray, columns: Iterable[str] = COVARIATES
) -> np.ndarray:
    """Standardised covariates, with mean/sd fitted on TRAIN rows only."""
    cols = list(columns)
    x = data[cols].to_numpy(dtype=float)
    mu = x[train_mask].mean(axis=0)
    sd = x[train_mask].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd


def chronological_masks(
    n: int, *, train_fraction: float = TRAIN_FRACTION, val_fraction: float = VAL_FRACTION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, validation, test) as the project defines them: 70 / 10 / 20, in time order.

    The test block is SPENT for search. It is returned only so a final one-shot report can
    index it, never for model selection.
    """
    lo = int(train_fraction * n)
    hi = int((train_fraction + val_fraction) * n)
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[:lo] = True
    val[lo:hi] = True
    test[hi:] = True
    return train, val, test


# ======================================================================================
# Benchmark: GARCH(1,1) x day-of-week
# ======================================================================================


def fit_garch11(
    y: np.ndarray, *, train_mask: np.ndarray, n_grid: int = 24
) -> dict[str, float]:
    """GARCH(1,1) with variance targeting, fitted on TRAIN rows only.

    ``omega`` is pinned by variance targeting (``omega = var * (1 - alpha - beta)``), which
    removes one parameter and makes a coarse 2-D grid plus local refinement sufficient. A
    full MLE would be marginally tighter and would add a scipy dependency to what is meant
    to be a hard-to-beat, easy-to-audit baseline.
    """
    r = np.asarray(y, dtype=float)
    tr = np.asarray(train_mask, dtype=bool)
    var_uncond = float(np.nanvar(r[tr]))
    if not np.isfinite(var_uncond) or var_uncond <= 0:
        raise ValueError("degenerate training variance")

    def nll(alpha: float, beta: float) -> float:
        if alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
            return np.inf
        omega = var_uncond * (1.0 - alpha - beta)
        h = var_uncond
        total = 0.0
        cnt = 0
        for i in range(len(r)):
            if tr[i]:
                total += math.log(h) + (r[i] ** 2) / h
                cnt += 1
            h = omega + alpha * (r[i] ** 2) + beta * h
        return total / max(cnt, 1)

    best = (np.inf, 0.05, 0.90)
    for a in np.linspace(0.01, 0.25, n_grid):
        for b in np.linspace(0.60, 0.98, n_grid):
            v = nll(float(a), float(b))
            if v < best[0]:
                best = (v, float(a), float(b))
    _, a0, b0 = best
    for scale in (0.02, 0.005):
        for a in np.linspace(max(1e-3, a0 - scale), a0 + scale, 9):
            for b in np.linspace(max(1e-3, b0 - scale), b0 + scale, 9):
                v = nll(float(a), float(b))
                if v < best[0]:
                    best = (v, float(a), float(b))
        _, a0, b0 = best
    return {
        "omega": var_uncond * (1.0 - best[1] - best[2]),
        "alpha": best[1],
        "beta": best[2],
        "uncond_var": var_uncond,
        "train_nll": best[0],
    }


def garch_dow_sigma(
    y: np.ndarray, dow: np.ndarray, *, train_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """One-step-ahead sigma from GARCH(1,1), multiplied by a day-of-week factor.

    Both components are fitted on TRAIN rows only. The weekday factor is the ratio of mean
    |residual| within each weekday to the overall mean, which is the six-number table that
    beat our own volatility ensemble in ``results/volatility_hypothesis_log.csv`` #3.
    """
    r = np.asarray(y, dtype=float)
    par = fit_garch11(r, train_mask=train_mask)
    h = par["uncond_var"]
    sig = np.empty(len(r))
    for i in range(len(r)):
        sig[i] = math.sqrt(max(h, 1e-12))  # forecast for bar i, made at i-1
        h = par["omega"] + par["alpha"] * (r[i] ** 2) + par["beta"] * h

    z = np.abs(r) / np.maximum(sig, 1e-12)
    tr = np.asarray(train_mask, dtype=bool)
    grand = float(np.nanmean(z[tr]))
    factors: dict[int, float] = {}
    d = np.asarray(dow, dtype=int)
    for k in range(7):
        sel = tr & (d == k)
        factors[k] = float(np.nanmean(z[sel]) / grand) if sel.sum() > 30 else 1.0
    mult = np.array([factors.get(int(k), 1.0) for k in d])
    par.update({f"dow_{k}": v for k, v in factors.items()})
    return sig * mult, par


# ======================================================================================
# Selective-prediction metrics
# ======================================================================================


def crps_gaussian_np(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (lower is better)."""
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / s
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1.0 + _ERF(z / math.sqrt(2.0)))
    return s * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def risk_coverage(
    losses: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep coverage from most- to least-confident. Returns (coverage, cumulative risk)."""
    loss = np.asarray(losses, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    ok = np.isfinite(loss) & np.isfinite(conf)
    loss, conf = loss[ok], conf[ok]
    order = np.argsort(-conf, kind="mergesort")
    ordered = loss[order]
    risk = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    coverage = np.arange(1, len(ordered) + 1) / len(ordered)
    return coverage, risk


def aurc(losses: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better.

    Comparable across models even when they abstain on different rows, because each model
    is swept over its OWN confidence ranking -- which is exactly why AURC is the right
    primary for a selective predictor and a single CRPS number is not.
    """
    coverage, risk = risk_coverage(losses, confidence)
    if len(coverage) < 2:
        return float("nan")
    return float(np.trapezoid(risk, coverage))


def paired_block_bootstrap_daurc(
    loss_a: np.ndarray,
    conf_a: np.ndarray,
    loss_b: np.ndarray,
    conf_b: np.ndarray,
    *,
    block_len: int = 24,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260810,
) -> dict[str, float]:
    """CI for AURC(b) - AURC(a) by moving-block bootstrap, resampling the same blocks for
    both models so the comparison stays paired. Positive delta = model A is better."""
    rng = np.random.default_rng(seed)
    n = len(loss_a)
    n_blocks = max(1, n // block_len)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
        idx = idx[idx < n]
        deltas[i] = aurc(loss_b[idx], conf_b[idx]) - aurc(loss_a[idx], conf_a[idx])
    lo, hi = np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2])
    point = aurc(loss_b, conf_b) - aurc(loss_a, conf_a)
    return {
        "delta_aurc": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy."""
    x = pd.Series(np.asarray(a, dtype=float))
    y = pd.Series(np.asarray(b, dtype=float))
    ok = x.notna() & y.notna()
    return float(x[ok].rank().corr(y[ok].rank()))


def partial_spearman(
    a: np.ndarray, b: np.ndarray, controls: np.ndarray
) -> float:
    """Rank correlation of ``a`` and ``b`` after linearly removing ``controls`` from both,
    on ranks.

    REQUIRED for the clock hypothesis, not optional. Measured on the real H1 validation
    slice, ``Spearman(log tick rate, |return|) = +0.4966`` -- activity and volatility are
    heavily entangled. So a raw correlation between ``tau_eff`` and the tick rate cannot
    distinguish "the cell learned an information clock" from "the cell learned volatility",
    and volatility is the thing every model here already tracks. Only the component of the
    tick rate that volatility does NOT explain can support the subordination claim.
    """
    x = pd.Series(np.asarray(a, dtype=float)).rank()
    y = pd.Series(np.asarray(b, dtype=float)).rank()
    c = np.asarray(controls, dtype=float)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    cr = np.column_stack([np.ones(len(c))] + [pd.Series(col).rank().to_numpy() for col in c.T])
    ok = x.notna() & y.notna() & np.isfinite(cr).all(axis=1)
    xr, yr, cc = x[ok].to_numpy(), y[ok].to_numpy(), cr[ok.to_numpy()]
    bx, *_ = np.linalg.lstsq(cc, xr, rcond=None)
    by, *_ = np.linalg.lstsq(cc, yr, rcond=None)
    rx, ry = xr - cc @ bx, yr - cc @ by
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ======================================================================================
# Hypothesis registry
# ======================================================================================

LTC_LOG = "results/ltc_hypothesis_log.csv"

LTC_LOG_COLUMNS: tuple[str, ...] = (
    "n",
    "date",
    "hypothesis",
    "arbiter",
    "primary_quantity",
    "point_estimate",
    "ci_low",
    "ci_high",
    "benchmark",
    "alpha_bonferroni",
    "cleared_bar",
    "verdict",
    "notes",
)


def init_hypothesis_log(path: str | Path = LTC_LOG, *, family_size: int = 2) -> Path:
    """Create the family registry with its pre-registration rows, PENDING.

    Two hypotheses are registered, not one, because two distinct claims are being made and
    both would be reported if they succeeded:

      3.1  forecast skill    AURC vs the GARCH(1,1) x day-of-week benchmark
      3.2  the clock claim   Spearman(tau_eff, realised tick rate)

    They are independent claims, so the family bar is ``alpha = 0.05 / 2 = 0.025`` for each.
    Registering 3.2 as a mere "secondary" of 3.1 would let us claim the scientifically
    interesting result at 0.05 while only ever having paid for one test -- and 3.2 is the
    result most worth having, since it is falsifiable without reference to P&L.

    Rows are written with ``verdict = PENDING`` BEFORE anything is scored. That is the
    point of pre-registration.
    """
    p = Path(path)
    if p.exists():
        return p
    alpha = 0.05 / family_size
    rows = [
        {
            "n": 1,
            "date": "",
            "hypothesis": "H_ltc.1_selective_cfc_lif_vs_garch_dow",
            "arbiter": "validation[70:80]",
            "primary_quantity": "delta_AURC (benchmark - model); positive favours the model",
            "point_estimate": "",
            "ci_low": "",
            "ci_high": "",
            "benchmark": "N(0, sigma) with sigma = GARCH(1,1) x day-of-week, fit on [0:70]",
            "alpha_bonferroni": alpha,
            "cleared_bar": "",
            "verdict": "PENDING",
            "notes": (
                "Target = next-bar log return in percent. Loss = CRPS. Model ranks by its "
                "own LIF membrane potential; the benchmark ranks by its own predicted "
                "sigma (smallest = most confident) -- pre-declared, and the only ranking a "
                "non-selective model has. Paired moving-block bootstrap, block_len=24, "
                "n=2000. CPU-only JAX."
            ),
        },
        {
            "n": 2,
            "date": "",
            "hypothesis": "H_ltc.2_learned_clock_tracks_tick_rate",
            "arbiter": "validation[70:80]",
            "primary_quantity": (
                "PARTIAL Spearman(tau_eff, log tick rate | abs_return_pct, parkinson_pct)"
            ),
            "point_estimate": "",
            "ci_low": "",
            "ci_high": "",
            "benchmark": "zero correlation (moving-block bootstrap CI must exclude 0)",
            "alpha_bonferroni": alpha,
            "cleared_bar": "",
            "verdict": "PENDING",
            "notes": (
                "The subordination claim: if the cell learned business time, its "
                "input-dependent relaxation rate should track realised information "
                "arrival. Sign is pre-declared NEGATIVE -- more ticks per unit calendar "
                "time means a faster clock, hence a SHORTER tau_eff. A positive "
                "correlation of any size is a DROP, not a discovery. "
                "CONFOUND CONTROL IS MANDATORY: on the real H1 validation slice "
                "Spearman(log tick rate, |return|) = +0.4966, so a raw correlation cannot "
                "separate 'learned an information clock' from 'learned volatility'. The "
                "primary is therefore the PARTIAL rank correlation controlling for "
                "abs_return_pct and parkinson_pct. The raw correlation is reported "
                "alongside it but does not gate."
            ),
        },
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(LTC_LOG_COLUMNS)).to_csv(p, index=False)
    return p
