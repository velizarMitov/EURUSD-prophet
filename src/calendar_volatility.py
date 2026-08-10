"""Calendar-seasonality volatility model — the thing that kept winning, built properly.

Three times in this project a plain calendar table beat a trained network on the
volatility/uncertainty problem, and each time it appeared only as an inline benchmark that
embarrassed something more complicated:

  * ``results/volatility_hypothesis_log.csv`` #3 (2026-08-07) — GARCH(1,1) x a six-number
    day-of-week table, fitted on ``[0:80%]`` only, beat the 5-seed multi-task LSTM ensemble
    outright on the held-out block: MAE 0.20521 vs 0.21925, corr +0.4005 vs +0.3453,
    dMAE -0.01403 CI95 [-0.01773, -0.01073].
  * ``results/ltc/`` — a dow x hour table beat the LTC+LIF selective forecaster on EURUSD
    H1 (AURC 0.025753 vs 0.029627).
  * ``results/spiking_readout/`` — the same table beat the GRU+LIF model on AUDUSD H1
    (AURC 0.048469 vs 0.055328), fitted train-only.

It has never been implemented as a model. ``src/`` contains no seasonality module. This is
that module.

--------------------------------------------------------------------------------------
WHY IT IS A SERIOUS MODEL AND NOT A CONSOLATION PRIZE
--------------------------------------------------------------------------------------
* ~7 free parameters on top of GARCH's 3. It cannot meaningfully overfit.
* numpy + pandas only. No TensorFlow, no JAX, no sklearn. Deterministic, and it serves on
  a machine where none of the deep-learning stack is installed.
* Every parameter is readable. Six weekday multipliers are auditable by a human in a way
  no LSTM weight matrix is.
* It is the only object in this project that has beaten the incumbent neural ensemble with
  a confidence interval excluding zero.

WHAT IT IS NOT: a directional edge. It forecasts the SIZE of the next move, not the sign.
Its use is position sizing, risk limits and stop placement -- none of which need direction.

--------------------------------------------------------------------------------------
FITTING CONVENTION
--------------------------------------------------------------------------------------
Target ``target_volatility_pct = |next-day log return| * 100``, matching the volatility
family exactly. GARCH(1,1) by variance targeting, then a multiplicative scale, then
multiplicative weekday factors -- each stage fitted on ``[0:70%]`` only and each fitted to
minimise MAE (weighted median of ratios), because MAE is the family's registered metric and
a least-squares fit would optimise the wrong loss.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10


# ======================================================================================
# Fitting helpers
# ======================================================================================


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The exact minimiser of ``sum_i w_i * |v_i - f|``.

    Used instead of a mean ratio because the registered metric is MAE. Minimising
    ``sum |y_i - f * x_i| = sum x_i * |y_i/x_i - f|`` is a weighted-median problem in the
    ratios, weighted by ``x_i`` -- a least-squares scale would be optimising MSE and would
    be pulled around by the fat right tail of realised volatility.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return 1.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def fit_garch11(returns_pct: np.ndarray, train_mask: np.ndarray, n_grid: int = 24) -> dict:
    """GARCH(1,1) by variance targeting, fitted on train rows only.

    ``omega`` is pinned by ``var * (1 - alpha - beta)``, which removes a parameter and
    makes a grid plus local refinement sufficient. Deliberately dependency-free: the
    ``arch`` package would give a marginally tighter MLE and a heavier install for a model
    whose selling point is that it needs nothing.
    """
    r = np.asarray(returns_pct, dtype=float)
    tr = np.asarray(train_mask, dtype=bool)
    var_uncond = float(np.nanvar(r[tr]))
    if not np.isfinite(var_uncond) or var_uncond <= 0:
        raise ValueError("degenerate training variance")

    def nll(alpha: float, beta: float) -> float:
        if alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
            return np.inf
        omega = var_uncond * (1.0 - alpha - beta)
        h, total, cnt = var_uncond, 0.0, 0
        for i in range(len(r)):
            if tr[i] and np.isfinite(r[i]):
                total += math.log(h) + (r[i] ** 2) / h
                cnt += 1
            h = omega + alpha * (r[i] ** 2 if np.isfinite(r[i]) else 0.0) + beta * h
        return total / max(cnt, 1)

    best = (np.inf, 0.05, 0.90)
    for a in np.linspace(0.01, 0.25, n_grid):
        for b in np.linspace(0.60, 0.98, n_grid):
            v = nll(float(a), float(b))
            if v < best[0]:
                best = (v, float(a), float(b))
    _, a0, b0 = best
    for scale in (0.02, 0.005):
        for a in np.linspace(max(1e-4, a0 - scale), a0 + scale, 9):
            for b in np.linspace(max(1e-4, b0 - scale), b0 + scale, 9):
                v = nll(float(a), float(b))
                if v < best[0]:
                    best = (v, float(a), float(b))
        _, a0, b0 = best
    return {
        "omega": var_uncond * (1.0 - best[1] - best[2]),
        "alpha": best[1],
        "beta": best[2],
        "uncond_var": var_uncond,
    }


def garch_sigma_path(returns_pct: np.ndarray, par: dict) -> np.ndarray:
    """One-step-ahead conditional sigma. Entry ``t`` is the forecast made using data up to
    and including ``t``, i.e. the forecast for ``target_volatility_pct`` at row ``t``."""
    r = np.asarray(returns_pct, dtype=float)
    h = par["uncond_var"]
    out = np.empty(len(r))
    for i in range(len(r)):
        ri = r[i] if np.isfinite(r[i]) else 0.0
        h = par["omega"] + par["alpha"] * (ri**2) + par["beta"] * h
        out[i] = math.sqrt(max(h, 1e-12))
    return out


# ======================================================================================
# Model
# ======================================================================================


@dataclass
class CalendarVolatilityParams:
    omega: float
    alpha: float
    beta: float
    uncond_var: float
    scale: float
    dow_factors: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class CalendarVolatilityModel:
    """GARCH(1,1) x scale x day-of-week multiplier.

    ``use_dow=False`` gives the plain scaled GARCH, which is the ablation arm: it isolates
    exactly what the six calendar numbers are worth.
    """

    def __init__(self, *, use_dow: bool = True) -> None:
        self.use_dow = use_dow
        self.params: CalendarVolatilityParams | None = None

    def fit(
        self, returns_pct: np.ndarray, y: np.ndarray, dow: np.ndarray, train_mask: np.ndarray
    ) -> "CalendarVolatilityModel":
        par = fit_garch11(returns_pct, train_mask)
        sigma = garch_sigma_path(returns_pct, par)
        tr = np.asarray(train_mask, dtype=bool) & np.isfinite(y) & (sigma > 0)

        # global MAE-optimal scale (E|r| = sigma*sqrt(2/pi) only under normality; fit it)
        scale = weighted_median(y[tr] / sigma[tr], sigma[tr])

        factors: dict[str, float] = {}
        if self.use_dow:
            base = sigma * scale
            d = np.asarray(dow, dtype=int)
            for k in range(7):
                sel = tr & (d == k)
                factors[str(k)] = (
                    weighted_median(y[sel] / base[sel], base[sel]) if sel.sum() >= 30 else 1.0
                )
        self.params = CalendarVolatilityParams(
            omega=par["omega"], alpha=par["alpha"], beta=par["beta"],
            uncond_var=par["uncond_var"], scale=float(scale), dow_factors=factors,
        )
        return self

    def predict(self, returns_pct: np.ndarray, dow: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("fit first")
        p = self.params
        sigma = garch_sigma_path(
            returns_pct,
            {"omega": p.omega, "alpha": p.alpha, "beta": p.beta, "uncond_var": p.uncond_var},
        )
        pred = sigma * p.scale
        if self.use_dow and p.dow_factors:
            mult = np.array([p.dow_factors.get(str(int(k)), 1.0) for k in dow])
            pred = pred * mult
        return pred

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.params.to_json())  # type: ignore[union-attr]


# ======================================================================================
# Evaluation
# ======================================================================================


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "mae": float(np.abs(y - p).mean()),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan,
    }


def bootstrap_delta_mae(
    y: np.ndarray,
    challenger: np.ndarray,
    baseline: np.ndarray,
    *,
    block_len: int = 5,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20260810,
) -> dict[str, float]:
    """Moving-block bootstrap on ``MAE(baseline) - MAE(challenger)``; positive favours the
    challenger. Block length 5 matches the volatility family's registered convention."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_blocks = max(1, n // block_len)
    d = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
        idx = idx[idx < n]
        d[i] = np.abs(y[idx] - baseline[idx]).mean() - np.abs(y[idx] - challenger[idx]).mean()
    lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
    point = float(np.abs(y - baseline).mean() - np.abs(y - challenger).mean())
    return {
        "delta_mae": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def chronological_masks(
    n: int, *, train_fraction: float = TRAIN_FRACTION, val_fraction: float = VAL_FRACTION
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = int(train_fraction * n)
    hi = int((train_fraction + val_fraction) * n)
    tr = np.zeros(n, dtype=bool); tr[:lo] = True
    va = np.zeros(n, dtype=bool); va[lo:hi] = True
    te = np.zeros(n, dtype=bool); te[hi:] = True
    return tr, va, te


def build_daily_dataset(
    features_csv: str = "results/eurusd_features.csv", *, euro_era_start: str = "1999-01-04"
) -> pd.DataFrame:
    """Daily close -> log return (%) and ``target_volatility_pct``.

    The euro-era truncation mirrors what the macro merge does inside
    ``src/volatility.py::build_volatility_matrix``; the pre-1999 rows in the features file
    are synthetic-history padding and are not modelled by any family.
    """
    d = pd.read_csv(features_csv)
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).set_index("time").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d[d.index >= pd.Timestamp(euro_era_start, tz="UTC")]
    close = d["close"].to_numpy(dtype=float)
    lr = np.concatenate([[np.nan], np.diff(np.log(close))]) * 100.0
    d["log_return_pct"] = lr
    d["target_volatility_pct"] = np.abs(np.concatenate([lr[1:], [np.nan]]))
    d["dow"] = d.index.dayofweek
    return d.dropna(subset=["log_return_pct", "target_volatility_pct"]).copy()
