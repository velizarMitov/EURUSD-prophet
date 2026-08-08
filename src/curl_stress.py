"""Discrete-Hodge curl ("network stress") on the FX currency graph — RESEARCH ONLY.

Scope
-----
4 currencies (EUR, USD, JPY, GBP) -> complete graph K4 -> 6 edges (pairs), 4 triangles.
Log rates form a 1-cochain ``w`` on the edges. No-arbitrage at simultaneity makes ``w``
EXACT: ``w = d(phi)`` for a per-currency scalar potential ``phi``, hence every triangle
curl ``(dw)(T) = 0``.

Therefore, with real (asynchronously sampled) bar data:

    observed_curl = d(observation noise)          <-- there is NO price signal in it

The observed curl is a PURE MEASUREMENT-ERROR CHANNEL. That is not a defeat: the
*amplitude* of that error is economically meaningful, because staleness error scales as
``sigma * sqrt(time since last tick)``. The curl is therefore a cross-sectional estimator
of (volatility x illiquidity) -- the same object the two-scale realized-volatility
literature (Zhang/Mykland/Ait-Sahalia) estimates from signature plots, but measured across
instruments at one instant instead of across time scales.

The whole point of this module is to normalise that amplitude by what pure asynchronicity
alone predicts, so that what remains ("excess curl") is not just a repackaging of volatility.

Key quantities
--------------
``triangle_curl``            raw curl of close log-prices (units: log points)
``staleness_null``           P_T = sum_e Var_bar(e) / n_ticks(e) -- predicted curl variance
                             under pure asynchronous-observation noise, parameter-free
``excess_curl``              X_T = c^2 / (a + b * P_T), calibrated on the TRAIN slice only.
                             E[X] = 1 under the null. X >> 1 = dislocation beyond timing.
``triangle_infeasibility``   one-sided, timing-robust violation certificate built from the
                             bar High/Low intervals: g_T > 0 means NO choice of prices
                             inside the observed ranges reconciles the triangle.
``stress_index``             log-excess curl averaged over the 3 INDEPENDENT triangles.

Cycle-space rank
----------------
K4 has E - V + 1 = 6 - 4 + 1 = 3 independent cycles. The 4 triangles are linearly
DEPENDENT. ``CYCLE_BASIS`` (the 3 triangles containing EUR) spans the space; the
(USD, JPY, GBP) triangle is a combination of them. Never invert a 4x4 covariance of
the 4 triangle curls -- it is singular by construction.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Graph definition
# --------------------------------------------------------------------------------------

CURRENCIES: tuple[str, ...] = ("EUR", "USD", "JPY", "GBP")

#: Storage convention: the pair is stored as (base, quote) == market convention.
PAIRS: tuple[tuple[str, str], ...] = (
    ("EUR", "USD"),
    ("EUR", "JPY"),
    ("EUR", "GBP"),
    ("GBP", "USD"),
    ("GBP", "JPY"),
    ("USD", "JPY"),
)

PAIR_LOOKUP: dict[frozenset[str], tuple[str, str]] = {frozenset(p): p for p in PAIRS}

#: All 4 triangles of K4 (ordered, so the orientation A->B->C->A is well defined).
TRIANGLES: tuple[tuple[str, str, str], ...] = tuple(
    itertools.combinations(CURRENCIES, 3)
)  # type: ignore[assignment]

#: A cycle basis: the 3 triangles containing EUR (spanning tree = star at EUR).
CYCLE_BASIS: tuple[tuple[str, str, str], ...] = tuple(t for t in TRIANGLES if "EUR" in t)

FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "ticks")


def symbol(base: str, quote: str) -> str:
    """Canonical column prefix, e.g. ``EURUSD``."""
    return f"{base}{quote}"


def triangle_edges(tri: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Oriented edges of the cycle A->B->C->A."""
    a, b, c = tri
    return ((a, b), (b, c), (c, a))


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


def align_bars(
    frames: Mapping[tuple[str, str], pd.DataFrame],
    *,
    tick_col: str = "tick_volume",
) -> pd.DataFrame:
    """Inner-join the 6 pair frames onto one strictly common timestamp index.

    THIS IS THE LOAD-BEARING STEP. Any bar where a single pair is missing (holiday,
    half-day, broker gap) must be DROPPED, not forward-filled: an ffill'd bar
    manufactures curl out of nothing and it will look like the most exciting result
    in the study.

    Each input frame must be indexed by timestamp (tz-aware, UTC) with columns
    ``open/high/low/close`` and a tick-count column.
    """
    missing = set(PAIR_LOOKUP) - {frozenset(k) for k in frames}
    if missing:
        raise ValueError(f"missing pairs: {sorted(tuple(sorted(m)) for m in missing)}")

    wide: dict[str, pd.Series] = {}
    index: pd.Index | None = None
    for (base, quote), df in frames.items():
        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{symbol(base, quote)}: index not sorted")
        if df.index.has_duplicates:
            raise ValueError(f"{symbol(base, quote)}: duplicate timestamps")
        index = df.index if index is None else index.intersection(df.index)

    assert index is not None
    for (base, quote), df in frames.items():
        sub = df.loc[index]
        sym = symbol(base, quote)
        for field in ("open", "high", "low", "close"):
            wide[f"{sym}_{field}"] = sub[field].astype(float)
        wide[f"{sym}_ticks"] = sub[tick_col].astype(float).clip(lower=1.0)

    out = pd.DataFrame(wide, index=index).sort_index()
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("non-finite values after alignment")
    return out


# --------------------------------------------------------------------------------------
# Signed log rates (orientation handling)
# --------------------------------------------------------------------------------------


def signed_log_price(df: pd.DataFrame, a: str, b: str, field: str = "close") -> np.ndarray:
    """log(A/B) taken from whichever direction the pair is actually stored in."""
    base, quote = PAIR_LOOKUP[frozenset((a, b))]
    lp = np.log(df[f"{symbol(base, quote)}_{field}"].to_numpy(dtype=float))
    return lp if (base, quote) == (a, b) else -lp


def signed_log_interval(
    df: pd.DataFrame, a: str, b: str, *, buffer_log: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi] bracket for log(A/B) from the bar High/Low, inverted if needed."""
    base, quote = PAIR_LOOKUP[frozenset((a, b))]
    sym = symbol(base, quote)
    lo = np.log(df[f"{sym}_low"].to_numpy(dtype=float))
    hi = np.log(df[f"{sym}_high"].to_numpy(dtype=float))
    if (base, quote) != (a, b):
        lo, hi = -hi, -lo
    return lo - buffer_log, hi + buffer_log


# --------------------------------------------------------------------------------------
# Curl
# --------------------------------------------------------------------------------------


def triangle_curl(df: pd.DataFrame, tri: Sequence[str], field: str = "close") -> np.ndarray:
    """(dw)(T) = log(A/B) + log(B/C) + log(C/A). Exactly 0 under simultaneity."""
    return sum(signed_log_price(df, x, y, field) for x, y in triangle_edges(tri))  # type: ignore[return-value]


def curl_frame(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Curl of all 4 triangles (3 independent)."""
    return pd.DataFrame(
        {"_".join(t): triangle_curl(df, t, field) for t in TRIANGLES}, index=df.index
    )


def triangle_infeasibility(
    df: pd.DataFrame, tri: Sequence[str], *, buffer_log: float = 0.0
) -> np.ndarray:
    """Timing-robust violation certificate from the High/Low intervals.

    Sum the per-edge log intervals; ``g`` is the distance from 0 to that sum.
    ``g == 0``  -> some assignment of prices inside the observed ranges satisfies
                   no-arbitrage exactly: the raw curl is fully explained by within-bar
                   timing. INCONCLUSIVE, and this is the overwhelmingly common case.
    ``g > 0``   -> no within-bar timing can reconcile the triangle.

    CAVEAT (do not skip): observed High/Low are extremes of the *sampled* path, so the
    observed range is a SUBSET of the true range. That biases ``g`` upward -> false
    positives, not false negatives. ``buffer_log`` (half-spread + a tick allowance,
    in log units) restores conservatism. Calibrate it on quiet periods so that the
    unconditional rate of ``g > 0`` is near zero.
    """
    lo_sum = np.zeros(len(df))
    hi_sum = np.zeros(len(df))
    for x, y in triangle_edges(tri):
        lo, hi = signed_log_interval(df, x, y, buffer_log=buffer_log)
        lo_sum += lo
        hi_sum += hi
    return np.maximum(lo_sum, 0.0) + np.maximum(-hi_sum, 0.0)


# --------------------------------------------------------------------------------------
# The asynchronicity null
# --------------------------------------------------------------------------------------


def parkinson_variance(
    df: pd.DataFrame, base: str, quote: str, *, window: int = 4
) -> np.ndarray:
    """Parkinson (1980) high-low variance estimator, ``ln(H/L)^2 / (4 ln 2)``.

    Used instead of a rolling variance of close-to-close returns because the null needs
    an *instantaneous* variance, and squared returns are a hopeless per-bar estimator.
    Validated in ``src/curl_null_simulation.py``: swapping a 96-bar rolling return
    variance for a 4-bar Parkinson estimate lifts the correlation between the feed-only
    proxy and the true expected curl variance from 0.29 to 0.94, and makes the binned
    calibration slope land on the theoretical 0.5 instead of collapsing in the tail.

    This is the load-bearing use of the High/Low columns.
    """
    sym = symbol(base, quote)
    hl = np.log(
        df[f"{sym}_high"].to_numpy(dtype=float) / df[f"{sym}_low"].to_numpy(dtype=float)
    )
    v = pd.Series(hl**2 / (4.0 * np.log(2.0)), index=df.index)
    if window > 1:
        v = v.rolling(window, min_periods=1).mean()
    return v.to_numpy(dtype=float)


def staleness_null(
    df: pd.DataFrame,
    tri: Sequence[str],
    *,
    window: int = 4,
    method: str = "parkinson",
    return_window: int = 96,
) -> np.ndarray:
    """Predicted curl VARIANCE under pure asynchronous-observation noise.

    Derivation: if the recorded Close of edge ``e`` is the last tick ``tau_e`` seconds
    before the bar boundary and ticks are Poisson with count ``n_e`` over a bar of length
    ``D``, then ``E[tau_e] = D / n_e`` (truncated at ``D``; see ``_truncation_factor``)
    and the staleness error has variance ``sigma_e^2 * D / n_e = Var_bar(e) / n_e``.
    Summing over the triangle's edges:

        P_T = sum_{e in T} Var_bar(e) / n_ticks(e) * truncation(n_ticks(e))

    Parameter-free: it needs only a per-bar variance and a tick count, both of which a
    retail MT5 feed provides. ``Var_bar`` defaults to the Parkinson High/Low estimator
    because the null needs an INSTANTANEOUS variance -- see ``parkinson_variance``.

    NOTE this is a SCALE model: the exact constant depends on the currency-level variance
    decomposition. Theory says the binned regression slope of ``c^2`` on ``P_T`` should be
    ~0.5 when tick rates across the triangle are similar (confirmed at 0.54 in simulation),
    which is why ``calibrate_null`` fits the constant rather than assuming it.

    Sharpest consequence, and the best day-one diagnostic you have:
    ``E[tau] = D / n`` is independent of bar size at a fixed tick RATE, so under this null
    **curl variance is FLAT across M1/M5/M15/H1** while return variance grows linearly.
    A curl variance that instead grows with bar size is NOT asynchronicity.
    """
    total = np.zeros(len(df))
    for x, y in triangle_edges(tri):
        base, quote = PAIR_LOOKUP[frozenset((x, y))]
        sym = symbol(base, quote)
        if method == "rv":
            # realized variance accumulated from finer sub-bars (see src/curl_mt5.py).
            # Strictly better than Parkinson when you hold M1 data and are working at
            # M5 or coarser: it is a direct, unbiased, instantaneous variance estimate.
            var = df[f"{sym}_rv"].to_numpy(dtype=float)
        elif method == "parkinson":
            var = parkinson_variance(df, base, quote, window=window)
        elif method == "return":  # kept only as an independent cross-check
            lp = np.log(df[f"{sym}_close"])
            var = (
                lp.diff()
                .rolling(return_window, min_periods=max(4, return_window // 2))
                .var()
                .to_numpy(dtype=float)
            )
        else:
            raise ValueError(f"unknown method: {method!r}")
        n = np.maximum(df[f"{sym}_ticks"].to_numpy(dtype=float), 1.0)
        total += var / n * _truncation_factor(n)
    return total


def _truncation_factor(n: np.ndarray) -> np.ndarray:
    """Correct ``E[tau] = D / n`` for the fact that staleness cannot exceed one bar.

    For Poisson ticks the backward recurrence time is exponential TRUNCATED at the bar
    length, so ``E[tau] = D/n - D/(e^n - 1)``, i.e. the naive ``D/n`` must be scaled by
    ``1 - n/(e^n - 1)``. The correction is negligible above ~10 ticks per bar and large
    below that -- which is precisely the illiquid-bar regime (holidays, the 22:00 UTC
    rollover, the Sunday open) where the uncorrected null over-predicts curl variance by
    an order of magnitude and therefore *hides* real stress exactly when it matters.
    """
    n = np.asarray(n, dtype=float)
    return np.where(n > 30.0, 1.0, 1.0 - n / np.expm1(np.minimum(n, 30.0)))


def null_calibration_table(
    curl: np.ndarray, null_pred: np.ndarray, *, mask: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bin bars into equal-count deciles of ``P_T`` and compare mean ``c^2`` per bin.

    THIS, not the per-bar regression, is how you check the null. Conditional on ``P``,
    ``c^2`` is approximately ``P * chi^2_1``, whose coefficient of variation is sqrt(2).
    That multiplicative noise attenuates ``corr(c^2, P)`` to roughly ``CV(P)/sqrt(2)`` --
    about 0.1 for a realistic feed -- **even when the null is exactly true**. Reading that
    0.1 as "the null model does not work" is the trap. Binning averages the chi-square away
    and the relationship appears cleanly.
    """
    y = np.asarray(curl, dtype=float) ** 2
    x = np.asarray(null_pred, dtype=float)
    ok = mask & np.isfinite(x) & np.isfinite(y)
    q = pd.qcut(pd.Series(x[ok]), n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"P": x[ok], "c2": y[ok], "bin": q})
    g = frame.groupby("bin", observed=True).agg(
        n=("c2", "size"), mean_P=("P", "mean"), mean_c2=("c2", "mean")
    )
    g["ratio"] = g["mean_c2"] / g["mean_P"]
    return g.reset_index()


def calibrate_null(
    curl: np.ndarray,
    null_pred: np.ndarray,
    *,
    train_mask: np.ndarray,
    n_bins: int = 20,
) -> tuple[float, float]:
    """Fit ``E[c^2] = a + b * P_T`` on the TRAIN slice only -> (intercept, slope).

    Fitted on equal-count BIN MEANS rather than raw bars. A raw per-bar OLS is dominated
    by the chi-square noise in ``c^2`` and returns a badly attenuated slope with the level
    dumped into the intercept; binning first recovers the true relationship (theory: the
    slope should land near 0.5 when tick rates across the triangle are similar).

    The intercept absorbs per-quote noise (bid-ask bounce, rounding) that does not scale
    with staleness. Fitting anywhere other than ``[0:70%]`` reintroduces exactly the
    look-ahead the project bans.
    """
    tab = null_calibration_table(curl, null_pred, mask=train_mask, n_bins=n_bins)
    if len(tab) < 3:
        raise ValueError("not enough distinct bins to calibrate the null")
    design = np.column_stack([np.ones(len(tab)), tab["mean_P"].to_numpy()])
    coef, *_ = np.linalg.lstsq(design, tab["mean_c2"].to_numpy(), rcond=None)
    return float(coef[0]), float(coef[1])


def convention_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-triangle mean curl vs its own standard error — a quote-convention test.

    A static level or convention mismatch between feeds (different rounding, a synthetic
    cross, a pair pulled from a different source) shows up as a curl with a LARGE NON-ZERO
    MEAN and comparatively small variance. Microstructure staleness is mean-zero. If
    ``t_stat`` is large, fix the data before interpreting anything: you are looking at
    plumbing, not at the market.
    """
    rows = []
    curls = curl_frame(df)
    for col in curls:
        c = curls[col].to_numpy(dtype=float)
        c = c[np.isfinite(c)]
        se = c.std(ddof=1) / np.sqrt(len(c))
        rows.append(
            {
                "triangle": col,
                "mean_curl_bp": c.mean() * 1e4,
                "rms_curl_bp": np.sqrt((c**2).mean()) * 1e4,
                "t_stat": c.mean() / se if se > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def excess_curl(
    curl: np.ndarray, null_pred: np.ndarray, intercept: float, slope: float
) -> np.ndarray:
    """X_T = c^2 / (a + b * P_T). Dimensionless; E[X] = 1 under the null."""
    denom = intercept + slope * np.asarray(null_pred, dtype=float)
    denom = np.where(denom > 0, denom, np.nan)
    return np.asarray(curl, dtype=float) ** 2 / denom


def causal_excess_curl(
    curl: np.ndarray,
    null_pred: np.ndarray,
    *,
    min_train: int = 5_000,
    refit_every: int = 1_000,
) -> np.ndarray:
    """Excess curl with the null refitted on an EXPANDING window of strictly past bars.

    A single static calibration drifts: volatility and liquidity regimes are persistent,
    so a fit on ``[0:70%]`` can be materially mis-scaled several years later. In
    simulation the out-of-sample mean of ``X`` under a *true* null wandered between 0.57
    and 1.26 across runs purely from regime drift -- i.e. a static fit will hand you
    apparent "stress" that is only a stale normalisation.

    Use this form for anything forward-looking (the paper-trading ledger). Use the static
    ``calibrate_null`` + ``excess_curl`` for the pre-registered validation-slice test,
    where a frozen fit on ``[0:70%]`` is exactly what the protocol demands.

    Rows before ``min_train`` are NaN by construction -- there is no past to fit on.
    """
    n = len(curl)
    out = np.full(n, np.nan)
    for start in range(min_train, n, refit_every):
        mask = np.zeros(n, dtype=bool)
        mask[:start] = True  # strictly past
        try:
            a, b = calibrate_null(curl, null_pred, train_mask=mask)
        except ValueError:
            continue
        stop = min(start + refit_every, n)
        out[start:stop] = excess_curl(curl, null_pred, a, b)[start:stop]
    return out


# --------------------------------------------------------------------------------------
# Stress index
# --------------------------------------------------------------------------------------


def stress_index(
    df: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    window: int = 4,
    smooth: int = 96,
    basis: Iterable[Sequence[str]] = CYCLE_BASIS,
) -> pd.DataFrame:
    """Per-bar network stress from the 3 INDEPENDENT triangles.

    Columns: ``curl_<T>``, ``null_<T>``, ``excess_<T>``, plus

    ``stress``       mean excess over the basis -- single-bar, and therefore chi-square
                     noisy (CV ~ sqrt(2/3)). Do NOT use this as a feature.
    ``stress_smooth`` log of the rolling mean of ``stress`` over ``smooth`` bars. THIS is
                     the usable quantity: the estimand is a conditional SCALE, and a scale
                     needs a window. At M15 with ``smooth=96`` that is one trading day.
    """
    out: dict[str, np.ndarray] = {}
    excesses: list[np.ndarray] = []
    for tri in basis:
        name = "_".join(tri)
        c = triangle_curl(df, tri)
        p = staleness_null(df, tri, window=window)
        a, b = calibrate_null(c, p, train_mask=train_mask)
        x = excess_curl(c, p, a, b)
        out[f"curl_{name}"] = c
        out[f"null_{name}"] = p
        out[f"excess_{name}"] = x
        excesses.append(x)
    stress = np.nanmean(np.column_stack(excesses), axis=1)
    out["stress"] = stress
    frame = pd.DataFrame(out, index=df.index)
    roll = frame["stress"].rolling(smooth, min_periods=max(4, smooth // 4)).mean()
    frame["stress_smooth"] = np.log(roll.where(roll > 0))
    return frame


def residualise_calendar(series: pd.Series) -> pd.Series:
    """Strip hour-of-day and day-of-week means (fit in-sample; see plan doc for the
    train-only variant).

    NON-OPTIONAL. Liquidity collapses at the 22:00 UTC rollover and through the Asian
    session, so raw curl has a large deterministic intraday shape. Skipping this step
    reproduces the exact failure documented in ``results/volatility_hypothesis_log.csv``
    #3, where an apparent volatility edge turned out to be a day-of-week lookup table.
    """
    idx = series.index
    frame = pd.DataFrame({"y": series.to_numpy(dtype=float)}, index=idx)
    frame["hour"] = idx.hour  # type: ignore[union-attr]
    frame["dow"] = idx.dayofweek  # type: ignore[union-attr]
    grand = frame["y"].mean()
    hour_mean = frame.groupby("hour")["y"].transform("mean")
    dow_mean = frame.groupby("dow")["y"].transform("mean")
    return pd.Series(frame["y"] - hour_mean - dow_mean + grand, index=idx, name="stress_resid")


# --------------------------------------------------------------------------------------
# Diagnostics (the anti-fooling suite)
# --------------------------------------------------------------------------------------


def frequency_scaling(frames_by_tf: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """RMS curl and RMS pair return per timeframe.

    Asynchronicity predicts RMS curl FLAT across timeframes and RMS return ~ sqrt(D),
    hence the ratio decaying as D^-0.5. Curl RMS that instead RISES with bar size is
    either a real persistent dislocation or a stale/frozen feed -- disambiguate with
    ``curl_autocorrelation``.
    """
    rows = []
    for tf, df in frames_by_tf.items():
        curls = curl_frame(df)
        rets = np.column_stack(
            [np.diff(signed_log_price(df, b, q), prepend=np.nan) for b, q in PAIRS]
        )
        rows.append(
            {
                "timeframe": tf,
                "n_bars": len(df),
                "rms_curl": float(np.sqrt(np.nanmean(curls.to_numpy() ** 2))),
                "rms_return": float(np.sqrt(np.nanmean(rets**2))),
                "ratio": float(
                    np.sqrt(np.nanmean(curls.to_numpy() ** 2))
                    / np.sqrt(np.nanmean(rets**2))
                ),
            }
        )
    return pd.DataFrame(rows)


def curl_autocorrelation(df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """ACF of each triangle's curl.

    Near-zero / slightly negative at lag 1 -> asynchronous sampling + quote bounce (fine).
    Persistent POSITIVE autocorrelation over many lags -> one feed is stale or frozen.
    That is a data bug, and it is the single most likely source of a spurious result.
    """
    curls = curl_frame(df)
    return pd.DataFrame(
        {col: [curls[col].autocorr(lag=k) for k in range(1, lags + 1)] for col in curls},
        index=pd.RangeIndex(1, lags + 1, name="lag"),
    )


def shift_placebo(df: pd.DataFrame, pair: tuple[str, str], shift: int = 1) -> float:
    """Deliberately misalign one pair by ``shift`` bars and report the RMS curl blow-up.

    Calibrates the metric's sensitivity: it tells you how many bars of misalignment would
    be needed to explain the curl you actually observe. If one bar of shift only doubles
    the curl, your feed alignment is not the dominant term.
    """
    base, quote = PAIR_LOOKUP[frozenset(pair)]
    sym = symbol(base, quote)
    shifted = df.copy()
    for field in FIELDS:
        shifted[f"{sym}_{field}"] = shifted[f"{sym}_{field}"].shift(shift)
    shifted = shifted.dropna()
    before = float(np.sqrt(np.nanmean(curl_frame(df).to_numpy() ** 2)))
    after = float(np.sqrt(np.nanmean(curl_frame(shifted).to_numpy() ** 2)))
    return after / before


def cycle_rank(df: pd.DataFrame) -> int:
    """Numerical rank of the 4 triangle curls. MUST be 3 for K4."""
    m = curl_frame(df).dropna().to_numpy()
    return int(np.linalg.matrix_rank(m - m.mean(axis=0), tol=1e-10))
