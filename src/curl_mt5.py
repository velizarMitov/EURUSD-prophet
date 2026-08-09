"""Real-broker-data runner for the Idea-2 curl measurement — RESEARCH ONLY.

Takes six MT5 OHLCV frames (EUR/USD/JPY/GBP subgraph), and provides the three day-one
components:

  1. ``convention_report``            -- plumbing check: inversion, quote-side, level bugs
  2. ``frequency_scaling_from_base``  -- the sharpest discriminator, M1 -> M5 -> M15 -> H1
  3. ``calculate_causal_excess_curl`` -- the index, causally calibrated, decile-binned

``run_day1`` chains all of it and writes to ``results/curl/``.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path.

--------------------------------------------------------------------------------------
THREE THINGS ABOUT REAL MT5 DATA THAT THE SYNTHETIC STUDY COULD NOT TELL YOU
--------------------------------------------------------------------------------------
1. **MT5 bars are BID, not mid.** log(bid) = log(mid) - halfspread/mid, so the curl of a
   bid triangle carries a systematic offset of roughly the signed sum of the three
   relative half-spreads -- typically a few tenths of a bp on majors at M1, and larger on
   GBPJPY. This is EXPECTED, not a bug. It shows up as a constant, and ``convention_report``
   compares the observed mean curl against the offset predicted from the ``spread`` column
   so you can tell "bid convention" apart from "inverted a pair".

2. **You need ``tick_volume``.** ``mt5.copy_rates_range`` returns
   ``time, open, high, low, close, tick_volume, spread, real_volume``. If you slice down to
   OHLC you throw away the staleness estimator and the null loses its liquidity channel
   entirely -- it degenerates to "curl variance is proportional to volatility", which
   controls for the confound you already knew about and none of the one you are testing.
   Keep ``tick_volume`` and ``spread``.

3. **Spread is the control you must beat.** Curl-from-bid-quotes rises mechanically when
   spreads widen, and spreads widen in stress. So an excess-curl index that predicts
   anything will trivially be suspected of being a repackaged spread. ``run_day1``
   therefore reports the correlation of the stress index with the summed spread, and the
   predictive test must include spread as a control regressor. If excess curl adds nothing
   over spread, it is not a new observable.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src import curl_stress as cs

OHLC = ("open", "high", "low", "close")

#: pandas resample rules, ordered coarser
TIMEFRAMES: dict[str, int] = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}
RESAMPLE_RULE: dict[str, str] = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1h"}


# ======================================================================================
# 0. Ingestion
# ======================================================================================


def prepare_frames(
    raw: Mapping[tuple[str, str], pd.DataFrame],
    *,
    ticks_method: str = "auto",
    assume_utc: bool = True,
) -> pd.DataFrame:
    """Normalise six broker frames and align them onto one strictly common index.

    ``raw`` maps ``(base, quote)`` -> DataFrame with a DatetimeIndex and at least
    ``open/high/low/close``. ``tick_volume`` and ``spread`` are used when present.

    ``ticks_method``:
      ``"auto"``      use ``tick_volume`` if present, else fall back to ``"constant"``
      ``"constant"``  n_ticks := 1 for every bar. The staleness channel is then DEAD and
                      the null reduces to a pure volatility control. Emits a warning.
    """
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for pair, df in raw.items():
        key = cs.PAIR_LOOKUP.get(frozenset(pair))
        if key is None:
            raise ValueError(f"{pair} is not an edge of the {cs.CURRENCIES} subgraph")
        d = df.copy()
        d.columns = [str(c).lower() for c in d.columns]
        missing = [c for c in OHLC if c not in d.columns]
        if missing:
            raise ValueError(f"{cs.symbol(*key)}: missing columns {missing}")

        if not isinstance(d.index, pd.DatetimeIndex):
            raise TypeError(f"{cs.symbol(*key)}: index must be a DatetimeIndex")
        if d.index.tz is None:
            if not assume_utc:
                raise ValueError(f"{cs.symbol(*key)}: tz-naive index")
            d.index = d.index.tz_localize("UTC")
        else:
            d.index = d.index.tz_convert("UTC")
        d = d[~d.index.duplicated(keep="last")].sort_index()

        if "tick_volume" not in d.columns:
            if ticks_method == "auto":
                warnings.warn(
                    f"{cs.symbol(*key)}: no tick_volume -- falling back to a constant. "
                    "The null loses its liquidity channel; re-pull with "
                    "mt5.copy_rates_range which returns tick_volume and spread.",
                    stacklevel=2,
                )
            d["tick_volume"] = 1.0
        frames[key] = d

    aligned = cs.align_bars(frames)

    # carry spread through if every pair has it (used by convention_report)
    if all("spread" in df.columns for df in frames.values()):
        for (base, quote), df in frames.items():
            aligned[f"{cs.symbol(base, quote)}_spread"] = (
                df.loc[aligned.index, "spread"].astype(float).to_numpy()
            )
    return aligned


def resample_aligned(base: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate an aligned M1 frame to a coarser timeframe.

    Aggregating AFTER alignment guarantees every pair uses identical bar boundaries --
    resampling each pair separately and joining afterwards is how you manufacture curl.

    Also produces ``<SYM>_rv``: realized variance accumulated from the M1 sub-bar close
    returns. That is a direct instantaneous variance estimate and is strictly better than
    Parkinson at M5 and coarser (Parkinson's discrete-sampling bias is worst exactly where
    ticks are sparse). At M1 there are no sub-bars, so Parkinson remains the estimator.
    """
    if timeframe not in RESAMPLE_RULE:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    rule = RESAMPLE_RULE[timeframe]
    out: dict[str, pd.Series] = {}
    grouper = base.resample(rule, label="right", closed="right")
    for b, q in cs.PAIRS:
        sym = cs.symbol(b, q)
        out[f"{sym}_open"] = grouper[f"{sym}_open"].first()
        out[f"{sym}_high"] = grouper[f"{sym}_high"].max()
        out[f"{sym}_low"] = grouper[f"{sym}_low"].min()
        out[f"{sym}_close"] = grouper[f"{sym}_close"].last()
        out[f"{sym}_ticks"] = grouper[f"{sym}_ticks"].sum()
        r2 = np.log(base[f"{sym}_close"]).diff() ** 2
        out[f"{sym}_rv"] = r2.resample(rule, label="right", closed="right").sum()
        if f"{sym}_spread" in base.columns:
            out[f"{sym}_spread"] = grouper[f"{sym}_spread"].mean()
    frame = pd.DataFrame(out).dropna()
    return frame[frame[[f"{cs.symbol(*p)}_ticks" for p in cs.PAIRS]].gt(0).all(axis=1)]


# ======================================================================================
# 1. convention_report
# ======================================================================================


def convention_report(df: pd.DataFrame, *, train_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Plumbing check. Run this FIRST and do not read anything else until it is clean.

    For each of the 4 triangles:

    ``mean_curl_bp``     a constant offset. Inversion or level bugs live here.
    ``rms_curl_bp``      total scale.
    ``mean_over_rms``    THE diagnostic. Staleness noise is mean-zero, so this should be
                         small (< ~0.2). Near 1 means the curl is essentially a constant:
                         a convention bug (synthetic study: 30.5 bp constant, i.e. 30x the
                         genuine 0.9 bp timing noise) or the bid-side offset below.
    ``t_stat``           mean / s.e. Reported for completeness, but with n in the hundreds
                         of thousands even a trivial bias is "significant" -- judge
                         ``mean_curl_bp`` against a pip, not against a p-value.
    ``pred_bid_bp``      offset predicted from the ``spread`` columns if present. If
                         ``mean_curl_bp`` matches this, the constant is the BID CONVENTION
                         and is expected. If it does not, you have inverted a pair.
    """
    curls = cs.curl_frame(df)
    mask = np.ones(len(df), dtype=bool) if train_mask is None else train_mask

    pred: dict[str, float] = {}
    if all(f"{cs.symbol(*p)}_spread" in df.columns for p in cs.PAIRS):
        for tri in cs.TRIANGLES:
            total = np.zeros(len(df))
            for x, y in cs.triangle_edges(tri):
                b, q = cs.PAIR_LOOKUP[frozenset((x, y))]
                sym = cs.symbol(b, q)
                # spread is in PRICE units here; half-spread in log terms
                rel = df[f"{sym}_spread"].to_numpy(float) / (
                    2.0 * df[f"{sym}_close"].to_numpy(float)
                )
                total += rel if (b, q) == (x, y) else -rel
            # bid = mid - half-spread, so the bid-side curl offset is -sum(signed rel)
            pred["_".join(tri)] = float(-np.nanmean(total[mask]) * 1e4)

    rows = []
    for col in curls:
        c = curls[col].to_numpy(dtype=float)[mask]
        c = c[np.isfinite(c)]
        mean, rms = c.mean(), np.sqrt((c**2).mean())
        se = c.std(ddof=1) / np.sqrt(len(c))
        rows.append(
            {
                "triangle": col,
                "n": len(c),
                "mean_curl_bp": mean * 1e4,
                "rms_curl_bp": rms * 1e4,
                "mean_over_rms": abs(mean) / rms if rms > 0 else np.nan,
                "t_stat": mean / se if se > 0 else np.nan,
                "pred_bid_bp": pred.get(col, np.nan),
            }
        )
    out = pd.DataFrame(rows)
    # What matters is the offset that the bid convention does NOT explain. On real MT5
    # data every triangle carries a bid-side constant; that is expected and harmless
    # (demean_curl removes it). An offset that survives the spread prediction is not.
    out["resid_bp"] = out["mean_curl_bp"] - out["pred_bid_bp"].fillna(0.0)
    resid_over_rms = (out["resid_bp"].abs() / out["rms_curl_bp"]).to_numpy()
    out["verdict"] = np.where(
        resid_over_rms > 0.5,
        "UNEXPLAINED CONSTANT — check for an inverted pair before proceeding",
        np.where(
            resid_over_rms > 0.15,
            "residual offset — investigate (rounding? mixed sources?)",
            "clean (any constant is explained by the bid convention)",
        ),
    )
    return out


def demean_curl(curl: np.ndarray, *, train_mask: np.ndarray) -> np.ndarray:
    """Remove the constant (bid-side / convention) offset using TRAIN rows only."""
    c = np.asarray(curl, dtype=float)
    ok = train_mask & np.isfinite(c)
    return c - float(c[ok].mean())


# ======================================================================================
# 2. Frequency scaling
# ======================================================================================


def frequency_scaling_from_base(
    base: pd.DataFrame,
    *,
    timeframes: Sequence[str] = ("M1", "M5", "M15", "H1"),
    train_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Curl variance vs bar size, built by aggregating ONE aligned M1 frame.

    Returns ``(table, summary)``. ``summary["slope"]`` is the OLS slope of
    ``log(curl variance)`` on ``log(bar seconds)``.

    Reading it -- and this is a correction to the first-pass framing, which was too loose:

    ``slope ~ 0`` **and** mean curl ~ 0 **and** ACF ~ 0
        Pure asynchronous sampling. The null is true. Expected outcome.

    ``slope ~ 0`` **but** mean curl != 0
        A constant offset: bid convention, or a genuine persistent dislocation. A real
        dislocation of fixed magnitude is ALSO flat in bar size, so flatness alone does
        not rule it out -- separate the two with ``convention_report`` and the ACF.

    ``slope ~ 1``
        Curl variance scales like price variance, i.e. effective staleness grows with the
        bar. That is the signature of a feed whose update interval does not scale with the
        bar -- a pair that effectively prints once per bar. This is a DATA-QUALITY finding,
        not a discovery.
    """
    rows = []
    for tf in timeframes:
        frame = base if tf == "M1" else resample_aligned(base, tf)
        curls = cs.curl_frame(frame)
        if train_mask is not None and tf == "M1":
            curls = curls[train_mask]
        arr = curls.to_numpy(dtype=float)
        rets = np.column_stack(
            [np.diff(cs.signed_log_price(frame, b, q), prepend=np.nan) for b, q in cs.PAIRS]
        )
        # variance about the mean: strips the constant convention offset
        var_curl = float(np.nanvar(arr))
        rows.append(
            {
                "timeframe": tf,
                "bar_seconds": TIMEFRAMES[tf],
                "n_bars": len(frame),
                "mean_curl_bp": float(np.nanmean(arr) * 1e4),
                "rms_curl_bp": float(np.sqrt(np.nanmean(arr**2)) * 1e4),
                "sd_curl_bp": float(np.sqrt(var_curl) * 1e4),
                "rms_return_bp": float(np.sqrt(np.nanmean(rets**2)) * 1e4),
                "mean_ticks_eurusd": float(frame["EURUSD_ticks"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["noise_to_signal"] = table["sd_curl_bp"] / table["rms_return_bp"]

    x = np.log(table["bar_seconds"].to_numpy(dtype=float))
    y = np.log(table["sd_curl_bp"].to_numpy(dtype=float) ** 2)
    slope, intercept = np.polyfit(x, y, 1)
    ref = np.polyfit(x, np.log(table["rms_return_bp"].to_numpy(dtype=float) ** 2), 1)[0]
    summary = {
        "slope": float(slope),
        "intercept": float(intercept),
        "return_variance_slope": float(ref),
        "verdict_slope": (
            "FLAT — consistent with asynchronous sampling"
            if abs(slope) < 0.25
            else "RISING — suspect a feed that prints ~once per bar; audit before believing"
            if slope > 0.25
            else "FALLING — unexpected; investigate"
        ),
    }
    return table, summary


# ======================================================================================
# 3. calculate_causal_excess_curl
# ======================================================================================


def calculate_causal_excess_curl(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    min_train: int = 20_000,
    refit_every: int = 5_000,
    max_train: int = 250_000,
    n_bins: int = 20,
    smooth: int = 60,
    var_method: str | None = None,
    parkinson_window: int = 4,
    basis: Sequence[Sequence[str]] = cs.CYCLE_BASIS,
    demean: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The index. Returns ``(result, calibration_history)``.

    For each of the **3 independent** triangles (the 4th is a linear combination and is
    dropped -- see ``cs.CYCLE_BASIS``):

      curl  ->  de-mean on train rows (removes the bid/convention constant)
             ->  null P_T = sum_e Var(e)/n_e * truncation(n_e)
             ->  fit E[c^2] = a + b*P on EQUAL-COUNT DECILES of P, using only PAST rows
             ->  excess X = c^2 / (a + b*P)

    then average X across the basis and take the log of a trailing mean.

    Calibration is causal by construction: the fit for block ``[s, s+refit_every)`` uses
    only rows ``[s-max_train, s)``. ``max_train`` caps the window because a fully expanding
    fit re-bins the entire history at every refit, which is O(n^2) and unusable at M1
    scale; the cap also tracks regime drift, which a frozen fit demonstrably does not
    (synthetic: OOS mean X wandered 0.57-1.26 under a *true* null).

    ``var_method`` defaults to ``"rv"`` when sub-bar realized variance is available
    (i.e. the frame came out of ``resample_aligned``) and ``"parkinson"`` otherwise
    (i.e. at M1).
    """
    n = len(df)
    if var_method is None:
        var_method = "rv" if f"{cs.symbol(*cs.PAIRS[0])}_rv" in df.columns else "parkinson"
    if n < min_train + refit_every:
        raise ValueError(
            f"need at least {min_train + refit_every} bars, got {n}. "
            "Lower min_train or pull more history."
        )

    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(train_fraction * n)] = True

    out: dict[str, np.ndarray] = {}
    excesses: list[np.ndarray] = []
    history: list[dict[str, float | str]] = []

    for tri in basis:
        name = "_".join(tri)
        c = cs.triangle_curl(df, tri)
        if demean:
            c = demean_curl(c, train_mask=train_mask)
        p = cs.staleness_null(df, tri, method=var_method, window=parkinson_window)

        x = np.full(n, np.nan)
        for start in range(min_train, n, refit_every):
            lo = max(0, start - max_train)
            fit_mask = np.zeros(n, dtype=bool)
            fit_mask[lo:start] = True  # strictly past
            try:
                a, b = cs.calibrate_null(c, p, train_mask=fit_mask, n_bins=n_bins)
            except ValueError:
                continue
            stop = min(start + refit_every, n)
            x[start:stop] = cs.excess_curl(c, p, a, b)[start:stop]
            history.append(
                {
                    "triangle": name,
                    "block_start": int(start),
                    "timestamp": df.index[start],
                    "fit_rows": int(start - lo),
                    "intercept": a,
                    "slope": b,
                }
            )

        out[f"curl_{name}"] = c
        out[f"null_{name}"] = p
        out[f"excess_{name}"] = x
        excesses.append(x)

    result = pd.DataFrame(out, index=df.index)
    result["excess_mean"] = np.nanmean(np.column_stack(excesses), axis=1)
    roll = result["excess_mean"].rolling(smooth, min_periods=max(4, smooth // 4)).mean()
    result["stress"] = np.log(roll.where(roll > 0))
    result["stress_resid"] = cs.residualise_calendar(result["stress"])
    return result, pd.DataFrame(history)


# ======================================================================================
# Orchestrator
# ======================================================================================


def run_day1(
    raw: Mapping[tuple[str, str], pd.DataFrame],
    *,
    outdir: str | Path = "results/curl",
    train_fraction: float = 0.70,
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Full day-one sequence. Prints the report and writes CSVs to ``outdir``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    base = prepare_frames(raw)
    n = len(base)
    train = np.zeros(n, dtype=bool)
    train[: int(train_fraction * n)] = True
    print(f"[1/6] aligned {n:,} common bars  {base.index[0]} -> {base.index[-1]}")

    print("\n[1b/6] tick_volume audit — is the broker's tick count usable at all?")
    audit = tick_volume_audit(base)
    print(
        audit[
            ["pair", "median", "max", "pct_zero", "pct_at_max", "n_distinct", "verdict"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    for tri in cs.CYCLE_BASIS:
        ex = staleness_exponent_test(base, tri, train_mask=train)
        print(
            f"      {'_'.join(tri):<12} beta={ex['beta_ticks']:+.2f} "
            f"z_vs_shuffled={ex['z_vs_shuffled']:+.1f} -> {ex['verdict']}"
        )

    print("\n[2/6] convention_report (must be clean before anything else)")
    conv = convention_report(base, train_mask=train)
    print(conv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rank = cs.cycle_rank(base)
    print(f"      cycle rank = {rank} (must be 3)")

    print("\n[3/6] frequency scaling")
    table, summary = frequency_scaling_from_base(base)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"      log-log slope of curl variance = {summary['slope']:+.3f} "
        f"(return variance slope {summary['return_variance_slope']:+.3f})"
    )
    print(f"      {summary['verdict_slope']}")

    print("\n[4/6] curl autocorrelation (stale-feed detector)")
    acf = cs.curl_autocorrelation(base, lags=5)
    print(acf.round(4).to_string())

    print("\n[5/6] causal excess curl")
    result, calib = calculate_causal_excess_curl(
        base, train_fraction=train_fraction, **kwargs  # type: ignore[arg-type]
    )
    print(
        f"      stress finite on {np.isfinite(result['stress']).sum():,} bars; "
        f"calibration slope median {calib['slope'].median():.3f}, refits {len(calib):,}"
    )
    print(
        "      (theory says ~0.5; at M1 a slope above that is EXPECTED — Parkinson"
        "\n       underestimates the range when a bar holds few ticks, so P is biased low"
        "\n       and the fitted slope absorbs it. Watch the slope's STABILITY over blocks,"
        "\n       in calibration_history.csv, not its level.)"
    )

    print("\n[6/6] confounds — the stress index must beat these to be a new observable")
    diag = _confound_report(base, result)
    print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    result["stress_orthogonal"] = residualise_against_confounds(
        result["stress"], base, train_mask=train
    )
    diag_o = _confound_report(base, result, column="stress_orthogonal")
    print("\n      after residualising (this is the column to pre-register):")
    print(diag_o.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    diag_o.insert(0, "stage", "orthogonal")
    diag.insert(0, "stage", "raw")
    diag = pd.concat([diag, diag_o], ignore_index=True)

    audit.to_csv(out / "tick_volume_audit.csv", index=False)
    conv.to_csv(out / "convention_report.csv", index=False)
    table.to_csv(out / "frequency_scaling.csv", index=False)
    acf.to_csv(out / "curl_autocorrelation.csv")
    calib.to_csv(out / "calibration_history.csv", index=False)
    result.to_csv(out / "causal_excess_curl.csv")
    diag.to_csv(out / "confounds.csv", index=False)
    print(f"\nwrote -> {out.resolve()}")
    return {"convention": conv, "scaling": table, "acf": acf, "index": result, "confounds": diag}


# ======================================================================================
# 3b. Is the broker's tick_volume usable as a staleness proxy?
# ======================================================================================


def tick_volume_audit(base: pd.DataFrame) -> pd.DataFrame:
    """Health check on ``tick_volume`` before you trust it as the staleness estimator.

    WHAT tick_volume ACTUALLY IS: the count of price updates that *this broker's* server
    published during the bar. It is not traded volume -- FX is OTC and has no consolidated
    tape (``real_volume`` is 0 for essentially every retail FX broker).

    Why that is fine, and arguably ideal, HERE: the curl is created by staleness of the
    feed whose Close you are reading. So the right quantity is the update rate of that
    exact feed, which is what tick_volume measures. True interbank volume would be the
    *wrong* number. You do not need tick_volume to be "real".

    What would break it, and what each column detects:

    ``pct_zero``      bars with no update at all. Should be ~0 in liquid hours.
    ``pct_flat``      bars with high == low. A volume-free activity proxy; should track
                      ``pct_zero``. If pct_flat is high while tick_volume is not low, the
                      feed is publishing updates that do not move the price (or the counts
                      are synthetic).
    ``corr_ticks_flat`` correlation of log ticks with the flat-bar proxy computed WITHOUT
                      volume. Strong negative = the two independent activity measures
                      agree = tick_volume is measuring something real.
    ``pct_at_max``    saturation. Throttled feeds cap updates per second, so tick_volume
                      piles up at a ceiling during fast markets -- exactly when you need
                      it. A visible spike at the maximum is disqualifying for that pair.
    ``n_distinct``    quantisation. A tiny number of distinct values means the counts are
                      bucketed or synthetic.
    ``yearly_median_ratio`` max/min of the per-year median. Brokers change liquidity
                      providers and backfill old history from a different source; a large
                      ratio means the early history is not comparable to the recent history
                      and you should truncate rather than "fix" it.
    """
    rows = []
    for b, q in cs.PAIRS:
        sym = cs.symbol(b, q)
        n = base[f"{sym}_ticks"].astype(float)
        flat = (base[f"{sym}_high"] <= base[f"{sym}_low"]).astype(float)
        yearly = n.groupby(n.index.year).median()
        logn = np.log(n.where(n > 0))
        rows.append(
            {
                "pair": sym,
                "median": float(n.median()),
                "p01": float(n.quantile(0.01)),
                "p99": float(n.quantile(0.99)),
                "max": float(n.max()),
                "pct_zero": float((n <= 0).mean() * 100),
                "pct_flat": float(flat.mean() * 100),
                "corr_ticks_flat": float(
                    pd.concat([logn, flat], axis=1).dropna().corr().iloc[0, 1]
                ),
                "pct_at_max": float((n >= n.max() * 0.999).mean() * 100),
                "n_distinct": int(n.nunique()),
                "yearly_median_ratio": float(yearly.max() / max(yearly.min(), 1e-9))
                if len(yearly) > 1
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    # The flat-bar cross-check is only meaningful when flat bars actually occur. At M1 on
    # majors they are nearly absent, so a weak correlation there says nothing.
    flat_check_valid = out["pct_flat"] > 0.5
    out["verdict"] = np.where(
        (out["pct_at_max"] > 0.5) | (out["n_distinct"] < 50),
        "SUSPECT — saturated or quantised; do not use as a staleness proxy",
        np.where(
            flat_check_valid & (out["corr_ticks_flat"] > -0.15),
            "SUSPECT — disagrees with the volume-free activity proxy",
            np.where(
                out["yearly_median_ratio"] > 3.0,
                "level drifts across years — decide with staleness_exponent_by_era, "
                "NOT with this ratio",
                "usable",
            ),
        ),
    )
    return out


def staleness_exponent_test(
    df: pd.DataFrame,
    tri: Sequence[str],
    *,
    var_method: str = "parkinson",
    window: int = 4,
    n_bins: int = 5,
    train_mask: np.ndarray | None = None,
    n_permutations: int = 20,
    seed: int = 20260808,
) -> dict[str, float]:
    """THE decisive test of whether tick_volume is a valid staleness proxy.

    Theory says ``E[c^2] = 0.5 * Vsum / Nharm`` where ``Vsum`` is the summed edge variance
    and ``Nharm`` the variance-weighted harmonic mean tick count. So fitting

        log E[c^2]  =  k  +  alpha * log(Vsum)  +  beta * log(Nharm)

    must give ``alpha = +1`` and ``beta = -1``. This is a direct, falsifiable statement
    about YOUR broker's tick counts, measurable on your own data in minutes.

    ``beta`` near -1  : tick_volume tracks true staleness. Proceed.
    ``beta`` near  0  : tick_volume carries no staleness information (saturated, synthetic,
                        or constant). The null's liquidity channel is dead -- you can still
                        run the study, but only as a volatility-controlled measurement, and
                        you must say so in the log.
    ``alpha`` far from 1 : the variance estimator is the problem, not the tick counts.

    Fitted on 2-D quantile bin means, for the same chi-square reason as ``calibrate_null``.

    Two traps this function had to be hardened against, both found in testing:

    * **Variance/tick entanglement.** Parkinson underestimates the range when a bar holds
      few ticks, so the variance estimate is itself a function of the tick count and the
      fitted ``beta`` is contaminated. The variance here is therefore taken from the
      PRECEDING ``window`` bars (shifted by one), which decouples it from the current bar's
      tick count at the cost of a little staleness in the volatility estimate.
    * **Collinearity.** If tick counts barely vary (a constant or heavily throttled feed),
      ``log(Nharm)`` is nearly constant, the design is rank-deficient and ``lstsq`` splits
      the intercept arbitrarily -- returning a confident-looking ``beta`` that means
      nothing. The spread of ``log(Nharm)`` is checked explicitly.
    """
    n_rows = len(df)
    mask = np.ones(n_rows, dtype=bool) if train_mask is None else train_mask
    c = cs.triangle_curl(df, tri)
    c = demean_curl(c, train_mask=mask)

    variances: list[np.ndarray] = []
    ticks: list[np.ndarray] = []
    for x, y in cs.triangle_edges(tri):
        b, q = cs.PAIR_LOOKUP[frozenset((x, y))]
        sym = cs.symbol(b, q)
        if var_method == "rv":
            var = pd.Series(df[f"{sym}_rv"].to_numpy(dtype=float), index=df.index)
        else:
            var = pd.Series(cs.parkinson_variance(df, b, q, window=1), index=df.index)
        # variance from PRECEDING bars only -> independent of this bar's tick count
        variances.append(
            var.shift(1).rolling(window, min_periods=1).mean().to_numpy(dtype=float)
        )
        ticks.append(np.maximum(df[f"{sym}_ticks"].to_numpy(dtype=float), 1.0))

    def _fit(tick_cols: list[np.ndarray]) -> float:
        vsum = np.zeros(n_rows)
        vovern = np.zeros(n_rows)
        for var_lag, n in zip(variances, tick_cols):
            vsum = vsum + var_lag
            vovern = vovern + var_lag / n * cs._truncation_factor(n)
        nh = np.divide(vsum, vovern, out=np.full(n_rows, np.nan), where=vovern > 0)
        good = mask & np.isfinite(c) & (vsum > 0) & np.isfinite(nh) & (nh > 0)
        fr = pd.DataFrame({"c2": c[good] ** 2, "V": vsum[good], "N": nh[good]})
        fr["bv"] = pd.qcut(fr["V"], n_bins, labels=False, duplicates="drop")
        fr["bn"] = pd.qcut(fr["N"], n_bins, labels=False, duplicates="drop")
        gg = fr.groupby(["bv", "bn"], observed=True).agg(
            cnt=("c2", "size"), c2=("c2", "mean"), V=("V", "mean"), N=("N", "mean")
        )
        gg = gg[gg["cnt"] >= 30]
        if len(gg) < 6:
            return np.nan
        dm = np.column_stack(
            [np.ones(len(gg)), np.log(gg["V"].to_numpy()), np.log(gg["N"].to_numpy())]
        )
        cf, *_ = np.linalg.lstsq(dm, np.log(gg["c2"].to_numpy()), rcond=None)
        return float(cf[2])

    vsum = np.zeros(n_rows)
    vovern = np.zeros(n_rows)
    for var_lag, n in zip(variances, ticks):
        vsum = vsum + var_lag
        vovern = vovern + var_lag / n * cs._truncation_factor(n)
    nharm = np.divide(vsum, vovern, out=np.full(n_rows, np.nan), where=vovern > 0)

    ok = mask & np.isfinite(c) & (vsum > 0) & np.isfinite(nharm) & (nharm > 0)
    log_n_spread = float(np.nanstd(np.log(nharm[ok]))) if ok.any() else 0.0
    if log_n_spread < 0.05:
        return {
            "alpha_volatility": np.nan,
            "beta_ticks": np.nan,
            "beta_shuffled_mean": np.nan,
            "beta_shuffled_sd": np.nan,
            "z_vs_shuffled": np.nan,
            "r2": np.nan,
            "n_cells": 0,
            "log_nharm_sd": log_n_spread,
            "verdict": (
                "UNDEFINED — tick counts barely vary (constant/throttled feed). "
                "The staleness channel is unusable; treat the null as volatility-only."
            ),
        }
    frame = pd.DataFrame(
        {"c2": c[ok] ** 2, "V": vsum[ok], "N": nharm[ok]}
    )
    frame["bv"] = pd.qcut(frame["V"], n_bins, labels=False, duplicates="drop")
    frame["bn"] = pd.qcut(frame["N"], n_bins, labels=False, duplicates="drop")
    g = frame.groupby(["bv", "bn"], observed=True).agg(
        cnt=("c2", "size"), c2=("c2", "mean"), V=("V", "mean"), N=("N", "mean")
    )
    g = g[g["cnt"] >= 30]
    if len(g) < 6:
        raise ValueError("too few populated cells for the exponent test")

    design = np.column_stack(
        [np.ones(len(g)), np.log(g["V"].to_numpy()), np.log(g["N"].to_numpy())]
    )
    y = np.log(g["c2"].to_numpy())
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    r2 = 1.0 - float(resid.var() / y.var()) if y.var() > 0 else np.nan
    beta = float(coef[2])

    # PERMUTATION CONTROL. Do not judge beta against the textbook -1: the volatility term
    # is measured with error (lagged), and volatility and activity are correlated, so
    # errors-in-variables biases beta away from -1 by an amount that depends on the data.
    # On honest synthetic ticks this fit returns about -1.7, not -1.0. What is diagnostic
    # is beta versus the SAME statistic computed on shuffled tick counts, which destroys
    # any bar-level staleness information while preserving the marginal distribution.
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(max(0, n_permutations)):
        shuffled = [rng.permutation(t) for t in ticks]
        val = _fit(shuffled)
        if np.isfinite(val):
            perm.append(val)
    perm_arr = np.asarray(perm, dtype=float)
    perm_mean = float(perm_arr.mean()) if len(perm_arr) else np.nan
    perm_sd = float(perm_arr.std(ddof=1)) if len(perm_arr) > 1 else np.nan
    z = (beta - perm_mean) / perm_sd if perm_sd and perm_sd > 0 else np.nan

    if not np.isfinite(z):
        verdict = "INCONCLUSIVE — permutation control failed to fit"
    elif z < -4.0:
        verdict = "tick_volume CARRIES REAL staleness information — proceed"
    elif z < -2.0:
        verdict = "WEAK staleness signal — usable but report the limitation"
    else:
        verdict = (
            "tick_volume is INDISTINGUISHABLE FROM SHUFFLED — no staleness information. "
            "Treat the null as volatility-only and say so in the log."
        )
    return {
        "alpha_volatility": float(coef[1]),
        "beta_ticks": beta,
        "beta_shuffled_mean": perm_mean,
        "beta_shuffled_sd": perm_sd,
        "z_vs_shuffled": float(z) if np.isfinite(z) else np.nan,
        "r2": r2,
        "n_cells": int(len(g)),
        "log_nharm_sd": log_n_spread,
        "verdict": verdict,
    }


def staleness_exponent_by_era(
    base: pd.DataFrame,
    tri: Sequence[str],
    *,
    freq: str = "YS",
    min_rows: int = 30_000,
    **kwargs: object,
) -> pd.DataFrame:
    """Run ``staleness_exponent_test`` separately per era. THE test for feed drift.

    A multi-year tick-level drift is only a problem if it changes the RELATIONSHIP between
    reported tick counts and true staleness. A pure level shift is absorbed by the causal
    refit in ``calculate_causal_excess_curl`` and is harmless.

    So do not decide whether to truncate history from ``yearly_median_ratio`` -- that is a
    crude heuristic (it is a max/min over years, so one unusually quiet year sets it, and a
    partial final year distorts it further; it was written to catch BACKFILLED history from
    a different source, which appears as a step discontinuity, not as smooth growth).
    Decide from this table instead:

    ``z_vs_shuffled`` strongly negative in every era
        The tick counts carry staleness information throughout. The drift is a level
        effect. Keep the full history.
    ``z_vs_shuffled`` near zero in the early eras only
        The early feed is not measuring what the recent feed measures. Truncate to the
        first era where it is clearly negative, and record the cut in the log.
    """
    out = []
    for period, sub in base.groupby(pd.Grouper(freq=freq)):
        if len(sub) < min_rows:
            continue
        try:
            r = staleness_exponent_test(sub, tri, **kwargs)  # type: ignore[arg-type]
        except ValueError as exc:
            r = {"verdict": f"failed: {exc}"}
        row = {"era": str(period)[:10], "n_bars": len(sub)}
        row.update(
            {
                k: r.get(k)
                for k in (
                    "beta_ticks",
                    "beta_shuffled_mean",
                    "z_vs_shuffled",
                    "median_ticks_eurusd",
                    "verdict",
                )
            }
        )
        row["median_ticks_eurusd"] = float(sub["EURUSD_ticks"].median())
        out.append(row)
    return pd.DataFrame(out)


def tick_level_breaks(base: pd.DataFrame, *, freq: str = "MS") -> pd.DataFrame:
    """Separate smooth activity growth from a step change in the feed.

    Market activity drifts smoothly; a liquidity-provider or feed change is a STEP. Also
    reports ticks per unit of realised range: if tick counts rise while that ratio stays
    flat, the market got busier. If the ratio jumps, the plumbing changed.
    """
    rows = []
    for period, sub in base.groupby(pd.Grouper(freq=freq)):
        if len(sub) < 500:
            continue
        ticks = float(sub["EURUSD_ticks"].median())
        rng = float(
            np.log(sub["EURUSD_high"] / sub["EURUSD_low"]).replace(0, np.nan).median()
        )
        rows.append(
            {
                "period": str(period)[:7],
                "n": len(sub),
                "median_ticks": ticks,
                "median_log_range_bp": rng * 1e4,
                "ticks_per_bp_range": ticks / (rng * 1e4) if rng and rng > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["log_ticks_step"] = np.log(out["median_ticks"]).diff()
    out["log_ratio_step"] = np.log(out["ticks_per_bp_range"]).diff()
    return out


def confound_design(
    base: pd.DataFrame,
    *,
    detrend_window: int | None = None,
    n_tick_bins: int = 12,
    bin_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """The things the stress index is most likely to secretly be.

    Total tick count, summed H/L range (volatility), summed relative spread (when
    available), plus hour-of-day and day-of-week dummies.

    ``detrend_window`` optionally expresses each continuous confound relative to its own
    causal trailing median. **Default OFF, because it was measured and it did not work.**
    The reasoning was that a broker's tick level drifts 3-4x across years, so a global
    coefficient on absolute ``log_ticks`` would be mis-specified. On a synthetic feed with
    exactly that drift imposed, residual corr(stress, log ticks) on the validation slice
    was +0.233 with the absolute form, +0.244 relative, +0.240 with both, and +0.301 when
    the detrended series was also binned. The hypothesis was wrong; the option is retained
    only so the finding stays reproducible.

    What DID help: quantile dummies on absolute log tick count (+0.233 -> +0.126). The
    relationship is nonlinear in log space, not drifting.
    """
    cols: dict[str, np.ndarray] = {}

    def add(name: str, series: pd.Series) -> None:
        v = np.log(np.maximum(series.to_numpy(dtype=float), 1e-12))
        if detrend_window:
            ref = (
                pd.Series(v, index=series.index)
                .shift(1)
                .rolling(detrend_window, min_periods=max(50, detrend_window // 20))
                .median()
            )
            v = v - ref.bfill().to_numpy(dtype=float)
        cols[name] = v

    add("log_ticks", sum(base[f"{cs.symbol(*p)}_ticks"] for p in cs.PAIRS))
    ticks_for_bins = cols["log_ticks"]
    add(
        "log_range",
        sum(
            np.log(base[f"{cs.symbol(*p)}_high"] / base[f"{cs.symbol(*p)}_low"])
            for p in cs.PAIRS
        ),
    )
    if all(f"{cs.symbol(*p)}_spread" in base.columns for p in cs.PAIRS):
        add(
            "log_spread",
            sum(
                base[f"{cs.symbol(*p)}_spread"] / base[f"{cs.symbol(*p)}_close"]
                for p in cs.PAIRS
            ),
        )
    design = pd.DataFrame(cols, index=base.index)

    # Tick count enters as QUANTILE DUMMIES as well as linearly. The stress/activity
    # relationship is not linear in log space: on a drifting synthetic feed, a purely
    # linear confound model fitted on [0:70%] still left corr(residual, log ticks) = +0.23
    # on the validation slice, while adding tick deciles cut that to +0.13. Neither is
    # zero -- see the caveat in CURL_EXPERIMENT_PLAN.md 3.2b.
    finite = np.isfinite(ticks_for_bins)
    ref = ticks_for_bins[finite] if bin_mask is None else ticks_for_bins[finite & bin_mask]
    if len(ref) > 500:
        edges = np.quantile(ref, np.linspace(0, 1, n_tick_bins + 1)[1:-1])
        buckets = np.digitize(ticks_for_bins, edges)
        tick_d = pd.get_dummies(buckets, prefix="tq", drop_first=True).astype(float)
        tick_d.index = design.index
        design = pd.concat([design, tick_d], axis=1)

    hours = pd.get_dummies(design.index.hour, prefix="h", drop_first=True).astype(float)
    dows = pd.get_dummies(design.index.dayofweek, prefix="d", drop_first=True).astype(float)
    hours.index = design.index
    dows.index = design.index
    return pd.concat([design, hours, dows], axis=1)


def residualise_against_confounds(
    stress: pd.Series,
    base: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    detrend_window: int | None = None,
) -> pd.Series:
    """Regress the stress index on the confounds (TRAIN rows only) and return the residual.

    **This, not the raw index, is the quantity to pre-register.** On synthetic data where
    the asynchronicity null is TRUE BY CONSTRUCTION, the raw stress index still correlated
    +0.40 with total tick count and -0.35 with volatility and with spread: the null
    normalisation removes most of the liquidity/volatility channel but not all of it.
    Testing the raw index would therefore be testing a repackaged liquidity proxy, and it
    would "work" for reasons that have nothing to do with the Hodge decomposition.

    Coefficients are fitted on train rows only, so applying this to the validation slice
    introduces no look-ahead.
    """
    x = confound_design(base, detrend_window=detrend_window, bin_mask=train_mask)
    y = stress.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), xv])
    ok = train_mask & np.isfinite(y) & np.isfinite(design).all(axis=1)
    if ok.sum() < 10 * design.shape[1]:
        raise ValueError("not enough clean training rows to residualise")
    coef, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
    fitted = design @ coef
    return pd.Series(y - fitted, index=stress.index, name="stress_orthogonal")


def _confound_report(
    base: pd.DataFrame, result: pd.DataFrame, *, column: str = "stress"
) -> pd.DataFrame:
    """Correlate the stress index against the things it is most likely to just be."""
    s = result[column]
    rows: list[dict[str, object]] = []

    def add(name: str, other: pd.Series) -> None:
        joined = pd.concat([s, other], axis=1).dropna()
        if len(joined) > 100:
            a, b = joined.iloc[:, 0], joined.iloc[:, 1]
            rows.append(
                {
                    "confound": name,
                    "n": len(joined),
                    "corr": float(a.corr(b)),
                    # rank correlation without scipy: Pearson on ranks
                    "spearman": float(a.rank().corr(b.rank())),
                }
            )

    ticks = sum(base[f"{cs.symbol(*p)}_ticks"] for p in cs.PAIRS)
    add("total tick count", np.log(ticks.where(ticks > 0)))
    hl = sum(
        np.log(base[f"{cs.symbol(*p)}_high"] / base[f"{cs.symbol(*p)}_low"]) for p in cs.PAIRS
    )
    add("summed H/L range (volatility)", np.log(hl.where(hl > 0)))
    if all(f"{cs.symbol(*p)}_spread" in base.columns for p in cs.PAIRS):
        sp = sum(
            base[f"{cs.symbol(*p)}_spread"] / base[f"{cs.symbol(*p)}_close"] for p in cs.PAIRS
        )
        add("summed relative spread", np.log(sp.where(sp > 0)))
    add("hour of day (as ordinal)", pd.Series(s.index.hour, index=s.index, dtype=float))
    return pd.DataFrame(rows)
