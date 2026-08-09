"""Monte-Carlo validation of the asynchronicity null in ``src/curl_stress.py`` — RESEARCH ONLY.

Runs entirely on synthetic data. Its job is to prove, BEFORE you touch a broker feed, that
the estimator does what the theory says:

  A. Simultaneous (arbitrage-free) prices give machine-epsilon curl, and a mismatched
     quote-level convention produces a LARGE CONSTANT curl that has nothing to do with
     microstructure. (This is the first thing that will bite you on real data.)
  B. Asynchronous last-tick sampling of an EXACTLY arbitrage-free world manufactures curl
     whose per-bar variance is predicted, quantitatively, by the closed-form staleness
     formula and tracked by the feed-only proxy ``staleness_null``.
  C. Curl variance is FLAT across bar sizes at a fixed tick rate, while return variance
     grows linearly -- the day-one discriminator to run on the real feed.
  D. An injected genuine dislocation is picked up by excess curl, and the High/Low
     infeasibility certificate only fires when the dislocation is comparable to the bar
     range (so it is an M1-and-events tool, not an M15 tool).

Run:  python -m src.curl_null_simulation
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import curl_stress as cs

RNG_SEED = 20260808
N_BARS_M15 = 6_000

#: per-second variance rates of the currency potentials (arbitrary but distinct)
VAR_RATES: dict[str, float] = {
    "EUR": 4.0e-10,
    "USD": 3.0e-10,
    "JPY": 5.0e-10,
    "GBP": 6.0e-10,
}

#: baseline ticks per second per pair (retail-feed plausible on majors)
TICK_RATES: dict[tuple[str, str], float] = {
    ("EUR", "USD"): 0.40,
    ("EUR", "JPY"): 0.22,
    ("EUR", "GBP"): 0.18,
    ("GBP", "USD"): 0.30,
    ("GBP", "JPY"): 0.15,
    ("USD", "JPY"): 0.35,
}

#: log price LEVEL per currency. Pair levels are DERIVED from these, so the synthetic
#: world is arbitrage-free by construction. Hard-coding six independent pair levels
#: instead (the obvious way to write this fixture) injects a constant curl of tens of
#: basis points -- see test A.
CURRENCY_LEVEL: dict[str, float] = {
    "EUR": 0.0,
    "USD": -float(np.log(1.08)),
    "JPY": -float(np.log(170.0)),
    "GBP": -float(np.log(0.85)),
}


def base_log_level(base: str, quote: str) -> float:
    return CURRENCY_LEVEL[base] - CURRENCY_LEVEL[quote]


def _staleness_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E|tau_a - tau_b| for independent exponentials with means a, b."""
    return a + b - 2.0 * a * b / (a + b)


def simulate_bars(
    n_bars: int,
    bar_seconds: int,
    *,
    rng: np.random.Generator,
    dislocated_pair: tuple[str, str] | None = None,
    dislocation_log: float = 0.0,
    dislocation_mask: np.ndarray | None = None,
    activity_trend: np.ndarray | None = None,
    chunk: int = 400,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Simulate an EXACTLY arbitrage-free world, then observe it asynchronously.

    Currency potentials are independent Brownian motions on a 1-second grid, so every
    triangle has zero curl at every instant BY CONSTRUCTION. The only thing that creates
    curl is that each pair's OHLC is built from ITS OWN tick times.

    Volatility and liquidity are persistent (AR(1) in logs) so that a rolling-window
    variance estimate can actually track them -- which is what makes ``staleness_null``
    a non-trivial per-bar predictor rather than a constant.
    """
    # persistent log-vol and log-activity factors (vol clustering + liquidity cycle)
    def ar1(phi: float, sd: float) -> np.ndarray:
        e = rng.normal(0.0, sd, size=n_bars)
        out = np.empty(n_bars)
        acc = 0.0
        for i in range(n_bars):
            acc = phi * acc + e[i]
            out[i] = acc
        return out

    log_vol = ar1(0.995, 0.05)
    log_act = -0.6 * log_vol + ar1(0.995, 0.05)
    vol_fac = np.exp(log_vol - log_vol.mean())
    act_fac = np.exp(log_act - log_act.mean())
    if activity_trend is not None:
        # multi-year drift in the broker's tick level, holding the price process fixed
        act_fac = act_fac * np.asarray(activity_trend, dtype=float)

    cols: dict[str, list[np.ndarray]] = {
        f"{cs.symbol(b, q)}_{f}": [] for b, q in cs.PAIRS for f in cs.FIELDS
    }
    idx = {c: i for i, c in enumerate(cs.CURRENCIES)}
    sd_vec = np.sqrt(np.array([VAR_RATES[c] for c in cs.CURRENCIES]))

    done = 0
    while done < n_bars:
        m = min(chunk, n_bars - done)
        sl = slice(done, done + m)
        incr = rng.normal(0.0, 1.0, size=(m, bar_seconds, len(cs.CURRENCIES))) * sd_vec
        incr *= np.sqrt(vol_fac[sl])[:, None, None]
        phi = np.cumsum(incr.reshape(-1, len(cs.CURRENCIES)), axis=0).reshape(
            m, bar_seconds, len(cs.CURRENCIES)
        )

        for base, quote in cs.PAIRS:
            path = base_log_level(base, quote) + phi[:, :, idx[base]] - phi[:, :, idx[quote]]
            if dislocated_pair is not None and frozenset((base, quote)) == frozenset(
                dislocated_pair
            ):
                sel = (
                    np.ones(m, dtype=bool)
                    if dislocation_mask is None
                    else dislocation_mask[sl]
                )
                path = path + dislocation_log * sel[:, None]

            p_tick = np.clip(TICK_RATES[(base, quote)] * act_fac[sl], 1e-4, 0.9)
            ticks = rng.random((m, bar_seconds)) < p_tick[:, None]
            ticks[:, 0] = True  # guarantee >= 1 tick per bar
            n = ticks.sum(axis=1).astype(float)

            rows = np.arange(m)
            last = bar_seconds - 1 - np.argmax(ticks[:, ::-1], axis=1)
            first = np.argmax(ticks, axis=1)
            observed = np.where(ticks, path, np.nan)

            sym = cs.symbol(base, quote)
            cols[f"{sym}_open"].append(np.exp(path[rows, first]))
            cols[f"{sym}_high"].append(np.exp(np.nanmax(observed, axis=1)))
            cols[f"{sym}_low"].append(np.exp(np.nanmin(observed, axis=1)))
            cols[f"{sym}_close"].append(np.exp(path[rows, last]))
            cols[f"{sym}_ticks"].append(n)
        done += m

    index = pd.date_range("2024-01-01", periods=n_bars, freq=f"{bar_seconds}s", tz="UTC")
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()}, index=index)

    # closed-form E[c^2] per bar per triangle, using the REALISED tick counts
    truth: dict[str, np.ndarray] = {"vol_fac": vol_fac, "act_fac": act_fac}
    # truncated-exponential mean staleness: E[tau] = D/n - D/(e^n - 1)
    mean_stale = {}
    for b, q in cs.PAIRS:
        nt = np.maximum(df[f"{cs.symbol(b, q)}_ticks"].to_numpy(dtype=float), 1.0)
        mean_stale[(b, q)] = (bar_seconds / nt) * cs._truncation_factor(nt)
    for tri in cs.TRIANGLES:
        edges = cs.triangle_edges(tri)
        exp_c2 = np.zeros(n_bars)
        for cur in tri:
            at_cur = [e for e in edges if cur in e]
            m1 = mean_stale[cs.PAIR_LOOKUP[frozenset(at_cur[0])]]
            m2 = mean_stale[cs.PAIR_LOOKUP[frozenset(at_cur[1])]]
            exp_c2 += VAR_RATES[cur] * vol_fac * _staleness_gap(m1, m2)
        truth[f"exp_c2_{'_'.join(tri)}"] = exp_c2
    return df, truth


def simultaneous_frame(
    n_bars: int, *, rng: np.random.Generator, consistent: bool = True
) -> pd.DataFrame:
    """Perfectly simultaneous observation. ``consistent=False`` reproduces the
    quote-level convention bug on purpose."""
    phi = np.cumsum(rng.normal(0.0, 1e-4, size=(n_bars, len(cs.CURRENCIES))), axis=0)
    idx = {c: i for i, c in enumerate(cs.CURRENCIES)}
    bad = {
        ("EUR", "USD"): np.log(1.08),
        ("EUR", "JPY"): np.log(170.0),
        ("EUR", "GBP"): np.log(0.85),
        ("GBP", "USD"): np.log(1.27),
        ("GBP", "JPY"): np.log(200.0),
        ("USD", "JPY"): np.log(157.0),
    }
    data: dict[str, np.ndarray] = {}
    for base, quote in cs.PAIRS:
        level = base_log_level(base, quote) if consistent else bad[(base, quote)]
        px = np.exp(level + phi[:, idx[base]] - phi[:, idx[quote]])
        sym = cs.symbol(base, quote)
        data[f"{sym}_open"] = px
        data[f"{sym}_high"] = px * 1.0001
        data[f"{sym}_low"] = px * 0.9999
        data[f"{sym}_close"] = px
        data[f"{sym}_ticks"] = np.full(n_bars, 50.0)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    return pd.DataFrame(data, index=index)


def _report(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def main() -> None:
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    rng = np.random.default_rng(RNG_SEED)
    pd.set_option("display.width", 150)

    # ---------------------------------------------------------------- A. exactness
    _report("A. Simultaneity + quote-convention check")
    good = simultaneous_frame(5_000, rng=rng, consistent=True)
    bad = simultaneous_frame(5_000, rng=rng, consistent=False)
    print(f"consistent levels   : max |curl| = {cs.curl_frame(good).abs().to_numpy().max():.3e}")
    print(
        f"inconsistent levels : max |curl| = {cs.curl_frame(bad).abs().to_numpy().max():.3e} "
        f"({cs.curl_frame(bad).abs().to_numpy().max() * 1e4:.1f} bp, CONSTANT)"
    )
    print(
        "\nLesson for the real feed: a static level/convention mismatch shows up as a curl"
        "\nwith a large NON-ZERO MEAN and near-zero variance. Always run convention_check()"
        "\nand de-mean per triangle before interpreting anything."
    )

    # ------------------------------------------------------- B. asynchronicity null
    _report("B. Asynchronous sampling of an arbitrage-free world (M15, 900s bars)")
    df, truth = simulate_bars(N_BARS_M15, 900, rng=rng)
    n = len(df)
    train = np.zeros(n, dtype=bool)
    train[: int(0.70 * n)] = True
    print(f"numerical cycle rank of the 4 triangle curls (must be 3 for K4) = {cs.cycle_rank(df)}")

    rows = []
    for tri in cs.TRIANGLES:
        name = "_".join(tri)
        c = cs.triangle_curl(df, tri)
        p = cs.staleness_null(df, tri)
        exp_c2 = truth[f"exp_c2_{name}"]
        ok = np.isfinite(c) & np.isfinite(p)
        a, b = cs.calibrate_null(c, p, train_mask=train)
        x = cs.excess_curl(c, p, a, b)
        rows.append(
            {
                "triangle": name,
                "mean_curl_bp": float(np.nanmean(c) * 1e4),
                "rms_curl_bp": float(np.sqrt(np.nanmean(c**2)) * 1e4),
                "theory_rms_bp": float(np.sqrt(np.nanmean(exp_c2)) * 1e4),
                "realised/theory": float(np.nanmean(c**2) / np.nanmean(exp_c2)),
                "slope_on_P": b,
                "corr(c2,P)": float(np.corrcoef(c[ok] ** 2, p[ok])[0, 1]),
                "OOS_mean_X": float(np.nanmean(x[~train])),
            }
        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nTheory says realised/theory ~ 1.0 (closed form is exact up to the exponential"
        "\napproximation of the backward recurrence time), the BINNED slope of c^2 on the"
        "\nfeed-only proxy P ~ 0.5, and out-of-sample mean excess X ~ 1.0."
        "\nNote corr(c2,P) ~ 0.1 EVEN THOUGH THE NULL IS EXACTLY TRUE: conditional on P,"
        "\nc^2 is P*chi^2_1, and that multiplicative noise attenuates the correlation to"
        "\nabout CV(P)/sqrt(2). Never judge the null by the raw per-bar correlation."
    )

    _report("B2. Binned calibration table (EUR_USD_JPY) — the honest way to read the null")
    tab = cs.null_calibration_table(
        cs.triangle_curl(df, ("EUR", "USD", "JPY")),
        cs.staleness_null(df, ("EUR", "USD", "JPY")),
        mask=train,
    )
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nmean_c2 should rise monotonically with mean_P; 'ratio' should be roughly flat.")

    _report("B3. Convention check — must be run FIRST on real data")
    print(cs.convention_check(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------ C. frequency scaling
    _report("C. Frequency scaling — the day-one discriminator")
    frames = {
        "M1": simulate_bars(20_000, 60, rng=rng)[0],
        "M5": simulate_bars(6_000, 300, rng=rng)[0],
        "M15": df,
        "H1": simulate_bars(1_200, 3600, rng=rng)[0],
    }
    scal = cs.frequency_scaling(frames)
    scal["rms_curl_bp"] = scal["rms_curl"] * 1e4
    scal["rms_return_bp"] = scal["rms_return"] * 1e4
    print(
        scal[["timeframe", "n_bars", "rms_curl_bp", "rms_return_bp", "ratio"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print(
        "\nRMS curl FLAT in bar size, RMS return ~ sqrt(D). On the real feed:"
        "\n  flat            -> curl is timing noise (the expected, boring, correct result)"
        "\n  rising with D   -> real persistent dislocation OR a stale feed; separate the"
        "\n                     two with the autocorrelation check."
    )

    _report("C2. Curl autocorrelation (M15) — stale-feed detector")
    print(cs.curl_autocorrelation(df, lags=5).round(4).to_string())

    _report("C3. Shift placebo — how much misalignment would explain the curl?")
    for pair in (("EUR", "USD"), ("GBP", "JPY")):
        print(f"  shifting {cs.symbol(*pair)} by 1 bar inflates RMS curl x{cs.shift_placebo(df, pair):.1f}")
    print(
        "\nOne bar of misalignment is enormous relative to a few seconds of staleness."
        "\nIf your observed curl is anywhere near this scale, you have an alignment bug,"
        "\nnot a discovery."
    )

    # --------------------------------------------------------- D. real dislocation
    _report("D. Injected GENUINE dislocation vs pure asynchronicity")
    # the episode sits OUTSIDE the [0:70%] calibration slice on purpose: an episode
    # inside the training window inflates the fitted denominator and silently deflates
    # the whole index (verified — it makes 'quiet' drift down with dislocation size).
    mask = np.zeros(N_BARS_M15, dtype=bool)
    mask[4_800:5_200] = True
    print(f"{'size':>8} {'stress quiet':>13} {'stress episode':>15} {'AUC':>7} {'infeas episode':>15} {'infeas base':>12}")
    for size_bp in (2.0, 5.0, 15.0, 50.0):
        dis, _ = simulate_bars(
            N_BARS_M15,
            900,
            rng=np.random.default_rng(RNG_SEED + 1),
            dislocated_pair=("EUR", "GBP"),
            dislocation_log=size_bp * 1e-4,
            dislocation_mask=mask,
        )
        st = cs.stress_index(dis, train_mask=train)["stress_smooth"].to_numpy()
        fin = np.isfinite(st)
        q, l = st[fin & ~mask], st[fin & mask]
        ranks = pd.Series(st[fin]).rank().to_numpy()
        lab = mask[fin]
        auc = (ranks[lab].mean() - (lab.sum() + 1) / 2) / (~lab).sum()
        infeas = np.column_stack(
            [cs.triangle_infeasibility(dis, t, buffer_log=1e-5) for t in cs.CYCLE_BASIS]
        ).max(axis=1)
        print(
            f"{size_bp:6.1f}bp {np.nanmean(q):13.3f} {np.nanmean(l):15.3f} {auc:7.3f} "
            f"{100 * (infeas[mask] > 0).mean():14.1f}% {100 * (infeas[~mask] > 0).mean():11.1f}%"
        )
    print(
        "\nAUC = separability of stress episode from baseline using the stress index alone."
        "\nThe High/Low certificate is deliberately conservative -- at M15 the bar range is"
        "\nmuch wider than a few bp, so it only fires for large events. It is an M1 tool."
    )


if __name__ == "__main__":
    main()
