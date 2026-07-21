"""
COT positioning — WEEKLY-horizon exploratory side-check (its OWN hypothesis family).

Separate from the daily direction/return family (results/feature_hypothesis_log.csv)
and the volatility family (results/volatility_hypothesis_log.csv): a different
target HORIZON (forward WEEKLY return, not next-day), so it is NOT comparable to
either family's bar and is logged only in its own
results/cot_weekly_hypothesis_log.csv.

Research-only, regardless of outcome: no model, no variant, no serving change. A
KEEP-signal here means only "worth designing a proper weekly model later" — never
"wire COT into the daily API today".

PRE-REGISTERED (fixed before looking at any result; run ONCE, no iterating):
  Frame     daily close (results/eurusd_features.csv) resampled to weekly W-TUE
            bars (last close of each week), aligned to CFTC's Tuesday as-of cadence.
  Target    forward weekly log return  r_{t+1} = log(C_{t+1} / C_t);  direction = sign.
  Predictor cot_eur_zscore / cot_usdindex_zscore, joined by AVAILABILITY date via
            merge_asof(direction='backward') — each week carries the freshest COT
            reading already PUBLIC by that Tuesday's close; the SAME-week as_of
            report (published ~3 days later) is correctly excluded. Identical
            look-ahead discipline to add_cot_features, at weekly resolution.
  Split     chronological 70/80 at weekly resolution; fit only on train[0:70%];
            evaluate on validation[70%:80%]; test[80%:100%] reserved, NEVER read.
  Test      ONE pre-registered battery (report BOTH, never cherry-pick):
              PRIMARY       Spearman rho(each z-score, forward weekly return) on the
                            validation slice, 2000-resample bootstrap 95% CI.
                            Signal = CI excludes 0.
              CORROBORATING logistic(2 z-scores -> weekly direction) fit on train,
                            validation accuracy vs the train-majority-class baseline;
                            paired 2000-resample bootstrap 95% CI on (acc - baseline)
                            + exact McNemar.
  Decision  KEEP-signal ONLY if the PRIMARY (Spearman) 95% CI excludes 0 for at
            least one z-score; the logistic is corroborating context. If the
            PRIMARY is null the verdict is DROP even if the logistic squeaks past
            (this is the anti-cherry-pick rule). alpha = 0.05 (first test of this
            family; a future weekly-COT test becomes #2 and tightens the bar).
  Power     ~100 validation weeks -> only |rho| >~ 0.2 is detectable at all, so a
            null is WEAK evidence of absence, stated honestly in the output.

Run:  python -m src.cot_weekly_check
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from src.cot_data import fetch_cot_features, COT_FEATURE_COLUMNS

WEEKLY_LOG = "results/cot_weekly_hypothesis_log.csv"
WEEKLY_LOG_COLUMNS = [
    "n", "date", "hypothesis", "arbiter", "n_train", "n_val",
    "spearman_eur_rho", "spearman_eur_ci_low", "spearman_eur_ci_high",
    "spearman_usdidx_rho", "spearman_usdidx_ci_low", "spearman_usdidx_ci_high",
    "logit_val_acc", "majority_baseline_acc", "logit_delta_acc",
    "logit_delta_ci_low", "logit_delta_ci_high", "mcnemar_p",
    "alpha", "cleared_bar", "verdict", "notes",
]
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
HISTORY_CSV = "results/eurusd_features.csv"

# Hypothesis #2 (contrarian extremes) pre-registered constants — fixed a priori,
# NOT tuned against this data (see run_extremes docstring):
EXTREME_Z = 1.0             # |z|>1 ~ top/bottom 16% of a trailing-window z under normality
MIN_EXTREME_PER_SIDE = 5    # fewer than this on a side -> spread CI is not stable -> INCONCLUSIVE

# Shared family-log column groups so heterogeneous hypotheses (Spearman/logistic
# for #1, conditional-mean spread for #2) coexist as rows in ONE ledger with a
# stable, readable column order (union of each hypothesis's own metrics).
COMMON_LEAD = ["n", "date", "hypothesis", "arbiter", "n_train", "n_val"]
COMMON_TAIL = ["alpha", "cleared_bar", "verdict", "notes"]


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def _upsert_weekly_log(row, out_path):
    """Upsert one hypothesis row into the weekly family log by hypothesis name
    (re-run refreshes that row; other hypotheses' rows are preserved). Columns
    are the UNION across hypotheses — each row keeps only its own metrics, NaN
    elsewhere — ordered COMMON_LEAD + metrics + COMMON_TAIL. This is the single
    upsert path reused by both run() (#1) and run_extremes() (#2)."""
    new = pd.DataFrame([row])
    if os.path.exists(out_path):
        log = pd.read_csv(out_path)
        log = log[log["hypothesis"] != row["hypothesis"]]
        out = pd.concat([log, new], ignore_index=True)
    else:
        out = new
    out["n"] = range(1, len(out) + 1)
    mid = [c for c in out.columns if c not in COMMON_LEAD + COMMON_TAIL]
    ordered = [c for c in COMMON_LEAD if c in out.columns] + mid + \
              [c for c in COMMON_TAIL if c in out.columns]
    out = out[ordered]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def weekly_cot_target_frame(weekly_close: pd.Series, cot_frame: pd.DataFrame) -> pd.DataFrame:
    """Join availability-dated COT z-scores onto a weekly (W-TUE) close series and
    attach the forward weekly log return + direction. Look-ahead-safe by
    construction: merge_asof(direction='backward') gives each Tuesday week-close
    ONLY the last COT reading whose availability date <= that Tuesday, so the
    same-week as_of report (published ~3 days later) is excluded. Restricted to
    weeks with a genuine COT reading and a defined forward return. Isolated from
    I/O so it is unit-testable with synthetic inputs."""
    weekly = weekly_close.to_frame("close").sort_index()
    weekly.index = pd.DatetimeIndex(weekly.index)
    if weekly.index.tz is not None:
        weekly.index = weekly.index.tz_convert("UTC").tz_localize(None)

    cf = cot_frame.copy()
    ci = pd.DatetimeIndex(cf.index)
    cf.index = (ci.tz_convert("UTC").tz_localize(None) if ci.tz is not None else ci).normalize()
    cf = cf[[c for c in COT_FEATURE_COLUMNS if c in cf.columns]].sort_index()

    merged = pd.merge_asof(weekly, cf, left_index=True, right_index=True, direction="backward")
    merged["fwd_weekly_ret"] = np.log(merged["close"].shift(-1) / merged["close"])
    merged["direction"] = (merged["fwd_weekly_ret"] > 0).astype(int)
    return merged.dropna(subset=COT_FEATURE_COLUMNS + ["fwd_weekly_ret"]).copy()


def build_weekly_frame(base_dir="", history_csv=HISTORY_CSV):
    """Weekly analysis frame from the real daily history + fetched COT (offline-
    safe via cache). Thin I/O wrapper over weekly_cot_target_frame."""
    daily = pd.read_csv(_p(base_dir, history_csv), index_col="time", parse_dates=True)
    weekly_close = daily["close"].resample("W-TUE").last().dropna()

    cot_frame, source = fetch_cot_features(base_dir=base_dir)
    if cot_frame is None or cot_frame.empty:
        raise RuntimeError("COT frame unavailable (no API, no cache) — cannot run the weekly check.")
    return weekly_cot_target_frame(weekly_close, cot_frame), source


def _weekly_split(analysis):
    """The shared chronological 70/80 split at weekly resolution (same convention
    as every other family): fit only on train, arbiter = validation, test
    reserved and never read here. Returns (train, val, (n, train_end, val_end))."""
    n = len(analysis)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    return analysis.iloc[:train_end], analysis.iloc[train_end:val_end], (n, train_end, val_end)


def _spearman_bootstrap(x, y, n_boot=BOOTSTRAP_RESAMPLES, random_state=42):
    """Point Spearman rho + percentile 95% CI over paired bootstrap resamples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(random_state)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = spearmanr(x[idx], y[idx])[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def _mcnemar_exact(correct_a, correct_b):
    """Two-sided exact-binomial McNemar on the discordant pairs (baseline vs
    logistic) over the identical validation rows."""
    from math import comb
    b = int(np.sum((~correct_a) & correct_b))   # baseline wrong, logistic right
    c = int(np.sum(correct_a & (~correct_b)))   # baseline right, logistic wrong
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * p)


def _logistic_eval(train, val, n_boot=BOOTSTRAP_RESAMPLES, random_state=42):
    """Fit logistic(2 z-scores -> direction) on train, evaluate on val vs the
    train-majority-class baseline. Returns metrics + paired bootstrap CI of the
    accuracy delta + exact McNemar p."""
    Xtr = train[COT_FEATURE_COLUMNS].to_numpy()
    ytr = train["direction"].to_numpy()
    Xva = val[COT_FEATURE_COLUMNS].to_numpy()
    yva = val["direction"].to_numpy()

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    pred = clf.predict(Xva)
    maj_class = int(round(ytr.mean()))            # train majority class (no val peeking)
    base_pred = np.full(len(yva), maj_class)

    logit_correct = (pred == yva)
    base_correct = (base_pred == yva)
    acc = float(logit_correct.mean())
    base_acc = float(base_correct.mean())

    rng = np.random.default_rng(random_state)
    n = len(yva)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = logit_correct[idx].mean() - base_correct[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    _b, _c, mcp = _mcnemar_exact(base_correct, logit_correct)
    return {
        "logit_val_acc": round(acc, 4),
        "majority_baseline_acc": round(base_acc, 4),
        "logit_delta_acc": round(acc - base_acc, 4),
        "logit_delta_ci_low": round(float(lo), 4),
        "logit_delta_ci_high": round(float(hi), 4),
        "mcnemar_p": round(float(mcp), 4),
    }


def run(base_dir="", out_log=WEEKLY_LOG, random_state=42):
    analysis, source = build_weekly_frame(base_dir=base_dir)
    train, val, (n, train_end, val_end) = _weekly_split(analysis)
    # analysis.iloc[val_end:] is the reserved TEST block — deliberately untouched.

    print("=" * 78)
    print("COT WEEKLY-HORIZON SIDE-CHECK #1 — own family (NOT the daily / volatility bars)")
    print(f"  COT source: {source}  |  weekly analysis rows: {n:,} "
          f"({analysis.index.min().date()} -> {analysis.index.max().date()})")
    print(f"  split (weekly): train[0:{train_end}]  val[{train_end}:{val_end}]  "
          f"test[{val_end}:{n}] RESERVED")
    print(f"  validation weeks: {len(val)}  (LIMITED POWER: ~|rho|>0.2 detectable at all)")
    print(f"  alpha = {FAMILY_ALPHA} (first test of this family)")
    print("=" * 78)

    # ---- PRIMARY: Spearman of each z-score vs forward weekly return (val) -----
    yv = val["fwd_weekly_ret"].to_numpy()
    spear = {}
    print("\n--- PRIMARY: Spearman rho(z-score, forward weekly return) on validation ---")
    for col, tag in zip(COT_FEATURE_COLUMNS, ("eur", "usdidx")):
        rho, lo, hi = _spearman_bootstrap(val[col].to_numpy(), yv, random_state=random_state)
        excl0 = lo > 0 or hi < 0
        spear[tag] = (rho, lo, hi, excl0)
        print(f"  {col:22s}: rho={rho:+.4f}  95% CI[{lo:+.4f}, {hi:+.4f}]  "
              f"{'EXCLUDES 0' if excl0 else 'straddles 0'}")

    # ---- CORROBORATING: logistic direction vs majority baseline --------------
    logit = _logistic_eval(train, val, random_state=random_state)
    print("\n--- CORROBORATING: logistic(2 z-scores -> weekly direction) ---")
    print(f"  logistic val acc = {logit['logit_val_acc']:.4f}  |  "
          f"train-majority baseline = {logit['majority_baseline_acc']:.4f}  |  "
          f"delta = {logit['logit_delta_acc']:+.4f}")
    print(f"  delta 95% CI[{logit['logit_delta_ci_low']:+.4f}, {logit['logit_delta_ci_high']:+.4f}]  "
          f"McNemar p={logit['mcnemar_p']:.4f}")

    # ---- Decision (pre-registered: PRIMARY governs) --------------------------
    primary_signal = spear["eur"][3] or spear["usdidx"][3]
    if primary_signal:
        verdict = ("KEEP-signal (weekly) — a Spearman CI excludes 0; worth designing a "
                   "proper weekly model later (NOT wiring into the daily API)")
    else:
        verdict = ("DROP — no weekly COT edge: both Spearman CIs straddle 0 (logistic is "
                   "corroborating context only, not a second bite at significance)")
    print(f"\n  VERDICT: {verdict}")
    print(f"  (power caveat: {len(val)} validation weeks — a null is weak evidence of absence)")

    row = {
        "n": 1, "date": pd.Timestamp.utcnow().date().isoformat(),
        "hypothesis": "cot_weekly_positioning_vs_forward_return",
        "arbiter": "weekly_validation[70:80]",
        "n_train": len(train), "n_val": len(val),
        "spearman_eur_rho": round(spear["eur"][0], 4),
        "spearman_eur_ci_low": round(spear["eur"][1], 4),
        "spearman_eur_ci_high": round(spear["eur"][2], 4),
        "spearman_usdidx_rho": round(spear["usdidx"][0], 4),
        "spearman_usdidx_ci_low": round(spear["usdidx"][1], 4),
        "spearman_usdidx_ci_high": round(spear["usdidx"][2], 4),
        **logit,
        "alpha": FAMILY_ALPHA,
        "cleared_bar": bool(primary_signal),
        "verdict": verdict,
        "notes": ("exploratory weekly side-check; W-TUE close, forward weekly log return, "
                  "COT joined by availability date (merge_asof backward); test block reserved; "
                  "pre-registered Spearman-primary + logistic-corroborating, no iterating"),
    }
    out_path = _p(base_dir, out_log)
    _upsert_weekly_log(row, out_path)
    print(f"\nLogged: {out_path}")
    return row


def _binom_p(k, n):
    """Two-sided exact binomial p vs p0=0.5 (scipy binomtest, or the older
    binom_test on legacy scipy). NaN when there are no trials."""
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        from scipy.stats import binom_test
        return float(binom_test(k, n, 0.5, alternative="two-sided"))


def _extreme_contrarian_spread(z, y, n_boot=BOOTSTRAP_RESAMPLES, random_state=42):
    """PRIMARY hypothesis-#2 statistic for one z-score on the validation slice.

    spread = mean(y | z > +EXTREME_Z) - mean(y | z < -EXTREME_Z), where y is the
    forward weekly return. The contrarian hypothesis predicts spread < 0 (crowded
    -long weeks underperform crowded-short weeks). Paired bootstrap: resample the
    WHOLE validation slice with replacement and recompute BOTH conditional means
    each draw, so uncertainty in how MANY extreme weeks land in each tail is
    propagated. Draws with an empty tail yield NaN and are excluded (their share
    is reported as boot_degenerate_frac — a stability diagnostic)."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    long_mask = z > EXTREME_Z
    short_mask = z < -EXTREME_Z
    n_long, n_short = int(long_mask.sum()), int(short_mask.sum())
    point = (float(y[long_mask].mean() - y[short_mask].mean())
             if n_long and n_short else float("nan"))

    rng = np.random.default_rng(random_state)
    n = len(z)
    spreads = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        zb, yb = z[idx], y[idx]
        lb, sb = zb > EXTREME_Z, zb < -EXTREME_Z
        if lb.any() and sb.any():
            spreads[i] = yb[lb].mean() - yb[sb].mean()
    valid = spreads[~np.isnan(spreads)]
    degenerate = float(np.mean(np.isnan(spreads)))
    lo, hi = (np.percentile(valid, [2.5, 97.5]) if len(valid) >= 2
              else (float("nan"), float("nan")))

    # Corroborating exact binomial sign tests (context only).
    yl, ys = y[long_mask], y[short_mask]
    long_frac_neg = float((yl < 0).mean()) if n_long else float("nan")
    short_frac_pos = float((ys > 0).mean()) if n_short else float("nan")
    long_p = _binom_p(int((yl < 0).sum()), n_long)
    short_p = _binom_p(int((ys > 0).sum()), n_short)

    computable = n_long >= MIN_EXTREME_PER_SIDE and n_short >= MIN_EXTREME_PER_SIDE
    # KEEP-signal for THIS z-score: computable, expected sign, and the 95% CI
    # (the pre-registered decision interval) entirely below 0.
    signal = bool(computable and point == point and point < 0
                  and hi == hi and hi < 0)
    return {
        "n_long": n_long, "n_short": n_short,
        "point_spread": point, "ci_low": float(lo), "ci_high": float(hi),
        "boot_degenerate_frac": round(degenerate, 3),
        "long_frac_neg": long_frac_neg, "short_frac_pos": short_frac_pos,
        "long_sign_p": long_p, "short_sign_p": short_p,
        "computable": computable, "signal": signal,
    }


def run_extremes(base_dir="", out_log=WEEKLY_LOG, random_state=42):
    """Hypothesis #2 of the weekly COT family: positioning EXTREMES as a
    CONTRARIAN signal (mean reversion), NOT linear correlation. Same target
    horizon as #1 (forward weekly log return) -> SAME family -> this is #2 and
    tightens the family bar to alpha = 0.05/2 = 0.025. Logged only in
    results/cot_weekly_hypothesis_log.csv; the daily feature_hypothesis_log.csv
    and volatility_hypothesis_log.csv are NOT touched.

    PRE-REGISTERED (fixed before looking at any result; run ONCE, no iterating):
      Extreme definition — since cot_*_zscore is already a trailing-window
        z-score, "crowded long" = z > +1.0 and "crowded short" = z < -1.0
        (~top/bottom 16% under normality — a standard a priori cutoff, NOT chosen
        by scanning thresholds against this data).
      Contrarian hypothesis — crowded LONG predicts a NEGATIVE forward weekly
        return; crowded SHORT predicts a POSITIVE forward weekly return
        (mean reversion / unwind), for cot_eur_zscore and cot_usdindex_zscore
        independently.
      PRIMARY test (governs the verdict; bundles BOTH instruments into this ONE
        hypothesis entry, exactly as #1 bundled eur+usdindex) — on the validation
        slice, per z-score, spread = mean(fwd | z>+1.0) - mean(fwd | z<-1.0);
        contrarian signal requires spread < 0. Paired bootstrap 2000 resamples
        (resample validation weeks, recompute both conditional means each draw)
        -> 95% CI. KEEP-signal ONLY if the CI is entirely below 0 for at least
        one z-score AND the point spread has the expected (negative) sign — no
        cherry-picking a sign-flipped explanation after the fact.
      CORROBORATING (context only, never overrides the PRIMARY) — exact binomial
        sign test: among crowded-long weeks, fraction with negative fwd return
        vs 0.5; among crowded-short weeks, fraction with positive fwd return
        vs 0.5. Report p-values; do not let them flip the spread-CI decision.
      POWER / SAMPLE caveat — with ~99 validation weeks and ~16% tails, expect
        ~15-20 extreme weeks per side: VERY thin for a bootstrap CI. Treat any
        KEEP-signal as preliminary and any DROP as weak evidence of absence. Do
        NOT expand the threshold post-hoc; if a side has fewer than
        MIN_EXTREME_PER_SIDE (5) extreme weeks the spread CI is not stable and
        that z-score is reported INCONCLUSIVE/UNDERPOWERED, never re-tuned.
      Decision bar recorded as alpha = 0.025 (Bonferroni family accounting for
        the 2nd weekly test); the decision INTERVAL is the pre-registered 95% CI.
        A stricter 0.025-consistent (97.5%) CI would only RAISE the KEEP bar, so
        a null here is conservative.
    """
    analysis, source = build_weekly_frame(base_dir=base_dir)
    train, val, (n, train_end, val_end) = _weekly_split(analysis)
    # analysis.iloc[val_end:] is the reserved TEST block — deliberately untouched.
    alpha = round(FAMILY_ALPHA / 2, 4)   # #2 of the weekly family
    yv = val["fwd_weekly_ret"].to_numpy()

    print("=" * 78)
    print("COT WEEKLY-HORIZON SIDE-CHECK #2 — contrarian POSITIONING EXTREMES (same family)")
    print(f"  COT source: {source}  |  weekly analysis rows: {n:,}  |  validation weeks: {len(val)}")
    print(f"  extreme cutoff |z|>{EXTREME_Z} (pre-registered ~16% tails); "
          f"min {MIN_EXTREME_PER_SIDE}/side for a stable CI")
    print(f"  family bar alpha = {FAMILY_ALPHA}/2 = {alpha} (2nd weekly test); "
          f"decision interval = pre-registered 95% CI")
    print("=" * 78)

    res = {}
    for col, tag in zip(COT_FEATURE_COLUMNS, ("eur", "usdidx")):
        r = _extreme_contrarian_spread(val[col].to_numpy(), yv, random_state=random_state)
        res[tag] = r
        status = ("SIGNAL" if r["signal"] else
                  "inconclusive/underpowered" if not r["computable"] else "no signal")
        print(f"\n--- {col} (crowded-long n={r['n_long']}, crowded-short n={r['n_short']}) ---")
        if r["point_spread"] == r["point_spread"]:
            print(f"  spread = mean(fwd|long) - mean(fwd|short) = {r['point_spread']:+.6f}  "
                  f"95% CI[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]  "
                  f"(degenerate draws {r['boot_degenerate_frac']:.1%})")
        else:
            print("  spread undefined — a tail has 0 weeks on the validation slice")
        print(f"  contrarian sign check: crowded-long frac-negative={r['long_frac_neg']:.3f} "
              f"(p={r['long_sign_p']:.4f})  |  crowded-short frac-positive={r['short_frac_pos']:.3f} "
              f"(p={r['short_sign_p']:.4f})")
        print(f"  -> {status}")

    # Verdict (PRIMARY governs): KEEP if any z-score signals; INCONCLUSIVE if
    # neither is even computable; else DROP.
    any_signal = res["eur"]["signal"] or res["usdidx"]["signal"]
    any_computable = res["eur"]["computable"] or res["usdidx"]["computable"]
    if any_signal:
        verdict = ("KEEP-signal (weekly contrarian extremes) — a crowded-long-minus-short spread "
                   "95% CI is entirely below 0; PRELIMINARY given thin tails, worth a proper "
                   "weekly model later (NOT the daily API)")
    elif not any_computable:
        verdict = (f"INCONCLUSIVE / underpowered — fewer than {MIN_EXTREME_PER_SIDE} extreme weeks on "
                   "a tail for both z-scores; no stable CI (threshold NOT loosened post-hoc)")
    else:
        verdict = ("DROP — no contrarian weekly edge: no spread CI is entirely below 0 (sign tests "
                   "are corroborating context only, not a second decision point)")
    print(f"\n  VERDICT: {verdict}")

    def _r(x, nd=6):
        return round(float(x), nd) if x == x else x    # keep NaN as NaN
    row = {
        "date": pd.Timestamp.utcnow().date().isoformat(),
        "hypothesis": "cot_weekly_positioning_extremes_contrarian",
        "arbiter": "weekly_validation[70:80]",
        "n_train": len(train), "n_val": len(val),
        "spread_eur": _r(res["eur"]["point_spread"]),
        "spread_eur_ci_low": _r(res["eur"]["ci_low"]), "spread_eur_ci_high": _r(res["eur"]["ci_high"]),
        "spread_usdidx": _r(res["usdidx"]["point_spread"]),
        "spread_usdidx_ci_low": _r(res["usdidx"]["ci_low"]),
        "spread_usdidx_ci_high": _r(res["usdidx"]["ci_high"]),
        "n_eur_long": res["eur"]["n_long"], "n_eur_short": res["eur"]["n_short"],
        "n_usdidx_long": res["usdidx"]["n_long"], "n_usdidx_short": res["usdidx"]["n_short"],
        "eur_long_sign_p": _r(res["eur"]["long_sign_p"], 4),
        "eur_short_sign_p": _r(res["eur"]["short_sign_p"], 4),
        "usdidx_long_sign_p": _r(res["usdidx"]["long_sign_p"], 4),
        "usdidx_short_sign_p": _r(res["usdidx"]["short_sign_p"], 4),
        "alpha": alpha,
        "cleared_bar": bool(any_signal),
        "verdict": verdict,
        "notes": (f"exploratory weekly CONTRARIAN-EXTREMES side-check, hypothesis #2 of the weekly "
                  f"family (target horizon = forward weekly return, same as #1 -> alpha=0.05/2=0.025). "
                  f"Pre-registered |z|>{EXTREME_Z} crowded cutoff, spread=mean(fwd|long)-mean(fwd|short), "
                  f"95% decision CI, run once no iterating. Corroborating sign-test fractions: "
                  f"eur long-neg={res['eur']['long_frac_neg']:.3f}/short-pos={res['eur']['short_frac_pos']:.3f}, "
                  f"usdidx long-neg={res['usdidx']['long_frac_neg']:.3f}/short-pos={res['usdidx']['short_frac_pos']:.3f}; "
                  f"boot degenerate-draw frac eur={res['eur']['boot_degenerate_frac']:.3f}/"
                  f"usdidx={res['usdidx']['boot_degenerate_frac']:.3f}. Thin tails -> KEEP would be "
                  f"preliminary, DROP is weak evidence of absence. Research-only; test block reserved."),
    }
    out_path = _p(base_dir, out_log)
    _upsert_weekly_log(row, out_path)
    print(f"  (power caveat: thin extreme tails — see counts above; a null is weak evidence of absence)")
    print(f"\nLogged: {out_path}")
    return row


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["extremes"]:
        run_extremes()
    else:
        run()
