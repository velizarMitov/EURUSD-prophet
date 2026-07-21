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


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


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
    n = len(analysis)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train = analysis.iloc[:train_end]
    val = analysis.iloc[train_end:val_end]          # the arbiter
    # analysis.iloc[val_end:] is the reserved TEST block — deliberately untouched.

    print("=" * 78)
    print("COT WEEKLY-HORIZON SIDE-CHECK — own family (NOT the daily / volatility bars)")
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
    # Own family, idempotent by hypothesis name (upsert so a re-run refreshes the row).
    if os.path.exists(out_path):
        log = pd.read_csv(out_path)
        log = log[log["hypothesis"] != row["hypothesis"]]
        out = pd.concat([log, pd.DataFrame([row], columns=WEEKLY_LOG_COLUMNS)], ignore_index=True)
        out["n"] = range(1, len(out) + 1)
    else:
        out = pd.DataFrame([row], columns=WEEKLY_LOG_COLUMNS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nLogged: {out_path}")
    return row


if __name__ == "__main__":
    run()
