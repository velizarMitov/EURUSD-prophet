"""
Calibration audit of the PRODUCTION direction head — research-only diagnostic.

WHY THIS FILE EXISTS
--------------------
`PredictionService.compute_consensus` gates live output on
`CONFIDENCE_THRESHOLD = 0.52`, where each head's confidence is `max(p, 1-p)` of
a sigmoid/`predict_proba` output. That number is the only quantity in the
serving path that claims to express *how sure the system is*, and the project now
trades real money against it. So its calibration is a risk-management question,
not an academic one: if `p = 0.58` does not actually mean "58% of the time", the
guard is filtering on noise and the dashboard is advertising a belief the model
does not hold.

WHAT THIS IS NOT
----------------
* NOT a hypothesis test. It spends no alpha and must never be entered in
  `results/feature_hypothesis_log.csv`. Nothing here selects a feature, a model
  or a threshold — it measures an artifact that is already frozen and deployed.
* NOT a retraining path. Every number comes from `.predict()` / `.predict_proba()`
  on the artifacts already in `models/<variant>/`. No estimator is fit here; the
  only fitted objects touched are the ones loaded off disk.
* NOT a writer. By default this module writes nothing at all. `--out` is opt-in
  and lands in `results/calibration_audit/`, never in `models/`.

There is NO sigma head in production to audit. `_train_pipeline.py:388-391`
builds exactly two heads — `return_output` (Dense, linear) and `direction_output`
(Dense, sigmoid). The Gaussian `sigma` head whose anti-calibration was recorded
in `results/spiking_readout_hypothesis_log.csv` row 2 belongs to the research
substrate in `src/spiking_readout.py`, which never entered the serving path.

READ `ARCHITECTURE_DOCS.md` SS4.2.2 BEFORE INTERPRETING ANY OUTPUT. Probability
calibration was measured once before (GBM `predict_proba` + `CalibratedClassifierCV`
on the test block) and deliberately not adopted. This module differs from that
work in three ways that matter: it covers BOTH heads and the consensus rather
than the GBM alone, it is per-variant (SS4.2.2 predates the dual-variant split),
and it reports the reliability structure rather than a single Brier number.

THE CONTAMINATION WARNING, STATED UP FRONT
------------------------------------------
The requested arbiter is validation[70:80]. For this particular audit that slice
is NOT clean, and the direction of the bias is optimistic:

  * the GBM trains on [0:80%] -- validation[70:80] is INSIDE its training range,
    so its probabilities there are in-sample;
  * the LSTM trains on [0:70%] but early-stops on [70:80] with
    `restore_best_weights`, so that slice chose its epoch.

Neither head is honestly out-of-sample on [70:80]. `audit()` therefore reports
both slices: `validation` as specified, and `test` [80:100] as the out-of-sample
control. Reading the test block here does not violate the Production Methodology
rule, because that rule forbids using it as a SEARCH knob -- adding or judging a
feature by its test-block score. Nothing is being searched or selected here; this
is precisely the "one-shot final report on a frozen model" use the rule carves
out. If these two slices disagree, the test block is the one to believe.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Same bound _train_pipeline.py applies, and for the same reason: neither fredapi
# nor yfinance accepts a timeout, and a socket that connects but never answers
# would hang this diagnostic forever behind an `except Exception` that cannot see it.
NETWORK_SOCKET_TIMEOUT_S = 60

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The live gate, imported rather than repeated -- if production ever moves the
# threshold this audit follows it instead of silently measuring the old one.
from .inference import PredictionService  # noqa: E402
from .features import (  # noqa: E402
    LAG_COLUMNS,
    TARGET_DIRECTION_COLUMN,
    add_advanced_features,
    apply_lag_pca,
    merge_macro_features,
    model_input_columns,
)

CONFIDENCE_THRESHOLD = PredictionService.CONFIDENCE_THRESHOLD
N_BINS = 10


# ---------------------------------------------------------------------------
# Scoring primitives (pure functions -- unit-tested in tests/test_calibration_audit.py)
# ---------------------------------------------------------------------------
def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Bin predicted P(up) into `n_bins` EQUAL-COUNT bins and compare the bin's
    mean prediction against its realised hit rate.

    Equal-count (quantile) bins rather than equal-width: this target's
    probabilities pile up in a narrow band around 0.5 (SS4.2.2 measured a raw
    range of [0.333, 0.672]), and equal-width bins would leave most of them
    empty while hiding all the structure inside one central bar.

    `duplicates='drop'` is load-bearing rather than defensive: a head whose
    output is nearly constant produces repeated bin edges, and the honest
    response is fewer bins, not a crash.
    """
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "p": p, "y": y})
    table = frame.groupby("bin").agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        realised=("y", "mean"),
        p_low=("p", "min"),
        p_high=("p", "max"),
    )
    table["gap"] = table["mean_predicted"] - table["realised"]
    return table.reset_index(drop=True)


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Murphy (1973) decomposition: Brier = calibration - resolution + uncertainty.

    * calibration (a.k.a. reliability): squared distance between what each bin
      PREDICTED and what actually happened in it. Lower is better; 0 = perfectly
      calibrated.
    * resolution: how far each bin's realised rate departs from the overall base
      rate -- i.e. whether the model separates anything at all. Higher is better.
    * uncertainty: base_rate * (1 - base_rate), a property of the TARGET alone.
      No model can move it; it is the score a constant base-rate predictor gets.

    "Refinement" in the calibration/refinement split is `uncertainty - resolution`,
    reported alongside so both conventional framings are readable off one dict.

    `residual` is the identity check. It is returned rather than asserted because
    it is only exactly zero when the bins used here are the bins the decomposition
    is defined over; a non-trivial residual means the binning distorted the split
    and the components should not be quoted.
    """
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "p": p, "y": y})
    grouped = frame.groupby("bin").agg(n=("y", "size"), pbar=("p", "mean"), obar=("y", "mean"))

    n = len(p)
    base = float(y.mean())
    w = grouped["n"].to_numpy() / n
    calibration = float((w * (grouped["pbar"] - grouped["obar"]) ** 2).sum())
    resolution = float((w * (grouped["obar"] - base) ** 2).sum())
    uncertainty = float(base * (1.0 - base))
    brier = float(np.mean((p - y) ** 2))

    return {
        "brier": brier,
        "calibration": calibration,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "refinement": uncertainty - resolution,
        "base_rate": base,
        "mean_predicted": float(p.mean()),
        # Brier of the trivial "always predict the TRAIN base rate" rule is added
        # by the caller, which knows the train slice; this one is the in-slice
        # constant, i.e. the best any constant could do here.
        "brier_constant_best": uncertainty,
        "residual": brier - (calibration - resolution + uncertainty),
        "n_bins_used": int(grouped.shape[0]),
    }


def confidence_correctness_spearman(p: np.ndarray, y: np.ndarray) -> dict:
    """Spearman(|p - 0.5|, correctness): does the system's own confidence rank
    its own correctness?

    This is the question the 0.52 guard implicitly answers YES to. It is a rank
    correlation on purpose -- the guard only ever uses the ORDER of confidences
    (is it above the bar), never their arithmetic, so a monotone-but-miscalibrated
    confidence would still be a usable filter. A rho indistinguishable from 0
    means the guard is thresholding noise.
    """
    from scipy.stats import spearmanr

    conf = np.abs(p - 0.5)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    if conf.std() == 0 or correct.std() == 0:
        return {"rho": float("nan"), "pvalue": float("nan"), "accuracy": float(correct.mean())}
    res = spearmanr(conf, correct)
    return {
        "rho": float(res.statistic),
        "pvalue": float(res.pvalue),
        "accuracy": float(correct.mean()),
    }


def roc_auc_safe(p: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC, or NaN when the slice is single-class (which happens on the
    thresholded subsets and is information, not an error)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


@dataclass
class HeadReport:
    """Everything measured for one (variant, head, slice) triple."""

    variant: str
    head: str
    slice_name: str
    subset: str            # "all" | ">=0.52"
    n: int
    scores: dict = field(default_factory=dict)
    spearman: dict = field(default_factory=dict)
    auc: float = float("nan")
    reliability: pd.DataFrame | None = None
    coverage: float = float("nan")   # share of the parent slice this subset kept


# ---------------------------------------------------------------------------
# Data + artifacts
# ---------------------------------------------------------------------------
def build_engineered_frame(config: dict, base_dir: str = BASE_DIR, verbose: bool = True):
    """Rebuild the EXACT engineered row set `_train_pipeline.py` trains on
    (SS1 -> SS1B -> SS2), and return it with the shared chronological boundaries.

    This mirrors the training script rather than importing from it because
    importing `_train_pipeline` executes a full training run at module level.
    The mirrored block is four calls long and is asserted against the artifacts'
    own expected feature count in `load_frozen_variant`, so a drift shows up as a
    load error instead of a silently misaligned matrix.
    """
    socket.setdefaulttimeout(NETWORK_SOCKET_TIMEOUT_S)
    from .macro_data import fetch_macro_features

    raw = pd.read_csv(
        os.path.join(base_dir, config["data"]["history_csv_path"]),
        index_col="time",
        parse_dates=True,
    )[["open", "high", "low", "close", "tick_volume"]]

    macro_df, macro_sources = fetch_macro_features(
        raw.index.min(), raw.index.max(), config.get("macro", {})
    )
    raw = merge_macro_features(
        raw, macro_df if macro_df is not None else pd.DataFrame(index=raw.index)
    )
    engineered = add_advanced_features(raw)

    n = len(engineered)
    train_frac = config["split"]["train_fraction"]
    val_frac = config["split"]["val_fraction"]
    train_end = int(n * train_frac)                      # 80% -- test starts here
    lstm_train_end = int(n * (train_frac - val_frac))    # 70% -- validation starts here

    if verbose:
        print(f"Engineered rows: {n:,}  ({engineered.index[0].date()} -> {engineered.index[-1].date()})")
        print(f"  macro feed sources : {macro_sources}")
        print(f"  train    [0:{lstm_train_end}]        {engineered.index[0].date()} -> "
              f"{engineered.index[lstm_train_end - 1].date()}")
        print(f"  validation [{lstm_train_end}:{train_end}]   {engineered.index[lstm_train_end].date()} -> "
              f"{engineered.index[train_end - 1].date()}   <- requested arbiter")
        print(f"  test     [{train_end}:{n}]   {engineered.index[train_end].date()} -> "
              f"{engineered.index[-1].date()}   <- out-of-sample control")

    return engineered, {"n": n, "lstm_train_end": lstm_train_end, "train_end": train_end}


def load_frozen_variant(name: str, base_dir: str = BASE_DIR) -> dict:
    """Load one variant's frozen artifacts THROUGH THE PRODUCTION LOADER.

    `PredictionService.__init__` also pulls live market data, the H1 ensemble,
    the volatility ensemble and the TI family -- none of which this audit needs,
    and the live fetch would make the run non-deterministic. So the class is
    allocated without running `__init__` and only `_load_variant_artifacts` is
    invoked. That keeps the audit bound to the real loading logic (including the
    XGBoost `device='cpu'` reset) instead of a copy that could drift from it.
    """
    svc = PredictionService.__new__(PredictionService)
    svc.load_errors = []
    v = svc._load_variant_artifacts(os.path.join(base_dir, "models", name), name)
    v["load_errors"] = svc.load_errors
    return v


def variant_probabilities(v: dict, engineered: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Batch-run one variant's two direction heads over EVERY engineered row.

    The transform chain is production's, in production's order:
      select the variant's feature columns -> apply THAT variant's lag PCA ->
      reindex to `model_input_columns` -> that variant's global_scaler.

    The LSTM windows follow the SERVING convention (`_predict_lstm` takes
    `tail(time_steps)` of a continuous history), not the training convention
    (which re-windows inside each block and so discards the first `time_steps`
    rows of every block). A validation-slice window may therefore reach back
    into training rows -- which is exactly what happens live, every day.
    """
    window = engineered[v["feature_columns"]]
    reduced = apply_lag_pca(window, v["lag_scaler"], v["lag_pca"], lag_columns=LAG_COLUMNS)
    cols = model_input_columns(v["lag_pca"], base_columns=v["feature_columns"], lag_columns=LAG_COLUMNS)
    x_scaled = v["global_scaler"].transform(reduced[cols])

    out = pd.DataFrame(index=engineered.index)
    out["p_gbm"] = np.nan
    out["p_lstm"] = np.nan

    if v["gbm_ready"]:
        out["p_gbm"] = v["gbm_classifier"].predict_proba(x_scaled)[:, 1]

    if v["lstm_ready"]:
        steps = int(v["lstm_time_steps"])
        ends = np.arange(steps - 1, len(x_scaled))
        idx = ends[:, None] - (steps - 1) + np.arange(steps)[None, :]
        seqs = x_scaled[idx]
        if verbose:
            print(f"  LSTM batch: {seqs.shape[0]:,} windows of {steps} steps x {seqs.shape[2]} features")
        _, prob_up = v["lstm_model"].predict(seqs, verbose=0, batch_size=512)
        out.iloc[ends, out.columns.get_loc("p_lstm")] = prob_up.ravel()

    return out


def consensus_frame(p_gbm: np.ndarray, p_lstm: np.ndarray) -> pd.DataFrame:
    """Vectorised mirror of `PredictionService.compute_consensus` for two heads.

    Reproduced rather than called because the production method consumes one
    dict per prediction, and this audit needs ~7k rows. The logic is copied
    branch for branch, INCLUDING the asymmetry worth noticing: the 0.52 guard is
    applied ONLY on the unanimous branch. When the heads disagree, production
    takes the more confident head's direction and emits it with NO threshold
    check -- so a disagreement at confidence 0.505 still leaves the service as a
    directional call. `gated` below marks the subset that actually cleared the
    guard; `emitted_direction` marks everything the service would state a
    direction for.
    """
    conf_gbm = np.where(p_gbm >= 0.5, p_gbm, 1.0 - p_gbm)
    conf_lstm = np.where(p_lstm >= 0.5, p_lstm, 1.0 - p_lstm)
    up_gbm = p_gbm >= 0.5
    up_lstm = p_lstm >= 0.5
    agree = up_gbm == up_lstm

    mean_conf = 0.5 * (conf_gbm + conf_lstm)
    gbm_more_confident = conf_gbm >= conf_lstm
    best_conf = np.where(gbm_more_confident, conf_gbm, conf_lstm)
    best_up = np.where(gbm_more_confident, up_gbm, up_lstm)

    confidence = np.where(agree, mean_conf, best_conf)
    up = np.where(agree, up_gbm, best_up)
    downgraded = agree & (mean_conf < CONFIDENCE_THRESHOLD)

    return pd.DataFrame(
        {
            "p": np.where(up, confidence, 1.0 - confidence),   # signed back to P(up)
            "confidence": confidence,
            "up": up,
            "agree": agree,
            "downgraded": downgraded,                          # "MIXED / LOW CONFIDENCE"
            "gated": agree & ~downgraded,                      # cleared the live guard
            "emitted_direction": ~downgraded,                  # service states a direction
        }
    )


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------
def score_head(p: np.ndarray, y: np.ndarray, *, variant: str, head: str,
               slice_name: str, n_bins: int = N_BINS) -> list[HeadReport]:
    """Score one head on one slice, twice: on everything, and on the subset
    whose confidence clears the live 0.52 bar (the only rows production acts on)."""
    reports = []
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]

    for subset, mask in (("all", np.ones(len(p), bool)),
                         (f">={CONFIDENCE_THRESHOLD}", np.abs(p - 0.5) >= CONFIDENCE_THRESHOLD - 0.5)):
        ps, ys = p[mask], y[mask]
        if len(ps) < 2 * n_bins:
            reports.append(HeadReport(variant, head, slice_name, subset, len(ps),
                                      coverage=float(mask.mean())))
            continue
        reports.append(
            HeadReport(
                variant=variant, head=head, slice_name=slice_name, subset=subset, n=len(ps),
                scores=brier_decomposition(ps, ys, n_bins),
                spearman=confidence_correctness_spearman(ps, ys),
                auc=roc_auc_safe(ps, ys),
                reliability=reliability_table(ps, ys, n_bins),
                coverage=float(mask.mean()),
            )
        )
    return reports


def audit(config: dict, base_dir: str = BASE_DIR, n_bins: int = N_BINS, verbose: bool = True):
    """Run the full audit: every variant x {gbm, lstm, consensus} x
    {validation[70:80], test[80:100]} x {all, >=0.52}."""
    engineered, bounds = build_engineered_frame(config, base_dir, verbose=verbose)
    y_all = engineered[TARGET_DIRECTION_COLUMN].to_numpy().astype(int)

    slices = {
        "validation[70:80]": slice(bounds["lstm_train_end"], bounds["train_end"]),
        "test[80:100]": slice(bounds["train_end"], bounds["n"]),
    }
    train_base_rate = float(y_all[: bounds["lstm_train_end"]].mean())

    reports: list[HeadReport] = []
    probabilities = {}
    for name in config.get("variants", ["baseline", "with_macro"]):
        if verbose:
            print(f"\n--- loading frozen artifacts: models/{name}/ ---")
        v = load_frozen_variant(name, base_dir)
        if v["load_errors"]:
            print(f"  LOAD ERRORS: {v['load_errors']}")
        if not (v["pca_ready"] and v["scaler_ready"]):
            print(f"  SKIPPING '{name}' -- no usable transform chain.")
            continue

        probs = variant_probabilities(v, engineered, verbose=verbose)
        cons = consensus_frame(probs["p_gbm"].to_numpy(), probs["p_lstm"].to_numpy())
        probs["p_consensus"] = cons["p"].to_numpy()
        probs["gated"] = cons["gated"].to_numpy()
        probs["agree"] = cons["agree"].to_numpy()
        probs["downgraded"] = cons["downgraded"].to_numpy()
        probabilities[name] = probs

        for slice_name, sl in slices.items():
            y = y_all[sl]
            for head, col in (("gbm", "p_gbm"), ("lstm", "p_lstm"), ("consensus", "p_consensus")):
                reports += score_head(probs[col].to_numpy()[sl], y, variant=name,
                                      head=head, slice_name=slice_name, n_bins=n_bins)

    return {
        "reports": reports,
        "probabilities": probabilities,
        "engineered": engineered,
        "bounds": bounds,
        "y": y_all,
        "train_base_rate": train_base_rate,
        "slices": slices,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x, nd=5):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def print_report(result: dict, n_bins: int = N_BINS) -> None:
    reports = result["reports"]
    train_base = result["train_base_rate"]

    print("\n" + "=" * 100)
    print("RELIABILITY TABLES  (equal-count bins of predicted P(up); 'realised' = share that went up)")
    print("=" * 100)
    for r in reports:
        if r.reliability is None or r.subset != "all":
            continue
        print(f"\n[{r.variant} / {r.head} / {r.slice_name}]  n={r.n}")
        t = r.reliability
        print(f"  {'bin':>3} {'range':>17} {'n':>5} {'mean pred':>10} {'realised':>10} {'gap':>9}")
        for i, row in t.iterrows():
            print(f"  {i + 1:>3} [{row.p_low:.4f},{row.p_high:.4f}] {int(row.n):>5} "
                  f"{row.mean_predicted:>10.4f} {row.realised:>10.4f} {row.gap:>+9.4f}")

    print("\n" + "=" * 100)
    print("BRIER SCORE + MURPHY DECOMPOSITION   (Brier = calibration - resolution + uncertainty)")
    print(f"train-slice base rate = {train_base:.4f}  ->  a constant 'always predict the train base rate' "
          f"predictor is the honest floor")
    print("=" * 100)
    hdr = (f"{'variant':<11} {'head':<10} {'slice':<18} {'subset':<7} {'n':>5} {'cover':>6} "
           f"{'Brier':>8} {'calib':>8} {'resol':>8} {'uncert':>8} {'refine':>8} {'resid':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        if not r.scores:
            print(f"{r.variant:<11} {r.head:<10} {r.slice_name:<18} {r.subset:<7} {r.n:>5} "
                  f"{r.coverage:>6.1%}   (too few rows to bin)")
            continue
        s = r.scores
        print(f"{r.variant:<11} {r.head:<10} {r.slice_name:<18} {r.subset:<7} {r.n:>5} "
              f"{r.coverage:>6.1%} {s['brier']:>8.5f} {s['calibration']:>8.5f} {s['resolution']:>8.5f} "
              f"{s['uncertainty']:>8.5f} {s['refinement']:>8.5f} {s['residual']:>+9.1e}")

    print("\n  Brier of the trivial constant predictor, per slice (train base rate applied out-of-sample):")
    for slice_name in result["slices"]:
        for r in reports:
            if r.slice_name == slice_name and r.subset == "all" and r.head == "gbm" and r.scores:
                y_slice_base = r.scores["base_rate"]
                const = train_base ** 2 * (1 - y_slice_base) + (1 - train_base) ** 2 * y_slice_base
                print(f"    {slice_name:<18} base rate {y_slice_base:.4f}   "
                      f"Brier(const={train_base:.4f}) = {const:.5f}")
                break

    print("\n" + "=" * 100)
    print("SPEARMAN(|p - 0.5|, correct)   -- does higher confidence actually mean more correct?")
    print("=" * 100)
    hdr = (f"{'variant':<11} {'head':<10} {'slice':<18} {'subset':<7} {'n':>5} "
           f"{'rho':>9} {'p-value':>9} {'accuracy':>9} {'ROC-AUC':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        if not r.spearman:
            continue
        print(f"{r.variant:<11} {r.head:<10} {r.slice_name:<18} {r.subset:<7} {r.n:>5} "
              f"{_fmt(r.spearman['rho'], 4):>9} {_fmt(r.spearman['pvalue'], 4):>9} "
              f"{r.spearman['accuracy']:>9.4f} {_fmt(r.auc, 4):>9}")

    print("\n" + "=" * 100)
    print("THE LIVE GATE ITSELF  -- how the 0.52 guard partitions each slice")
    print("=" * 100)
    for name, probs in result["probabilities"].items():
        for slice_name, sl in result["slices"].items():
            sub = probs.iloc[sl]
            y = result["y"][sl]
            ok = np.isfinite(sub["p_gbm"]) & np.isfinite(sub["p_lstm"])
            sub, y = sub[ok.to_numpy()], y[ok.to_numpy()]
            gated = sub["gated"].to_numpy()
            up = sub["p_consensus"].to_numpy() >= 0.5
            acc_gated = float((up[gated] == y[gated]).mean()) if gated.any() else float("nan")
            acc_rest = float((up[~gated] == y[~gated]).mean()) if (~gated).any() else float("nan")
            print(f"  [{name} / {slice_name}]  n={len(sub)}  heads agree {sub['agree'].mean():.1%}  "
                  f"downgraded to MIXED {sub['downgraded'].mean():.1%}  cleared the guard {gated.mean():.1%}")
            print(f"      accuracy WHEN GATED = {_fmt(acc_gated, 4)}   "
                  f"accuracy otherwise = {_fmt(acc_rest, 4)}   "
                  f"delta = {_fmt(acc_gated - acc_rest, 4)}")


def write_csvs(result: dict, out_dir: str) -> None:
    """Opt-in only (`--out`). Writes under results/, never models/."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for r in result["reports"]:
        row = {"variant": r.variant, "head": r.head, "slice": r.slice_name,
               "subset": r.subset, "n": r.n, "coverage": r.coverage, "auc": r.auc}
        row.update({k: v for k, v in r.scores.items()})
        row.update({f"spearman_{k}": v for k, v in r.spearman.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)

    rel = []
    for r in result["reports"]:
        if r.reliability is None:
            continue
        t = r.reliability.copy()
        t.insert(0, "subset", r.subset)
        t.insert(0, "slice", r.slice_name)
        t.insert(0, "head", r.head)
        t.insert(0, "variant", r.variant)
        rel.append(t)
    if rel:
        pd.concat(rel).to_csv(os.path.join(out_dir, "reliability_bins.csv"), index=False)
    print(f"\nWrote {out_dir}/calibration_summary.csv and reliability_bins.csv")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--bins", type=int, default=N_BINS)
    parser.add_argument("--out", default=None,
                        help="optional directory for CSVs (default: write nothing)")
    args = parser.parse_args(argv)

    with open(os.path.join(BASE_DIR, "config.json")) as f:
        config = json.load(f)

    print("=" * 100)
    print("PRODUCTION DIRECTION-HEAD CALIBRATION AUDIT")
    print(f"live gate: CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD} (imported from PredictionService)")
    print("diagnostic only -- spends no alpha, fits nothing, writes nothing unless --out is passed")
    print("=" * 100)

    result = audit(config, n_bins=args.bins)
    print_report(result, n_bins=args.bins)
    if args.out:
        write_csvs(result, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
