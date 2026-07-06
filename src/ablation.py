"""
Feature-ablation harness — VALIDATION-ONLY arbiter (post-defense methodology).

Why this module exists
----------------------
Earlier feature KEEP/DROP decisions in this project (the original
`yield_differential` and the three macro features `usd_index_return`,
`policy_rate_differential`, `inflation_differential`) were scored on the FINAL
held-out TEST block `[80%:100%]`. Reusing that same block to *decide* which
features to keep is data-snooping: with enough features tried, some cross a
naive 0.05 bar by chance alone, not because they carry real signal. Now that
the system trades real money, a false-positive feature is live capital risk,
not a lost exam point.

This module fixes that permanently. Every KEEP/DROP decision is made on the
VALIDATION slice `[70%:80%]` only; the TEST block `[80%:100%]` is NEVER read
here. The test block stays a one-shot final report produced by
`_train_pipeline.py`, never a feature-search criterion.

Splits (from `config.json`, identical boundaries to `_train_pipeline.py`)
------------------------------------------------------------------------
    train  [0    : 70%]    PCA + scaler + model are fit HERE only
    val    [70%  : 80%]    ablation KEEP/DROP arbiter (this module)
    test   [80%  : 100%]   NEVER touched here

The PCA/scaler are fit on `[0:70%]` (train only) rather than the production
`[0:80%]`, so the validation slice is genuinely held out from everything the
ablation experiment fits — the val block is the arbiter, so it must not leak
into the fit.

Run standalone (offline-safe; uses the cached FRED CSVs if the API is down):

    python -m src.ablation                 # ablate the standard 4-feature set
    python -m src.ablation usd_index_return  # one feature

Writes `results/feature_ablation_validation.csv`.
"""
import json
import os
from math import comb

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.macro_data import fetch_macro_features
from src.features import (
    load_history, merge_macro_features, add_advanced_features,
    FEATURE_COLUMNS, LAG_COLUMNS, TARGET_DIRECTION_COLUMN,
    fit_lag_pca, apply_lag_pca, model_input_columns,
)

# The standard retroactive hypothesis set — the features whose KEEP decisions
# were originally (incorrectly) made on the test block and must be re-scored on
# validation. `yield_differential_delta` is the model-facing diff feature.
STANDARD_FEATURES = [
    'yield_differential_delta',
    'usd_index_return',
    'policy_rate_differential',
    'inflation_differential',
]

BOOTSTRAP_RESAMPLES = 2000
VALIDATION_CSV = 'results/feature_ablation_validation.csv'


def _canonical_split(n: int, train_fraction: float, val_fraction: float) -> dict:
    """The three chronological boundaries shared with `_train_pipeline.py`.
    `train_end` is the 70% mark (end of the ablation-fit block); `val_end` is
    the 80% mark (end of the validation arbiter block, start of the untouched
    test block)."""
    train_end = int(n * (train_fraction - val_fraction))   # 70%
    val_end = int(n * train_fraction)                       # 80%
    return {"train_end": train_end, "val_end": val_end, "n": n}


def build_matrix(config: dict, base_dir: str = ""):
    """Engineer the full feature matrix once and return everything the ablation
    loop needs: the PCA-reduced frame, the canonical model input columns, the
    split boundaries, and the direction target. PCA is fit on the 70% train
    block only, so nothing the validation arbiter sees has leaked into the fit.
    Offline-safe: `fetch_macro_features` falls back to the cached FRED CSVs."""
    def _p(rel):
        return os.path.join(base_dir, rel) if base_dir else rel

    ohlcv = load_history(_p(config['data']['history_csv_path']))
    macro, _ = fetch_macro_features(
        ohlcv.index.min(), ohlcv.index.max(), config['macro'], base_dir=base_dir
    )
    feat = add_advanced_features(merge_macro_features(ohlcv.copy(), macro))

    split = _canonical_split(
        len(feat), config['split']['train_fraction'], config['split']['val_fraction']
    )
    lag_scaler, lag_pca = fit_lag_pca(
        feat.iloc[:split['train_end']], lag_columns=LAG_COLUMNS,
        variance_threshold=config['pca']['variance_threshold'],
    )
    red = apply_lag_pca(feat, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    cols_full = model_input_columns(lag_pca, base_columns=FEATURE_COLUMNS, lag_columns=LAG_COLUMNS)
    return red, cols_full, split


def _fit_predict_val(red, cols, split, random_state):
    """Fit the quick-GBM on the 70% train block and predict on the VALIDATION
    slice `[70%:80%]`. Mirrors the quick-GBM used in the prior ablations
    (sklearn GBM, no grid search) so the numbers are comparable — the only
    change is the evaluation block (validation, not test)."""
    tr, va = slice(0, split['train_end']), slice(split['train_end'], split['val_end'])
    y = red[TARGET_DIRECTION_COLUMN].values
    X_tr, X_va = red[cols].iloc[tr], red[cols].iloc[va]
    sc = StandardScaler().fit(X_tr)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   random_state=random_state)
    m.fit(sc.transform(X_tr), y[tr])
    prob = m.predict_proba(sc.transform(X_va))[:, 1]
    return prob, (prob >= 0.5).astype(int), y[va]


def _mcnemar_exact(correct_wo, correct_w):
    """Two-sided exact-binomial McNemar on the discordant pairs of the WITHOUT
    vs WITH direction calls over the identical validation rows. `b` = WITHOUT
    wrong & WITH correct, `c` = WITHOUT correct & WITH wrong."""
    b = int(np.sum((~correct_wo) & correct_w))
    c = int(np.sum(correct_wo & (~correct_w)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * p)


def evaluate_feature(feature, red, cols_full, split, random_state=42,
                     n_boot=BOOTSTRAP_RESAMPLES, alpha=0.05, prob_w=None, pred_w=None, y_val=None):
    """Ablate one feature on the VALIDATION slice with bootstrap CI + McNemar.

    WITH = full model; WITHOUT = drop exactly `feature` (rows held identical, so
    the delta isolates that feature). Returns a result dict; `alpha` is the
    significance bar the verdict is judged against (a flat 0.05 here in Step 1;
    Step 2 passes a Bonferroni-corrected value). The full-model predictions can
    be passed in (`prob_w`/`pred_w`/`y_val`) to avoid refitting it per feature.
    """
    rng = np.random.default_rng(random_state)
    if prob_w is None:
        prob_w, pred_w, y_val = _fit_predict_val(red, cols_full, split, random_state)

    cols_wo = [c for c in cols_full if c != feature]
    prob_wo, pred_wo, _ = _fit_predict_val(red, cols_wo, split, random_state)

    correct_w = (pred_w == y_val)
    correct_wo = (pred_wo == y_val)
    n_val = len(y_val)

    point_dacc = accuracy_score(y_val, pred_w) - accuracy_score(y_val, pred_wo)
    point_dauc = roc_auc_score(y_val, prob_w) - roc_auc_score(y_val, prob_wo)

    d_acc = np.empty(n_boot)
    d_auc = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n_val, n_val)          # paired resample
        yt = y_val[idx]
        d_acc[i] = accuracy_score(yt, pred_w[idx]) - accuracy_score(yt, pred_wo[idx])
        if len(np.unique(yt)) == 2:
            d_auc[i] = roc_auc_score(yt, prob_w[idx]) - roc_auc_score(yt, prob_wo[idx])

    ci_acc = np.percentile(d_acc, [2.5, 97.5])
    ci_auc = np.nanpercentile(d_auc, [2.5, 97.5])
    frac_pos = float(np.mean(d_acc > 0))
    b, c, pval = _mcnemar_exact(correct_wo, correct_w)

    ci_excludes_zero = ci_acc[0] > 0 or ci_acc[1] < 0
    significant = ci_excludes_zero and pval < alpha
    if significant:
        verdict = f"KEEP (CI excludes 0 and McNemar p<{alpha:.4g})"
    else:
        verdict = "KEEP-provisional (not distinguishable from noise at the current bar)"

    return {
        "feature": feature,
        "arbiter": "validation[70:80]",
        "n_val": n_val,
        "point_delta_acc": round(point_dacc, 4),
        "ci95_dacc_low": round(ci_acc[0], 4),
        "ci95_dacc_high": round(ci_acc[1], 4),
        "frac_dacc_positive": round(frac_pos, 3),
        "point_delta_auc": round(point_dauc, 4),
        "ci95_dauc_low": round(ci_auc[0], 4),
        "ci95_dauc_high": round(ci_auc[1], 4),
        "mcnemar_b_0to1": b,
        "mcnemar_c_1to0": c,
        "mcnemar_p": round(pval, 4),
        "alpha_bar": alpha,
        "verdict": verdict,
    }


def run(features=None, config_path='config.json', base_dir='', out_csv=VALIDATION_CSV,
        alpha=0.05, random_state=None):
    """Ablate `features` (default: the standard 4) on the validation slice and
    write `out_csv`. Prints a report whose header states the arbiter block and
    the significance bar in force so it can never be silently forgotten."""
    features = features or STANDARD_FEATURES
    with open(os.path.join(base_dir, config_path) if base_dir else config_path) as f:
        config = json.load(f)
    random_state = config['random_state'] if random_state is None else random_state

    red, cols_full, split = build_matrix(config, base_dir=base_dir)
    prob_w, pred_w, y_val = _fit_predict_val(red, cols_full, split, random_state)
    acc_w = accuracy_score(y_val, pred_w)
    auc_w = roc_auc_score(y_val, prob_w)

    print("=" * 78)
    print("FEATURE ABLATION — arbiter = VALIDATION slice [70%:80%] (test block NOT touched)")
    print(f"  full model on validation: acc={acc_w:.4f}  auc={auc_w:.4f}  "
          f"(n_val={len(y_val):,} rows)")
    print(f"  significance bar in force: alpha={alpha:.4g}")
    print("=" * 78)

    rows = []
    for feat in features:
        res = evaluate_feature(feat, red, cols_full, split, random_state=random_state,
                               alpha=alpha, prob_w=prob_w, pred_w=pred_w, y_val=y_val)
        rows.append(res)
        print(f"\n=== {feat} ===")
        print(f"  point delta acc={res['point_delta_acc']:+.4f}  auc={res['point_delta_auc']:+.4f}")
        print(f"  95% CI d_acc=[{res['ci95_dacc_low']:+.4f}, {res['ci95_dacc_high']:+.4f}]  "
              f"frac(d_acc>0)={res['frac_dacc_positive']:.3f}")
        print(f"  McNemar b={res['mcnemar_b_0to1']} c={res['mcnemar_c_1to0']}  p={res['mcnemar_p']:.4f}")
        print(f"  VERDICT: {res['verdict']}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, out_csv) if base_dir else out_csv
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


if __name__ == "__main__":
    import sys
    feats = sys.argv[1:] or None
    run(features=feats)
