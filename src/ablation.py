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

# Multiple-comparisons bookkeeping. Every feature ever ablated is one hypothesis
# spent against the same data; testing enough of them guarantees a naive-0.05
# "winner" by chance. `feature_hypothesis_log.csv` is the running family count
# (seeded retroactively with the 4 already-tested features), and every KEEP
# decision from now on must clear a Bonferroni-corrected bar, not a flat 0.05.
HYPOTHESIS_LOG = 'results/feature_hypothesis_log.csv'
HYPOTHESIS_LOG_COLUMNS = [
    'n', 'date', 'feature', 'arbiter', 'point_delta_acc',
    'ci95_dacc_low', 'ci95_dacc_high', 'mcnemar_p',
    'alpha_bonferroni', 'cleared_bar', 'verdict', 'notes',
]
FAMILY_ALPHA = 0.05


def _hyp_path(base_dir=''):
    return os.path.join(base_dir, HYPOTHESIS_LOG) if base_dir else HYPOTHESIS_LOG


def load_hypothesis_log(base_dir=''):
    """The running family of feature hypotheses (empty frame if none yet)."""
    p = _hyp_path(base_dir)
    if not os.path.exists(p):
        return pd.DataFrame(columns=HYPOTHESIS_LOG_COLUMNS)
    return pd.read_csv(p)


def bonferroni_alpha(family_size, family_alpha=FAMILY_ALPHA):
    """The corrected per-hypothesis bar: family_alpha / family_size. Guards
    against family_size 0 (returns the uncorrected bar)."""
    return family_alpha / max(1, family_size)


def register_hypothesis(result, base_dir='', notes=''):
    """Append a NEWLY tested feature to the hypothesis log and return its row.
    The recorded Bonferroni bar reflects the family size AFTER this addition, so
    the log always shows the exact bar the decision was judged against. A feature
    already in the log is not double-counted (idempotent by feature name)."""
    log = load_hypothesis_log(base_dir)
    if result['feature'] in set(log['feature']):
        return log
    n = len(log) + 1
    row = {
        'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
        'feature': result['feature'], 'arbiter': result['arbiter'],
        'point_delta_acc': result['point_delta_acc'],
        'ci95_dacc_low': result['ci95_dacc_low'], 'ci95_dacc_high': result['ci95_dacc_high'],
        'mcnemar_p': result['mcnemar_p'],
        'alpha_bonferroni': round(bonferroni_alpha(n), 4),
        'cleared_bar': result['mcnemar_p'] < bonferroni_alpha(n)
                       and (result['ci95_dacc_low'] > 0 or result['ci95_dacc_high'] < 0),
        'verdict': result['verdict'], 'notes': notes,
    }
    new = pd.DataFrame([row], columns=HYPOTHESIS_LOG_COLUMNS)
    out = new if log.empty else pd.concat([log, new], ignore_index=True)
    out.to_csv(_hyp_path(base_dir), index=False)
    return out


def _canonical_split(n: int, train_fraction: float, val_fraction: float) -> dict:
    """The three chronological boundaries shared with `_train_pipeline.py`.
    `train_end` is the 70% mark (end of the ablation-fit block); `val_end` is
    the 80% mark (end of the validation arbiter block, start of the untouched
    test block)."""
    train_end = int(n * (train_fraction - val_fraction))   # 70%
    val_end = int(n * train_fraction)                       # 80%
    return {"train_end": train_end, "val_end": val_end, "n": n}


def build_matrix(config: dict, base_dir: str = "", extra_feature_columns=None):
    """Engineer the full feature matrix once and return everything the ablation
    loop needs: the PCA-reduced frame, the canonical model input columns, the
    split boundaries, and the direction target. PCA is fit on the 70% train
    block only, so nothing the validation arbiter sees has leaked into the fit.
    Offline-safe: `fetch_macro_features` falls back to the cached FRED CSVs.

    `extra_feature_columns` appends candidate columns (currently the FOMC
    calendar trio, src/fomc_calendar.py) to the model input set for an
    ADDITION hypothesis test — the row set is identical with or without them,
    so the comparison isolates the columns themselves."""
    def _p(rel):
        return os.path.join(base_dir, rel) if base_dir else rel

    ohlcv = load_history(_p(config['data']['history_csv_path']))
    macro, _ = fetch_macro_features(
        ohlcv.index.min(), ohlcv.index.max(), config['macro'], base_dir=base_dir
    )
    feat = add_advanced_features(merge_macro_features(ohlcv.copy(), macro))

    extra = list(extra_feature_columns or [])
    if extra:
        from src.fomc_calendar import add_fomc_features, FOMC_FEATURE_COLUMNS
        from src.cot_data import add_cot_features, COT_FEATURE_COLUMNS
        from src.fibonacci_fractals import add_fibonacci_features, FIBONACCI_FEATURE_COLUMNS
        from src.vix_features import add_vix_features, VIX_FEATURE_COLUMNS
        if any(c in FOMC_FEATURE_COLUMNS for c in extra):
            feat = add_fomc_features(feat, base_dir=base_dir)
        if any(c in COT_FEATURE_COLUMNS for c in extra):
            # Availability-date as-of ffill onto the modeled rows, neutral 0
            # before COT exists / z-score warm-up (see src/cot_data.py).
            feat = add_cot_features(feat, base_dir=base_dir, config=config)
        if any(c in FIBONACCI_FEATURE_COLUMNS for c in extra):
            # Confirmed-fractal swing geometry from OHLC alone (no feed). Every
            # column is neutral 0 until a confirmed structure exists, so the
            # row set is identical with or without them (see src/fibonacci_fractals.py).
            feat = add_fibonacci_features(feat)
        if any(c in VIX_FEATURE_COLUMNS for c in extra):
            # VIX regime z-score + shock, computed on native cadence and joined
            # by availability date (print + 1 business day, STEP 0), neutral 0
            # before warm-up / when unreachable (see src/vix_features.py).
            feat = add_vix_features(feat, base_dir=base_dir, config=config)
        assert not feat[extra].isna().any().any(), \
            "extra candidate columns must be fully defined on the modeled rows"

    split = _canonical_split(
        len(feat), config['split']['train_fraction'], config['split']['val_fraction']
    )
    lag_scaler, lag_pca = fit_lag_pca(
        feat.iloc[:split['train_end']], lag_columns=LAG_COLUMNS,
        variance_threshold=config['pca']['variance_threshold'],
    )
    red = apply_lag_pca(feat, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    cols_full = model_input_columns(lag_pca, base_columns=FEATURE_COLUMNS + extra,
                                    lag_columns=LAG_COLUMNS)
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
                     n_boot=BOOTSTRAP_RESAMPLES, alpha=0.05, prob_w=None, pred_w=None,
                     y_val=None, feature_name=None):
    """Ablate one feature on the VALIDATION slice with bootstrap CI + McNemar.

    WITH = full model; WITHOUT = drop exactly `feature` (rows held identical, so
    the delta isolates that feature). `feature` may be a single column name or
    a LIST of columns tested as one bundled hypothesis (e.g. the FOMC calendar
    trio — several views of one underlying fact spend one Bonferroni slot);
    `feature_name` labels a bundle in the result/log. Returns a result dict;
    `alpha` is the significance bar the verdict is judged against (a flat 0.05
    here in Step 1; Step 2 passes a Bonferroni-corrected value). The full-model
    predictions can be passed in (`prob_w`/`pred_w`/`y_val`) to avoid refitting
    it per feature.
    """
    drop = [feature] if isinstance(feature, str) else list(feature)
    rng = np.random.default_rng(random_state)
    if prob_w is None:
        prob_w, pred_w, y_val = _fit_predict_val(red, cols_full, split, random_state)

    cols_wo = [c for c in cols_full if c not in set(drop)]
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
        "feature": feature_name or (feature if isinstance(feature, str) else '+'.join(feature)),
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
        register=False, random_state=None):
    """Ablate `features` (default: the standard 4) on the validation slice and
    write `out_csv`. The significance bar is the Bonferroni-corrected
    `0.05 / family_size`, where family_size counts every feature hypothesis ever
    tested (`feature_hypothesis_log.csv`) plus any genuinely new feature in this
    run. The header prints that corrected bar so it can never be silently
    forgotten. `register=True` appends new features to the hypothesis log."""
    features = features or STANDARD_FEATURES
    with open(os.path.join(base_dir, config_path) if base_dir else config_path) as f:
        config = json.load(f)
    random_state = config['random_state'] if random_state is None else random_state

    # Bonferroni bar: family = everything already logged, unioned with whatever
    # this run tests. A re-run of already-logged features leaves the family size
    # unchanged; a genuinely new feature grows it and tightens the bar for all.
    log = load_hypothesis_log(base_dir)
    already = set(log['feature'])
    new_feats = [f for f in features if f not in already]
    family_size = len(already | set(features))
    alpha = bonferroni_alpha(family_size)

    red, cols_full, split = build_matrix(config, base_dir=base_dir)
    prob_w, pred_w, y_val = _fit_predict_val(red, cols_full, split, random_state)
    acc_w = accuracy_score(y_val, pred_w)
    auc_w = roc_auc_score(y_val, prob_w)

    print("=" * 78)
    print("FEATURE ABLATION — arbiter = VALIDATION slice [70%:80%] (test block NOT touched)")
    print(f"  full model on validation: acc={acc_w:.4f}  auc={auc_w:.4f}  "
          f"(n_val={len(y_val):,} rows)")
    print(f"  hypotheses tested to date: {len(already)}  |  this run adds "
          f"{len(new_feats)} new  ->  family size {family_size}")
    print(f"  BONFERRONI-CORRECTED BAR: alpha = {FAMILY_ALPHA} / {family_size} "
          f"= {alpha:.4g}  (a KEEP must clear THIS, not a flat 0.05)")
    print("=" * 78)

    rows = []
    for feat in features:
        res = evaluate_feature(feat, red, cols_full, split, random_state=random_state,
                               alpha=alpha, prob_w=prob_w, pred_w=pred_w, y_val=y_val)
        rows.append(res)
        if register and feat in new_feats:
            register_hypothesis(res, base_dir=base_dir)
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


def run_addition_test(name, columns, config_path='config.json', base_dir='',
                      register=True, random_state=None, note_suffix='',
                      out_csv='results/feature_addition_validation.csv'):
    """Test ADDING a bundled column block to the current full feature set as
    ONE hypothesis (one Bonferroni slot for the whole bundle — its columns are
    several views of one underlying fact, e.g. the FOMC calendar trio).

    WITH = FEATURE_COLUMNS + bundle, WITHOUT = the current full set, identical
    rows, judged on the validation arbiter with the same paired bootstrap +
    exact McNemar used for the macro features. The family bar counts every
    hypothesis in `feature_hypothesis_log.csv` at RUN TIME plus this one."""
    with open(os.path.join(base_dir, config_path) if base_dir else config_path) as f:
        config = json.load(f)
    random_state = config['random_state'] if random_state is None else random_state

    log = load_hypothesis_log(base_dir)
    family_size = len(set(log['feature']) | {name})
    alpha = bonferroni_alpha(family_size)

    red, cols_full, split = build_matrix(config, base_dir=base_dir,
                                         extra_feature_columns=list(columns))
    prob_w, pred_w, y_val = _fit_predict_val(red, cols_full, split, random_state)

    print("=" * 78)
    print(f"FEATURE ADDITION TEST — {name} {list(columns)} as ONE bundled hypothesis")
    print(f"  arbiter = VALIDATION slice [70%:80%] (test block NOT touched)")
    print(f"  hypotheses tested to date: {len(set(log['feature']))}  ->  family size "
          f"{family_size}  ->  BONFERRONI BAR alpha = {FAMILY_ALPHA} / {family_size} "
          f"= {alpha:.4g}")
    print("=" * 78)

    res = evaluate_feature(list(columns), red, cols_full, split,
                           random_state=random_state, alpha=alpha,
                           prob_w=prob_w, pred_w=pred_w, y_val=y_val,
                           feature_name=name)
    # ADD-test verdict wording: an addition that fails the bar is DROPPED
    # outright (never "KEEP-provisional" — that status is for features that
    # already shipped before the validation-arbiter methodology existed).
    if res['verdict'].startswith('KEEP ('):
        res['verdict'] = (f'KEEP — bundle clears the corrected bar '
                          f'(CI excludes 0 and McNemar p<{alpha:.4g})')
    else:
        res['verdict'] = ('DROP — addition indistinguishable from noise at the '
                          'corrected bar (do not add)')
    print(f"  point delta acc={res['point_delta_acc']:+.4f}  auc={res['point_delta_auc']:+.4f}")
    print(f"  95% CI d_acc=[{res['ci95_dacc_low']:+.4f}, {res['ci95_dacc_high']:+.4f}]  "
          f"frac(d_acc>0)={res['frac_dacc_positive']:.3f}")
    print(f"  95% CI d_auc=[{res['ci95_dauc_low']:+.4f}, {res['ci95_dauc_high']:+.4f}]")
    print(f"  McNemar b={res['mcnemar_b_0to1']} c={res['mcnemar_c_1to0']}  p={res['mcnemar_p']:.4f}")
    print(f"  VERDICT: {res['verdict']}")

    if register:
        notes = (f'ADD-test of bundled block {",".join(columns)} '
                 f'(one hypothesis for the whole bundle)')
        if note_suffix:
            notes = f'{notes} | {note_suffix}'
        register_hypothesis(res, base_dir=base_dir, notes=notes)
    out_path = os.path.join(base_dir, out_csv) if base_dir else out_csv
    pd.DataFrame([res]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return res


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ['fomc']:
        from src.fomc_calendar import FOMC_FEATURE_COLUMNS
        run_addition_test('fomc_calendar_block', FOMC_FEATURE_COLUMNS)
    elif sys.argv[1:2] == ['cot']:
        # The EUR + USD-index positioning pair is ONE bundled hypothesis (a
        # single positioning theme), one Bonferroni slot — same precedent as
        # the FOMC calendar trio and the policy-rate block.
        from src.cot_data import COT_FEATURE_COLUMNS
        run_addition_test('cot_positioning_block', COT_FEATURE_COLUMNS)
    elif sys.argv[1:2] == ['fib']:
        # Hypothesis #7: fractal breakout + Fibonacci retracement geometry as
        # ONE bundled hypothesis (three views of one swing-structure fact), same
        # convention as the FOMC calendar trio and the COT positioning pair.
        from src.fibonacci_fractals import FIBONACCI_FEATURE_COLUMNS
        run_addition_test(
            'fibonacci_retracement_block', FIBONACCI_FEATURE_COLUMNS,
            note_suffix=('CONTINGENCY: hypothesis #8 (dist_to_nearest_fib_extension_pct, '
                         '3-point extension/projection) is BUILT but runs ONLY if this '
                         '#7 bundle clears its bar; if #7 is DROP, #8 is not tested at all.'))
    elif sys.argv[1:2] == ['vix']:
        # Hypothesis #8: VIX regime z-score + day-over-day shock as ONE bundled
        # hypothesis (two views of one equity-risk-sentiment fact), same
        # convention as the fibonacci / COT / FOMC blocks.
        from src.vix_features import VIX_FEATURE_COLUMNS
        run_addition_test(
            'vix_regime_block', VIX_FEATURE_COLUMNS,
            note_suffix=('VIX (VIXCLS) via shared FRED framework; conservative D-1 '
                         'availability (print + 1 business day, verified in STEP 0: '
                         'FRED publishes VIXCLS with a business-day lag). z-score on '
                         'native business-day cadence, window 756 / min 252.'))
    else:
        run(features=sys.argv[1:] or None)
