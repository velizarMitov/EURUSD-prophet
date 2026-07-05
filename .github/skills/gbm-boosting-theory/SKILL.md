---
name: gbm-boosting-theory
description: 'Gradient Boosting Theorist grounded in Hastie/Tibshirani/Friedman "The Elements of Statistical Learning" Ch.10 (Boosting and Additive Trees). Use when: tuning the production GradientBoostingClassifier/Regressor in _train_pipeline.py, deciding the huber_alpha / learning_rate / subsample tradeoff, adding stochastic gradient boosting, extracting or interpreting feature importances from the GBM, explaining why loss=huber was chosen for the regressor, diagnosing whether the GBM param_grid is actually well-regularized. Triggers: gradient boosting, GBM, shrinkage, learning rate, subsample, stochastic gradient boosting, huber loss, huber_alpha, feature importance, partial dependence, boosting regularization, MART, param_grid.'
argument-hint: 'Describe the boosting task (e.g. "should I add subsample to the GBM grid", "explain huber_alpha=0.9", "extract feature importance from best_gbm_eurusd")'
---

# Gradient Boosting Theory (ESL Ch. 10)

## Role
You are a Gradient Boosting Theorist. Every recommendation about `best_gbm_eurusd.pkl` / `best_gbm_regressor_eurusd.pkl` is grounded in Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2nd ed.), Chapter 10 — the book that formalizes the exact algorithm `sklearn.ensemble.GradientBoostingClassifier/Regressor` implements. This complements `model-tuning-bias-variance` (generic CV/search mechanics) and `deep-learning-timeseries` (the LSTM side) — this skill is specifically about what the boosting math implies for `config.json`'s `gbm` block and `_train_pipeline.py`.

## When to Use
- Explaining or revisiting `config.json → gbm.huber_alpha` / `param_grid`
- Deciding whether to add stochastic subsampling to the GBM
- Extracting and interpreting feature importance from the trained GBM (currently not done anywhere in the repo)
- Understanding why `loss='huber'` was chosen over squared error for the return regressor
- Reviewing whether the current `learning_rate` grid is actually in the regime the theory recommends

---

## 1. Why `loss='huber'` for the return regressor (ESL §10.6, pp. 349–350)

> "On finite samples squared-error loss places much more emphasis on observations with large absolute residuals... it is thus far less robust, and its performance severely degrades for long-tailed error distributions and especially for grossly mis-measured y-values ('outliers')... One such criterion is the Huber loss criterion."

Daily EUR/USD returns are exactly this kind of long-tailed target (occasional large jumps around macro releases, otherwise near-zero noise) — the book's own justification for Huber over squared error is precisely the "grossly mis-measured / outlier-prone target" scenario. This is a solid citation to put in `ARCHITECTURE_DOCS.md §3.1` next to the current `huber_alpha=0.9` mention — right now the docs state *what* loss is used but not *why*, from first principles.

**One nuance worth knowing:** ESL's Huber loss (eq. 10.23) uses a fixed threshold `δ`. sklearn's `GradientBoostingRegressor(loss='huber', alpha=0.9)` instead sets `δ` **adaptively at each boosting iteration** to the `alpha`-quantile (here, the 90th percentile) of the current absolute residuals. So `huber_alpha=0.9` does not mean "delta = 0.9" — it means "let the top 10% largest-residual points be treated as outliers (linear penalty) at each iteration, and the inner 90% as squared-error." That is worth stating explicitly wherever `huber_alpha` is documented, since the name invites the wrong mental model.

---

## 2. Shrinkage — is `learning_rate` actually in the right regime? (ESL §10.12.1, pp. 364–365)

> "The best strategy appears to be to set ν to be very small (ν < 0.1) and then choose M by early stopping... Smaller values of ν lead to larger values of M for the same training risk, so there is a tradeoff."

Current `config.json`:
```json
"param_grid": { "n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5] }
```
Two of the three `learning_rate` candidates (0.01, 0.05) are already in the book's recommended `ν < 0.1` regime — good. But per the shrinkage/iteration tradeoff, a small `ν` needs a **correspondingly larger `M` (`n_estimators`)** to reach the same training risk; capping `n_estimators` at `[100, 200]` may not give `ν=0.01` enough iterations to be competitive with `ν=0.1`, which could bias `GridSearchCV` toward picking the larger learning rate simply because the smaller one wasn't given enough trees to converge. Worth widening the grid, e.g. `"n_estimators": [100, 200, 500, 1000]`, so each `learning_rate` is judged at its own natural scale rather than a shared iteration budget.

---

## 3. Missing: stochastic subsampling (ESL §10.12.2, pp. 365–366)

> "With stochastic gradient boosting, at each iteration we sample a fraction η of the training observations (without replacement)... Not only does the sampling reduce the computing time... it actually produces a more accurate model... It appears that subsampling without shrinkage does poorly [combine both]."

`_train_pipeline.py`'s `param_grid` (checked directly — grep confirms no `subsample` key anywhere in the file) never sets `GradientBoostingClassifier`/`Regressor`'s `subsample` parameter, so it silently defaults to `1.0` (no subsampling) for both production heads. Per the book this leaves free accuracy and free compute time on the table, precisely because shrinkage (`learning_rate < 0.1`) is *already* in place — the combination is what the book found best, not shrinkage alone. Concrete change:

```python
# config.json → gbm.param_grid
"param_grid": {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.5, 0.8, 1.0]   # ESL §10.12.2 — new axis
}
```
This is a 3x expansion of the grid (compute cost), so if `GridSearchCV` runtime is already a concern, `RandomizedSearchCV` or `Optuna` (see `model-tuning-bias-variance`) is the better search strategy once this axis is added rather than exhaustive grid search.

---

## 4. Feature importance is computed by neither model — a real gap (ESL §10.13.1, pp. 367–368)

> "In data mining applications the input predictor variables are seldom equally relevant. Often only a few of them have substantial influence on the response... It is often useful to learn the relative importance... of each input variable."

`best_gbm_eurusd.pkl` / `best_gbm_regressor_eurusd.pkl` expose `.feature_importances_` (the sklearn implementation of ESL eq. 10.43 — importance averaged over all `M` trees) for free, but nothing in `_train_pipeline.py` or the notebook currently extracts or persists it. Given the FRED ablation already shows `yield_differential` is net-negative (`results/2C_fred_ablation.csv`), a full importance ranking across all 24 `FEATURE_COLUMNS` (or `model_input_columns` post-PCA) would show whether other features are similarly dead weight — directly actionable before the next retrain:

```python
import pandas as pd

importances = pd.Series(
    gbm_classifier.feature_importances_,
    index=model_input_columns  # src/features.py — exact post-PCA column order
).sort_values(ascending=False)

importances.to_csv("results/gbm_feature_importance.csv")
print(importances)
```

Per the book, because of shrinkage (§10.12.1) "the masking of important variables by others with which they are highly correlated is much less of a problem" for this averaged measure — so this ranking is trustworthy even with correlated inputs like the four SMA windows or the six lag/PCA columns, unlike a single-tree importance.

**Follow-up (ESL §10.13.2, Partial Dependence Plots):** for the top 3–4 important features, a partial dependence plot (`sklearn.inspection.PartialDependenceDisplay`) shows the *shape* of the relationship the GBM learned — useful to sanity-check that, e.g., `volatility_20` or `ATR_14` isn't driving predictions through some degenerate/monotonic artifact of the PCA-reduced lag block.

---

## Quick Reference

| Question | ESL section | Current repo state | Action |
|---|---|---|---|
| Why Huber, not squared error, for the return head? | §10.6 | Used (`huber_alpha=0.9`), undocumented rationale | Cite in `ARCHITECTURE_DOCS.md §3.1` |
| Is `learning_rate` small enough? | §10.12.1 | 0.01/0.05/0.1 — good range, but `n_estimators` may be too capped for the smallest rate | Widen `n_estimators` grid |
| Is subsampling used? | §10.12.2 | **Not set — defaults to 1.0** | Add `subsample` axis to `param_grid` |
| Which of the 24 features actually matter? | §10.13.1 | **Never extracted** | Add `feature_importances_` export after training |
| What's the shape of the top features' effect? | §10.13.2 | Not done | `PartialDependenceDisplay` on top-4 importances |
