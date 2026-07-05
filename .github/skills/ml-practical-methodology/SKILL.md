---
name: ml-practical-methodology
description: 'ML Methodology Auditor grounded in Goodfellow/Bengio/Courville "Deep Learning" Ch.7.7 (Multi-Task Learning) and Ch.11 (Practical Methodology). Use when: deciding whether a coin-flip / near-chance metric (ROC-AUC ~0.50) reflects a real Bayes-error floor vs a fixable bug, deciding whether to gather more data vs tune hyperparameters vs change algorithm, auditing whether the shared LSTM trunk (multi-task return+direction heads) is actually justified, debugging a model that "sort of works" but you cannot tell if it is a software defect or a genuine capacity/data limit, choosing which performance metric to optimize before touching code, inspecting the worst individual mispredictions instead of only aggregate metrics. Triggers: practical methodology, train vs test gap, Bayes error, is this a bug or underfitting, multi-task learning, shared trunk, worst mistakes, default baseline, gather more data, performance metric selection, debug ML pipeline, coin flip accuracy, efficient market.'
argument-hint: 'Describe the diagnostic question (e.g. "is 0.50 ROC-AUC a bug or the real ceiling", "should I gather more data or tune more", "is the shared LSTM trunk justified")'
---

# Practical Methodology & Multi-Task Justification

## Role
You are an ML Methodology Auditor. Your job is not to make the model better by guessing — it is to run the diagnostic loop from Goodfellow et al., *Deep Learning* (MIT Press), Chapter 11 ("Practical Methodology") **before** anyone touches a hyperparameter, and to check whether the multi-task architecture (Ch. 7.7) this project already uses is actually earning its keep.

This project (`eurusdprophet`) already sits at an unusual point: `ARCHITECTURE_DOCS.md §4.2.1` documents that the GBM regressor is *statistically indistinguishable from "always predict the historical mean"* on the held-out test block (MAE 0.2959% vs baseline 0.2958%), and both direction heads sit at ROC-AUC ≈ 0.50. Before spending effort "fixing" this, use the book's own design process to determine whether that is even a fixable thing.

## When to Use
- Someone asks "why is this only 50/50" or wants to "improve accuracy" without first framing what improving it would even mean
- Deciding whether the next experiment should be: gather more data, change capacity, add regularization, or accept the current ceiling
- Reviewing whether `shared_lstm_trunk` (the `LSTM(units=64)` shared between `return_output` and `direction_output` in `_train_pipeline.py`) is justified, or whether it should be split into two independent single-task models
- Debugging a pipeline change that "seems to work" but you can't tell if a metric improvement is real signal or a measurement bug
- Before adding a new feature/model, deciding what single metric will judge whether it helped

---

## Part A — The Design Loop (Ch. 11, pp. 421–422)

The book's four-step loop, applied to this repo's actual files:

1. **Determine the goal metric first.** For this project that is already decided and documented (`direction_accuracy` / `direction_roc_auc`, `return_mae` — see `ARCHITECTURE_DOCS.md §4.2`). Any new experiment (new feature, new model, FRED ablation) must report *the same* metric on the *same* held-out `[80%:100%]` split, or it cannot be compared to `results/comparison_table.csv`.
2. **Establish an end-to-end pipeline before optimizing any single piece.** Already true here — don't lose it. Any change under consideration (e.g. dropping `yield_differential`, adding a new feature) should be run through the *whole* `_train_pipeline.py`, not evaluated in isolation in a notebook cell.
3. **Instrument to find *which* component underperforms** — see Part C (debugging strategies) below.
4. **Make one incremental change at a time**, driven by what step 3 found — not by trying five things at once and hoping.

### The "gather more data vs. change the algorithm" decision tree (Ch. 11.3, pp. 425–426)

```
Compute train-set metric (NOT just val/test).
Is train_roc_auc ALSO ~0.50 (same as test)?
├── YES → small train/test gap → this is NOT overfitting.
│         Per the book: "if both train and test error are high... the model
│         may be underfitting due to fundamental algorithmic reasons" —
│         OR the ceiling is close to Bayes error (irreducible noise).
│         Gathering more daily EURUSD history will NOT help — a random walk
│         does not become less random with more rows. The only lever left
│         is genuinely new, informative FEATURES (order flow, positioning,
│         intraday microstructure — see the h1_features.py module), not more
│         data, not a bigger model, not more tuning.
└── NO  → train_roc_auc is meaningfully higher than test_roc_auc
          → THIS is classic overfitting/variance. Standard fixes apply:
          increase regularization, reduce capacity, add dropout, or (per
          the book) gather more data — see model-tuning-bias-variance skill.
```

**Action item for this repo:** `_train_pipeline.py` currently logs test-set `direction_accuracy` / `direction_roc_auc` to MLflow but does **not** compute or log the equivalent train-set metric. Without it, you cannot actually run the decision tree above — you cannot tell from the current MLflow runs whether ROC-AUC ≈ 0.50 is "the market is efficient" (train ≈ test ≈ 0.50, matches `ARCHITECTURE_DOCS.md §4.2.1`) or "the model isn't even fitting the training data" (train also ≈ 0.50 for a *different* reason — e.g. a feature leak zeroing out signal, or a scaler bug). Add one line per model:

```python
train_roc_auc = roc_auc_score(y_train, clf.predict_proba(X_train_scaled)[:, 1])
mlflow.log_metric("train_direction_roc_auc", train_roc_auc)
```

If `train_roc_auc` also comes back ≈ 0.50, that is strong independent confirmation of the "efficient market" conclusion already written up in the architecture docs — worth adding to `§4.2.1` as corroborating evidence. If it comes back noticeably higher (e.g. 0.65+) while test stays at 0.50, that reopens the overfitting hypothesis and points back to `model-tuning-bias-variance`.

---

## Part B — Is the Multi-Task LSTM Trunk Actually Justified? (Ch. 7.7, pp. 244–245)

The book's own framing of multi-task learning:

> "Multi-task learning... is a way to improve generalization by pooling the examples... arising out of several tasks... when part of a model is shared across tasks, that part of the model is more constrained towards good values (**assuming the sharing is justified**), often yielding better generalization." — Ch. 7.7

> "The underlying prior belief is... among the factors that explain the variations observed in the data associated with the different tasks, **some are shared** across two or more tasks."

This project's `lstm_multitask_eurusd.keras` is exactly the architecture in the book's Figure 7.2: a `shared_lstm_trunk` feeding two task-specific heads, `return_output` (regression) and `direction_output` (classification). That architectural choice is only earning the generalization benefit the book describes **if the factors that predict tomorrow's return magnitude and tomorrow's direction actually overlap**. That is an empirical, checkable assumption — not something to take on faith because it's a popular pattern.

**Concrete check before trusting the shared trunk:**

```python
# On the held-out test block, do sign(predicted_return) and direction_output
# actually agree more often than chance? If the two heads are frequently
# INCONSISTENT with each other, the "shared factors" assumption is weak and
# the shared trunk may be constraining the return head with gradients that
# don't help it (or vice versa).
sign_agreement = (np.sign(y_pred_return) == (y_pred_direction_prob > 0.5).astype(int) * 2 - 1).mean()
print(f"Return-sign vs direction-head agreement: {sign_agreement:.3f}")
```

If this agreement rate is barely above 50%, the two heads are not actually pulling from a shared signal — consider training two single-task models instead (a plain regressor + a plain classifier) and comparing against the multi-task model on the *same* test block, exactly as the book frames it: multi-task learning is a hypothesis to test, not a default to assume. `compute_consensus`'s `CONFIDENCE_THRESHOLD = 0.52` guard in `src/inference.py` is already a tacit admission that the direction head is near chance — this check tells you whether the *architecture*, not just the *market*, might be part of why.

---

## Part C — Debugging Strategies (Ch. 11.5, pp. 436–439)

Two of the book's debugging techniques map directly onto tools this project already has but doesn't fully use:

### 1. "Visualize the worst mistakes" (p. 437)

> "By viewing the training set examples that are the hardest to model correctly, one can often discover problems with the way the data has been preprocessed or labeled."

This project already logs every live prediction to `results/prediction_log.csv` and scores it against realized closes in `/history` (`src/tracking.py`). That log is the raw material for this technique but nothing currently sorts it by error magnitude. Add a quick pass:

```python
import pandas as pd

log = pd.read_csv("results/prediction_log.csv", parse_dates=["forecasting_date"])
log["abs_error"] = (log["actual_return_pct"] - log["predicted_return_pct"]).abs()
worst = log.sort_values("abs_error", ascending=False).head(20)
print(worst[["forecasting_date", "data_source", "predicted_return_pct", "actual_return_pct", "abs_error"]])
```

Look for a *pattern* in `worst` (e.g. clustering right after weekends/holidays where `yield_differential` was stale per `ARCHITECTURE_DOCS.md §4.5`, or clustering on `history_fallback`/`+history_backfill` days where the live feed was thin). A systematic pattern here is a data/pipeline problem you can fix directly; random scatter across sources and dates is further evidence of genuine irreducible noise, not a bug.

### 2. Reasoning from train vs. test error to tell "bug" from "limit" (p. 437)

> "If training error is low but test error is high... the model is overfitting... If **both** train and test error are high, then it is difficult to determine whether there is a software defect or whether the model is underfitting due to fundamental algorithmic reasons. This scenario requires further tests" — namely, **fit a tiny dataset** (p. 438): a correct implementation should be able to drive training error to ~0 on a handful of examples; if it can't even memorize 5 rows, that's a software defect, not a market-efficiency story.

```python
# Sanity check: can the GBM/LSTM at least memorize a trivial slice?
X_tiny, y_tiny = X_train[:5], y_train[:5]
model.fit(X_tiny, y_tiny)
print("Tiny-fit train error:", mean_absolute_error(y_tiny, model.predict(X_tiny)))
# Expect ~0. If not, there is a bug in the training loop/loss/scaling,
# not "the market is efficient".
```

Run this once whenever a change to `_train_pipeline.py`, `src/features.py`, or the LSTM architecture is suspected of silently breaking something — it's a 5-line, 2-second test that rules out an entire class of "is my code broken" questions before you spend hours on hyperparameters.

---

## Anti-Patterns This Skill Guards Against

| Anti-Pattern | Why it's wrong here | Book reference |
|---|---|---|
| Tuning hyperparameters harder to fix ROC-AUC ≈ 0.50 | If train ROC-AUC is *also* ≈ 0.50, no amount of tuning fixes a Bayes-error floor | Ch. 11.3 |
| Assuming the shared LSTM trunk automatically helps | Multi-task sharing only helps if tasks share explanatory factors — check, don't assume | Ch. 7.7 |
| Judging the pipeline only by aggregate MAE/ROC-AUC | Aggregate metrics can hide a fixable systematic error pattern | Ch. 11.5 |
| Concluding "the market is efficient" without a train-set control | The same symptom (train≈test≈chance) can also mean a silent bug (e.g. scaler/leak zeroing signal) | Ch. 11.5, "fit a tiny dataset" |
| Changing five things at once after a bad metric | Makes it impossible to attribute the outcome to any one change | Ch. 11, intro |
