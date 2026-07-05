# Improvement Log

Source of truth for the backlog grounded in `.github/skills/ml-practical-methodology`
and `.github/skills/gbm-boosting-theory` (plus the other 5 project skills). Read this
file before starting new work in this repo — it records what's done, in progress, or
still open. One backlog item = one commit; do not batch items together.

## Backlog

- [x] Log train-set `direction_roc_auc`/`accuracy` in `_train_pipeline.py` (currently
      only test-set metrics are logged to MLflow). Needed to run the
      ml-practical-methodology decision tree: is ROC-AUC~0.50 a Bayes-error floor
      (train≈test) or a bug (train also low for a different reason)?
      **Done:** added `train_direction_accuracy`/`train_direction_roc_auc` to both
      the GBM (`X_gb_train_s`) and LSTM (`X_train_seq`) MLflow metric blocks, plus a
      printed train-test ROC-AUC gap. Next retrain will show whether the gap is near
      zero (efficient-market floor, matches `ARCHITECTURE_DOCS.md §4.2.1`) or large
      (real overfitting).
- [ ] Add a sign-agreement check between the LSTM's `return_output` and
      `direction_output` heads on the test block, to verify the shared multi-task
      trunk is actually justified (ml-practical-methodology Part B).
- [ ] Add `subsample` to `config.json`'s `gbm.param_grid` (currently defaults to 1.0 —
      stochastic gradient boosting per gbm-boosting-theory §3 is missing).
- [ ] Extract and persist GBM `feature_importances_` to
      `results/gbm_feature_importance.csv` after training (gbm-boosting-theory §4) —
      currently never computed anywhere.
- [ ] Add a "worst mistakes" pass over `results/prediction_log.csv` (sort by
      abs_error, inspect for systematic patterns vs random noise) per
      ml-practical-methodology Part C.
- [ ] Update `ARCHITECTURE_DOCS.md`: document the H1→Daily ensemble module
      (`src/h1_features.py`, `h1_ready` gate) which currently has zero coverage
      despite being wired into `predict()`; also add the `huber_alpha` adaptive-
      quantile clarification from gbm-boosting-theory §1.
- [ ] Revisit the FRED `yield_differential` feature — `results/2C_fred_ablation.csv`
      shows it's net-negative (-0.0039 acc). Decide: drop it, or replace with a
      momentum/delta version instead of raw level.
- [ ] (Stretch) Add probability calibration (`CalibratedClassifierCV`) to the GBM
      classifier so `CONFIDENCE_THRESHOLD=0.52` in `src/inference.py` is judging a
      calibrated probability, not a raw `predict_proba`.
- [ ] (Stretch) Add a simple backtest with transaction costs on the held-out test
      block to see if any edge survives spread/slippage.

## Working rules

- Before each item: re-read the matching skill (`ml-practical-methodology` or
  `gbm-boosting-theory`) so the change stays grounded in it, not just memory.
- After each code change: run `python -m pytest -q`, fix anything that breaks, add
  test coverage if the change needs it.
- After each item is verified working: mark it `[x]` here with a one-line note on
  what changed and why, then commit that item alone.
- If interrupted, this file is the handoff document — resume by reading it, not by
  re-deriving the plan.
