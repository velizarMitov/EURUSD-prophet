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
- [x] Add a sign-agreement check between the LSTM's `return_output` and
      `direction_output` heads on the test block, to verify the shared multi-task
      trunk is actually justified (ml-practical-methodology Part B).
      **Done:** added `sign_agreement_lstm` computed on `X_test_seq` (comparing
      `sign(return_output)` vs `direction_output`'s call), printed with a
      <0.55 "weak" flag, and logged to MLflow as `multitask_sign_agreement`. If it
      comes back near 50% on the next retrain, the shared trunk isn't earning its
      keep and two single-task models should be A/B'd against it.
      **Retrain result:** 0.7249 ("shared signal present") — well above the 0.55
      threshold. The shared `lstm_trunk` is empirically justified; no need to A/B
      against two single-task models at this time.
- [x] Add `subsample` to `config.json`'s `gbm.param_grid` (currently defaults to 1.0 —
      stochastic gradient boosting per gbm-boosting-theory §3 is missing).
      **Done:** added `"subsample": [0.5, 0.8, 1.0]` (gbm-boosting-theory §3's own
      suggested values) to `config.json → gbm.param_grid`; applies to both the
      XGBClassifier and XGBRegressor grids in `_train_pipeline.py` with no code
      change needed (XGBoost's sklearn API already accepts `subsample`). Verified
      with a smoke-fit. Note: this 3x's the grid size on top of the existing 12
      combos — the gbm-boosting-theory skill's `n_estimators` widening ([100,200] →
      +500,1000) is a separate, not-yet-done follow-up since small `learning_rate`
      needs more trees to converge; left out of this commit to keep it scoped.
- [x] Extract and persist GBM `feature_importances_` to
      `results/gbm_feature_importance.csv` after training (gbm-boosting-theory §4) —
      currently never computed anywhere.
      **Done:** after the artifact `joblib.dump`s in `_train_pipeline.py`, both the
      direction classifier's and return regressor's `feature_importances_` are
      captured against `MODEL_INPUT_COLUMNS` (the exact post-PCA column order),
      written to `results/gbm_feature_importance.csv`, and printed. Smoke-tested
      the export logic against real fitted XGBoost models. Next retrain will show
      whether other features are dead weight the way `yield_differential` already
      is (`results/2C_fred_ablation.csv`) — feeds directly into backlog item 7.
- [x] Add a "worst mistakes" pass over `results/prediction_log.csv` (sort by
      abs_error, inspect for systematic patterns vs random noise) per
      ml-practical-methodology Part C.
      **Done:** added `worst_mistakes()` to `src/tracking.py` — reuses
      `_actual_closes()` (same realised-close join as `build_history_html`) since
      the log has no `actual_return_pct` column of its own, computes `abs_error`
      per resolved row, and returns the worst `n` sorted descending. Excludes
      still-pending rows (undefined error) and MIXED/LOW-CONFIDENCE rows with no
      numeric `pred_return_pct`. Runnable standalone via `python -m src.tracking`
      (writes `results/worst_mistakes.csv`). Verified against the real log: the 9
      resolved forecasts show errors scattered 0.02%–0.47% with no obvious
      clustering by date — consistent with the irreducible-noise conclusion in
      `ARCHITECTURE_DOCS.md §4.2.1`, not a fixable pattern. Added
      `test_worst_mistakes_ranks_by_absolute_error_and_excludes_pending` to
      `tests/test_unit.py`.
- [x] Update `ARCHITECTURE_DOCS.md`: document the H1→Daily ensemble module
      (`src/h1_features.py`, `h1_ready` gate) which currently has zero coverage
      despite being wired into `predict()`; also add the `huber_alpha` adaptive-
      quantile clarification from gbm-boosting-theory §1.
      **Done:** added new §3.4 (data/features/training/serving/response-shape for
      the H1 ensemble), a Component Map row for `src/h1_features.py`, an H1
      artifacts table in §5.1, an H1 row in §4.6 test inventory, and two Appendix A
      failure-mode rows (`h1_ready==False`, post-feature-change shape mismatch).
      **Also found and corrected while there:** §3.1 said the return regressor
      uses sklearn's `GradientBoostingRegressor(loss='huber', alpha=0.9)` with the
      adaptive-quantile semantics gbm-boosting-theory describes — but the actual
      code trains `xgb.XGBRegressor(objective='reg:pseudohubererror')`, and
      `config.json`'s `huber_alpha` is **only logged to MLflow, never passed to the
      constructor** (XGBoost's real threshold knob is `huber_slope`, unset, default
      `1.0`). Documented this as an open doc/config drift rather than silently
      "fixing" model behavior in a docs-only commit — also corrected the stale
      `GradientBoostingClassifier`/`GBClassifier`/`GBRegressor` labels in §3.1 and
      §5.1 to the real `xgb.XGBClassifier`/`xgb.XGBRegressor`, and the test count
      (18 → 28, `worst_mistakes` + 5 H1 tests already existed but were uncounted).
- [x] Revisit the FRED `yield_differential` feature — `results/2C_fred_ablation.csv`
      shows it's net-negative (-0.0039 acc). Decide: drop it, or replace with a
      momentum/delta version instead of raw level.
      **Done:** replaced the raw-level model feature with `yield_differential_delta`
      (`diff(1)`, derived in `compute_features` exactly like `log_return` is
      derived from `close` — same stationarity rationale). The raw level merge
      (`merge_macro_features`) is untouched and still feeds the dashboard's human-
      readable display (`bar_used.yield_differential`); only the model-facing
      feature changed. Standalone ablation (notebook §2C's exact quick-GBM
      methodology, same split): raw level was net-negative (acc −0.0039, auc
      −0.0021); the delta flips both positive (acc +0.0029, auc +0.0032) —
      confirmed by a **full retrain**: `yield_differential_delta` ranks 22/24 by
      Gini importance (0.0358, comparable to other mid-pack features, not dead
      weight). Added `test_yield_differential_delta_no_lookahead_on_weekend_gap`
      proving the diff never leaks a future Monday value into a ffilled
      Sat/Sun. Updated `results/2C_fred_ablation.csv` (kept the old raw-level row
      for the before/after record) and `ARCHITECTURE_DOCS.md` §2.3/§2.4/§4.3/§4.5.
      End-to-end verified: `bar_used.yield_differential` still shows the raw level
      (1.4335) post-retrain, confirming display/model decoupling. All 32 tests pass.
- [x] (Stretch) Add probability calibration (`CalibratedClassifierCV`) to the GBM
      classifier so `CONFIDENCE_THRESHOLD=0.52` in `src/inference.py` is judging a
      calibrated probability, not a raw `predict_proba`.
      **Evaluated, NOT adopted** (a real decision, not a no-op): implemented
      `CalibratedClassifierCV(method='sigmoid', cv=TimeSeriesSplit)` in
      `_train_pipeline.py`, smoke-tested it end-to-end (joblib round-trip,
      `hasattr(_, 'get_booster')` device-guard compatibility confirmed), then ran a
      quick standalone check against the real project data before committing to a
      full retrain: Brier score barely moves (raw 0.25063 -> sigmoid 0.25025 ->
      isotonic 0.25038), and a **trivial "always predict the train base rate"**
      baseline already scores 0.25013 -- the raw classifier's probabilities are
      already statistically indistinguishable from (microscopically worse than)
      that constant baseline, so there is essentially nothing for calibration to
      fix. Sigmoid's tiny Brier "improvement" comes entirely from collapsing the
      predicted-probability range from `[0.333,0.672]` to `[0.461,0.513]`, which
      would push `CONFIDENCE_THRESHOLD=0.52` to almost never trigger again --
      trading a real, user-visible behavior change (dashboard defaults to
      `MIXED / LOW CONFIDENCE` almost permanently) for a negligible accuracy-of-
      belief gain. **Reverted the code change** (confirmed via `git diff` showing
      zero diff on `_train_pipeline.py` against the item-7 commit) rather than
      ship it. Documented the full evidence trail as new `ARCHITECTURE_DOCS.md`
      §4.2.2, mirroring §4.2.1's "predict the mean" framing — this is the
      classification-side twin of that same efficient-market finding. If a future
      retrain shows the raw classifier pulling meaningfully away from chance
      ROC-AUC, this decision should be revisited.
- [ ] (Stretch) Add a simple backtest with transaction costs on the held-out test
      block to see if any edge survives spread/slippage.
- [ ] **NEW (discovered via item 1's train-set diagnostic):** the GBM direction
      classifier shows train ROC-AUC=0.7798 vs test ROC-AUC=0.5059 (gap +0.2740) on
      the item-7 retrain — this is **not** the efficient-market floor documented in
      `ARCHITECTURE_DOCS.md §4.2.1` (train≈test≈0.50), it's classic high-variance
      overfitting per the model-tuning-bias-variance decision tree. Selected
      hyperparameters this run: `max_depth=5, n_estimators=100, subsample=1.0` (grid
      did NOT pick a subsample <1.0 despite item 3's new axis). The LSTM shows the
      same pattern but far milder (train 0.59 vs test 0.4993, gap +0.0907 — its
      Dropout is doing real regularization work the GBM lacks). Since train-set
      logging never existed before this session, it's unknown whether this gap is
      new or has always been there, silently. Action for next pass: constrain
      `max_depth` lower (e.g. `[2, 3]` instead of `[3, 5]`) and/or force
      `subsample <= 0.8` in `config.json → gbm.param_grid`, then re-check the gap on
      a retrain — per model-tuning-bias-variance, do not conclude "efficient
      market" for the GBM head until this is addressed. Also worth updating
      `ARCHITECTURE_DOCS.md §4.2`'s "Honest performance reality" table, which
      currently only reports test-side numbers for the daily heads.

## Working rules

- Before each item: re-read the matching skill (`ml-practical-methodology` or
  `gbm-boosting-theory`) so the change stays grounded in it, not just memory.
- After each code change: run `python -m pytest -q`, fix anything that breaks, add
  test coverage if the change needs it.
- After each item is verified working: mark it `[x]` here with a one-line note on
  what changed and why, then commit that item alone.
- If interrupted, this file is the handoff document — resume by reading it, not by
  re-deriving the plan.
