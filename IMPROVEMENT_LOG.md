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
- [x] (Stretch) Add a simple backtest with transaction costs on the held-out test
      block to see if any edge survives spread/slippage.
      **Done:** added `src/backtest.py::simulate_strategy`/`backtest_table` — a
      minimal daily long/short strategy on the GBM direction signal, scored
      against the realised `target_return` on the SAME held-out test block, cost
      charged only on position-change days (not every day). Wired into
      `_train_pipeline.py` right after GBM test evaluation, saving
      `results/backtest_transaction_costs.csv`. Added 2 unit tests (hand-computed
      cost-on-flip-only math, wrong-signal negative-return sanity check).
      **Real result** (current production artifacts, test block = 3,103 days ≈12
      years): gross (no cost) = **+29.08%** total (hit rate 0.5021, barely above
      chance); 1-pip spread more than halves it to **+16.93%**; a typical retail
      2-pip spread leaves **+4.77%** (~0.4%/year) — economically negligible.
      Documented as `ARCHITECTURE_DOCS.md` §4.7, framed as the third independent
      confirmation (after §4.2.1's regression-MAE finding and §4.2.2's
      Brier-vs-baseline finding) of the same efficient-market conclusion, now in
      P&L terms.
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
- [ ] **NEW (flagged during the macro-expansion pass, deliberately not acted on):**
      `yield_differential_delta` shows a small *negative* ablation delta on the
      euro-era 1999+ row set (−0.0041 acc / −0.0045 auc, `results/new_macro_ablation.csv`)
      — opposite sign from its +0.0029/+0.0032 on the 1971+ set. Both within noise.
      Re-evaluate whether to keep/drop the yield feature on the 1999+ set as its own
      one-change-at-a-time pass (with the same bootstrap/McNemar rigor), separate
      from the three features added alongside it.

## Backlog — Macro feature expansion (3 new FRED features, added 2026-07-05, PRODUCTION phase)

Grounded in the CLAUDE.md invariant rule: any `FEATURE_COLUMNS` change updates
`config.json`, `src/features.py`, `src/macro_data.py`, `_train_pipeline.py`,
notebook macro cells, and `src/inference.py` together, no look-ahead. Goal:
`usd_index_return` (log-ret of DTWEXAFEGS), `policy_rate_differential`
(DFEDTARU − ECBDFR), `inflation_differential` (US CPI YoY − DE CPI YoY).

- [x] Step 0: verify the 5 new series IDs resolve via the **official FRED API tier**
      (not the public-CSV fallback). **Done:** throwaway script confirmed all 5
      (DTWEXAFEGS/DFEDTARU/ECBDFR/CPIAUCSL/DEUCPIALLMINMEI) return via `fredapi.Fred`.
      Key findings that shape the impl: ECBDFR is negative (−0.5, fine for a diff);
      CPIAUCSL/DEUCPIALLMINMEI are monthly INDEX levels needing a YoY transform with
      ≥12-month lookback (so the fetcher must extend the start window, else NaN at
      the live edge — a real production bug now that money is at stake). Script deleted.
- [x] Step 1: generalized `src/macro_data.py` — `fetch_fred_feature` writes the
      API→public→cache chain once; `fetch_macro_features` fans out to all four;
      `fetch_yield_differential` kept as thin wrapper. Per-feature cache files.
      Old 4 FRED tests adapted (raw-level fetchers) + new generic fallback test.
- [x] Step 2: `config.json` macro block — `macro.features` with series IDs +
      cache paths + `yoy_lookback_days` for inflation.
- [x] Step 3: `src/features.py` — `merge_macro_features` as-of ffills all 4 macro
      columns (monthly CPI propagates onto daily, no look-ahead); `FEATURE_COLUMNS`
      24→27; `usd_index_return` derived in `compute_features` from the merged level
      (fillna 0 pre-2006); `add_advanced_features` dropna on `FEATURE_COLUMNS+targets`
      only (so `usd_index`'s 2006 start doesn't drag the floor). Floor = 1999-01-04.
- [x] Step 4: `_train_pipeline.py` + `src/inference.py` switched to
      `fetch_macro_features`; `bar_used` surfaces all 4 macro values + per-feed
      sources. **Notebook Section 2B still TODO** (secondary artifact; retrain uses
      `_train_pipeline.py`).
- [x] Step 5: no-look-ahead test per new feature (usd weekend-flat + pre-2006 zero,
      policy ffill, inflation monthly→daily) + generic fallback test. 37 unit/smoke pass.
- [x] Step 6: ablated each new feature individually on the FIXED 1999+ row set
      (`results/new_macro_ablation.csv`). All three point-positive → kept (27 cols).
      **Strengthened with a proper significance test** (`results/new_macro_significance.csv`,
      2000 paired bootstrap resamples + McNemar):
      - `usd_index_return`: Δacc +0.0064, 95% CI [−0.0099, +0.0234], McNemar p=0.499
      - `policy_rate_differential`: Δacc +0.0047, 95% CI [−0.0099, +0.0181], p=0.568
      - `inflation_differential`: Δacc +0.0123, 95% CI [−0.0064, +0.0298], p=0.210
      **All three CIs straddle 0 and all McNemar p ≫ 0.05 → status is KEEP —
      PROVISIONAL (no proven edge, not distinguishable from noise).** Kept in the
      model on a nominally-positive point estimate, but flagged as a live-money
      caveat to revisit with a longer test window. `comparison_table.csv` old
      24-col/1971+ baseline row confirmed intact (retrain never writes it).
      Full 27-col retrain run to regenerate all production artifacts.
- [ ] Step 7: docs — `ARCHITECTURE_DOCS.md` §2.4 table (done), §2.6 macro-expansion
      (done), §4.3.1 ablation+significance (done); notebook Section 2B cell (TODO).

## Backlog — Production methodology hardening (post-defense, added 2026-07-06, REAL-MONEY phase)

The exam is passed; the system now trades real capital. A false-positive feature
is live risk, not a lost grade. Root problem: every feature KEEP/DROP decision so
far (`yield_differential` + the 3 macro features) was scored on the SAME fixed
held-out TEST block `[80%:100%]` — that is data-snooping, and it applies
retroactively to what's already been tested.

- [x] Step 1 — Split validation from test, permanently. Audited the split: the
      prior ablation/significance scratchpad scripts fit on `[0:80%]` and scored
      KEEP/DROP on `iloc[80%:]`, i.e. **on the final test block** — confirmed
      data-snooping. Fixed with a permanent, committed harness `src/ablation.py`
      whose sole arbiter is the VALIDATION slice `[70%:80%]`; the test block is
      never indexed there. PCA/scaler/model are fit on `[0:70%]` only so the val
      block is genuinely held out from the fit. Re-ran all 4 features on
      validation (`results/feature_ablation_validation.csv`):
      - `yield_differential_delta`: Δacc −0.0047, 95% CI [−0.0245, +0.0152], McNemar p=0.728
      - `usd_index_return`: Δacc +0.0058, 95% CI [−0.0152, +0.0245], p=0.645
      - `policy_rate_differential`: Δacc +0.0058, 95% CI [−0.0152, +0.0257], p=0.653
      - `inflation_differential`: Δacc +0.0093, 95% CI [−0.0175, +0.0362], p=0.563
      **All four straddle 0 and all McNemar p ≫ 0.05 on validation too** — the
      test-block "positive point estimates" do not survive being moved to a clean
      arbiter, exactly the efficient-market expectation. Added 2 unit tests
      guarding the split-boundary invariant and the McNemar helper. 41 tests pass.
- [x] Step 2 — Cumulative hypothesis counter + Bonferroni-corrected bar. Added
      `results/feature_hypothesis_log.csv`, seeded retroactively with the 4
      hypotheses already spent (`yield_differential_delta`, `usd_index_return`,
      `policy_rate_differential`, `inflation_differential`). `src/ablation.py`
      now reads that family count and judges every KEEP against a Bonferroni bar
      `alpha = 0.05 / family_size` (currently **0.05 / 4 = 0.0125**), not a flat
      0.05 — printed in the header of every report so it can't be silently
      forgotten. A genuinely new feature grows the family and tightens the bar
      for all; `register_hypothesis()` appends it (idempotent by name). At the
      current bar all 4 stay **KEEP-provisional** (smallest McNemar p is 0.5625,
      ~45x the 0.0125 bar). Added `bonferroni_alpha` + register-idempotency tests.
      43 tests pass.
- [x] Step 3 — Forward/paper-trading harness (the real arbiter going forward).
      Added `src/paper_trading.py`: reconstructs a simulated, cost-net position
      ledger on demand from `results/prediction_log.csv` joined to realised
      closes (UP→long, DOWN→short, MIXED→flat), charging a configurable retail
      spread (`config.json → paper_trading.spread_pips`, default 1.5) once per
      taken position. Writes `results/paper_trading_log.csv` and a scorecard
      (cumulative net pips/%, win rate, annualised Sharpe-like ratio, max
      drawdown). Surfaced at `GET /paper-trading` (HTML) and `GET /api/paper-trading`
      (JSON), cross-linked with `/history`. **Simulated only** — no broker/order
      code (per the task's explicit no-real-execution scope). Current forward
      window is tiny (7 settled positions, net −180 pips / −1.58%) and not yet
      meaningful — the ledger is meant to accumulate over months before it can
      arbitrate production-worthiness. Added 3 unit tests (direction/cost sign,
      pending-exclusion, drawdown/cumulative math). 46 tests pass.
- [x] Step 4 — Document the new methodology. Added a prominent **Production
      Methodology (post-defense)** section to `ARCHITECTURE_DOCS.md` (right after
      the Component Map) stating (a) the test block is spent for feature search,
      (b) all new claims clear the Bonferroni bar, (c) the paper-trading ledger is
      the primary production-worthiness signal — plus a "methodology superseded"
      banner on §4.3.1, new Component-Map / §5.3 / §5.4 rows for the three new
      modules and endpoints. Mirrored the three rules into `CLAUDE.md` as a
      governing section. Docs-only; 46 tests still pass.

## Backlog — Dual model variants (baseline vs with_macro, added 2026-07-06)

The 4 macro features are KEEP-provisional (no proven edge), so instead of betting
the single production model on them, train + persist + serve BOTH a price-only
baseline and the with-macro variant side by side on every prediction, and let the
two forward paper-trading ledgers arbitrate.

- [x] `config.json → variants: ["baseline","with_macro"]` (note: the spec said to
      replace `macro.experimental_features_enabled`, but no such boolean ever
      existed in config.json — the list was added directly). `src/features.py`
      gained `MACRO_FEATURE_COLUMNS` / `PRICE_FEATURE_COLUMNS` (23 = 27 − 4;
      baseline resolved as strictly price-only since the spec's "original
      24-column (no macro block)" was self-contradictory — the 24-col set
      included `yield_differential_delta`, which IS macro) and
      `variant_feature_columns()`.
- [x] `_train_pipeline.py` refactored into ONE `train_variant()` body looped over
      both variants (no copy-paste): shared engineered euro-era row set, unified
      split, per-variant lag_scaler/lag_pca/global_scaler/GBM heads/LSTM under
      `models/<variant>/`, per-variant MLflow runs + per-variant
      `results/gbm_feature_importance_{variant}.csv` and
      `results/backtest_transaction_costs_{variant}.csv`. Stale root-level daily
      artifacts removed post-run. Per the user's safer-default instruction the
      lag PCA is duplicated per variant even though the lag block is currently
      identical across them.
- [x] `src/inference.py` loads both variants with independent gates
      (`baseline_ready`/`macro_ready`; per-family flags inside
      `service.variants[name]`); one shared data fetch feeds both committees;
      response = `{baseline, with_macro, variant_agreement}` (None when a side is
      degraded — honest "can't compare", not fake agreement).
- [x] `src/tracking.py` log schema: `pred_*` columns stay = with_macro committee
      (continuous lineage for the macro ledger), new `baseline_*` +
      `variant_agreement` columns. `src/paper_trading.py`: `direction_column`
      param + `build_all_ledgers`; pre-dual rows are SKIPPED (not flat-logged) by
      the baseline ledger since no baseline forecast existed. Two ledgers:
      `results/paper_trading_log_baseline.csv` / `_macro.csv` (config
      `paper_trading.ledgers`); old single `paper_trading_log.csv` superseded.
- [x] `static/index.html`: side-by-side variant panels (responsive grid), macro
      panel badged "⚠ experimental / unproven" with the Bonferroni context in the
      tooltip (honesty over cosmetic confidence), agreement/disagreement banner +
      border color driven by `variant_agreement`.
- [x] Tests: `test_smoke` asserts all 14 per-variant artifacts; integration
      asserts the dual response shape + `variant_agreement` consistency; unit
      tests for the variant column contract, dual-consensus logging, and
      baseline-ledger skip semantics.
- [ ] **Follow-up:** port notebook Section 19 to `train_variant()` — it still
      trains the pre-dual single 27-col pipeline into root-level `models/` paths
      production no longer loads (flagged in CLAUDE.md as known drift).

## Bug fixes

- [x] **H1 consensus 2-2 tie bug (live dashboard report, 2026-07-07).** A 2-2
      vote split displayed "UP — 50% model agreement" over a NEGATIVE −0.0131%
      averaged return. Root cause: `compute_h1_consensus` broke ties with
      `up >= down` (arbitrary UP) while `predicted_return_pct` was the full-panel
      mean — a vote-derived label shown next to a magnitude-derived number from a
      contradicting definition. The daily committee's design is vote-based with
      an explicit MIXED downgrade, so H1 was standardized to match (design (a)):
      strict-majority direction, `MIXED / TIE` on an exact split, and the
      consensus return now averages the MAJORITY side only (sign-consistent with
      the label by construction; the old full-panel mean could contradict a 3-1
      vote too — the prior unit test literally asserted UP over a −0.025 mean).
      UI updated (neutral tie card, "Majority-side averaged return" label);
      ARCHITECTURE_DOCS §3.4 states the tie handling; regression test
      `test_compute_h1_consensus_exact_tie_is_mixed_not_arbitrary_up` pins the
      exact reported numbers. `/history` already treats a non-UP/DOWN h1 call as
      unscored. 50 tests pass.
      **Related observation (not touched, one change at a time):** the DAILY
      committee can also surface a sign-conflicting pair on the per-model level —
      each model's direction comes from its classifier/direction head while its
      return comes from the regression head (multi-task sign agreement is only
      0.61–0.72). E.g. the 2026-06-29 logged row: pred DOWN with +0.0104% return.
      That is an architectural property of the dual heads, already measured by
      the `multitask_sign_agreement` metric; whether the consensus should
      reconcile it (e.g. defer the displayed return to the direction-consistent
      head) is a separate decision worth its own pass.

## Working rules

- Before each item: re-read the matching skill (`ml-practical-methodology` or
  `gbm-boosting-theory`) so the change stays grounded in it, not just memory.
- After each code change: run `python -m pytest -q`, fix anything that breaks, add
  test coverage if the change needs it.
- After each item is verified working: mark it `[x]` here with a one-line note on
  what changed and why, then commit that item alone.
- If interrupted, this file is the handoff document — resume by reading it, not by
  re-deriving the plan.
