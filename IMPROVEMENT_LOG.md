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

## Backlog — Next-day realized volatility target (added 2026-07-07)

A genuinely different task from direction/return, where real signal is far more
plausible (volatility clustering is an established FX stylized fact, unlike
next-day direction). Ran entirely under the Production Methodology: validation
arbiter only, mandatory GARCH baseline before any NN, its own hypothesis family.

- [x] Step 0 — Prereqs: `config.json` was already valid in the working tree
      (no restore needed; `json.load` passes). `arch>=6.0.0` added to
      requirements.txt (installed 8.0.0). ONE volatility model, not
      per-variant — GARCH/volatility are price-only by nature.
- [x] Step 1 — Target + honest baselines FIRST. `target_volatility_pct =
      |next-day log return| * 100` in `src/features.py::add_advanced_features`
      (same `shift(-1)` geometry as `target_return`; unit tests mirror the
      existing no-look-ahead pattern). `src/volatility.py`: GARCH(1,1) fit on
      the train block ONLY then FIXED-parameter-rolled (`ARCHModel.fix`) —
      genuinely out-of-sample one-step forecasts; E|r| via √(2/π)·σ. On
      validation `[70:80]`: GARCH MAE 0.2038% / R² +0.009, persistence
      MAE 0.2611% / R² −0.842 — the bar the NN had to clear, computed BEFORE
      building it.
- [x] Step 2 — Dedicated price-only LSTM (fit `[0:63]`, ES tail `[63:70]` so
      the arbiter stays clean) beat GARCH on the first run (MAE 0.1897,
      R² +0.118, 95% CIs excluding 0)… but the follow-up multi-task experiment
      exposed **TF/oneDNN training nondeterminism of the same order as the
      deltas** (identical-seed MAE 0.190→0.197), which flipped the primary
      re-check at the tightened family bar. Honest fix: ONE final
      pre-registered ship gate — the 5-seed ensemble (mean over 42–46) of the
      3-head multi-task trunk (which beat the dedicated model head-to-head;
      sharing HELPS the vol head: corr(vol_head,|return_head|)=+0.16) vs
      GARCH at family_size=3 Bonferroni α=0.0167. **CLEARED decisively**:
      ensemble MAE 0.1859 / R² +0.144, ΔMAE CI98.33 [+0.0111,+0.0242],
      ΔR² CI [+0.080,+0.183], frac=1.000; every individual seed also beat
      GARCH (`results/volatility_seed_ensemble.csv`).
- [x] Step 3 — Separate hypothesis family:
      `results/volatility_hypothesis_log.csv` (3 hypotheses spent: dedicated
      vs GARCH, MT-head vs dedicated, MT-ensemble vs GARCH — the family is
      now SPENT on this arbiter; further volatility claims need genuinely new
      data, i.e. the forward window). Never mixed with the 4-feature
      direction/return family count.
- [x] Step 4 — Shipped with honest framing (the gate cleared, so it ships as
      validated — the exact mirror of the macro features shipping as
      unproven): `train_production_volatility_model` in `_train_pipeline.py`
      §12B → `models/volatility/` (5 seed models + own PCA/scalers fit
      `[0:80]` + `vol_metrics.json` carrying the one-shot test report:
      ensemble MAE 0.2188 / R² +0.110 vs GARCH 0.2326 / +0.036 — edge
      generalizes to the untouched test block). `PredictionService.vol_ready`
      all-or-nothing gate (a partial ensemble is an unvalidated object →
      refuses to serve), `volatility_forecast` response block with
      `vs_garch_baseline`/`vs_persistence_baseline` evidence, UI card
      "Next-Day Realized Volatility ✓ validated vs GARCH(1,1)" —
      direction-free magnitude framing. `test_smoke` asserts the 10 new
      artifacts; ARCHITECTURE_DOCS.md §3.5. 52 tests pass.

## Backlog — Volatility candidate input features (RSI_14 / BB %B, added 2026-07-17)

Owner-directed re-opening of the volatility hypothesis family (previously
marked spent on the validation arbiter) under the standard Bonferroni
widening: two volatility-regime oscillators, pre-declared up front and tested
ONE AT A TIME against the validated 5-seed ensemble, each judged at the
final-family bar `0.05/5 = 0.01` (99% CIs) — stricter than the 0.0167 the
original ship gate cleared. Runner: `python -m src.volatility candidates`
(`run_candidate_feature_tests`); results
`results/volatility_candidate_features.csv`; both registered in
`results/volatility_hypothesis_log.csv` (hypotheses 4–5).

- [x] Step 1 — Features constructed with no look-ahead, volatility-only:
      `RSI_14` literally reuses `src/h1_features.py::_rsi` (period=14, daily
      close); `BB_percent_b = (close − lower) / (upper − lower)` from the
      SAME 20-day rolling mean/σ that `BB_width` uses (σ=0 → neutral 0.5,
      mirroring `_rsi`'s neutral-50). Computed on full history before the
      euro-era dropna (warm-ups never reach modeled rows); NEVER added to
      `FEATURE_COLUMNS` (direction/return capacity question is closed per the
      Ch.11 diagnostic — a unit test enforces the separation). 3 new tests:
      formula consistency, future-truncation invariance, family separation.
- [x] Step 2 — Both hypotheses tested vs the SAME freshly trained same-seeds
      base ensemble (never vs each other), paired bootstrap 2000 resamples on
      identical validation rows, test block never indexed. Base: val MAE
      0.184755% / R² +0.1445 (GARCH context 0.203794% / +0.0094).

      | hypothesis | Δ MAE (base−cand) | ΔMAE CI99 | ΔR² CI99 | verdict |
      |---|---|---|---|---|
      | #4 +RSI_14 | −0.0010% | [−0.00222, +0.00013] | [−0.0216, +0.0016] | **DROP** |
      | #5 +BB_percent_b | −0.0015% | [−0.00254, −0.00036] | [−0.0121, +0.0069] | **DROP** |

      Neither CI excludes 0 in the feature's favor; BB %B's ΔMAE CI is
      entirely negative — CI-confirmed HARM (frac(ΔMAE>0)=0.000): the extra
      input dilutes the validated ensemble. Textbook illustration of why the
      Bonferroni + validation-arbiter discipline exists: two "obvious"
      volatility-regime oscillators, honestly tested, add nothing the price
      features don't already carry.
- [x] Step 3 — Null results documented (this entry + hypothesis log +
      ARCHITECTURE_DOCS §3.5), exactly like the macro-feature nulls — not
      discarded quietly. Production `models/volatility/` UNTOUCHED (no
      retrain, no UI change — the KEEP branch never triggered); the family
      count is now 5, so any future volatility hypothesis faces
      `alpha = 0.05/6 = 0.0083`. Full suite green (58 tests).

## OWNER OVERRIDE — H1 TI-LSTM wired into production DESPITE its DROP verdict (2026-07-18)

**Read this before trusting the TI-LSTM panel: its presence in production is
NOT validation.** By explicit owner decision (2026-07-18), the H1
technical-indicator LSTM below — which FAILED its own pre-registered
hypothesis bar (DROP: test AUC 0.5128 vs the existing H1 ensemble's 0.5283,
ΔAUC −0.015 CI [−0.072, +0.042] — includes 0, point estimate NEGATIVE) — was
promoted to serving anyway, for **transparent forward observation only**.
This overrides the original "only ship if it clears the bar" rule for this
one model. It does NOT get the volatility model's validated framing, nor the
macro features' "nominally positive, unproven" framing — its labels state
plainly that it demonstrated no edge.

- [x] Artifacts `models/ti_lstm_h1/` (2×64, seed 42, 13 epochs, CUDA;
      `ti_metrics.json` carries `validated: false` + the real numbers) —
      trained via `python -m src.ti_lstm_h1_experimental train-production`.
- [x] `ti_h1_ready` all-or-nothing gate in `src/inference.py` (mirrors
      `vol_ready`); distinct `ti_h1_forecast` response block with
      `validated: false` + verbatim test evidence — never merged into the H1
      ensemble's block. The torch-trained `.keras` is backend-portable and
      serves under tf.keras (verified) — serving has NO torch dependency.
- [x] Amber warning UI card: "⚠ Not Validated — No Demonstrated Edge (test
      AUC ≈0.51, ΔAUC CI included 0 vs H1 ensemble)" with the actual numbers
      rendered from the API block, visually distinct from the validated
      volatility card.
- [x] Retrain flow §12C in `_train_pipeline.py`: runs the TI retrain as a
      SUBPROCESS — mandatory, not just safer: KERAS_BACKEND freezes at the
      first keras import and the pipeline process already imported tf.keras.
      Verified end-to-end: a TF-backend host process spawned the CUDA torch
      subprocess (rc=0) and hot-reloaded the fresh artifact.
- [x] Third forward paper-trading ledger
      (`results/paper_trading_log_ti_h1.csv`, driven by the new
      `ti_h1_direction` log column) — the honest way to watch it: forward
      P&L accumulation, no historical-edge claim.
- [x] Full suite green; live end-to-end `/api/predict` confirmed all panels
      (validated + observational) render together.

## Experiment — Standalone H1 technical-indicator LSTM (2026-07-18, research-only, VERDICT: DROP)

Isolated experiment (`src/ti_lstm_h1_experimental.py`, own entry point, never
imported by api.py / inference / _train_pipeline; zero artifacts under
models/) testing whether an H1-native LSTM over a classic TI set beats the
shipped H1 ensemble. Own hypothesis family
(`results/ti_lstm_h1_hypothesis_log.csv`, n=1, bar 0.05); full run report
`results/ti_lstm_h1_validation.csv`.

- [x] **Backend**: Keras 3 on the PYTORCH backend, real CUDA on the RTX 4070
      Laptop GPU (torch 2.11.0+cu128), process-local `KERAS_BACKEND=torch`,
      loud-fail gate (no silent CPU fallback). Production stays on tf.keras.
- [x] **Determinism finding (contrast with TF/oneDNN)**: two identical
      seed-42 runs were BIT-IDENTICAL (max |Δprob| = 0.0) — Keras3/torch/cuDNN
      is run-to-run deterministic here, so single-seed comparisons are valid
      in this setup (the 5-seed fallback was armed but not needed).
- [x] **Data/target**: existing H1 chain only (MT5 refreshed 60k bars live);
      2500 complete sessions 2016-11-24 → 2026-07-16; (24, 8) right-aligned
      hourly tensor; target = NEXT-DAY percent return / sign via the H1
      module's `build_daily_target` shift(-1) — not next-hour. Indicators at
      the exact specified params: %B(20, 2σ), MACD 13/34 with 8-SMA signal
      (spec'd Fibonacci variant, not 12/26/9), trend vs SMA-504/168, RSI_24
      (REUSED from `h1_features._rsi` — flagged assumption: period 24 per the
      H1 module convention, not the daily-conventional 14), CCI-20 (Lambert
      0.015), ADX-14 (Wilder; closed-form unit test verifies the recursion).
- [x] **Architecture sweep (validation only)**: 1×32 AUC 0.506, 1×64 0.524,
      2×32 0.514, 2×64 0.542 → 2×64 selected; early stopping fired at epochs
      12–28 of 100 in every config (capacity self-limits, as Ch.11 predicted).
- [x] **Step 6 comparisons (full coverage, no denominator tricks)**:
      validation — TI 0.5421 vs existing H1 ensemble 0.5902 (caveat: the
      ensemble TRAINED on [0:80%], so the val slice favors it), ΔAUC −0.048
      CI95 [−0.135, +0.040]; vs daily baseline GBM 0.5325: ΔAUC +0.010
      CI [−0.074, +0.092]. One-shot TEST block (fair — both out-of-sample):
      TI AUC 0.5128 / acc 0.5060 vs ensemble 0.5283, ΔAUC −0.015
      CI [−0.072, +0.042].
- [x] **VERDICT: DROP — research-only.** No CI-confirmed edge anywhere; the
      TI-LSTM lands at ≈0.51 out-of-sample — the same near-chance floor as
      the reviewed paper's TI_LSTM (~52%) and this project's own Ch.11
      evidence. NOT wired into serving (no `ti_h1_ready` gate, no API block,
      no UI card — the KEEP branch conditions were never met). The module
      stays as reusable research scaffolding (CUDA backend gate, indicator
      library with closed-form tests, own hypothesis log).

## Backlog — FOMC meeting-day calendar features (added 2026-07-17, tested BOTH families)

Genuinely new information (a scheduled-event calendar — nothing like it tried
before in either family): `is_fomc_day` / `days_to_next_fomc` /
`days_since_last_fomc`, bundled as ONE hypothesis per family (three views of
the same calendar fact — they never eat three Bonferroni slots).

- [x] Step 1 — Calendar built from real sources, verified live ("probe before
      you build"). **ALFRED release-dates for rid=101 REJECTED by the probe**:
      it returns ~3.7k near-DAILY dates — per its own header, "dates when any
      series from this release was revised" (data-revision timestamps), not
      meeting days. Replacement (same official domain): fomccalendars.htm
      (2021–2027) + fomchistorical{1998..2020}.htm, parsed from the meeting
      headings with a statement-link cross-check that fails loudly on parser
      drift. `results/fomc_dates.csv`: **240 statement days (1998-02-04 →
      2027-12-08)**, statement day = LAST day of each SCHEDULED meeting.
      Unscheduled/notation-vote/cancelled actions excluded BY DESIGN — a
      surprise action was not knowable in advance, so a countdown including it
      would inject future knowledge into pre-announcement rows (the 2020-03
      emergency cuts are the canonical case). Small static reference file
      (~8 dates/yr), NOT a live-fetch feed; **needs a manual annual refresh**
      (`python -m src.fomc_calendar`) when the Fed publishes the next year.
      Scheduled dates are public years ahead → no publish-lag/look-ahead
      surface (unlike COT/FRED); correctness tests cover the join geometry
      and known/excluded dates (no look-ahead test — nothing to guard).
- [x] Step 2 — ONE hypothesis per family at each family's CURRENT bar
      (counts read from the logs at run time):

      | family | hypothesis # | bar | result | evidence |
      |---|---|---|---|---|
      | volatility | #6 | 0.05/6 = 0.0083 | **DROP** | ΔMAE = −0.0003% CI99.2 [−0.0024, +0.0016]; ΔR² [−0.0207, +0.0174]; frac(ΔMAE>0)=0.344 |
      | direction/return | #5 | 0.05/5 = 0.01 | **DROP** | Δacc = −0.0292, 95% CI [−0.0549, −0.0035] (entirely harmful); McNemar b=46/c=71, p=0.0261 > bar |

      Volatility: same 5-seed (42–46) ensemble methodology vs a same-seeds
      base (val MAE 0.183148% / R² +0.1545), paired bootstrap 2000 resamples,
      validation [70:80] only (`run_candidate_feature_tests`, bundle-aware).
      Direction/return: ADD-test via `src/ablation.py::run_addition_test`
      (paired bootstrap + exact McNemar, same as the macro features). The
      direction result matches the tempered expectation set by the Ch.11
      diagnostic — and the point estimate is actively negative, consistent
      with FOMC being a volatility event, not a directional-bias event…
      except the volatility family ALSO shows nothing: the price features
      (volatility_20/ATR/BB_width) apparently already carry the FOMC-day
      variance the LSTM can use.
- [x] Step 3 — Both nulls registered (`results/volatility_hypothesis_log.csv`
      #6, `results/feature_hypothesis_log.csv` #5) and documented here +
      ARCHITECTURE_DOCS (§3.5 and the Production-Methodology bar, now
      0.05/5=0.01 for direction/return; CLAUDE.md refreshed). No retrain, no
      UI change — the KEEP branch never triggered in either family. Next
      bars: volatility 0.05/7 ≈ 0.0071, direction/return 0.05/6 ≈ 0.0083.
      Full suite green (60 tests).

## Diagnostics — Ch.11 train-vs-test capacity check (2026-07-17, diagnostic only)

Settles the repeatedly-flagged question: could more capacity (epochs/layers)
help the direction/return GBM + LSTM, or are they at a Bayes-error floor?
Read-only pass — no architecture/config change. Script reconstructed the exact
training matrix from the persisted artifacts (cache-only macro, n=8560 matching
the 2026-07-10 retrain); recomputed train direction metrics match
`results/retrain.log` digit-for-digit (with_macro test metrics differ in the
3rd decimal only because FRED cache revisions since the retrain touched a few
test-era rows). Full table: `results/train_vs_test_diagnostic.csv`.

- [x] Step 1 — Train-vs-test metrics (each model's OWN train rows: GBM
      `[0:80%]`, LSTM `[0:70%]`; identical held-out test `[80%:100%]`):

      | model | variant    | AUC train | AUC test | gap    | Acc train | Acc test | R² train | R² test |
      |-------|------------|-----------|----------|--------|-----------|----------|----------|---------|
      | GBM   | baseline   | 0.6157    | 0.5220   | +0.094 | 0.5796    | 0.5093   | +0.023   | −0.002  |
      | GBM   | with_macro | 0.6166    | 0.5218   | +0.095 | 0.5897    | 0.5035   | +0.025   | −0.002  |
      | LSTM  | baseline   | 0.5575    | 0.5046   | +0.053 | 0.5343    | 0.5154   | +0.024   | −0.007  |
      | LSTM  | with_macro | 0.5697    | 0.5302   | +0.040 | 0.5484    | 0.5118   | +0.031   | −0.032  |

      Return-head caveat: train MAE (0.39–0.41%) > test MAE (0.30%) is a
      target-dispersion artifact (the 1999–2019 train era contains 2008 etc.);
      R² is the cross-split-comparable number, and it says: ~2–3% of variance
      fit in-sample, ≤ 0 out-of-sample (worse than predicting the mean).
- [x] Step 2 — Ch.11 classification: **mild overfit above a Bayes floor at
      chance**, for all four models. Train is modestly above chance (the models
      already have enough capacity to memorize +0.04–0.09 AUC of noise), test
      is at chance — the branch whose remedy is MORE regularization, never
      more capacity. And the regularizers are already maxed in the useful
      direction: all 4 GBM grid searches (2 heads × 2 variants) picked the
      MINIMUM-capacity corner of the grid (`n_estimators=100, max_depth=3,
      lr=0.01`) — larger capacity was offered and lost in TimeSeriesSplit CV.
- [x] Step 2b — Software-defect rule-out ("fit a tiny dataset"): a fresh LSTM
      with the exact production architecture/loss memorized 5 training rows —
      total loss 0.796 → 0.017 (46×), direction 5/5 (probs saturated
      0.006/0.984), return MAE 0.037% vs target scale ~0.37%. PASS: the
      training loop/loss/scaling can learn when signal exists; flat test
      metrics are a property of the data, not a bug.
- [x] Step 3 — "More epochs" is mechanically moot: early stopping
      (patience=10, restore_best_weights) fired at epoch 14 (baseline) / 13
      (with_macro) of the 100-epoch cap — best validation weights came from
      epoch ~4/~3. The models already stop themselves ~86 epochs before the
      cap; raising it changes nothing.
- [x] **Conclusion — do NOT scale epochs/layers on the direction/return
      models.** Two independent locks: (1) extra capacity was already offered
      and rejected (GBM CV chose the grid floor 4/4; LSTM quits after ~4
      useful epochs); (2) what capacity does fit on train (+0.05–0.09 AUC,
      +0.02–0.03 R²) generalizes to exactly nothing (test AUC ≈ 0.50–0.53,
      R² ≤ 0), with Step 2b ruling out a defect. Only genuinely new
      information can move test performance — different features (the forward
      paper-trading arbiter) or a different target (exactly how the
      volatility family found its validated edge). Documented in
      ARCHITECTURE_DOCS.md §4.2.1.
- [x] **COT (Commitment of Traders) net-speculative positioning — new candidate
      feature, tested in BOTH families, DROPPED in both (2026-07-20).** Added
      `src/cot_data.py`: leveraged-funds net position (long − short) for EURO FX
      and ICE's USD INDEX (DX) futures from the CFTC "Traders in Financial
      Futures" Socrata dataset (`gpe5-46if`), z-scored over a trailing 3-year
      weekly window (raw contract counts are non-stationary — OI has grown for
      two decades — so raw levels were deliberately not used). Genuinely
      different information (positioning/sentiment), not another price/macro
      transform. Reused the owner's working CFTC client from
      `C:\Users\test\PycharmProjects\COTForex\cftc_api.py` (dataset id, endpoint,
      `lev_money_positions_*` fields), ported self-contained into this repo with
      the same API→cache→None fallback chain and merge-not-overwrite caching as
      `src/macro_data.py`.
      **Look-ahead discipline (the whole risk of a positioning feature):** CFTC
      reports a Tuesday "as of" date but publishes the following Friday (~3-day
      lag), delayed irregularly by holidays/shutdowns. Verified against the live
      API that Socrata's true publish timestamp `:created_at` is reliable ONLY
      for rows after the 2022-09-13 bulk reload (every earlier row carries that
      single reload date, not its original publish). So `availability_date()` is
      hybrid: trust `:created_at` when its lag over as_of is plausible (recent
      rows — handles the real holiday/shutdown delays, e.g. as_of 2026-06-30 →
      published 2026-07-06), else fall back to a CONSERVATIVE `as_of + 10 days`
      (deep history) — never a fixed +3, which would leak during exactly those
      gaps. The z-score is computed on the native WEEKLY cadence then ffilled by
      availability date onto daily bars (`add_cot_features`), so a bar only ever
      sees a reading already public. Residual honesty caveat: a pre-2022
      multi-week shutdown could exceed the +10 buffer and be treated available a
      few days early (a handful of deep-train rows); all live/forward data uses
      the true `:created_at`. `cot_staleness_days()` diagnostic added, mirroring
      the `macro_source`/`h1_data_source` convention; it is wired into serving
      only if COT ever ships (it did not — see below — so it stays a module-level
      diagnostic rather than dead code in the live response).
      **Both families reject it, judged validation-only at the Bonferroni bar,
      test block never touched:**
        * Direction/return (bundled ONE hypothesis via `src/ablation.py cot`;
          family 5→6, bar 0.05/6 = 0.0083): Δacc **−0.0035**, Δauc −0.0099,
          95% CI dacc [−0.0234, +0.0175], McNemar **p=0.83** →
          **DROP** (`feature_hypothesis_log.csv` n=6). Direction family now 0/6.
        * Volatility (bundled ONE hypothesis via `src/volatility.py candidates
          cot`, same 5-seed MT ensemble methodology; family 6→7, bar
          0.05/7 = 0.0071): base ensemble MAE 0.1822% vs +COT 0.1889%, ΔMAE
          **−0.0067%** CI **[−0.0091, −0.0045]** (entirely BELOW 0 — reliably
          WORSE, not merely indistinguishable), ΔR² −0.0249 CI [−0.046, −0.008],
          frac(dMAE>0)=0.000 → **DROP** (`volatility_hypothesis_log.csv` n=7).
          The challenger's best seed (0.1824%) still lost to the base ensemble
          mean, so it is not a training-noise artifact.
      **Conclusion:** COT positioning carries no next-day EUR/USD edge in either
      task — consistent with COT being documented for multi-week reversals, not
      daily moves, and with this project's efficient-market result. Kept the
      module + both harness hooks + tests so the finding is reproducible, but
      COT is NOT added to `FEATURE_COLUMNS`, NOT served, NOT in any variant, and
      triggered no retrain. A null (here, negative) result honestly recorded —
      not a failure hidden. New COT claims need genuinely new forward data. See
      ARCHITECTURE_DOCS.md §4.3.2.
- [x] **COT weekly-horizon side-check — exploratory, own family, DROP
      (2026-07-21).** Follow-up to the above: does COT positioning carry an edge
      on a WEEKLY horizon (its documented use — multi-week reversals), even
      though it failed both DAILY families? Added `src/cot_weekly_check.py`: daily
      close (`results/eurusd_features.csv`) resampled to weekly **W-TUE** bars
      (last close, aligned to CFTC's Tuesday as-of cadence); target = forward
      weekly log return (`shift(-1)` weekly, direction = sign); predictors =
      `cot_eur_zscore`/`cot_usdindex_zscore` joined by **availability date**
      (`merge_asof` backward — each Tuesday close carries only the last COT
      reading already public, the same-week as_of report published ~3 days later
      excluded; the weekly analogue of `add_cot_features`, unit-tested by
      `test_weekly_cot_asof_join_backward_no_lookahead`). Chronological 70/80
      split at weekly resolution (992 analysis weeks 2007-06→2026-06; train 694,
      **val 99**, test 199 reserved and untouched).
      **This is a SEPARATE hypothesis family** (different target horizon → not
      comparable to the daily/volatility bars): logged ONLY in the new
      `results/cot_weekly_hypothesis_log.csv` at alpha=0.05 (first test; a future
      weekly-COT test becomes #2 and tightens it). The daily
      `feature_hypothesis_log.csv` and `volatility_hypothesis_log.csv` were NOT
      touched.
      **Pre-registered ONE battery, run once, no iterating** (Spearman primary +
      logistic corroborating; decision governed by the primary to bar cherry-
      picking):
        * PRIMARY Spearman rho on validation — `cot_eur_zscore` rho **+0.061**
          CI [−0.125, +0.243]; `cot_usdindex_zscore` rho **+0.061** CI
          [−0.145, +0.257] — **both straddle 0**.
        * CORROBORATING logistic(2 z-scores → direction): val acc **0.5354** ==
          the train-majority baseline exactly (Δacc **+0.0000**, CI
          [−0.152, +0.172], McNemar **p=1.0**) — the model collapses to the base
          rate; the z-scores add no separating signal.
      **Verdict DROP — no weekly COT edge.** Stated power caveat: 99 validation
      weeks only detect |rho| ≳ 0.2, so a null is *weak* evidence of absence, not
      proof. Research-only outcome — no model, no variant, no serving change (as
      pre-committed). A genuine future test needs either a longer forward window
      or a properly-designed weekly model; this side-check just says the raw
      signal is not obviously there.

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

- [x] **H1 frozen 'as of' date — stale cache served on every prediction (live
      dashboard report, 2026-07-10).** The dashboard showed "DATA 'AS OF'
      2026-07-07" on 2026-07-10: the H1 panel had been re-serving the same
      pre-retrain session for three days. Root cause: `load_h1_frame` was
      cache-first — the live fetch sat in an `elif` that was unreachable
      whenever `results/eurusd_h1.csv` existed, so the only thing that ever
      refreshed the H1 stream was the full retrain (`_train_pipeline.py`),
      which happened to rewrite the cache as a side effect. Fix in
      `src/h1_features.py::refresh_h1_frame`, now the inference load path:
      (1) live-first with the cache as fallback; (2) a mandatory staleness
      gate — the live chain is hit only when the cache's last COMPLETE session
      (same `MIN_HOURS>=12` rule as `aggregate_daily_features`) is behind the
      expected latest weekday session, so dashboard loads don't refetch when
      current; (3) thin live pulls are merged onto cached history (dedup by
      index, live wins) and the merged frame is rewritten to the cache, so the
      SMA504/RSI trailing warm-ups can never be silently truncated (the H1
      analogue of the old daily SMA_200 warm-up bug, §4.5.1); (4) the response
      `h1` block now carries `data_source` ("live" / "cache" /
      "live+history_backfill"), surfaced in the UI panel — same transparency
      convention as the daily `data_source` and `macro_source`. Training keeps
      cache-first `load_h1_frame` (the pipeline refreshes explicitly).
      Regression tests: `test_h1_inference_refreshes_stale_cache_live_first`,
      `test_h1_staleness_gate_skips_live_fetch_when_cache_current`,
      `test_h1_thin_live_fetch_backfills_history_from_cache`. 55 tests pass;
      live `/api/predict` end-to-end confirmed the served H1 day advanced
      2026-07-07 → 2026-07-09 (the correct latest complete session).

## Working rules

- Before each item: re-read the matching skill (`ml-practical-methodology` or
  `gbm-boosting-theory`) so the change stays grounded in it, not just memory.
- After each code change: run `python -m pytest -q`, fix anything that breaks, add
  test coverage if the change needs it.
- After each item is verified working: mark it `[x]` here with a one-line note on
  what changed and why, then commit that item alone.
- If interrupted, this file is the handoff document — resume by reading it, not by
  re-deriving the plan.
