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

## Backlog — Fibonacci retracement + Williams fractal features (added 2026-07-26, hypothesis #7, VERDICT: DROP)

Genuinely new information for the direction/return family: the GEOMETRY of
recent swing structure (where price sits vs. the last confirmed swing high/low
and the Fibonacci retracement grid), derived from OHLC alone. New module
`src/fibonacci_fractals.py`.

- [x] Step 1 — Williams 5-bar fractal detection with the confirmation lag as
      the explicit design centre: a fractal at bar i is a strict extremum of
      high/low[i-2:i+3], so it is only KNOWABLE at bar i+2. The reveal walk
      (`confirmed_high_low_levels`, `_swing_walk`) reveals the fractal at index
      t-2 exactly at step t, so a fractal forming at i is INVISIBLE on bars i,
      i+1 and first usable on i+2. A dedicated unit test asserts this directly
      (`test_fractal_confirmation_lag_no_lookahead`), mirroring the FRED/COT
      no-look-ahead guards. All features are neutral 0 / NaN-safe until a
      confirmed structure exists, so the modeled row set is unchanged.
- [x] Step 2 — HYPOTHESIS #7: `fractal_breakout_up` / `fractal_breakout_down`
      (close vs. most recent confirmed high/low fractal) + `dist_to_nearest_fib_pct`
      (signed, swing-range-normalized distance to the nearest retracement level
      of the most recent confirmed 2-point swing), bundled as ONE hypothesis
      (three views of one swing-geometry fact → one Bonferroni slot). ADD-test
      via `src/ablation.py::run_addition_test` (paired bootstrap 2000 resamples
      + exact McNemar), validation slice [70:80] only, test block reserved —
      exactly the macro/COT/FOMC convention. Bar tightened to **0.05/7 ≈ 0.0071**.

      | family | hypothesis # | bar | result | evidence |
      |---|---|---|---|---|
      | direction/return | #7 | 0.05/7 = 0.0071 | **DROP** | Δacc = +0.0035, 95% CI [−0.0187, +0.0257] (includes 0); McNemar b=49/c=46, p=0.8376 ≫ bar; ΔAUC +0.0047 CI [−0.0136, +0.0228] |

      Run ONCE, no threshold/window tuning after seeing the result. Registered
      as `feature_hypothesis_log.csv` #7. NOT added to FEATURE_COLUMNS, NOT
      served, NOT in any variant — a null recorded, not a failure hidden.
- [x] Step 3 — HYPOTHESIS #8 (Fibonacci extension/projection from a confirmed
      3-point swing A→B→C, `dist_to_nearest_fib_extension_pct`) was BUILT and
      unit-tested this pass (`add_fibonacci_extension_features`, same
      confirmation-lag guard now on 3 swing points, with a chronology-order
      assertion) but, per the **pre-registered contingency**, is **NOT tested**:
      it runs only if #7 clears its bar, and #7 is DROP. A more discretionary
      3-point feature is not worth a hypothesis slot when the simpler 2-point
      version already failed. Contingency recorded in the module docstring, in
      `feature_hypothesis_log.csv`'s notes for #7, and here. Next
      direction/return bar if #8 is ever spent: 0.05/8 = 0.00625. Full suite
      green (80 tests).

## Backlog — VIX (CBOE Volatility Index) regime features (added 2026-07-26, hypothesis #8, VERDICT: DROP)

Genuinely new information for the direction/return family: broad EQUITY risk
sentiment (the "fear gauge"), unlike price, rates, positioning, or swing
geometry. Ingested through the SHARED FRED framework (no parallel fetch):
`config.json macro.features.vix` → `src/macro_data.py::_combine_vix` (raw level,
cache `results/vix.csv`), with the two stationary transforms living downstream in
new module `src/vix_features.py`.

- [x] STEP 0 — VERIFY BEFORE BUILDING (the COT/FOMC "probe, don't assume"
      precedent). EURUSD D1 (src/live_data.py: MT5 TIMEFRAME_D1 / yfinance daily,
      tz-naive) closes at the retail-FX rollover **~17:00 ET**; VIXCLS is the CBOE
      close **~16:15 ET** — a ~45-min, DST-fragile margin. A **live FRED probe on
      2026-07-26 (Sun)** returned the most recent VIXCLS print as **2026-07-23
      (Thu)** — Friday 07-24's print was still absent two days later, confirming
      FRED's **business-day publish lag**. **DECISION (conservative, err later):**
      a print dated D is available only on **D + 1 business day** → day D's bar
      uses **day D-1's VIX**. Pinned by
      `test_vix_availability_is_one_business_day_after_the_print_no_lookahead` and
      `test_vix_value_never_usable_on_the_bar_it_would_leak_into`.
- [x] Step 1 — Two candidate features, bundled as ONE hypothesis (one Bonferroni
      slot): `vix_zscore` (trailing rolling z-score, window **756 / min 252**
      trading days — VIX has genuine multi-year regime drift, so a raw level is
      non-stationary, the COT z-score treatment not the stationary-differential
      pass-through) and `vix_change_pct` (day-over-day % change, the shock
      component). Both computed on the **native business-day cadence** (never the
      ffilled FX-daily series, whose weekend/holiday duplicate bars — the history
      carries Sunday bars — would corrupt the 756-day window), then stamped at the
      availability date and as-of ffilled onto daily bars. Missing feed →
      neutral 0 (graceful degradation), guarded by
      `test_add_vix_features_neutral_zero_when_feed_unavailable`.
- [x] Step 2 — ADD-test via `src/ablation.py::run_addition_test` (paired bootstrap
      2000 resamples + exact McNemar), validation slice [70:80] only, test block
      reserved. Bar tightened to **0.05/8 = 0.00625**.

      | family | hypothesis # | bar | result | evidence |
      |---|---|---|---|---|
      | direction/return | #8 | 0.05/8 = 0.00625 | **DROP** | Δacc = −0.0117, 95% CI [−0.0327, +0.0105] (includes 0); McNemar b=43/c=53, p=0.3584 ≫ bar; ΔAUC +0.0040 CI [−0.0134, +0.0226] |

      Run ONCE, no window/transform tuning after seeing the result. Registered as
      `feature_hypothesis_log.csv` #8. The point estimate is actively negative —
      consistent with equity fear being a *volatility* event, not a next-day
      *directional* EUR/USD signal, and with the efficient-market result of §4.2.
- [x] Step 3 — NOT added to `FEATURE_COLUMNS`, NOT served, NOT in any variant, no
      retrain. The raw VIX level is deliberately kept OUT of
      `features.MACRO_MERGE_COLUMNS` (guarded by
      `test_vix_features_stay_out_of_direction_return_models`): the shared macro
      fetch maintains `results/vix.csv`, but nothing merges the level into the
      model frame, so predictions are byte-identical. A null recorded, not a
      failure hidden. Next direction/return bar: 0.05/9 ≈ 0.00556. Full suite
      green (86 tests).
- [x] Step 4 — SEPARATE follow-up test in the VOLATILITY family (2026-07-26,
      `results/volatility_hypothesis_log.csv`, NOT `feature_hypothesis_log.csv`
      — a distinct target/metric/bar per the family rules). Rationale: equity-vol
      → FX-vol spillover is a much better-established relationship than
      equity-vol → FX *direction* (which just dropped above) — a genuinely
      different, better-targeted hypothesis, not a re-test of the same idea.
      Reused `src/vix_features.py` / `results/vix.csv` and the conservative D-1
      availability convention EXACTLY as built for Step 0–3 above — no new
      fetch, no new look-ahead surface. Wired as one bundled candidate
      (`vix_zscore` + `vix_change_pct`) into `build_volatility_matrix`'s
      `extra_feature_columns`, tested via the existing
      `run_candidate_feature_tests` 5-seed multi-task LSTM ensemble methodology
      (`python -m src.volatility candidates vix`) — same paired bootstrap 2000
      resamples on validation[70:80], base ensemble freshly retrained on the
      same 5 seeds (42–46), test block reserved. Confirmed family size = 7
      before running → this is volatility hypothesis **#8**, bar
      `0.05/8 = 0.00625`.

      | family | hypothesis # | bar | result | evidence |
      |---|---|---|---|---|
      | volatility | #8 | 0.05/8 = 0.00625 | **DROP** | base MAE=0.185044%/R²=+0.1452 vs +VIX MAE=0.187440%/R²=+0.1466; ΔMAE=−0.002396% CI[−0.006231, +0.001763] (includes 0); ΔR²=+0.0014 CI[−0.0444, +0.0426] (includes 0); frac(ΔMAE>0)=0.044 |

      Run ONCE, no window/seed tuning after seeing the result. Registered as
      `volatility_hypothesis_log.csv` #8. Even the better-motivated hypothesis
      (equity-vol → FX-vol, not FX-direction) found nothing the price-only
      5-seed ensemble doesn't already carry — consistent with the RSI_14 /
      BB_percent_b / FOMC / COT nulls already in this family (§3.5). NOT added
      to any input set, `models/volatility/` UNTOUCHED, no retrain. Next
      volatility bar: 0.05/9 ≈ 0.00556. Full suite green (86 tests).

## Backlog — Volatility ensemble's own forecast as a direction/return input (added 2026-07-26, hypothesis #9, VERDICT: DROP)

CROSS-FAMILY REUSE, not a new raw data source: does conditioning the
direction/return model on "how much movement is expected tomorrow" — a signal
this project already validated in the SEPARATE volatility family (§3.5, the
ONLY neural family with a CI-confirmed edge over its baseline) — help predict
direction? Rationale: trend persistence vs. mean-reversion often differs by
volatility regime, so this is a mechanistically distinct hypothesis from
another fresh external feed, reusing information already proven out elsewhere
in this project.

- [x] Implementation — NO retraining of the volatility ensemble. New functions
      in `src/volatility.py`:
      `load_frozen_volatility_ensemble` loads the PRODUCTION `models/volatility/`
      artifacts (5 seed `.keras` models + `lag_scaler`/`lag_pca`/`global_scaler`,
      fit ONCE on train[0:80%] by `train_production_volatility_model`) via
      `joblib.load` / Keras `load_model` only — no fitting.
      `batch_predict_frozen_ensemble_vol_pct` runs pure batch INFERENCE
      (`.transform()` + `.predict()` only) across the FULL historical row set to
      produce `predicted_vol_pct` for every row — the exact same
      transform/predict calls `src/inference.py::_predict_volatility` makes for
      one live window, vectorized over history instead. Same idiom
      `src/ablation.py::build_matrix` already uses for its own once-fit PCA
      applied across train+val+test — not a new or different look-ahead
      surface; row t's prediction still depends only on rows <= t
      (`make_sequences`' existing sliding-window geometry). `add_volatility_forecast_feature`
      neutral-fills warm-up rows (before the first full `time_steps` window) to
      0.0, the same convention every other candidate module uses.
      `test_frozen_volatility_ensemble_batch_inference_never_fits` monkeypatches
      `StandardScaler.fit`/`fit_transform` and `PCA.fit`/`fit_transform` to raise,
      then exercises the full code path (with synthetic already-fitted stand-in
      artifacts) across a full synthetic row range standing in for
      train+val+test at once — confirming no fitting step is ever triggered,
      regardless of which rows are passed.
- [x] Wired into `src/ablation.py::build_matrix`'s `extra_feature_columns`
      handling (mirrors the FOMC/COT/fibonacci/VIX branches) and a
      `python -m src.ablation volforecast` CLI entry. Single-column bundle
      (`predicted_vol_pct`) — one Bonferroni slot. Confirmed family size = 8
      before running → this is direction/return hypothesis **#9**, bar
      `0.05/9 ≈ 0.00556`. ADD-test via `run_addition_test` (paired bootstrap
      2000 resamples + exact McNemar), validation[70:80] only, test block
      reserved, run ONCE, no tuning after seeing results.

      | family | hypothesis # | bar | result | evidence |
      |---|---|---|---|---|
      | direction/return | #9 | 0.05/9 ≈ 0.00556 | **DROP** | Δacc = −0.0093, 95% CI [−0.0304, +0.0105] (includes 0); McNemar b=35/c=43, p=0.4282 ≫ bar; ΔAUC +0.0009 CI [−0.0136, +0.0146] |

      Registered as `feature_hypothesis_log.csv` #9. `volatility_hypothesis_log.csv`
      and the weekly COT log untouched (this is a direction/return hypothesis
      only). Even reusing an already-proven signal from a different family
      found nothing the existing 27-column input set doesn't already carry.
- [x] NOT added to `FEATURE_COLUMNS`, NOT served, NOT in any variant, no
      retrain. Even had this cleared, shipping it would introduce a
      **serving-order dependency** (the volatility ensemble must run before
      direction/return can consume its output) — a discussion point, not
      automatic, on a clear KEEP; moot here since it DROPped. Next
      direction/return bar: 0.05/10 = 0.005. Full suite green (88 tests).

## Backlog — H1 harmonic-pattern event-conditional model, TRIPLE-BARRIER labeling (added 2026-07-26, OWN family, hypotheses H1.1 + H1.2, VERDICT: BOTH DROP)

A genuinely new EVENT UNIVERSE and TARGET, not another daily/H1 feature: does
price behave differently in the ~120 H1 bars (~5 days) after a classical
XABCD harmonic-pattern reversal signal completes? Its own hypothesis family
(`results/harmonic_pattern_hypothesis_log.csv`, first budget alpha=0.05,
split across two sequentially-scaled sub-hypotheses run together this pass,
alpha=0.025 each) — separate from the daily direction/return, daily
volatility, and weekly-COT families; none of those logs are touched.

**Scope note (owner-confirmed):** `src/harmonic_patterns.py` (XABCD ratio
scoring) did NOT already exist in this project — the initial task assumed it
did ("reuse … UNCHANGED"). Confirmed absent by a full-repo search, flagged to
the owner, and built as explicitly NEW, unvalidated code this pass (its own
module docstring says so plainly). Only the fractal/swing PRIMITIVES it calls
(`src.fibonacci_fractals.detect_fractals` / `_push_swing` / `CONFIRMATION_LAG`)
have a prior track record — genuinely reused UNCHANGED, not reimplemented.

- [x] Step 1 — `src/harmonic_patterns.py`: an XABCD pattern is 5 alternating
      confirmed swing points (X→A→B→C→D), scored against 4 published
      Fibonacci ratio templates (Gartley/Bat/Butterfly/Crab — NOT tuned
      against this project's data) via `r_AB=|AB|/|XA|`, `r_BC=|BC|/|AB|`,
      `r_CD=|CD|/|BC|`, `r_AD=|AD|/|XA|`; `best_fit_score` = best-template
      match in [0,1]; `direction` = +1 bullish (D is a LOW) / −1 bearish (D is
      a HIGH) — D's kind alone determines the sign, by construction.
      Confirmation lag INHERITED unchanged from `src.fibonacci_fractals`: the
      whole pattern is only confirmed at `D_idx + CONFIRMATION_LAG` (2 bars),
      since D is itself a fractal. Event filter (pre-registered, no post-hoc
      tuning): `best_fit_score >= 0.5`. On `results/eurusd_h1.csv`
      (60,000 H1 bars): **14,144 raw XABCD completions, 4,161 clearing the
      0.5 filter.**
- [x] Step 2 — `src/triple_barrier.py` (Lopez de Prado, event-source-agnostic,
      new but generically reusable): `r_ewma_std` = EWMA std of H1 log
      returns, span=24; `horizon_vol = r_ewma_std * sqrt(120)` — square-root-
      of-time scaled to the 120-bar holding horizon. **This replaced an
      earlier plain-ATR draft per owner review**: ATR measures PER-BAR range,
      not the dispersion an entry should expect over its full multi-bar
      holding period, so it is the wrong volatility unit for a fixed-horizon
      barrier — the sqrt-time-scaled EWMA is the measure genuinely matched to
      the horizon. `entry` = close at `confirmed_at_idx`; `target = entry *
      exp(direction * 1.5 * horizon_vol)`; `stop = entry * exp(-direction *
      1.0 * horizon_vol)`; time barrier = 120 bars, fixed. Same-bar
      target+stop ambiguity (OHLC cannot resolve true intrabar order) ties
      toward the STOP — conservative, never overstates the edge. Time-barrier
      resolution requires the signed move to clear the transaction cost,
      **explicit pip→price conversion**: `config.json`
      `paper_trading.spread_pips = 1.5`; EURUSD 1 pip = 0.0001
      (`src.paper_trading.PIP_SIZE`) → **1.5 × 0.0001 = 0.00015** raw price
      units — a move smaller than that is not a realizable win, mirroring how
      `paper_trading.py` already nets cost rather than scoring a bare
      `sign(>0)`. Events within 120 bars of the end of history are EXCLUDED
      (never padded): **14 of 4,161** filtered events, leaving a **final
      4,147-event labeled dataset** (label 1 rate 45.3%).
- [x] Step 3 — `src/harmonic_event_check.py`, the pipeline + BOTH
      sub-hypotheses on the identical event subset / chronological 70/15/15
      split (2,902 train / 622 val / 623 test, test reserved untouched) /
      identical 8 features (`r_AB, r_BC, r_CD, r_AD, best_fit_score,
      direction, swing_duration_bars, norm_amplitude` — the last is the
      XA-leg amplitude normalized by the SAME `horizon_vol` already computed
      at the event, not a second ad hoc normalizer) / `class_weight=
      'balanced'` on both models (**rationale**: the closer 1.0x stop is
      geometrically more likely to be touched before the farther 1.5x target
      under a pure random walk, independent of any real edge — without
      balancing either model could trivially collapse to the majority class
      and become indistinguishable from the baseline it's judged against,
      silently underpowering the test) / `random_state=42` (project
      convention).

      **H1.1 (linear baseline) — LogisticRegression**, judged against the
      train-majority-class baseline.
      **H1.2 (non-linear) — feed-forward MLP**, RAW PYTORCH (deliberately not
      Keras, unlike every other neural model in this project — see
      *"H1.2 framework correction"* below for why), NOT an LSTM — these are
      already-extracted cross-sectional per-event ratios, not a time series:
      Linear(16,L2=1e-3 via `weight_decay`)→ReLU→Dropout(0.3)→
      Linear(8,L2=1e-3)→ReLU→Dropout(0.3)→Linear(1)→Sigmoid, Adam lr=0.001,
      ≤100 epochs, early-stop patience=10 on val loss, batch_size=32 (this
      project's existing H1-LSTM convention) — architecture FIXED, no tuning
      after results. H1.2's **PRIMARY** decision test compares against
      **H1.1's own predictions on the identical validation rows** (not a
      fresh baseline) — the real "is the extra complexity worth it" question,
      and ALONE governs H1.2's verdict; MLP-vs-majority-baseline is
      corroborating context only (anti-cherry-pick rule, same convention as
      the weekly-COT Spearman-primary / logistic-corroborating test).
- [x] Step 4 — Paired bootstrap 2000 resamples + exact McNemar, BOTH
      hypotheses, alpha = 0.05/2 = **0.025** each (CI width itself
      alpha-scaled — 97.5% — matching the volatility-family / weekly-COT-
      extremes convention: a stricter alpha only ever raises the bar). Run
      ONCE, no tuning of the event threshold, EWMA span, sqrt-horizon
      scaling, barrier multipliers, spread-cost threshold, class weighting,
      or MLP architecture after seeing results.

      | hypothesis | comparison | val acc (challenger / reference) | Δacc | 97.5% CI | McNemar p | verdict |
      |---|---|---|---|---|---|---|
      | H1.1 LogisticRegression | vs train-majority baseline | 0.4936 / 0.5305 | −0.0370 | [−0.1061, +0.0322] | 0.2671 | **DROP** |
      | H1.2 MLP (PRIMARY) | vs H1.1's own val predictions | 0.4678 / 0.4936 | −0.0257 | [−0.0643, +0.0145] | 0.1812 | **DROP** |

      H1.2's corroborating check (vs majority baseline) was ALSO negative
      (Δacc −0.0627, McNemar p=0.0882) — no ambiguity to arbitrate; both
      paths agree. Non-event random-sample baseline (b), descriptive only:
      label==1 rate 0.4608 (n=4,147) — close to the event dataset's own
      45.3%, consistent with the target/stop geometric distance asymmetry
      being the dominant driver of the label distribution, not the harmonic
      signal itself.
- [x] Step 5 — Both rows registered in
      `results/harmonic_pattern_hypothesis_log.csv` (n=1, n=2). No model,
      serving, or API change regardless of outcome (both DROPped anyway).
      13 new unit tests: exact-Gartley ratio/direction hand-check, degenerate-
      swing None-guard, confirmation-lag truncation (event invisible before
      `D_idx+2`, present exactly at it), no-look-ahead future-truncation
      equivalence, all 4 triple-barrier outcomes (target-first, stop-first,
      time-win, time-loss) plus insufficient-history exclusion plus the
      short-direction mirror, the `sqrt(120)` scaling math (both a direct
      unit check and an end-to-end barrier-placement check), EWMA causality,
      the generic paired-bootstrap helper's swap-test (proving H1.2's PRIMARY
      comparison is a genuine row-for-row comparison against whatever
      predictions are passed in, never a hidden independent baseline), and
      the FEATURE_COLUMNS-exclusion guard. Full suite green (102 tests).

**H1.2 framework correction (same day, before shipping).** The first draft of
H1.2 used Keras (`tensorflow.keras`, matching every other neural model in
this project). Owner review flagged that a Keras implementation cannot
demonstrate two PyTorch-specific correctness pitfalls that matter for any
FUTURE model in this codebase written in raw PyTorch, so H1.2 was rewritten
in raw PyTorch specifically to guard against them, in `train_h1_2_mlp`
(`src/harmonic_event_check.py`): **(1)** PyTorch does NOT auto-toggle
Dropout between train/eval like Keras's `.fit()`/`.predict()` — `model
.train()` before every training batch, `model.eval()` + `torch.no_grad()`
before every validation-loss check and the final prediction; forgetting this
leaves Dropout active during validation with no error, silently corrupting
both the early-stopping signal and the reported accuracy. **(2)** the
architecture keeps an explicit `Sigmoid` output (matching the original
`Dense(1, sigmoid)` spec), so the loss must be `BCELoss` (not
`BCEWithLogitsLoss`), and plain `BCELoss` has no `pos_weight` argument (that
is `BCEWithLogitsLoss`-only) — class balancing is done via an explicit
PER-SAMPLE weight tensor (`weight[i] = class_weight[y[i]]`) rebuilt each
batch and passed to `BCELoss(weight=...)`. Validation loss (the early-
stopping signal) is deliberately UNWEIGHTED, matching Keras's own actual
default (`class_weight` only affects the training loss, never `val_loss`).
CPU-only by choice (matches this project's other neural models' determinism
convention, even though CUDA happens to be available in this environment).
L2=1e-3 is applied via Adam's `weight_decay` — the standard PyTorch idiom,
stated honestly as NOT numerically identical to Keras's loss-added
`kernel_regularizer=l2` (same L2 strength, different mechanism). **The whole
pre-registered hypothesis was re-run once, in full, with this corrected
implementation** (a genuine correctness fix, not post-hoc tuning of a
hyperparameter): H1.1 reproduced identically (unaffected — still
scikit-learn), H1.2's numbers changed (Δacc +0.0177 → −0.0257) but the
**verdict is unchanged: DROP**. `results/harmonic_pattern_hypothesis_log.csv`
row 2 reflects the corrected run; this document's numbers above are the final
ones.
      Power caveat stated plainly: 622 validation events is a small-n family;
      both DROPs are correspondingly weak (not strong) evidence of absence.

## Backlog — ZigZag swing basis for the harmonic-pattern model (added 2026-07-26, SAME family, hypotheses H1.3 + H1.4, VERDICT: BOTH DROP)

An ALTERNATIVE swing-point source for the same harmonic-pattern hypothesis
family (`results/harmonic_pattern_hypothesis_log.csv`) — hypotheses #3/#4,
family grown from 2 to 4, bar tightened to alpha = 0.05/4 = **0.0125** each.
Not a new event universe or target: EVERYTHING else in the pipeline (event
filter `best_fit_score>=0.5`, triple-barrier labeling, the 8 features,
chronological 70/15/15 split, `class_weight='balanced'`, `random_state=42`,
LogReg + PyTorch MLP) stays byte-identical to H1.1/H1.2 — only the swing
points feeding `score_xabcd` change.

**Why try a second swing basis.** A Williams fractal's 5-bar window is
FIXED-LENGTH regardless of volatility. On H1 bars specifically, a fixed
5-bar window is short enough that it likely flags a lot of noisy
MICRO-swings — local wiggles that are technically "extrema" but not
genuinely meaningful market structure — feeding XABCD scoring with
unrepresentative points. A ZigZag whose reversal threshold ADAPTS to current
volatility (ATR-scaled) targets cleaner, more meaningful swings: a pivot
only confirms once price has genuinely reversed by a volatility-relative
amount, not merely "more extreme than its 2 immediate neighbors on each
side". On the real H1 data this produced markedly fewer, presumably
higher-quality swings: **6,969 raw XABCD completions (vs the fractal path's
14,144) → 2,307 clearing the 0.5 filter (vs 4,161) → 2,303 labeled events
(vs 4,147)**.

- [x] Step 1 — **Elevated look-ahead risk, stated honestly, and mitigated.**
      A Williams fractal's confirmation lag is FIXED (exactly 2 bars,
      trivial to reason about). A ZigZag pivot's confirmation lag
      (`reveal_bar - idx`) is VARIABLE and UNBOUNDED — a pivot can sit
      unconfirmed for a handful of bars or for hundreds, depending on how
      long price takes to reverse by the ATR-scaled threshold. This is a
      genuinely easier algorithm to get wrong in a way that REPAINTS (the
      classic ZigZag bug: compute local extrema over the WHOLE series first,
      then apply the threshold — using full hindsight, because "was this
      the local max" implicitly depends on bars that had not happened yet).
      Mitigation: new module `src/zigzag_swings.py` processes bars STRICTLY
      IN ORDER, one at a time — no "scan the whole array for extrema" step
      anywhere (the only vectorized computation is ATR itself, which is
      ITSELF purely causal — `.ewm`, reusing `src.features.py`'s exact
      `ATR_14` formula, so vectorizing that specific recurrence introduces
      no look-ahead). Every pivot carries both `idx` (where the extreme
      price occurred) and `reveal_bar` (when the reversal was actually
      detected) — the variable-length analogue of `CONFIRMATION_LAG`.
      Guarded by what is, deliberately, the **highest-priority test set in
      this whole family**: a pivot demonstrably invisible to any query
      truncated before its own `reveal_bar`
      (`test_zigzag_pivot_invisible_before_reveal_bar`), and — THE core
      repainting guard — a confirmed pivot's `(idx, level, reveal_bar)`
      provably IDENTICAL whether computed causally up to its own
      `reveal_bar` or with arbitrarily more future bars appended afterward
      (`test_zigzag_confirmed_pivot_is_stable_under_future_extension`), plus
      a threshold-sensitivity sanity check (a monotonic path with no
      reversal at all produces zero confirmed pivots).
- [x] Step 2 — Algorithm (pre-registered, fixed before running):
      `threshold[t] = 1.5 * ATR(14)[t]` (reuses this project's existing
      1.5x multiplier convention, ATR computed causally). Maintain a running
      candidate extreme + search direction; extend the candidate on a new
      intrabar extreme (`high`/`low`), confirm the reversal on `close`
      crossing the ATR-scaled threshold away from the candidate — then flip
      direction and reseed. Bootstrap (bar 0, "seeking a high") is an
      arbitrary but immaterial, explicitly stated choice. Pivots strictly
      alternate H/L by construction (no `_push_swing`-style same-kind
      collapse needed, unlike the independent fractal detector).
      New `src.harmonic_patterns.detect_harmonic_events_from_pivots` reuses
      `score_xabcd` UNCHANGED — only the swing-point source differs; an
      event's `confirmed_at_idx` is its own D pivot's `reveal_bar`, not
      `D_idx + CONFIRMATION_LAG`.
- [x] Step 3 — `src/harmonic_event_check.py`'s `build_event_dataset`/`run`
      parametrized by `swing_source` ('fractal' | 'zigzag'); the family's
      Bonferroni bar is now computed DYNAMICALLY at run time from however
      many distinct hypothesis names are already registered (matching
      `src.ablation.run` / `src.volatility.run_candidate_feature_tests`'s
      convention) rather than a hardcoded constant — H1.1/H1.2's
      ALREADY-LOGGED alpha (0.025) is never retroactively rewritten; only
      this new run is judged at the freshly-tightened bar. Run via
      `python -m src.harmonic_event_check zigzag`.

      | hypothesis | comparison | val acc (challenger / reference) | Δacc | 97.5%→98.8% CI | McNemar p | verdict |
      |---|---|---|---|---|---|---|
      | H1.3 LogisticRegression | vs train-majority baseline | 0.5130 / 0.5043 | +0.0087 | [−0.0928, +0.1072] | 0.8856 | **DROP** |
      | H1.4 MLP (PRIMARY) | vs H1.3's own val predictions | 0.5362 / 0.5130 | +0.0232 | [−0.0276, +0.0768] | 0.3497 | **DROP** |

      H1.4's corroborating check (vs majority baseline) was directionally
      positive but also not CI-confirmed (Δacc +0.0319, McNemar p=0.5012) —
      consistent, not ambiguous. Non-event random-sample baseline (b),
      descriptive only: label==1 rate 0.4637 (n=2,303) — close to this
      dataset's own 46.3%, the same target/stop geometric-distance-asymmetry
      pattern as the fractal run.
- [x] Step 4 — Both rows registered as `harmonic_pattern_hypothesis_log.csv`
      n=3/n=4. No model, serving, or API change regardless of outcome. 7 new
      unit tests (the reveal-lag invisibility check, the repainting guard,
      threshold-sensitivity zero-pivots, an ATR-formula-matches-`src.features`
      equality check, strict H/L alternation, `detect_harmonic_events_from_pivots`
      reusing `score_xabcd` unchanged with a variable `confirmed_at_idx`, and
      a swing-source routing check). Full suite green (109 tests). Cleaner
      swings did not translate into a detectable edge either — a materially
      different swing basis on the SAME idea still finds nothing, modest
      further evidence (not proof) that the underlying null is about the
      harmonic-pattern hypothesis itself, not an artifact of the fractal
      window's noise. Next bar in this family if a 5th hypothesis is ever
      spent: 0.05/5 = 0.01.

## Backlog — Fractal-breakout drift/continuation event-study (added 2026-07-26, NEW OWN family, VERDICT: DROP)

A genuinely different question from hypothesis #7 (`feature_hypothesis_log.csv`,
DROPped): #7 asked whether `fractal_breakout_up`/`fractal_breakout_down` help
predict the SINGLE NEXT day's direction as an input feature. This asks the
classic breakout-MOMENTUM thesis instead — conditional on a confirmed breakout
today, does price keep moving in that direction over the next few days? A
forward multi-day event-study, not a same-day feature-addition test, so it gets
its own brand-new family log
(`results/fractal_breakout_driftcheck_hypothesis_log.csv`) and does NOT touch
`feature_hypothesis_log.csv`, `volatility_hypothesis_log.csv`,
`cot_weekly_hypothesis_log.csv`, or `harmonic_pattern_hypothesis_log.csv`.

Research-only, regardless of outcome: no model, feature, or serving change at
this stage. A KEEP-signal would only be the TRIGGER to design a proper
dedicated event-conditional model later (mirroring how the harmonic-pattern H1
model was built, `src/harmonic_event_check.py`) — never an automatic
feature/serving change itself.

- [x] Step 1 — New module `src/fractal_breakout_driftcheck.py`. Reuses
      `confirmed_high_low_levels()`/`add_fibonacci_features()` from
      `src.fibonacci_fractals` UNCHANGED to get `fractal_breakout_up`/
      `fractal_breakout_down` per day on `results/eurusd_features.csv` (the
      same series hypothesis #7 used) — fractal detection itself is not
      rebuilt, and the confirmation-lag look-ahead guard is already baked into
      those columns by construction.
- [x] Step 2 — Event definition: day t is an event iff exactly one of
      `fractal_breakout_up[t]`/`fractal_breakout_down[t]` fires (`event_direction
      = +1`/`-1`); the rare day where BOTH fire (3 times in the full 1971-2026
      daily history) has an undefined direction and is excluded, counted
      separately rather than arbitrarily signed. For each event and horizon N
      in {2, 3, 5}: `signed_continuation_N = event_direction *
      log(close[t+N]/close[t])`.
- [x] Step 3 — SAME chronological daily split convention as every other family
      (`config.json`: train_fraction=0.80/val_fraction=0.10 ->
      train[0:70%]/validation[70%:80%]/test[80%:100%] RESERVED, identical
      formula to `src.ablation._canonical_split`). A validation-slice event
      whose forward window would cross INTO the reserved test block is
      excluded for that horizon even though the underlying CSV physically has
      more rows there (this is daily history extending to the present, not a
      short series) — the same `max_idx` boundary parameter in
      `compute_signed_continuation` also handles genuine "insufficient forward
      history" at the true end of the series, so there is only one place this
      look-ahead rule can be gotten wrong.
- [x] Step 4 — PRE-REGISTERED test: PRIMARY = mean(signed_continuation_3) over
      validation-slice events, paired bootstrap (2000 resamples), 95% CI;
      KEEP-signal only if the CI is entirely > 0. CORROBORATING (context only,
      never a second path to KEEP): the same statistic for N=2 and N=5 — if
      N=3 is null but N=2/N=5 look significant, the verdict stays DROP (same
      anti-cherry-pick convention as `harmonic_h1_2_mlp_vs_h1_1_primary` and
      every other bundled hypothesis). alpha = 0.05 (first hypothesis of this
      brand-new family).
- [x] Step 5 — One-shot run on real data: validation-slice raw event counts
      breakout_up=260, breakout_down=380 (both-flags-excluded=0) — a decently
      powered 640 total, no thin-tails caveat needed this time.

      | horizon | role | n used | mean signed_continuation | 95% CI | verdict contribution |
      |---|---|---|---|---|---|
      | N=2 | corroborating | 638 | −0.000233 | [−0.000771, +0.000343] | straddles 0 |
      | N=3 | **PRIMARY** | 637 | −0.000155 | [−0.000856, +0.000551] | straddles 0 → **DROP** |
      | N=5 | corroborating | 637 | −0.000112 | [−0.001001, +0.000762] | straddles 0 |

      All three horizons are directionally NEGATIVE (mild reversal, not
      momentum) but none clears the pre-registered bar — consistent with the
      rest of this project's near-efficient-market findings; a confirmed
      fractal breakout carries no detectable forward drift at 2/3/5-day
      horizons. `n_used` dips slightly below the raw event counts purely from
      the split-boundary exclusion near `val_end`, confirming that guard is
      live, not dead code.
- [x] Step 6 — Logged as `fractal_breakout_continuation_3day`, n=1, in the new
      `results/fractal_breakout_driftcheck_hypothesis_log.csv`. No other
      hypothesis log touched. 5 new unit tests (signed_continuation
      direction-sign construction for both up/down breakouts, both-flags-
      same-day exclusion, insufficient-forward-history exclusion, the
      validation/test split-boundary exclusion — proven against an identical
      unbounded run to show the data genuinely exists beyond `val_end` and is
      deliberately not read —, and a split-formula equality check against
      `src.ablation`'s convention). Full suite green (114 tests). No model,
      feature, or serving change. This DROP does not retire the underlying
      curiosity forever — it retires 3/2/5-day continuation specifically; a
      different horizon or a volatility-conditioned variant would be hypothesis
      #2 of this family (alpha tightening to 0.05/2 = 0.025).

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
      **Housekeeping note (2026-07-21):** the daily-COT commit `198fb3b` also
      incidentally restaged a full regeneration of `results/eurusd_h1.csv` (an
      unrelated H1-cache rewrite that was sitting in the working tree at commit
      time) — it is NOT part of the COT feature work. Left as published; no
      history rewrite.
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
- [x] **COT weekly hypothesis #2 — contrarian positioning EXTREMES —
      INCONCLUSIVE/underpowered (2026-07-21).** Second pre-registered test in the
      SAME weekly family (target = forward weekly return), so it tightens the
      family bar to **alpha = 0.05/2 = 0.025**; logged as row #2 of
      `results/cot_weekly_hypothesis_log.csv` (upsert-by-name, union columns), the
      daily `feature_hypothesis_log.csv` and `volatility_hypothesis_log.csv`
      untouched. Reused `build_weekly_frame`/`weekly_cot_target_frame` + the same
      70/80 split unchanged (added `run_extremes()` to `src/cot_weekly_check.py`).
      **Pre-registered (fixed before results, no tuning):** crowded-long = z>+1.0,
      crowded-short = z<−1.0 (~top/bottom 16% under normality — a priori, not
      threshold-scanned); contrarian hypothesis = crowded-long → NEGATIVE forward
      weekly return, crowded-short → POSITIVE. PRIMARY (governs verdict, bundles
      both instruments): spread = mean(fwd|z>+1) − mean(fwd|z<−1) per z-score,
      paired 2000-bootstrap 95% CI; KEEP-signal only if a CI is entirely below 0
      AND the point spread is negative. Corroborating exact binomial sign tests
      (context only). Pre-registered underpowered rule: <5 extreme weeks on a tail
      → INCONCLUSIVE, do NOT loosen the cutoff.
      **Outcome — INCONCLUSIVE / underpowered, and a clean vindication of the
      pre-registration.** The validation window (~2020-10→2022-08) had one-sided
      positioning: `cot_eur_zscore` was crowded-long **30** weeks but crowded-short
      **0** (spread undefined); `cot_usdindex_zscore` was crowded-short **27** but
      crowded-long only **2** (< the 5 minimum; 12% of bootstrap draws degenerate).
      Neither z-score had enough two-sided extremes for a stable CI, so the
      verdict is INCONCLUSIVE (`cleared_bar=False`), NOT a KEEP and NOT a clean
      DROP. The one point estimate that could be formed (usdidx spread **+0.0026**)
      was even the *wrong* sign for the contrarian story. A less disciplined pass
      would have dropped the cutoff to |z|>0.5 to manufacture rows — exactly the
      post-hoc tuning the pre-registration forbade; the threshold was left at 1.0
      and the thin/one-sided tails reported plainly. Research-only: no model, no
      variant, no serving change. A real test needs a longer/again-two-sided
      forward window. See ARCHITECTURE_DOCS.md §4.3.2.

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
