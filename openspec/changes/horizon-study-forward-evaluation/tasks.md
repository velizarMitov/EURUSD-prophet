# Tasks

## 1. Package scaffolding and safety guards

- [x] 1.1 Create the `src/forecast_eval/` package with the module layout from design D1 and add `research_models/` to `.gitignore`; verify `python -c "import src.forecast_eval"` succeeds and `git check-ignore research_models/x` matches
- [x] 1.2 Add an AST test that fails if any module in `src/forecast_eval/` references an MT5 trade function (`order_send`, `order_check`, `positions_*`, `orders_*`, `history_orders_*`); verify it raises on a probe source containing `mt5.order_send(...)` and passes on the package
- [x] 1.3 Add a test asserting no module in `src/forecast_eval/` contains a write path under `models/`, and that importing the package changes no file pinned by `tests/fixtures/*_protected_sha256.json`; verify with a negative probe and the existing checksum tests

## 2. Targets, costs and breakeven

- [x] 2.1 Implement horizon direction, return-percent and volatility targets on log returns in `targets.py`, with zero-move exclusion counted; verify with tests for h ∈ {1, 24}, the zero-move count, and an AST check that the package scales by 100 only inside `targets.py`
- [x] 2.2 Implement the cost model in `costs.py`: measured spread from `results/curl/m1_coverage.csv`, the `config.json` round trip as sensitivity, cost charged only on position change, and refusal when spread data is missing; verify with unit tests for each rule
- [x] 2.3 Implement the breakeven `0.5 + c / (2·E|r_h|)` per horizon; verify a test reproduces the design calculation on the EURUSD train slice (h = 1: 53.32 % at 0.5 pip, 59.97 % at 1.5 pip, ±0.01 pp)

## 3. Leakage-free splitting

- [x] 3.1 Implement the expanding-window purged walk-forward with purge = h and embargo ≥ h in `splits.py`; verify tests show no train/test label-window overlap at h = 24, embargo ≥ 48 at h = 48, and every train row preceding every test row
- [x] 3.2 Implement CPCV; verify N = 6, k = 2 yields 15 splits and 5 paths, and each path holds exactly one out-of-sample forecast per observation
- [x] 3.3 Implement label-uniqueness weights and effective n; verify all weights equal 1 at h = 1, effective n ≪ row count at h = 24, and parity with `h1_direction_model.mean_label_uniqueness` on identical spans

## 4. Benchmarks and statistical tests

- [x] 4.1 Implement the coin-flip, train-majority, zero-return random-walk and random-sign matched-turnover benchmarks in `benchmarks.py`; verify each random strategy's turnover equals the model's and the model's percentile is reported
- [x] 4.2 Implement the Pesaran–Timmermann test and the moving-block bootstrap (block ≥ h, refusal when n ≤ block) in `stats.py`; verify on synthetic perfect-skill and no-skill series, and a parity test against `h1_direction_model._block_bootstrap_delta` yields identical intervals
- [x] 4.3 Implement Clark–West for nested forecasts and Diebold–Mariano with the HLN correction for non-nested ones; verify the zero forecast against itself is non-significant, and that a nested comparison is routed to Clark–West
- [x] 4.4 Implement Bonferroni and Romano–Wolf step-down adjustment; verify Romano–Wolf adjusted p ≤ Bonferroni adjusted p, and a small global-null simulation keeps the family-wise error near 0.05

## 5. Overfitting diagnostics

- [x] 5.1 Implement the append-only trial log `results/horizon_study/trial_log.csv` and the Deflated Sharpe Ratio using its cumulative count; verify 40 prior trials plus 10 new gives a count of 50, and that no earlier row can be changed or removed
- [x] 5.2 Implement PBO via CSCV with S = 16; verify PBO ≈ 0.5 for a pure-noise candidate set and ≈ 0 when one candidate has real skill

## 6. Read-only feature adapters

- [x] 6.1 Implement `features.py` adapters over `compute_features` (price-only and with-macro), `compute_pooled_features`, `enrich_h1_with_indicators` and the `build_h1_datasets` features, discarding any target they return; verify features equal the production code's output on sampled bars
- [x] 6.2 Implement the feature-module digest used by the manifest and the logger guard; verify the digest changes when a probe copy of a feature module changes, and that no pinned file was modified

## 7. Challenger models

- [x] 7.1 Define the study-record schema (grid, per-challenger configuration starting from `config.json` values read-only, tuning grids and rule thresholds, cap = 3 years, seeds), written before the first fit; verify a test refuses a grid change after the first fit
- [x] 7.2 Implement the daily GBM challenger for both feature sets on CPU; verify fit/predict on a small fixture and identical configuration hashes across horizons
- [x] 7.3 Implement the daily multi-task LSTM challenger for both feature sets on CPU; verify fit/predict on a small fixture with a fixed seed reproducing identical predictions
- [x] 7.4 Implement the H1→daily ensemble challenger (XGBoost, Random Forest, SVM, LSTM); verify fit/predict on a fixture and that scalers are fitted inside the training fold only
- [x] 7.5 Implement the TI-LSTM challenger with a CPU path; verify the test runs with CUDA unavailable
- [x] 7.6 Implement the H1 next-bar GBM challenger; verify fit/predict on a fixture at h ∈ {1, 4}
- [x] 7.7 Implement the 5-seed volatility multi-task LSTM challenger, plus the GARCH(1,1) × day-of-week and random-walk volatility baselines fitted on the training fold only; verify with fixture tests that the baselines see no test rows
- [x] 7.8 Implement the Kronos adapter, run as-is with forecast length = h, which records a failure when torch or the checkpoint is absent; verify the existing `tests/test_external_kronos.py` still passes unchanged
- [x] 7.9 Implement the nested purged-CV tuning rule from design D9; verify exactly one configuration leaves each cell and both variants appear in the trial log

## 8. Horizon study run

- [x] 8.1 Implement `study.py` orchestration with per-cell failure isolation, writing artifacts to `research_models/horizon_study/` and the tracked manifest (path, SHA-256, family, horizon, configuration hash, training range, feature digests); verify an end-to-end run on a tiny synthetic dataset
- [x] 8.2 Run the full study on history and deliver `results/horizon_study/` curves (accuracy, AUC, gross/net per trade, breakeven at both costs, PT p-value, CPCV paths, PBO, DSR) with a DESCRIPTIVE banner and date range; verify every cell is present or failed with a reason, `GET /api/provenance` reports nothing undeclared, and every existing hypothesis log is byte-identical; record the refit compute time
- [x] 8.3 Write `results/horizon_study/README.md` stating that the curves are descriptive and describe the challengers, not production; verify the file exists and every path it cites exists

## 9. Forward logger

- [x] 9.1 Implement the read-only `mt5_reader.py` (bars with spread, symbol swap, server name, never the login); verify with an injected fake MT5 module, and that the AST guard from 1.2 covers it
- [x] 9.2 Implement the logger predict step (lock file, H1 and D1 cadence detection, latest manifest version with `train_end < as_of`, idempotent key, every spec field, `phase` flag); verify with a fake reader and clock that a repeat run is a no-op
- [x] 9.3 Implement gap records with no backfill; verify three missed bars produce three gap records and a later run never predicts them
- [x] 9.4 Implement the append-only settlement step from MT5 closes, excluding non-MT5 sources and other servers; verify a `yfinance` row is logged but never settled for scoring, and that existing rows are unchanged
- [x] 9.5 Implement per-model failure isolation and the feature-digest guard; verify one missing artifact yields one failure record while every other model's row is still written
- [x] 9.6 Add Windows Task Scheduler install/uninstall scripts and `results/forward_eval/RUNBOOK.md` (keep AutoTrading off, disable sleep, how to read gaps and failures); verify `schtasks /Query` shows the task after install and not after uninstall

## 10. Walk-forward refits

- [x] 10.1 Implement `refit.py`: monthly weekend refits on an expanding window to the last closed Friday bar, frozen configuration, new manifest versions, and the previous version kept on failure; verify with a fake clock that every version's `train_end` precedes the first prediction that uses it, and one version per month appears
- [x] 10.2 Add the scheduled refit task to the install script and the runbook; verify it is listed by `schtasks /Query`

## 11. Forward dry run

- [ ] 11.1 Run the logger in `phase = dry_run` for at least two weeks and deliver `results/forward_eval/dry_run_coverage.md` (fraction of bar closes logged per cadence, failures by model); verify the report exists and no `dry_run` row was ever settled for scoring. Started 2026-10-01 for H1 and D1; the M15 cadence joined 2026-10-02, so its own two weeks run to 2026-10-16

## 12. Pre-registration and verdicts

- [x] 12.1 Extend the power table and fixed-point family in `prereg.py` to the M15 cadence, taking the eligible-trade rate per horizon from measurement (7,523 / 3,632 / 1,816 / 778 / 259 a year at h = 1 / 2 / 4 / 8 / 16) rather than as rate/h, with the edge declared as 3 pp over 50 %; verify at full coverage it reproduces the design D13 table (46 → 10 → 12 cells, alpha 0.004167, n = 3,817, M15 at 0.51 / 1.05 / 2.10 years and H1 at 0.62 / 1.24 / 2.47 years)
- [x] 12.2 Compute the volatility family power table from the development variance; verify a test on fixture variance gives the expected time-to-decision
- [ ] 12.3 Write and commit `results/forward_eval/PRE_REGISTRATION.md` covering all 12 admitted cells including the six M15 session cells (cells, estimand, price definition, session window, cost level, coverage assumption per cadence, families, alpha, verdict rule, refit cadence, expected outcome per cell from the development study, and the bid-versus-mid parity tolerance) with empty `forward_hypothesis_log.csv` and `forward_vol_hypothesis_log.csv`; verify the commit exists before any scoring, that section 14 is complete first, and that the expected outcomes match design D13 (the two 15-minute cells and the three H1 GBM cells expected KEEP, the rest DROP)
- [x] 12.4 Implement the scoring-mode gate (refuse unless the registration commit is an ancestor of HEAD); verify the logger refuses to score without the commit and scores with it
- [x] 12.5 Implement the verdict engine (KEEP, DROP, undecided at the registered n only; interim figures labelled; "predictive, not cost-viable" label); verify tests show no verdict before n and the correct label for a cell above 50 % but below breakeven

## 13. Integration

- [x] 13.1 Run the full test suite; verify it passes, the pinned checksum tests pass without any re-baseline, the provenance check reports nothing undeclared, every existing hypothesis log is byte-identical, and the feature bar is still 0.05/9
- [x] 13.2 Update the test counts in `README.md` and `HOW_TO_RUN.md`, and add a short pointer to the program in `CLAUDE.md`; verify the documented count equals `pytest --collect-only -q`

## 14. M15 session layer

This section must complete before 12.3, because the registration fixes the family for
good (design D13, Migration Plan phase 1b).

- [x] 14.1 Implement `m15_data.py` aggregating `results/curl/raw/EURUSD_M1.parquet` to M15 bars that carry bid OHLC, the mid close (`close + spread/2`), summed tick volume and the median spread of their minutes, refusing a source with no spread field and failing with a message naming `pyarrow` when it is absent; verify the built series has 199,097 bars over 2018-08-08 → 2026-08-07, that `results/eurusd_m15.csv` is neither read for targets nor written, and that a spread-less probe source is refused
- [x] 14.2 Record the M1 source file's SHA-256 and row count in the study record so a regenerated copy can be proven identical; verify a test detects a one-row difference in a probe file
- [x] 14.3 Establish the bar-label clock from the weekly market boundary and record it with its evidence, refusing a run whose weekly open label stops matching; verify the measured boundary is `Sun 23:00` year-round and `22:00` in the US/EU daylight-saving mismatch weeks of March and late October, that only `Europe/Berlin` fits all four cases, and that a probe frame shifted by an hour is refused
- [x] 14.4 Implement the session window in label time (`14:30 <= label < 22:00` on weekdays, the owner's 15:30–23:00 Europe/Sofia) with a trade eligible only when its as-of bar AND its target bar fall inside the session on the same label date; verify the in-session as-of count is 62,238 bars over 2,075 session days, the eligible counts are 60,163 / 58,088 / 53,938 at h = 1 / 2 / 4, and that the window is the same label time of day in January and July
- [x] 14.5 Add the M15 cadence to the study-record grid {1, 2, 4, 8, 16, 26} and to the splitting, uniqueness and block-length arithmetic (block = max(h, 26), one session); verify purge and embargo leave no train/test label-window overlap at h = 26, uniqueness weights are all 1 at h = 1, and the grid cannot change after the first fit
- [x] 14.6 Implement the two session challengers (`m15_session_gbm`, `m15_session_lstm`) on M15 mid features including intrabar movement, tick volume, the bar's spread and its position within the session, configurations declared once and frozen across horizons, CPU-only; verify fit/predict on a fixture at h ∈ {1, 4}, identical configuration hashes across horizons, and that no feature reads a bar later than its as-of bar
- [x] 14.7 Implement the bid-versus-mid parity check: score every M15 cell on both price definitions, report both and their difference, and label a cell "spread artifact" when the gap exceeds the declared tolerance of 1.0 pp; verify the check flags the New York 17:00 rollover window (label hour 23: bid 72.2 % against mid 60.5 % at a 1-hour horizon) and flags no eligible in-session hour, where the two agree to within 0.69 pp
- [x] 14.8 Run the session study and extend `results/horizon_study/` with the M15 curves at both price definitions, the session breakeven at the measured session spread (0.50 pip median, 0.60 pip p90) and at the config round trip, PBO and DSR; verify every M15 cell is present or failed with a reason, the trial log grew by the cells fitted, every pinned file is byte-identical and `GET /api/provenance` reports nothing undeclared
- [x] 14.9 Add the M15 cadence to the forward logger (session-restricted bar closes, bid close and the bar's spread recorded, mid used at settlement, no gap records outside the session, a bar without a spread excluded from scoring) and to the refit scheduler; verify with a fake reader that an out-of-session close writes neither a prediction nor a gap, that a spread-less bar is excluded, and that the AST guard from 1.2 still covers the new modules
- [x] 14.10 Extend the scheduled task and `RUNBOOK.md` to the 15-minute cadence with the expected row counts per session day; verify `schtasks /Query` shows the M15 trigger and that a `dry_run` M15 row is never settled for scoring
- [ ] 14.11 Run the M15 dry run for at least two weeks and extend `results/forward_eval/dry_run_coverage.md` with M15 coverage and failures by model; verify the report covers every cadence separately and that the measured coverage is fed back into 12.1's power table before 12.3

## 15. Operator view

The owner asked for the forward log in an HTML UI, and for cost arithmetic to be
kept off their screen (2026-10-02). Added after section 14 was built.

- [x] 15.1 Implement `report.py` writing a self-contained `results/forward_eval/dashboard.html` from the logs only: the latest forecast per cell with the window it covers in Europe/Sofia, the running accuracy against the registered n with the remaining count, the session state, and the gaps and failures; verify it renders with no forward rows, with dry-run rows only, and that it contains no spread, breakeven or net-profit figure
- [x] 15.2 Regenerate the view at the end of every logging run without letting a view failure affect logging, and document it in `RUNBOOK.md`; verify a run still succeeds when the view raises, and that `api.py`, `src/inference.py` and `src/paper_trading.py` are byte-identical
- [x] 15.3 Serve the view from a separate read-only loopback server (`report.py --serve`, default port 8001) with a double-click launcher, re-rendering per request so a reload is always current; verify it returns the page with `Cache-Control: no-store`, 404s every other path, 405s a POST, binds only to 127.0.0.1, turns a render failure into a 500 and keeps serving, and imports no part of the served application. The owner chose this over a route in `api.py` (2026-10-02), which keeps the serving Non-Goals intact
