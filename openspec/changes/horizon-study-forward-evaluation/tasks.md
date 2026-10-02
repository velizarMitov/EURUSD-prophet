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

- [ ] 11.1 Run the logger in `phase = dry_run` for at least two weeks and deliver `results/forward_eval/dry_run_coverage.md` (fraction of bar closes logged, failures by model); verify the report exists and no `dry_run` row was ever settled for scoring

## 12. Pre-registration and verdicts

- [x] 12.1 Implement the power table and fixed-point family in `prereg.py` (Bonferroni, power 0.8, 3 pp over breakeven, 3-year cap, measured coverage); verify at full coverage it reproduces the design table (34 → 6 cells, alpha 0.00833, 0.5 / 1.1 / 2.2 years)
- [x] 12.2 Compute the volatility family power table from the development variance; verify a test on fixture variance gives the expected time-to-decision
- [ ] 12.3 Write and commit `results/forward_eval/PRE_REGISTRATION.md` (cells, estimand, cost level, coverage assumption, families, alpha, verdict rule, refit cadence, expected outcome) with empty `forward_hypothesis_log.csv` and `forward_vol_hypothesis_log.csv`; verify the commit exists before any scoring
- [x] 12.4 Implement the scoring-mode gate (refuse unless the registration commit is an ancestor of HEAD); verify the logger refuses to score without the commit and scores with it
- [x] 12.5 Implement the verdict engine (KEEP, DROP, undecided at the registered n only; interim figures labelled; "predictive, not cost-viable" label); verify tests show no verdict before n and the correct label for a cell above 50 % but below breakeven

## 13. Integration

- [x] 13.1 Run the full test suite; verify it passes, the pinned checksum tests pass without any re-baseline, the provenance check reports nothing undeclared, every existing hypothesis log is byte-identical, and the feature bar is still 0.05/9
- [x] 13.2 Update the test counts in `README.md` and `HOW_TO_RUN.md`, and add a short pointer to the program in `CLAUDE.md`; verify the documented count equals `pytest --collect-only -q`
