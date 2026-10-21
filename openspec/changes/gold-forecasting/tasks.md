# Tasks

## 1. Package scaffold and isolation guard

- [x] 1.1 Create `src/gold/__init__.py` (module docstring stating the isolation contract) and `src/gold/gold_config.json`. The config holds: symbol aliases `["GOLD","XAUUSD"]`, D1/H1 bar counts (H1 below 100,000), split fractions 0.70/0.80, clock threshold 0.95, the GBM grid, the LSTM configs from design D6, seeds 42–46, bootstrap resamples 2,000, and every results path. Verify `python -c "import json; json.load(open('src/gold/gold_config.json'))"` succeeds.
- [x] 1.2 Add `tests/test_gold_isolation.py`. It asserts that `api.py`'s model path, `src/inference.py`, `_train_pipeline.py` and `src/paper_trading.py` never import `src.gold`, that `src/gold/` contains no write path under `models/` and no reference to `config.json` or `results/forward_eval/`, and that the existing protected-set fixtures still pass. Verify `python -m pytest -q tests/test_gold_isolation.py tests/test_h1_production.py -k "protected or gold"` passes.
  - Note 2026-10-21: `test_protected_set_is_sha256_identical` is red before this change. Commit `418fe94` (Copilot) committed the undeclared 2026-10-14 retrain, and all 19 mismatching pinned files are under `models/`. Gold changes nothing there, and the four gold isolation tests pass. Re-check at 11.2 once the retrain is declared or reverted.

## 2. Data acquisition and clock guard

- [x] 2.1 Implement `src/gold/data.py`:
  - resolve the symbol through the ordered aliases with `pooled_h1_data.resolve_symbol`;
  - fetch D1 and H1 with `copy_rates_from_pos` behind `assert_coverage`;
  - keep only `time, open, high, low, close`;
  - write an append-only cache to `results/gold/gold_{d1,h1}.csv`, fall back to it when
    MT5 is unreachable, and fail clearly when neither MT5 nor a cache is available.

  Verify with `tests/test_gold_data.py`: a fake MT5 returning volume and spread fields
  yields a cache whose columns are exactly the five names; there is a cache-fallback
  case and a no-source error case. Then run once live and check `results/gold/gold_h1.csv`
  has about 70,500 rows from 2013-08-08.
  - Done 2026-10-21: live D1 has 3,466 closed bars (2013-08-08 → 2026-10-20). Live H1 has
    70,015 bars from 2015-01-02, not 2013-08-08: the leading daily-like "H1" months are
    trimmed and Good Friday closures are exempt (design D2).
- [x] 2.2 Implement `src/gold/clock.py::verify(h1)`. It returns the share of weekly opens labelled Monday 00:00 or Sunday 23:00 and raises below 0.95. Verify with `tests/test_gold_clock.py`: a synthetic week set passes at 0.99 and fails at 0.90, and the live cache reports about 0.994.
  - Done 2026-10-21 with the stricter predicted-hour method (design D3). The live cache
    reports 0.976 over 624 weeks; the misses are a 14-week broker-clock anomaly in winter
    2016/17 plus two holiday opens. Task 8.1 adds the logger's live-window check (last 26
    weeks at 0.6).

## 3. Features, targets and eligibility

- [x] 3.1 Implement `src/gold/features.py`. H1 features come from `pooled_h1_model.compute_pooled_features` unchanged. D1 features are the scale-free set in design D4, plus a guard that rejects raw-level column names. Verify with `tests/test_gold_features.py`: OHLC × 3.0 gives identical features (both cadences); a frame with `close` or `SMA_50` as an input column is refused; no column contains `volume` or `spread`; altering bars after t leaves the features at t unchanged.
- [x] 3.2 Implement targets (via `forecast_eval.targets`: log return %, direction, |return| %) and the H1 eligibility rule in design D5. Verify with tests: the last bar before a daily break, the first bar after a weekend and a DST-mismatch-week break are all ineligible, and an ordinary mid-session hour is eligible. Run the rule on the live cache, record the eligible share, and confirm about 6,400 eligible H1 validation rows.
  - Live 2026-10-21: 91.9 % of H1 rows are eligible, and 6,392 eligible H1 validation rows.
- [x] 3.3 Implement the split helper (fractions from config, with boundary purge at h = 1). Verify that a test shows no validation index in training, that scalers fitted on training are unchanged when validation rows are perturbed, and that the D1 split gives about 2,425 / 347 / 693 rows.

## 4. Challengers and baselines

- [x] 4.1 Implement `d1_gbm` and `h1_gbm`:
  - XGB classifier plus pseudo-Huber regressor;
  - the declared grid tuned by inner `TimeSeriesSplit(5)` on training rows only;
  - every grid point appended to `results/gold/trial_log.csv` through
    `overfit.append_trials(path=…)`.

  Verify that a test with shuffled labels gives a validation accuracy in [0.48, 0.52],
  and that the trial log gains exactly one row per grid point.
- [x] 4.2 Implement `d1_lstm` and `h1_lstm`: multi-task heads for return (Huber), direction (BCE) and volatility (softplus, MAE), with early stopping on the last 1/8 of training and a 5-seed mean that is all or nothing. Verify with tests: a missing seed makes the challenger emit no prediction, the outputs carry `p_up`, `ret_pct` and `vol_pct`, and a 2-seed CPU smoke fit on a small slice finishes in under 2 minutes.
- [x] 4.3 Implement `d1_garch` (GARCH(1,1), train-only fit, one-step forecasts by filtering forward without refit) and `h1_garch_hod` (returns ÷ train-only hour-of-day mean |r|, GARCH(1,1), rescaled). Verify with tests: the parameters are identical when validation rows are perturbed, and the forecasts are positive and finite on the live caches.

## 5. Pre-registration (before any validation score)

- [x] 5.1 Implement `src/gold/prereg.py`:
  - build the power table with `prereg.n_required` and `fixed_point_family` on the
    measured eligible validation n;
  - write `results/gold/PRE_REGISTRATION.md` and `.json`, covering cells, families, α,
    verdict rules, the 0.95 clock threshold, eligibility, the grid and configs, the CI
    method, and the expected outcomes in design D9;
  - provide `scoring_allowed()` via `committed_and_clean`.

  Verify with a test that `scoring_allowed()` is False for an uncommitted or modified
  file and True for a committed, clean one.
- [x] 5.2 Generate the pre-registration and create the three empty registries (`results/gold/gold_{direction,return,volatility}_hypothesis_log.csv`). Have the owner review them, then commit them by hand with an explicit message and nothing else in the commit. Verify `git log -1 --stat` lists only those files, and that the D1 direction cells read UNDERPOWERED — NO DECISION while the H1 direction cells are admitted.
  - Deviation 2026-10-21: Copilot committed the registration inside `ab420d3` together with code, caches and three EUR/USD files, not on its own. It was still committed before any score, and the gate, the config digest and the data pins all check out. Recorded in `results/gold/RESULTS.md`; history is not rewritten.

## 6. Validation stage

- [x] 6.1 Implement `src/gold/study.py::run(stage)`, with `python -m src.gold.study --stage validation`:
  - check the clock, then call `scoring_allowed()`;
  - fit on [0:70] and score [70:80] on eligible rows, against each cell's baseline;
  - compute the verdicts in design D7;
  - append the registry rows;
  - write `results/gold/validation_report.md` with the descriptive extras: hour-of-day
    concentration flag, PBO and DSR.

  Verify with tests that scoring refuses without a committed pre-registration and that
  registry rows are appended, never rewritten. Then run it live and confirm every
  admitted cell has exactly one registry row.
- [ ] 6.2 Commit the validation registries and report by hand, with an explicit message. Write any correction to `results/gold/RESULTS.md`, never into the pre-registration. Verify `git diff HEAD~1 -- results/gold/PRE_REGISTRATION.md` is empty.

## 7. One-shot test and final artifacts

- [ ] 7.1 Implement `--stage test`. It refuses when `results/gold/test_report.json` exists in HEAD or when any validation registry row is uncommitted. Otherwise it refits on [0:80], scores [80:100] once, and writes the report. Verify with tests that a second run refuses and changes nothing. Then run it live once and commit the report.
- [ ] 7.2 Implement `--stage final`. It refits on [0:100] into `research_models/gold/<cell>/` and writes SHA-256 digests to `results/gold/artifact_manifest.csv`. Verify that the manifest digests match the files on disk and that `git status` shows nothing under `models/`.

## 8. Forward logger

- [x] 8.1 Implement `src/gold/forward_logger.py`:
  - lock, clock check and manifest digest check;
  - forecasts for the newest closed H1 bar and day whose target is still open, with a
    `phase` stamp;
  - gap and failure records;
  - settlement with the D5 targets and eligibility.

  Verify with tests: no row is written when the target bar has closed; a digest mismatch
  fails only that cell; a missed bar yields a gap and no back-fill; rows that border a
  break settle as ineligible; nothing under `results/forward_eval/` is touched.
- [ ] 8.2 Add `scripts/gold/install_task.ps1` and `uninstall_task.ps1` for `EURUSDProphet-GoldLogger` (hourly at :03, using the repo's `.venv` Python). Add `results/gold/forward/RUNBOOK.md`. Verify that `Get-ScheduledTask EURUSDProphet-GoldLogger` exists after install and is gone after uninstall, and that one manual run appends rows with `phase=dry_run`.
- [ ] 8.3 Run the dry run for at least two weeks. Then write `results/gold/forward/dry_run_coverage.md`, covering the share of bar closes logged per cadence and failures by cell. Verify that the report exists and that no `dry_run` row was ever scored.

## 9. Forward pre-registration

- [ ] 9.1 Write `results/gold/forward/PRE_REGISTRATION.md`, covering:
  - the forward cells and families;
  - the power table at the measured coverage;
  - a 3-year cap and a minimum 6-month window;
  - frozen artifacts with their manifest digests;
  - the verdict rules;
  - "gross of cost — not a trading result".

  Commit it by hand on its own. Verify that forward scoring refuses before the commit
  and is allowed after it (test plus live check).
- [x] 9.2 Implement the forward scoring report (`python -m src.gold.forward_logger --report`). It produces registered-phase rows only and no P&L, pips or Sharpe. Verify with a test that the report never contains those words or figures and that dry-run rows are excluded.

## 10. Read-only gold view

- [x] 10.1 Implement `src/gold/summary.py::build()`. It takes the newest row per cell whole, omits finished windows on the bar-label clock, derives the implied price, and attaches recorded verdicts and the "research — no proven edge" label. Verify with tests:
  - a blank p_up is not filled from an older row;
  - a closed target bar is omitted;
  - 4,131.00 with a +0.04 % return gives 4,132.65;
  - the build changes no file and imports no TensorFlow or XGBoost module;
  - an empty log yields the no-forecast state.
- [x] 10.2 Add `GET /api/gold` and `GET /gold` to `api.py`, add `static/gold.html` (bilingual BG/EN, with no spread, pips or P&L), and add one link in `static/index.html`. Verify with `TestClient`: both routes return 200 and the page renders the verdict labels. Then check `/gold` on the running app.
  - 2026-10-21: verified through TestClient on `api.app` (both routes return 200; labels shown; no cost words). The server that was running predates the routes and returns 404 until restarted; the launcher restarts it automatically when `.py` files are newer.

## 11. Documentation and integration

- [x] 11.1 Add a gold section to `ARCHITECTURE_DOCS.md`: the package layout, the data facts, the three families, the stages, the logger and the view. Add a short gold paragraph under Production Methodology in `CLAUDE.md`: its own families, gross of cost, never under `models/`. Verify that both documents name the registry paths and the scheduled-task name exactly as implemented.
- [ ] 11.2 Run the full suite `python -m pytest -q`. Verify it is green, that the protected-set fixtures pass, and that `git status` shows no change under `models/`, `config.json` or `results/forward_eval/` from this change.
