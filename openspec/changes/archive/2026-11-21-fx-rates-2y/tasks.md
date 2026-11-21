# Tasks

## 1. Scaffolding and protection

- [x] 1.1 Create `src/fx_rates/` with `__init__.py` (`load_config`, `repo_path`, `config_digest`, `feature_list_digest`, `GROSS_HEADER`) and `fx_rates_config.json`. The config holds:
  - pairs `["EURUSD", "GBPUSD"]` with foreign series `{EURUSD: de2y, GBPUSD: uk2y}`;
  - the source URLs of design D3, the FX history start `2012-01-01` and `min_h1_per_d1 = 12`;
  - horizon 5, lag 1 FX day, fill limit 3 business days;
  - the windows 5/20/60, the z-score window 60 and the divergence window 252;
  - the ordered 12 features;
  - folds from `2015-01-01`, 6 months each, validation 0.15, purge 5, embargo 5;
  - the LightGBM settings of D7, and the MLP settings with `allow_cpu = false`;
  - scoring: alpha 0.0125, block 10, 10 000 resamples, seed 20261123, power 0.8.

  Verify with `tests/test_fx_rates_config.py`: the config loads, the digest is stable, alpha is 0.05/4, the block is at least 5, and there are no class weights.
- [x] 1.2 Add `tests/test_fx_rates_protected.py`. It checks that `models/`, `config.json`, every other registry, the fx_session caches and results, fx_fib's results and the production macro caches are unchanged by importing and configuring the family. It also checks that fx_fib's and fx_session's configs match the digests recorded with their verdicts, and that no family source names a protected write target. Verify it passes.

## 2. Yield data

- [x] 2.1 Implement `rates_loader.py`:
  - FRED CSV for `DGS2` and `DGS10`, dropping blank and `.` values;
  - Bundesbank REST CSV, skipping the metadata and `.` rows;
  - the Bank of England stdlib `.xlsx` reader: the "spot curve" sheet, the 2.0 column found by its 2.0→2.5 header, Excel dates, archive plus current-month with current-month winning;
  - append-only caches, revision counting and `refresh_log.csv`;
  - per-series failure isolation and a generic User-Agent.

  Verify with offline tests on small fixture files built in the test (FRED, Bundesbank and a minimal `.xlsx` written with `zipfile`):
  - parsing;
  - picking 2.0 and not the short-end sheet;
  - append and revision counting;
  - one failing source leaving its cache unchanged while the others refresh.
- [x] 2.2 Implement the causal alignment in `align.py`: FX D1 from the fx_session H1 caches (read-only, `min_h1_per_d1`, mid close), then for each FX date `t` the yields as of the previous FX date via a backward `merge_asof`, with the 3-business-day limit, `yield_stale_flag` and the "stale yields" exclusion. Verify with tests:
  - a same-day yield is not used;
  - a US holiday fills and flags;
  - a 5-business-day gap excludes the day;
  - no back-fill ever happens (the first rows before any yield are excluded);
  - a one-bar day has no D1 bar.
- [x] 2.3 Run the real refresh. Verify each cache's first and last date and the refresh log, and that UK 2Y matches the probe: 2010-01-04 ≈ 1.45, with no value from the short-end sheet. Record the sources, coverage and lags in `results/fx_rates/README.md`.

## 3. Features

- [x] 3.1 Implement `features.py` with the 12 columns of design D5 and `assemble`, which raises on an extra or missing column. Verify with tests:
  - hand-computed values for the spread level, the 5-day change, the z-score, the slope and the divergence;
  - the scale test (FX prices × 2 leave the features unchanged);
  - the missing-column error;
  - warm-up exclusion.
- [x] 3.2 Add the truncation test: for 20 random days, cutting the FX bars after `t` and the yields from `t` on leaves the row unchanged. Run on the real data, and print row counts, exclusions by reason and the feature summary with no constant column.

## 4. Target, label report and power

- [x] 4.1 Implement `labeling.py`: the 5-day mid-sign target in FX trading days, ties dropped and counted, and the `dataset()` join of features and targets with reasons. Verify with tests for the up-week and flat-week scenarios, and that no target looks past `t+5`.
- [x] 4.2 Write `results/fx_rates/labels.md` with the gross header; sources and coverage; per pair usable days, out-of-sample days, ties, up-rate overall and per year, stale-filled and stale-excluded days; and the power table (VIF from the training-era target sign autocorrelation at lags 1–4, minimum detectable accuracy on `n / VIF` and on `n / 5`). It runs the config guard. Verify with a test that the 3 080-day, VIF 3.7 case gives about 55 % and about 56 %, and that no model output is used. Generate it on the real data.

## 5. Folds and models

- [x] 5.1 Implement `splits.py` with shared 6-month folds from 2015-01-01, the last-15 % validation slice, and a 5 + 5 day purge and embargo on both gaps. Verify with tests for shared boundaries, gaps of at least 10 days, the absence of any fit target window reaching validation, and a scaler fit on the fit slice only.
- [x] 5.2 Wire LightGBM, the prior and the MLP through the fx_session model and artifact helpers (read-only import) with this family's config and `research_models/fx_rates/`. Verify with tests that a same-seed LightGBM run is identical, the prior equals the fit up-rate, and the MLP refuses to run without CUDA.

## 6. Walk-forward, scoring and registry

- [x] 6.1 Implement `walk_forward.py --stage lgbm`. It writes `oos_lgbm.csv.gz` and `run_lgbm.json`, with the config and feature digests, the fx_session cache digests and the yield cache digests. It runs the guard first. Verify with a synthetic full-stage test and a guard-refusal test.
- [x] 6.2 Implement `registry.py` and `score.py --stage lgbm|mlp`:
  - block 10, alpha 0.0125, mode 'x' per cell;
  - notes with the gross statement and the prior facts (#1, #3, horizon-study daily cells, fx-fibonacci, fx-session);
  - NOT TRIGGERED rows with no alpha reassignment;
  - accuracy recorded with a Wilson interval on `n / VIF`.

  Verify with tests:
  - KEEP on a planted edge and DROP on none;
  - a second scoring refused with the registry byte-identical;
  - a changed config refused by the label, walk-forward and score stages;
  - the both-DROP path;
  - other registries unchanged.
- [x] 6.3 Implement `walk_forward.py --stage mlp`. It refuses without both LightGBM rows and trains only the KEEP pairs. Verify with tests for the refusal, KEEP-only training on the GPU, and the both-DROP path that trains nothing.

## 7. Report and docs

- [x] 7.1 Implement `report.py`, which writes `results/fx_rates/walk_forward_report.md`. It contains:
  - the gross header, the verdicts or "Not scored yet", and the power table;
  - pooled and per-fold log-loss and Brier;
  - accuracy with a Wilson interval on `n / VIF`, accuracy per year and on stale-flag days;
  - reliability;
  - the weekly rule per offset and averaged, against an always-long baseline.

  Verify with tests that two builds differ only in the generation line, and that the weekly rule's pips and offsets are computed correctly on a toy series.
- [x] 7.2 Complete `results/fx_rates/README.md`, add `ARCHITECTURE_DOCS.md §3.16`, and add CLAUDE.md invariant 11. Verify that every path the three name exists.

## 8. Real run and integration

- [x] 8.1 Run the real stage-1 walk-forward. Verify it completes, both pairs' folds are recorded, and the OOS file has no NaN or duplicate days.
- [x] 8.2 Score the LightGBM cells once, then run stage 2 and score or record the MLP cells. Verify the registry has exactly four rows and every other registry is byte-identical.
- [x] 8.3 Generate the report twice and verify the builds are identical apart from the timestamp. Run `python -m pytest -q tests/test_fx_rates_*.py tests/test_fx_session_protected.py tests/test_fx_fib_protected.py`, and verify all tests pass and `git status --short models/` is empty.

## Workflow follow-up

- Report the four verdicts to the owner beside the power table, gross of cost and not a trading result.
- Archive with `/opsx:archive` after the owner's review.
