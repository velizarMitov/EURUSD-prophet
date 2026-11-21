# Tasks

## 1. Scaffolding and protection

- [x] 1.1 Create `src/fx_session/` with `__init__.py` and `fx_session_config.json`. The config holds:
  - pairs `["EURUSD", "GBPUSD"]` and the factor pair `AUDUSD`; history start `2010-01-01`, with the stop date `2011-01-01`;
  - entry label hour 14 and exit label hour 21, the session windows (Asia 0–8, London 9–13, data hour 14), the ATR period 14 and the 20-day range window;
  - the ordered feature list;
  - folds from `2013-01-01`, 6 months each, validation fraction 0.15, embargo 1 day;
  - the LightGBM settings of design D5 and the MLP settings of D6, with `allow_cpu = false`;
  - scoring: alpha 0.0125, block 5, 10 000 resamples, seed 20261122; parity tolerance 1.0 pp.

  Add `config_digest` and `load_config`. Verify with `tests/test_fx_session_config.py`, which checks that the config loads, the digest is stable, and the alpha is 0.05/4.
- [x] 1.2 Add `tests/test_fx_session_protected.py`. It digests `models/`, `config.json`, every other family's registry, `results/fomc_dates.csv` and the pooled caches before and after a no-op family run. It also asserts that `src/fx_fib/fx_fib_config.json` still matches the digest in `results/fx_fib/walk_forward_run.json`. Verify it passes.

## 2. Market data

- [x] 2.1 Implement `data_loader.py`:
  - an MT5 H1 refresh for EURUSD, GBPUSD and AUDUSD into `results/fx_session/data/<PAIR>_h1.csv`, with the columns time, OHLC and spread;
  - reads from position 1, with the count clamped to `maxbars − 1`;
  - append-only, with overlap-mismatch counting;
  - the point size and the start date written to `refresh_log.csv`;
  - the stop when any pair starts after 2011-01-01;
  - symbols resolved via `resolve_symbol`, with every pair refused if one is unresolved;
  - the clock checked with `verify_clock`.

  Verify with fake-MT5 tests covering append, the forming bar, a shallow history stop, an unresolved symbol, a shifted clock, a declared broker gap kept unchanged and an undeclared thin month refused.
- [x] 2.2 Implement `mid_close` and `mid_open` from bid plus half the spread × point, and the trading-day calendar with the entry, exit and eligibility rules. Verify with tests:
  - the spec's mid example;
  - a DST-mismatch day that keeps the 14 and 21 label hours;
  - an early-close day excluded as an incomplete session;
  - joint eligibility across both pairs;
  - the per-week clock gate: a Monday 00:00 open, a factor pair off the clock, and a mismatch week opening at 23:00 each make the week an unverified clock week.
- [x] 2.3 Run the real refresh. Verify each cache starts by 2012-01-31 (owner decision: history from 2012 after the GBPUSD 2011 clock shift), only the declared GBPUSD 2014-03 gap is accepted, there are no duplicate labels, the clock match rate is printed, and the refresh log holds the point sizes. Record the actual start dates in `results/fx_session/README.md`.

## 3. Features

- [x] 3.1 Implement `features.py` for the price-structure block. It covers the Asia, London and data-hour moves; the range ratios; the position in the London range; yesterday's move, range and close location; the move from yesterday's close; the broke-high and broke-low flags; and ATR as a fraction of the close. Use D1 bars and ATR from fx_fib's `build_d1` and `atr`, read-only. Verify with hand-computed synthetic-day tests and the scale test (all prices × 2 leaves the features unchanged).
- [x] 3.2 Add the dollar factor (pre-session and data-hour) and the pair residual. Verify with the spec's common-move scenario: three pairs +0.5 ATR give a factor of 0.5 and residuals of 0.
- [x] 3.3 Add the calendar flags: day of week, NFP by the first-Friday rule, FOMC from `results/fomc_dates.csv` (scheduled days only), and DST mismatch from the `zoneinfo` offset gap. Verify with tests: 2026-03-10 gives mismatch 1, and a mid-June day gives 0; a listed FOMC day gives 1; the first Friday gives NFP 1.
- [x] 3.4 Implement `assemble`, which raises on a missing or extra column, and the incomplete-input exclusions with counted reasons. Verify with the truncation-invariance test: cutting the history after D's entry bar leaves D's row unchanged, for 20 random days on both pairs. Run on the real caches and print the row counts and exclusion reasons.

## 4. Target, label report and power

- [x] 4.1 Implement `labeling.py`. It builds the binary mid-sign target from the 14:00 close to the 21:00 close, drops and counts ties, builds the bid-sign twin for the parity check, and checks that consecutive targets do not overlap. Verify with tests for the spec's up-day, flat-session and consecutive-days scenarios.
- [x] 4.2 Write `results/fx_session/labels.md`. It holds, per pair: eligible days; ties; the up-rate overall and per year; the mid-versus-bid up-rate gap; excluded days by reason; DST-mismatch day count; and the power table (MDE in accuracy and log-loss from the out-of-sample day counts, alpha 0.0125, 80 % power). It runs the config guard and carries the gross header. Verify with a test that the report has these sections and uses no model output, and that the 3 450-day case gives about 52.6 %. Then generate it on the real data.

## 5. Folds and models

- [x] 5.1 Implement `splits.py`. It builds 6-month folds from 2013-01-01 with shared boundaries, the last-15 % validation slice, a 1-day embargo on both gaps, and a RobustScaler fit on the fit slice only. Verify with tests for shared boundaries, the day gaps and scaler isolation.
- [x] 5.2 Implement `model_lgbm.py` with the fixed D5 settings, early stopping on validation and no class weights, plus the fit-slice prior. Verify with tests that two runs with the same seed give identical probabilities and the prior equals the fit up-rate.
- [x] 5.3 Implement `model_mlp.py` per D6. It is binary with BCEWithLogits, runs on CUDA with no CPU fallback, uses no class weights, a 5-seed mean ensemble and the fx_fib determinism helpers. Save artifacts under `research_models/fx_session/<PAIR>/fold_XX` with a SHA-256 manifest. Verify with GPU tests:
  - a same-seed rerun matches within 1e-5;
  - CUDA unavailable raises;
  - a manifest mismatch refuses loading;
  - a partial ensemble raises.

## 6. Walk-forward, scoring and registry

- [x] 6.1 Implement `walk_forward.py --stage lgbm`. It runs every fold for both pairs and writes `oos_lgbm.csv.gz` (prior and LightGBM probabilities, mid and bid targets, entry and exit prices) and `run_lgbm.json` (config digest, feature digest, row counts). It runs the guard first. Verify with a synthetic-data test of the full stage and the guard refusal.
- [x] 6.2 Implement `score.py --stage lgbm` and `--stage mlp`:
  - one-sided circular block bootstrap (block 5, alpha 0.0125), reusing fx_fib's `test_cell` read-only;
  - the registry `results/fx_session/fx_session_hypothesis_log.csv`, written with mode 'x' per cell;
  - notes with the decision basis, the gross statement and the prior facts (horizon-study session 4 h and 6.5 h, H1 direction family, fx-fibonacci DROPs);
  - `NOT TRIGGERED` rows for MLP cells whose LightGBM cell is DROP, with no alpha reassignment.

  Verify with tests:
  - a clear edge gives KEEP and zero edge gives DROP;
  - a second scoring is refused and the registry is byte-identical;
  - a changed config is refused by the label, walk-forward and score stages;
  - the untriggered path writes NOT TRIGGERED;
  - other registries are unchanged.
- [x] 6.3 Implement `walk_forward.py --stage mlp`. It refuses without both LightGBM registry rows, trains only the KEEP pairs and writes `oos_mlp.csv.gz` and `run_mlp.json`. Verify with tests for the refusal, the KEEP-only training and the both-DROP path that trains nothing.

## 7. Report and docs

- [x] 7.1 Implement `report.py`, which writes `results/fx_session/walk_forward_report.md`. It contains:
  - the gross header, the verdict table or "Not scored yet", and the power table;
  - per pair, pooled and per-fold log-loss, Brier and accuracy with Wilson 95 % CI, accuracy per year and on DST-mismatch days;
  - reliability, the bid-versus-mid parity with the 1.0 pp flag, and the sign rule's gross pips and percent.

  Verify with tests that two builds differ only in the generation line, and that the parity flag triggers at 1.4 pp.
- [x] 7.2 Write `results/fx_session/README.md` (commands, data and clock, entry/exit, features, target, cells and the decision rule, the NFP approximation), a new `ARCHITECTURE_DOCS.md §3.15`, and CLAUDE.md invariant 10 (family name, alpha 0.05/4, gross, walk-forward only, conditional MLP, never `models/`). Verify that every path both documents name exists.

## 8. Real run and integration

- [x] 8.1 Run the real stage-1 walk-forward. Verify it completes, `run_lgbm.json` records both pairs' folds, and the OOS file has no NaN probabilities or duplicate days.
- [x] 8.2 Score the LightGBM cells once. Then run stage 2: train the MLP only for KEEP pairs, and score the MLP cells once or record them NOT TRIGGERED. Verify the registry has exactly four rows and the other families' registries are byte-identical.
- [x] 8.3 Generate the report twice and verify the two builds are identical apart from the timestamp. Run `python -m pytest -q tests/test_fx_session_*.py tests/test_fx_fib_protected.py`, and verify all tests pass and `git status --short models/` is empty.

## Workflow follow-up

- Report the four verdicts to the owner, gross of cost and not a trading result, with the power table beside them.
- Archive the change with `/opsx:archive` after the owner's review.
