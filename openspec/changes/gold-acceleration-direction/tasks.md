# Tasks

## 1. Package and features

- [x] 1.1 Create `src/gold_accel/` with:
  - `__init__.py` (isolation contract docstring, `load_config`, `repo_path`);
  - `accel_config.json`, holding the feature names, the 60-day profile, the seven
    models with their columns and years, the walk-forward settings (warm-up 3 years, M15
    warm-up 1 year), stats (block 120 rows / 20 days, 2000 draws, seed 42), costs (floor
    0.0062 %, stress 0.0122 %), the window as-of labels 14–20 and the paths.

  Verify:
  - a config test asserts each value;
  - a static test finds no write target under `models/`, no `config.json` or
    `gold_config.json` write, no import of `src.gold.versions`, `src.gold.prereg*` or
    the loggers, and no buy or sell wording.
- [x] 1.2 In `features.py`, implement `accel_1h`, `activity_rel`, `accel_expected_next`
  (trailing 60-trading-day profile, current day excluded) and `accel_m15`, and
  `build_inputs(h1, m15, columns)`. The last returns gold `Inputs` with exactly those
  columns, gold's `eligible` rule and `complete`.

  Verify with tests:
  - changing every bar after t leaves every feature at t unchanged;
  - multiplying prices by 3 leaves every feature unchanged;
  - a hand series gives `activity_rel` = 2.0;
  - `accel_m15` is NaN when an M15 bar is missing;
  - `assert_scale_free` accepts every column list.

## 2. Walk-forward

- [x] 2.1 In `walkforward.py`:
  - pin the H1 and M15 inputs under `research_models/gold/accel/inputs/` with digests
    in the manifest;
  - fit and predict every model and year with `src.gold.challengers.make(base, gold_cfg,
    {'columns': cols})`, resumable through `done.json`, with failures recorded and never
    filled;
  - reuse the `gold-window-strategy` `h1_lstm` predictions only when the configuration
    and input digests match.

  Verify with tests using a fake `make`:
  - the fit ends before the year;
  - a resumed year is not refit;
  - a failed year has no predictions;
  - the M15 models start in 2024;
  - nothing is written outside `research_models/gold/accel/` and `results/gold/accel/`.
- [x] 2.2 Run `python -m src.gold_accel.walkforward` on this PC, the GBM models first,
  then `h1_lstm_accel` (and `h1_lstm` if not reused).

  Verify:
  - every (model, year) has `done.json`;
  - the log shows the fit ranges;
  - `git status` shows no change under `models/`, `src/gold/` or `results/gold/`
    outside `accel/`.

## 3. Report

- [x] 3.1 In `backtest.py`, implement the scoring of design D5 (same rows, hit rate, AUC,
  Brier, coin, training majority, paired block-bootstrap of with minus without, all
  hours, window and per year) and the one cost check of D6, and render
  `results/gold/accel/backtest.md` with the label in every section.

  Verify with tests on synthetic predictions:
  - identical models give a difference of exactly 0;
  - a perfect model gives hit rate 1;
  - the window view uses only as-of labels 14–20;
  - an interval that holds 0 is worded as not distinguishable;
  - the report has no buy, sell or "edge" wording.
- [x] 3.2 Run `python -m src.gold_accel.backtest`, and append a dated section to
  `results/gold/RESULTS.md` with the main table and the plain reading.

  Verify:
  - `backtest.md` exists with every model and view;
  - the RESULTS section quotes its numbers exactly.

## 4. Documentation and integration

- [x] 4.1 Add a paragraph to `ARCHITECTURE_DOCS.md` §3.12 and one bullet under gold item 7
  in `CLAUDE.md`.

  Verify:
  - `pytest tests/test_gold_accel_*.py tests/test_gold_*.py` passes;
  - `openspec validate gold-acceleration-direction --strict` is clean;
  - the registered gold artifacts, `gold_config.json` and the registries are
    byte-identical to before (`git diff --quiet`).
