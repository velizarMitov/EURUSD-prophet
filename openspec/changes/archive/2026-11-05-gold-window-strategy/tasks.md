# Tasks

## 1. Scaffolding and descriptive report

- [x] 1.1 Create `src/gold_window/` with `__init__.py` and `window_config.json`. The
  config holds:
  - the paths;
  - the window labels;
  - the four rules' parameters (80th percentile, median, 250 days, the 15:30–16:30 range,
    the 20:00 cut-off);
  - the bootstrap settings: 95 %, block 20, 2,000 resamples, seed 42;
  - the cost floor 0.0062 % and the stress 0.0122 %;
  - the walk-forward warm-up of 3 calendar years.

  It has no registry, registration or archive settings.

  Verify: a test loads the config, asserts each value, and asserts that no path names
  `models/`, a registry or a registration.
- [x] 1.2 Write `describe.py`, which reproduces the 2026-11-04 statistics from the MT5
  caches into `results/gold/window/DESCRIPTIVE.md`:
  - hour, session, weekday and month tables per span;
  - best-day concentration;
  - size quintiles;
  - the bid-limitation note.

  Verify:
  - two runs differ only in the timestamp line;
  - the session table matches the scratch figures (Asia +15.4 / +6.4 / +13.3 %, window
    −3.6 / +1.1 / −0.4 %) to 0.1;
  - every section carries the descriptive label.

## 2. Bars and spreads

- [x] 2.1 In `bars.py`, implement the year-chunked read-only MT5 fetch of H1 spreads and
  M15 bars into gitignored `research_models/gold/window/mt5/`, with digests in
  `artifact_manifest.csv`. Add the loaders that return H1 (cache + joined spread), D1 and
  M15 frames (design D1).

  Verify with tests using a fake MT5 module:
  - the fetch is chunked by year;
  - only time, OHLC and spread are kept;
  - labels are stored unchanged;
  - the H1 join keeps every cache row;
  - the manifest digest matches the file.
- [x] 2.2 Run the fetch. Report the H1-spread and M15 spans and the share of zero spreads
  by year.

  Verify: the manifest lists both files, and M15 starts on or about 2022-08-12.

## 3. Rules and simulation

- [x] 3.1 In `rules.py`, implement the four rules as pure functions from bars and
  forecasts to trade lists: entry time and price, side, exit time and price, gate value.
  Also implement window eligibility and the strictly-prior gate quantiles (D3).

  Verify with unit tests for every scenario in the spec, among them:
  - a closed and an open gate;
  - P(up) = 0.47 with the gate open giving a short;
  - no breakout;
  - the first break down at 17:15 Sofia, with later breaks ignored;
  - the Monday call coming from the last D1 bar before Monday;
  - the 250-day warm-up;
  - a missing 18:00 H1 bar excluding the day;
  - no position open after 23:00 Sofia.
- [x] 3.2 In `simulate.py`, implement bid-plus-spread fills, the per-trade cost floor and
  stress, and the daily series with zeros on non-trade days (D4).

  Verify with tests:
  - a zero-spread trade is charged 0.0062 %;
  - a buy pays the bar's spread and a sell does not;
  - a seven-trade H1 day sums into one daily value;
  - no swap is ever charged.

## 4. Walk-forward predictions

- [x] 4.1 In `walkforward.py`, implement the yearly expanding refit of `h1_lstm`, `d1_lstm`
  and the D1 GARCH through `challengers.make` (D2):
  - it holds `gpu_job()` when TensorFlow sees a GPU;
  - it checkpoints per year;
  - it writes artifacts, predictions and digests only under `research_models/gold/window/`
    and to `artifact_manifest.csv`.

  Verify with tests on a small synthetic series:
  - every fit's last target is labelled before its prediction year;
  - a year missing a seed yields no neural predictions;
  - a resume skips completed years;
  - nothing is written outside `research_models/gold/window/` and `results/gold/window/`.
- [x] 4.2 Run the walk-forward. Report the per-year fit times and that every year has all
  5 seeds, or name the years that do not.

  Verify:
  - the prediction files' digests are in the manifest;
  - `git status` shows no change under `models/`, in `research_models/gold/` outside
    `window/`, or in any gold registry or registration.

## 5. Backtest report

- [x] 5.1 In `backtest.py`, build the four daily series, the comparators (gate always open,
  always long in the window, the time-shift placebo) and the 95 % intervals. Render
  `results/gold/window/backtest.md` (D5).

  Verify with tests on synthetic inputs:
  - every section carries the descriptive label;
  - the words CLEARED and DROP never appear;
  - no buy or sell instruction appears;
  - a synthetic always-profitable rule shows a positive interval;
  - the placebo share is between 0 and 1;
  - no registry file is written.
- [x] 5.2 Run the backtest, and show the owner the report in Bulgarian.

  Verify: the report's trade counts match the trade lists, and the spans match the
  prediction years.

## 6. Documentation and integration

- [x] 6.1 Add the window-strategy subsection to `ARCHITECTURE_DOCS.md` §3.12 and one bullet
  to `CLAUDE.md` gold item 7. Record the descriptive summary in `results/gold/RESULTS.md`.

  Verify:
  - `pytest tests/test_gold_*.py` passes;
  - `openspec validate gold-window-strategy --strict` is clean;
  - `git status` shows changes only in `src/gold_window/`, `tests/test_gold_window_*`,
    `results/gold/window/`, the docs, `RESULTS.md` and `.gitignore`.
