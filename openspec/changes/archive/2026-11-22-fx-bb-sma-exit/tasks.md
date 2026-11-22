# Tasks

## 1. Scaffolding and protection

- [x] 1.1 Create `src/fx_bb_sma/` with `__init__.py` (`load_config`, `repo_path`, `config_digest`, `feature_list_digest`, `COST_HEADER`) and `fx_bb_sma_config.json`. The config holds:
  - fx_bb's `pairs`, `pip`, `data`, `bands`, `atr_period`, `reference_quorum` and `features` blocks, copied unchanged;
  - the trade rule: the take profit at SMA20, a stop multiple of 1.0, a time stop of 10 valid days, the stop-wins tie rule and the half-spread cost model;
  - the exclusion window `t−19 … t+10`;
  - folds from `2015-01-01`, 12 months each, validation 0.15, a 5 + 5 business-day gap measured from the exit date;
  - fx_bb's LightGBM settings;
  - scoring: alpha 0.025, family size 2, block 10, 10 000 resamples, a fixed seed, power 0.8, a declared SD of 1.0 R and a parity tolerance of 0.05 R.

  Verify with `tests/test_fx_bb_sma_config.py`: the shared blocks equal fx_bb's config, alpha is 0.05/2, the block is 10, the time stop is 10, there are no class weights, and the digest is stable.
- [x] 1.2 Add `tests/test_fx_bb_sma_protected.py`. It checks:
  - `models/`, `config.json`, every other registry, and the fx_bb, fx_session, fx_rates and fx_fib results, caches and sources are unchanged by importing and configuring the family;
  - the four earlier configs match the digests recorded with their verdicts;
  - no family source names a protected write target.

  Verify it passes.

## 2. The trade simulation

- [x] 2.1 Implement `trades.py` with:
  - entries on first breaches, from fx_bb's indicators;
  - the no-overlap stream, where excluded trades hold their slot to `t+10`;
  - the H1 scan from the bar after the entry bar through the closing bar of valid day `t+10`, on mid high and low;
  - the stop-wins tie rule;
  - fills at the level, with the stop gap at the mid open;
  - the time stop at the D1 mid close of `t+10`;
  - gross and net results in price, pips and R, net of half the entry and half the exit spread;
  - the exclusions `open at data end`, `pair gap` and `corrupt spread`;
  - `exit_times`, which returns timestamps only.

  Verify with unit tests on synthetic H1 bars:
  - a take profit on day `t+3`;
  - a stop;
  - a same-bar conflict recorded as a stop;
  - a gap through the stop filled at the open;
  - a time stop at the close of `t+10`;
  - the entry bar is not scanned;
  - a breach during an open trade opens nothing, while a time-stop exit and a new entry at the same close are allowed;
  - the hand-computed net R of +0.98;
  - a corrupt-spread bar excludes the trade and still blocks the pair;
  - short trades mirror long trades.
- [x] 2.2 Implement the bid/ask control fill in `trades.py`. Verify with tests that a long enters at the bid plus the spread and exits on the bid, that a short mirrors it, and that with a zero spread the control equals the gross mid result.
- [x] 2.3 Add the real-data truncation test: for 20 random trades, removing every bar after the entry bar leaves the entry, the levels and `dist` unchanged. Run it on the real caches, and print the trade counts per pair and the exclusions by reason, with no result.

## 3. Power before any model

- [x] 3.1 Implement `labels.py`. It writes `results/fx_bb_sma/labels.md` and `labels.json` with the cost header. Per pair and pooled, overall, from 2015 and per year, it gives:
  - first breaches, trades, breaches skipped while a trade was open, exclusions by reason and distinct entry dates;
  - Cell 1's minimum detectable mean, with an SD of 1.0 R;
  - Cell 2's minimum detectable accuracy, each on trades and on dates.

  It runs the config guard first. Verify with tests:
  - 1 000 trades on 750 dates give 0.089 R and 0.102 R;
  - the report succeeds while the function that computes exit reasons and results raises if called;
  - a changed config is refused.

  Generate it on the real data.

## 4. Folds, model and bootstrap

- [x] 4.1 Implement `splits.py`: expanding 12-month folds over the entry dates, with the gap measured from each trade's exit date. Verify with tests that:
  - no training exit is within 10 business days of the next slice;
  - the 2019-12-10 → 2019-12-24 example is not in the 2020 fold's training data;
  - every out-of-sample trade appears once;
  - the folds are contiguous.
- [x] 4.2 Implement the day-block ratio bootstrap with 10-day blocks in `score.py`. It reuses or mirrors fx_bb's, with the block from this config. Verify with tests that:
  - with one trade a day it reproduces `circular_block_means` exactly;
  - a planted positive mean is KEEP and a zero mean is DROP for Cell 1;
  - Cell 2's side keeps only below zero.
- [x] 4.3 Wire the pooled LightGBM and the fit-slice win rate through fx_session's `model_lgbm` read-only, with this family's config and `research_models/fx_bb_sma/`. Use fx_bb's ten features and the target `net > 0`. Verify with tests that a same-seed run is identical and that the base rate equals the fit slice's win rate.

## 5. Walk-forward, scoring and registry

- [x] 5.1 Implement `walk_forward.py`. It runs the guard, and refuses before `labels.json` exists. It writes:
  - `oos_trades.csv.gz`, with `p_prior` and `p_lightgbm`;
  - `run.json`, with the config and feature digests, the six cache digests and the folds.

  Verify with a synthetic full-stage test and a guard-refusal test.
- [x] 5.2 Implement `registry.py` and `score.py --cell rule|model`. Each cell is scored once with mode `'x'`. Cell 1 saves `cell1_trades.csv.gz`, records the bid/ask control and the parity flag, and its note names the spread-only cost model and the earlier fx-bb-reversion DROP. Verify with tests that:
  - a second scoring is refused and the registry stays byte-identical;
  - a changed config is refused by the label, walk-forward and score stages;
  - the parity flag never changes the verdict;
  - other registries are unchanged.

## 6. Report and docs

- [x] 6.1 Implement `report.py`, which writes `results/fx_bb_sma/walk_forward_report.md`. It contains:
  - the cost header, the verdicts or "Not scored yet", and the power table;
  - the plain-language section in Bulgarian: trades, exit shares (take profit, stop loss, time stop), win rate, average win and loss in pips, profit factor gross and net, and total R gross and net;
  - the five example trades, chosen by the fixed rule;
  - breakdowns per pair, per year and per direction, the mean holding time, the number of conflict bars and the bid/ask line;
  - for Cell 2: log-loss and Brier per fold, reliability, and the trades above and below a win probability of 0.5.

  Verify with tests that:
  - two builds differ only in the generation line;
  - nothing is shown before scoring;
  - the exit shares add up to 100 %;
  - USDJPY and EURUSD pips are correct on a toy set;
  - the example rule picks all three exit reasons and both directions.
- [x] 6.2 Write `results/fx_bb_sma/README.md`, `ARCHITECTURE_DOCS.md §3.19` and a CLAUDE.md invariant entry, covering the spread-only cost model, the seen-events note and the 10-day blocks. Verify that every path the three name exists.

## 7. Real run and integration

- [x] 7.1 Run the real walk-forward. Verify it completes, every fold is recorded, and the out-of-sample file has no NaN or duplicate trade.
- [x] 7.2 Score Cell 1 and then Cell 2, once each. Verify the registry has exactly two rows and every other registry is byte-identical.
- [x] 7.3 Generate the report twice and verify the builds are identical apart from the timestamp. Run `python -m pytest -q tests/test_fx_bb_sma_*.py tests/test_fx_bb_*.py tests/test_fx_rates_protected.py tests/test_fx_session_protected.py tests/test_fx_fib_protected.py`, and verify all tests pass and `git status --short models/` is empty.

## Workflow follow-up

- Report both verdicts to the owner beside the power table, with the plain-language trade summary. State that the result is net of the recorded spread only and is not a trading result.
- Archive with `/opsx:archive` after the owner's review.
