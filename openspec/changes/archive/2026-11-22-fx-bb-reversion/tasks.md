# Tasks

## 1. Scaffolding and protection

- [x] 1.1 Create `src/fx_bb/` with `__init__.py` (`load_config`, `repo_path`, `config_digest`, `feature_list_digest`, `GROSS_HEADER`) and `fx_bb_config.json`. The config holds:
  - the six pairs, with their pip sizes and data sources (fx_session caches for EURUSD, GBPUSD and AUDUSD; `results/fx_bb/data` for USDJPY, USDCAD and USDCHF);
  - the per-pair starts (`2012-01-01`, USDCHF `2015-02-01`), `latest_allowed_start`, `known_gaps` `{USDCAD: ["2013-05"]}`, `min_h1_per_d1 = 12`, `max_spread_pips = 50`, the label time zone and the 17:00 New York day boundary;
  - band length 20, width 2, ATR 20, horizon 3, cooldown 3, and the reference-calendar quorum of 4;
  - the ordered 10 features;
  - folds from `2015-01-01`, 12 months each, validation 0.15, purge 5, embargo 5;
  - the LightGBM settings of design D7;
  - scoring: alpha 0.025, family size 2, block 5, 10 000 resamples, a fixed seed, power 0.8, and a parity tolerance of 0.02 ATR.

  Verify with `tests/test_fx_bb_config.py`: the config loads, the digest is stable, alpha is 0.05/2, there are no class weights, and the band, horizon and cooldown values are 20, 2, 3 and 3.
- [x] 1.2 Add `tests/test_fx_bb_protected.py`. It checks that `models/`, `config.json`, every other registry and the fx_session, fx_fib and fx_rates caches and results are unchanged by importing and configuring the family. It also checks that those three families' configs match the digests recorded with their verdicts, and that no family source names a protected write target. Verify it passes.

## 2. Market data

- [x] 2.1 Implement `data_loader.py`:
  - a refresh for USDJPY, USDCAD and USDCHF, reusing fx_session's `Mt5Source`, `coverage_holes` and `verify_clock_segments` read-only;
  - per-pair start dates, declared gaps, and an all-or-nothing write;
  - `symbols.json` and `refresh_log.csv`;
  - the per-year lag check against EURUSD (lags −1, 0 and +1) on refresh and on load for all six pairs.

  Verify with offline tests using a fake source:
  - one failing pair writes nothing for any pair;
  - an undeclared hole is refused and a declared one accepted;
  - a synthetic one-hour shift fails the lag check while passing the weekly-open check;
  - USDCHF bars before its start are never used;
  - the fx_session caches stay byte-identical.
- [x] 2.2 Implement `daily.py`: D1 bars closing at 17:00 New York, built from the Berlin labels. They hold the bid OHLC, the mid close, the bid close and `n_h1`, drop days with fewer than 12 bars, and apply the corrupt-spread rule. Verify with tests:
  - the Sunday reopen belongs to Monday;
  - a Wednesday closes on the 22:00 Berlin bar, and on the 21:00 bar in a mismatch week;
  - the 23:00 rollover bar opens the next day;
  - a short day is dropped;
  - a 300 000-point spread voids that day's mid close but not the bid prices.
- [x] 2.3 Run the real refresh of the three new pairs. Verify the first and last bars, the refresh log, a lag check passing every year from each pair's start, and that USDCHF 2014 would fail it. Record coverage, the lag table, the corrupt AUDUSD day and the declared gaps in `results/fx_bb/README.md`.

## 3. Events and outcomes

- [x] 3.1 Implement `events.py`: bands, ATR20, first breaches, the 3-day cooldown, the reference calendar and the pair-gap exclusion, outcomes `r` in ATR units, and binary targets with ties counted. Verify with tests:
  - an upper breach gives direction −1, and a second day outside is not an event;
  - a re-breach inside the cooldown is not an event;
  - the hand-computed +0.5 outcome;
  - a GBPUSD 2014-03 window is excluded as a pair gap, while a common holiday excludes nothing;
  - truncating the bars after `t` leaves the bands, ATR and event flag unchanged.
- [x] 3.2 Write `results/fx_bb/labels.md` with the gross header, and the following per pair and pooled, overall, from 2015 and per year:
  - events, cooldown removals, exclusions by reason, ties and distinct dates;
  - the minimum detectable mean outcome, from the all-days dispersion, and the minimum detectable accuracy, each on events and on distinct dates.

  It runs the config guard. Verify with a test that 1 054 events on 751 dates give about 54.3 % and 55.1 %, and that no outcome of an event is read. Generate it on the real data.

## 4. Features

- [x] 4.1 Implement `features.py` with the 10 oriented columns of design D5 and `assemble`, which raises on an extra or missing column. Verify with tests:
  - the hand-computed z-score;
  - the mirrored-path test;
  - `close_pos_in_bar` equals 1 at the extreme;
  - `simultaneous_frac` on a five-pair day;
  - the scale test (prices × 100);
  - warm-up exclusion from Cell 2 only.
- [x] 4.2 Add the truncation test: for 20 random events, cutting every pair's bars after `t` leaves the row unchanged. Run on the real data, and print the event counts with features, exclusions by reason, and a feature summary with no constant column.

## 5. Folds, model and bootstrap

- [x] 5.1 Implement `splits.py`: expanding 12-month folds from 2015-01-01 over the event dates, with fit, validation and test slices, and the 10-business-day gap after each slice's last `t+3` date. Verify with tests that no training `t+3` date is within 10 business days of the next slice, that every out-of-sample event appears once, and that the folds are contiguous.
- [x] 5.2 Wire the pooled LightGBM and the fit-slice base rate through fx_session's `model_lgbm` (read-only import) with this family's config and `research_models/fx_bb/`. Verify with tests that a same-seed run is identical, and that the base rate equals the fit slice's up-rate.
- [x] 5.3 Implement the day-block ratio bootstrap in `score.py`. Verify with a test that, with one event every day, it reproduces `fx_fib.score.circular_block_means` exactly for the same seed. Also verify that a planted positive mean is KEEP and a zero mean is DROP for Cell 1.

## 6. Walk-forward, scoring and registry

- [x] 6.1 Implement `walk_forward.py`. It writes `oos_events.csv.gz` (out-of-sample events with `p_prior` and `p_lightgbm`) and `run.json`, which holds the config and feature digests, the six cache digests and the folds. It runs the guard first. Verify with a synthetic full-stage test and a guard-refusal test.
- [x] 6.2 Implement `registry.py` and `score.py --cell rule|model`:
  - alpha 0.025, block 5, mode `'x'` per cell;
  - the note with the gross statement and the earlier fx families;
  - the Cell 1 window, the bid statistic and the parity flag.

  Verify with tests:
  - a second scoring is refused and the registry stays byte-identical;
  - a changed config is refused by the label, walk-forward and score stages;
  - other registries are unchanged.

## 7. Report and docs

- [x] 7.1 Implement `report.py`, which writes `results/fx_bb/walk_forward_report.md`. It contains:
  - the gross header, the verdicts or "Not scored yet", and the power table;
  - for every event, per pair, per year and per breach side: win rate with Wilson on distinct dates, mean `r`, and mean pips and profit factor, gross and after the mean spread;
  - the mid-versus-bid line;
  - for Cell 2: log-loss and Brier per fold, reliability, and the events above and below P(win) 0.5.

  Verify with tests that two builds differ only in the generation line, and that pips are computed correctly for USDJPY and EURUSD on a toy event set.
- [x] 7.2 Complete `results/fx_bb/README.md`, add `ARCHITECTURE_DOCS.md §3.17` and a CLAUDE.md invariant entry (including the USDCHF 2012–2014 clock shift and the corrupt AUDUSD spread). Verify that every path the three name exists.

## 8. Real run and integration

- [x] 8.1 Run the real walk-forward. Verify it completes, every fold is recorded, and the out-of-sample file has no NaN or duplicate event.
- [x] 8.2 Score Cell 1 and then Cell 2, once each. Verify the registry has exactly two rows and every other registry is byte-identical.
- [x] 8.3 Generate the report twice and verify the builds are identical apart from the timestamp. Run `python -m pytest -q tests/test_fx_bb_*.py tests/test_fx_rates_protected.py tests/test_fx_session_protected.py tests/test_fx_fib_protected.py`, and verify all tests pass and `git status --short models/` is empty.

## Workflow follow-up

- Report both verdicts to the owner beside the power table, gross of cost and not a trading result.
- Archive with `/opsx:archive` after the owner's review.
