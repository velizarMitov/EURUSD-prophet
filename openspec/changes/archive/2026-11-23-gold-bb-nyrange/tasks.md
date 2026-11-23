# Tasks

## 1. Package, config and protection

- [x] 1.1 Create `src/gold_bbnr/__init__.py` with `load_config`, `repo_path`, `config_digest` and `GROSS_HEADER`, and create `gold_bbnr_config.json`. The config holds the window 2015-01-02 .. 2026-11-20, bands, ATR, cooldown, trade, the eleven (Part 1) and fourteen (Part 2) feature lists, the DXY symbol, sources and windows, the folds, the thin-fold minimum of 60, LightGBM and MLP hyperparameters, and scoring (the alphas, blocks 5/10/5, 10 000 resamples, seeds, declared SDs). Verify with `tests/test_gold_bbnr_config.py`: the digest is stable, and the LightGBM, MLP and band blocks equal fx_bb's, fx_bb_sma's and fx_nyvol's.
- [x] 1.2 Add `tests/test_gold_bbnr_protected.py`. Verify that running the label stage leaves `results/gold/gold_h1.csv`, its meta file, every gold and fx registry, `gold_config*.json` and `models/` byte-identical, and that no module in the package writes outside `results/gold_bbnr/` and `research_models/gold_bbnr/`.

## 2. Gold bars

- [x] 2.1 Write `bars.py`: read the H1 cache (OHLC only, window-clipped), run `src.gold.clock.verify` and refuse on failure, and build New York 17:00 D1 bars with validity (at least 12 H1 bars, weekday). Verify with `tests/test_gold_bbnr_bars.py`: a synthetic DST-mismatch 23:00 bar goes to the next day, a short day is invalid, bars after 2026-11-20 are ignored, and a shifted clock is refused.

- [x] 2.2 Write `dxy.py`:
  - `fetch --mt5-path`: market-data calls only, OHLC only, 2017-01-03 .. 2026-11-20, into `data/dxy_h1.csv`.
  - `fetch --yahoo`: `DX-Y.NYB` daily, 2014-06-01 .. 2026-11-20, generic client name, into `data/dxy_d1.csv`.
  - `load_h1`: runs the yearly lag-0 check against fx_session's EURUSD.
  - `load_d1`.

  Verify with `tests/test_gold_bbnr_dxy.py`:
  - a year shifted by one hour is refused;
  - the fetch module references no MT5 order or position function;
  - the Yahoo request carries only the generic client name (mocked).
- [x] 2.3 Run both real fetches once. Verify the files and their row counts, the first H1 bar on 2017-01-03, the June 2024 hole, and that the lag check passes. Record the digests.

## 3. Part 1a: Bollinger reversion

- [x] 3.1 Write `bb_events.py`: bands, ATR20, first breaches, the 3-day cooldown, and the outcome `r` in a separate function that labels never calls. Verify with `tests/test_gold_bbnr_bb.py`: the hand-built upper and lower breaches, the cooldown removal, `r = +1.0` on the spec example, and identical detection on truncated data.
- [x] 3.2 Write `bb_features.py`: the nine gold features (fx_bb's definitions, `simultaneous_frac` absent) plus `dxy_zscore_20d` and `dxy_ret_5d`, from DXY closes dated before `t` and signed by the event direction. Verify that:
  - the columns equal the eleven-name config list;
  - features on data cut at `t` are identical;
  - planting a huge DXY close on day `t` leaves day `t`'s features unchanged;
  - `warm-up` exclusions are counted.

## 4. Part 1b: the SMA20 trade

- [x] 4.1 Write `bb_trades.py`: entries without cooldown, the take profit at SMA20, the symmetric stop, the H1 first-touch scan (stop wins a tie, a gap through the stop fills at the open), the time stop at the close of `t+10`, one open trade, the `no exit data` exclusion, and R. Verify with `tests/test_gold_bbnr_trades.py`: the spec examples for levels, a tie, a gap, R = −1 at the stop, blocking during an open trade, and an exit at the same close as a new entry.

## 5. Part 2: the NY range

- [x] 5.1 Write `ny_target.py` and `ny_features.py`: target eligibility (bars 14–21 present), the thirteen gold and calendar features, and `dollar_pre_range_ratio` from hourly DXY, which is missing when fewer than 10 bars fall between 00:00 and 14:00. Verify with `tests/test_gold_bbnr_ny.py` that:
  - a missing 21:00 bar makes the day ineligible;
  - doubling prices leaves everything unchanged;
  - features on data cut after 14:00 are identical;
  - a DXY day with 5 bars gives `warm-up`.

## 6. Labels and pre-registration

- [x] 6.1 Write `labels.py`: counts per year and overall, exclusions by reason (including `thin fold` from event dates), distinct dates, and the MDE per cell from the declared dispersions. Write `labels.md` and `labels.json`. Verify with a test that mocks the outcome, exit-scan and target functions to raise if called.
- [x] 6.2 Write `prereg.py` with `build`, `write`, `gate` and a `--write` / `--check` CLI, using fx_ny_orb's git-gate pattern. The pre-registration includes the two DXY cache digests, and the gate re-hashes both caches. Verify with `tests/test_gold_bbnr_prereg.py` in a temporary git repo: the gate refuses an untracked file, a modified file, a config-digest mismatch and an edited DXY cache, and passes a clean commit.
- [x] 6.3 Run the real label stage and `prereg --write` on the gold cache. Check that the counts match the measured 179 breaches before cooldown, and record the power table in `results/gold_bbnr/README.md`. Verify that `labels.md` and both pre-registration files exist and no outcome file exists.
- [x] 6.4 **Owner step:** review and commit `results/gold_bbnr/PRE_REGISTRATION.md` and `.json`. Verify that `python -m src.gold_bbnr.prereg --check` reports them as committed and unchanged.

## 7. Walk-forward

- [x] 7.1 Write `walk_forward.py --part bb|bb_sma`: 12-month folds from 2018-01-01, the 5-day purge (by `t+3` for 1a and by exit for 1b), 15 % early stopping and thin folds skipped. Gate first, then write `bb_predictions.csv.gz` and `bb_sma_predictions.csv.gz`. Verify on synthetic data in a test with a monkeypatched gate: the purge boundary, a thin fold predicting nothing, and determinism (two runs give identical files).
- [x] 7.2 Write `walk_forward.py --part nyvol`: 6-month folds from 2018-01-01, all four models on the same rows with a 1-day embargo; HAR and HAR-X by OLS on all training days; LightGBM and the 5-seed MLP with 15 % early stopping; the MLP refuses without CUDA. Write `nyvol_predictions.csv.gz`. Verify on synthetic data: the fold boundary, the CUDA refusal (monkeypatched), and that HAR-X equals OLS on the fourteen columns.

## 8. Scoring and report

- [x] 8.1 Write `registry.py` and `score.py` for the three families and seven cells, reusing fx_bb's ratio day-block bootstrap, with KEEP and DROP as in the spec. Each cell is scored once, and `AlreadyScored` leaves the file unchanged. Verify with `tests/test_gold_bbnr_score.py`: a planted positive effect gives KEEP, zero gives DROP, re-scoring is refused, and other registries are byte-identical.
- [x] 8.2 Write `report.py` (pre-scoring: power and "not scored yet"; post-scoring: the spec's tables), and `frozen.py` for the Part 2 refit with a digest manifest and a refusal on a digest mismatch. Verify with tests that the report is identical apart from its timestamp, and that an altered frozen file raises.
- [x] 8.3 After 6.4, run the real walk-forwards, score all three families once, run `frozen`, and build the report twice. Verify that the three registries hold 2 + 2 + 3 rows, that every other registry is byte-identical, and that `models/` is unchanged.

## 9. Docs

- [x] 9.1 Add an `ARCHITECTURE_DOCS.md` section and a `CLAUDE.md` invariant for `gold_bbnr`, covering the package, the read-only inputs, gross on bid prices, the three registries and the results once scored. Verify that the full `tests/test_gold_bbnr_*` suite and the other families' protection tests pass.
