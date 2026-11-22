# Tasks

## 1. Scaffolding and protection

- [ ] 1.1 Create `src/fx_nyvol/` with `__init__.py` (`load_config`, `repo_path`, `config_digest`, `feature_list_digest`, `GROSS_HEADER`) and `fx_nyvol_config.json`. The config holds:
  - pairs `["EURUSD", "GBPUSD"]`, factor pairs adding AUDUSD, and the fx_session data path (read-only);
  - the session window 15–21 Berlin, the close-of-decision hour 14, Asia 0–8, London 9–13, the data hour 14, the range window 20 and the HAR windows 1/5/20;
  - `max_spread_pips = 50` and the pip sizes;
  - the ordered 14 features;
  - folds from `2013-01-01`, 6 months each, validation 0.15, embargo 1;
  - the LightGBM settings of design D6 with the `l2` objective, and the MLP settings with `allow_cpu = false`;
  - scoring: alpha 0.0125, family size 4, block 5, 10 000 resamples, a fixed seed, power 0.8, and the hot-day quantile 0.75 over 60 days.

  Verify with `tests/test_fx_nyvol_config.py`: the config loads, the digest is stable, alpha is 0.05/4, and the window is 15–21.
- [ ] 1.2 Add `tests/test_fx_nyvol_protected.py`. It checks that `models/`, `config.json`, the production volatility results, every other registry and the fx_session, fx_fib and fx_rates caches and results are unchanged by importing and configuring the family. It also checks that those three families' configs match the digests recorded with their verdicts, and that no family source names a protected write target. Verify it passes.

## 2. Eligibility and target

- [ ] 2.1 Implement `target.py`:
  - eligibility from fx_session's `session_calendar` (read-only), complete pre-session windows, and the corrupt-spread rule;
  - the mid-range target `y = log((mid_high − mid_low) / close14)` over bars 15–21;
  - the bid twin `y_bid`;
  - an exclusion reason for each day that is not eligible.

  Verify with tests:
  - the hand-computed target;
  - the 22:00 rollover bar does not change it;
  - a missing 18:00 GBPUSD bar excludes the day for both pairs;
  - a 300 000-point AUDUSD spread excludes the day as `corrupt spread`;
  - an unverified clock week is excluded.
- [ ] 2.2 Write `results/fx_nyvol/labels.md` with the gross header; per pair the weekdays, eligible days, out-of-sample days from 2013, exclusions by reason and eligible days per year; the mean and standard deviation of `y`; and the detectable difference in units of its standard deviation. It runs the config guard. Verify with a test that 3 470 days give about 0.05 standard deviations, and that no model output is used. Generate it on the real data.

## 3. Features

- [ ] 3.1 Implement `features.py` with the 14 columns of design D4, reusing fx_session's window and calendar helpers read-only, and `assemble`, which raises on an extra or missing column. Verify with tests:
  - the hand-computed London range;
  - `cross_pre_range_ratio` = 0.5 on a quiet morning;
  - the HAR lags are unchanged when today's range is doubled;
  - the scale test (prices × 100);
  - the missing-column error;
  - warm-up exclusion.
- [ ] 3.2 Add the truncation test: for 20 random days, removing every bar after the one labelled 14:00 leaves the row unchanged. Run on the real data, and print row counts, exclusions by reason and a feature summary with no constant column.

## 4. Folds and models

- [ ] 4.1 Implement `splits.py` with shared 6-month folds from 2013-01-01, the last-15 % validation slice and the 1-day embargo. Verify with tests for shared boundaries, the embargoed day, every out-of-sample day predicted once, and a scaler fit on the fit slice only.
- [ ] 4.2 Implement `models.py`:
  - the HAR OLS on the training days;
  - the LightGBM `l2` regressor with early stopping;
  - the MSE MLP: `32-16`, LayerNorm, GELU, a linear head, a standardised target, the mean of 5 seeds, CUDA only.

  Verify with tests:
  - HAR recovers known coefficients on a synthetic series and ignores test days;
  - a same-seed LightGBM run is identical;
  - the MLP refuses to run without CUDA, and on CUDA beats a constant on a learnable toy series.

## 5. Walk-forward, scoring and registry

- [ ] 5.1 Implement `walk_forward.py --stage lgbm`. It writes `oos_lgbm.csv.gz` (`y`, `y_bid`, `p_har`, `p_lightgbm`, the fold, the calendar flags) and `run_lgbm.json` (config and feature digests, the fx_session cache digests, the folds). It runs the guard first. Verify with a synthetic full-stage test and a guard-refusal test.
- [ ] 5.2 Implement `registry.py` and `score.py --stage lgbm|mlp`:
  - `fx_fib.score.test_cell` on the per-day squared-error differences, with block 5 and alpha 0.0125;
  - mode `'x'` per cell;
  - the relative MSE reduction;
  - the gross note and the earlier fx families;
  - NOT TRIGGERED rows with no alpha reassignment.

  Verify with tests:
  - KEEP on a planted improvement and DROP on none;
  - a second scoring is refused and the registry stays byte-identical;
  - a changed config is refused by the label, walk-forward and score stages;
  - the both-DROP path;
  - other registries are unchanged.
- [ ] 5.3 Implement `walk_forward.py --stage mlp`. It refuses without both LightGBM rows and trains only the KEEP pairs. Verify with tests for the refusal, KEEP-only training on the GPU, and the both-DROP path that trains nothing.

## 6. Report and docs

- [ ] 6.1 Implement `report.py`, which writes `results/fx_nyvol/walk_forward_report.md`. It contains:
  - the gross header, the verdicts or "Not scored yet", and the power line;
  - MSE, MAE, R², the MSE ratio to HAR and Spearman's ρ;
  - the causal-Q75 hot-day precision, recall and accuracy;
  - MSE per year, per fold and on NFP, FOMC and DST-mismatch days;
  - the HAR-X line;
  - the bid parity line.

  Verify with tests that two builds differ only in the generation line, and that the Q75 threshold on a toy series uses only the previous 60 days.
- [ ] 6.2 Complete `results/fx_nyvol/README.md`, add `ARCHITECTURE_DOCS.md §3.18` and a CLAUDE.md invariant entry. Verify that every path the three name exists.

## 7. Real run and integration

- [ ] 7.1 Run the real stage-1 walk-forward. Verify it completes, both pairs' folds are recorded, and the out-of-sample file has no NaN or duplicate day.
- [ ] 7.2 Score the LightGBM cells once, then run stage 2 and score or record the MLP cells. Verify the registry has exactly four rows and every other registry is byte-identical.
- [ ] 7.3 Generate the report twice and verify the builds are identical apart from the timestamp. Run `python -m pytest -q tests/test_fx_nyvol_*.py tests/test_fx_rates_protected.py tests/test_fx_session_protected.py tests/test_fx_fib_protected.py`, and verify all tests pass and `git status --short models/` is empty.

## Workflow follow-up

- Report the four verdicts to the owner with the relative MSE reductions and the HAR-X line, gross of cost and not a trading result.
- Archive with `/opsx:archive` after the owner's review.
