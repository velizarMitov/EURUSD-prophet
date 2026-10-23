# Tasks

Before starting:
- Ask the owner to turn off VS Code Copilot auto-commit and auto-push (memory:
  project-copilot-auto-commits).
- Take a full-suite baseline (`python -m pytest -q`). The known baseline failures are
  the 7 protected-set and provenance tests from the undeclared 2026-10-14 retrain, plus
  the `DATA.md` row count.

Each group ends with no new failures against that baseline.

## 1. Logger reliability (runs first: the forward data needs it)

- [ ] 1.1 Update `scripts/gold/install_task.ps1` to set the following, and print the
  active power plan's "Allow wake timers" state from `powercfg /q` without changing it:
  - WakeToRun and StartWhenAvailable;
  - AllowStartIfOnBatteries and DontStopIfGoingOnBatteries;
  - MultipleInstances IgnoreNew;
  - ExecutionTimeLimit 20 minutes.

  Verify: reinstall, then `Get-ScheduledTask EURUSDProphet-GoldLogger` settings show
  each value, and the installer output names the wake-timer state.
- [ ] 1.2 Add `src/gold/health.py`:
  - `record(pass_result)` appends to `results/gold/forward/health.csv`;
  - `status(now)` gives the last pass, the last success, missed hours in the last 24 h,
    and whether the market is open, using `clock.py`'s weekly-open logic;
  - `stalled(status)` applies the 2-hour and 3-missed rule outside weekends.

  Wrap `forward_logger.run_once` so that every pass, including a failure or an
  unreachable MT5, writes one health row, leaving round-one files untouched. Verify with
  `tests/test_gold_health.py`:
  - fixed-clock cases: weekday stall, weekend quiet, MT5 unreachable;
  - one row per pass;
  - `predictions.csv` and `settlements.csv` headers unchanged.
- [ ] 1.3 Add `GET /api/gold/health` (additive in `api.py`) and a stall banner on
  `/gold` (BG/EN). Widen the `tests/test_gold_isolation.py` allowlist to the read-only
  views `health` and `leaderboard`. Verify:
  - the test client returns the status;
  - the page source has the banner keys in both languages;
  - the `api.py` numstat shows 0 removed lines.
- [ ] 1.4 Document the task settings, the health file and the stall rule in
  `results/gold/forward/RUNBOOK.md` and `ARCHITECTURE_DOCS.md §3.12`. Verify that both
  name `health.csv` and WakeToRun.

## 2. Round-two registration

- [ ] 2.1 Add `src/gold/prereg2.py`. It builds `results/gold/PRE_REGISTRATION_V2.{md,json}`
  with:
  - the range, macro and direction2 families, cells, alphas, metrics, baselines and
    expected outcomes (design D1);
  - the power table from `prereg.direction_family`, with the edge, power and family α
    unchanged;
  - the selective coverage floor, `max(0.45, n_required / n_validation)`;
  - the barrier k and m and the range W, γ and tolerance;
  - the statement that validation was seen in round one and the test block is spent;
  - pinned data digests.

  Add `init_registries` for `gold_{range,macro,direction2}_hypothesis_log.csv`. Verify
  in `tests/test_gold_prereg2.py`:
  - the round-one registries and `PRE_REGISTRATION.json` are byte-identical before and
    after;
  - an underpowered cell is recorded UNDERPOWERED and the family α comes from the
    registered size;
  - a test-block request is refused.
- [ ] 2.2 Add a `scoring_allowed_v2` gate, using `committed_and_clean` on
  `PRE_REGISTRATION_V2.json`. Verify that tests refuse scoring when the file is
  uncommitted or modified.

## 3. Price ranges

- [ ] 3.1 Add `src/gold/ranges.py`:
  - `sigma_from_abs(e_abs)` (√(π/2));
  - `ACI(level, gamma, window)` with `step(z_abs, covered)` and `bounds(close, sigma)`,
    using past-only updates;
  - `winkler(lo, hi, y, alpha)`.

  Verify with unit tests: no settlement at or after t enters bar t's quantile; on an
  i.i.d. normal synthetic series, coverage converges to within ±2 points of the target;
  the Winkler score matches a hand-computed case.
- [ ] 3.2 Add the study path `range` to the round-two stage. It reads the saved
  validation predictions, uses the first W rows as warm-up, scores ACI-LSTM against
  static-normal GARCH on the same rows, and appends the verdict to the range registry.
  The static-Gaussian LSTM range is reported as a comparison row. Verify that a test
  reproduces the score from the saved CSVs without refitting, and that coverage and
  tolerance appear in the report.
- [ ] 3.3 Live ranges: the logger writes `ranges.csv` rows for h1_lstm and d1_lstm at 80
  and 95 %, seeded per design D2, and settles `inside` from the round-one settlement of
  the same key. Verify:
  - a fixture pass writes 4 range rows per bar;
  - a later pass settles them;
  - the round-one files are unchanged.
- [ ] 3.4 The range card on `/gold` (*Now* tab) shows the 80 and 95 % ranges, and
  *History* gains a range-coverage column, with no target, stop or entry words. Verify
  that the page-source tests (both languages, banned words) pass, then check in the
  browser at 360 and 1280 px.

## 4. Macro drivers

- [ ] 4.1 Add `src/gold/macro.py`:
  - the FRED series through `src.macro_data.fetch_fred_feature`, with caches under
    `results/gold/macro/`;
  - MT5 `SILVER` and `Brent` D1 caches through a helper factored out of
    `src/gold/data.py`;
  - alignment per design D4, plus `macro_features(d1_index)`.

  Verify:
  - the Monday real-yield case takes Friday or earlier;
  - a missing silver bar carries the past value forward;
  - `assert_scale_free` rejects a raw level;
  - an unreachable source names the series;
  - gold isolation passes.
- [ ] 4.2 Add the variants `d1_lstm_macro` and `d1_gbm_macro`, with round-one
  hyperparameters, to the round-two validation stage. Score vol, return and range
  against the same model without macro on identical rows, and append them to the macro
  registry. Verify: both variants are scored on the same row set, and the registry gains
  exactly the registered cells.

## 5. Direction, new approach

- [ ] 5.1 Add `src/gold/direction2.py` with:
  - `h1_selective`: τ chosen on inner training folds to meet the floor, with
    abstentions logged;
  - `h1_barrier`: EWMA(20) barriers, k = 6, the purge;
  - `h4_dir`: non-overlapping windows, any missing hour makes the window ineligible.

  Verify:
  - τ never sees validation rows;
  - abstentions are excluded from hit counts;
  - a label crossing the training end is purged;
  - a 4-hour window with a missing hour is ineligible.
- [ ] 5.2 Score the admitted direction2 cells on validation against the training
  majority and a coin. Record the UNDERPOWERED ones without scoring. Verify that the
  registry rows match the registration's cell list exactly.

## 6. The round-two validation run

- [ ] 6.1 Commit `PRE_REGISTRATION_V2.{md,json}` on its own, and confirm no score
  exists yet. Then run the round-two validation stage once, and write
  `results/gold/validation_report_v2.md` and a dated section in `RESULTS.md`, with
  every verdict marked "needs forward confirmation". Verify:
  - the stage refuses a second run;
  - the round-one registries are unchanged;
  - the report lists every registered cell.
- [ ] 6.2 Fit the admitted round-two models for live use with the `final` path into
  `research_models/gold/v2/`, with manifest digests. The logger forecasts them into
  `predictions_v2.csv` and `settlements_v2.csv` in the dry-run phase. Verify that a
  fixture pass writes v2 rows with digests and settles them, and that round-one files
  are unchanged.

## 7. Large-move alerts

- [ ] 7.1 Add `src/gold/alerts.py`:
  - `--calibrate` writes the 90th-percentile threshold and value to `gold_config.json`;
  - `check(forecast)` decides whether to alert, deduplicated by bar;
  - `notify(text)` raises a PowerShell toast with a 10 s timeout, and a failure goes to
    the health record;
  - `alerts.csv` rows are settled with the realised move.

  Verify:
  - the threshold equals the percentile of the fixture forecasts;
  - a second pass on the same bar makes no new alert;
  - a failed notification still writes the row;
  - the text has no buy, sell, entry, stop or target words in either language.
- [ ] 7.2 Add an `alert` block to `/api/gold` (additive field) and a banner on `/gold`
  while the target hour is open. Verify with the test client and in the browser.

## 8. Leaderboard

- [ ] 8.1 Add `src/gold/leaderboard.py` and `GET /api/gold/leaderboard`. Each row is
  (cell, digest), with the count, the Wilson hit rate, move error against GARCH on
  shared bars, range coverage, phase and the shadow flag. Rows under 30 say "too few".
  Each row is labelled "descriptive — not a verdict". Alerted hours are compared with
  non-alerted ones. Verify in `tests/test_gold_leaderboard.py`:
  - two digests make two rows;
  - a shadow version is flagged;
  - sorting puts rows under 30 last;
  - building it writes no file.
- [ ] 8.2 Add a *Leaderboard* tab on `/gold` (BG/EN, the shell's components, sortable,
  phone-friendly). Verify the page-source tests, then check in the browser at 360 and
  1280 px, light and dark.

## 9. Model versions

- [ ] 9.1 Add `src/gold/versions.py`:
  - `refit()` runs the `final` path into `versions/<YYYYMMDD>/` with a manifest, under
    its own lock;
  - `active.json` holds the newest version only;
  - `promote(id, reason)` refuses while the forward window is open, and otherwise swaps
    the manifest, records the change in `RESULTS.md` and supersedes the forward
    registration.

  Verify:
  - a refit leaves `artifact_manifest.csv` and `research_models/gold/<cell>/`
    byte-identical;
  - promote refuses in an open window;
  - an overlapping refit exits and records the skip.
- [ ] 9.2 The logger forecasts with the active shadow version into `predictions_v2.csv`
  with `shadow=1`, and `forward_report` ignores shadow rows. Verify that a fixture pass
  writes shadow rows and that the forward-report test sees only registered digests.
- [ ] 9.3 Add `scripts/gold/install_retrain_task.ps1` (monthly on the 1st at 02:30,
  through `run_hidden.vbs`, with the same wake and battery settings) and an uninstall
  script, documented in RUNBOOK. Verify that the task is registered and that a manual
  run writes a version folder.

## 10. Integration checks

- [ ] 10.1 Run browser QA on `/gold` (Now with ranges and alert banner, History,
  Leaderboard, stall banner) at 360 and 1280 px, light and dark, BG and EN, with no
  console errors.
- [ ] 10.2 Run the full suite and confirm no new failures against the baseline. Check
  that `api.py` has 0 removed lines, the round-one gold files are byte-identical, and
  nothing changed under `models/`.
- [ ] 10.3 Restart the server through the launcher and reinstall both tasks. Confirm
  that the next hourly pass writes health, range and v2 rows, then report to the owner
  with links and the round-two validation verdicts.
