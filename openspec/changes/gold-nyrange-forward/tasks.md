# Tasks

## 1. Package, config and protection

- [x] 1.1 Create `src/gold_nyfwd/` with `__init__.py` (`REPO`, `load_config`, `repo_path`, `config_digest`, `sha256_file`, `GROSS_HEADER`) and `gold_nyfwd_config.json`. The config covers:
  - the paths `results/gold_nyfwd` and `research_models/gold_nyfwd`;
  - the terminal path, gold symbol `GOLD`, the contract prefix `USDInd` and a roll margin of 2 trading days;
  - 4 000 bars per top-up and the lag window ±3;
  - forecast after 15:00 and settlement after 22:00 (label clock);
  - the forward horizon of 60 days or 2027-04-30;
  - scoring: alpha 0.05, family size 1, block 5, 10 000 resamples, seed 20261201;
  - the export subdirectory and file, and 60 display days.

  Verify with `tests/test_gold_nyfwd_config.py`: the config loads and the digest is stable.
- [x] 1.2 Add `tests/test_gold_nyfwd_protected.py`. Verify that:
  - no module names a write target outside `results/gold_nyfwd/` and `research_models/gold_nyfwd/`, apart from the export;
  - no source contains `order_send`, `positions_get` or `TRADE_ACTION`;
  - a fake-feed forecast and settlement leave `models/`, `config.json`, `results/gold/`, `results/gold_bbnr/`, `results/fx_ny_orb/`, `research_models/gold_bbnr/` and every other registry byte-identical.

## 2. Live data (spec live-data)

- [x] 2.1 Write `feed.py`, a read-only MT5 wrapper that can be injected for tests. It provides `symbols` with expiries, `closed_bars(symbol, n)` from position 1, and `label_now()` from `forecast_eval.m15_data`. Write `cache.py` with append-only per-symbol H1 caches. Verify with `tests/test_gold_nyfwd_data.py`:
  - a repeated top-up stores each closed bar once;
  - a changed earlier bar is never rewritten;
  - a Saturday `label_now` makes every pass report `weekend`.
- [x] 2.2 Write `contracts.py`. It covers:
  - the roll rule: nearest expiry at least 2 trading days after the day;
  - the per-day contract map used for the dollar frame;
  - the EURUSD lag check (negative return correlation peaking at lag 0 within ±3 h), recorded in `contracts.json`.

  Verify in the same test file:
  - Dec26 is chosen on 2026-12-09 and Mar27 on 2026-12-10;
  - with no successor, the day is `no dollar data`;
  - a contract shifted by one hour is refused;
  - a 20-day mean spanning a roll mixes both contracts' ranges.

## 3. Pinned coefficients and pre-registration (spec forward-evaluation)

- [x] 3.1 Write `prereg.py` with `build`, `write`, `gate` and a `--write` / `--check` CLI, following fx_ny_orb's git-gate pattern.
  - `build` copies the HAR-X (15) and HAR (4) coefficients, their column order and the `ols.json` SHA-256 from gold_bbnr, read-only.
  - It also states the cell, alpha, the horizon, the 34 % power with its calculation, the contract-roll rule and that a DROP is not evidence of failure.
  - `gate` refuses an untracked or modified file and a config-digest mismatch, and returns the commit date.

  Verify with `tests/test_gold_nyfwd_prereg.py` in a temporary git repo:
  - each refusal case is refused;
  - a clean commit passes;
  - an altered `ols.json` makes the coefficient loader refuse;
  - the copied coefficients equal gold_bbnr's.
- [x] 3.2 Run `prereg --write` on the real gold_bbnr artifacts. Verify that both files exist, that the coefficients match `ols.json`, and that no forward log exists.
- [ ] 3.3 **Owner step:** review and commit `results/gold_nyfwd/PRE_REGISTRATION.md` and `.json`. Verify that `python -m src.gold_nyfwd.prereg --check` reports them as committed and unchanged.

## 4. Daily passes (spec daily-passes)

- [x] 4.1 Write `live.py` with the forecast pass. It works as follows:
  - it runs gate first, then the clock window, then once-per-day;
  - it tops up the caches and truncates both frames to bars labelled ≤ 14:00;
  - it runs gold_bbnr's `d1_ny` and `ny_features.dataset` and applies the pinned coefficients;
  - it logs `p_harx`, `p_har`, `F = exp(p_harx) × close14`, close14 and the contract, or the reason: `missing pre-session bar`, `warm-up`, `no dollar data` or `unverified clock week`.

  Verify with `tests/test_gold_nyfwd_live.py` on fake feeds:
  - a start at 14:50 logs nothing;
  - a second run changes nothing;
  - a pass at 18:00 gives the same row as one at 16:08;
  - the features equal those gold_bbnr computes from the same bars;
  - a missing 13:00 bar logs its reason.
- [x] 4.2 Add the settlement pass. It settles the due unsettled forecasts with gold_bbnr's eligibility and target on bars through 21:00. It records `y`, `se_harx`, `se_har`, `realised_high` and `realised_low`, or the ineligibility reason. A day with no forecast row is never settled. Verify in the same file:
  - one settlement per day;
  - a second run changes nothing;
  - an early close is settled `incomplete session` with no squared errors;
  - a weekday with no forecast stays absent.
- [x] 4.3 Add `scripts/gold_nyfwd/` with `run_hidden.vbs`, `run_module.cmd`, `install_tasks.ps1` (16:08 and 23:08 Mon–Fri, with `-DryRun`) and `uninstall_tasks.ps1`. Verify that the `-DryRun` output names exactly the two tasks at those times, and that the command length is at most 261 characters.

## 5. Scoring and report (spec forward-evaluation)

- [x] 5.1 Write `registry.py` and `score.py`. They cover:
  - the cell `gold_harx_vs_har_forward` scored with `fx_fib.score.test_cell` (alpha 0.05, block 5);
  - forward days strictly after the commit date;
  - a refusal before 60 days and before 2027-04-30;
  - `UNDERPOWERED — NO DECISION` at the deadline;
  - `AlreadyScored` on a second run.

  Verify with `tests/test_gold_nyfwd_score.py`:
  - a planted improvement is KEEP;
  - zero is DROP;
  - 40 days before the deadline is refused;
  - 40 days at the deadline is UNDERPOWERED;
  - re-scoring leaves the registry byte-identical.
- [x] 5.2 Write `report.py`. It shows the forward count, the days without a forecast by reason, missed weekdays and the contracts used. Before scoring it says `not scored yet` and shows no error figure. After scoring it shows the verdict, monthly MSE and HAR-X's MSE relative to HAR's. Verify in the same file that the report is deterministic apart from the generated-at line, and that the pre-score report has no `se_` or MSE figure.

## 6. Chart (spec chart-display)

- [x] 6.1 Write `export.py`. It rebuilds `Common\Files\EURUSDProphet\gold_ny_range.tsv` after each pass: a header with the forward count out of 60 and the verdict text, then 60 days of rows with date, status, close14, F, and the realised high and low. A failed write reports `export failed` and leaves the logs unchanged. Verify with `tests/test_gold_nyfwd_export.py`:
  - the rows are correct after settlement;
  - the export failure case behaves as stated;
  - no trade word appears in the file.
- [x] 6.2 Write `mql5/Indicators/ProphetGoldNYRange.mq5`, read-only. It draws only on `GOLD*` at H1 and below: the ±F/2 band around close14 for 15:00–22:00 Berlin, and the dotted realised box. Its corner shows the count out of 60 and the no-verdict text. Add `scripts/gold_nyfwd/install_mt5_indicator.ps1` and `uninstall_mt5_indicator.ps1`. Verify that:
  - it compiles with MetaEditor, with 0 errors;
  - a test scans its string literals for trade words and its source for order functions;
  - the uninstall leaves ProphetNYOrb and ProphetGoldSignals untouched.

## 7. Documentation and integration

- [x] 7.1 Write `results/gold_nyfwd/README.md` and `RUNBOOK.md`. They cover the once-steps, the daily timetable, the December contract check, what the chart shows and what to do on each refusal. Add an `ARCHITECTURE_DOCS.md` §3.22 and a `CLAUDE.md` invariant 17 for `gold_nyfwd`. Verify that every command in the RUNBOOK runs as written (`--help` or `-DryRun`).
- [ ] 7.2 Integration check:
  - run the full `tests/test_gold_nyfwd_*` suite and the protection tests of gold_bbnr, fx_ny_orb, fx_nyvol, fx_bb and fx_bb_sma;
  - after 3.3, run one real forecast pass and one real settlement pass;
  - confirm that the logs hold one row each, that the TSV exists, and that `models/` and every other registry are unchanged.

## Workflow follow-up

- After 3.3 and 7.2, the owner installs the tasks and the indicator per the RUNBOOK.
- Score once at 60 forward days or on 2027-04-30 with `python -m src.gold_nyfwd.score`, then record the result in the docs.
- Archive the change after implementation; the scoring may come later.
