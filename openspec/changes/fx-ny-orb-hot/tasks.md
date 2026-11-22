# Tasks

## 1. Scaffolding and protection

- [x] 1.1 Create `src/fx_ny_orb/` with `__init__.py` (`load_config`, `repo_path`, `config_digest`, `COST_HEADER`) and `fx_ny_orb_config.json`. The config holds:
  - the pairs, the M1 parquet paths and the fx_nyvol results path, all read-only;
  - the range labels 14:00–14:45, the first breakout bar 15:00, the entry cut-off 19:00 and the last bar 20:45;
  - the multiples 0.50, 0.25, 0.25 and 0.60, and the 0.4 bps slippage;
  - the window 2018-08-08 → 2026-08-07 and the frozen fit end 2026-11-20;
  - the forward rule: 270 days or 2028-01-31;
  - scoring: alpha 0.025 per family, 5-day blocks, 10 000 resamples, fixed seeds, a declared SD of 1.5 R;
  - the export path and the 60-day display window.

  Verify with `tests/test_fx_ny_orb_config.py`: the values, a stable digest, and no search settings.
- [x] 1.2 Add `tests/test_fx_ny_orb_protected.py`. Verify that:
  - importing and configuring the family changes no protected path: `models/`, `config.json`, the results and sources of fx_nyvol, fx_bb_sma, fx_bb, fx_rates, fx_session and fx_fib, `research_models/horizon_study/`, `results/curl/raw/`, and every other registry;
  - those families' configs match their verdict digests;
  - no family source names a protected write target.

## 2. M15 data

- [x] 2.1 Implement `m15.py`:
  - historical bars from `aggregate_m15(load_m1(path))`, with no cache write, the Berlin clock check, mid OHLC and the 50-pip void;
  - live bars from MT5, starting from position 1, into `results/fx_ny_orb/live/`, append-only, with overlap revisions counted.

  Verify with tests:
  - mid high = bid high + spread / 2;
  - a corrupt spread voids the day;
  - a shifted clock is refused;
  - a fake MT5 source never returns the forming bar;
  - `research_models/horizon_study/` is untouched.

## 3. The rule

- [x] 3.1 Implement `rule.py` with `entries` (stops at the entry decision) and `simulate_day` (the full trade), per the orb-rule spec and design D3. Verify with hand-built M15 days:
  - a missing 14:30 bar means no trade;
  - a range too wide, a breakout too far and an 18:45 breakout each mean no trade;
  - a long reaches its target;
  - a short reaches its stop;
  - a same-bar conflict counts as a stop;
  - the stop fills at the gap open;
  - a forced exit at the 20:45 close;
  - a stopped trade never re-enters;
  - the hand-computed +2.306 R net example;
  - the non-hot control is labelled;
  - long and short mirror each other on a reflected day.
- [x] 3.2 Add the historical day assembly: join fx_nyvol's `p_harx`, `hot_threshold` and eligibility to the M15 days in the window. Verify with tests that:
  - `close14` from the 14:45 M15 bar matches fx_nyvol's H1 close to within 1 pip on 99 % of the real days;
  - the hot counts reproduce 230 and 185;
  - no exit is computed by the assembly.

## 4. Power and pre-registration

- [x] 4.1 Implement `labels.py`. It writes `results/fx_ny_orb/labels.md` and `labels.json`, per pair and pooled, hot and non-hot: eligible days, trades, non-trade reasons, distinct dates, and the minimum detectable mean with SD 1.5 R. Verify with tests that:
  - the report succeeds while the exit scan raises;
  - a changed config is refused.

  Generate it on the real data.
- [x] 4.2 Implement `prereg.py`. It writes `PRE_REGISTRATION.md` and `.json` (the rule, costs, cells, window, config digest, the power table from labels.json, and the HAR-X selection fact), and adds the git gate. Verify with tests that the gate refuses an untracked, a modified and a missing file and passes a clean committed one, using a temporary git repo. Generate both files.
- [ ] 4.3 **Owner step:** review and commit `results/fx_ny_orb/PRE_REGISTRATION.md` and `.json`. Verify that `python -m src.fx_ny_orb.prereg --check` reports both files committed and unchanged.

## 5. Historical simulation and scoring

- [x] 5.1 Implement `registry.py` and `score.py --family historical`:
  - the gate first;
  - the simulation of every eligible day, saved to `hist_trades.csv.gz`;
  - the two cells with the four-series day-block bootstrap;
  - mode `'x'` per cell, and the note of the evaluation spec.

  Verify with tests that:
  - with no non-hot trades, the difference bootstrap reproduces fx_bb's `ratio_block_means` for cell 1 exactly;
  - a planted edge is KEEP and none is DROP;
  - a second scoring is refused, and the registry stays byte-identical;
  - the gate refusal leaves no outcome on disk;
  - other registries are unchanged.
- [x] 5.2 Implement `report.py`, which writes `results/fx_ny_orb/report.md`. It contains:
  - the cost header, the verdicts or "not scored yet", and the power table;
  - hot and non-hot by pair and by year: trades, exit shares, win rate with Wilson on dates, mean net R, pips and profit factor gross and net, conflicts and DST-mismatch days;
  - five example trades by a fixed rule.

  Verify with tests that two builds differ only in the generation line, and that nothing appears before scoring.
- [ ] 5.3 Run the real historical scoring once, after task 4.3. Verify that the registry has exactly two rows and every other registry is byte-identical, then build the report twice.

## 6. Frozen models and live passes

- [x] 6.1 Implement `frozen.py`: OLS HAR-X and HAR per pair on fx_nyvol's eligible days up to 2026-11-20, the JSON files under `research_models/fx_ny_orb/` and the digest manifest. Verify with tests that:
  - refitting reproduces the files byte for byte;
  - an edited file is refused;
  - the HAR-X coefficients reproduce fx_nyvol's last-fold `p_harx` within 1e-6 on its last fold's training days when fitted on the same rows.
- [x] 6.2 Implement `live.py --pass forecast|settle`, per the range-forward spec and design D7, with a fake MT5 source in the tests. Verify:
  - the early-start refusal;
  - one forecast row per day, unchanged on a rerun;
  - a settlement written once;
  - a missed day is logged as missing and never backfilled;
  - only bars labelled 14:00 or earlier are used for the forecast (truncation test);
  - protected paths are unchanged.
- [x] 6.3 Implement `live.py --pass signal`, and the settlement of signals into `forward/signal_ledger.csv`, per the range-forward spec and design D7. Verify with a fake MT5 source:
  - one signal row per pair per day, unchanged by later passes;
  - the forming bar is never used (a pass at 15:52 sees bars up to 15:30);
  - a signal is logged whatever the hot flag and the gate;
  - a missed day is marked and never backfilled;
  - the ledger's net R equals `rule.simulate_day` from the logged entry.
- [x] 6.4 Implement `score.py --family forward`: refuse before 270 settled days per pair unless it is 2028-01-31 or later; then two cells, or `UNDERPOWERED — NO DECISION`, once each. Verify with synthetic logs: the early refusal, KEEP and DROP, the underpowered label at the deadline, and a second scoring refused.

## 7. Chart display

- [x] 7.1 Implement `export.py`: the atomic write to `Common\Files\EURUSDProphet\ny_orb.tsv`, with the header block, the BG and EN texts and the 60-day rows. Verify with tests that:
  - the current day has signal fields only when `orb_hot_net_r` is KEEP, today is hot and a signal is logged, and has none in each of the three closed-gate cases;
  - no text contains an order instruction, a lot size or a promise of profit;
  - every live signal label names its historical verdict and forward record;
  - every label of a past trade says simulation;
  - every time is a Berlin label;
  - the file is replaced atomically, since a reader never sees a partial file;
  - nothing in `src/` reads it back.
- [x] 7.2 Write `mql5/Indicators/ProphetNYOrb.mq5`, per the chart-display spec and design D8. Write `scripts/fx_ny_orb/install_mt5_indicator.ps1` and `uninstall_mt5_indicator.ps1`, modelled on the gold scripts. Verify that the install script compiles the indicator in every local terminal with 0 errors, and attach the compile log to the README.
- [ ] 7.3 Write `scripts/fx_ny_orb/install_tasks.ps1` and `uninstall_tasks.ps1`, for the three scheduled tasks, Monday to Friday, local time: the forecast at 16:05, the signal every 15 minutes from 16:01 to 20:01, and the settlement at 23:05. Verify with a dry-run flag that prints the task definitions without registering them. **Owner step:** run the installer.

## 8. Docs and integration

- [x] 8.1 Write `results/fx_ny_orb/README.md` and `results/fx_ny_orb/forward/RUNBOOK.md`, add `ARCHITECTURE_DOCS.md §3.20` and a CLAUDE.md invariant entry, covering the two families, the gate, the chart rules and the facts about HAR-X selection. Verify that every path they name exists.
- [x] 8.2 Run `python -m pytest -q tests/test_fx_ny_orb_*.py` and the protection tests of fx_nyvol, fx_bb_sma, fx_bb, fx_rates, fx_session and fx_fib. Verify that all pass and `git status --short models/` is empty.

## Workflow follow-up

- The owner commits the pre-registration (task 4.3), installs the indicator and the tasks, and attaches the indicator to EURUSD and GBPUSD M15 charts.
- Report the two historical verdicts with the plain trade summary, net of the recorded spread plus 0.4 bps, not a trading result.
- The forward cells are scored once at 270 days per pair or on 2028-01-31. Archive with `/opsx:archive` after the historical verdicts, and keep the forward part running.
