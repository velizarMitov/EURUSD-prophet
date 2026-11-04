# Tasks

## 1. Scaffolding and descriptive report

- [ ] 1.1 Create `src/gold/window/` with `__init__.py` and `window_config.json`. The
  config holds:
  - the paths;
  - the window labels 14:30–22:00 Berlin;
  - the four cells' parameters (80th percentile, median, 250 days, the 15:30–16:30 range,
    the 20:00 cut-off);
  - alpha 0.0125, block 20, 2,000 resamples, seed 42;
  - the cost floor 0.0062 %, the stress 0.0122 %, the trade minimum 300, warm-up 3 years.

  Add `research_data/` to `.gitignore`.

  Verify: `git check-ignore research_data/dukascopy/x` succeeds, and a test loads the config
  and asserts each value.
- [ ] 1.2 Write `describe.py`, which reproduces the 2026-11-04 statistics from the MT5
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

## 2. Archive download, clock and guard

- [ ] 2.1 In `dukascopy.py`, fetch one day of BID and ASK M1 candles in the overlap
  (2014-03-03), and determine the record layout and the price divisor. Fix them as
  constants, with a test on a stored fixture of that day.

  Verify: the decoded mid close at 15:00 Berlin is within $1.00 of the MT5 cache close plus
  the half-spread.
- [ ] 2.2 Add the bulk downloader:
  - resumable, at most 2 requests a second, backoff with 5 retries;
  - writes year files to `research_data/dukascopy/`;
  - records gaps;
  - writes `results/gold/window/data_manifest.csv` with each year's digest, span, days and
    gaps.

  Verify with tests using a fake HTTP layer:
  - a resume skips stored days;
  - a failed day becomes a gap;
  - the manifest digest matches the file.
- [ ] 2.3 In `clock.py`, convert to Berlin labels and build M15, H1 and D1 bars from M1.

  Verify with tests:
  - 13:00 UTC in July is labelled 15:00 and in January 14:00;
  - an H1 bar is the last M1 close of its hour;
  - the D1 day is the Berlin calendar day.
- [ ] 2.4 Add the archive read guard (design D2): `load()` refuses for any range before
  2013-08-08 unless both registration files are committed and clean. `timestamps()` and
  `counts()` are always allowed.

  Verify with tests:
  - a refusal before registration;
  - success with patched `committed_and_clean`;
  - a static check that no other `src/gold/window` module opens `research_data` files.
- [ ] 2.5 Run the full download, then the overlap verification (D3) and the weekly-open
  table, writing both into the manifest.

  Verify:
  - lag 0 is best;
  - ρ ≥ 0.95;
  - the median |Δ| is at most $1.00;
  - the first available archive day is reported to the owner.

  If a check fails, stop and report. Do not register.

## 3. Rules and simulation

- [ ] 3.1 In `rules.py`, implement the four cells as pure functions from bars and
  forecasts to trade lists: entry time, side, exit time and gate value. Also implement
  window eligibility and the strictly-prior gate quantiles (D5).

  Verify with unit tests for every scenario in the window-strategy spec, among them:
  - a closed and an open gate;
  - P(up) = 0.47 with the gate open giving a short;
  - no breakout;
  - the first break down at 17:15 Sofia, with later breaks ignored;
  - the Monday call coming from the last closed D1 bar;
  - the 250-day warm-up;
  - a missing 18:00 M15 bar excluding the day;
  - no position open after 23:00 Sofia.
- [ ] 3.2 In `simulate.py`, implement ask/bid fills, the per-trade cost floor and stress,
  and the daily series with zeros on non-trade days (D6).

  Verify with tests:
  - an archive round trip of 0.004 % is charged 0.0062 %;
  - a long fills at the ask and exits at the bid;
  - a seven-trade H1 day sums into one daily value;
  - no swap is ever charged.

## 4. Registration

- [ ] 4.1 In `prereg.py`, write the registration builder:
  - the rules and parameters from `window_config.json`;
  - the confirmation span (first walk-forward year through 2013-07-31);
  - the archive digests from the manifest;
  - the frozen LSTM settings and the digest of `gold_config.json`;
  - the power statement from the seen span only (D7).

  The writer refuses if verification failed or the files are already committed.

  Verify with tests:
  - the power function returns only dispersion and counts;
  - a second write after a commit raises;
  - the builder reads no archive price before registration (guard patched to raise).
- [ ] 4.2 Check `git log` for new Copilot commits and ask the owner to turn Copilot
  auto-commit off. Then run `python -m src.gold.window.prereg --write` and show the owner
  `PRE_REGISTRATION.md`, including the span, N and the detectable effect. Commit only the
  two registration files, after the owner approves.

  Verify: `committed_and_clean` holds for both files, and the commit contains nothing else.

## 5. Walk-forward predictions

- [ ] 5.1 In `walkforward.py`, implement the yearly expanding refit of `h1_lstm`, `d1_lstm`
  and the D1 GARCH through `challengers.make` (D4):
  - it refuses before registration;
  - it holds `gpu_job()`;
  - it checkpoints per year;
  - it writes artifacts and digests to `research_models/gold/window/` and
    `artifact_manifest.csv`.

  Verify with tests on a small synthetic series:
  - every fit ends before its prediction year;
  - a year missing a seed yields no neural predictions;
  - a resume skips completed years;
  - nothing is written under `models/` or `research_models/gold/{d1_lstm,h1_lstm,versions}`.
- [ ] 5.2 Run the walk-forward on the archive. Report the per-year fit times and that every
  year has all 5 seeds, or name the years that do not.

  Verify:
  - the prediction files' digests are in the artifact manifest;
  - `git status` shows no change under `models/` or in the existing gold registries.

## 6. Confirmation and replay

- [ ] 6.1 In `confirm.py`, implement the one-shot scoring (spec "scored once"):
  - it refuses before registration and when a digest differs;
  - it builds the four daily series on the confirmation span;
  - it computes the bootstrap interval;
  - it labels CLEARED, DROP or UNDERPOWERED;
  - it appends once to `results/gold/gold_window_hypothesis_log.csv`;
  - it renders `confirmation_report.md` with per-year rows and the simulated-only wording.

  Verify with tests:
  - a synthetic positive series is CLEARED;
  - a positive mean whose interval includes 0 is a DROP;
  - 250 trades is UNDERPOWERED;
  - a second run appends nothing;
  - the report has no buy or sell instruction;
  - the other gold registries are byte-identical.
- [ ] 6.2 Run the confirmation once and show the owner the report.

  Verify: the registry holds exactly four new rows, and the report matches them.
- [ ] 6.3 In `replay.py`, write the seen-span replay (D9) to `replay_seen.md`, under the
  "already seen, never evidence" banner. It refuses before registration, and MT5 is read
  only for M15 bars and spread.

  Verify with a test that it refuses before registration and never writes to the
  registry. Then run it and show the owner both reports side by side.

## 7. Documentation and integration

- [ ] 7.1 Add the window-strategy subsection to `ARCHITECTURE_DOCS.md` §3.12 and one
  bullet to `CLAUDE.md` gold item 7. Record the verdicts and the replay summary in
  `results/gold/RESULTS.md`.

  Verify:
  - `pytest tests/test_gold_*.py` passes;
  - `openspec validate gold-window-strategy --strict` is clean;
  - `git status` shows changes only in `src/gold/window/`, `tests/test_gold_window_*`,
    `results/gold/window/`, the new registry, the docs and `.gitignore`.
