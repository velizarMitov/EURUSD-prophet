# Design

## Context

- The gold caches (`results/gold/gold_{d1,h1}.csv`) start in 2013-08 and 2015-01. MT5
  history for the symbol starts on 2013-08-08. Every year since then has been looked at,
  in the gold study and in the 2026-11-03/04 backtests.
- The H1 cache holds bid closes only. MT5's per-bar `spread` is a single value per bar:
  0.00 for 2015–2017 and a flat 0.25 or 0.50 later. It cannot show the widening at the
  daily break, so the 01:00–02:00 Sofia rise in `DESCRIPTIVE.md` cannot be told apart from
  a bid artifact with repo data.
- Labels are Europe/Berlin wall time stored as UTC (CLAUDE.md item 6). Sofia is
  Berlin + 1 h all year, including 2004–2013, because both follow the EU rule.
- Reusable pieces, all used read-only:
  - `src/gold/challengers.py`: `GoldLSTM` (heads ret, dir, vol; 5 seeds; a partial
    ensemble refused) and `GoldGarch` (D1, parameters from training rows only);
  - `src/gold/features.py`: scale-free features, with spread and volume forbidden;
  - `src/gold/targets.py`;
  - `src/forecast_eval/stats.block_bootstrap_ci`;
  - `src/forecast_eval/prereg.committed_and_clean`;
  - `src/external/tsfm.gpu_job` (the shared GPU lock).
- `gold_config.json` is pinned by round one. It is read for the LSTM settings and never
  written.

## Goals / Non-Goals

**Goals:**
- One honest, net-of-cost answer per cell, on years no one has looked at, with the rule
  frozen before any of those prices is read.
- The seen-span numbers kept clearly apart from that answer.

**Non-Goals:**
- Tuning anything on the archive. There is no parameter search; each parameter is fixed
  in the registration.
- Trading outside the window, an overnight hold, or a live signal.
- Making the walk-forward models a version, a challenger or a forward-logged model.

## Decisions

### D1. Source: Dukascopy daily M1 candle files, bid and ask

The files are `https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM-1:02d}/{DD:02d}/{BID|ASK}_candles_min_1.bi5`:
one LZMA file per side per day, of fixed-size big-endian records. The first task
verifies the record layout and the price divisor on a day inside the MT5 overlap before
any bulk fetch.

Storage is one gzip CSV per year per side under `research_data/dukascopy/`,
which is gitignored. Downloads are limited to at most 2 requests per second, with
exponential backoff and 5 retries.

- **Why M1, not ticks:** M15, H1 and D1 are all built from it, and it is about 7,000
  files instead of about 70,000.
- **Alternatives rejected:**
  - yfinance `GC=F`: futures, not spot, and intraday history only for recent years;
  - HistData: bid only, so the spread question stays open.

### D2. Archive read guard

`archive.load(kind, start, end)` returns prices only through one of two gates:
- the requested range is on or after 2013-08-08 (the overlap and later);
- `committed_and_clean` holds for both registration files.

`archive.timestamps()` and `archive.counts()` are always allowed. Every price-returning
path in `src/gold/window/` goes through `archive.load`, and a test asserts that no other
module opens the raw files.

### D3. Clock and scale verification

On the overlap (2013-08-08 to 2014-12-31):
- H1 mid closes are built from the archive;
- they are inner-joined with the MT5 cache on label;
- the correlation of hourly log returns is computed at lags −2 to +2.

The thresholds, from the spec, are lag 0 best, ρ ≥ 0.95 and a median |Δclose| of at most
$1.00. The MT5 cache is bid-only, so the comparison uses archive mid against MT5 bid plus
the median MT5 half-spread. The weekly open label distribution is tabulated over the whole
archive from timestamps only and written to the manifest.

### D4. Walk-forward schedule

The calendar year is the refit unit.
- Years 1–3 of the archive are fit-only.
- For year Y, every model is fit on all bars labelled before Y-01-01, then predicts Y.
- The same `GoldLSTM` and `GoldGarch` classes are used, through `challengers.make`, with
  settings from `gold_config.json`.
- Seeds are `[42..46]`. H1 uses the H1 feature set and D1 the D1 set, exactly as the gold
  study does.
- Early stopping uses the classes' own last-1/8 split of the fit rows. No archive row
  after the fit end is ever seen.

Artifacts go to `research_models/gold/window/<model>/<YYYY>/`, with SHA-256 digests in
`results/gold/window/artifact_manifest.csv`. Predictions go to
`research_models/gold/window/predictions_{h1,d1}.csv.gz`. Those are gitignored, but their
digests are tracked.

The run holds `gpu_job()` for its whole duration. There are about 7 yearly refits of
5 seeds for H1, the expensive part, which is expected to take hours. The run resumes from
the last completed year.

- **Alternative rejected:** a monthly refit. That is 12 times the cost, for a benefit the
  gold study never showed.

### D5. Gate thresholds

- Each gate compares today's forecast with a quantile of the cell's own forecasts on the
  previous 250 trading days: the 80th percentile for `h1_gated_direction`, and the median
  for the GARCH and `d1_lstm` gates.
- For `h1_gated_direction`, the pool is the in-window forecasts of those days, as-of
  labels 14–20.
- The quantile is computed with `numpy.quantile(..., method='linear')` on a strictly-prior
  slice.
- With fewer than 250 prior days, the gate is closed. So each cell's first tradable day
  is about one year after its first prediction year.

### D6. Simulation

- Mid is (bid + ask) / 2 from the archive, per bar close. Signals use mid. Fills use the
  ask for a buy and the bid for a sell, at the stated bar's close.
- The cost floor is applied per trade: if the realised round-trip cost, as a fraction of
  the entry mid, is below 0.0062 %, the difference is charged.
- The stress figure uses a 0.0122 % floor.
- A daily series has one value per eligible weekday: the sum of that day's trade returns,
  or 0.

Eligibility means every M15 bar labelled 14:30–21:45 Berlin is present and the weekday
is Monday to Friday. For `h1_gated_direction`, at most 7 one-hour trades can sum into one
day's value. That keeps every cell on the same daily unit and block length.

### D7. Power, from the seen span only

The registration computes, for each cell, the standard deviation of its daily net series
replayed on MT5 2015-01 to 2022-10. That span has spread-free bid data, so the `bid + 0.25`
fill is an approximation, stated as such. The neural cells use the honest validation-block
predictions.

From that standard deviation it gives:
- N = the expected eligible days in the confirmation span (about 250 per year × the
  number of walk-forward years after warm-up);
- the detectable mean daily return at 80 % power and alpha 0.0125 (normal
  approximation, block dependence ignored and stated).

No archive return enters this, because the registration is written before any archive
price is read.

The power function records only dispersion and counts, never a mean, so it cannot steer
a rule. For the neural cells it draws on the validation-block predictions, which overlap
the D9 replay span, and only their dispersion is recorded. A test asserts that it returns
only standard deviations and counts.

### D8. Registry and report

- Each confirmation run appends one row per cell to `results/gold/gold_window_hypothesis_log.csv`
  with the gold registries' columns: `n`, `date`, `cell`, `family`, …, `verdict`, `notes`.
- The `notes` hold: the trades, the days, the gross and net mean, the stress mean, the hit
  rate, and the "simulated, net of the stated costs" label.
- `confirmation_report.md` renders the same information with a per-year table.
- A run where all four cell rows already exist appends nothing.

### D9. Seen-span replay

This runs after registration. It covers November 2022 to 2026 on MT5 M15 and H1, with the
existing honest predictions:
- `validation_predictions_{d1,h1}.csv` and `test_predictions_{d1,h1}.csv`, for the
  neural gates;
- the GARCH refit on the same yearly schedule over the MT5 caches.

M15 is read from MT5 read-only, because the repo has no gold M15 cache. Fills are bid,
plus the per-bar MT5 spread or the floor. It is written to `replay_seen.md` with the
"already seen, never evidence" banner.

### D10. Placement

The code goes in the new package `src/gold/window/`. It imports gold, forecast_eval and
external_tsfm helpers read-only. It does not read `config.json`, writes nothing under
`models/` and is not imported by `api.py`. There is no page or MT5 change in this change.

## Risks / Trade-offs

- **[Dukascopy gold history may start later than 2004, or have thin early years]** →
  The availability check reads timestamps only and runs before registration. The
  registration records the actual span and recomputes N and power. If fewer than 4
  walk-forward years remain after warm-up, the registration says so, and the owner
  decides before committing.
- **[Archive spreads (ECN) differ from the broker's]** → The broker floor makes every
  trade at least as expensive as today's window median. A 0.0122 % stress figure is
  reported.
- **[The 2004–2013 market (pre-2013 crash, the 2011 peak) is a different regime]** →
  That is the point of an independent span. A rule that only works in 2022–26 should fail
  there. The per-year table shows whether a CLEARED cell depends on one year.
- **[Walk-forward neural fits are slow and could hang the GPU]** → There is a resumable
  per-year checkpoint and the shared GPU lock. A failed year leaves that year without
  neural predictions, so those cells trade nothing then. It is never patched with a
  partial ensemble.
- **[Four cells at 0.0125 over about 6 years are weakly powered]** → The power statement
  says so before commit. An UNDERPOWERED or DROP result is a valid answer.
- **[Data snooping through the descriptive report]** → The four rules were chosen after
  the 2015–2026 statistics were seen. That is exactly why the confirmation uses only
  2004–2013. Each rule's parameters (80th percentile, median, 15:30–16:30 range, 20:00
  cut-off) are fixed in the registration, and no other variant is ever scored on the
  archive.
- **[Copilot auto-commit could commit the registration before the owner reviews it]**
  (this happened in `ab420d3` and `dec983e`) → The owner turns auto-commit off before the
  registration is written. Task 4.2 checks `git log` first.

## Migration Plan

1. Build the download, the guard and the verification, and the descriptive report.
2. Fetch the archive. Verify the clock and scale on the overlap.
3. Write the registration. The owner reviews and commits it alone.
4. Fit walk-forward. Score the confirmation once. Run the seen-span replay.
5. Record the verdicts in `results/gold/RESULTS.md`.

Rollback: before step 3, delete `src/gold/window/` and the gitignored folders. After
step 3, the family stays registered. An unfinished run only means fewer verdicts.
