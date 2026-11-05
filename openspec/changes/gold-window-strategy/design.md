# Design

## Context

- Data available:
  - `results/gold/gold_h1.csv` from 2015-01-02 and `gold_d1.csv` from 2013-08-08. Both
    are bid OHLC only.
  - MT5, read-only, has gold H1 bars with a per-bar `spread` from 2014-12 and M15 bars
    only from 2022-08-12. The bar spread is 0 for 2015–2017 and partly 2017, and a flat
    0.25 or 0.50 later.
- Every year has been looked at, and the four rules were chosen after the descriptive
  statistics. The owner declined an independent history on 2026-11-04. So nothing this
  change produces is evidence (proposal, Why).
- Labels are Europe/Berlin wall time stored as UTC (CLAUDE.md item 6). Sofia is
  Berlin + 1 h all year.
- Reusable pieces, all used read-only:
  - `src/gold/challengers.py`: `GoldLSTM` (heads ret, dir, vol; 5 seeds; a partial
    ensemble refused) and `GoldGarch` (D1, parameters from fit rows only), built through
    `challengers.make` and `build_inputs`;
  - `src/gold/targets.build_targets`;
  - `src/forecast_eval/stats.block_bootstrap_ci`;
  - `src/external/tsfm.gpu_job`, the shared GPU lock;
  - `src/gold/data.resolve_gold_symbol`.
- `gold_config.json` is pinned by round one. It is read for the model settings and never
  written.

## Goals / Non-Goals

**Goals:**
- Honest model inputs: every prediction comes from a fit that ended before its year.
- Honest costs, with an explicit floor.
- A report that cannot be mistaken for a verdict.

**Non-Goals:**
- Any verdict, registration, registry row, or forward logging.
- Tuning: no threshold, window or horizon search, and no extra rule variants.
- Making the walk-forward models a version, a challenger or a logged model.

## Decisions

### D1. Bars and spreads

`bars.py` returns three frames for a given span:
- **H1:** the cache, plus the MT5 per-bar spread joined on label.
- **D1:** the cache.
- **M15:** MT5 bars with their spread.

MT5 fetches are chunked by year, because one long `copy_rates_range` fails with
"Invalid params". They are saved once to gitignored
`research_models/gold/window/mt5/{h1_spread,m15}.csv.gz`, with digests in
`artifact_manifest.csv`, so a backtest re-run reads the same inputs. Each function takes
frames, so the tests never need MT5.

- **Alternative rejected:** an H1 version of the breakout. It would change a rule the
  owner saw, and the shorter M15 span is stated in the report instead.

### D2. Walk-forward schedule

The calendar year is the refit unit, on an expanding window.
- H1 is fit-only for 2015–2017, then predicts 2018 onward. That is about 9 yearly fits of
  `h1_lstm`, 5 seeds each.
- D1 is fit-only for 2013 (partial), 2014 and 2015, then predicts 2016 onward. `d1_lstm`
  and the D1 GARCH are each refit about 11 times.
- For year Y, inputs are built over the whole cache, which is causal: features read bars
  ≤ t. The model is fit on positions whose bar and next-bar target are labelled before
  Y-01-01. It predicts every row labelled in Y.
- Early stopping uses the classes' own last-1/8 split of those fit rows.

Artifacts go to `research_models/gold/window/<model>/<YYYY>/`, and predictions to
`research_models/gold/window/predictions_{h1,d1}.csv.gz`, with digests in
`artifact_manifest.csv`. The run resumes from the last completed year. It holds `gpu_job()` only
when TensorFlow sees a GPU. The `.venv` TensorFlow on this PC is CPU-only, and there the lock
would only push the hourly external pass onto the same CPU. A full H1 fit took 172 s on CPU in
the gold study.

- **Alternative rejected:** a monthly refit, about 12 times the GPU time for a refit
  frequency the gold study never showed helps.

### D3. Gates

- Each gate compares today's forecast with a quantile of the rule's own forecasts on the
  previous 250 trading days: the 80th percentile for `h1_gated_direction`, and the median
  for the GARCH and `d1_lstm` gates.
- For `h1_gated_direction`, the pool is the in-window forecasts of those days, as-of
  labels 14–20.
- The quantile uses `numpy.quantile(..., method='linear')` on a strictly-prior slice.
- With fewer than 250 prior days, the gate is closed.

### D4. Simulation

- Signals use the bid close. A buy fills at the bid plus the bar's spread, and a sell at
  the bid.
- A round trip whose spread cost, as a fraction of the entry price, is below 0.0062 % is
  topped up to it. The stress series uses a 0.0122 % floor.
- The daily series has one value per eligible weekday: the sum of that day's trade
  returns, or 0. For `h1_gated_direction`, up to 7 one-hour trades sum into one day.
- Eligibility:
  - the H1 rules need H1 bars labelled 14:00–21:00;
  - the breakout needs M15 bars labelled 14:30–21:45;
  - every rule needs a Monday-to-Friday day.

### D5. Report contents

For each rule, `backtest.md` gives, net and at the stress cost:
- the span, the eligible days and the trades;
- the mean daily return with a 95 % moving-block bootstrap interval (block 20, 2,000
  resamples, seed 42);
- the total compounded return, maximum drawdown and hit rate;
- a per-year table;
- comparators on the same days:
  - the rule with its gate always open;
  - always long from 16:00 to 23:00 Sofia;
  - a placebo that rolls the rule's gate series by every offset from 20 to N − 20 days
    and reports the share of offsets doing at least as well.

Each section carries the descriptive label. The words CLEARED and DROP are never written.
A test asserts both.

### D6. Placement

The code goes in the top-level package `src/gold_window/`, not under `src/gold/`. The
`src/gold` isolation contract (its package docstring, and
`tests/test_gold_isolation.py::test_gold_never_reads_spread_or_volume_as_data`) says
spread is used nowhere, "not as costs". This family is net of cost by design, so it gets
its own package with its own contract instead of an exception carved into gold's. The
gold contract and its tests stay byte-identical.

`src/gold_window/` imports gold, forecast_eval and external_tsfm helpers read-only. It
does not read `config.json`, writes nothing under `models/`, and is not imported by
`api.py`. This change adds no page and changes nothing in MT5.

## Risks / Trade-offs

- **[The report looks like evidence]** → Every section is labelled. No verdict words are
  used. The proposal and `RESULTS.md` say that the rules were chosen after the data was
  seen.
- **[MT5 spread is 0 or flat for many years]** → The 0.0062 % floor applies. The stress
  figure doubles it.
- **[The breakout covers only about 3 years]** → The report states its span. The gate
  warm-up comes from the GARCH, which has D1 history from 2013, so it costs no M15 days.
- **[Walk-forward neural fits are slow, and could hang the GPU or contend with the hourly
  pass]** → There is a resumable per-year checkpoint and the shared GPU lock. A failed
  year has no neural predictions, and its rules trade nothing that year.
- **[Copilot auto-commits mid-task]** (as in `9211520`) → Nothing in this change depends
  on a commit, so an early commit is only untidy. Before any commit Claude makes, it
  checks `git log`.

## Migration Plan

Build the bars, rules and simulation with tests. Run the walk-forward. Run the backtest.
Record the summary in `RESULTS.md`.

Rollback: delete `src/gold_window/`, `results/gold/window/` and
`research_models/gold/window/`. Nothing else depends on them.
