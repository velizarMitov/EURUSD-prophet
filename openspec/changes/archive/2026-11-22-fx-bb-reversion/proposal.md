# Proposal

## Why

Three FX families ended without an edge on 2026-11-21:
- **fx-fibonacci:** all four cells DROP.
- **fx-session:** both LightGBM cells DROP; both MLP cells NOT TRIGGERED.
- **fx-rates:** both LightGBM cells DROP; both MLP cells NOT TRIGGERED.

Each one forecast every day, and every test could see only a large edge (54–56 %).

The owner's next hypothesis tests a published, fixed rule only on the days it fires. Barbara
Rockefeller (*The Foreign Exchange Matrix*, Ch. 8) treats a daily close outside the
Bollinger Bands (20-day SMA ± 2 SD) as a counter-trend signal. The price is said to return
toward the middle band within about 3 trading days.

The rule has no free parameters to fit (20, 2 and 3 come from the book). Its events are
pooled over six liquid pairs, so it gets many more independent trials than a two-pair
daily model.

## What Changes

- **A new research family, `fx_bb`.** It covers six pairs: EURUSD, GBPUSD, USDJPY, AUDUSD,
  USDCAD and USDCHF. It has its own package `src/fx_bb/`, config, registry and results
  directory.
- **Data from 2012-01-01.**
  - **EURUSD, GBPUSD, AUDUSD:** read-only from the fx_session H1 caches.
  - **USDJPY, USDCAD, USDCHF:** new append-only H1 caches with the bar spread, under the
    same weekly-open clock gate.
  - **A lag check, new to this family.** Each pair's hourly returns must line up with
    EURUSD at lag 0 in every calendar year.
- **USDCHF starts on 2015-02-01.** The probe of 2026-11-21 found two problems before that
  date:
  - **Shifted clock.** Its 2012-01..2014-11 bars are one hour off: the correlation with
    EURUSD peaks at lag +1 (0.76–0.95). They still pass the 96 % weekly-open check. This
    is the same defect as GBPUSD 2011.
  - **Missing data.** The broker has no USDCHF prices for 76 hours from the SNB floor
    removal on 2015-01-15.
- **Declared gap and corrupt spreads.** USDCAD 2013-05 is a declared broker gap
  (232 hours missing). A bar whose spread exceeds 50 pips is treated as corrupt, for
  example AUDUSD on 2014-01-07, where all 19 bars read 300 000 points.
- **D1 bars built from H1, closing at 17:00 New York.** These are not MT5's own D1 bars.
  - On this Berlin-clock server, MT5's D1 bars include a one-hour Sunday bar every week.
  - They also close inside the rollover hour, when the median spread is 8 points instead
    of 5.
  - A D1 bar needs at least 12 H1 bars. The close is the mid price.
- **Events at the close of day `t`.** An event is the first close outside the bands:
  - the close before was inside the bands;
  - the bands use the mid closes `t−19 … t`;
  - a 3-day cooldown per pair applies.

  The direction is −1 above the upper band and +1 below the lower band.
- **Cell 1, the rule, with no ML.** It asks whether the mean counter-trend return over
  3 trading days, in units of ATR(20), is above zero.
  - The test is a one-sided circular block bootstrap over 5-day calendar blocks. This
    covers same-day clustering across pairs and the 3-day horizon.
  - It uses every event from 2012.
- **Cell 2, an ML filter on the events.** Pooled LightGBM, on 10 oriented features, is
  compared with the training base rate on log-loss.
  - The target is binary: a profitable counter-trend move to `t+3`. There are no class
    weights.
  - It runs walk-forward from 2015-01-01 in 12-month folds, with a 5-day purge and
    embargo.
- **Two cells at alpha = 0.05/2, each scored once.** MLP is not part of this family.
- **The power check comes before any model.** `labels.md` gives the event counts and the
  minimum detectable effects. The probe found:
  - about 1 300 events from 2012 and about 1 050 from 2015, on about 750 distinct dates;
  - a detectable mean of about 0.09–0.11 ATR for Cell 1;
  - a detectable accuracy of about 54.3–55.1 % for Cell 2.
- **Gross of cost for every verdict.** The report also gives profit factor, win rate and
  mean pips, both gross and after each pair's mean spread. These are descriptive only.

## Capabilities

### New Capabilities
- `fx-bb-reversion/market-data`:
  - the six-pair H1 caches, the clock gate and the lag check;
  - per-pair start dates, declared gaps and the corrupt-spread rule;
  - the 17:00 New York mid D1 bars.
- `fx-bb-reversion/band-events`:
  - the Bollinger event definition, the first-breach and cooldown rules;
  - the 3-day ATR-unit outcome, the binary target and the pair-gap exclusion;
  - the power report.
- `fx-bb-reversion/event-features`: the fixed ten oriented features known at the close of
  `t`.
- `fx-bb-reversion/evaluation`:
  - the rule cell and its day-block bootstrap;
  - the pooled LightGBM walk-forward against the base rate;
  - the two-cell registry, one-shot scoring and the config guard;
  - the descriptive gross and after-spread statistics.

### Modified Capabilities
(none)

## Impact

- **New code and results.** The code lives in `src/fx_bb/` and the tests in
  `tests/test_fx_bb_*.py`. Results go to `results/fx_bb/`, with the new caches in
  `results/fx_bb/data/`. Frozen artifacts go to the gitignored `research_models/fx_bb/`.
- **Read-only inputs.** These are:
  - the fx_session H1 caches and loader helpers;
  - `fx_fib.score.circular_block_means`;
  - the fx_session LightGBM helper.

  fx_fib, fx_session and fx_rates keep their configs and results unchanged, and a test
  pins their digests.
- **MT5, read only.** The family needs three more symbols: USDJPY, USDCAD and USDCHF. No
  other vendor is used, and there is no network access beyond MT5.
- **Nothing else changes.** `models/`, `config.json`, the served API and every other
  registry stay untouched. The family never touches the `0.05/9` bar.
- **Docs.** `ARCHITECTURE_DOCS.md §3.17` and a CLAUDE.md invariant entry.
- **Dependencies.** None new.
