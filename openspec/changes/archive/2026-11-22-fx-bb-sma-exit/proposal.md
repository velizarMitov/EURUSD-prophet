# Proposal

## Why

`fx-bb-reversion` scored its two cells once on 2026-11-22, and both were DROP.
- **The rule.** Holding a Bollinger first breach blindly for 3 days averaged +0.014 ATR
  (95 % interval −0.06 to +0.09).
- **The filter.** The pooled LightGBM filter was worse than the base rate.

A read-only audit found no code defect.

The owner's next question is about a trade a person would actually place on the same
events. It enters at the breach close, takes profit at the 20-day mean (`SMA20[t]`), stops
at the same distance on the other side, and exits after 10 trading days if neither level is
hit. Exits are checked hour by hour on H1 bars, and costs use the recorded spread. This is
a different payoff from the 3-day close-to-close move, so it is a new hypothesis family.

**This family is a follow-up on events whose 3-day outcomes were already seen.** The
registry note records that as a fact, and the result can only confirm or reject the
pre-declared rule below.

## What Changes

- **A new research family, `fx_bb_sma`.** It has its own package `src/fx_bb_sma/`, config,
  registry `results/fx_bb_sma/fx_bb_sma_hypothesis_log.csv` and results directory.
- **The data is reused read-only and never refreshed.**
  - It covers six pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD and USDCHF.
  - It uses the fx-bb-reversion H1 caches and the 17:00 New York D1 bars, with their clock,
    lag, gap and corrupt-spread (more than 50 pips) rules.
  - USDCHF starts on 2015-02-01 and every other pair on 2012-01-01.
- **Entry.** Each trade opens on the first D1 mid close outside the Bollinger Bands
  (20, 2 SD), at that close.
  - The direction is −1 above the upper band and +1 below the lower band.
  - A pair never holds two trades at once.
- **Exit, scanned hour by hour on the H1 bars after the entry close.**
  - **Take profit** at the fixed `SMA20[t]`.
  - **Stop loss** at `dist = |Close[t] − SMA20[t]|` on the other side.
  - **When one bar touches both, it counts as a stop.**
  - **Time stop** at the D1 close of trading day `t+10`.
- **The verdict is net of spread.** Each trade's result is in units of risk,
  `R = PnL / dist`, after half the recorded spread at entry and half at exit (a full
  round trip from mid). Slippage, swap and commission are not modelled.
  - **This departs from the gross-of-cost rule of the earlier fx families,** at the
    owner's request for this family.
  - A bid/ask control fill is reported for parity and never changes a verdict.
- **Cell 1, the rule.** It asks whether the mean net R per trade is above zero, using
  every eligible trade from 2012.
- **Cell 2, a LightGBM filter.** It uses the ten fx-bb-reversion features. The target is
  whether the trade made money after spread.
  - It runs walk-forward from 2015 and is compared with the base rate on log-loss.
  - The purge is measured from each trade's exit date.
- **Two cells at alpha = 0.05/2, each scored once.** Both use a day-block bootstrap, with
  blocks of 10 business days to cover the longest trade.
- **The trade count and the minimum detectable effects** go in `labels.md` before any
  model trains. That report shows no exit reason and no result.
- **A plain-language section in the report.** It gives the share of exits at the take
  profit, at the stop and at the time stop; win rate; average win and loss in pips; profit
  factor gross and net; total R; and five example trades.

## Capabilities

### New Capabilities
- `fx-bb-sma-exit/trade-rule`: the entry on a first band breach, the no-overlap rule, the
  H1 exit scan with its take profit, stop and conflict rule, the 10-day time stop, the
  spread charges, and the exclusions. It reuses the fx-bb-reversion data read-only.
- `fx-bb-sma-exit/evaluation`: the two cells, the day-block bootstrap, the walk-forward,
  the registry with one-shot scoring and the config guard, and the power report written
  before any model.
- `fx-bb-sma-exit/trade-report`: the plain-language trade summary, the five example
  trades, and the descriptive breakdowns.

### Modified Capabilities
(none)

## Impact

- **New code and results.** The code lives in `src/fx_bb_sma/` and the tests in
  `tests/test_fx_bb_sma_*.py`. Results go to `results/fx_bb_sma/`, and frozen models to
  the gitignored `research_models/fx_bb_sma/`.
- **Read-only inputs.**
  - From `src/fx_bb`: the loader, the D1 builder, the indicators and the features.
  - The caches in `results/fx_bb/data` and `results/fx_session/data`.
  - From fx_session and fx_fib: the LightGBM helper and the bootstrap check.

  A test pins the configs of fx_bb, fx_rates, fx_session and fx_fib to the digests
  recorded with their verdicts.
- **Nothing else changes.** `models/`, `config.json`, the API and every other registry
  stay untouched, and the family never touches the `0.05/9` bar. It has no order, broker
  or live code: the exits are simulated on history only.
- **Docs.** `ARCHITECTURE_DOCS.md §3.19` and a CLAUDE.md invariant entry. §3.18 is
  reserved by the pending change fx-ny-volatility.
- **Dependencies.** None new.
