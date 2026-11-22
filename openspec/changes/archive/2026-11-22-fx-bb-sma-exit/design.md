# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-bb-sma-exit/*`.

**What fx-bb-reversion established on 2026-11-22.** It is archived at
`openspec/changes/archive/2026-11-22-fx-bb-reversion/`, with its results in
`results/fx_bb/`.

- **Data.** The six H1 caches pass the clock and lag checks from each pair's start.
  - USDCHF is one hour off before 2014-12, so it starts on 2015-02-01.
  - USDCAD 2013-05 and GBPUSD 2014-03 are declared gaps.
  - AUDUSD and USDJPY carry corrupt spreads on 2014-01-07.
- **D1 bars close at 17:00 New York.** The closing bar is labelled 22:00 Berlin, or 21:00
  in DST-mismatch weeks. The next day opens with the 23:00 rollover bar.
- **Events.** 1 369 first breaches from 2012. The mean closing-bar spread is 0.68–1.22 pips
  by pair.
- **Its 3-day outcomes are known.** The mean is +0.014 ATR and the win rate 50.7 %. This
  family is designed after that result, which the registry note records.

**How the earlier families' code is reused.**
- **fx_bb, read-only:**
  - `data_loader.load_all` runs the clock and lag checks.
  - `daily.build_all` and `daily.trading_dates` give the D1 bars and the trading dates.
  - `events.indicators` and `events.reference_calendar` give the bands and the reference
    dates.
  - `features.build_features` gives the ten features.
- **fx_session, read-only:** `model_lgbm` and `artifacts`.
- **fx_fib, read-only:** `circular_block_means`, for the self-check.

## Goals / Non-Goals

**Goals:**
- A single, pre-declared simulation of the owner's trade: take profit at the mean, a
  symmetric stop, a 10-day time stop and hourly exits.
- A conservative tie rule and fills without optimism.
- One-shot verdicts net of the recorded spread, with the discipline of the earlier
  families.

**Non-Goals:**
- Any search over the band, the take-profit level, the stop multiple, the time stop or the
  features.
- Position sizing, portfolio limits across pairs, swap, commission or slippage models.
- Order, broker, live or forward code. Nothing is added to the API or the UI.
- Re-scoring fx-bb-reversion, or changing its files.

## Decisions

### D1. Its own package, with the event and feature definitions pinned to fx_bb

**Layout.** `src/fx_bb_sma/` holds:
- `fx_bb_sma_config.json`;
- the modules `trades`, `labels`, `splits`, `walk_forward`, `registry`, `score` and
  `report`;
- the registry `results/fx_bb_sma/fx_bb_sma_hypothesis_log.csv`.

**Config.** Its config repeats fx_bb's `pairs`, `pip`, `data`, `bands`, `atr_period`,
`reference_quorum` and `features` blocks. A test asserts they equal fx_bb's, so the
events and features cannot drift from the validated definitions.

**Why not import fx_bb's config directly.** The guard must cover every input of this
family's verdicts, so its digest must change when anything here changes.

### D2. The signal stream and the no-overlap rule

`trades.simulate(pair)` walks the pair's first breaches in date order.
- **Opening.** A breach on `t` opens a trade when the pair's last trade exited at or before
  the entry bar, the closing H1 bar of `t`.
- **Excluded trades** (D6) hold their slot until the closing bar of `t+10`. So the set of
  trades never depends on whether a corrupt or holed window happened to hit a level.
- **Breaches skipped** because a trade was open are counted per pair.

An exit at the time stop and a new entry can share one close: the exit happens first.

### D3. The hourly exit scan

**Which bars.** The bars scanned are the pair's H1 bars labelled after the entry bar,
through the closing bar of the pair's valid day `t+10`, using `daily.trading_dates`.
- Bars of short days, with fewer than 12 bars, are scanned: the market traded.
- They do not count toward the 10 valid days.

**The checks.**
- **Mid prices.** A bar's mid high is its bid high plus half its spread, and the same for
  the low and the open.
- **Long.** The stop is hit when `mid_low ≤ SL`, and the take profit when `mid_high ≥ TP`.
  A short is the mirror image.
- **Conflict.** A bar that hits both is a stop loss.

**Fills.**
- **Take profit:** at TP.
- **Stop:** at SL, or at the mid open when the bar opens beyond SL. This applies only to
  the stop.
- **Time stop:** the D1 mid close of `t+10`.

**The first bar scanned is the 23:00 Berlin rollover bar.** Its recorded spread is wider,
which raises the cost of an exit there. The mid limits the effect on touches, since a
wider spread moves the bid down and adds back half the width.

**Alternative rejected:** checking M15 or ticks inside the bar to order the take-profit
and stop touches. These caches are H1 only. The owner chose the worst case.

### D4. Costs, net of the recorded spread only

- **Gross result.** `dir · (exit_mid − entry_mid)`.
- **Net result.** The gross result minus half the entry bar's spread, minus half the exit
  bar's spread:
  - the entry bar is the D1 closing bar of `t`;
  - for a take profit or a stop, the exit bar is the H1 bar that hit;
  - for a time stop, it is the closing bar of `t+10`.

  This is a full round trip from mid: ask in and bid out for a long.
- **R.** `net / dist`. Pips use the pair's pip size.
- **Cost in R** is `spread / dist`, so it is small when the bands are wide and larger when
  they are narrow. This weighting is part of the hypothesis.
- **The recorded spread** is MT5's per-bar spread field. It may be below the spread at the
  moment of a fill, so costs can be understated. The registry note and the report say
  "net of the recorded spread only".
- **Swap** (up to 10 nights), commission and slippage are not modelled.

**This family departs from the gross-of-cost rule of the earlier fx families.** The owner
asked for this cell to be net. The verdict is still not a trading result.

### D5. The bid/ask control

The same trades and the same mid levels, with fills on executable prices:
- **Long:** it enters at `bid_close + spread·point` (the ask) and exits on the bid.
  - The take profit is hit when `bid_high ≥ TP`, and the stop when `bid_low ≤ SL`.
  - It fills at the level, or at the bid open on a gap through the stop.
- **Short:** it enters at the bid and exits on the ask. The ask is the bid plus the bar's
  spread.
- **Same order of checks and same tie rule.**

The control's mean R goes in the Cell 1 registry row. A flag is set when it differs from
the net mid mean by more than 0.05 R. The tolerance is wider than fx_bb's 0.02 ATR
because touch timing differs between the two fills, not only the price. The flag never
changes a verdict.

### D6. Exclusions

Checked in this order:
1. **`open at data end`:** day `t+10` does not exist yet and no level was hit.
2. **`pair gap`:** fx_bb's reference calendar, over the window `t−19 … t+10`. The window
   is fixed, so the exclusion never depends on when the trade exits.
3. **`corrupt spread`:** any scanned H1 bar, or the entry bar, has a spread above 50 pips.

### D7. Power before any model

**What the report needs.** The no-overlap rule needs each trade's exit time, so
`labels.py` calls `trades.exit_times`.
- That function returns only the exit timestamp.
- The reason, the price and the result come from a separate function, which the label
  report never calls. A test asserts this.
- The report counts trades, skipped breaches, exclusions and distinct dates.

**Cell 1's minimum detectable mean** uses a declared SD of 1.0 R.
- A take profit is about +1 R gross and a stop about −1 R. Time stops fall in between,
  and gaps go beyond −1.
- So 1.0 is a fixed, outcome-blind ceiling for the dispersion. No data is read for it.

**Cell 2** uses `0.5 + 2.80 · 0.5 / √n`.

**Expected size.** About 1 000–1 250 trades are expected; fewer than the 1 369 first
breaches, because trades last up to 10 days. With 1 000 trades on 750 dates, the minimum
detectable mean is about 0.09–0.10 R.

### D8. Bootstrap blocks of 10 business days

- **How it works.** S_d and N_d are summed per business day of entry. The same blocks are
  drawn for S and N, and the statistic is ΣS*/ΣN*.
  - Cell 1 sums the net R.
  - Cell 2 sums the log-loss difference.
- **Settings.** 10 000 resamples and a fixed seed. With N ≡ 1, the bootstrap must reproduce
  `circular_block_means`.
- **Why 10 days.** A trade lives up to 10 business days, so trades entered on different
  pairs up to 10 days apart share market days and are dependent.
  - Blocks of 5 days, as requested, would break that dependence and understate the
    variance. That would loosen the bar.
  - Blocks of 10 cover the longest trade. This is the more conservative choice and the
    project's practice for overlapping targets (fx_rates).

### D9. The walk-forward and the model

- **Folds.** Expanding 12-month folds from 2015-01-01.
  - **Fit:** trades whose exit date is more than 10 business days before the fold start,
    a 5-day purge plus a 5-day embargo measured from the actual exit.
  - **Validation:** the last 15 % of those trades by entry date, with the same gap before
    it.
- **Model.** fx_bb's LightGBM settings: 4 leaves, at least 20 rows per leaf, learning rate
  0.02, `lambda_l2` 1.0, bagging 0.8, up to 2 000 trees with early stopping after 100,
  and seed 42.
- **Reference.** The prior is the fit slice's win rate.
- **Target.** `net > 0`. A net result of exactly zero is a loss, because it did not make
  money.
- **Features.** Events without full features (warm-up) leave Cell 2 only.

### D10. The report

- **Language.** The plain-language section is in Bulgarian, for the owner. The rest
  follows the English format of the earlier fx reports.
- **What it contains.** The specs list the tables. The five examples follow the fixed rule
  in the trade-report spec, never the result.
- **When it shows results.** The report reads only saved files, so two builds match. It
  shows a cell's numbers only after that cell's verdict.

## Risks / Trade-offs

- **[The events were seen in fx-bb-reversion]** → The rule and its parameters come from
  the owner's request and are frozen now. The note records the earlier DROP. A KEEP here
  would still need forward data before any real use.
- **[The recorded spread may understate the cost, especially at rollover and at news
  times]** → It is labelled "net of the recorded spread only". The bid/ask control shows
  how much fill timing matters. Swap is not modelled.
- **[H1 cannot order a take-profit and a stop touch inside one hour]** → The stop wins, as
  the owner chose. The report counts the conflict bars, so the size of this pessimism is
  visible.
- **[A stop gap at the Sunday reopen or on news can lose more than 1 R]** → It fills at the
  mid open. The distribution of R is in the report.
- **[With 10-day blocks, the test has less power than with 5-day blocks]** → This is the
  honest variance for 10-day trades. The label report states the minimum detectable effect
  before any model.
- **[Trade count and power are unknown until labels.md]** → No probe was run, because
  counting trades needs exit times. The label report is the first and only place they
  appear before the verdict.

## Migration Plan

Nothing to migrate. Rolling back means deleting `src/fx_bb_sma/`,
`tests/test_fx_bb_sma_*.py`, `results/fx_bb_sma/`, `research_models/fx_bb_sma/` and the doc
entries.
