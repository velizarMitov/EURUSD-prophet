# Design

## Context

See proposal.md for the motivation. Three earlier families already hold the machinery:

- `src/fx_bb/` holds the events, the features, the ratio day-block bootstrap (`day_sums`, `ratio_block_means`, `test_ratio`) and the label and power maths.
- `src/fx_bb_sma/` holds the H1 first-touch exit scan.
- `src/fx_nyvol/` holds the OLS, LightGBM and CUDA MLP trainers and the walk-forward pattern.
- `src/fx_ny_orb/prereg.py` holds the git gate.

Gold has its own clock check (`src/gold/clock.py`) and a cache that holds only OHLC.

Measured before writing this, using no outcome: gold has 3 068 days with H1 bars between 2015-01-02 and 2026-11-20. There are 179 first band breaches (106 up, 73 down), 139 of them from 2018. The H1 hours 00–21 Berlin are present on about 3 000 days; 23:00 appears only in the DST-mismatch weeks.

## Goals / Non-Goals

**Goals:**
- Reproduce the three currency designs on gold, changing only what gold forces.
- Keep every outcome behind the committed pre-registration.
- Make the low power of Part 1 visible before any result.

**Non-Goals:**
- No cost model, no MT5 refresh, no indicator, no forward logger. A forward or chart step for Part 2 would be its own change, after a KEEP.
- No tuning or search. Every number is fixed in the config before labels run.
- No change to any existing family's code, config, cache or registry.

## Decisions

**D1. One package `src/gold_bbnr/`, three registries.**
- Parts 1a, 1b and 2 share the data layer and the gate. Their questions differ, so each gets its own registry and Bonferroni bar.
- The package lives outside `src/gold/`, so the never-edit rule on `gold_config*.json` and the gold versions machinery is untouched.
- Alternative considered: three packages. Rejected, because it would duplicate the data layer and the gate three times.

**D2. Import pure helpers; keep our own config.**
- The family imports the stateless functions: bootstrap, RSI, OLS, the LightGBM and MLP trainers, and first touch. It never imports or calls anything that writes another family's paths.
- Hyperparameters are copied into `gold_bbnr_config.json`. A test pins them equal to fx_bb's, fx_bb_sma's and fx_nyvol's blocks, so no knob was quietly tuned for gold.
- Alternative considered: copy all the code. Rejected, because the copies would drift.

**D3. D1 by New York 17:00 from Berlin labels.**
- Each H1 label is read as Berlin time, converted to New York time and shifted +7 h. The date of the result is the D1 day.
- This matches fx_bb exactly and puts the DST-mismatch 23:00 bar into the next day.
- Gold's daily break (17:00–18:00 New York) falls on this boundary, so no session is split.

**D4. Bid only, gross.**
- The cache whitelists OHLC, so there is no spread to build a mid price or a cost.
- Gross matches the gold family and is the owner's decision. Bid-on-bid differences cancel to first order in ranges and returns.

**D5. The cooldown applies only in Part 1a.**
- In 1a the cooldown is fx_bb's and the outcome window is 3 days.
- 1b follows fx_bb_sma, where the one-open-trade rule already prevents overlap.
- Both use the same events.

**D6. Thin folds are skipped, not merged.**
- With about 40 events before 2018, the early LightGBM folds would train on a handful of rows. A fold with fewer than 60 training rows predicts nothing.
- Which folds are thin depends only on event dates, so the label report counts them before any outcome.
- Alternative considered: one expanding fold starting later. Rejected, because it is still a choice of start date; the count rule is mechanical.

**D7. Part 2 cells are against HAR-X, not chained.**
- On currencies, HAR-X beat both non-linear models. So the questions are: does pre-session information help (HAR-X vs HAR), and does non-linearity add anything beyond it (LightGBM vs HAR-X, MLP vs HAR-X)?
- All three cells are independent at 0.05/3. No cell waits on another.

**D8. Folds for Part 2.**
- 6-month folds from 2018-01-01, with a 1-day embargo.
- Hourly DXY starts on 2017-01-03 and needs 20 days of warm-up, so the first training set is about 220 days. That gives about 2 250 scored days.
- All four models use the same rows: the days with all fourteen features.

**D9. The dollar index (owner request), from two sources.**
- **Part 2 uses MT5's `USDIndDec26` H1 bars.** They replace fx_nyvol's three-pair ratio with the DXY pre-session range ratio.
  - Checked before writing this, with no gold outcome: real H1 bars start on 2017-01-03 (2014–2016 hold only daily bars).
  - The returns are at lag 0 against EURUSD, with correlation −0.91 to −0.95 every year.
  - The largest residual moves against EURUSD (up to 1.1 %) are Monday 02:00 weekend gaps, not contract rolls.
  - There is one 10-day hole in June 2024.
  - The spread is recorded only from 2024, so validity is by bar count (at least 10 bars between 00:00 and 14:00), not by spread.
- **Part 1 uses Yahoo's daily `DX-Y.NYB`.**
  - Its history covers 2015, so no Bollinger event is lost.
  - Only closes dated before gold day `t` are used. A Yahoo date is a New York session, and its settlement time is not pinned to our 17:00 close, so same-day use would risk look-ahead.
  - The two features are signed by the gold event's direction, so a model can read "dollar stretched the same way" without needing the side.
- **Both caches are fetched once, before the pre-registration.**
  - Their digests are frozen in it, and gated stages refuse a changed cache.
  - The MT5 contract `USDIndDec26` expires on 2026-12-11, so a refetch after that date could not reproduce the cache. Pinning prevents silent drift.
  - The MT5 fetch takes the terminal path as an argument: the series exists in the logged-in `mt5app` ActivTrades terminal, not in every local terminal.
- **Alternative considered: keep EURUSD.** It has a longer history, but the owner asked for the dollar index, and DXY measures the dollar directly. The 2015–2016 loss leaves Part 2 well powered.

**D10. The gate reuses fx_ny_orb's pattern.**
- `git ls-files --error-unmatch`, then `git diff --quiet HEAD`, then the config-digest match, before any outcome, target of a scored day, or verdict.
- The labels and the pre-registration build stage are ungated.

**D11. Stages and CLI.**
- `fetch_dxy --mt5-path <terminal64.exe>` and `fetch_dxy --yahoo`, run once each.
- `labels`, then `prereg --write`; the owner commits.
- `walk_forward --part bb`, `walk_forward --part bb_sma` and `walk_forward --part nyvol`, gated.
- `score --family bb|bb_sma|nyvol`, once each, gated.
- `frozen` (Part 2 refit), gated.
- `report`.
- Predictions are written to `results/gold_bbnr/*.csv.gz`, and frozen models to `research_models/gold_bbnr/`.

## Risks / Trade-offs

- [About 170 events give little power for Part 1] → The minimum detectable effect is printed before any outcome; a DROP will say "underpowered" in the report, not "no effect".
- [Gold's H1 history was already used by the gold family's validation and test blocks] → Recorded as a fact in the pre-registration. The targets here (band events, NY session range) were never scored there.
- [The currency results inspired the cell choice (HAR-X)] → That is the intended use of a prior; it is named in the pre-registration.
- [Bid-only highs and lows can be exaggerated by spread spikes at the 23:00 rollover] → The rollover is outside both the Part 2 window and the D1 close. Part 1b's H1 scan can cross the rollover, which is recorded as a limitation.
- [The MT5 DXY series may be a stitched forward contract, so a roll could create a step] → No step is visible in the residuals against EURUSD. A day whose open-to-previous-close gap exceeds 1 % is listed in the label report, which reads no gold outcome.
- [Yahoo's daily series could later be revised] → The cache is fetched once and pinned by digest.
- [Without CUDA, the MLP cannot run] → The stage refuses rather than silently using the CPU; the other cells still run.

## Migration Plan

New files only; there is nothing to migrate. Rollback means deleting `src/gold_bbnr/`, `results/gold_bbnr/`, `research_models/gold_bbnr/` and the tests.
