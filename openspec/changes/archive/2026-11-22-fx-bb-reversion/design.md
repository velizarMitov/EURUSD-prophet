# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-bb-reversion/*`.
A read-only probe of 2026-11-21 established the facts below. Its scripts are kept outside
the repo, and it wrote nothing. It counted events only and never looked at an event's
outcome.

**MT5 coverage.** USDJPY, USDCAD and USDCHF resolve to their own names, with H1 history
from October 2010 to 2026-11-20.
- **USDJPY:** point 0.001, no holes.
- **USDCAD:** one interior hole, 2013-05-03 22:00 → 2013-05-13 14:00 (232 h).
- **USDCHF:** one interior hole, 2015-01-15 19:00 → 2015-01-18 23:00 (76 h), the SNB
  floor removal.
- **Clock check:** every pair matches the Berlin weekly-open clock on 96.4 % of 790 weeks.

**The lag check found a shifted clock.** The probe correlated each pair's hourly returns
with EURUSD's at lags −1, 0 and +1, per calendar year and per month.
- **USDCHF 2012-01 … 2014-11:** the correlation peaks at lag +1, at 0.76–0.95, so the bars
  are one hour off. December 2014 peaks at lag 0 (0.95), and 2015-02 onward is clean.
  The 96 % weekly-open check does not catch this.
- **Every other pair** peaks at lag 0 in every year from 2012.

**A corrupt spread field.** The fx_session AUDUSD cache has 19 bars on 2014-01-07 whose
spread reads 300 000 points (3.0 in price).
- Every other spread is far smaller. The maximum is 90 points (EURUSD) or 162 (GBPUSD),
  and the 99.9th percentile is at most 100 points.
- fx_session used only AUDUSD's bid prices, for its dollar factor, so its verdicts are
  unaffected. Any mid price here would be.

**The day boundary.** MT5's own EURUSD D1 bars include a Sunday bar every week (66 of
400). That bar is the one-hour reopen labelled 23:00 Berlin.
- Median EURUSD spread by Berlin label hour is 5 points at 20:00–22:00, 8 at 23:00 and 5
  at 00:00. For USDJPY it is 5, 5, 6, 9 and 5.
- The bar labelled 23:00 Berlin is 17:00–18:00 New York, the rollover hour. A D1 close at
  Berlin midnight is therefore the close of the rollover hour, which is what the
  existing D1 builders produce.

**Events.** These counts use D1 bars that close at 17:00 New York, with the mid close and
at least 12 H1 bars per day, from 2012. The cooldown is 3 trading days.

| | EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD | USDCHF |
|---|---|---|---|---|---|---|
| Days | 3 861 | 3 840 | 3 861 | 3 861 | 3 855 | 3 860 |
| Closes outside the bands | 469 | 469 | 470 | 438 | 449 | 447 |
| First breaches | 241 | 241 | 239 | 240 | 227 | 233 |
| Events after the cooldown | 224 | 221 | 222 | 215 | 214 | 212 |
| Events from 2015 | 186 | 179 | 172 | 173 | 175 | 169 |

**Totals.** There are 1 308 events in all and 1 054 from 2015, on 751 distinct dates.
Events per date from 2015:

| Events on the date | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Dates | 540 | 136 | 63 | 8 | 3 | 1 |

**Dispersion.** The 3-day mid return divided by ATR(20), over all days (not events), has
a standard deviation of 1.12–1.24 for five pairs. AUDUSD reads 4.4, because of the
corrupt spread day. This dispersion is the input to the power calculation.

## Goals / Non-Goals

**Goals:**
- An honest, one-shot test of Rockefeller's Bollinger reversion rule as published.
- A second, separate question: whether a small pooled model can tell the events that
  revert from the ones that do not.
- The discipline of the earlier families:
  - a frozen config and a digest guard;
  - one-shot scoring;
  - a power report before any model;
  - a reproducible report.

**Non-Goals:**
- Searching over the band length, width, horizon or cooldown, or over features. The
  values are 20, 2, 3 and 3.
- An MLP. It could only be a separate later family, and only behind a Cell 2 KEEP.
- Stops, targets, sizing or execution. Costs never enter a verdict.
- A forward logger, live inference or UI.

## Decisions

### D1. Its own package, `src/fx_bb/`, with read-only reuse

The family gets `fx_bb_config.json`, `results/fx_bb/fx_bb_hypothesis_log.csv`,
`results/fx_bb/` and `research_models/fx_bb/`.

- **Imported read-only:**
  - from `fx_session.data_loader`: `Mt5Source`, `load_h1`, `load_points`,
    `coverage_holes` and `verify_clock_segments`;
  - `pooled_h1_data.resolve_symbol`;
  - `fx_fib.score.circular_block_means`, used for a self-check of the new ratio bootstrap;
  - `fx_session.model_lgbm`.
- **Written new:**
  - a family registry module;
  - a cache refresh that allows a start date per pair, which fx_session's refresh cannot
    do;
  - the ratio bootstrap of D8.
- **A test pins the earlier families.** fx_fib, fx_session and fx_rates must still match
  the config digests recorded with their verdicts.

### D2. H1 data, per-pair starts, gaps and corrupt spreads

**Sources.**
- **EURUSD, GBPUSD, AUDUSD** are read from `results/fx_session/data/`. They are never
  refreshed here.
- **USDJPY, USDCAD, USDCHF** go into `results/fx_bb/data/<PAIR>_h1.csv`, with the columns
  time, OHLC (bid) and spread (points).
  - The caches are append-only, and the refresh is all-or-nothing.
  - Point sizes go to `symbols.json`, and every refresh writes `refresh_log.csv`.

**Checks on every refresh.**
- **Coverage.** Holes are refused unless their month is declared. USDCAD `2013-05` is
  declared.
- **Clock.** The weekly-open clock gate must be at least 90 %, run on the segments between
  declared gaps.
- **Lag check (new).** Within each pair's history, and for each calendar year, the
  absolute correlation of the pair's hourly log returns with EURUSD's must peak at lag 0
  among −1, 0 and +1. A failing year refuses the refresh. On load, the three fx_session
  pairs get the same check, and a failure there refuses the run.

**Start dates.**
- **Every pair except USDCHF** starts on 2012-01-01. GBPUSD's shifted 2011 falls before
  that.
- **USDCHF** starts on 2015-02-01. This date clears two problems:
  - the shifted 2012-01 … 2014-11 clock;
  - the SNB month, which has a 76-hour hole. Until 2015-01-15 the SNB floor also pinned
    USDCHF to EURUSD, so its events would mostly duplicate EURUSD's.

  This follows the owner's precedent for GBPUSD 2011: start after a proven clock defect,
  never shift the labels. Its first events come after the band warm-up, in March 2015.

**Corrupt spread.** A bar whose spread exceeds `max_spread_pips` is treated as corrupt.
That limit is 50 pips: 500 points on 5-digit pairs and 50 on USDJPY.
- Any mid price that needs that bar is missing, and its day counts as missing.
- Bid prices are unaffected.

### D3. D1 bars that close at 17:00 New York

Each H1 label is the Berlin wall clock, so it is localised to `Europe/Berlin` and converted
to `America/New_York`. Its trading date is the New York time plus 7 hours, truncated to the
date.

- **What this gives.** The day runs from 17:00 New York on the previous day to 17:00 New
  York. Its last bar is the one labelled 22:00 Berlin, or 21:00 in the US/EU DST-mismatch
  weeks.
- **The Sunday reopen.** The 23:00 Berlin bar (22:00 in mismatch weeks) is 17:00 New York,
  so it opens Monday's day. No Sunday day exists.
- **Ambiguous labels** fall only on weekend nights, when there are no bars. One still
  raises.
- **What each D1 bar holds:**
  - open, high and low from the bid;
  - `close_bid`;
  - `close`, the mid: the last bar's bid close plus half its spread;
  - `n_h1`.
- **Which days are dropped.** A Monday–Friday day with fewer than 12 H1 bars is missing.
  This covers holidays, Friday bars after 17:00 New York in mismatch weeks, and thin
  months. A day whose closing mid uses a corrupt spread is also missing.
- **Why.** It is the conventional FX daily close. It also avoids both the Sunday bar and
  the rollover hour in the closing mid.
- **The per-week clock gate is not applied**, as in fx_rates D2. A one-hour label error
  moves one daily close by an hour within a 3-day horizon. Excluding off-clock weeks
  would void every event whose 23-day window touches one, about 3.5 % of weeks. The lag
  check and the pooled weekly-open gate still apply.

### D4. Bands, ATR, events and outcomes

All windows run over the pair's own valid D1 days.

**Indicators.**
- **Bands.** `SMA20` and `SD20` (population, `ddof = 0`) of the mid closes over
  `t−19 … t`. The bands are `SMA20 ± 2·SD20`.
- **ATR20.** The simple mean of the true range over `t−19 … t`. The true range uses the bid
  high, the bid low and the previous bid close.

**Events.**
- **Breach.** `close[t] > upper[t]` gives direction −1, and `close[t] < lower[t]` gives +1.
- **First breach.** `close[t−1]` was inside the bands of `t−1`.
- **Cooldown.** An event on a pair is accepted only if at least 3 valid days have passed
  since that pair's last accepted event. Same-pair horizons therefore never overlap.

**Outcome.** `r = dir · (close[t+3] − close[t]) / ATR20[t]`, on mid closes, with `t+3` the
third valid day after `t`. The binary target is `1` if `r > 0` and `0` if `r < 0`. Ties are
dropped and counted.

**Pair gaps.**
- **The reference calendar** is every date on which at least 4 of the pairs that have
  started have a valid D1 bar.
- **Exclusion.** An event is excluded as a pair gap if the pair misses any reference date
  between the date of `t−19` and the date of `t+3`.

  This catches GBPUSD 2014-03, USDCAD 2013-05 and AUDUSD 2014-01-07 without filling
  anything. Holidays common to all pairs are not reference dates.

### D5. The ten features of Cell 2

Each feature is known at the close of `t`. Each uses only that pair's D1 bars up to `t`,
plus the other pairs' closes on `t`, since all pairs close together at 17:00 New York.

The breach side `s` is `−dir`: +1 for a breach of the upper band and −1 for the lower.
Directional features are multiplied by `s`, so pooled long and short events share one
meaning.

| Column | Definition |
|---|---|
| `bb_zscore_abs` | `abs(close − SMA20) / SD20` |
| `bb_width_rank_60d` | the fraction of `t−59 … t` whose band width `4·SD20/SMA20` is at most today's |
| `rsi_14_dist` | `s · (RSI14 − 50)`, Wilder RSI on mid closes |
| `bar_range_vs_atr20` | `(high − low) / ATR20` |
| `close_pos_in_bar` | the bid close's position in the day's range, toward the breach side: 1 means it closed at the extreme |
| `ret_3d_atr` | `s · (close[t] − close[t−3]) / ATR20` |
| `trend_ma50_alignment` | `s · (close − SMA50) / ATR20`: positive when the breach is on the side the close already sits on relative to SMA50 |
| `atr_ratio_20_60` | `ATR20 / ATR60` |
| `simultaneous_frac` | the share of the other started pairs whose close on `t` is outside their own bands |
| `breach_side` | `s` |

- **Warm-up.** `SMA50`, `ATR60` and the 60-day width rank need about 80 valid days. Events
  without full features are excluded from Cell 2 only. Cell 1 needs only the bands and
  ATR20.
- **No pair identity is a feature.** This keeps the model pooled.

### D6. Cell 1 uses every event; Cell 2 is walk-forward from 2015

**Cell 1** has no fitted parameter, so it is scored on every eligible event: from the first
complete bands in 2012 (USDCHF from 2015) to the last event whose `t+3` exists.

- The window is fixed now, before any outcome is computed.
- The probe found about 1 260 events once USDCHF's pre-2015 events are removed.

**Cell 2** uses expanding 12-month folds from 2015-01-01.
- **Fit.** All events whose date of `t+3` is more than 10 business days before the fold
  start: a 5-day purge plus a 5-day embargo.
- **Validation.** The last 15 % of the fit events by date is held out for early stopping,
  with the same 10-day gap before it.
- **Test.** The fold's events.
- **The prior** is the up-rate of the fit slice, as in fx_session.

### D7. LightGBM settings for small samples

The first fold trains on about 200 events from 2012–2014.

| Setting | Value |
|---|---|
| Objective | binary, no class weights |
| Leaves | 4 |
| Rows per leaf, minimum | 20 |
| Learning rate | 0.02 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| L2 | 1.0 |
| Trees | up to 2 000, early stopping after 100 |
| Seed | 42, deterministic |

These are fixed now, not tuned.

### D8. Scoring

Both cells resample whole days, so events on the same date and on neighbouring dates stay
together. Per business day `d` (Monday–Friday) from the first to the last scored event:

- **Cell 1** uses `S_d = Σ r` and `N_d =` the number of events dated `d`. The statistic is
  the event mean, `ΣS / ΣN`.
- **Cell 2** uses `S_d = Σ (logloss_lgbm − logloss_prior)` over the out-of-sample events
  dated `d`.

**The bootstrap** is circular, over 5-business-day blocks, with 10 000 resamples and a fixed
seed. It draws the same blocks for `S` and `N`, and recomputes `ΣS* / ΣN*`.
- **Cell 1 is KEEP** if the alpha-quantile of the bootstrap means is above 0.
- **Cell 2 is KEEP** if the (1 − alpha)-quantile is below 0.
- **alpha = 0.025 each** (0.05/2). Each cell is scored once, with file mode `'x'`.

**A self-check pins the ratio bootstrap.** With `N ≡ 1` it must reproduce
`fx_fib.score.circular_block_means` exactly for the same seed.

**What the registry records:**
- the event and date counts;
- the statistic and the bound;
- the effective size, as the number of distinct dates;
- for Cell 1, the same statistic on bid closes, with bands and returns from bid, and a
  parity flag when the two differ by more than 0.02 ATR. This follows the mid-versus-bid
  practice of invariant 6. The flag is recorded, but it does not change the verdict.

### D9. Descriptive statistics, gross and after spread

**Units.**
- **Pip size:** 0.0001, or 0.01 for USDJPY.
- **The pair's mean spread** is the mean spread, in pips, of the D1 closing bar over the
  scored period. It is charged once per event, a full round trip from mid to mid.

**The report gives:**
- **For Cell 1:**
  - per pair, pooled, per year and per breach side: events, win rate with Wilson 95 % on
    the distinct dates, mean `r`, and mean and total pips;
  - profit factor and mean pips, gross and after spread;
  - the mid-versus-bid parity line.
- **For Cell 2:** the out-of-sample comparison of all events against the events with
  P(win) > 0.5, using the same statistics, plus log-loss and Brier per fold and a
  reliability table.

All of it is descriptive. Only the two registry rows are verdicts.

## Risks / Trade-offs

- **[The detectable effect is about 0.1 ATR (Cell 1) and about 55 % (Cell 2)]** → stated in
  `labels.md` before any model. A DROP means "no edge of that size".
- **[USD shocks fire several pairs on one day]** → day blocks keep them together. The
  effective size is reported as distinct dates (about 750 from 2015), not events.
- **[Excluding USDCHF before 2015 removes about 40 events]** → those bars are provably on
  the wrong clock. Most fell in the SNB-floor era, when USDCHF moved almost one-for-one
  with EURUSD.
- **[Another broker defect not yet seen]** → the lag check and the corrupt-spread rule run
  on every load. A new failure refuses the run instead of being absorbed.
- **[The D1 definition differs from fx_rates (Berlin midnight)]** → intentional, and the
  design records the spread evidence. fx_rates' results are not touched.
- **[MT5 history can be revised]** → the caches are append-only, overlap mismatches are
  logged, and the run records the cache digests.

## Migration Plan

Nothing to migrate. Rolling back means deleting `src/fx_bb/`, `tests/test_fx_bb_*.py`,
`results/fx_bb/`, `research_models/fx_bb/` and the doc entries.
