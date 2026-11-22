# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-ny-volatility/*`.

**Facts carried over from fx-session-ny.** Its `results/fx_session/labels.md`, generated
2026-11-21, gives these counts. EURUSD and GBPUSD each have 3 875 weekdays from 2012 and
about 3 690 usable days, of which about 3 470 fall from 2013-01-01.

| Days excluded per pair | Count |
|---|---|
| Week with an unverified clock (pre-2017 DST-mismatch weeks) | 116 |
| Incomplete session | 42 |
| Warm-up and previous-day gaps | about 15 |

This family's eligibility differs in two ways:
- the HAR inputs need 20 earlier eligible days;
- a day needs a usable range, not a non-zero move, so there are no ties.

Expect about the same counts.

**Facts from the fx-bb-reversion probe (2026-11-21):**
- AUDUSD's cache has 19 bars on 2014-01-07 whose spread reads 300 000 points.
- The largest EURUSD spread is 90 points and the largest GBPUSD spread is 162.
- fx_session's features read only bid prices, so its verdicts are unaffected.
- The median spread by Berlin label hour is 5 points from 20:00 to 22:00 and 8 at 23:00.
  The window here stops at the bar labelled 21:00, which closes at 22:00 Berlin, before
  the rollover.

**Already known in this repo:**
- Range and volatility are predictable from their own history.
- A model with today's London range will almost certainly beat a constant.

The registered reference is therefore HAR, as the owner specified.

## Goals / Non-Goals

**Goals:**
- An honest test of whether pre-session information plus a non-linear model beats HAR on
  the 16:00–23:00 Sofia range.
- The fx-session-ny discipline: a frozen config, a digest guard, one-shot scoring, a
  conditional MLP, a power report before models, and a reproducible report.

**Non-Goals:**
- Direction, entries, stops, sizing rules or execution.
- Any search over windows, features or targets. The target is `log(range / close14)`, and
  the alternative ratio-to-20-day target is not tested.
- A forward logger, live inference or UI.
- Any change to the production volatility family.

## Decisions

### D1. Its own package, `src/fx_nyvol/`, with read-only reuse

The family gets `fx_nyvol_config.json`, `results/fx_nyvol/fx_nyvol_hypothesis_log.csv`,
`results/fx_nyvol/` and `research_models/fx_nyvol/`.

- **Imported read-only:**
  - from `fx_session.data_loader`: `load_h1` and `load_points`;
  - from `fx_session.session`: `session_calendar`, `verified_weeks` and `mid`;
  - from `fx_session.features`: `hour_grid`, `_window`, `_ratio_to_mean` and
    `calendar_block`;
  - `fx_fib.score.test_cell`.
- **Written new:**
  - the regression trainers: LightGBM with the `l2` objective, and an MSE MLP;
  - the HAR OLS;
  - a family registry.

  fx_session's model helpers are binary, BCE and log-loss, so they are not reused for
  training. The MLP copies their architecture and their CUDA determinism pattern.
- **A test pins the earlier families.** fx_fib, fx_session and fx_rates must still match
  the config digests recorded with their verdicts.

### D2. Eligibility

The fx_session config's session hours are 14–21. A day is eligible when all of the
following hold:
1. fx_session's `session_calendar` marks it eligible. That means every bar labelled
   14:00–21:00 exists for both scored pairs, and the week's weekly open is on the Berlin
   clock for every loaded pair.
2. Its pre-session windows are complete: bars 0–8, 9–13 and 14.
3. No bar it uses carries a corrupt spread, meaning more than 50 pips. That covers bars
   0–21 of the scored pair, and bars 0–14 of each pair in the cross-pair feature.
4. Its HAR inputs and 20-day ratios have full windows.

Each exclusion has a reason: `incomplete session`, `unverified clock week`,
`missing pre-session bar`, `corrupt spread` or `warm-up`. `labels.md` counts them.

### D3. Target

For pair `p` and day `d`, take the bars labelled 15:00–21:00 Berlin:
- `mid_high = max(high + spread·point/2)`;
- `mid_low = min(low + spread·point/2)`;
- `close14` is the mid close of the bar labelled 14:00.

The target is `y = log((mid_high − mid_low) / close14)`.

- **The window is fixed in Berlin labels.** Sofia is Berlin + 1 hour all year, so it is
  always 16:00–23:00 Sofia.
- **The data hour moves.** In DST-mismatch weeks the 08:30 New York release falls in the
  bar labelled 13:00, and `dst_mismatch` marks those days.
- **A bid twin**, `y_bid`, is built from bid highs and lows. It is reported for parity
  only: their correlation and the cells' MSE ratio on `y_bid`. It is never registered.

### D4. The 14 features

Every feature is known at the close of the bar labelled 14:00 Berlin. Ranges are bid
`max(high) − min(low)` over complete windows.

| Column | Definition |
|---|---|
| `log_ny_range_lag1` | `y` on the previous eligible day |
| `log_ny_range_ma5` | mean of `y` over the previous 5 eligible days |
| `log_ny_range_ma20` | mean of `y` over the previous 20 eligible days |
| `log_asia_range` | `log(range(bars 0–8) / close14_bid)` |
| `log_london_range` | `log(range(bars 9–13) / close14_bid)` |
| `log_data_hour_range` | `log(range(bar 14) / close14_bid)` |
| `abs_data_hour_return` | `abs(close14 − open14) / close14`, bid |
| `london_range_ratio` | London range / mean of the previous 20 days that had one (fx_session's definition) |
| `data_hour_range_ratio` | the same for bar 14 |
| `cross_pre_range_ratio` | mean over EURUSD, GBPUSD and AUDUSD of `range(bars 0–14)` divided by its previous-20-day mean |
| `day_of_week`, `is_nfp_day`, `is_fomc_day`, `dst_mismatch` | fx_session's `calendar_block` |

- **"Eligible" in the HAR lags** means the day had a target. The HAR lags therefore never
  include today.
- **A data-hour range of zero** is floored at one point before the log, and is counted.
  No other range can be zero across 5 or 9 bars.

### D5. Folds

- **Folds.** 6-month folds from 2013-01-01 to the last eligible day, with shared
  boundaries for both pairs, expanding.
- **Training.** The eligible days before the fold, minus a 1-day embargo.
- **Validation.** The last 15 % of the training days, minus a 1-day embargo before it, is
  held out for early stopping.

Only the HAR lags reach into the past, and none reaches forward. The target is the same
day, so nothing overlaps.

### D6. Models

**HAR.**
- An OLS of `y` on `[1, lag1, ma5, ma20]`.
- It is fitted per pair per fold on all training days (fit plus validation), because it
  has no hyper-parameter.
- It is closed-form and deterministic.

**LightGBM.**

| Setting | Value |
|---|---|
| Objective | `l2` |
| Leaves | 7 |
| Rows per leaf, minimum | 50 |
| Learning rate | 0.02 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| L2 | 1.0 |
| Trees | up to 2 000, early stopping after 100 on validation L2 |
| Seed | 42, deterministic |

There is one model per pair. The settings are fx_session's, with only the objective
changed.

**The MLP.**
- `32-16`, LayerNorm and GELU, dropout 0.2, and a linear output.
- MSE on `y`, standardised by the fit slice's mean and standard deviation, with features
  standardised the same way.
- AdamW 1e-3 with weight decay 1e-3, batch 256, patience 20, plateau schedule.
- The prediction is the mean of 5 seeds, de-standardised.
- CUDA only. It is trained only for a pair whose LightGBM cell is KEEP.

### D7. Scoring

- **The per-day difference** is the squared error of the model minus the squared error
  of the reference. The reference is HAR for cells 1–2 and LightGBM for cells 3–4.
- **The test** is `fx_fib.score.test_cell` with block 5, 10 000 resamples, a fixed seed and
  alpha 0.0125. It is KEEP when the upper bound is below 0.
- **The pipeline has two stages.**
  - Stage 1 runs `walk_forward --stage lgbm` (HAR and LightGBM), then
    `score --stage lgbm`.
  - Stage 2 runs `walk_forward --stage mlp`, then `score --stage mlp`. It refuses before
    both LightGBM rows exist. It trains only the KEEP pairs and records NOT TRIGGERED
    rows without training.
- **What the registry records:**
  - the days and both MSEs;
  - the mean difference, its bound and `mde_80`;
  - the verdict and the digests;
  - the relative MSE reduction;
  - a note with the gross statement and the earlier fx families, including fx-session-ny's
    direction DROP.

### D8. Descriptive report

The report gives, per pair:
- **Accuracy of the forecast:**
  - MSE and MAE;
  - out-of-sample R², as `1 − MSE / var(y_oos)`;
  - the MSE ratio to HAR;
  - Spearman's ρ.
- **Hot sessions.** The threshold is the causal Q75: the 75th percentile of `y` over the
  previous 60 eligible days. The report gives precision, recall and accuracy for calling
  a day hot.
- **Breakdowns.** MSE per year and per fold, and on NFP, FOMC and DST-mismatch days.
- **The HAR-X line.** The same OLS fitted with all 14 features. It is descriptive, and it
  separates "pre-session information helps" from "a non-linear model helps".
- **The bid parity line.**

A test checks that two builds differ only in the generation line.

## Risks / Trade-offs

- **[A KEEP against HAR may mostly reflect today's London range, which a linear model
  also captures]** → the cells stay as the owner registered them. The HAR-X line shows
  this openly, and the MLP cells test only non-linear gain over LightGBM. Making HAR-X the
  registered reference would be a different hypothesis, and has to be chosen before any
  run.
- **[Mid highs use each bar's own spread, which may be the bar's minimum and not the spread
  at the high]** → the bid twin is reported. A range is a difference, so a constant offset
  cancels.
- **[About 116 days per pair are lost to unverified clock weeks]** → the owner's rule from
  fx-session-ny is reused unchanged, so the two families share a calendar.
- **[An effect of 0.05 σ of the loss difference is detectable]** → this is small. A KEEP
  can be practically tiny, so the report gives the relative MSE reduction next to the
  verdict.
- **[The fx_session caches are refreshed between stages]** → the run records their
  digests, and scoring uses only the saved predictions.

## Migration Plan

Nothing to migrate. Rolling back means deleting `src/fx_nyvol/`, `tests/test_fx_nyvol_*.py`,
`results/fx_nyvol/`, `research_models/fx_nyvol/` and the doc entries.
