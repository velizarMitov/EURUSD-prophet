# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-session/*`.

The repo already provides these pieces:

- **MT5 access.** Symbol resolution lives in `src/pooled_h1_data.py::resolve_symbol`. The
  Berlin clock check is `src/forecast_eval/m15_data.py::verify_clock`. Bars are read from
  position 1 with a count of at most `maxbars − 1`, because the terminal's `maxbars` is
  100 000. That is the fx_fib finding.
- **fx_fib helpers.** `src/fx_fib/` is decided and frozen; its four DROPs are registered.
  It has the D1 aggregation (`mtf_aligner.build_d1`, reopen rule, midnight close), Wilder
  ATR, the circular block bootstrap (`score.circular_block_means`, `score.test_cell`)
  and the CUDA determinism helpers.
- **The fx_fib MLP is three-class.** Its output size is fixed at 3.
- **FOMC dates.** `results/fomc_dates.csv` lists the scheduled statement days from 1998
  to 2027.
- **AUDUSD.** It exists only in the pooled cache, from 2015-04-27, and without spread.
- **Session facts.** The horizon study (`results/horizon_study/README.md` §1b, §1c) shows
  coin-level accuracy for the 4 h and 6.5 h session horizons. It also shows a bid-only
  artifact at the 17:00 New York rollover.

## Goals / Non-Goals

**Goals:**
- One clean daily example per pair: features at 16:00 Sofia, target at 23:00 Sofia.
- The largest honest out-of-sample sample. History goes back to 2010, and evaluation
  starts in 2013.
- The fx_fib discipline: a frozen config, a digest guard, one-shot scoring and a
  reproducible report.

**Non-Goals:**
- Any hyper-parameter, feature, entry-hour or exit-hour search.
- Rates (the 2Y differential) and external intermarket data such as DXY, S&P or DAX.
  These are separate, later hypotheses.
- Costs, position sizing and execution.
- Any forward logger, live inference or UI.

## Decisions

### D1. Its own package, `src/fx_session/`, with read-only reuse

The family gets its own package, config (`fx_session_config.json`), registry and
results, like fx_fib.

- **Imported read-only:** `resolve_symbol`, `verify_clock`, fx_fib's `build_d1`, `atr`,
  `circular_block_means` and `test_cell`, and the determinism helpers.
- **Written fresh:** the MT5 loader and the MLP.
  - The fx_fib loader drops the spread column.
  - The fx_fib MLP is three-class, and changing either file would move the fx_fib
    digests.
- **A test pins fx_fib.** It checks that `src/fx_fib/fx_fib_config.json` still matches
  the digest recorded with fx_fib's verdicts.

*Alternative considered:* generalising the fx_fib modules. Rejected, because a decided
family's code would change after its verdicts.

### D2. The loader stores the spread and the point size (revised after the first refresh)

`fetch_h1` keeps MT5's `spread` field, in points. Each pair's `symbol_info.point` is
written to the refresh log.

The request starts from the newest closed bar and goes back `maxbars − 1` bars. That is
about 16 years, so roughly November 2010. To get the full 2010-01-01 start, raise
**Tools → Options → Charts → Max bars in chart** in the MT5 terminal.

The spec's stop rule applies only if a pair begins after 2011-01-01. Reaching 2010-11
passes, and the actual start is reported.

**The first real refresh (2026-11-21) changed the start date.**

- **History depth.** With `maxbars` 100 000, MT5 reached EURUSD 2010-10-21, GBPUSD
  2010-09-20 and AUDUSD 2010-10-21.
- **GBPUSD's clock was one hour off for a year.** From 2010-12 to 2011-11, GBPUSD's
  hourly returns correlate with EURUSD's at about 0.6 only after GBPUSD is shifted back
  one hour, and near 0 at lag 0. From 2011-12 on they correlate at lag 0. The pooled
  clock check still passes at 91 % (threshold 90 %), so a whole wrong year hides inside
  the threshold.
  - Owner decision: history starts on **2012-01-01** for all three pairs.
  - The shifted bars are dropped, never corrected.
  - Evaluation still starts on 2013-01-01, so the first fold trains on about a year.
- **GBPUSD March 2014 holds one bar per day.** It is a real broker gap, the same on
  every fetch. It is declared in `data.known_gaps`: it is cached as returned, left out
  of the clock check, and never filled.
  - Any other hole still refuses the refresh.
  - D1 bars built from fewer than `min_h1_per_d1 = 12` H1 bars count as missing, so
    that month cannot distort ATR.
- **Before 2017, DST-mismatch weeks open at 23:00 instead of 22:00.** In those weeks
  the labels were probably not on Berlin time.
  - Owner decision: a per-week gate. A day is eligible only if every loaded pair's
    weekly open for that week matches the Berlin rule.
  - This drops about 100 days in 2012–2016.

AUDUSD gets its own cache with spread. The pooled AUDUSD file is not reused: it starts in
2015 and has no spread.

### D3. One clock for the whole family: Berlin labels, with Sofia = Berlin + 1 h

- **Entry and exit are fixed label hours:** the bar labelled 14:00 for the entry and the
  bar labelled 21:00 for the exit.
- **No New York time conversion anywhere.** It matches invariant 6 and the owner's
  fixed 15:30–23:00 Sofia window.
- **The DST-mismatch flag uses `zoneinfo`.** It compares the UTC offsets of
  `America/New_York` and `Europe/Berlin` at noon on day D. It is 1 when the gap is not
  6 hours.

In the mismatch weeks two things move:
- The 08:30 ET release falls in the 13:00 bar, which belongs to the London window.
- The exit close lands at the 17:00 New York rollover.

The flag lets the model and the report see both. The mid-based target with the parity
check limits the exit effect.

*Alternative considered:* a fixed New York clock (09:00–16:00 ET). Rejected for these
reasons:
- the session would move inside the owner's window twice a year;
- a second clock would be needed;
- the label-hour arithmetic, which every cache relies on, would be lost.

### D4. Day eligibility is joint across the two scored pairs

- **A day is eligible only if both EURUSD and GBPUSD have every bar from 14:00 to 21:00.**
  The dollar factor also needs AUDUSD's pre-session bars.
- **So both pairs score on the identical day set.** This keeps the fold boundaries and
  the report rows aligned.
- **The cost is small.** Under 1 % of days are expected to be lost to a single pair's
  gap. The exact count is reported.

### D5. Fixed LightGBM settings for small samples

The first fold trains on about 500 days, and the last on about 4 000. The config fixes:

| Setting | Value |
|---|---|
| `objective` | binary |
| `num_leaves` | 7 |
| `min_data_in_leaf` | 50 |
| `learning_rate` | 0.02 |
| `feature_fraction` | 0.8 |
| `bagging_fraction` | 0.8, every bag |
| `lambda_l2` | 1.0 |
| `n_estimators` | 2 000, early stopping after 100 on the validation log-loss |
| seed | 42 |
| `deterministic` | true |
| `force_col_wise` | true |

There are no class weights, because the target is near 50/50.

*Alternative considered:* fx_fib's LightGBM settings. Rejected, because they were sized
for about 50 000 rows a fold.

### D6. Conditional MLP: 32-16, binary, 5 seeds, CUDA only

The config fixes the architecture now:

| Setting | Value |
|---|---|
| Layers | Linear → LayerNorm → GELU → Dropout 0.2 (×2: 32, 16), then a linear output to 1 logit |
| Loss | `BCEWithLogits` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-3 |
| Batch | 256 |
| Epochs | at most 300, early-stopping patience 20 |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 8 |
| Seeds | 11, 22, 33, 44, 55; the ensemble is their mean probability |
| Scaling | RobustScaler fit on the fit slice |

There is no CPU fallback. The fx_fib lesson is built in: there are no class weights, so
the probabilities stay calibrated to the base rate.

### D7. Two-stage pipeline and registry

```
 labels + power report --> stage 1: walk-forward LightGBM --> score lgbm (2 rows)
                                                                   |
                    +---------------- KEEP pairs only -------------+
                    v
           stage 2: walk-forward MLP --> score mlp (2 rows: verdict or NOT TRIGGERED)
```

- **Stage 2 needs stage 1's two registry rows.** It refuses to start without them.
- **A pair whose LightGBM cell is DROP gets no MLP training.** It gets a `NOT TRIGGERED`
  row instead.
- **Every registry row stores the config digest.** Each stage checks it.

### D8. A 5-day bootstrap block

The daily targets do not overlap. Model errors can still cluster in a regime week, so a
block of 5 eligible days is used. This is the same bootstrap code as fx_fib, with
`block = 5`.

### D9. The power report sits inside the label report

- **The label stage writes `labels.md` before any model exists.** It contains the counts
  and the minimum detectable accuracy:
  `0.5 + (z_{1−α} + z_{0.8}) · 0.5 / √n`, with α = 0.0125.
- **It also gives the matching log-loss improvement.** That is
  `ln 2 − H(0.5 + MDE)` in the same units, with H the binary entropy.
- **It never touches a model output.**

### D10. Calendar sources

- **FOMC** comes from `results/fomc_dates.csv`, read-only. Only scheduled statement days
  count.
- **NFP** is the first-Friday rule. It is a declared approximation, because the BLS
  occasionally publishes on the second Friday. It is not corrected by hand, to avoid
  outcome-aware edits.

## Risks / Trade-offs

- [The broker's history may not reach 2010] → D2's stop rule. The actual start goes into
  `labels.md`, and the MDE is recomputed from the real counts.
- [The MT5 bar `spread` field is not the spread at the close] → the parity check
  reports the bid-versus-mid gap per pair and fold. Verdicts use mid, and results are
  gross anyway.
- [About 3 450 days per pair can only detect accuracy of about 52.6 %] → the power
  report shows this before scoring. The horizon study's session cells sat near 49 %,
  so DROP is the expected outcome, and the report says so.
- [The NFP approximation is wrong on some months] → it stays a fixed, declared rule, and
  the mismatch is noted in the README.
- [Early folds train on about 500 days] → strong LightGBM regularisation (D5). Per-fold
  log-loss is reported, so early-fold noise is visible.
- [The fx_fib helpers could change later and silently change this family] → the config
  digest covers only this family. The digest test on fx_fib's config, plus the frozen
  fx_fib verdicts, make such a change visible.

## Migration Plan

Nothing to migrate. Rolling back means deleting `src/fx_session/`,
`tests/test_fx_session_*.py`, `results/fx_session/`, `research_models/fx_session/` and
the doc entries. No shared file changes.
