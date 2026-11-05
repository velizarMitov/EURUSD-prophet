# Design

## Context

See proposal.md, Why. What already exists on this PC:

- **Models.** `src.gold.challengers.make(name, cfg, overrides)` builds `GoldGBM` or
  `GoldLSTM` from a config whose `columns` list is only checked by
  `features.assert_scale_free`. The models read whatever matrix `Inputs.X` holds. So a
  column override plus a custom `Inputs` gives an exact copy with extra columns, without
  editing `src/gold`.
- **Tuning.** `GoldGBM.fit` tunes inside the training block over the declared grid
  (inner walk-forward, best AUC), so each yearly refit tunes itself on the past only.
- **Walk-forward.** `src/gold_window/walkforward.py` already has `year_positions`
  (it drops the last pre-year row, whose target is in the year), `prediction_years` and
  a CPU-safe `_gpu_lock`. It has also produced walk-forward `h1_lstm` predictions for
  2018–2026 with the registered configuration.
- **Data.**
  - H1 cache `results/gold/gold_h1.csv`, from 2015-01, bid OHLC on Berlin labels.
  - M15 cache `research_models/gold/analytics/bars_M15.csv.gz`, from 2022-08-12.
  - H1 spread `research_models/gold/window/mt5/h1_spread.csv.gz`, with the cost floor
    0.0062 % per round trip and the stress 0.0122 %.
  - The owner's window: as-of H1 labels 14–20, so the target bar is inside 15:30–23:00
    Sofia.

## Goals / Non-Goals

**Goals:**
- One feature module whose every column is causal, scale-free and needs no fit.
- Exact like-for-like pairs: with and without acceleration, same configuration, rows,
  seeds and years.
- A resumable run, since the LSTM takes about an hour on CPU.

**Non-Goals:**
- Size or volatility targets (the owner chose direction only).
- Any registration, forward logger, page or `api.py` route.
- Searching thresholds, holding times or feature subsets.

## Decisions

### D1. A separate package, `src/gold_accel/`

The package uses the gold challengers read-only, as `src/gold_window` does. It stays
outside `src/gold` for two reasons:
- its one cost check reads the spread, which gold's contract forbids;
- adding feature columns inside `src/gold` would risk the pinned round-one digests.

Its settings live in `accel_config.json`: features, models, walk-forward, stats, costs
and paths.

- **Alternative rejected:** shadow versions through `src/gold/versions.py`. Those are
  refits of the registered cells, would appear on the leaderboard, and assume the
  registered columns.

### D2. Features without a fitted profile

The move is `r_t = 100 · ln(close_t / close_{t−1})` on H1.

| feature | definition |
|---|---|
| `accel_1h` | \|r_t\| − \|r_{t−1}\| |
| `activity_rel` | \|r_t\| / mean \|r\| at the same label hour over the 60 previous trading days, current day excluded |
| `accel_expected_next` | the same trailing profile at hour(t + 1 h) minus at hour(t) |
| `accel_m15` | \|m₄\| − mean(\|m₁\|, \|m₂\|, \|m₃\|), where m₁…m₄ are the M15 log moves inside bar t; NaN unless all four exist |

The trailing profile is a rolling window over past days only. So there is nothing to fit,
and the inputs can be built once over the whole cache, as the walk-forward of
`gold-window-strategy` does.

- **Alternative rejected:** a profile fitted on each year's training block. It is equally
  honest, but it needs inputs rebuilt per year, and nothing gained justifies that.
- **Known limitation:** the profile runs on the Berlin label hour, so in the US/EU
  mismatch weeks a US release sits one hour earlier. `accel_1h` and `activity_rel`
  still see the realised move.

### D3. Models

| model | base | columns | years |
|---|---|---|---|
| `h1_gbm` | GoldGBM, registered config | the 15 H1 columns | 2018–2026 |
| `h1_gbm_accel` | the same | 15 + `accel_1h`, `activity_rel`, `accel_expected_next` | 2018–2026 |
| `h1_lstm` | GoldLSTM, registered config | 15 | 2018–2026 |
| `h1_lstm_accel` | the same | 15 + 3 | 2018–2026 |
| `h1_accel` | GoldGBM, registered grid | the 3 + `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | 2018–2026 |
| `h1_gbm_accel_m15` | GoldGBM | 15 + 3 + `accel_m15` | 2024–2026 |
| `h1_accel_m15` | GoldGBM | `h1_accel` + `accel_m15` | 2024–2026 |

Each model is `make(base, gold_cfg, {'columns': cols})` over an `Inputs` whose `X` holds
exactly those columns. The `eligible` rule is gold's (an H1 gap rule), and `complete`
drops rows with any NaN.

The three-year warm-up gives 2018 onward. The M15 models use rows from 2022-08 with one
year of warm-up, so their first prediction year is 2024.

**`h1_lstm` baseline reuse.** The `gold-window-strategy` predictions are reused when the
challenger configuration digest and the H1 input digest both match. Otherwise
`h1_lstm` is refit here. The report states which happened.

### D4. Walk-forward

`year_positions`, `prediction_years` and `_gpu_lock` are reused from `src.gold_window`.

- Each model and year writes `done.json` (ok or error, fit rows, last fit label,
  seconds) and its predictions file.
- A failed year stays failed.
- Run starts pin the H1 and M15 inputs as gzip copies under
  `research_models/gold/accel/inputs/`, with digests in
  `results/gold/accel/artifact_manifest.csv`. A rerun on the same pinned inputs gives the
  same rows.

### D5. Scoring

Every comparison uses the intersection of the rows both sides predicted that are
eligible and target-defined.

- **Hit rate:** (p_up > 0.5) == dir. A p_up of exactly 0.5 counts as a miss for both
  sides.
- **AUC and Brier score.**
- **Baselines:** the coin is 0.5, and the training-majority class comes from each year's
  fit rows.
- **Differences** use a paired moving-block bootstrap of the per-row hit difference:
  block 120 H1 rows (about one trading week), 2000 draws, seed 42, 95 %.
- **Views:** all hours, the owner's window (as-of labels 14–20), and per year.

### D6. The one cost check

For every as-of bar in the window:
- side = sign(p_up − 0.5);
- entry at the bar's close, exit at the next bar's close;
- cost = the spread paid (a long pays the entry spread, a short the exit spread), with
  the floor 0.0062 %, through `src.gold_window.simulate.trade_returns`.

The bootstrap of daily sums uses block 20 days. The stress cost 0.0122 % is reported
beside it. Nothing else is varied.

### D7. Report

`results/gold/accel/backtest.md` holds:
- the label in the header and in every section;
- the data digests and the run times;
- the tables of D5 and D6;
- a plain-language summary that never says "edge", "improves" or a trade word unless an
  interval excludes zero, and even then calls it descriptive.

## Risks / Trade-offs

- **[The owner saw the acceleration statistics first]** → The features are fixed in this
  design before any model result. The report is labelled descriptive, and nothing is
  tuned after it.
- **[Multiple models inflate a lucky win]** → Seven models over three views. The report
  lists every one, and says that one interval excluding zero among about twenty is
  expected by chance.
- **[LSTM runtime on CPU]** → The run is resumable per year and logged. The GBM models run
  first, so a partial run still reports them.
- **[The H1 cache grows hourly]** → The inputs are pinned at the run start.
- **[M15 years are few]** → The M15 models carry their own note. Their intervals will be
  wide, and the report says so.

## Migration Plan

1. Build features and tests, then run the walk-forward GBM models and then the LSTM.
2. Write the report, append the result to `results/gold/RESULTS.md`, and update the
   docs.
3. Rollback: delete `src/gold_accel/`, `results/gold/accel/` and
   `research_models/gold/accel/`. Nothing else depends on them.
