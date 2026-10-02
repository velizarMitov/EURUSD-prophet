# Horizon study `hs-20261001`: results

> **DESCRIPTIVE. Not a verdict.** These curves describe the **challenger**
> models built by `src/forecast_eval/`, not the production models in `models/`.
> They were measured on development data that includes the spent test blocks,
> so they can rank and explain, but they cannot prove anything. The only arbiter
> for a horizon claim is the forward data (`results/forward_eval/`). No
> hypothesis log was written and no alpha was spent.

## What was run

- **Record:** [`study_record.json`](study_record.json). It fixed the horizon grid,
  every configuration, the tuning rule and the 3-year cap, and was written
  before the first fit.
- **Grid:** H1 cadence at {1, 2, 4, 6, 12, 24, 48, 120} bars. Daily cadence at
  {1, 2, 5} days.
- **Cells:** 38. That is 34 direction cells (10 challengers), the volatility
  ensemble at {1, 2, 5}, and Kronos volatility at 24 bars. **0 failures.**
- **Data:**
  - daily: 1999-01-04 → 2026-08-10, 8,606 euro-era rows; macro from FRED
    (public CSV, on a sandboxed copy of the caches);
  - H1: 2017-01-20 → 2026-09-30, 60,312 bars;
  - Kronos: scored only after its 2024-06 training cutoff, on every 12th bar.
- **Method:**
  - an expanding purged walk-forward (5 folds, test on the later half);
  - CPCV (N = 6, k = 2; N = 4 for the volatility ensemble);
  - nested-CV tuning for the GBM cells under the declared rule.
- **Costs:** the measured median spread of 0.5 pip is the primary cost. The
  1.5-pip round trip in `config.json` is the sensitivity case. Breakeven
  accuracy = 0.5 + cost / (2 · E|move over h|).
- **Compute:** 111 minutes of fitting across all cells, plus about 108 minutes of
  Kronos sampling on the GPU.

## What it says

### 1. The H1 GBM has real directional skill. It fades with the horizon and meets the cost at 4–12 h.

| h (bars) | accuracy | 95 % CI | breakeven 0.5 pip | breakeven 1.5 pip | net / trade 0.5 pip | net / trade 1.5 pip | CPCV median net | DSR |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 52.66 % | 52.13–53.22 | 53.69 % | 61.06 % | +0.19 pip | −0.81 | +0.12 | 0.000 |
| 2 | 52.25 % | 51.64–52.90 | 52.59 % | 57.76 % | +0.36 | −0.64 | +0.37 | 0.001 |
| **4** | **52.01 %** | **51.23–52.79** | **51.80 %** | 55.40 % | +0.64 | −0.36 | +0.86 | 0.019 |
| 6 | 51.24 % | 50.37–52.13 | 51.44 % | 54.33 % | −0.63 | −1.63 | +1.06 | 0.000 |
| 12 | 51.47 % | 50.32–52.57 | 50.99 % | 52.96 % | +2.03 | +1.03 | +3.44 | 0.259 |
| 24 | 50.82 % | 49.48–52.18 | 50.67 % | 52.02 % | +0.43 | −0.57 | +1.92 | 0.110 |
| 48 | 50.31 % | 48.51–52.08 | 50.47 % | 51.41 % | −1.24 | −2.24 | +3.90 | 0.121 |
| 120 | 47.95 % | 45.83–50.10 | 50.30 % | 50.90 % | +4.77 | +3.77 | −5.87 | 0.379 |

About 30,000 scored bars per row. The Pesaran–Timmermann p-value is below 0.001
for every h ≤ 24.

- **Accuracy falls as the horizon grows. The breakeven falls faster at first.**
  At 1 h the model is about 1 pp short of the breakeven. Around **4 h** the two
  cross. Up to 12 h the point estimates stay near or above the breakeven.
- **Nowhere is the accuracy significantly above the breakeven.** At h = 4 the
  lower end of the interval is 51.23 %, below the 51.80 % breakeven. After
  deflation for the trials logged so far (52 by the end of the run), every
  Deflated Sharpe ratio is at most 0.26.
- **At the 1.5-pip round trip, every horizon except 12 h loses money.** The live
  spread was 0.5–0.8 pip when sampled, and execution adds slippage. The edge is
  of the same size as the cost.
- **PBO across horizons is 0.16.** Choosing the best horizon of this model in
  sample is unlikely to be pure luck: the *shape* of the curve is stable, even
  though no single point clears its cost with confidence.

### 2. The daily models are coin flips at every horizon

- Every daily interval contains 50 %. The one exception is `daily_gbm_price` at
  h = 1 (50.03–52.90 %), whose Deflated Sharpe is 0.007.
- The CPCV median net is negative for every daily cell.
- Choosing a daily horizon overfits badly: PBO is 0.90 for `daily_gbm_price` and
  0.96 for `daily_lstm_price`.
- The "above breakeven" labels on some daily rows mean nothing. The daily
  breakeven is about 50.6 %, so any point estimate above it gets the label, even
  when the interval straddles 50 %.

### 3. TI-LSTM, the H1→daily ensemble and Kronos direction: no evidence

- **TI-LSTM, h = 2:** 52.75 %, but with only 1,255 days the interval
  (49.96–55.54 %) contains 50 %, and its DSR is 0.14.
- **H1→daily ensemble:** below 50 % at every horizon.
- **Kronos direction:** every interval contains 50 %, with only about 1,160
  strided bars.

### 4. No volatility challenger beats GARCH(1,1) × day-of-week

| cell | MAE model | MAE GARCH × DoW | ΔMAE (95 % CI) |
|---|---:|---:|---|
| ensemble h = 1 | 0.2173 | 0.1929 | +0.0243 (0.0215, 0.0270) |
| ensemble h = 2 | 0.2997 | 0.2816 | +0.0181 (0.0146, 0.0216) |
| ensemble h = 5 | 0.4687 | 0.4498 | +0.0190 (0.0135, 0.0246) |
| Kronos h = 24 bars | 0.2159 | 0.2015 | +0.0144 (0.0068, 0.0232) |

Every interval lies entirely **above** zero: the challengers are *worse*. At
h = 1 even plain GARCH(1,1) (0.2152) beats the ensemble. This agrees with row 3
of `results/volatility_hypothesis_log.csv`, reached independently on the
production ensemble.

## What this means for the forward test

The six cells admitted to the forward family were fixed by arithmetic before any
of this was seen. They are the H1 GBM and Kronos direction at 1, 2 and 4 h. The
study gives each one a prior:

- **1 h and 2 h:** expected *predictive, not cost-viable*. The model is reliably
  above 50 % and reliably below the breakeven.
- **4 h:** the only cell where the H1 GBM sits at its breakeven. It is the cell
  worth watching. At full coverage, a forward decision on it takes about
  2.2 years.
- **Kronos:** expected undecided or DROP.

Every daily cell remains `UNDERPOWERED — NO DECISION` by design.

## Files

| File | Contents |
|---|---|
| [`horizon_curves.csv`](horizon_curves.csv) | one row per cell: accuracy, CI, AUC, PT, breakevens, gross/net at both costs, random-sign percentile, Clark–West, CPCV summary, DSR, the tuning decision |
| [`cpcv_paths.csv`](cpcv_paths.csv) | per-path accuracy and net for every direction cell |
| [`pbo_across_horizons.csv`](pbo_across_horizons.csv) | per model: PBO of picking its best horizon in sample |
| [`trial_log.csv`](trial_log.csv) | append-only count of every configuration ever fitted (the DSR's N) |
| [`artifact_manifest.csv`](artifact_manifest.csv) | SHA-256 of every fitted challenger in `research_models/` (gitignored), which the forward logger loads |
| [`study_record.json`](study_record.json) | the locked declaration (first fit 2026-10-01) |
| [`run_meta_hs-20261001.json`](run_meta_hs-20261001.json) | data ranges, macro sources, per-cell timings of the resumed run |

Re-running: `python -m src.forecast_eval.study --study-id hs-20261001` skips the
finished cells. A new grid or new configuration needs a new study id.
