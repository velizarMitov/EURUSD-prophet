# Horizon study: results (`hs-20261001` daily/H1, `hs-m15-20261002` M15 session)

> **DESCRIPTIVE. Not a verdict.** These curves describe the **challenger**
> models built by `src/forecast_eval/`, not the production models in `models/`.
> They were measured on development data that includes the spent test blocks,
> so they can rank and explain, but they cannot prove anything. The only arbiter
> for a horizon claim is the forward data (`results/forward_eval/`). No
> hypothesis log was written and no alpha was spent.

## What was run

Two studies, because adding a cadence changes the locked grid and so needs its
own record: `hs-20261001` (daily and H1) and `hs-m15-20261002` (the M15 session
layer). The earlier record is archived as
[`study_record.hs-20261001.json`](study_record.hs-20261001.json).

- **Record:** [`study_record.json`](study_record.json). It fixed the horizon grid,
  every configuration, the session declaration, the tuning rule and the 3-year
  cap, and was written before the first fit.
- **Grid:** M15 session cadence at {1, 2, 4, 8, 16, 26} bars. H1 cadence at
  {1, 2, 4, 6, 12, 24, 48, 120} bars. Daily cadence at {1, 2, 5} days.
- **Cells:** 38 in `hs-20261001` — 34 direction cells (10 challengers), the
  volatility ensemble at {1, 2, 5}, and Kronos volatility at 24 bars — plus 12 in
  `hs-m15-20261002` (2 session challengers × 6 horizons). **Every cell produced a
  result.** [`failures.csv`](failures.csv) keeps six records from the first M15 GBM
  attempt, which hit a bug in the nested-tuning call (the session-filtered rows are
  not a contiguous block, which the inner walk-forward needs). The fix passes the
  contiguous block plus the eligibility mask; the six cells were re-run and
  succeeded. The log is append-only, so the record of the failed attempt stays.
- **Data:**
  - daily: 1999-01-04 → 2026-08-10, 8,606 euro-era rows; macro from FRED
    (public CSV, on a sandboxed copy of the caches);
  - H1: 2017-01-20 → 2026-09-30, 60,312 bars;
  - M15: 2018-08-08 → 2026-08-07, 199,097 bars aggregated from
    `results/curl/raw/EURUSD_M1.parquet` (2,980,060 M1 bars, digest in the
    record). 62,238 fall inside the session over 2,075 session days;
  - Kronos: scored only after its 2024-06 training cutoff, on every 12th bar.
- **Method:**
  - an expanding purged walk-forward (5 folds, test on the later half);
  - CPCV (N = 6, k = 2; N = 4 for the volatility ensemble and the session LSTM);
  - nested-CV tuning for the GBM cells under the declared rule;
  - M15 cells train and score on session rows only, while the splits run on the
    full bar grid so the purge stays correct.
- **Costs:** the measured median spread of 0.5 pip is the primary cost (for M15
  the in-session median, which is also 0.50 pip, p90 0.60). The 1.5-pip round
  trip in `config.json` is the sensitivity case. Breakeven accuracy =
  0.5 + cost / (2 · E|move over h|), with E|move| taken over the rows each cell
  is actually scored on.
- **Compute:** 111 minutes for `hs-20261001`, plus about 108 minutes of Kronos
  sampling on the GPU; 13 minutes for the 12 M15 cells.

### The bar-label clock

The bar labels are **not UTC**. They are **Europe/Berlin** wall clock, which the
study establishes from the data rather than assuming. The forex week opens Sunday
17:00 America/New_York; read on Berlin's clock that instant is 23:00 normally and
22:00 in the March and late-October weeks when US and EU daylight saving disagree.
Measured over every weekend in the M1 file
([`run_meta_hs-m15-20261002.json`](run_meta_hs-m15-20261002.json)):

| weeks | n | predicted label hour | observed modal | match |
|---|---:|---:|---:|---:|
| daylight saving aligned | 396 | 23 | 23 | 99.2 % |
| US/EU mismatch | 29 | 22 | 22 | 100 % |
| all | 425 | | | **99.3 %** |

No other zone fits both rows — UTC would predict 21:00/22:00, New York a constant
17:00. A run whose weekly opens stop matching is **refused**, because every
intraday rule rests on this one fact. Every session rule is expressed against that
clock, so the owner's **15:30–23:00 Europe/Sofia** session is the fixed label
window **14:30–22:00** with no timezone conversion anywhere (Sofia is Berlin + 1 h
on every day of the year, under one EU daylight-saving rule).

An earlier draft of this program read the labels as UTC and converted them to
`America/New_York`. That put the session two hours off and mislocated the rollover
artifact of §1c at "19:00 New York". Every figure here is from the corrected clock.

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

### 1b. In the owner's session, only the 15-minute horizon beats a coin — and it is five points short of the spread

Both session challengers, on mid prices, inside 15:30–23:00 Europe/Sofia:

| horizon | model | accuracy | 95 % CI | breakeven 0.5 pip | net / trade | PT p | bid−mid gap |
|---|---|---:|---|---:|---:|---:|---:|
| **15 min** | GBM | **51.01 %** | **50.46–51.56** | 56.02 % | −0.3 pip | 0.0002 | 0.04 pp |
| **15 min** | LSTM | **51.12 %** | **50.56–51.66** | 56.02 % | −0.2 pip | 0.0001 | 0.20 pp |
| 30 min | GBM | 50.58 % | 49.95–51.22 | 54.27 % | −0.4 | 0.024 | 0.36 pp |
| 30 min | LSTM | 50.43 % | 49.76–51.12 | 54.27 % | −0.3 | 0.071 | 0.37 pp |
| 1 h | GBM | 49.66 % | 48.89–50.47 | 53.01 % | −0.6 | 0.87 | 0.06 pp |
| 1 h | LSTM | 49.89 % | 48.96–50.83 | 53.01 % | −0.5 | 0.65 | 0.70 pp |
| 2 h | GBM | 49.63 % | 48.52–50.74 | 52.10 % | −0.3 | 0.84 | 0.60 pp |
| 4 h | GBM | 48.58 % | 47.18–50.14 | 51.46 % | −2.3 | 1.00 | 0.14 pp |
| 6.5 h | GBM | 48.84 % | 46.60–50.99 | 51.08 % | −1.8 | 0.92 | 0.62 pp |

(The LSTM's 2 h, 4 h and 6.5 h rows behave the same; 22,000–29,000 scored rows at
15–60 min.)

- **Two cells have an interval above 50 %: both models at 15 minutes.** The edge
  is about one point, and the Pesaran–Timmermann p-value is below 0.001.
- **It cannot pay the spread.** At 15 minutes the mean move is 4.15 pips against
  a 0.5-pip spread, so breakeven is **56.02 %** — five points above what the
  models reach. Every net figure, at every cost level, for all twelve cells, is
  negative. Every Deflated Sharpe is 0.000.
- **Nothing survives past 30 minutes.** From 1 hour on, the point estimates sit
  *below* 50 %.
- **The LSTM's PBO across horizons is 0.77** (the GBM's is 0.28): picking the
  session LSTM's best horizon in sample is mostly overfitting.
- **The bid-versus-mid parity check passed on all twelve cells**, worst gap
  0.70 pp against a declared 1.0 pp tolerance. That is the measurement that the
  session window is clean, not an assumption.

### 1c. The rollover artifact the session rule exists to exclude

One hour after the session ends, a bid-only series reads a 72 % one-hour
"accuracy". It is not skill:

| label hour | New York | up-rate on bid | up-rate on mid | gap | median spread |
|---|---|---:|---:|---:|---:|
| 14:00–21:00 (session) | 08:00–15:00 | 48.0–50.5 % | 48.1–50.5 % | ≤ 0.69 pp | 0.5 pip |
| 22:00 | 16:00 | 46.60 % | 57.08 % | +10.48 pp | 0.9 pip |
| 23:00 | **17:00 — rollover** | **72.20 %** | 60.48 % | −11.72 pp | 2.4 pip |

The spread quintuples at the New York 17:00 rollover; the bid falls because the
spread opens, not because the market falls. A sign model fitted there would
backtest above 70 % and lose money at any cost level — its breakeven is near
80 %. This is why every sub-hourly target is built on the **mid** price from each
bar's own spread, why the session stops at 23:00 Sofia, and why a trade must both
start and end inside the session to be eligible.

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

The forward verdict asks the owner's question: **does the model call the direction
(+ or -) better than a coin?** KEEP means the lower bound on accuracy is above 50 %.
The spread result is reported beside it but does not decide it. The owner chose this
on 2026-10-02. ActivTrades charges no commission, so the spread is the only cost.

The **twelve** cells admitted to the forward family were fixed by arithmetic before any
of this was seen: 46 candidates iterate to 12 at alpha = 0.05/12 = 0.004167, needing
3,817 forecasts each. The study gives each one a prior:

| cell | horizon | years to decide | expected verdict |
|---|---|---:|---|
| `m15_session_gbm`, `m15_session_lstm` | 15 min | 0.51 | **KEEP**, cost-labelled |
| `m15_session_gbm`, `m15_session_lstm` | 30 min | 1.05 | DROP |
| `m15_session_gbm`, `m15_session_lstm` | 1 h | 2.10 | DROP |
| `h1_gbm` | 1, 2, 4 h | 0.62 / 1.24 / 2.47 | **KEEP**, 1 h and 2 h cost-labelled |
| `kronos_direction` | 1, 2, 4 h | 0.62 / 1.24 / 2.47 | DROP |

- **The 15-minute session cells and the H1 GBM** are the only cells whose study
  interval cleared 50 %, so they are the ones expected to end KEEP. Every one of them
  is expected to carry the cost label *predictive, not cost-viable* — the 15-minute
  cells by five points, the H1 1 h cell by one.
- **Kronos direction and everything from 30 minutes up** are expected to end DROP.
- The first possible verdict is about **six months** in: the 15-minute cells at 0.51
  years and the H1 1-hour cells at 0.62.
- The other 34 candidate cells stay `UNDERPOWERED — NO DECISION` by design — the M15
  cells at 2 h and beyond (4.9–14.7 years), H1 at h ≥ 6, and every daily cell.

## Files

| File | Contents |
|---|---|
| [`horizon_curves.csv`](horizon_curves.csv) | one row per cell: accuracy, CI, AUC, PT, breakevens, gross/net at both costs, random-sign percentile, Clark–West, CPCV summary, DSR, the tuning decision |
| [`cpcv_paths.csv`](cpcv_paths.csv) | per-path accuracy and net for every direction cell |
| [`pbo_across_horizons.csv`](pbo_across_horizons.csv) | per model: PBO of picking its best horizon in sample |
| [`trial_log.csv`](trial_log.csv) | append-only count of every configuration ever fitted (the DSR's N) |
| [`artifact_manifest.csv`](artifact_manifest.csv) | SHA-256 of every fitted challenger in `research_models/` (gitignored), which the forward logger loads |
| [`study_record.json`](study_record.json) | the locked declaration for `hs-m15-20261002`: grid, configurations, the session rules, the M1 source digest |
| [`study_record.hs-20261001.json`](study_record.hs-20261001.json) | the archived declaration of the daily/H1 study (first fit 2026-10-01) |
| [`run_meta_hs-20261001.json`](run_meta_hs-20261001.json) | data ranges, macro sources, per-cell timings of the resumed run |
| [`run_meta_hs-m15-20261002.json`](run_meta_hs-m15-20261002.json) | the same for the M15 session layer, plus the session summary and the clock evidence |

`horizon_curves.csv` holds both studies; the `study_id` column separates them, and
the M15 rows add `n_eligible`, `price_definition`, `accuracy_bid`,
`parity_diff_pp`, `parity_noise_floor_pp`, `parity_threshold_pp` and
`parity_label`.

Re-running: `python -m src.forecast_eval.study --study-id hs-m15-20261002` skips the
finished cells. A new grid or new configuration needs a new study id. The M15 layer
needs `results/curl/raw/EURUSD_M1.parquet`, which is excluded from git (`DATA.md` §7)
and regenerable with `pip install pyarrow && python -m src.curl_mt5_fetch`; its SHA-256
and row count are in the record so a rebuilt copy can be proven identical.
