# Gold pre-registration

Committed before any validation score exists (openspec change `gold-forecasting`). This file is never edited after its commit; corrections go to `results/gold/RESULTS.md`.

- Created: 2026-10-21T14:34:31+00:00
- Config digest (`src/gold/gold_config.json`): `3d3dd5bf03e05d4098850b38268307a9195eff17dbc2e02a19d27a46cd1c31bc`
- Price definition: MT5 bid OHLC; spread, tick volume and real volume used nowhere
- **Cost basis: GROSS of cost -- statistical skill only, never a trading result**

## Pinned data

| cadence | symbol | rows | first bar | last bar | sha256 |
|---|---|---|---|---|---|
| D1 | GOLD | 3466 | 2013-08-08 00:00:00+00:00 | 2026-10-20 00:00:00+00:00 | `8c050f2ccd8fae74…` |
| H1 | GOLD | 70015 | 2015-01-02 04:00:00+00:00 | 2026-10-21 15:00:00+00:00 | `01db01fc7593fdb6…` |

## Clock

97.6% of 624 weekly opens land on the predicted Europe/Berlin hour (threshold 95%).

## Split

| cadence | train | validation | test | eligible validation rows |
|---|---|---|---|---|
| D1 | 2426 | 346 | 694 | 346 |
| H1 | 49010 | 7002 | 14003 | 6376 |

Eligibility: H1: the as-of and target bars are consecutive hours and the as-of bar follows its predecessor by one hour (no bar bordering a market gap); D1: every row with a next bar. Zero moves are excluded from direction scoring.

## Families

### Direction (alpha = 0.02500)

Edge 3% over 50 % at 80% power. Baselines: training-majority class and a 50 % coin, on the same rows.

| cell | cadence | validation n | required n | status |
|---|---|---|---|---|
| d1_gbm | D1 | 346 | 2640 | UNDERPOWERED — NO DECISION |
| d1_lstm | D1 | 346 | 2640 | UNDERPOWERED — NO DECISION |
| h1_gbm | H1 | 6376 | 2640 | ADMITTED |
| h1_lstm | H1 | 6376 | 2640 | ADMITTED |

Verdict: CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap CI of mean(correct_model - correct_majority) AND of mean(correct_model - 0.5) both have lower bound > 0; else DROP. Pesaran-Timmermann reported alongside, not decisive.

### Return (alpha = 0.01250)

Cells: d1_gbm, d1_lstm, h1_gbm, h1_lstm. Baseline: the zero-return forecast (driftless random walk).

Verdict: CLEARED iff the one-sided Clark-West p-value < alpha; else DROP.

### Volatility (alpha = 0.02500)

- `d1_lstm` vs `d1_garch` (D1)
- `h1_lstm` vs `h1_garch_hod` (H1)

Verdict: CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap CI of mean(|err_baseline| - |err_model|) has lower bound > 0; else DROP.

## CI method

paired circular moving-block bootstrap, block length D1 = 5, H1 = 24, 2000 resamples, seed 42.

## Models

- GBM grid: `{"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.03, 0.1]}`; tuning: inner purged walk-forward on the training block only; best mean inner ROC-AUC.
- LSTM D1: `{"time_steps": 20, "units": 64, "dropout": 0.3, "learning_rate": 0.001, "epochs": 100, "batch_size": 64, "patience": 10}`
- LSTM H1: `{"time_steps": 24, "units": 32, "dropout": 0.2, "learning_rate": 0.001, "epochs": 60, "batch_size": 256, "patience": 8}`
- Seeds: [42, 43, 44, 45, 46] (mean of all five; a partial ensemble is never scored).
- d1_garch: GARCH(1,1), constant mean, normal, fit on training returns only.
- h1_garch_hod: GARCH(1,1), zero mean, on returns / training mean |r| per label hour; rescaled by the next hour.

## Descriptive only (never a verdict)

- accuracy by hour of day, flagged when > 1/2 of excess accuracy comes from the 00:00-06:00 labels (high-spread hours)
- Deflated Sharpe of a gross sign strategy over the gold trial log
- PBO across cells

## Stages

- **validation**: fit [0:70 %], score [70:80 %] -- the only arbiter
- **test**: after every validation verdict is committed: fit [0:80 %], score [80:100 %], once
- **final**: fit [0:100 %] -> frozen forward artifacts

## Expected outcomes (written before any score)

| family | cell | expected |
|---|---|---|
| direction | d1_gbm | UNDERPOWERED — NO DECISION (~50 %) |
| direction | d1_lstm | UNDERPOWERED — NO DECISION (~50 %) |
| direction | h1_gbm | possible CLEARED at 51-53 % (H_dir.1 GBM analogue: KEEP on EUR/USD 52.7 %, AUD/USD, CHF/USD; DROP on GBP/USD -- h1_direction_hypothesis_log.csv) |
| direction | h1_lstm | DROP more likely than CLEARED (H_dir.2: the EUR/USD H1 LSTM scored 51.6 % vs the GBM 52.7 %, DROP) |
| return | d1_gbm | DROP (Huber-loss shrinkage to the ~0 % conditional mean) |
| return | d1_lstm | DROP (Huber-loss shrinkage to the ~0 % conditional mean) |
| return | h1_gbm | DROP (Huber-loss shrinkage to the ~0 % conditional mean) |
| return | h1_lstm | DROP (Huber-loss shrinkage to the ~0 % conditional mean) |
| volatility | d1_lstm | CLEARED (volatility clustering; EUR/USD analogue CLEARED) |
| volatility | h1_lstm | CLEARED |
