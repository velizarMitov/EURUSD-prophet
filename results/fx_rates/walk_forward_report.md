# fx-rates walk-forward

> **Gross of spread and slippage — not a trading result.**

Stage-1 run 2026-11-21T21:06:29+00:00, config digest `d9a6001de49d`, feature list `937460de3544`. Decision at the D1 close of `t`, yields from `t−1`, target = sign of the mid move to `t+5`. Out-of-sample days 2015-01-02..2026-11-13 in 24 folds of 6 months; purge 5 + embargo 5 days between slices.

## Verdicts

alpha = 0.0125 (0.05/4), one-sided circular block bootstrap, block 10 days, 10000 resamples, on the per-day log-loss difference. KEEP needs the whole interval below zero. An untriggered cell keeps its alpha unspent.

| Cell | Days | Model log-loss | Reference log-loss | Accuracy | Wilson 95 % on n_eff | Mean diff | Upper bound | Verdict |
|---|---|---|---|---|---|---|---|---|
| EURUSD_lightgbm_vs_prior | 3,078 | 0.69429 | 0.69333 | 50.58% | 47.2%–53.9% (n_eff 848) | +0.00096 | +0.00369 | **DROP** |
| GBPUSD_lightgbm_vs_prior | 3,077 | 0.69346 | 0.69293 | 52.52% | 49.1%–55.9% (n_eff 812) | +0.00053 | +0.00895 | **DROP** |
| EURUSD_mlp_vs_lightgbm | — | — | — | — | — | — | — | **NOT TRIGGERED** |
| GBPUSD_mlp_vs_lightgbm | — | — | — | — | — | — | — | **NOT TRIGGERED** |

## Power (from day counts and the target overlap)

| Pair | Out-of-sample days | VIF | Effective n | MDE on effective n | Weeks | MDE on weeks |
|---|---|---|---|---|---|---|
| EURUSD | 3,078 | 3.63 | 848 | 55.29% | 615 | 56.22% |
| GBPUSD | 3,077 | 3.79 | 812 | 55.41% | 615 | 56.22% |

## Pooled out-of-sample metrics (descriptive)

### EURUSD (3,078 days, up-rate 48.44%, effective n 848)

| Model | Log-loss | Brier | Accuracy | Wilson 95 % on n_eff | Accuracy on stale-yield days |
|---|---|---|---|---|---|
| class prior (fit slice) | 0.69333 | 0.25009 | 51.56% | 48.2%–54.9% | 50.3% (n = 199) |
| LightGBM | 0.69429 | 0.25057 | 50.58% | 47.2%–53.9% | 47.7% (n = 199) |

| Year | Days | class prior (fit slice) accuracy | LightGBM accuracy |
|---|---|---|---|
| 2015 | 259 | 56.0% | 56.0% |
| 2016 | 259 | 53.3% | 52.1% |
| 2017 | 259 | 40.2% | 40.2% |
| 2018 | 259 | 56.4% | 52.5% |
| 2019 | 259 | 51.4% | 51.4% |
| 2020 | 260 | 42.3% | 42.3% |
| 2021 | 260 | 57.7% | 57.7% |
| 2022 | 259 | 55.6% | 55.6% |
| 2023 | 259 | 49.4% | 48.6% |
| 2024 | 260 | 55.0% | 55.0% |
| 2025 | 259 | 42.5% | 42.5% |
| 2026 | 226 | 60.2% | 53.5% |

### GBPUSD (3,077 days, up-rate 49.01%, effective n 812)

| Model | Log-loss | Brier | Accuracy | Wilson 95 % on n_eff | Accuracy on stale-yield days |
|---|---|---|---|---|---|
| class prior (fit slice) | 0.69293 | 0.24989 | 51.06% | 47.6%–54.5% | 47.2% (n = 199) |
| LightGBM | 0.69346 | 0.24995 | 52.52% | 49.1%–55.9% | 49.7% (n = 199) |

| Year | Days | class prior (fit slice) accuracy | LightGBM accuracy |
|---|---|---|---|
| 2015 | 259 | 51.7% | 51.7% |
| 2016 | 259 | 51.7% | 51.7% |
| 2017 | 258 | 48.1% | 61.6% |
| 2018 | 259 | 56.8% | 54.4% |
| 2019 | 259 | 50.2% | 47.1% |
| 2020 | 260 | 45.4% | 45.4% |
| 2021 | 260 | 49.6% | 53.8% |
| 2022 | 259 | 57.1% | 53.7% |
| 2023 | 259 | 48.3% | 48.3% |
| 2024 | 260 | 55.4% | 55.4% |
| 2025 | 259 | 44.8% | 47.5% |
| 2026 | 226 | 54.0% | 60.6% |

## Log-loss per fold

| Fold | Start | EURUSD prior | EURUSD lightgbm | GBPUSD prior | GBPUSD lightgbm |
|---|---|---|---|---|---|
| 0 | 2015-01-02 | 0.69118 | 0.68208 | 0.69418 | 0.69179 |
| 1 | 2015-07-01 | 0.68069 | 0.68260 | 0.69187 | 0.69105 |
| 2 | 2016-01-04 | 0.69709 | 0.69330 | 0.69440 | 0.69536 |
| 3 | 2016-07-01 | 0.68557 | 0.69580 | 0.69162 | 0.68100 |
| 4 | 2017-01-02 | 0.71270 | 0.71237 | 0.69398 | 0.71486 |
| 5 | 2017-07-03 | 0.71573 | 0.71702 | 0.69514 | 0.67316 |
| 6 | 2018-01-02 | 0.68681 | 0.70067 | 0.68930 | 0.70376 |
| 7 | 2018-07-02 | 0.68732 | 0.68730 | 0.68929 | 0.68773 |
| 8 | 2019-01-02 | 0.69381 | 0.69337 | 0.69240 | 0.69253 |
| 9 | 2019-07-01 | 0.69172 | 0.69021 | 0.69507 | 0.69742 |
| 10 | 2020-01-02 | 0.69250 | 0.69965 | 0.69089 | 0.69087 |
| 11 | 2020-07-01 | 0.71453 | 0.71772 | 0.70198 | 0.69994 |
| 12 | 2021-01-04 | 0.68932 | 0.68924 | 0.69666 | 0.69486 |
| 13 | 2021-07-01 | 0.68301 | 0.68261 | 0.69216 | 0.69835 |
| 14 | 2022-01-03 | 0.68241 | 0.68176 | 0.68584 | 0.69245 |
| 15 | 2022-07-01 | 0.69545 | 0.69469 | 0.69303 | 0.69567 |
| 16 | 2023-01-02 | 0.69651 | 0.69666 | 0.69414 | 0.69473 |
| 17 | 2023-07-03 | 0.69012 | 0.69354 | 0.69229 | 0.69189 |
| 18 | 2024-01-02 | 0.69395 | 0.69211 | 0.69320 | 0.69367 |
| 19 | 2024-07-01 | 0.68419 | 0.68464 | 0.68668 | 0.68598 |
| 20 | 2025-01-02 | 0.70770 | 0.70746 | 0.69943 | 0.71111 |
| 21 | 2025-07-01 | 0.69413 | 0.69338 | 0.69290 | 0.69572 |
| 22 | 2026-01-02 | 0.68646 | 0.68401 | 0.69143 | 0.68502 |
| 23 | 2026-07-01 | 0.68520 | 0.68977 | 0.69226 | 0.68064 |

## Reliability (predicted P(up) vs observed up-rate)

**EURUSD — LightGBM**

| P(up) bin | Days | Mean P(up) | Observed up-rate |
|---|---|---|---|
| 0.00-0.40 | 60 | 0.3861 | 0.4333 |
| 0.40-0.45 | 429 | 0.4333 | 0.4779 |
| 0.45-0.48 | 1,466 | 0.4701 | 0.4973 |
| 0.48-0.50 | 1,021 | 0.4864 | 0.4848 |
| 0.50-0.52 | 72 | 0.5067 | 0.3056 |
| 0.52-0.55 | 28 | 0.5326 | 0.5000 |
| 0.55-0.60 | 2 | 0.5605 | 0.0000 |
| 0.60-1.00 | 0 | — | — |

**GBPUSD — LightGBM**

| P(up) bin | Days | Mean P(up) | Observed up-rate |
|---|---|---|---|
| 0.00-0.40 | 246 | 0.3312 | 0.4390 |
| 0.40-0.45 | 239 | 0.4287 | 0.4477 |
| 0.45-0.48 | 416 | 0.4694 | 0.4207 |
| 0.48-0.50 | 1,511 | 0.4899 | 0.5043 |
| 0.50-0.52 | 281 | 0.5095 | 0.5125 |
| 0.52-0.55 | 230 | 0.5328 | 0.5478 |
| 0.55-0.60 | 93 | 0.5703 | 0.4731 |
| 0.60-1.00 | 61 | 0.6269 | 0.6885 |

## Weekly rule — gross of spread and slippage, descriptive, not a trading result

Every 5th out-of-sample day from a start offset: long when P(up) > 0.5, else short; held 5 FX days, mid to mid, no overlap, no cost. Offset 0 and the mean over offsets 0–4.

| Pair | Model | Offset | Weeks | Win rate | Wilson 95 % | Mean pips | Total pips | Sharpe (√52) | Profit factor |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | always long | 0 | 616 | 49.5% | 45.6%–53.5% | -1.7 | -1027 | -0.10 | 0.963 |
| EURUSD | always long | mean 0–4 | 616 | 48.4% | — | -1.5 | -906 | -0.09 | 0.968 |
| EURUSD | LightGBM | 0 | 616 | 50.2% | 46.2%–54.1% | +2.7 | +1665 | 0.16 | 1.062 |
| EURUSD | LightGBM | mean 0–4 | 616 | 50.6% | — | -0.3 | -202 | -0.02 | 0.994 |
| GBPUSD | always long | 0 | 616 | 49.5% | 45.6%–53.5% | -3.4 | -2121 | -0.14 | 0.946 |
| GBPUSD | always long | mean 0–4 | 616 | 49.0% | — | -3.3 | -2036 | -0.14 | 0.948 |
| GBPUSD | LightGBM | 0 | 616 | 52.1% | 48.2%–56.0% | +4.2 | +2570 | 0.17 | 1.069 |
| GBPUSD | LightGBM | mean 0–4 | 616 | 52.5% | — | +7.2 | +4402 | 0.30 | 1.124 |

_Generated 2026-11-21T21:07:10+00:00._
