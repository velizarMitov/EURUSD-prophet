# fx-fibonacci walk-forward

> **Gross of spread and slippage — not a trading result.**

Run 2026-11-21T13:50:22+00:00 on NVIDIA GeForce RTX 4070 Laptop GPU (torch 2.11.0+cu128, CUDA 12.8). Config digest `8792332ffa33`, feature list `a6d3bb9995ef`. Out-of-sample rows 2019-07-01..2026-11-19 in 15 folds of 6 months, purge + embargo 24 bars each side.

## Verdicts

alpha = 0.0125 (0.05/4), one-sided circular block bootstrap, block 24 bars, 10000 resamples. Difference = model − reference log-loss; KEEP needs the whole interval below zero.

| Cell | Rows | Model log-loss | Reference log-loss | Mean diff | Upper bound | MDE (80 %) | Verdict |
|---|---|---|---|---|---|---|---|
| EURUSD_mlp_vs_prior | 46,025 | 1.07788 | 0.69088 | +0.38699 | +0.41515 | 0.03943 | **DROP** |
| EURUSD_mlp_vs_lightgbm | 46,025 | 1.07788 | 0.94281 | +0.13507 | +0.16457 | 0.04214 | **DROP** |
| GBPUSD_mlp_vs_prior | 46,025 | 1.06733 | 0.69308 | +0.37425 | +0.40232 | 0.03972 | **DROP** |
| GBPUSD_mlp_vs_lightgbm | 46,025 | 1.06733 | 0.95012 | +0.11721 | +0.14747 | 0.04218 | **DROP** |

- **EURUSD:** Decided once on this family's pooled out-of-sample walk-forward rows (2019-07-01..2026-11-19), owner decision 2026-11-21. Gross predictive result, not a trading edge: spread and slippage are excluded by owner decision. Its 2024-11-19..2026-07-28 rows overlap the H1 direction family block read once on 2026-07-30 (results/h1_direction_hypothesis_log.csv), and daily EURUSD Fibonacci features were rejected as hypothesis #7 (results/feature_hypothesis_log.csv). Recorded as a fact, not a blocker.
- **GBPUSD:** Decided once on this family's pooled out-of-sample walk-forward rows (2019-07-01..2026-11-19), owner decision 2026-11-21. Gross predictive result, not a trading edge: spread and slippage are excluded by owner decision. GBPUSD H1 history was used by the H1 direction family replication H_dir.3 (DROP, results/h1_direction_hypothesis_log.csv). Recorded as a fact, not a blocker.

## Pooled out-of-sample metrics

### EURUSD (46,025 rows; neutral / up / down = 77.4% / 11.4% / 11.2%)

| Model | Log-loss | Macro F1 | Brier | ECE |
|---|---|---|---|---|
| class prior (train fold) | 0.69088 | 0.2909 | 0.37521 | 0.0027 |
| MLP 5-seed ensemble (registered) | 1.07788 | 0.2860 | 0.65578 | 0.0354 |
| LightGBM (registered baseline) | 0.94281 | 0.3315 | 0.53500 | 0.0616 |
| L2 logistic (descriptive) | 1.09964 | 0.2836 | 0.66731 | 0.0383 |
| MLP 64-32 (descriptive) | 1.07351 | 0.2992 | 0.65322 | 0.0513 |

### GBPUSD (46,025 rows; neutral / up / down = 77.3% / 11.2% / 11.5%)

| Model | Log-loss | Macro F1 | Brier | ECE |
|---|---|---|---|---|
| class prior (train fold) | 0.69308 | 0.2907 | 0.37673 | 0.0041 |
| MLP 5-seed ensemble (registered) | 1.06733 | 0.2986 | 0.64870 | 0.0568 |
| LightGBM (registered baseline) | 0.95012 | 0.3290 | 0.54618 | 0.0597 |
| L2 logistic (descriptive) | 1.10480 | 0.2750 | 0.67238 | 0.0519 |
| MLP 64-32 (descriptive) | 1.06610 | 0.3076 | 0.64986 | 0.0735 |

## Log-loss per fold

| Fold | Start | EURUSD prior | EURUSD mlp | EURUSD lightgbm | GBPUSD prior | GBPUSD mlp | GBPUSD lightgbm |
|---|---|---|---|---|---|---|---|
| 0 | 2019-07-01 | 0.67300 | 0.99048 | 0.78289 | 0.76097 | 1.07936 | 1.05626 |
| 1 | 2020-01-01 | 0.76209 | 1.08649 | 0.95615 | 0.71478 | 1.08741 | 1.07668 |
| 2 | 2020-07-01 | 0.66085 | 1.09340 | 0.89393 | 0.71165 | 1.05505 | 0.97573 |
| 3 | 2021-01-03 | 0.66072 | 1.06446 | 0.90853 | 0.60641 | 1.03924 | 0.81972 |
| 4 | 2021-07-01 | 0.65694 | 1.03741 | 0.91244 | 0.63857 | 1.04375 | 0.91705 |
| 5 | 2022-01-02 | 0.75797 | 1.04770 | 1.04138 | 0.73634 | 1.10220 | 1.02306 |
| 6 | 2022-07-01 | 0.68540 | 1.09803 | 0.94552 | 0.70423 | 1.08243 | 1.01278 |
| 7 | 2023-01-02 | 0.66888 | 1.12554 | 1.02279 | 0.66120 | 1.05908 | 0.87239 |
| 8 | 2023-07-02 | 0.71408 | 1.08939 | 1.03570 | 0.67368 | 1.09089 | 0.94771 |
| 9 | 2024-01-02 | 0.69132 | 1.06932 | 1.03930 | 0.62548 | 1.03527 | 0.89920 |
| 10 | 2024-07-01 | 0.72136 | 1.10185 | 0.91446 | 0.76704 | 1.10777 | 0.97260 |
| 11 | 2025-01-01 | 0.70135 | 1.13575 | 0.91082 | 0.74320 | 1.05584 | 1.00686 |
| 12 | 2025-07-01 | 0.60077 | 1.07800 | 0.91488 | 0.68993 | 1.04199 | 0.90493 |
| 13 | 2026-01-01 | 0.68630 | 1.07039 | 0.82629 | 0.67696 | 1.05147 | 0.86802 |
| 14 | 2026-07-01 | 0.73362 | 1.08234 | 1.06556 | 0.68201 | 1.08084 | 0.88259 |

## MLP reliability (top-class confidence vs accuracy)

**EURUSD**

| Confidence | Rows | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 0 | — | — |
| 0.1-0.2 | 0 | — | — |
| 0.2-0.3 | 0 | — | — |
| 0.3-0.4 | 23,753 | 0.3698 | 0.3652 |
| 0.4-0.5 | 16,166 | 0.4416 | 0.3919 |
| 0.5-0.6 | 4,763 | 0.5370 | 0.4375 |
| 0.6-0.7 | 1,128 | 0.6370 | 0.4805 |
| 0.7-0.8 | 180 | 0.7330 | 0.4444 |
| 0.8-0.9 | 35 | 0.8342 | 0.3714 |
| 0.9-1.0 | 0 | — | — |

**GBPUSD**

| Confidence | Rows | Mean confidence | Accuracy |
|---|---|---|---|
| 0.0-0.1 | 0 | — | — |
| 0.1-0.2 | 0 | — | — |
| 0.2-0.3 | 0 | — | — |
| 0.3-0.4 | 26,135 | 0.3752 | 0.4283 |
| 0.4-0.5 | 18,763 | 0.4329 | 0.3851 |
| 0.5-0.6 | 1,126 | 0.5191 | 0.2256 |
| 0.6-0.7 | 1 | 0.6095 | 0.0000 |
| 0.7-0.8 | 0 | — | — |
| 0.8-0.9 | 0 | — | — |
| 0.9-1.0 | 0 | — | — |

## Fixed rule — gross of spread and slippage, descriptive, not a trading result

Acts when the predicted class is up or down, one position per pair, exit at the first barrier or the time limit. Percent of the entry close.

| Pair | Model | Trades | Skipped | Win rate | Expectancy % | Profit factor | Sharpe (√252) | Max DD % | Total % |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | MLP 5-seed ensemble (registered) | 1,667 | 25,511 | 0.511 | 0.0065 | 1.039 | 0.22 | 18.51 | 10.78 |
| EURUSD | LightGBM (registered baseline) | 1,091 | 10,561 | 0.490 | -0.0084 | 0.952 | -0.23 | 19.80 | -9.13 |
| GBPUSD | MLP 5-seed ensemble (registered) | 1,661 | 24,122 | 0.489 | -0.0180 | 0.913 | -0.54 | 31.53 | -29.86 |
| GBPUSD | LightGBM (registered baseline) | 1,102 | 11,197 | 0.514 | -0.0069 | 0.965 | -0.16 | 25.24 | -7.62 |

_Generated 2026-11-21T14:08:07+00:00._
