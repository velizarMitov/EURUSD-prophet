# gold-bb-nyrange report

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

Only the registry rows are verdicts; every table below is descriptive and none is a trading result.

## Verdicts

| Family | Cell | Statistic | Bound | MDE (80 %) | Rows | Verdict |
|---|---|---|---|---|---|---|
| bb | `gold_bb_rule_mean_reversion` | -0.25053 | -0.46316 | 0.3018 | 173 | **DROP** |
| bb | `gold_bb_lightgbm_vs_base_rate` | +0.01196 | +0.03179 | 0.0266 | 106 | **DROP** |
| bb_sma | `gold_bb_sma_rule_mean_r` | -0.08650 | -0.22111 | 0.1949 | 151 | **DROP** |
| bb_sma | `gold_bb_sma_lightgbm_vs_base_rate` | -0.00350 | +0.03110 | 0.0491 | 77 | **DROP** |
| nyvol | `gold_harx_vs_har` | -0.02755 | -0.02000 | 0.0110 | 2211 | **KEEP** |
| nyvol | `gold_lgbm_vs_harx` | +0.02595 | +0.03489 | 0.0118 | 2211 | **DROP** |
| nyvol | `gold_mlp_vs_harx` | +0.01899 | +0.02920 | 0.0129 | 2211 | **DROP** |

## Power (labels.md, before any outcome)

| Cell | Sample | Minimum detectable effect |
|---|---|---|
| 1a rule | 173 events | 0.257 ATR |
| 1a model | 106 events | accuracy 0.636 |
| 1b rule | ≤ 179 trades | ≥ 0.209 R |
| 1b model | ≤ 110 trades | accuracy ≥ 0.634 |
| 2 (each) | 2211 days | 0.0632 SD |

## Part 1a — Bollinger reversion (3-day, ATR units)

Events 173; win (r > 0) 43.4%; mean r -0.251 ATR.

| Side | N | Win | Mean |
|---|---|---|---|
| above band (short) | 103 | 35.9% | -0.371 |
| below band (long) | 70 | 54.3% | -0.073 |

| Year | N | Win | Mean |
|---|---|---|---|
| 2015 | 12 | 50.0% | -0.113 |
| 2016 | 13 | 30.8% | -0.351 |
| 2017 | 15 | 46.7% | -0.106 |
| 2018 | 15 | 60.0% | +0.114 |
| 2019 | 12 | 41.7% | -0.383 |
| 2020 | 13 | 38.5% | -0.296 |
| 2021 | 20 | 45.0% | -0.087 |
| 2022 | 14 | 35.7% | -0.236 |
| 2023 | 16 | 43.8% | -0.364 |
| 2024 | 13 | 38.5% | -0.637 |
| 2025 | 17 | 29.4% | -0.609 |
| 2026 | 13 | 61.5% | +0.033 |

## Part 1b — the SMA20 trade (R)

Opened trades 152 (blocked by an open trade 28); with an exit 151; win (R > 0) 45.0%; mean R -0.087.
Exits: stop 38%, take profit 29%, time stop 33%.

| Side | N | Win | Mean |
|---|---|---|---|
| above band (short) | 89 | 40.4% | -0.183 |
| below band (long) | 62 | 51.6% | +0.052 |

| Year | N | Win | Mean |
|---|---|---|---|
| 2015 | 12 | 58.3% | +0.096 |
| 2016 | 12 | 41.7% | -0.350 |
| 2017 | 13 | 61.5% | +0.256 |
| 2018 | 14 | 35.7% | -0.043 |
| 2019 | 11 | 45.5% | -0.020 |
| 2020 | 12 | 58.3% | +0.154 |
| 2021 | 16 | 43.8% | -0.128 |
| 2022 | 11 | 27.3% | -0.343 |
| 2023 | 13 | 23.1% | -0.470 |
| 2024 | 12 | 50.0% | -0.025 |
| 2025 | 14 | 42.9% | -0.102 |
| 2026 | 11 | 54.5% | -0.066 |

## Part 2 — the NY session range

Scored days 2211; mean y -4.777.

| Year | Days | HAR-X / HAR | LightGBM / HAR | MLP / HAR |
|---|---|---|---|---|
| 2018 | 250 | 1.019 | 1.022 | 0.976 |
| 2019 | 250 | 0.827 | 1.086 | 0.928 |
| 2020 | 251 | 0.847 | 1.200 | 1.219 |
| 2021 | 250 | 0.797 | 0.944 | 0.879 |
| 2022 | 250 | 0.784 | 0.907 | 0.843 |
| 2023 | 249 | 0.908 | 0.966 | 0.920 |
| 2024 | 243 | 0.837 | 0.907 | 0.796 |
| 2025 | 245 | 0.852 | 0.917 | 0.892 |
| 2026 | 223 | 0.758 | 0.892 | 1.067 |
| all | 2211 | 0.847 | 0.991 | 0.952 |

_Generated 2026-11-23T06:00:19+00:00._
