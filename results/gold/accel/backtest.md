# Gold acceleration and next-hour direction (descriptive: every year has been seen; not evidence)

> **descriptive: every year has been seen; not evidence.** Walk-forward refits on the past only, but every year has been seen and the acceleration statistics were looked at before this study. No verdict, registration or registry row.

Generated 2026-11-04 18:10 UTC. Inputs: `h1` 70246 rows 2015-01-02 04:00:00+00:00 .. 2026-11-04 16:00:00+00:00 (sha256 62747dbab65b), `m15` 100024 rows 2022-08-12 04:30:00+00:00 .. 2026-11-04 15:45:00+00:00 (sha256 ca4d68551d1a).

h1_lstm baseline: the gold-window-strategy run recorded no digest of its H1 input, so h1_lstm is refit here on the pinned input.

Models and years (`ok` / `failed`): `h1_gbm` 2018-2026 ok; `h1_gbm_accel` 2018-2026 ok; `h1_lstm` 2018-2026 ok; `h1_lstm_accel` 2018-2026 ok; `h1_accel` 2018-2026 ok; `h1_gbm_accel_m15` 2024-2026 ok; `h1_accel_m15` 2024-2026 ok.

## All hours (descriptive: every year has been seen; not evidence)

| model | rows | hit rate | AUC | Brier | hit - coin [95 %] | hit - training majority [95 %] |
|---|---|---|---|---|---|---|
| `h1_gbm` | 47,620 | 51.28 % | 0.518 | 0.2507 | +1.28 pp [+0.85, +1.72] | +1.73 pp [+1.09, +2.35] |
| `h1_gbm_accel` | 47,620 | 51.21 % | 0.518 | 0.2501 | +1.21 pp [+0.78, +1.62] | +1.66 pp [+1.06, +2.20] |
| `h1_lstm` | 47,620 | 51.10 % | 0.515 | 0.2503 | +1.10 pp [+0.68, +1.53] | +1.55 pp [+0.98, +2.11] |
| `h1_lstm_accel` | 47,620 | 51.01 % | 0.515 | 0.2503 | +1.01 pp [+0.58, +1.45] | +1.46 pp [+0.85, +2.08] |
| `h1_accel` | 47,620 | 50.34 % | 0.507 | 0.2508 | +0.34 pp [-0.07, +0.79] | +0.79 pp [+0.20, +1.40] |
| `h1_gbm_accel_m15` | 15,350 | 50.40 % | 0.503 | 0.2629 | +0.40 pp [-0.37, +1.18] | +2.23 pp [+1.22, +3.28] |
| `h1_accel_m15` | 15,350 | 50.03 % | 0.502 | 0.2550 | +0.03 pp [-0.80, +0.89] | +1.87 pp [+0.70, +2.98] |

## The owner's window (as-of labels 14-20, 15:30-23:00 Sofia) (descriptive: every year has been seen; not evidence)

| model | rows | hit rate | AUC | Brier | hit - coin [95 %] | hit - training majority [95 %] |
|---|---|---|---|---|---|---|
| `h1_gbm` | 15,797 | 50.84 % | 0.511 | 0.2512 | +0.84 pp [+0.06, +1.52] | +1.73 pp [+0.61, +2.62] |
| `h1_gbm_accel` | 15,797 | 50.81 % | 0.509 | 0.2507 | +0.81 pp [+0.06, +1.44] | +1.71 pp [+0.63, +2.57] |
| `h1_lstm` | 15,797 | 50.83 % | 0.510 | 0.2508 | +0.83 pp [+0.01, +1.58] | +1.72 pp [+0.69, +2.64] |
| `h1_lstm_accel` | 15,797 | 50.29 % | 0.508 | 0.2509 | +0.29 pp [-0.44, +1.01] | +1.18 pp [+0.19, +2.08] |
| `h1_accel` | 15,797 | 49.45 % | 0.492 | 0.2514 | -0.55 pp [-1.37, +0.21] | +0.34 pp [-0.82, +1.37] |
| `h1_gbm_accel_m15` | 5,102 | 50.53 % | 0.506 | 0.2627 | +0.53 pp [-0.70, +1.88] | +2.82 pp [+1.18, +4.46] |
| `h1_accel_m15` | 5,102 | 49.49 % | 0.499 | 0.2560 | -0.51 pp [-1.88, +0.87] | +1.78 pp [-0.08, +3.49] |

## With acceleration minus without (descriptive: every year has been seen; not evidence)

Each model against its pair on exactly the same rows. A hit-rate difference in percentage points.

| model | pair | view | rows | hit | pair hit | difference [95 %] | reading |
|---|---|---|---|---|---|---|---|
| `h1_gbm_accel` | `h1_gbm` | all | 47,620 | 51.21 % | 51.28 % | -0.07 pp [-0.43, +0.30] | not distinguishable from zero |
| `h1_gbm_accel` | `h1_gbm` | window | 15,797 | 50.81 % | 50.84 % | -0.03 pp [-0.69, +0.66] | not distinguishable from zero |
| `h1_lstm_accel` | `h1_lstm` | all | 47,620 | 51.01 % | 51.10 % | -0.09 pp [-0.40, +0.24] | not distinguishable from zero |
| `h1_lstm_accel` | `h1_lstm` | window | 15,797 | 50.29 % | 50.83 % | -0.54 pp [-1.01, -0.01] | interval below zero (descriptive) |
| `h1_accel` | `h1_gbm` | all | 47,620 | 50.34 % | 51.28 % | -0.94 pp [-1.48, -0.36] | interval below zero (descriptive) |
| `h1_accel` | `h1_gbm` | window | 15,797 | 49.45 % | 50.84 % | -1.39 pp [-2.39, -0.30] | interval below zero (descriptive) |
| `h1_gbm_accel_m15` | `h1_gbm` | all | 15,350 | 50.40 % | 51.13 % | -0.74 pp [-1.70, +0.25] | not distinguishable from zero |
| `h1_gbm_accel_m15` | `h1_gbm` | window | 5,102 | 50.53 % | 50.86 % | -0.33 pp [-1.71, +1.36] | not distinguishable from zero |
| `h1_accel_m15` | `h1_gbm` | all | 15,350 | 50.03 % | 51.13 % | -1.10 pp [-2.27, -0.01] | interval below zero (descriptive) |
| `h1_accel_m15` | `h1_gbm` | window | 5,102 | 49.49 % | 50.86 % | -1.37 pp [-2.91, +0.31] | not distinguishable from zero |

## Per year, with acceleration minus without (descriptive: every year has been seen; not evidence)

| model | year | view | rows | difference [95 %] | reading |
|---|---|---|---|---|---|
| `h1_gbm_accel` | 2018 | all | 5,371 | +0.54 pp [-0.44, +1.46] | not distinguishable from zero |
| `h1_gbm_accel` | 2018 | window | 1,778 | +0.67 pp [-0.50, +2.50] | not distinguishable from zero |
| `h1_gbm_accel` | 2019 | all | 5,372 | -0.04 pp [-1.02, +0.80] | not distinguishable from zero |
| `h1_gbm_accel` | 2019 | window | 1,775 | -0.79 pp [-2.50, +1.00] | not distinguishable from zero |
| `h1_gbm_accel` | 2020 | all | 5,400 | -0.43 pp [-1.31, +0.39] | not distinguishable from zero |
| `h1_gbm_accel` | 2020 | window | 1,790 | -1.40 pp [-2.72, +0.17] | not distinguishable from zero |
| `h1_gbm_accel` | 2021 | all | 5,372 | -0.19 pp [-1.09, +0.54] | not distinguishable from zero |
| `h1_gbm_accel` | 2021 | window | 1,776 | +0.23 pp [-1.17, +1.72] | not distinguishable from zero |
| `h1_gbm_accel` | 2022 | all | 5,385 | +0.33 pp [-0.85, +1.52] | not distinguishable from zero |
| `h1_gbm_accel` | 2022 | window | 1,788 | -0.17 pp [-2.33, +2.50] | not distinguishable from zero |
| `h1_gbm_accel` | 2023 | all | 5,370 | +0.09 pp [-1.28, +1.63] | not distinguishable from zero |
| `h1_gbm_accel` | 2023 | window | 1,788 | +0.73 pp [-1.44, +3.83] | not distinguishable from zero |
| `h1_gbm_accel` | 2024 | all | 5,408 | +0.13 pp [-1.21, +1.43] | not distinguishable from zero |
| `h1_gbm_accel` | 2024 | window | 1,800 | +1.28 pp [-0.67, +3.28] | not distinguishable from zero |
| `h1_gbm_accel` | 2025 | all | 5,390 | -0.15 pp [-1.13, +0.74] | not distinguishable from zero |
| `h1_gbm_accel` | 2025 | window | 1,792 | +0.28 pp [-1.72, +1.89] | not distinguishable from zero |
| `h1_gbm_accel` | 2026 | all | 4,552 | -1.05 pp [-2.43, +0.07] | not distinguishable from zero |
| `h1_gbm_accel` | 2026 | window | 1,510 | -1.26 pp [-4.36, +1.03] | not distinguishable from zero |
| `h1_lstm_accel` | 2018 | all | 5,371 | -0.32 pp [-1.31, +0.48] | not distinguishable from zero |
| `h1_lstm_accel` | 2018 | window | 1,778 | -0.96 pp [-2.11, +0.06] | not distinguishable from zero |
| `h1_lstm_accel` | 2019 | all | 5,372 | -0.26 pp [-1.35, +0.63] | not distinguishable from zero |
| `h1_lstm_accel` | 2019 | window | 1,775 | -0.51 pp [-2.44, +0.94] | not distinguishable from zero |
| `h1_lstm_accel` | 2020 | all | 5,400 | +0.78 pp [-0.02, +1.46] | not distinguishable from zero |
| `h1_lstm_accel` | 2020 | window | 1,790 | +0.34 pp [-1.06, +1.72] | not distinguishable from zero |
| `h1_lstm_accel` | 2021 | all | 5,372 | -0.43 pp [-1.33, +0.44] | not distinguishable from zero |
| `h1_lstm_accel` | 2021 | window | 1,776 | -0.17 pp [-1.83, +1.17] | not distinguishable from zero |
| `h1_lstm_accel` | 2022 | all | 5,385 | -0.67 pp [-1.70, +0.24] | not distinguishable from zero |
| `h1_lstm_accel` | 2022 | window | 1,788 | -0.78 pp [-2.44, +0.56] | not distinguishable from zero |
| `h1_lstm_accel` | 2023 | all | 5,370 | -0.30 pp [-1.28, +0.72] | not distinguishable from zero |
| `h1_lstm_accel` | 2023 | window | 1,788 | -1.12 pp [-2.78, +0.67] | not distinguishable from zero |
| `h1_lstm_accel` | 2024 | all | 5,408 | +0.94 pp [+0.07, +1.87] | interval above zero (descriptive; by chance about one interval in twenty excludes zero) |
| `h1_lstm_accel` | 2024 | window | 1,800 | -0.17 pp [-1.89, +1.33] | not distinguishable from zero |
| `h1_lstm_accel` | 2025 | all | 5,390 | -0.32 pp [-1.17, +0.46] | not distinguishable from zero |
| `h1_lstm_accel` | 2025 | window | 1,792 | -0.67 pp [-1.67, +0.83] | not distinguishable from zero |
| `h1_lstm_accel` | 2026 | all | 4,552 | -0.29 pp [-1.34, +0.83] | not distinguishable from zero |
| `h1_lstm_accel` | 2026 | window | 1,510 | -0.86 pp [-2.37, +0.90] | not distinguishable from zero |
| `h1_accel` | 2018 | all | 5,371 | -0.35 pp [-1.87, +1.63] | not distinguishable from zero |
| `h1_accel` | 2018 | window | 1,778 | -0.06 pp [-2.11, +3.95] | not distinguishable from zero |
| `h1_accel` | 2019 | all | 5,372 | -1.21 pp [-2.98, +0.52] | not distinguishable from zero |
| `h1_accel` | 2019 | window | 1,775 | -2.03 pp [-5.11, +1.28] | not distinguishable from zero |
| `h1_accel` | 2020 | all | 5,400 | -1.50 pp [-3.30, -0.17] | interval below zero (descriptive) |
| `h1_accel` | 2020 | window | 1,790 | -1.68 pp [-4.72, +2.00] | not distinguishable from zero |
| `h1_accel` | 2021 | all | 5,372 | -0.86 pp [-2.72, +0.43] | not distinguishable from zero |
| `h1_accel` | 2021 | window | 1,776 | +0.62 pp [-3.00, +2.83] | not distinguishable from zero |
| `h1_accel` | 2022 | all | 5,385 | -0.11 pp [-1.48, +1.52] | not distinguishable from zero |
| `h1_accel` | 2022 | window | 1,788 | -1.96 pp [-3.11, +1.06] | not distinguishable from zero |
| `h1_accel` | 2023 | all | 5,370 | -1.23 pp [-2.85, +0.50] | not distinguishable from zero |
| `h1_accel` | 2023 | window | 1,788 | -1.01 pp [-3.72, +2.67] | not distinguishable from zero |
| `h1_accel` | 2024 | all | 5,408 | -0.63 pp [-2.34, +1.41] | not distinguishable from zero |
| `h1_accel` | 2024 | window | 1,800 | -1.00 pp [-3.56, +1.72] | not distinguishable from zero |
| `h1_accel` | 2025 | all | 5,390 | -1.09 pp [-2.74, +0.41] | not distinguishable from zero |
| `h1_accel` | 2025 | window | 1,792 | -2.46 pp [-5.39, +0.83] | not distinguishable from zero |
| `h1_accel` | 2026 | all | 4,552 | -1.54 pp [-3.51, +0.57] | not distinguishable from zero |
| `h1_accel` | 2026 | window | 1,510 | -3.25 pp [-6.86, +0.45] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2024 | all | 5,408 | -0.07 pp [-1.96, +1.74] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2024 | window | 1,800 | +0.50 pp [-1.83, +3.50] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2025 | all | 5,390 | -1.02 pp [-2.70, +0.63] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2025 | window | 1,792 | -0.11 pp [-2.72, +2.22] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2026 | all | 4,552 | -1.19 pp [-2.76, +0.13] | not distinguishable from zero |
| `h1_gbm_accel_m15` | 2026 | window | 1,510 | -1.59 pp [-4.23, +0.51] | not distinguishable from zero |
| `h1_accel_m15` | 2024 | all | 5,408 | -1.18 pp [-3.41, +0.98] | not distinguishable from zero |
| `h1_accel_m15` | 2024 | window | 1,800 | -1.06 pp [-3.39, +1.61] | not distinguishable from zero |
| `h1_accel_m15` | 2025 | all | 5,390 | -1.13 pp [-2.94, +0.50] | not distinguishable from zero |
| `h1_accel_m15` | 2025 | window | 1,792 | -2.01 pp [-4.33, +0.83] | not distinguishable from zero |
| `h1_accel_m15` | 2026 | all | 4,552 | -0.97 pp [-3.22, +0.94] | not distinguishable from zero |
| `h1_accel_m15` | 2026 | window | 1,510 | -0.99 pp [-5.00, +2.12] | not distinguishable from zero |

## The one cost check (descriptive: every year has been seen; not evidence)

Fixed before any result: inside the window, the sign of p_up - 0.5 at a bar close, held one hour, paying the real spread (floor 0.0062 %; stress 0.0122 %). Days resampled in blocks of 20. A simulation of past bars, not a trade.

| model | trades | mean gross / cost | mean net per trade [95 %] | winning share [95 %] | cumulative net [95 %] | cumulative net, stress cost |
|---|---|---|---|---|---|---|
| `h1_gbm` | 15,797 | +0.0010 % / 0.0154 % | -0.0144 % [-0.0189, -0.0105] | 46.4 % [45.5, 47.2] | -227.6 % [-298.4, -165.3] | -247.4 % |
| `h1_gbm_accel` | 15,797 | -0.0016 % / 0.0154 % | -0.0171 % [-0.0215, -0.0132] | 46.4 % [45.6, 47.3] | -269.6 % [-340.0, -209.3] | -289.4 % |
| `h1_lstm` | 15,797 | -0.0016 % / 0.0154 % | -0.0170 % [-0.0216, -0.0124] | 46.3 % [45.3, 47.3] | -269.1 % [-340.7, -196.4] | -289.0 % |
| `h1_lstm_accel` | 15,797 | -0.0039 % / 0.0154 % | -0.0193 % [-0.0236, -0.0150] | 45.8 % [44.9, 46.8] | -305.1 % [-372.4, -237.7] | -325.0 % |
| `h1_accel` | 15,797 | -0.0012 % / 0.0154 % | -0.0167 % [-0.0201, -0.0133] | 45.0 % [44.1, 45.9] | -263.1 % [-317.9, -210.1] | -283.0 % |
| `h1_gbm_accel_m15` | 5,102 | +0.0006 % / 0.0083 % | -0.0078 % [-0.0161, +0.0002] | 48.2 % [47.0, 49.6] | -39.6 % [-81.9, +1.2] | -59.4 % |
| `h1_accel_m15` | 5,102 | -0.0039 % / 0.0083 % | -0.0123 % [-0.0208, -0.0045] | 47.8 % [46.2, 49.2] | -62.5 % [-106.5, -23.1] | -82.3 % |

## Reading (descriptive: every year has been seen; not evidence)

- 10 with-minus-without intervals are reported here (plus the per-year ones), so one excluding zero is expected by chance even with no real effect.
- A hit rate is a share of next-hour directions called right; 50 % is a coin. A difference that is not distinguishable from zero means acceleration did not change how often the direction was called.
- Nothing here was searched: the features, models, views and the single cost rule were fixed in the design before any result.
