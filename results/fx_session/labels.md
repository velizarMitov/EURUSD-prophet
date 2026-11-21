# fx-session labels and power

> **Gross of spread and slippage — not a trading result.**

Target: sign of the mid close of the bar labelled 21:00 Berlin minus the mid close of the bar labelled 14:00 Berlin, same day. Ties dropped. Evaluation from 2013-01-01.

| Pair | Weekdays | Usable days | Usable from first fold | Ties | Up-rate (mid) | Up-rate (bid) | Gap (pp) | DST-mismatch days |
|---|---|---|---|---|---|---|---|---|
| EURUSD | 3,875 | 3,692 | 3,470 | 6 | 50.16% | 50.16% | 0.00 | 175 |
| GBPUSD | 3,875 | 3,693 | 3,471 | 5 | 50.61% | 50.58% | 0.03 | 175 |

A bid-versus-mid gap above 1.0 pp would be flagged ⚠ as a possible spread artifact.

## Up-rate per year (usable days)

| Year | EURUSD days | EURUSD up | GBPUSD days | GBPUSD up |
|---|---|---|---|---|
| 2012 | 222 | 55.9% | 222 | 50.5% |
| 2013 | 235 | 60.9% | 235 | 54.9% |
| 2014 | 228 | 44.3% | 226 | 46.5% |
| 2015 | 238 | 52.1% | 238 | 54.6% |
| 2016 | 240 | 47.9% | 240 | 49.2% |
| 2017 | 250 | 54.0% | 248 | 56.0% |
| 2018 | 254 | 46.5% | 255 | 46.3% |
| 2019 | 258 | 47.7% | 259 | 49.0% |
| 2020 | 260 | 49.2% | 260 | 52.7% |
| 2021 | 258 | 50.8% | 259 | 51.0% |
| 2022 | 254 | 45.7% | 255 | 48.6% |
| 2023 | 250 | 51.2% | 250 | 54.4% |
| 2024 | 256 | 49.6% | 256 | 48.4% |
| 2025 | 258 | 52.7% | 259 | 52.1% |
| 2026 | 231 | 44.6% | 231 | 44.6% |

## Excluded days

| Pair | Reason | Days |
|---|---|---|
| EURUSD | unverified clock week | 116 |
| EURUSD | incomplete session | 42 |
| EURUSD | no previous day | 9 |
| EURUSD | warm-up | 6 |
| EURUSD | tie | 6 |
| EURUSD | missing pre-session bar | 4 |
| GBPUSD | unverified clock week | 116 |
| GBPUSD | incomplete session | 42 |
| GBPUSD | no previous day | 9 |
| GBPUSD | warm-up | 6 |
| GBPUSD | tie | 5 |
| GBPUSD | missing pre-session bar | 4 |

## Power — computed from day counts only, before any model

One-sided alpha 0.0125 (0.05/4), power 80%. The minimum detectable accuracy is `0.5 + (z(1−α) + z(power)) · 0.5 / √n`; the log-loss gain is `ln 2 − H(MDE)`.

| Pair | Out-of-sample days | Minimum detectable accuracy | Matching log-loss gain |
|---|---|---|---|
| EURUSD | 3,470 | 52.62% | 0.00137 |
| GBPUSD | 3,471 | 52.62% | 0.00137 |

_Generated 2026-11-21T14:45:03+00:00._
