# fx-rates labels and power

> **Gross of spread and slippage — not a trading result.**

Target: sign of the D1 mid close 5 FX trading days after `t` minus the mid close of `t`. Yields enter as of the previous FX trading day (fill limit 3 business days). Evaluation from 2015-01-01.

## Sources

| Series | First | Last | Rows |
|---|---|---|---|
| us2y | 2005-01-03 | 2026-11-19 | 5,475 |
| us10y | 2005-01-03 | 2026-11-19 | 5,475 |
| de2y | 2005-01-03 | 2026-11-20 | 5,562 |
| uk2y | 2005-01-04 | 2026-11-19 | 5,530 |

## Days

| Pair | D1 days | Usable | Usable from first fold | Ties | Up-rate | Stale-filled usable days | Excluded as stale yields |
|---|---|---|---|---|---|---|---|
| EURUSD | 3,861 | 3,583 | 3,078 | 0 | 47.92% | 231 | 0 |
| GBPUSD | 3,840 | 3,561 | 3,077 | 1 | 48.92% | 231 | 0 |

## Up-rate per year (usable days)

| Year | EURUSD days | EURUSD up | GBPUSD days | GBPUSD up |
|---|---|---|---|---|
| 2013 | 246 | 52.4% | 246 | 55.3% |
| 2014 | 259 | 37.5% | 238 | 41.2% |
| 2015 | 259 | 44.0% | 259 | 49.0% |
| 2016 | 259 | 46.7% | 259 | 48.3% |
| 2017 | 259 | 59.8% | 258 | 51.9% |
| 2018 | 259 | 43.6% | 259 | 43.2% |
| 2019 | 259 | 48.6% | 259 | 49.8% |
| 2020 | 260 | 57.7% | 260 | 54.6% |
| 2021 | 260 | 42.3% | 260 | 50.4% |
| 2022 | 259 | 44.4% | 259 | 42.9% |
| 2023 | 259 | 50.6% | 259 | 51.7% |
| 2024 | 260 | 45.0% | 260 | 44.6% |
| 2025 | 259 | 57.5% | 259 | 55.2% |
| 2026 | 226 | 39.8% | 226 | 46.0% |

## Excluded days

| Pair | Reason | Days |
|---|---|---|
| EURUSD | warm-up | 272 |
| EURUSD | no target yet | 5 |
| EURUSD | no lagged date | 1 |
| GBPUSD | warm-up | 272 |
| GBPUSD | no target yet | 5 |
| GBPUSD | no lagged date | 1 |
| GBPUSD | tie | 1 |

## Power — from day counts and the target's own overlap, before any model

One-sided alpha 0.0125 (0.05/4), power 80%. Overlapping 5-day targets are dependent: VIF = 1 + 2·Σρ_k over the target sign autocorrelation at lags 1–4 on training-era days (before 2015-01-01). Minimum detectable accuracy = 0.5 + (z(1−α) + z(power)) · 0.5 / √n.

| Pair | Out-of-sample days | ρ lags 1–4 | VIF | Effective n | MDE on effective n | Independent weeks | MDE on weeks | Log-loss gain at MDE |
|---|---|---|---|---|---|---|---|---|
| EURUSD | 3,078 | 0.69, 0.49, 0.31, 0.18 | 4.36 | 706 | 55.80% | 615 | 56.22% | 0.00675 |
| GBPUSD | 3,077 | 0.65, 0.52, 0.33, 0.20 | 4.38 | 702 | 55.82% | 615 | 56.22% | 0.00678 |

Only an edge of about this size can be seen. A DROP means "no edge of this size", not "no edge".

_Generated 2026-11-21T21:03:03+00:00._
