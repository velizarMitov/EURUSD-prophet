# fx-ny-volatility labels and power

> **Gross of spread and slippage — not a trading result.**

Target: y = log((mid high − mid low) / mid close14) over the bars labelled 15:00–21:00 Berlin (16:00–23:00 Sofia); features at the close of the bar labelled 14:00. Out of sample from 2013-01-01. No model output is used.

## Days

| Pair | Weekdays | Eligible | Eligible from 2013 | Mean y | SD y | Mean y_bid | corr(y, y_bid) |
|---|---|---|---|---|---|---|---|
| EURUSD | 3,875 | 3,692 | 3,475 | -5.5413 | 0.5075 | -5.5413 | 1.0000 |
| GBPUSD | 3,875 | 3,692 | 3,475 | -5.4872 | 0.4846 | -5.4871 | 1.0000 |

## Exclusions (one reason per day, shared by both pairs)

| Reason | Days |
|---|---|
| incomplete session | 42 |
| unverified clock week | 116 |
| missing pre-session bar | 4 |
| corrupt spread | 1 |
| warm-up | 20 |

## Eligible days per year

| Year | Eligible |
|---|---|
| 2012 | 217 |
| 2013 | 235 |
| 2014 | 227 |
| 2015 | 238 |
| 2016 | 240 |
| 2017 | 250 |
| 2018 | 255 |
| 2019 | 259 |
| 2020 | 260 |
| 2021 | 260 |
| 2022 | 255 |
| 2023 | 250 |
| 2024 | 256 |
| 2025 | 259 |
| 2026 | 231 |

## Power — before any model

One-sided alpha 0.0125 (0.05/4), power 80%. The detectable mean of the per-day squared-error difference, in units of that difference's own standard deviation, is (z(1−α) + z(power)) / √n. Five-day blocks barely change it for a same-day target.

| Pair | Out-of-sample days | Detectable difference (σ units) |
|---|---|---|
| EURUSD | 3,475 | 0.0523 |
| GBPUSD | 3,475 | 0.0523 |

A KEEP of this size can be practically tiny; the report gives the relative MSE reduction next to it.

_Generated 2026-11-22T13:38:01+00:00._
