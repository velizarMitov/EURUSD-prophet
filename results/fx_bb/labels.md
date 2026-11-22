# fx-bb-reversion labels and power

> **Gross of spread and slippage — not a trading result.**

Events: the first D1 mid close outside the 20-day, 2-SD Bollinger Bands (D1 closes at 17:00 New York), counter-trend direction, 3-day cooldown per pair. Outcome over 3 valid days in ATR(20) units. This report reads no event outcome and no model output.

## D1 days and breaches

| Pair | Start | Valid D1 days | Closes outside | First breaches | Removed by cooldown | Accepted events |
|---|---|---|---|---|---|---|
| EURUSD | 2012-01-01 | 3,861 | 469 | 241 | 17 | 224 |
| GBPUSD | 2012-01-01 | 3,840 | 469 | 241 | 19 | 222 |
| USDJPY | 2012-01-01 | 3,860 | 471 | 238 | 17 | 221 |
| AUDUSD | 2012-01-01 | 3,860 | 440 | 242 | 25 | 217 |
| USDCAD | 2012-01-01 | 3,855 | 449 | 227 | 13 | 214 |
| USDCHF | 2015-02-01 | 3,062 | 355 | 180 | 13 | 167 |
| pooled | — | 22,338 | 2653 | 1369 | 104 | 1265 |

## Cell 1 — every event from each pair's start

| Pair | Accepted | Pair gap | No outcome yet | Eligible | Ties | Distinct dates |
|---|---|---|---|---|---|---|
| EURUSD | 224 | 0 | 0 | 224 | 0 | 224 |
| GBPUSD | 222 | 1 | 1 | 220 | 0 | 220 |
| USDJPY | 221 | 0 | 0 | 221 | 0 | 221 |
| AUDUSD | 217 | 0 | 0 | 217 | 0 | 217 |
| USDCAD | 214 | 1 | 0 | 213 | 0 | 213 |
| USDCHF | 167 | 0 | 0 | 167 | 1 | 167 |
| pooled | 1265 | 2 | 1 | 1262 | 1 | 928 |

## Cell 2 — out-of-sample events from 2015-01-01

| Pair | Accepted | Pair gap | No outcome yet | Eligible | Ties | Distinct dates |
|---|---|---|---|---|---|---|
| EURUSD | 186 | 0 | 0 | 186 | 0 | 186 |
| GBPUSD | 180 | 0 | 1 | 179 | 0 | 179 |
| USDJPY | 172 | 0 | 0 | 172 | 0 | 172 |
| AUDUSD | 173 | 0 | 0 | 173 | 0 | 173 |
| USDCAD | 175 | 0 | 0 | 175 | 0 | 175 |
| USDCHF | 167 | 0 | 0 | 167 | 1 | 167 |
| pooled | 1053 | 0 | 1 | 1052 | 1 | 751 |

## Eligible events per year

| Year | EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD | USDCHF | pooled | distinct dates |
|---|---|---|---|---|---|---|---|---|
| 2012 | 9 | 10 | 18 | 15 | 10 | 0 | 62 | 57 |
| 2013 | 12 | 16 | 18 | 14 | 13 | 0 | 73 | 58 |
| 2014 | 17 | 15 | 13 | 15 | 15 | 0 | 75 | 62 |
| 2015 | 15 | 15 | 13 | 16 | 15 | 13 | 87 | 61 |
| 2016 | 13 | 11 | 14 | 16 | 15 | 13 | 82 | 66 |
| 2017 | 19 | 16 | 13 | 18 | 15 | 14 | 95 | 71 |
| 2018 | 15 | 17 | 15 | 13 | 14 | 18 | 92 | 70 |
| 2019 | 15 | 16 | 12 | 15 | 13 | 15 | 86 | 66 |
| 2020 | 13 | 13 | 12 | 17 | 13 | 12 | 80 | 55 |
| 2021 | 16 | 15 | 21 | 15 | 15 | 14 | 96 | 70 |
| 2022 | 15 | 11 | 15 | 10 | 12 | 13 | 76 | 54 |
| 2023 | 18 | 16 | 17 | 16 | 17 | 13 | 97 | 68 |
| 2024 | 16 | 17 | 16 | 14 | 15 | 15 | 93 | 60 |
| 2025 | 15 | 19 | 13 | 11 | 15 | 13 | 86 | 55 |
| 2026 | 16 | 13 | 11 | 12 | 16 | 14 | 82 | 55 |

## Events per date (eligible, pooled)

| Events on the date | Dates overall | Dates from 2015-01-01 |
|---|---|---|
| 1 | 689 | 540 |
| 2 | 161 | 137 |
| 3 | 66 | 63 |
| 4 | 8 | 7 |
| 5 | 3 | 3 |
| 6 | 1 | 1 |

## Power — before any model

One-sided alpha 0.025 (0.05/2), power 80%, z(1−α) + z(power) = 2.8016.

**Cell 1** (mean outcome > 0): minimum detectable mean = 2.80 · sd / √n, with sd the standard deviation of (close[t+3] − close[t]) / ATR20[t] over ALL valid days, never over events.

| Pair | sd, all days | sd, all days from 2015-01-01 | Eligible events | MDE on events (ATR) | Distinct dates | MDE on dates (ATR) |
|---|---|---|---|---|---|---|
| EURUSD | 1.131 | 1.142 | 224 | 0.212 | 224 | 0.212 |
| GBPUSD | 1.169 | 1.174 | 220 | 0.221 | 220 | 0.221 |
| USDJPY | 1.240 | 1.242 | 221 | 0.234 | 221 | 0.234 |
| AUDUSD | 1.090 | 1.096 | 217 | 0.207 | 217 | 0.207 |
| USDCAD | 1.116 | 1.127 | 213 | 0.214 | 213 | 0.214 |
| USDCHF | 1.156 | 1.156 | 167 | 0.251 | 167 | 0.251 |
| pooled | 1.152 | 1.157 | 1262 | 0.091 | 928 | 0.106 |

**Cell 2** (pooled LightGBM vs the base rate, from 2015-01-01): minimum detectable accuracy = 0.5 + 2.80 · 0.5 / √n, on the non-tie events and on their distinct dates.

| Pair | Events with a target | MDE on events | Distinct dates | MDE on dates |
|---|---|---|---|---|
| EURUSD | 186 | 60.27% | 186 | 60.27% |
| GBPUSD | 179 | 60.47% | 179 | 60.47% |
| USDJPY | 172 | 60.68% | 172 | 60.68% |
| AUDUSD | 173 | 60.65% | 173 | 60.65% |
| USDCAD | 175 | 60.59% | 175 | 60.59% |
| USDCHF | 166 | 60.87% | 166 | 60.87% |
| pooled | 1051 | 54.32% | 751 | 55.11% |

Cell 2 is pooled, so only the pooled row is a registered test; the per-pair rows show scale. Only an effect of about this size can be seen. A DROP means "no effect of this size", not "no effect".

_Generated 2026-11-22T08:15:09+00:00._
