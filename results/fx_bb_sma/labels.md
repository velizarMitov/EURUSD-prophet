# fx-bb-sma-exit labels and power

> **Net of the recorded bar spread only — slippage, swap and commission are not included; not a trading result.**

Trades: entry at the D1 mid close of a first breach of the 20-day, 2-SD Bollinger Bands; take profit at SMA20[t], stop at the same distance beyond the entry, time stop at the close of valid day t+10; one open trade per pair. Exits are scanned hour by hour on H1 bars. This report reads each trade's exit TIME only (for the one-trade rule): no exit reason, exit price or result is read or shown.

## Trades

| Pair | Start | Valid D1 days | First breaches | Skipped (trade open) | Trades | Pair gap | Corrupt spread | Open at data end | Eligible | Distinct dates |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | 2012-01-01 | 3,861 | 241 | 54 | 187 | 0 | 0 | 0 | 187 | 187 |
| GBPUSD | 2012-01-01 | 3,840 | 241 | 53 | 188 | 1 | 0 | 1 | 186 | 186 |
| USDJPY | 2012-01-01 | 3,860 | 238 | 47 | 191 | 0 | 0 | 1 | 190 | 190 |
| AUDUSD | 2012-01-01 | 3,860 | 242 | 45 | 197 | 0 | 0 | 0 | 197 | 197 |
| USDCAD | 2012-01-01 | 3,855 | 227 | 42 | 185 | 1 | 0 | 1 | 183 | 183 |
| USDCHF | 2015-02-01 | 3,062 | 180 | 34 | 146 | 0 | 0 | 0 | 146 | 146 |
| pooled | — | 22,338 | 1369 | 275 | 1094 | 2 | 0 | 3 | 1089 | 821 |

## Eligible trades per year

| Year | EURUSD | GBPUSD | USDJPY | AUDUSD | USDCAD | USDCHF | pooled | distinct dates |
|---|---|---|---|---|---|---|---|---|
| 2012 | 8 | 8 | 15 | 13 | 9 | 0 | 53 | 50 |
| 2013 | 11 | 12 | 14 | 12 | 13 | 0 | 62 | 53 |
| 2014 | 15 | 12 | 12 | 15 | 12 | 0 | 66 | 55 |
| 2015 | 11 | 14 | 12 | 14 | 12 | 12 | 75 | 52 |
| 2016 | 11 | 10 | 13 | 15 | 12 | 11 | 72 | 56 |
| 2017 | 15 | 13 | 10 | 14 | 13 | 13 | 78 | 62 |
| 2018 | 11 | 13 | 12 | 13 | 11 | 17 | 77 | 61 |
| 2019 | 13 | 14 | 12 | 15 | 13 | 13 | 80 | 60 |
| 2020 | 11 | 13 | 11 | 14 | 9 | 9 | 67 | 47 |
| 2021 | 13 | 11 | 18 | 15 | 15 | 13 | 85 | 65 |
| 2022 | 15 | 10 | 14 | 9 | 11 | 12 | 71 | 52 |
| 2023 | 14 | 14 | 14 | 13 | 15 | 13 | 83 | 60 |
| 2024 | 12 | 13 | 13 | 13 | 13 | 11 | 75 | 50 |
| 2025 | 13 | 18 | 11 | 10 | 13 | 10 | 75 | 50 |
| 2026 | 14 | 11 | 9 | 12 | 12 | 12 | 70 | 48 |

## Power — before any model

One-sided alpha 0.025 (0.05/2), power 80%, z(1−α) + z(power) = 2.8016. Bootstrap blocks of 10 business days (a trade lives up to 10), so the effective size is nearer the date count than the trade count.

**Cell 1** (mean net R > 0, every eligible trade): minimum detectable mean = 2.80 · 1 R / √n. The SD of 1.0 R is declared, not estimated: a take profit is about +1 R and a stop about −1 R gross.

| Pair | Eligible trades | MDE on trades (R) | Distinct dates | MDE on dates (R) |
|---|---|---|---|---|
| EURUSD | 187 | 0.205 | 187 | 0.205 |
| GBPUSD | 186 | 0.205 | 186 | 0.205 |
| USDJPY | 190 | 0.203 | 190 | 0.203 |
| AUDUSD | 197 | 0.200 | 197 | 0.200 |
| USDCAD | 183 | 0.207 | 183 | 0.207 |
| USDCHF | 146 | 0.232 | 146 | 0.232 |
| pooled | 1089 | 0.085 | 821 | 0.098 |

**Cell 2** (pooled LightGBM vs the base rate, from 2015-01-01): minimum detectable accuracy = 0.5 + 2.80 · 0.5 / √n. Only the pooled row is a registered test.

| Pair | Eligible trades | MDE on trades | Distinct dates | MDE on dates |
|---|---|---|---|---|
| EURUSD | 153 | 61.32% | 153 | 61.32% |
| GBPUSD | 154 | 61.29% | 154 | 61.29% |
| USDJPY | 149 | 61.48% | 149 | 61.48% |
| AUDUSD | 157 | 61.18% | 157 | 61.18% |
| USDCAD | 149 | 61.48% | 149 | 61.48% |
| USDCHF | 146 | 61.59% | 146 | 61.59% |
| pooled | 908 | 54.65% | 663 | 55.44% |

Only an effect of about this size can be seen. A DROP means "no effect of this size", not "no effect".

_Generated 2026-11-22T13:24:30+00:00._
