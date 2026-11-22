# fx-ny-orb-hot labels and power

> **Net of the recorded spread plus 0.4 bps slippage (no swap: intraday) — not a trading result.**

NY opening range = M15 bars 14:00, 14:15, 14:30, 14:45 Berlin; first M15 close outside it from 15:00; entry at the next open before 19:00; range ≤ 0.5 F, breakout ≤ 0.25 F; SL 0.25 F, TP 0.6 F, forced exit 20:45. Window 2018-08-08 .. 2026-08-07. This report runs the rule only up to the entry decision: no exit, later price or result is read.

## Days and entry decisions

| Pair | Side | Days | Simulable | Trades | Dates | No range | Too wide | No breakout | Too far | Too late | No M15 | Corrupt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | hot | 230 | 230 | 84 | 84 | 0 | 142 | 0 | 4 | 0 | 0 | 0 |
| EURUSD | cold | 1828 | 1828 | 986 | 986 | 0 | 718 | 13 | 99 | 11 | 0 | 0 |
| GBPUSD | hot | 185 | 185 | 86 | 86 | 0 | 93 | 1 | 3 | 2 | 0 | 0 |
| GBPUSD | cold | 1873 | 1873 | 965 | 965 | 0 | 788 | 6 | 103 | 10 | 0 | 0 |
| pooled | hot | 415 | 415 | 170 | 121 | 0 | 235 | 1 | 7 | 2 | 0 | 0 |
| pooled | cold | 3701 | 3701 | 1951 | 1290 | 0 | 1506 | 19 | 202 | 21 | 0 | 0 |

## Hot-day trades per year

| Year | EURUSD | GBPUSD | pooled |
|---|---|---|---|
| 2018 | 3 | 3 | 6 |
| 2019 | 7 | 8 | 15 |
| 2020 | 19 | 18 | 37 |
| 2021 | 12 | 9 | 21 |
| 2022 | 9 | 19 | 28 |
| 2023 | 6 | 5 | 11 |
| 2024 | 10 | 9 | 19 |
| 2025 | 8 | 7 | 15 |
| 2026 | 10 | 8 | 18 |

## Power — before any outcome

One-sided alpha 0.025 (0.05/2), power 80%, z(1−α) + z(power) = 2.8016, declared SD 1.5 R (a stop is −1 R and a target +2.4 R; forced exits shrink the spread).

| Cell | Trades | Dates | MDE on trades (R) | MDE on dates (R) |
|---|---|---|---|---|
| orb_hot_net_r (hot, pooled) | 170 | 121 | 0.322 | 0.382 |
| non-hot control (pooled) | 1951 | 1290 | 0.095 | 0.117 |

The difference cell is limited by the hot side. A DROP means "no effect of this size", not "no effect".

_Generated 2026-11-22T15:04:52+00:00._
