# Gold window backtest

Generated 2026-11-04 10:25 UTC by `python -m src.gold_window.backtest`.

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

- Every trade opens and closes inside 15:30–23:00 Sofia (labels 14:30–22:00 Berlin); nothing is held overnight, so no swap applies.
- Prices are MT5 bid closes. A long entry pays its entry bar's MT5 spread, a short exit pays its exit bar's; every round trip costs at least 0.0062 % of the entry price (stress: 0.0122 %).
- Models are refit each calendar year on the past only (walk-forward); gates compare a forecast with its own trailing 250 trading days.
- Inputs: H1 cache to 2026-11-04 10:00, MT5 M15 from 2022-08-12, walk-forward predictions h1_lstm 2018 on, d1_lstm and d1_garch 2016 on.

## Summary

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

| rule | span | days | trades | mean net %/day | 95 % interval | total net % | total stress % | max drawdown % | hit % | gate always open: total % | always long 16–23: total % | placebo share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `h1_gated_direction` | 2019-01-02 .. 2026-11-03 | 1960 | 3349 | -0.0433 | [-0.0713, -0.0189] | -58.3 | -60.5 | -59.0 | 47.6 | -89.2 | -23.7 | 0.99 |
| `continuation` | 2016-12-21 .. 2026-11-03 | 2467 | 1095 | +0.0093 | [-0.0074, +0.0280] | +21.9 | +19.7 | -14.8 | 50.2 | +38.4 | -31.3 | 0.31 |
| `us_open_breakout` | 2022-08-12 .. 2026-11-03 | 1056 | 534 | -0.0059 | [-0.0378, +0.0251] | -7.2 | -8.7 | -16.0 | 47.6 | -0.8 | -12.4 | 0.75 |
| `d1_gated_direction` | 2016-12-30 .. 2026-11-03 | 2461 | 1312 | -0.0153 | [-0.0357, +0.0047] | -33.6 | -35.1 | -41.6 | 49.7 | -44.3 | -31.8 | 0.72 |

The placebo share is the fraction of time-shifted gates (the same gate decisions moved along the candidate trades) whose mean daily net is at least the real gate's: near 0 means the timing of the gate mattered in these years, near 0.5 or above means it did not.

## `h1_gated_direction`

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

Gross (before cost) total -29.8 %, mean -0.0169 %/day. Gate always open: 13720 trades, total -89.2 %, mean -0.1110 %/day.

| year | total net % | days | trades |
|---|---|---|---|
| 2019 | -14.8 | 250 | 519 |
| 2020 | -30.5 | 251 | 524 |
| 2021 | -6.6 | 250 | 187 |
| 2022 | -2.1 | 250 | 504 |
| 2023 | -5.8 | 249 | 193 |
| 2024 | -12.2 | 250 | 499 |
| 2025 | +0.5 | 249 | 485 |
| 2026 | -7.1 | 211 | 438 |

## `continuation`

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

Gross (before cost) total +43.3 %, mean +0.0159 %/day. Gate always open: 2466 trades, total +38.4 %, mean +0.0152 %/day.

| year | total net % | days | trades |
|---|---|---|---|
| 2016 | +0.0 | 7 | 0 |
| 2017 | -4.4 | 250 | 23 |
| 2018 | -5.6 | 250 | 42 |
| 2019 | -3.5 | 250 | 126 |
| 2020 | +9.2 | 251 | 195 |
| 2021 | -2.2 | 250 | 10 |
| 2022 | +4.8 | 250 | 179 |
| 2023 | +4.8 | 249 | 88 |
| 2024 | +9.9 | 250 | 151 |
| 2025 | -2.9 | 249 | 128 |
| 2026 | +11.8 | 211 | 153 |

## `us_open_breakout`

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

Gross (before cost) total -2.3 %, mean -0.0010 %/day. Gate always open: 948 trades, total -0.8 %, mean +0.0008 %/day.

| year | total net % | days | trades |
|---|---|---|---|
| 2022 | +1.5 | 97 | 63 |
| 2023 | -7.9 | 249 | 79 |
| 2024 | -0.2 | 250 | 136 |
| 2025 | +0.8 | 249 | 117 |
| 2026 | -1.3 | 211 | 139 |

## `d1_gated_direction`

_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. Not evidence of an edge._

Gross (before cost) total -18.9 %, mean -0.0072 %/day. Gate always open: 2461 trades, total -44.3 %, mean -0.0217 %/day.

| year | total net % | days | trades |
|---|---|---|---|
| 2016 | -0.6 | 1 | 1 |
| 2017 | -6.4 | 250 | 151 |
| 2018 | -7.6 | 250 | 116 |
| 2019 | -6.9 | 250 | 97 |
| 2020 | -14.4 | 251 | 218 |
| 2021 | -3.7 | 250 | 57 |
| 2022 | -8.3 | 250 | 131 |
| 2023 | +7.7 | 249 | 108 |
| 2024 | +1.8 | 250 | 141 |
| 2025 | +7.9 | 249 | 169 |
| 2026 | -7.2 | 211 | 123 |
