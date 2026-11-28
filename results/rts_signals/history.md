# RTS signals — history / история

> **descriptive — no verdict.** The results below were seen before this check was built; they cannot decide anything. The decision is the forward cell `rts_spot_net_r` (see report.md). / Резултатите са видени преди проверката; само тестът напред решава.

> Net of the recorded MT5 bar spread only; slippage, swap and commission are not modelled — not a trading result.

_Generated 2026-11-28 12:20 UTC_

## 1. RTS's own spot track record on our prices / Отчетът на RTS върху нашите цени

105 closed trades 2026-11-02 .. 2026-11-26 (latest version of each line). `spread` = the broker's recorded spread at the minute our prices first trade the entry.

| pair | trades | RTS points | spread (pips) | RTS points − spread | confirmed | different exit | not reproducible |
|---|---:|---:|---:|---:|---:|---:|---:|
| GBPUSD | 18 | +82 | 61.4 | +20.6 | 17 | 1 | 0 |
| EURUSD | 19 | +107 | 48.1 | +58.9 | 18 | 1 | 0 |
| AUDUSD | 17 | -36 | 41.8 | -77.8 | 17 | 0 | 0 |
| USDJPY | 18 | +29 | 53.6 | -24.6 | 17 | 1 | 0 |
| USDCHF | 16 | +57 | 91.7 | -34.7 | 16 | 0 | 0 |
| USDCAD | 17 | -88 | 68.1 | -156.1 | 16 | 1 | 0 |
| **total** | 105 | +151 | 364.7 | -213.7 | 101 | 4 | 0 |

- `confirmed`: after the entry, our prices trade RTS's exit price (±3 points) by its exit date, and before the stop when the rule fixes it. It shows the trade was possible on our prices, not that every RTS decision is reproduced.
- RTS enters Rule 0 trades at 18:00 New York, the broker's daily rollover, when the spread is widest. That is why the spread column is large.

Not confirmed:

| entry date | pair | side | entry | rule | exit | RTS points | label | note |
|---|---|---|---:|---:|---:|---:|---|---|
| 2026-11-02 | GBPUSD | buy | 1.3382 | 0 | 1.3454 | 72 | different exit | exit price never traded after the entry |
| 2026-11-18 | USDJPY | sell | 159.34 | 0 | 159.05 | 29 | different exit | exit price never traded after the entry |
| 2026-11-12 | USDCAD | buy | 1.4249 | 0 | 1.4285 | 36 | different exit | exit price never traded after the entry |
| 2026-11-03 | EURUSD | buy | 1.1184 | 1 | 1.1102 | -82 | different exit | exit price never traded after the entry |

## 2. Base trades of the received reports / Базови сделки от получените отчети

Reports 2026-11-20, 2026-11-23, 2026-11-24, 2026-11-25, 2026-11-27. Status counts: entry not reached 4, filled 18, npr 8.

| pair | filled | target | stop | time | gross pips | net pips | gross R | net R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GBPUSD | 3 | 1 | 2 | 0 | -31.7 | -34.8 | -1.19 | -1.26 |
| EURUSD | 3 | 1 | 0 | 2 | +46.1 | +44.3 | +1.15 | +1.10 |
| AUDUSD | 3 | 0 | 1 | 2 | -15.1 | -17.2 | -0.49 | -0.55 |
| USDJPY | 3 | 1 | 1 | 1 | +25.8 | +22.1 | +0.65 | +0.55 |
| USDCHF | 3 | 1 | 2 | 0 | -35.2 | -40.4 | -0.75 | -0.93 |
| USDCAD | 3 | 2 | 1 | 0 | +33.8 | +10.7 | +0.57 | -0.10 |
| **total** | 18 | 6 | 7 | 5 | +23.7 | -15.3 | -0.06 | -1.20 |

Too few trades to mean anything. descriptive — no verdict.
