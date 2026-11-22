# fx-ny-orb-hot report

> **Net of the recorded spread plus 0.4 bps slippage (no swap: intraday) — not a trading result.**

New York opening-range breakout (M15 bars 14:00–14:45 Berlin), taken on days the walk-forward HAR-X calls hot; non-hot days run the same rule as the control.

## Verdicts

| Cell | Statistic | Hot trades | Non-hot trades | Value | Bound (α-quantile) | mde_80 | Verdict |
|---|---|---|---|---|---|---|---|
| orb_hot_net_r | mean net R, hot days | 170 | 1951 | -0.1645 R | -0.3773 | 0.3129 | **DROP** |
| orb_hot_minus_cold | mean net R, hot minus non-hot | 170 | 1951 | -0.1330 R | -0.3594 | 0.3272 | **DROP** |

## Forward HAR-X check

Running: scored once after 270 settled forward days per pair or on 2028-01-31.

## Power (labels.md, before any outcome)

| Side | Trades | Dates | MDE on trades | MDE on dates |
|---|---|---|---|---|
| hot | 170 | 121 | 0.322 R | 0.382 R |
| cold | 1951 | 1290 | 0.095 R | 0.117 R |

## Trades (descriptive)

Win = net R > 0. Wilson intervals count distinct dates. A bar touching both SL and TP counts as SL (conflicts).

| Group | Trades | TP | SL | Forced | Win (net) | Wilson 95 % (dates) | Mean net R | Pips gross / net | PF gross / net | Conflicts |
|---|---|---|---|---|---|---|---|---|---|---|
| hot, pooled | 170 | 19% | 66% | 15% | 30.0% | 22.6%–38.7% | -0.164 | -1.4 / -2.5 | 0.895 / 0.817 | 1 |
| non-hot, pooled | 1951 | 24% | 59% | 17% | 35.7% | 33.1%–38.3% | -0.031 | +0.8 / -0.4 | 1.113 / 0.950 | 2 |
| hot, EURUSD | 84 | 15% | 65% | 19% | 28.6% | 20.0%–39.0% | -0.219 | -3.5 / -4.4 | 0.706 / 0.645 | 0 |
| non-hot, EURUSD | 986 | 24% | 59% | 17% | 35.6% | 32.7%–38.6% | -0.019 | +0.9 / -0.1 | 1.145 / 0.983 | 0 |
| hot, GBPUSD | 86 | 22% | 66% | 12% | 31.4% | 22.6%–41.8% | -0.111 | +0.7 / -0.7 | 1.048 / 0.954 | 1 |
| non-hot, GBPUSD | 965 | 24% | 60% | 16% | 35.8% | 32.8%–38.8% | -0.044 | +0.7 / -0.7 | 1.089 / 0.925 | 2 |
| hot, long | 95 | 20% | 64% | 16% | 32.6% | 22.9%–44.1% | -0.101 | +0.6 / -0.6 | 1.050 / 0.958 | 1 |
| hot, short | 75 | 17% | 68% | 15% | 26.7% | 17.1%–39.1% | -0.245 | -3.9 / -5.0 | 0.720 / 0.656 | 0 |
| hot, DST-mismatch days | 32 | 25% | 69% | 6% | 28.1% | 14.2%–48.1% | -0.130 | +2.5 / +1.3 | 1.173 / 1.087 | 0 |
| hot, 2018 | 6 | 0% | 100% | 0% | 0.0% | 0.0%–56.1% | -1.056 | -20.4 / -21.6 | 0.000 / 0.000 | 0 |
| hot, 2019 | 15 | 33% | 53% | 13% | 40.0% | 16.8%–68.7% | +0.261 | +2.7 / +1.5 | 1.257 / 1.139 | 0 |
| hot, 2020 | 37 | 16% | 65% | 19% | 32.4% | 18.5%–50.3% | -0.196 | +2.1 / +1.0 | 1.152 / 1.066 | 0 |
| hot, 2021 | 21 | 19% | 62% | 19% | 28.6% | 11.3%–55.6% | -0.154 | -0.9 / -2.1 | 0.909 / 0.813 | 1 |
| hot, 2022 | 28 | 32% | 64% | 4% | 32.1% | 16.3%–53.5% | +0.066 | +2.7 / +1.5 | 1.187 / 1.100 | 0 |
| hot, 2023 | 11 | 18% | 64% | 18% | 36.4% | 12.2%–70.2% | -0.063 | +0.2 / -0.9 | 1.017 / 0.927 | 0 |
| hot, 2024 | 19 | 11% | 79% | 11% | 15.8% | 4.7%–41.6% | -0.600 | -9.8 / -10.9 | 0.296 / 0.263 | 0 |
| hot, 2025 | 15 | 7% | 67% | 27% | 33.3% | 12.1%–64.6% | -0.226 | -4.7 / -6.0 | 0.657 / 0.590 | 0 |
| hot, 2026 | 18 | 17% | 61% | 22% | 33.3% | 14.8%–59.1% | -0.079 | -1.6 / -2.8 | 0.851 / 0.755 | 0 |

### Five example hot-day trades (fixed rule: latest TP, SL, forced exit, then latest long/short)

| Pair | Dir | Entry (Berlin label) | Entry | SL | TP | Exit (label) | Reason | Pips gross / net | Net R |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD | long | 2026-07-29 15:30:00+00:00 | 1.13854 | 1.13733 | 1.14147 | 2026-07-29 20:00:00+00:00 | take profit | +29.3 / +28.3 | +2.32 |
| GBPUSD | long | 2026-03-18 17:15:00+00:00 | 1.33320 | 1.33120 | 1.33799 | 2026-03-18 19:30:00+00:00 | stop loss | -20.0 / -21.5 | -1.08 |
| EURUSD | short | 2026-04-29 15:30:00+00:00 | 1.16908 | 1.17071 | 1.16515 | 2026-04-29 20:45:00+00:00 | forced exit | +18.6 / +17.6 | +1.08 |
| EURUSD | long | 2026-03-18 16:00:00+00:00 | 1.15163 | 1.14956 | 1.15663 | 2026-03-18 19:30:00+00:00 | stop loss | -20.8 / -21.8 | -1.05 |
| EURUSD | short | 2026-06-17 17:00:00+00:00 | 1.15894 | 1.16049 | 1.15523 | 2026-06-17 20:00:00+00:00 | take profit | +37.1 / +36.1 | +2.33 |

Only the registry rows are verdicts; everything else is descriptive and none is a trading result.

_Generated 2026-11-22T16:02:45+00:00._
