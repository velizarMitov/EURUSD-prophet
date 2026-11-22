# fx-ny-orb-hot — pre-registration

Written before any breakout outcome is computed. Commit this file and PRE_REGISTRATION.json unchanged; scoring refuses otherwise.

Config digest `cb6fd6a7d4f3b830d5b2d640eaa935cbc90c65b4e378fe4851aae29f5d056c41`.

## The rule (fixed, never searched)

- NY opening range: the M15 bars labelled 14:00, 14:15, 14:30, 14:45 Berlin (15:00–16:00 Sofia).
- Hot day: the walk-forward HAR-X forecast above the causal Q75 of the previous 60 eligible days.
- Entry: the first M15 close outside the range from 15:00; entry at the next bar's mid open, only if range ≤ 0.5 F, breakout ≤ 0.25 F beyond the level, entry before 19:00. Otherwise no trade that day.
- SL 0.25 F, TP 0.6 F from the entry; a bar touching both is SL; forced exit at the close of the bar labelled 20:45; one trade per day per pair.
- Costs: half the recorded spread at entry and at exit, plus 0.4 bps slippage. R = SL distance.

## Cells (historical family, alpha 0.05/2 each, scored once)

- `orb_hot_net_r`: mean net R of hot-day trades > 0 (pooled EURUSD+GBPUSD).
- `orb_hot_minus_cold`: mean net R of hot-day trades - mean net R of non-hot trades > 0.
- Window 2018-08-08 .. 2026-08-07; circular 5-business-day block bootstrap, 10000 resamples, seed 20261127.
- Power: 170 hot trades on 121 dates; minimum detectable mean 0.322 R on trades, 0.382 R on dates (SD 1.5 R).

## Forward family (alpha 0.05/2 each, scored once)

- `EURUSD_harx_vs_har`, `GBPUSD_harx_vs_har`: frozen OLS HAR-X vs HAR (fit to 2026-11-20), per-day squared error, after 270 settled forward days per pair or on 2028-01-31 (then UNDERPOWERED — NO DECISION if fewer). Forward days start after this commit.

## Chart gate

- a live entry sign is exported only when orb_hot_net_r is KEEP.

## Facts recorded

- HAR-X (OLS on fx_nyvol's 14 features) was chosen after its range MSE was seen in fx-ny-volatility (2026-11-22, descriptive line, MSE 0.80x / 0.85x HAR); its walk-forward predictions are reused unchanged.
- The breakout outcomes on these days were not computed or looked at before this pre-registration.
- Earlier fx families: fx-fibonacci, fx-session-ny, fx-rates-2y, fx-bb-reversion and fx-bb-sma-exit were DROP; fx-ny-volatility LightGBM vs HAR KEEP for both pairs.
