# gold-bb-nyrange — pre-registration

Written before any gold outcome, exit or target is computed. Commit this file and PRE_REGISTRATION.json unchanged; every outcome stage refuses otherwise.

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

Config digest `03ecd5a5372c38cc732e4fa33c9233d038fd83a04f6f313fbb5e67783f849cb8`. Window 2015-01-02 .. 2026-11-20.

Input caches (pinned; a changed cache stops every gated stage):

- `dxy_d1` `8da163572167113c7c1bc0598414cda75785c809285460c744c1f70218f19ba0`
- `dxy_h1` `b4d8df711f01c85f86c2374f9ad899bcdcee2a89444145e841df08c03551a7c5`
- `gold_h1` `023a72fc14abf6647c792889c30987a80a942e9ba8f8f6a0b4c9d367a485f39c`

## Data

- Gold: results/gold/gold_h1.csv (bid OHLC, Berlin label clock, read-only). D1 bars close at 17:00 New York; valid with >= 12 H1 bars on a weekday.
- DXY hourly: MT5 ActivTrades USDIndDec26, from 2017-01-03, lag-0 checked against EURUSD every year.
- DXY daily: Yahoo DX-Y.NYB, closes dated strictly before the gold day only.

## Part 1a — Bollinger reversion

- Event: the first D1 close outside SMA20 ± 2 SD20 (population SD); 3-valid-day cooldown.
- Outcome r = dir × (close[t+3] − close[t]) / ATR20[t].
- Model: binary LightGBM (fx_bb settings) on 11 features (9 gold, 2 previous-day DXY), 12-month folds from 2018-01-01, 10-business-day purge + embargo from t+3, folds with < 60 training events predict nothing.
- Cells at alpha 0.025 each, 5-day blocks, seed 20261201:
  - `gold_bb_rule_mean_reversion`: mean r = dir * (close[t+3] - close[t]) / ATR20 > 0 over every event with an outcome.
  - `gold_bb_lightgbm_vs_base_rate`: mean log-loss of the walk-forward LightGBM below the fit-slice base rate of r > 0.
- Power: 173 rule events → minimum detectable mean 0.257 ATR; 106 model events → minimum detectable accuracy 0.636.

## Part 1b — the SMA20 trade

- Entry at the D1 close of every first breach; TP = SMA20[t]; stop the same distance beyond the entry; time stop at the close of t+10; exits scanned on H1 bars (a bar touching both is a stop; a gap through the stop fills at the open; a gap through the target fills at the target); one open trade.
- R = dir × (exit − entry) / |entry − stop|.
- Model: as Part 1a, on R > 0, the purge measured from the exit date.
- Cells at alpha 0.025 each, 10-day blocks, seed 20261202:
  - `gold_bb_sma_rule_mean_r`: mean R > 0 over every opened trade with an exit.
  - `gold_bb_sma_lightgbm_vs_base_rate`: mean log-loss of the walk-forward LightGBM below the fit-slice base rate of R > 0.
- Power (upper bounds): at most 179 trades → minimum detectable mean ≥ 0.209 R (SD 1.0 R); at most 110 model trades → minimum detectable accuracy ≥ 0.634.

## Part 2 — the NY session range

- y = log((max high − min low of the bars labelled 15:00–21:00 Berlin) / close of the 14:00 bar).
- Fourteen features at the 14:00 close (nine gold, the DXY 00:00–14:00 range ratio, four calendar flags).
- HAR (OLS on 3 lags), HAR-X (OLS on 14), LightGBM (fx_nyvol settings), 5-seed CUDA MLP; 6-month folds from 2018-01-01, 1-day embargo; all four models on the same rows.
- Three independent cells at alpha 0.01667 each (0.05/3), 5-day blocks, seed 20261203:
  - `gold_harx_vs_har`: mean squared error of y: HAR-X below HAR.
  - `gold_lgbm_vs_harx`: mean squared error of y: LightGBM below HAR-X.
  - `gold_mlp_vs_harx`: mean squared error of y: 5-seed MLP below HAR-X.
- Power: 2211 scored days → minimum detectable mean difference 0.0632 SD.

## Decision rule

- Whole business days are resampled (circular blocks, 10000 resamples, fixed seed). A "> 0" cell is KEEP when the alpha-quantile of the bootstrap means is above 0; a "below" cell is KEEP when the (1 − alpha)-quantile of the mean difference is below 0. Everything else is DROP. Each cell is scored once.

## Facts recorded

- Owner decisions (2026-11-22): both Bollinger variants, gross of cost (the gold cache has no spread), HAR/HAR-X/LightGBM/MLP for the range, and the dollar index (MT5 hourly for Part 2, Yahoo daily for Part 1).
- Gold H1 2015-2026 was already used by the gold family (validation and one-shot test blocks): its D1/H1 volatility LSTMs CLEARED against GARCH and its range intervals were DROP. Neither scored band events or the NY session range.
- Currency precedents: fx-bb-reversion and fx-bb-sma-exit DROP on six pairs; fx-ny-volatility LightGBM KEEP for both pairs and the descriptive HAR-X beat LightGBM and the MLP, which is why HAR-X is registered here; fx-ny-orb-hot DROP.
- Before this pre-registration only event counts, bar presence, the DXY clock and gaps were looked at (179 first breaches 2015-2026 measured before the plan). No gold outcome, exit or target was computed.
- The MT5 contract USDIndDec26 expires on 2026-12-11; its cache is pinned by digest and never refetched.
