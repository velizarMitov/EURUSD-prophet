# fx-rates research family

Does the 2-year yield differential predict the sign of the EUR/USD or GBP/USD move over the
next 5 trading days?
- **The decision** is at the close of D1 bar `t`.
- **Yields** enter only from the previous FX trading day.
- **The target** is the sign of the mid move from `t` to `t+5`.

The OpenSpec change is `fx-rates-2y`, and the code lives in `src/fx_rates/`.

> **Gross of spread and slippage — not a trading result.** This is the owner's decision
> of 2026-11-21. Nothing here is broker, order or execution code. Nothing here feeds the
> served API, `models/` or the paper-trading ledgers.

## Status

**Decided 2026-11-21.** Both LightGBM cells are DROP and both MLP cells are NOT TRIGGERED
(`fx_rates_hypothesis_log.csv`, `walk_forward_report.md`).

| Pair | LightGBM log-loss | Prior log-loss | Accuracy | Wilson 95 % on n_eff | Upper bound |
|---|---|---|---|---|---|
| EURUSD | 0.69429 | 0.69333 | 50.58 % | 47.2–53.9 % (n_eff 848) | +0.00369 |
| GBPUSD | 0.69346 | 0.69293 | 52.52 % | 49.1–55.9 % (n_eff 812) | +0.00895 |

- **The bar.** The test could detect accuracy of about 55.3–55.8 %, so a DROP means
  "no weekly edge of that size".
- **GBPUSD's gross weekly rule** has a profit factor of 1.07–1.12 (descriptive). Its log-loss
  is still worse than the prior's, and the accuracy interval includes 50 %.

## Data — `data/`

| Series | Source | First | Last (2026-11-21 refresh) | Rows |
|---|---|---|---|---|
| `us2y` | FRED `DGS2` (public CSV) | 2005-01-03 | 2026-11-19 | 5 475 |
| `us10y` | FRED `DGS10` (public CSV) | 2005-01-03 | 2026-11-19 | 5 475 |
| `de2y` | Bundesbank, Svensson term structure of Federal securities, residual maturity 2.0 years (BBSIS `…R02XX…`) | 2005-01-03 | 2026-11-20 | 5 562 |
| `uk2y` | Bank of England nominal spot curve, 2.0 years: sheet "4. spot curve" of the GLC daily archive plus the current-month file | 2005-01-04 | 2026-11-19 | 5 530 |

**How the caches are kept:**
- **Append-only.** Revised values are counted in `refresh_log.csv`, never rewritten.
- **Each series fails on its own.** A failed series keeps its cache unchanged.
- **One generic client name** is used everywhere: `Mozilla/5.0 (research data client)`.
  FRED drops the connection for a fake browser User-Agent; this was found on the first
  refresh.

**Bank of England quirks:**
- **No 2-year code in its database (IADB).** `IUDSNZC` is 5Y and `IUDMNZC` is 10Y. The 2Y
  point comes from the yield-curve `.xlsx` files, read with the standard library.
- **Not the short-end sheet.** Sheet "3. spot, short end" has a 24-month column too, and
  is never used.
- **A check value.** UK 2Y on 2010-01-04 is 1.45 %.

**FX** comes read-only from the fx_session H1 caches (`results/fx_session/data/`):
- D1 bars are built by trading date from 2012-01-01. A D1 bar needs at least 12 H1 bars.
- The mid close is the last H1 bid close plus half its spread.
- The broker's shifted GBPUSD 2011 clock falls before the start.

## Alignment

- **Yields lag one FX trading day.** For FX date `t`, each series is taken as of the
  previous FX date: its latest value dated on or before that day. Nothing ever looks
  forward.
- **Short gaps are filled and flagged.** An observation up to 3 business days old is
  allowed and sets `yield_stale_flag`. Older ones exclude the day as "stale yields".

## Commands

```powershell
python -m src.fx_rates.rates_loader                # refresh us2y, us10y, de2y, uk2y (each series on its own)
python -m src.fx_rates.labeling                    # labels.md: sources, days, up-rates, overlap-aware power (no model)
python -m src.fx_rates.walk_forward --stage lgbm   # prior + LightGBM -> oos_lgbm.csv.gz, run_lgbm.json
python -m src.fx_rates.score --stage lgbm          # score the two LightGBM cells ONCE
python -m src.fx_rates.walk_forward --stage mlp    # MLP only behind a LightGBM KEEP (CUDA)
python -m src.fx_rates.score --stage mlp           # score the MLP cells ONCE, or record NOT TRIGGERED
python -m src.fx_rates.report                      # walk_forward_report.md
python -m pytest -q tests/test_fx_rates_config.py tests/test_fx_rates_protected.py tests/test_fx_rates_data.py tests/test_fx_rates_features.py tests/test_fx_rates_labels.py tests/test_fx_rates_models.py tests/test_fx_rates_walkforward.py tests/test_fx_rates_report.py
```

## Features (`src/fx_rates/features.py`)

There are 12 columns, in the order listed in `src/fx_rates/fx_rates_config.json`. The
spread is foreign 2Y − US 2Y: DE for EURUSD, UK for GBPUSD.

| Column | Definition |
|---|---|
| `spread_2y_level`, `spread_2y_chg_{5,20,60}d`, `spread_2y_zscore_60d` | level, changes over 5, 20 and 60 FX days, 60-day z-score |
| `us_2s10s_slope` | US10Y − US2Y |
| `fx_ret_{5,20}d` | log return divided by vol20 · √h |
| `fx_vs_spread_div_20d` | z of the 20-day FX log return minus z of the 20-day spread change, each against the previous 252 days |
| `fx_ma20_vs_ma50`, `fx_vol_20d_ratio` | log(MA20 / MA50) / vol20; vol20 / vol60 |
| `yield_stale_flag` | 1 when any yield was forward-filled |

## Target and decision

- **The target** is 1 when the mid close 5 FX trading days later is higher. Ties are
  dropped. There are no class weights.
- **Overlap.** Overlapping 5-day targets make about 3 080 days worth about 810–850
  independent ones (VIF ≈ 3.6–3.8).
- **Folds** run every 6 months from 2015-01-01, with a 5-day purge plus a 5-day embargo
  on both gaps.
- **Four cells at `alpha = 0.05/4`:**
  - LightGBM vs the prior, per pair;
  - the MLP vs LightGBM, run only after that pair's LightGBM KEEP. Otherwise the cell is
    NOT TRIGGERED, and its alpha is never reassigned.
- **The test** is a one-sided circular block bootstrap with 10-day blocks on the per-day
  log-loss difference. Each cell is scored once.
- **A changed config is a new hypothesis.** Every stage refuses one.
