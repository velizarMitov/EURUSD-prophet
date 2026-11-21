# fx-session walk-forward

> **Gross of spread and slippage — not a trading result.**

Stage-1 run 2026-11-21T14:51:49+00:00, config digest `ae7be028d5af`, feature list `46087c0df6bf`. One forecast per day: features at the close of the bar labelled 14:00 Berlin (16:00 Sofia), target = sign of the mid close at 21:00 Berlin (23:00 Sofia). Out-of-sample days 2013-01-03..2026-11-20 in 28 folds of 6 months.

## Verdicts

alpha = 0.0125 (0.05/4), one-sided circular block bootstrap, block 5 days, 10000 resamples. Difference = model − reference log-loss; KEEP needs the whole interval below zero. An untriggered cell keeps its alpha unspent.

| Cell | Days | Model log-loss | Reference log-loss | Accuracy | Mean diff | Upper bound | Verdict |
|---|---|---|---|---|---|---|---|
| EURUSD_lightgbm_vs_prior | 3,470 | 0.69597 | 0.69509 | 51.27% | +0.00088 | +0.00274 | **DROP** |
| GBPUSD_lightgbm_vs_prior | 3,471 | 0.69488 | 0.69350 | 50.65% | +0.00139 | +0.00433 | **DROP** |
| EURUSD_mlp_vs_lightgbm | — | — | — | — | — | — | **NOT TRIGGERED** |
| GBPUSD_mlp_vs_lightgbm | — | — | — | — | — | — | **NOT TRIGGERED** |

## Power (from day counts only)

| Pair | Out-of-sample days | Minimum detectable accuracy |
|---|---|---|
| EURUSD | 3,470 | 52.62% |
| GBPUSD | 3,471 | 52.62% |

## Pooled out-of-sample metrics (descriptive)

### EURUSD (3,470 days, up-rate 49.80%)

| Model | Log-loss | Brier | Accuracy | Wilson 95 % | Accuracy on DST-mismatch days |
|---|---|---|---|---|---|
| class prior (fit slice) | 0.69509 | 0.25096 | 49.80% | 48.14%–51.46% | 53.71% (n = 175) |
| LightGBM | 0.69597 | 0.25138 | 51.27% | 49.60%–52.93% | 50.86% (n = 175) |

| Year | Days | class prior (fit slice) accuracy | LightGBM accuracy |
|---|---|---|---|
| 2013 | 235 | 60.9% | 60.9% |
| 2014 | 228 | 44.3% | 44.3% |
| 2015 | 238 | 52.1% | 52.1% |
| 2016 | 240 | 47.9% | 46.2% |
| 2017 | 250 | 54.0% | 54.0% |
| 2018 | 254 | 46.5% | 46.9% |
| 2019 | 258 | 47.7% | 45.3% |
| 2020 | 260 | 49.2% | 52.7% |
| 2021 | 258 | 50.8% | 53.5% |
| 2022 | 254 | 45.7% | 46.9% |
| 2023 | 250 | 51.2% | 54.4% |
| 2024 | 256 | 49.6% | 53.5% |
| 2025 | 258 | 52.7% | 52.3% |
| 2026 | 231 | 44.6% | 55.0% |

### GBPUSD (3,471 days, up-rate 50.62%)

| Model | Log-loss | Brier | Accuracy | Wilson 95 % | Accuracy on DST-mismatch days |
|---|---|---|---|---|---|
| class prior (fit slice) | 0.69350 | 0.25017 | 49.96% | 48.29%–51.62% | 50.86% (n = 175) |
| LightGBM | 0.69488 | 0.25079 | 50.65% | 48.99%–52.31% | 50.86% (n = 175) |

| Year | Days | class prior (fit slice) accuracy | LightGBM accuracy |
|---|---|---|---|
| 2013 | 235 | 45.1% | 49.8% |
| 2014 | 226 | 46.5% | 53.1% |
| 2015 | 238 | 54.6% | 54.2% |
| 2016 | 240 | 49.2% | 49.2% |
| 2017 | 248 | 56.0% | 52.4% |
| 2018 | 255 | 46.3% | 45.9% |
| 2019 | 259 | 49.0% | 49.0% |
| 2020 | 260 | 52.7% | 52.7% |
| 2021 | 259 | 51.0% | 45.2% |
| 2022 | 255 | 48.6% | 48.6% |
| 2023 | 250 | 54.4% | 54.4% |
| 2024 | 256 | 48.4% | 48.4% |
| 2025 | 259 | 52.1% | 52.9% |
| 2026 | 231 | 44.6% | 54.1% |

## Log-loss per fold

| Fold | Start | EURUSD prior | EURUSD lightgbm | GBPUSD prior | GBPUSD lightgbm |
|---|---|---|---|---|---|
| 0 | 2013-01-03 | 0.68195 | 0.68118 | 0.69356 | 0.69593 |
| 1 | 2013-07-01 | 0.66796 | 0.67330 | 0.69438 | 0.68408 |
| 2 | 2014-01-02 | 0.70907 | 0.70896 | 0.69835 | 0.68880 |
| 3 | 2014-07-01 | 0.74358 | 0.74394 | 0.70002 | 0.70063 |
| 4 | 2015-01-02 | 0.68887 | 0.69006 | 0.68273 | 0.67681 |
| 5 | 2015-07-01 | 0.69753 | 0.69810 | 0.69367 | 0.73591 |
| 6 | 2016-01-04 | 0.68906 | 0.71244 | 0.69582 | 0.69598 |
| 7 | 2016-07-01 | 0.70725 | 0.71072 | 0.69334 | 0.69358 |
| 8 | 2017-01-09 | 0.69557 | 0.69540 | 0.68724 | 0.68606 |
| 9 | 2017-07-03 | 0.68537 | 0.68661 | 0.69224 | 0.71112 |
| 10 | 2018-01-08 | 0.69694 | 0.69831 | 0.69594 | 0.69695 |
| 11 | 2018-07-02 | 0.69783 | 0.69833 | 0.69702 | 0.70469 |
| 12 | 2019-01-02 | 0.69833 | 0.69727 | 0.69467 | 0.69501 |
| 13 | 2019-07-01 | 0.69460 | 0.69558 | 0.69433 | 0.69395 |
| 14 | 2020-01-02 | 0.69292 | 0.69029 | 0.69203 | 0.69266 |
| 15 | 2020-07-01 | 0.69472 | 0.69302 | 0.69226 | 0.69290 |
| 16 | 2021-01-04 | 0.69219 | 0.69315 | 0.69203 | 0.69916 |
| 17 | 2021-07-01 | 0.69383 | 0.69284 | 0.69409 | 0.69385 |
| 18 | 2022-01-03 | 0.69527 | 0.69384 | 0.69325 | 0.69282 |
| 19 | 2022-07-01 | 0.69491 | 0.69775 | 0.69475 | 0.69469 |
| 20 | 2023-01-09 | 0.69218 | 0.69012 | 0.68818 | 0.68820 |
| 21 | 2023-07-03 | 0.69347 | 0.69361 | 0.69385 | 0.69509 |
| 22 | 2024-01-08 | 0.69196 | 0.69205 | 0.69384 | 0.69347 |
| 23 | 2024-07-01 | 0.69414 | 0.69261 | 0.69413 | 0.69421 |
| 24 | 2025-01-02 | 0.69247 | 0.69238 | 0.69334 | 0.69328 |
| 25 | 2025-07-01 | 0.69288 | 0.69267 | 0.69140 | 0.69024 |
| 26 | 2026-01-02 | 0.69402 | 0.69416 | 0.69533 | 0.69291 |
| 27 | 2026-07-01 | 0.69457 | 0.68984 | 0.69664 | 0.67869 |

## Reliability (predicted P(up) vs observed up-rate)

**EURUSD — LightGBM**

| P(up) bin | Days | Mean P(up) | Observed up-rate |
|---|---|---|---|
| 0.00-0.40 | 8 | 0.3747 | 0.7500 |
| 0.40-0.45 | 9 | 0.4294 | 0.2222 |
| 0.45-0.48 | 101 | 0.4706 | 0.4653 |
| 0.48-0.50 | 541 | 0.4928 | 0.4603 |
| 0.50-0.52 | 1,109 | 0.5091 | 0.4977 |
| 0.52-0.55 | 1,118 | 0.5346 | 0.5125 |
| 0.55-0.60 | 536 | 0.5735 | 0.5112 |
| 0.60-1.00 | 48 | 0.6442 | 0.5208 |

**GBPUSD — LightGBM**

| P(up) bin | Days | Mean P(up) | Observed up-rate |
|---|---|---|---|
| 0.00-0.40 | 38 | 0.3477 | 0.4737 |
| 0.40-0.45 | 77 | 0.4285 | 0.4416 |
| 0.45-0.48 | 121 | 0.4674 | 0.4711 |
| 0.48-0.50 | 315 | 0.4923 | 0.5270 |
| 0.50-0.52 | 2,031 | 0.5117 | 0.5057 |
| 0.52-0.55 | 653 | 0.5274 | 0.5008 |
| 0.55-0.60 | 130 | 0.5705 | 0.5385 |
| 0.60-1.00 | 106 | 0.6496 | 0.5472 |

## Bid-versus-mid parity

A gap above 1.0 pp is flagged ⚠ as a possible spread artifact. Verdicts use the mid target only.

| Pair | Up-rate (mid) | Up-rate (bid) | Gap (pp) | Flag |
|---|---|---|---|---|
| EURUSD | 49.80% | 49.80% | 0.00 | — |
| GBPUSD | 50.62% | 50.59% | 0.03 | — |

## Sign rule — gross of spread and slippage, descriptive, not a trading result

Long at the 16:00 Sofia mid close when P(up) > 0.5, else short; exit at the 23:00 Sofia mid close. One position per pair per day, no sizing, no cost.

| Pair | Model | Days | Long share | Win rate | Mean pips | Total pips | Total % | Profit factor |
|---|---|---|---|---|---|---|---|---|
| EURUSD | always long | 3,470 | 100.0% | 49.8% | +0.38 | +1304 | +13.13 | 1.030 |
| EURUSD | LightGBM | 3,470 | 81.0% | 51.3% | +1.00 | +3462 | +30.33 | 1.082 |
| GBPUSD | always long | 3,471 | 100.0% | 50.6% | +0.58 | +2003 | +17.34 | 1.038 |
| GBPUSD | LightGBM | 3,471 | 84.1% | 50.6% | +0.34 | +1165 | +9.29 | 1.022 |

_Generated 2026-11-21T14:52:42+00:00._
