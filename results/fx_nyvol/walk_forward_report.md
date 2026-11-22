# fx-ny-volatility walk-forward

> **Gross of spread and slippage — not a trading result.**

Target: y = log(New York session mid range / mid close of the 14:00 bar), bars labelled 15:00–21:00 Berlin (16:00–23:00 Sofia); features at the close of the bar labelled 14:00 Berlin. Reference: HAR OLS on the range's own 1/5/20-day history, fitted per fold on the training days only.

## Verdicts

alpha = 0.0125 (0.05/4), one-sided circular block bootstrap, block 5 days, 10000 resamples, on the per-day squared-error difference (model − reference). KEEP needs the whole interval below zero. An untriggered cell keeps its alpha unspent.

| Cell | Days | Model MSE | Reference MSE | Relative MSE reduction | Mean diff | Upper bound | mde_80 | Verdict |
|---|---|---|---|---|---|---|---|---|
| EURUSD_lightgbm_vs_har | 3,475 | 0.17315 | 0.19143 | +9.55% | -0.01828 | -0.01136 | 0.00956 | **KEEP** |
| GBPUSD_lightgbm_vs_har | 3,475 | 0.16105 | 0.17170 | +6.20% | -0.01065 | -0.00452 | 0.00856 | **KEEP** |
| EURUSD_mlp_vs_lightgbm | 3,475 | 0.16805 | 0.17315 | +2.95% | -0.00510 | -0.00122 | 0.00544 | **KEEP** |
| GBPUSD_mlp_vs_lightgbm | 3,475 | 0.15992 | 0.16105 | +0.70% | -0.00113 | +0.00310 | 0.00567 | **DROP** |

## Power (from labels.md, before any model)

| Pair | Out-of-sample days | Detectable difference (σ units) |
|---|---|---|
| EURUSD | 3,475 | 0.0523 |
| GBPUSD | 3,475 | 0.0523 |

## Out-of-sample accuracy (descriptive; run 2026-11-22T13:43:17+00:00, config `570a8661f040`)

R² is 1 − MSE / var(y) on the out-of-sample days. HAR-X separates "pre-session information helps" (a linear model gets it too) from "a non-linear model helps" (only LightGBM or the MLP gets it).

### EURUSD (3,475 days, 28 folds; mean y -5.5479, SD 0.5106)

| Model | MSE | MAE | R² | MSE / HAR | Spearman ρ |
|---|---|---|---|---|---|
| HAR (reference) | 0.19143 | 0.34230 | 0.2656 | 1.0000 | 0.5132 |
| HAR-X (OLS, all 14 features; descriptive) | 0.15273 | 0.30731 | 0.4141 | 0.7979 | 0.6270 |
| LightGBM | 0.17315 | 0.32624 | 0.3357 | 0.9045 | 0.5773 |
| MLP 5-seed ensemble | 0.16805 | 0.32155 | 0.3553 | 0.8779 | 0.5999 |

**Hot sessions** — y above its causal Q75 (the previous 60 eligible days only). Share of hot days: 26.6%.

| Model | Days called hot | Precision | Recall | Accuracy |
|---|---|---|---|---|
| HAR (reference) | 133 | 0.526 | 0.076 | 0.736 |
| HAR-X (OLS, all 14 features; descriptive) | 404 | 0.678 | 0.297 | 0.776 |
| LightGBM | 417 | 0.561 | 0.253 | 0.749 |
| MLP 5-seed ensemble | 470 | 0.579 | 0.294 | 0.755 |

| Year | Days | HAR (reference) MSE | HAR-X (OLS, all 14 features; descriptive) MSE | LightGBM MSE |
|---|---|---|---|---|
| 2013 | 235 | 0.20540 | 0.19664 | 0.22790 |
| 2014 | 227 | 0.22859 | 0.18275 | 0.29486 |
| 2015 | 238 | 0.19767 | 0.15160 | 0.20246 |
| 2016 | 240 | 0.19220 | 0.13796 | 0.15688 |
| 2017 | 250 | 0.18932 | 0.16114 | 0.17688 |
| 2018 | 255 | 0.15214 | 0.12938 | 0.13849 |
| 2019 | 259 | 0.22988 | 0.16207 | 0.19281 |
| 2020 | 260 | 0.19365 | 0.16811 | 0.16691 |
| 2021 | 260 | 0.16992 | 0.13250 | 0.15133 |
| 2022 | 255 | 0.17195 | 0.13331 | 0.13585 |
| 2023 | 250 | 0.17512 | 0.13357 | 0.13504 |
| 2024 | 256 | 0.18743 | 0.14031 | 0.14846 |
| 2025 | 259 | 0.19137 | 0.15841 | 0.15620 |
| 2026 | 231 | 0.20136 | 0.15633 | 0.15718 |

| Day type | Days | HAR MSE | LightGBM MSE | MSE / HAR |
|---|---|---|---|---|
| NFP | 157 | 0.16451 | 0.13532 | 0.8225 |
| FOMC | 102 | 0.51442 | 0.35466 | 0.6894 |
| DST mismatch | 175 | 0.20487 | 0.16195 | 0.7905 |
| other | 3064 | 0.18312 | 0.17028 | 0.9299 |

**Bid parity:** corr(y, y_bid) = 1.0000; LightGBM MSE / HAR MSE against the bid target = 0.9045 (against the mid target 0.9045).

### GBPUSD (3,475 days, 28 folds; mean y -5.4736, SD 0.4847)

| Model | MSE | MAE | R² | MSE / HAR | Spearman ρ |
|---|---|---|---|---|---|
| HAR (reference) | 0.17170 | 0.32391 | 0.2692 | 1.0000 | 0.4704 |
| HAR-X (OLS, all 14 features; descriptive) | 0.14552 | 0.29922 | 0.3806 | 0.8475 | 0.5721 |
| LightGBM | 0.16105 | 0.31438 | 0.3145 | 0.9380 | 0.5299 |
| MLP 5-seed ensemble | 0.15992 | 0.31182 | 0.3193 | 0.9314 | 0.5539 |

**Hot sessions** — y above its causal Q75 (the previous 60 eligible days only). Share of hot days: 25.9%.

| Model | Days called hot | Precision | Recall | Accuracy |
|---|---|---|---|---|
| HAR (reference) | 139 | 0.576 | 0.089 | 0.747 |
| HAR-X (OLS, all 14 features; descriptive) | 355 | 0.625 | 0.247 | 0.767 |
| LightGBM | 322 | 0.556 | 0.199 | 0.752 |
| MLP 5-seed ensemble | 299 | 0.555 | 0.185 | 0.751 |

| Year | Days | HAR (reference) MSE | HAR-X (OLS, all 14 features; descriptive) MSE | LightGBM MSE |
|---|---|---|---|---|
| 2013 | 235 | 0.16565 | 0.15956 | 0.16928 |
| 2014 | 227 | 0.17586 | 0.13939 | 0.22317 |
| 2015 | 238 | 0.14216 | 0.12525 | 0.14002 |
| 2016 | 240 | 0.17703 | 0.15478 | 0.21496 |
| 2017 | 250 | 0.17664 | 0.16115 | 0.16590 |
| 2018 | 255 | 0.16792 | 0.14805 | 0.15471 |
| 2019 | 259 | 0.21349 | 0.18469 | 0.18284 |
| 2020 | 260 | 0.18639 | 0.15709 | 0.16030 |
| 2021 | 260 | 0.13342 | 0.11292 | 0.12418 |
| 2022 | 255 | 0.19557 | 0.15438 | 0.16619 |
| 2023 | 250 | 0.15914 | 0.12397 | 0.13339 |
| 2024 | 256 | 0.15341 | 0.12496 | 0.13420 |
| 2025 | 259 | 0.16710 | 0.13521 | 0.14114 |
| 2026 | 231 | 0.18984 | 0.15645 | 0.15327 |

| Day type | Days | HAR MSE | LightGBM MSE | MSE / HAR |
|---|---|---|---|---|
| NFP | 157 | 0.17411 | 0.15478 | 0.8890 |
| FOMC | 102 | 0.37144 | 0.27630 | 0.7439 |
| DST mismatch | 175 | 0.21677 | 0.18936 | 0.8735 |
| other | 3064 | 0.16292 | 0.15590 | 0.9569 |

**Bid parity:** corr(y, y_bid) = 1.0000; LightGBM MSE / HAR MSE against the bid target = 0.9381 (against the mid target 0.9380).

## MSE per fold

| Fold | Start | EURUSD HAR | EURUSD LightGBM | GBPUSD HAR | GBPUSD LightGBM |
|---|---|---|---|---|---|
| 0 | 2013-01-03 | 0.21731 | 0.23614 | 0.17095 | 0.18394 |
| 1 | 2013-07-01 | 0.19455 | 0.22039 | 0.16083 | 0.15593 |
| 2 | 2014-01-02 | 0.23152 | 0.31522 | 0.17212 | 0.21902 |
| 3 | 2014-07-01 | 0.22612 | 0.27765 | 0.17902 | 0.22668 |
| 4 | 2015-01-02 | 0.20234 | 0.23073 | 0.15629 | 0.16375 |
| 5 | 2015-07-01 | 0.19346 | 0.17691 | 0.12938 | 0.11856 |
| 6 | 2016-01-04 | 0.17036 | 0.14286 | 0.17154 | 0.24399 |
| 7 | 2016-07-01 | 0.21368 | 0.17067 | 0.18242 | 0.18641 |
| 8 | 2017-01-09 | 0.16357 | 0.15297 | 0.18554 | 0.15923 |
| 9 | 2017-07-03 | 0.21508 | 0.20078 | 0.16773 | 0.17256 |
| 10 | 2018-01-08 | 0.17076 | 0.14655 | 0.16898 | 0.16010 |
| 11 | 2018-07-02 | 0.13424 | 0.13073 | 0.16690 | 0.14953 |
| 12 | 2019-01-02 | 0.22101 | 0.19431 | 0.23328 | 0.20563 |
| 13 | 2019-07-01 | 0.23855 | 0.19135 | 0.19416 | 0.16057 |
| 14 | 2020-01-02 | 0.19920 | 0.16411 | 0.23652 | 0.20131 |
| 15 | 2020-07-01 | 0.18819 | 0.16967 | 0.13702 | 0.11992 |
| 16 | 2021-01-04 | 0.16361 | 0.14843 | 0.13401 | 0.12721 |
| 17 | 2021-07-01 | 0.17604 | 0.15414 | 0.13285 | 0.12124 |
| 18 | 2022-01-03 | 0.17422 | 0.12600 | 0.20700 | 0.16695 |
| 19 | 2022-07-01 | 0.16962 | 0.14593 | 0.18386 | 0.16542 |
| 20 | 2023-01-09 | 0.20265 | 0.16304 | 0.16586 | 0.13909 |
| 21 | 2023-07-03 | 0.14760 | 0.10704 | 0.15242 | 0.12769 |
| 22 | 2024-01-08 | 0.19467 | 0.15881 | 0.14718 | 0.12470 |
| 23 | 2024-07-01 | 0.18051 | 0.13857 | 0.15935 | 0.14326 |
| 24 | 2025-01-02 | 0.20331 | 0.15107 | 0.16602 | 0.14048 |
| 25 | 2025-07-01 | 0.17971 | 0.16121 | 0.16815 | 0.14179 |
| 26 | 2026-01-02 | 0.20931 | 0.16650 | 0.19507 | 0.15683 |
| 27 | 2026-07-01 | 0.19148 | 0.14561 | 0.18334 | 0.14885 |

Only the registry rows are verdicts; every other number is descriptive and none is a trading result.

_Generated 2026-11-22T14:00:07+00:00._
