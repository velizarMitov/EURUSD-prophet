# External foundation models: clean-window replay

**DESCRIPTIVE: chooses a configuration only, never a verdict.** Window: as-of 2025-11-01 .. 2026-10-31; first half before 2026-05-01. Rule: lowest first-half MAE of the size forecast vs realised |return| (%). Verdicts come only from forward rows (results/external_tsfm/PRE_REGISTRATION.md).

## gold

5374 eligible as-of rows (2025-11-03 01:00:00+00:00 .. 2026-10-30 20:00:00+00:00), 2635 in the first half.

| configuration | first-half MAE | second-half MAE | mean size % | mean P(up) |
|---|---|---|---|---|
| `chronos2:zero_shot:returns:2048` | 0.16571 | 0.13478 | 0.1625 | 0.513 |
| `chronos2:zero_shot:returns:8192` | 0.16601 | 0.13544 | 0.1609 | 0.512 |
| `chronos2:zero_shot:closes:2048` | 0.18248 | 0.15385 | 0.2458 | 0.493 |
| `chronos2:zero_shot:closes:8192` | 0.32711 | 0.16835 | 0.3744 | 0.357 |
| `chronos2:fine_tuned:returns:2048` | 0.16490 | 0.13384 | 0.1616 | 0.508 |
| `chronos2:fine_tuned:returns:8192` **(chosen)** | 0.16477 | 0.13422 | 0.1589 | 0.506 |
| `chronos2:fine_tuned:closes:2048` | 0.19353 | 0.15680 | 0.2664 | 0.497 |
| `chronos2:fine_tuned:closes:8192` | 0.34970 | 0.18266 | 0.4057 | 0.396 |
| `timesfm25:zero_shot:returns:2048` **(chosen)** | 0.16686 | 0.13494 | 0.1567 | 0.505 |
| `timesfm25:zero_shot:returns:16256` | 0.16838 | 0.13553 | 0.1439 | 0.510 |
| `timesfm25:zero_shot:closes:2048` | 0.17982 | 0.14376 | 0.2331 | 0.507 |
| `timesfm25:zero_shot:closes:16256` | 0.17526 | 0.14004 | 0.2101 | 0.542 |
| `timesfm25:fine_tuned:returns:2048` | 0.16974 | 0.13571 | 0.1961 | 0.505 |
| `timesfm25:fine_tuned:returns:16256` | 0.16710 | 0.13500 | 0.1829 | 0.510 |
| `timesfm25:fine_tuned:closes:2048` | 0.22619 | 0.17419 | 0.3243 | 0.491 |
| `timesfm25:fine_tuned:closes:16256` | 0.31703 | 0.21092 | 0.4201 | 0.223 |

Comparators on the same rows (descriptive; they inform no choice):

| comparator | first-half MAE | second-half MAE | rows |
|---|---|---|---|
| baseline | 0.18701 | 0.14360 | 2635 + 2586 |
| h1_lstm | 0.16357 | 0.13451 | 2635 + 2586 |

## eurusd

6084 eligible as-of rows (2025-11-03 00:00:00+00:00 .. 2026-10-30 20:00:00+00:00), 2994 in the first half.

| configuration | first-half MAE | second-half MAE | mean size % | mean P(up) |
|---|---|---|---|---|
| `chronos2:zero_shot:returns:2048` | 0.03324 | 0.02847 | 0.0362 | 0.492 |
| `chronos2:zero_shot:returns:8192` | 0.03309 | 0.02847 | 0.0351 | 0.498 |
| `chronos2:zero_shot:closes:2048` | 0.03647 | 0.03121 | 0.0467 | 0.478 |
| `chronos2:zero_shot:closes:8192` | 0.04764 | 0.03487 | 0.0636 | 0.479 |
| `chronos2:fine_tuned:returns:2048` | 0.03314 | 0.02827 | 0.0361 | 0.491 |
| `chronos2:fine_tuned:returns:8192` **(chosen)** | 0.03309 | 0.02834 | 0.0351 | 0.490 |
| `chronos2:fine_tuned:closes:2048` | 0.03748 | 0.03188 | 0.0504 | 0.503 |
| `chronos2:fine_tuned:closes:8192` | 0.05206 | 0.03650 | 0.0695 | 0.513 |
| `timesfm25:zero_shot:returns:2048` | 0.03423 | 0.02908 | 0.0324 | 0.492 |
| `timesfm25:zero_shot:returns:16256` | 0.03428 | 0.02921 | 0.0317 | 0.491 |
| `timesfm25:zero_shot:closes:2048` | 0.03657 | 0.03167 | 0.0479 | 0.493 |
| `timesfm25:zero_shot:closes:16256` | 0.03671 | 0.03053 | 0.0448 | 0.494 |
| `timesfm25:fine_tuned:returns:2048` **(chosen)** | 0.03328 | 0.02844 | 0.0360 | 0.491 |
| `timesfm25:fine_tuned:returns:16256` | 0.03331 | 0.02833 | 0.0355 | 0.494 |
| `timesfm25:fine_tuned:closes:2048` | 0.03631 | 0.03221 | 0.0478 | 0.545 |
| `timesfm25:fine_tuned:closes:16256` | 0.08931 | 0.05163 | 0.1023 | 0.163 |

Comparators on the same rows (descriptive; they inform no choice):

| comparator | first-half MAE | second-half MAE | rows |
|---|---|---|---|
| baseline | 0.04126 | 0.03569 | 2994 + 3090 |

## Choice

| model × instrument | chosen | replay argmin | frozen by registration |
|---|---|---|---|
| chronos2_gold | `chronos2:fine_tuned:returns:8192` | `chronos2:fine_tuned:returns:8192` | False |
| timesfm25_gold | `timesfm25:zero_shot:returns:2048` | `timesfm25:zero_shot:returns:2048` | False |
| chronos2_eurusd | `chronos2:fine_tuned:returns:8192` | `chronos2:fine_tuned:returns:8192` | False |
| timesfm25_eurusd | `timesfm25:fine_tuned:returns:2048` | `timesfm25:fine_tuned:returns:2048` | False |
