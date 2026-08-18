# Replication: Yıldırım, Toroslu & Fiore (2021)

*Forecasting directional movement of Forex data using LSTM with technical and macroeconomic indicators*, **Financial Innovation 7:1**, [doi:10.1186/s40854-020-00220-2](https://doi.org/10.1186/s40854-020-00220-2) (open access, CC BY 4.0).

Generated 2026-08-15 19:23 UTC — 180 LSTM fits, seeds [42, 43, 44, 45, 46], scales ['primary', 'secondary'].

> **This study costs zero alpha.** It registers no hypothesis in any family, adds no row to any `*_hypothesis_log.csv`, and does not tighten any Bonferroni bar. It is a sensitivity analysis of an *external* published result, not a claim about EUR/USD predictability. Nothing here may be used to justify a production feature.

## 1. What the paper reports

The split, quoted: *"The data set was split into the training and test sets, with ratios of 80% and 20%, respectively. The training phase was carried out with different numbers of iterations"*; and for the extended set, *"split into training and test sets, with ratios of 90% and 10%"*. **There is no validation set anywhere in the paper.** Every number below is therefore a test-block number, including the ones that were compared against each other to make a choice.

Single-model profit_accuracy, averaged over the four iteration counts (Tables 4-6, 8-10, 12-14) and the hybrid averaged over the 4x4 iteration grid (Tables 7, 11, 15):

| Horizon | ME_LSTM | TI_LSTM | ME_TI_LSTM | Hybrid | test rows |
|---|---|---|---|---|---|
| 1-day | 50.69% | 52.18% | 53.05% | **77.32%** | 243 |
| 3-day | 51.31% | 48.58% | 53.84% | **78.98%** | 243 |
| 5-day | 47.31% | 49.88% | 48.73% | **84.08%** | 242 |

The paper's own summary table (Table 20) disagrees with its per-model tables — 1-day ME/TI/ME-TI/Hybrid are 50.16 / 51.43 / 49.89 / 73.09 there versus 50.69 / 52.18 / 53.05 / 77.32 in Tables 4-7. We compare against Tables 4-7 throughout and note the discrepancy as-is.

At n=243 test rows and ~150 transactions, the three single-model figures are indistinguishable from a coin flip: the 95% binomial interval around 50% on 150 trials is roughly [42%, 58%]. The paper's headline is not the single models — it is the hybrid's ~25-point jump, bought by abstaining from about three quarters of the trades.

## 2. Our A0 (clean) next to the paper

A0 = threshold fitted on training rows only, iteration count chosen on the validation block, hybrid rules tuned on the validation block, test block scored exactly once. Mean over 5 seeds; +/- is the seed standard deviation.

**Scale: `primary`** — the paper's own window trimmed to its own row count: 1214 weekday bars, 2013-01-02 .. 2017-09-01.

The paper covers January 2013 - January 2018 in 1214 bars ('days in which the markets were open'); our weekday history carries ~260 bars/year against their ~241, so the same 1214-row count runs out earlier in calendar time. Row count is what fixes the 243-row test block and therefore the statistical power, so the count is what we matched.

| Horizon | Model | paper | our A0 (clean) | our A4 (all 3 leaks) | A0 transactions | paper transactions |
|---|---|---|---|---|---|---|
| 1-day | ME_LSTM | 50.69% | 48.85% ± 0.99 | 50.48% ± 0.24 | 230.2/243 | 149.50/243 |
| 1-day | TI_LSTM | 52.18% | 49.61% ± 1.95 | 49.93% ± 2.16 | 164.4/243 | 155.25/243 |
| 1-day | ME_TI_LSTM | 53.05% | 48.51% ± 1.01 | 47.15% ± 7.33 | 201.8/243 | 157.25/243 |
| 1-day | HYBRID | 77.32% | 48.91% ± 4.05 | 54.15% ± 1.70 | 126.2/243 | 64.75/243 |
| 3-day | ME_LSTM | 51.31% | 46.78% ± 0.48 | 46.78% ± 0.48 | 145.8/243 | 174.50/243 |
| 3-day | TI_LSTM | 48.58% | 48.82% ± 3.62 | 48.82% ± 3.62 | 201.2/243 | 146.50/243 |
| 3-day | ME_TI_LSTM | 53.84% | 47.40% ± 5.84 | 45.91% ± 3.36 | 128.0/243 | 158.50/243 |
| 3-day | HYBRID | 78.98% | 46.62% ± 1.14 | 56.21% ± 8.78 | 92.8/243 | 65.13/243 |
| 5-day | ME_LSTM | 47.31% | 44.21% ± 0.00 | 44.21% ± 0.00 | 145.2/242 | 206.25/242 |
| 5-day | TI_LSTM | 49.88% | 47.07% ± 3.88 | 47.07% ± 3.88 | 200.0/242 | 151.50/242 |
| 5-day | ME_TI_LSTM | 48.73% | 44.83% ± 2.09 | 44.83% ± 2.09 | 153.6/242 | 138.75/242 |
| 5-day | HYBRID | 84.08% | 45.00% ± 2.47 | 56.93% ± 12.97 | 92.2/242 | 69.31/242 |

**Scale: `secondary`** — our full euro-era daily set: 7185 weekday bars, 1999-01-04 .. 2026-08-10.

| Horizon | Model | paper | our A0 (clean) | our A4 (all 3 leaks) | A0 transactions | paper transactions |
|---|---|---|---|---|---|---|
| 1-day | ME_LSTM | 50.69% | 50.32% ± 2.28 | 50.35% ± 0.97 | 418.0/1437 | 149.50/243 |
| 1-day | TI_LSTM | 52.18% | 49.35% ± 2.14 | 50.67% ± 0.74 | 665.4/1437 | 155.25/243 |
| 1-day | ME_TI_LSTM | 53.05% | 48.51% ± 5.51 | 49.72% ± 1.83 | 553.2/1437 | 157.25/243 |
| 1-day | HYBRID | 77.32% | 53.70% ± 6.11 | 50.35% ± 3.65 | 158.0/1437 | 64.75/243 |
| 3-day | ME_LSTM | 51.31% | 48.80% ± 2.54 | 48.80% ± 2.54 | 1046.4/1437 | 174.50/243 |
| 3-day | TI_LSTM | 48.58% | 50.85% ± 0.62 | 50.85% ± 0.62 | 1038.6/1437 | 146.50/243 |
| 3-day | ME_TI_LSTM | 53.84% | 49.22% ± 3.27 | 48.09% ± 3.05 | 1042.6/1437 | 158.50/243 |
| 3-day | HYBRID | 78.98% | 48.44% ± 4.86 | 50.63% ± 6.83 | 639.2/1437 | 65.13/243 |
| 5-day | ME_LSTM | 47.31% | 49.14% ± 3.62 | 49.14% ± 3.62 | 1046.0/1437 | 206.25/242 |
| 5-day | TI_LSTM | 49.88% | 49.82% ± 0.99 | 49.82% ± 0.99 | 1037.6/1437 | 151.50/242 |
| 5-day | ME_TI_LSTM | 48.73% | 47.93% ± 4.36 | 47.93% ± 4.36 | 1026.2/1437 | 138.75/242 |
| 5-day | HYBRID | 84.08% | 47.06% ± 5.58 | 52.14% ± 8.96 | 738.8/1437 | 69.31/242 |

### Fidelity check: the labelling algorithm reproduces Table 3

The paper's Table 3 reports the thresholds its entropy search selected: **0.0023 / 0.0040 / 0.0055** for 1 / 3 / 5 days ahead. Our independent implementation of Algorithm 1 + Algorithm 2, run on our own EUR/USD history, lands here:

| Scale | Horizon | τ (train-only, A0/A2/A3) | τ (full series, A1/A4) | paper Table 3 |
|---|---|---|---|---|
| primary | 1-day | 0.00230 | 0.00226 | 0.0023 |
| primary | 3-day | 0.00421 | 0.00391 | 0.0040 |
| primary | 5-day | 0.00571 | 0.00542 | 0.0055 |
| secondary | 1-day | 0.00272 | 0.00236 | 0.0023 |
| secondary | 3-day | 0.00483 | 0.00420 | 0.0040 |
| secondary | 5-day | 0.00660 | 0.00566 | 0.0055 |

On the paper-scale window the three train-only thresholds agree with Table 3 to within a fraction of a pip. That is a meaningful check: the histogram upper bound and the entropy sweep were re-derived from the prose alone, on different data, and landed on the paper's numbers. Whatever else this replication does or does not reproduce, the labelling stage is faithful.

It also shows how *small* Leak 1 is mechanically: moving the threshold's scope from train-only to the full series shifts τ by only a few 1e-5. Leak 1 is real but its lever is short, and the leak decomposition below should be read with that in mind.

## 3. Leak decomposition

Each arm changes **one** selection input and nothing else. The training rows, the scoring rows, the architecture, the seeds and the transaction-count floor are identical across arms; A1/A4 retrain because a different threshold changes the training labels, and every arm reads the same fitted weights everywhere else. A positive gap means the leak inflates the reported figure.

| Arm | what leaks |
|---|---|
| A0 | clean: threshold on train, iterations on val, hybrid on val, test scored once |
| A1 | leak 1 only: threshold over the full series |
| A2 | leak 2 only: iteration count selected on test |
| A3 | leak 3 only: hybrid rules tuned on test |
| A4 | all three leaks (approximates as-published) |

CIs are 95% moving-block (circular, block length 20, 2000 resamples) paired bootstraps over the test rows, on `src.walk_forward_validation._circular_block_bootstrap_indices`. profit_accuracy has a random denominator, so each resample recomputes sum(wins)/sum(transactions).

**Scale: `primary`**

| Horizon | Model | gap | Δ profit_accuracy | 95% CI | ≠ 0? |
|---|---|---|---|---|---|
| 1-day | ME_LSTM | A1−A0 | +1.57 pp | [-8.92, +11.89] | no |
| 1-day | ME_LSTM | A2−A0 | +1.42 pp | [+0.20, +2.99] | yes |
| 1-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | ME_LSTM | A4−A0 | +1.57 pp | [-8.92, +11.89] | no |
| 1-day | TI_LSTM | A1−A0 | +0.36 pp | [-2.76, +4.22] | no |
| 1-day | TI_LSTM | A2−A0 | +2.11 pp | [+0.03, +4.14] | yes |
| 1-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | TI_LSTM | A4−A0 | +0.36 pp | [-2.76, +4.22] | no |
| 1-day | ME_TI_LSTM | A1−A0 | +0.76 pp | [-4.38, +5.74] | no |
| 1-day | ME_TI_LSTM | A2−A0 | +1.96 pp | [+0.06, +4.04] | yes |
| 1-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | ME_TI_LSTM | A4−A0 | +1.22 pp | [-3.40, +5.55] | no |
| 1-day | HYBRID | A1−A0 | +1.24 pp | [-7.34, +10.17] | no |
| 1-day | HYBRID | A2−A0 | +1.57 pp | [-1.65, +4.61] | no |
| 1-day | HYBRID | A3−A0 | +1.52 pp | [-0.63, +4.06] | no |
| 1-day | HYBRID | A4−A0 | +4.77 pp | [-5.00, +13.50] | no |
| 3-day | ME_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_TI_LSTM | A1−A0 | -0.57 pp | [-3.90, +3.17] | no |
| 3-day | ME_TI_LSTM | A2−A0 | +0.65 pp | [-0.74, +1.99] | no |
| 3-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_TI_LSTM | A4−A0 | -0.57 pp | [-3.90, +3.17] | no |
| 3-day | HYBRID | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | HYBRID | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | HYBRID | A3−A0 | +6.11 pp | [+1.70, +10.57] | yes |
| 3-day | HYBRID | A4−A0 | +6.11 pp | [+1.70, +10.57] | yes |
| 5-day | ME_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A3−A0 | +6.55 pp | [+1.22, +11.15] | yes |
| 5-day | HYBRID | A4−A0 | +6.55 pp | [+1.22, +11.15] | yes |

**Scale: `secondary`**

| Horizon | Model | gap | Δ profit_accuracy | 95% CI | ≠ 0? |
|---|---|---|---|---|---|
| 1-day | ME_LSTM | A1−A0 | +0.76 pp | [-0.88, +2.42] | no |
| 1-day | ME_LSTM | A2−A0 | +0.36 pp | [-1.75, +2.62] | no |
| 1-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | ME_LSTM | A4−A0 | +0.76 pp | [-0.88, +2.42] | no |
| 1-day | TI_LSTM | A1−A0 | +0.88 pp | [-0.92, +2.70] | no |
| 1-day | TI_LSTM | A2−A0 | +0.84 pp | [-0.67, +2.36] | no |
| 1-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | TI_LSTM | A4−A0 | +0.88 pp | [-0.92, +2.70] | no |
| 1-day | ME_TI_LSTM | A1−A0 | -0.90 pp | [-2.70, +0.98] | no |
| 1-day | ME_TI_LSTM | A2−A0 | +1.31 pp | [-0.67, +3.32] | no |
| 1-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 1-day | ME_TI_LSTM | A4−A0 | -0.59 pp | [-2.46, +1.27] | no |
| 1-day | HYBRID | A1−A0 | -1.45 pp | [-4.85, +1.77] | no |
| 1-day | HYBRID | A2−A0 | -2.02 pp | [-5.35, +1.14] | no |
| 1-day | HYBRID | A3−A0 | +1.92 pp | [-0.01, +3.93] | no |
| 1-day | HYBRID | A4−A0 | -1.41 pp | [-4.94, +2.07] | no |
| 3-day | ME_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_TI_LSTM | A1−A0 | -1.08 pp | [-2.13, +0.03] | no |
| 3-day | ME_TI_LSTM | A2−A0 | +0.04 pp | [-0.29, +0.38] | no |
| 3-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | ME_TI_LSTM | A4−A0 | -1.08 pp | [-2.13, +0.03] | no |
| 3-day | HYBRID | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | HYBRID | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 3-day | HYBRID | A3−A0 | +0.90 pp | [-1.38, +3.14] | no |
| 3-day | HYBRID | A4−A0 | +0.90 pp | [-1.38, +3.14] | no |
| 5-day | ME_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A3−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | ME_TI_LSTM | A4−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A1−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A2−A0 | +0.00 pp | [+0.00, +0.00] | no |
| 5-day | HYBRID | A3−A0 | +3.24 pp | [-0.79, +7.63] | no |
| 5-day | HYBRID | A4−A0 | +3.24 pp | [-0.79, +7.63] | no |

### What the leak decomposition does not explain

The gaps above are differences between OUR arms. They are only half the picture, because our A4 — the arm that takes all three shortcuts and is the closest thing here to 'as published' — does not land anywhere near the paper's reported hybrid:

| Scale | Horizon | paper hybrid | our A4 hybrid | our A0 hybrid | residual (paper − A4) |
|---|---|---|---|---|---|
| primary | 1-day | 77.32% | 54.15% | 48.91% | +23.2 pp |
| primary | 3-day | 78.98% | 56.21% | 46.62% | +22.8 pp |
| primary | 5-day | 84.08% | 56.93% | 45.00% | +27.1 pp |
| secondary | 1-day | 77.32% | 50.35% | 53.70% | +27.0 pp |
| secondary | 3-day | 78.98% | 50.63% | 48.44% | +28.3 pp |
| secondary | 5-day | 84.08% | 52.14% | 47.06% | +31.9 pp |

So the three selection choices under study move the hybrid by a few percentage points, while the distance between our as-published arm and the published figure is an order of magnitude larger. **The leaks do not account for the reported improvement.** Stated carefully, this replication establishes:

- Leak 3 (tuning the hybrid's abstention rule on the test block) is the only choice with a consistently non-zero effect here, worth roughly +6 pp at the 3- and 5-day horizons on the paper-scale window, with intervals excluding zero. Leak 2 (iteration count on test) is worth ~+1.5 to +2 pp for the single models at 1 day. Leak 1 (threshold scope) is not separable from zero anywhere — consistent with the fidelity table above, where widening the threshold's scope moves τ by only a few 1e-5.
- None of that explains a hybrid at 77-84%. Our hybrid sits near 50% on every arm, at every horizon, on both scales.

What this does **not** establish: that the paper is wrong. We reproduce one reading of an underspecified method (Section 4 lists 16 places where a reading had to be chosen) on a different vendor's EUR/USD series, with a different LSTM configuration, and with macro inputs mapped onto FRED series rather than the paper's own sources. Any of those could carry the difference. The honest summary is that **the reported hybrid improvement did not reproduce here, and the three selection leaks we can measure are not large enough to be its explanation** — which leaves it unexplained rather than explained away.

### What abstention actually does to profit_accuracy

profit_accuracy is a ratio whose denominator the model chooses, so it is worth measuring directly what happens as the hybrid's confidence floor rises and the transaction count collapses. `abstention_curve.csv` does that at the 1-day horizon with the 10% selection floor REMOVED, holding the iteration counts at A0's validation-chosen values so that only the confidence floor varies (mean over 5 seeds):

**`primary`**

| confidence floor | profit_accuracy | transactions | of rows |
|---|---|---|---|
| 0.00 | 48.1% | 158.0 | 243 |
| 0.34 | 48.1% | 158.0 | 243 |
| 0.40 | 48.1% | 158.0 | 243 |
| 0.45 | 48.2% | 157.8 | 243 |
| 0.50 | 48.1% | 152.8 | 243 |
| 0.55 | 48.6% | 147.6 | 243 |
| 0.60 | 49.3% | 142.6 | 243 |
| 0.65 | 49.0% | 122.6 | 243 |
| 0.70 | 44.3% | 90.0 | 243 |
| 0.75 | 40.5% | 43.6 | 243 |
| 0.80 | 38.4% | 30.4 | 243 |
| 0.85 | 31.8% | 25.8 | 243 |
| 0.90 | 51.3% | 11.8 | 243 |
| 0.95 | 50.0% | 6.8 | 243 |

**`secondary`**

| confidence floor | profit_accuracy | transactions | of rows |
|---|---|---|---|
| 0.00 | 63.5% | 167.4 | 1437 |
| 0.34 | 63.5% | 167.4 | 1437 |
| 0.40 | 63.9% | 142.8 | 1437 |
| 0.45 | 54.0% | 102.0 | 1437 |
| 0.50 | 54.9% | 35.0 | 1437 |
| 0.55 | 63.4% | 7.8 | 1437 |
| 0.60 | 73.3% | 3.4 | 1437 |
| 0.65 | 100.0% | 0.6 | 1437 |
| 0.70 | 100.0% | 0.2 | 1437 |
| 0.75 | n/a (no trades) | 0.0 | 1437 |
| 0.80 | n/a (no trades) | 0.0 | 1437 |
| 0.85 | n/a (no trades) | 0.0 | 1437 |
| 0.90 | n/a (no trades) | 0.0 | 1437 |
| 0.95 | n/a (no trades) | 0.0 | 1437 |

The curve does **not** show abstention buying accuracy. On the paper-scale window profit_accuracy stays in a 48-51% band while transactions fall from 158 to single digits, and the individual cells become erratic rather than better (31.8% on ~26 trades, then 51.3% on ~12). On the euro-era window the only cells that reach 100% are the ones where the denominator has collapsed below one trade per seed.

That reframes the paper's most striking cells. Table 7's 100.00% on 8 transactions, Table 11's 100.00% on 2, and Table 15's `Nan` on 0 are what a vanishing denominator looks like, not evidence that the filter is finding better trades — and averaging them into a headline gives them the same weight as a full-coverage cell. Our arms impose the 10% floor (UNDERSPECIFIED[#8]) precisely to keep *selection* out of that regime; the curve above is what the floor is protecting against.

## 4. Where the paper is underspecified, and what we chose

This list is part of the result. A replication of an underspecified method is a replication of one reading of it.

**1. Histogram upper bound**
- *Paper:* "the maximum difference value of the last bin added was used as the upper bound of the threshold value" — ambiguous between the bin's upper EDGE and the largest difference OBSERVED inside that bin.
- *Our choice:* We take the largest observed |difference| inside the last bin added. This is the only reading consistent with the paper's own worked example: with max difference 0.029 and 10 equal-width bins the edges are multiples of 0.0029, and the reported bound 0.00652 is not one of them. The bin-edge reading is available as histogram_threshold_upper_bound(..., use_bin_edge=True) and is unit-tested alongside the primary reading.

**2. Histogram bin ties**
- *Paper:* Bins are 'sorted in descending order' by count; ties are not resolved.
- *Our choice:* Stable descending sort, so equal counts keep ascending bin order (the lower/denser bin is consumed first).

**3. Entropy base**
- *Paper:* 'Entropy = - sum p_i * log p_i' does not state the logarithm base.
- *Our choice:* Natural log. The base is a positive constant factor and cannot change the argmax, so this choice is inert for the threshold actually used.

**4. Difference definition**
- *Paper:* 'the EUR/USD ratio differences between consecutive days' is stated for the 1-day case; the 3- and 5-day-ahead labels are never re-defined.
- *Our choice:* diff_t = close_{t+n} - close_t (absolute price difference, not a return) for n in {1,3,5}. Consistent with Table 3, whose thresholds grow with the horizon (0.0023 / 0.0040 / 0.0055) exactly as an n-day price difference would.

**5. Zero difference**
- *Paper:* profit_accuracy converts a predicted inc/dec on a true no_act row into a correct prediction 'if the actual movement is in the same direction'. An exactly-zero move has no direction.
- *Our choice:* diff == 0 counts as a LOSS on any transaction (it can never be 'in the same direction'). Affects a handful of rows.

**6. LSTM architecture**
- *Paper:* No layer count, unit count, activation, optimizer, loss, batch size, lookback window or scaler is given anywhere in the paper.
- *Our choice:* One LSTM layer of 32 units -> Dropout(0.2) -> Dense(3, softmax); Adam(lr=0.001); sparse categorical cross-entropy; batch 32; lookback 20 bars (config.json lstm.time_steps). 'Iterations' is read as epochs. Held IDENTICAL across all five arms, so it cannot affect the leak decomposition — only the absolute level.

**7. Feature scaling**
- *Paper:* No scaling is described, yet the feature set is dominated by non-stationary LEVELS (close, SMA, S&P 500, DAX, policy rates).
- *Our choice:* StandardScaler fit on TRAIN rows only (never val/test — a full-series scaler would be a fourth leak, and this study measures exactly three), then clipped to +/-5.0 sigma so a trending level cannot saturate the LSTM into a constant output. Identical in every arm. This penalises the level-heavy ME feature set on the long secondary scale, and that is an honest consequence of the paper's feature choice, not a bug.

**8. Selection criterion floor**
- *Paper:* The paper reports profit_accuracy over a self-selected transaction subset with no floor on the transaction count — Table 7 reports 100.00% on 8/243 transactions, Table 11 on 2/243.
- *Our choice:* Any configuration is selectable only if it transacts on at least 10% of the SELECTION block's rows; if none clears the floor we fall back to the configuration with the most transactions. Applied identically in all five arms. Without such a floor the metric is maximised by trading once, which is the mechanism behind the paper's own 100% cells.

**9. Hybrid tunable surface**
- *Paper:* The combiner's 'smart decision rules' are given as three fixed rules, but the paper also reports two variants ('modification based on ME_LSTM' / 'based on TI_LSTM') that are never defined, and describes the mechanism as 'eliminating transactions with weaker confidence' without naming a confidence level.
- *Our choice:* The two reported variants are read as the tie-break model. The confidence mechanism is exposed as one parameter, min_confidence, over the grid (0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7); min_confidence=0.0 reproduces the paper's three rules EXACTLY, so the published rule is a member of every arm's grid.

**10. Iteration comparison scope**
- *Paper:* 'we performed 50, 100, 150, and 200 iterations ... to properly compare different models' — the block on which that comparison was made is never named, and no validation block exists.
- *Our choice:* This is Leak 2 and is exactly what arms A0/A2 vary. For the hybrid, each constituent's iteration count is inherited from that constituent's own single-model choice, which keeps Leak 2 and Leak 3 separable.

**11. Iteration grid inconsistency**
- *Paper:* 'Training classifiers and labeling the data' says iterations of (50, 100, 150); 'Experiments' says (50, 100, 150, 200) and every results table reports 200.
- *Our choice:* We use (50, 100, 150, 200), the grid the tables actually report.

**12. Macroeconomic sources**
- *Paper:* The ME set is 'Interest Rate_GER', 'Interest Rate_EU', 'FED Funds Rate', 'Inflation Rate_EU', 'Inflation Rate_USA', 'Close(S&P 500)', 'Close(DAX)', sourced from ECB SDW / BLS / FRED / Yahoo Finance. Exact series IDs are not given.
- *Our choice:* Mapped onto this project's existing FRED framework: rate_de = IRLTLT01DEM156N (DE long-term rate, monthly), rate_eu = ECBDFR (ECB deposit facility), rate_fed = DFF (effective fed funds), infl_eu = DE HICP YoY (CP0000DEM086NEST, Germany standing in for the euro area, the project's existing choice), infl_us = US CPI YoY (CPIAUCSL). S&P 500 and DAX have no series in config.json's FRED set (FRED's SP500 carries only a rolling 10 years and FRED has no DAX at all), so they come from yfinance ^GSPC / ^GDAXI — already a project dependency — and are cached to results/yildirim_replication/equity_indices.csv.

**13. Technical indicator outputs**
- *Paper:* 'MACD with short- and long-term periods of 12 and 26' names no signal line; 'BB with period of 20' names no band or output; 'MA with a period of 10' does not say whether the raw level or a ratio is fed.
- *Our choice:* MACD = EMA12 - EMA26 (line only, no signal line specified). Bollinger is fed as %B, period 20, 2.0 sigma, reusing src.ti_lstm_h1_experimental.bollinger_percent_b. MA(10) is fed as the raw level, as written. RSI(10) is src.h1_features.wilder_rsi (the paper cites Wilder 1978). CCI(20) is src.ti_lstm_h1_experimental.cci (Lambert 1980, c=0.015). ROC(2) = (close/close_{t-2} - 1)*100, Momentum(4) = close - close_{t-4}.

**14. Trading calendar**
- *Paper:* '1234 data points in which the markets were open' (~247 rows/year) — the paper's bars are weekday bars.
- *Our choice:* Our MT5-derived daily history carries a short Sunday session bar (~1421 of 8606 euro-era rows). Sunday bars are dropped so the cadence matches the paper's; the remaining Mon-Fri rows run ~260/year, and holidays survive as flat bars.

**15. Sequence warm-up**
- *Paper:* The lookback window is never stated, so neither is what happens to the first rows of the training block.
- *Our choice:* Sequences end-index at or after row 19, so the training block loses its first 19 rows. The validation and test blocks are untouched, which keeps the test count at exactly 243/243/242 for horizons 1/3/5 — the paper's own counts.

**16. Minibatch shuffling**
- *Paper:* Not mentioned.
- *Our choice:* Keras default shuffle=True. Each sample is a self-contained trailing window, so shuffling minibatches cannot leak the future; it only affects optimisation.

**17. Macro publication lag (not a choice, an observation)**
- *Paper:* 'Monthly inflation rates were collected from the websites of central banks, and they were repeated for all days of the corresponding month to fill the fields in our daily records.' A month's CPI is not known until the following month, so a January CPI stamped on 2 January is a value the market did not have.
- *Our choice:* We reproduce the paper's convention exactly (month-stamped level, forward-filled across the month) because deviating from it would stop being a replication. This is a fourth, built-in leak that the paper's design carries independently of the three under study; it is present identically in ALL FIVE arms, so it inflates the absolute level of every ME-containing arm without contaminating any arm-to-arm gap. Flagged rather than fixed.

**18. Reported-figure inconsistency (not a choice, an observation)**
- *Paper:* The paper's per-model tables and its summary table disagree. Tables 4/5/6 give 1-day averages of 50.69 / 52.18 / 53.05, while Table 20 gives 50.16 / 51.43 / 49.89 for the same three models. Table 7 gives hybrid 1-day averages of 77.32 / 77.76, while Table 20 gives 73.09.
- *Our choice:* We quote both in the report and compare against the per-model tables (Tables 4-7), which the brief cites and which are the more detailed of the two.

## 5. Which gaps are distinguishable from zero

7 of 96 gaps have a 95% interval excluding zero:

| Scale | Horizon | Model | gap | Δ | 95% CI |
|---|---|---|---|---|---|
| primary | 1-day | ME_LSTM | A2−A0 | +1.42 pp | [+0.20, +2.99] |
| primary | 1-day | TI_LSTM | A2−A0 | +2.11 pp | [+0.03, +4.14] |
| primary | 1-day | ME_TI_LSTM | A2−A0 | +1.96 pp | [+0.06, +4.04] |
| primary | 3-day | HYBRID | A3−A0 | +6.11 pp | [+1.70, +10.57] |
| primary | 3-day | HYBRID | A4−A0 | +6.11 pp | [+1.70, +10.57] |
| primary | 5-day | HYBRID | A3−A0 | +6.55 pp | [+1.22, +11.15] |
| primary | 5-day | HYBRID | A4−A0 | +6.55 pp | [+1.22, +11.15] |

The remaining 89 gaps have intervals covering zero and are **not** distinguishable from noise.

### Power, stated plainly

The primary scale has 243 test rows and, after abstention, often fewer than 100 transactions. A 95% interval on a proportion from ~100 trials is about ±10 percentage points before any block-dependence widening. Any gap smaller than that is unresolvable here **by construction** — no amount of seeding fixes it, because the limit is the number of scoring rows the paper's design provides, not the number of runs. The secondary scale exists precisely to check whether a gap that is invisible at n=243 becomes visible at euro-era length.

### Files

- `arms.csv` — one row per (scale, arm, horizon, model, seed), with the selection actually made (iteration counts, hybrid parameters, threshold).
- `summary.csv` — `record=arm_mean` rows (seed means/sds) and `record=gap` rows (bootstrap CIs per leak gap).
- `abstention_curve.csv` — profit_accuracy vs transaction count as the hybrid's confidence floor rises, with the selection floor removed (`--abstention-curve`).
- `panel_cache.csv`, `equity_indices.csv` — the assembled input panel, cached so the run is reproducible offline.
- `run_meta.json` — run parameters.
