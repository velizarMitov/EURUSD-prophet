# fx-ny-volatility — forecasting the New York session range (16:00–23:00 Sofia)

> **Gross of spread and slippage — not a trading result.**

Openspec change `fx-ny-volatility`. Code `src/fx_nyvol/`, tests `tests/test_fx_nyvol_*.py`, docs
`ARCHITECTURE_DOCS.md §3.18`.

**Target.** `y = log((mid high − mid low) / mid close14)` over the bars labelled
15:00–21:00 Berlin. All 14 features are known at the close of the bar labelled 14:00 Berlin
(16:00 Sofia).

**Reference.** HAR: an OLS on the range's own 1/5/20-day history, fitted per fold on the
training days.

**Evaluation.** Expanding 6-month folds from 2013-01-01 with a 1-day embargo, and 3 475
out-of-sample days per pair.

## Result (scored once, 2026-11-22)

| Cell | MSE model | MSE reference | Relative MSE reduction | Upper bound | Verdict |
|---|---|---|---|---|---|
| `EURUSD_lightgbm_vs_har` | 0.17315 | 0.19143 | 9.6 % | −0.0114 | **KEEP** |
| `GBPUSD_lightgbm_vs_har` | 0.16105 | 0.17170 | 6.2 % | −0.0045 | **KEEP** |
| `EURUSD_mlp_vs_lightgbm` | 0.16805 | 0.17315 | 3.0 % | −0.0012 | **KEEP** |
| `GBPUSD_mlp_vs_lightgbm` | 0.15992 | 0.16105 | 0.7 % | +0.0031 | **DROP** |

alpha = 0.05/4 per cell, tested with a one-sided circular block bootstrap on the per-day
squared-error difference: 5-day blocks, 10 000 resamples. The detectable difference stated
in `labels.md` before any model was 0.052 σ.

### What the KEEPs mean — read this before using them

**A linear model on the same 14 features (HAR-X) beats both non-linear models.** HAR-X is
descriptive and not registered.

| Pair | MSE / HAR: HAR-X | LightGBM | MLP | R² out of sample: HAR | HAR-X | LightGBM | MLP |
|---|---|---|---|---|---|---|---|
| EURUSD | **0.798** | 0.905 | 0.878 | 0.266 | **0.414** | 0.336 | 0.355 |
| GBPUSD | **0.848** | 0.938 | 0.931 | 0.269 | **0.381** | 0.315 | 0.319 |

- **The registered cells confirm one thing:** the day's pre-session information (the Asia,
  London and data-hour ranges and the cross-pair morning) predicts the New York range
  beyond its own history.
- **They do not show that a non-linear model helps.** On these features a plain OLS does
  better than LightGBM, and EURUSD's MLP KEEP only says it is less bad than LightGBM.
- **In early folds the trees do worse than HAR.** EURUSD's LightGBM MSE is worse than HAR's
  in 2013–2015, with few training days.
- **HAR-X's advantage is descriptive.** Making HAR-X the registered model would be a new
  hypothesis, and it would need its own pre-registration and forward data.

**Other descriptive points:**
- **Hot sessions** are days whose range is above the causal Q75 of the previous 60 days.
  They make up about 26 % of days.

  | Pair | Model | Precision | Recall |
  |---|---|---|---|
  | EURUSD | HAR | 0.53 | 0.08 |
  | EURUSD | HAR-X | 0.68 | 0.30 |
  | EURUSD | MLP | 0.58 | 0.29 |
  | GBPUSD | HAR-X | 0.63 | 0.25 |

- **Event days gain the most.** On FOMC days LightGBM's MSE is 0.69 × HAR's (EURUSD), and
  on NFP days 0.82 ×.
- **Mid and bid agree.** corr(y, y_bid) is 1.000, and the MSE ratios on the bid target
  equal those on the mid target.

A range forecast is not a trading result. Whether a hot-session filter adds anything after
costs would be a separate, pre-registered question.

## Files

| File | What it is |
|---|---|
| `labels.md`, `labels.json` | days, exclusions and the power table, before any model |
| `fx_nyvol_hypothesis_log.csv` | the four verdict rows |
| `oos_lgbm.csv.gz`, `run_lgbm.json` | stage 1: y, y_bid, HAR, HAR-X and LightGBM predictions, the causal Q75; folds, HAR coefficients, the fx_session cache digests |
| `oos_mlp.csv.gz`, `run_mlp.json` | stage 2: the MLP ensemble's predictions; best epochs and the device (RTX 4070 Laptop, CUDA) |
| `artifact_manifest.csv` | the digests of the frozen fold models in `research_models/fx_nyvol/` |
| `walk_forward_report.md` | the descriptive report |

Stages, in order:
1. `python -m src.fx_nyvol.labels`
2. `python -m src.fx_nyvol.walk_forward --stage lgbm`
3. `python -m src.fx_nyvol.score --stage lgbm`
4. `python -m src.fx_nyvol.walk_forward --stage mlp`
5. `python -m src.fx_nyvol.score --stage mlp`
6. `python -m src.fx_nyvol.report`

The data is read-only from `results/fx_session/data`.

**Eligibility is joint for both pairs.** The day must be eligible in fx_session's session
calendar, have bars 00:00–14:00 for EURUSD, GBPUSD and AUDUSD, and have no spread above
50 pips. Under that rule 116 days fall in unverified-clock weeks, 42 sessions are
incomplete, 4 days miss a pre-session bar, 1 has a corrupt spread (AUDUSD 2014-01-07), and
20 are warm-up.
