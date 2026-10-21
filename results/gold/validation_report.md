# Gold validation report

Arbiter: `validation[70:80]`. **Gross of cost** -- statistical skill only, never a trading result. Registered protocol: `results/gold/PRE_REGISTRATION.md`.

- Run: 2026-10-21T14:59:57+00:00 (194 s)

| family | cell | n | model | baseline | delta | CI | p | alpha | verdict |
|---|---|---|---|---|---|---|---|---|---|
| direction | d1_gbm | 346 | 0.43931 | 0.50578 | -0.06647 | [-0.1387, 0.0116] | 0.9889 | 0.02500 | UNDERPOWERED — NO DECISION |
| direction | d1_lstm | 346 | 0.54335 | 0.50578 | 0.03757 | [-0.0376, 0.1127] | 0.0538 | 0.02500 | UNDERPOWERED — NO DECISION |
| return | d1_gbm | 346 | 0.79142 | 0.67892 | -0.11249 |  | 0.5074 | 0.01250 | DROP (Clark-West p >= alpha) |
| return | d1_lstm | 346 | 0.67759 | 0.67892 | 0.00134 |  | 0.1496 | 0.01250 | DROP (Clark-West p >= alpha) |
| volatility | d1_lstm | 346 | 0.38869 | 0.42224 | 0.03354 | [0.0150, 0.0503] |  | 0.02500 | CLEARED (beats d1_garch: CI of MAE_baseline - MAE_model excludes 0) |
| direction | h1_gbm | 6376 | 0.51035 | 0.50737 | 0.00298 | [-0.0185, 0.0213] | 0.0493 | 0.02500 | DROP (a CI includes 0) |
| direction | h1_lstm | 6376 | 0.50894 | 0.50737 | 0.00157 | [-0.0204, 0.0212] | 0.0528 | 0.02500 | DROP (a CI includes 0) |
| return | h1_gbm | 6392 | 0.03032 | 0.02999 | -0.00033 |  | 0.4827 | 0.01250 | DROP (Clark-West p >= alpha) |
| return | h1_lstm | 6392 | 0.03004 | 0.02999 | -0.00006 |  | 0.4862 | 0.01250 | DROP (Clark-West p >= alpha) |
| volatility | h1_lstm | 6392 | 0.07371 | 0.08138 | 0.00767 | [0.0058, 0.0096] |  | 0.02500 | CLEARED (beats h1_garch_hod: CI of MAE_baseline - MAE_model excludes 0) |

## Descriptive only (never a verdict)

- D1 PBO across ['d1_gbm', 'd1_lstm']: 0.076
- h1_gbm: share of excess accuracy from 00:00-06:00 labels = 0.205
- h1_lstm: share of excess accuracy from 00:00-06:00 labels = -0.184
- H1 PBO across ['h1_gbm', 'h1_lstm']: 0.913
- d1_gbm: gross sign-strategy per-trade Sharpe -0.0394, SR0 0.0199 over 16 trials, DSR 0.138
- d1_lstm: gross sign-strategy per-trade Sharpe 0.0424, SR0 0.0199 over 16 trials, DSR 0.662
- h1_gbm: gross sign-strategy per-trade Sharpe 0.0079, SR0 0.0199 over 16 trials, DSR 0.170
- h1_lstm: gross sign-strategy per-trade Sharpe 0.0065, SR0 0.0199 over 16 trials, DSR 0.142

_Deflated Sharpe recomputed 2026-10-21 from the saved per-row predictions after the run first charged N = 1 instead of the 16 logged grid trials; verdicts unaffected (RESULTS.md)._
