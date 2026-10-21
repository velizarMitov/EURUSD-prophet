# Gold test report

Arbiter: `test[80:100] one-shot`. **Gross of cost** -- statistical skill only, never a trading result. Registered protocol: `results/gold/PRE_REGISTRATION.md`.

- Run: 2026-10-21T15:37:46+00:00 (271 s)

| family | cell | n | model | baseline | delta | CI | p | alpha | verdict |
|---|---|---|---|---|---|---|---|---|---|
| direction | d1_gbm | 693 | 0.53391 | 0.55700 | -0.02309 | [-0.0722, 0.0289] | 0.0902 | 0.02500 | UNDERPOWERED — NO DECISION |
| direction | d1_lstm | 693 | 0.55556 | 0.55700 | -0.00144 | [-0.0505, 0.0447] | 0.0179 | 0.02500 | UNDERPOWERED — NO DECISION |
| return | d1_gbm | 693 | 1.93933 | 1.86502 | -0.07431 |  | 0.1150 | 0.01250 | DROP (Clark-West p >= alpha) |
| return | d1_lstm | 693 | 1.87584 | 1.86502 | -0.01083 |  | 0.1935 | 0.01250 | DROP (Clark-West p >= alpha) |
| volatility | d1_lstm | 693 | 0.62793 | 0.63360 | 0.00567 | [-0.0195, 0.0302] |  | 0.02500 | DROP (CI includes 0 or favours d1_garch) |
| direction | h1_gbm | 12767 | 0.50819 | 0.48531 | 0.02287 | [0.0102, 0.0358] | 0.0184 | 0.02500 | DROP (a CI includes 0) |
| direction | h1_lstm | 12767 | 0.51335 | 0.48531 | 0.02804 | [0.0157, 0.0410] | 0.0008 | 0.02500 | CLEARED (beats the training majority and a coin: both CIs exclude 0) |
| return | h1_gbm | 12780 | 0.08238 | 0.08091 | -0.00147 |  | 0.2831 | 0.01250 | DROP (Clark-West p >= alpha) |
| return | h1_lstm | 12780 | 0.08104 | 0.08091 | -0.00013 |  | 0.3908 | 0.01250 | DROP (Clark-West p >= alpha) |
| volatility | h1_lstm | 12780 | 0.11702 | 0.12835 | 0.01133 | [0.0089, 0.0139] |  | 0.02500 | CLEARED (beats h1_garch_hod: CI of MAE_baseline - MAE_model excludes 0) |

## Descriptive only (never a verdict)

- D1 PBO across ['d1_gbm', 'd1_lstm']: 0.461
- h1_gbm: share of excess accuracy from 00:00-06:00 labels = -0.005
- h1_lstm: share of excess accuracy from 00:00-06:00 labels = 0.079
- H1 PBO across ['h1_gbm', 'h1_lstm']: 0.294
- d1_gbm: gross sign-strategy per-trade Sharpe 0.0460, SR0 0.0247 over 32 trials, DSR 0.715
- d1_lstm: gross sign-strategy per-trade Sharpe 0.0773, SR0 0.0247 over 32 trials, DSR 0.919
- h1_gbm: gross sign-strategy per-trade Sharpe -0.0001, SR0 0.0247 over 32 trials, DSR 0.003
- h1_lstm: gross sign-strategy per-trade Sharpe 0.0104, SR0 0.0247 over 32 trials, DSR 0.054
