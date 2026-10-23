# Gold pre-registration, round two

Created 2026-10-23T07:33:27+00:00 (openspec change `gold-forecasting-v2`). Every verdict below is **needs forward confirmation**. GROSS of cost.

## Disclosure

The validation block [70:80] was scored in round one (2026-10-21) and its results were seen before this registration. The test block [80:100] was spent in round one and is never scored in round two. Every round-two verdict needs forward confirmation.

## Round one is untouched

- Round-one registration digest: `7da929a2424201d6e6fd9df7886701e5f11cd7511617f3bd0a2deb383bde075d` (commit `ab420d3412250834feb50d87c6547b4d1ec98f76`)
- Round-one config digest: `3d3dd5bf03e05d4098850b38268307a9195eff17dbc2e02a19d27a46cd1c31bc`; its families direction, return and volatility keep their registries and alphas.
- Round-two study settings digest (`src/gold/gold_config_v2.json` → `study`): `27e03c50945fc1130d7997b6de4c29b041153036b7e2ca8bb6b1ca2bfee3e9c0`

## Families

### range (alpha 0.0250)

- Cells: h1_range, d1_range
- Definition: sigma = E|r| x sqrt(pi/2) from the frozen LSTM volatility forecast; range = as-of close x exp(+/- q_t x sigma / 100); q_t by adaptive conformal inference (gamma 0.005, window {'H1': 500, 'D1': 250}) from past settled |z| only; the first `window` validation rows warm up and are not scored
- Baseline: GARCH sigma (h1_garch_hod / d1_garch) with the static normal quantile, same rows
- Verdict: CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap CI of mean(winkler_baseline - winkler_model) at the 80% level has lower bound > 0 AND the 80% coverage is within +/-3% of target; else DROP. The 95 % level and a static-Gaussian LSTM range are reported, not decisive.

### macro (alpha 0.0125)

- Cells: d1_lstm_macro:vol, d1_lstm_macro:ret, d1_gbm_macro:ret, d1_range_macro
- Timing: FRED values dated on or before the previous US business day; MT5 bars labelled on or before the gold D1 bar; forward-fill only
- Pinned feature file: `results/gold/macro/pinned_features_d1.csv` (SHA-256 `c639bd37073e647e3875524177b357ba215ebe6ac5a1621fddbf12bd552c354d`, 2821 complete rows from 2016-01-26 00:00:00+00:00)
- Baseline: the same model and hyperparameters without the macro block, on identical rows
- Verdict: vol and range: CLEARED iff the paired bootstrap CI of mean(|err_without| - |err_with|) (Winkler for range) has lower bound > 0; return: CLEARED iff the one-sided Clark-West p < alpha against the model without macro; else DROP.

### direction2 (alpha 0.0250)

Edge 3% over 50 % at 80% power, the round-one rule unchanged.

| cell | validation n | required n | status |
|---|---|---|---|
| h1_selective | 2869 | 2640 | ADMITTED |
| h1_barrier | 4487 | 2640 | ADMITTED |
| h4_dir | 1219 | 2640 | UNDERPOWERED — NO DECISION |
| d1_gbm_macro:dir | 346 | 2640 | UNDERPOWERED — NO DECISION |
| d1_lstm_macro:dir | 346 | 2640 | UNDERPOWERED — NO DECISION |

- Selective coverage floor: 0.450
- Barrier: k = 6 h, m = 1.0, EWMA span 20; validation time-outs: 685
- Verdict: CLEARED iff both paired bootstrap CIs (vs majority, vs coin) have lower bound > 0; abstentions and time-outs are never scored; else DROP.

## Expected outcomes

- `h1_range`: likely CLEARED: it rides on the h1_lstm volatility edge that cleared validation and test
- `d1_range`: uncertain: the d1_lstm volatility edge cleared validation but not the test block
- `d1_lstm_macro:vol`: likely DROP: 346 validation days leave little room to show a paired gain
- `d1_lstm_macro:ret`: DROP expected: no gold return cell has ever cleared
- `d1_gbm_macro:ret`: DROP expected: no gold return cell has ever cleared
- `d1_range_macro`: likely DROP, for the same reason as d1_lstm_macro:vol
- `h1_selective`: DROP expected: round-one H1 direction sat at a coin; abstaining rarely creates an edge
- `h1_barrier`: DROP expected, if admitted
- `h4_dir`: UNDERPOWERED expected
- `d1_gbm_macro:dir`: UNDERPOWERED expected: 346 validation days
- `d1_lstm_macro:dir`: UNDERPOWERED expected: 346 validation days
