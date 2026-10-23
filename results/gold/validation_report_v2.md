# Gold validation report, round two

Arbiter: `validation[70:80] round two (seen in round one)`. **Gross of cost.** Every verdict **needs forward confirmation**.

The validation block [70:80] was scored in round one (2026-10-21) and its results were seen before this registration. The test block [80:100] was spent in round one and is never scored in round two. Every round-two verdict needs forward confirmation.

- Registration: `results/gold/PRE_REGISTRATION_V2.md` (commit `417af006aa329fdb90cbb96b93aebb9ff046924d`)
- Run: 2026-10-23T07:36:00+00:00 (127 s)

| family | cell | n | model | baseline | delta | CI | p | alpha | verdict |
|---|---|---|---|---|---|---|---|---|---|
| range | h1_range | 5936 | 0.55864 | 0.56124 | 0.00261 | [-0.0048, 0.0108] |  | 0.02500 | DROP (CI includes 0 or favours h1_garch_hod) |
| range | d1_range | 96 | 2.54400 | 2.56953 | 0.02552 | [-0.1123, 0.1335] |  | 0.02500 | DROP (CI includes 0 or favours d1_garch) |
| macro | d1_lstm_macro:vol | 346 | 0.38819 | 0.38897 | 0.00079 | [-0.0047, 0.0058] |  | 0.01250 | DROP (CI includes 0 or favours no macro) |
| macro | d1_range_macro | 96 | 2.55188 | 2.51438 | -0.03750 | [-0.1350, 0.0406] |  | 0.01250 | DROP (CI includes 0 or favours no macro) |
| macro | d1_lstm_macro:ret | 346 | 0.67943 | 0.68417 |  |  | 0.0499 | 0.01250 | DROP (Clark-West p >= alpha) |
| macro | d1_gbm_macro:ret | 346 | 0.80973 | 0.80043 |  |  | 0.0156 | 0.01250 | DROP (Clark-West p >= alpha) |
| direction2 | h4_dir | 1219 |  |  |  |  |  | 0.02500 | UNDERPOWERED — NO DECISION |
| direction2 | d1_gbm_macro:dir | 346 |  |  |  |  |  | 0.02500 | UNDERPOWERED — NO DECISION |
| direction2 | d1_lstm_macro:dir | 346 |  |  |  |  |  | 0.02500 | UNDERPOWERED — NO DECISION |
| direction2 | h1_selective | 2769 | 0.50921 | 0.50957 | -0.00036 | [-0.0311, 0.0307] | 0.1642 | 0.02500 | DROP (a CI includes 0) |
| direction2 | h1_barrier | 4280 | 0.50958 | 0.50374 | 0.00584 | [-0.0273, 0.0409] | 0.1031 | 0.02500 | DROP (a CI includes 0) |

## Notes

- `h1_range`: Winkler@80% model 0.55864 vs h1_garch_hod 0.56124 vs static-Gaussian h1_lstm 0.58200 (reported, not decisive); coverage 80%: 80.0%, 95%: 95.0%; h1_garch_hod coverage 86.1%; warm-up 500 rows; block 24; gross of cost
- `d1_range`: Winkler@80% model 2.54400 vs d1_garch 2.56953 vs static-Gaussian d1_lstm 2.52622 (reported, not decisive); coverage 80%: 86.5%, 95%: 96.9%; d1_garch coverage 84.4%; warm-up 250 rows; block 5; gross of cost
- `d1_lstm_macro:vol`: MAE with macro 0.38819 % vs without 0.38897 %; block 5; fitted and scored on rows where every macro input exists; gross of cost
- `d1_range_macro`: Winkler@80%; warm-up 250 rows; block 5; fitted and scored on rows where every macro input exists; gross of cost
- `d1_lstm_macro:ret`: MSPE with vs without macro; CW stat 1.646; fitted and scored on rows where every macro input exists; gross of cost
- `d1_gbm_macro:ret`: MSPE with vs without macro; CW stat 2.156; fitted and scored on rows where every macro input exists; gross of cost
- `h4_dir`: not scored: below the registered power requirement
- `d1_gbm_macro:dir`: not scored: below the registered power requirement
- `d1_lstm_macro:dir`: not scored: below the registered power requirement
- `h1_selective`: tau 0.0340 from inner training folds; covered 2769 of 6376 validation rows (43.4%); abstentions not scored; accuracy 0.5092; train-majority class 0; vs coin delta +0.0092 CI [-0.0110, +0.0287]; PT stat 0.977; block 24; gross of cost
- `h1_barrier`: k 6 h, m 1.0 x EWMA(20); 30822 purged training labels; time-outs not scored; accuracy 0.5096; train-majority class 1; vs coin delta +0.0096 CI [-0.0117, +0.0297]; PT stat 1.264; block 24; gross of cost
