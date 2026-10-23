# Gold results and corrections

The corrections document named by `PRE_REGISTRATION.md`. The pre-registration itself is
never edited.

## 2026-10-21 — the registration was not committed on its own

The registration was committed in `ab420d3` (2026-10-21 17:51 +0300, titled "Refactor code
structure for improved readability and maintainability"). VS Code Copilot made that commit
and pushed it. Task 5.2 required the registration to be committed alone. Instead, this
commit also carries:

- all of `src/gold/` and `tests/test_gold_*.py`;
- the OHLC caches;
- the openspec change;
- three EUR/USD `results/forward_eval/` files.

What still holds, checked the same day before any score:

- No validation score existed when it was committed. All three registries were empty, and
  no validation report existed.
- `scoring_allowed()` is True: both files are committed, unmodified, and ancestors of
  HEAD.
- The pinned D1 (3,466 rows) and H1 (70,015 rows) digests and the `gold_config.json`
  digest match the registration.

What is lost: a commit you can review as the registration alone. The pushed history is
not rewritten. Read `results/gold/PRE_REGISTRATION.*` at `ab420d3` as the registered
protocol.

## 2026-10-21 — validation verdicts (arbiter `validation[70:80]`, gross of cost)

One run (194 s). Each cell has exactly one row in its family registry.

| family | cell | result | verdict | registered expectation |
|---|---|---|---|---|
| volatility | d1_lstm | MAE 0.3887 % vs GARCH(1,1) 0.4222 %, CI [0.0150, 0.0503] | CLEARED | CLEARED ✓ |
| volatility | h1_lstm | MAE 0.0737 % vs GARCH-HOD 0.0814 %, CI [0.0058, 0.0096] | CLEARED | CLEARED ✓ |
| direction | h1_gbm | 51.04 % vs majority 50.74 %, CI [−0.0185, +0.0213] | DROP | possible CLEARED ✗ |
| direction | h1_lstm | 50.89 % vs majority 50.74 % | DROP | DROP ✓ |
| direction | d1_gbm / d1_lstm | 346 rows (43.9 % / 54.3 %) | UNDERPOWERED | UNDERPOWERED ✓ |
| return | all four | Clark–West p ≥ 0.0125 | DROP | DROP ✓ |

Nine of ten cells came out as registered. The H1 GBM direction edge seen on EUR/USD,
AUD/USD and CHF/USD does not carry over to gold on this slice: the point gain is +0.3 pp
against a CI of about ±2 pp. The D1 direction figures are descriptive only; 346 rows
cannot decide anything.

### Correction: Deflated Sharpe trial count

The first rendering of `validation_report.md` charged the descriptive Deflated Sharpe for
N = 1 trial. It was computed before the run appended its 16 GBM grid trials to
`trial_log.csv`. It has been recomputed from the saved per-row predictions
(`validation_predictions_{d1,h1}.csv`) with N = 16, and the per-trade Sharpe values match
the run exactly. DSR is now 0.14 (d1_gbm), 0.66 (d1_lstm), 0.17 (h1_gbm) and 0.14
(h1_lstm), down from 0.24, 0.78, 0.74 and 0.70. The code now computes it after the trials
are appended. No verdict depends on it.

## 2026-10-21 — one-shot test report (`test[80:100]`, gross of cost)

The test block was scored once, after the validation verdicts were committed (`9b4c1fe`).
The run took 271 s. **The test never changes a validation verdict.** The registries are
not touched; the figures live in `test_report.{json,md}` and `test_predictions_{d1,h1}.csv`.

| family | cell | validation (arbiter) | test (one-shot) |
|---|---|---|---|
| volatility | h1_lstm | CLEARED | CLEARED: MAE 0.1170 vs 0.1284, CI [0.0089, 0.0139] |
| volatility | d1_lstm | CLEARED | not cleared: MAE 0.6279 vs 0.6336, CI [−0.0195, 0.0302] |
| direction | h1_lstm | DROP | cleared: 51.34 % over 12,767 rows; vs coin and vs majority both exclude 0; PT p = 0.0008 |
| direction | h1_gbm | DROP | not cleared: 50.82 %; the CI against a coin includes 0 |
| direction | d1_gbm / d1_lstm | UNDERPOWERED | UNDERPOWERED (693 rows) |
| return | all four | DROP | DROP |

What this means:

- **H1 volatility is the robust finding.** It cleared on both blocks, as it did for
  EUR/USD.
- **D1 volatility did not replicate.** Its test CI includes 0. The validation CLEARED
  rests on 346 rows and should be read as fragile.
- **The H1 LSTM direction result is not trend-riding.** On test it predicted up on 47.9 %
  of rows while the market rose on 51.5 %. It beat the base rate by about 1.4 pp on both
  its up calls (52.9 % vs 51.5 %) and its down calls (49.9 % vs 48.5 %). Validation shows
  the same small lift (+0.8 pp up, +1.2 pp down), which the arbiter's 6,376 rows could
  not separate from 0. The reading is a faint, consistent signal of about 1 pp at most.
  Its registered verdict stays **DROP**. Only the forward window can decide it, and even
  a forward KEEP would be gross skill, not a trading result.
- **The test majority baseline (48.5 %) is below a coin.** The [0:80] training majority
  is "down" (hourly gold has more small down bars), while the test block rose. That is
  why the registered rule requires beating both the majority and a coin.
- **Descriptive:** DSR over 32 trials is 0.003 for h1_gbm and 0.054 for h1_lstm (a gross
  sign strategy shows nothing). H1 PBO is 0.29. No high-spread-hour concentration flag was
  raised (shares −0.005 and 0.079).

## 2026-10-23 — round two, validation verdicts (`validation[70:80]`, seen in round one; gross of cost)

Openspec change `gold-forecasting-v2`. Registration: `results/gold/PRE_REGISTRATION_V2.md`, committed on its own in
`417af00` before any round-two score. Run once, 127 s; full table in `results/gold/validation_report_v2.md`. The
validation block was already seen in round one, so every verdict needs forward confirmation. The test block was not
touched.

| family (alpha) | cell | n | result | verdict |
|---|---|---|---|---|
| range (0.025) | h1_range | 5,936 | Winkler 0.5586 vs GARCH 0.5612, CI [-0.0048, +0.0108]; coverage 80.0 % / 95.0 % (GARCH 86.1 % at 80 %) | DROP |
| range (0.025) | d1_range | 96 | CI [-0.112, +0.134]; only 96 rows after the 250-row warm-up | DROP |
| macro (0.0125) | d1_lstm_macro:vol | 346 | MAE 0.3882 % with vs 0.3890 % without, CI [-0.0047, +0.0058] | DROP |
| macro (0.0125) | d1_range_macro | 96 | CI [-0.135, +0.041] | DROP |
| macro (0.0125) | d1_lstm_macro:ret | 346 | Clark–West p 0.050 | DROP |
| macro (0.0125) | d1_gbm_macro:ret | 346 | Clark–West p 0.016 | DROP |
| direction2 (0.025) | h1_selective | 2,769 covered of 6,376 | 50.9 % vs majority 51.0 %, vs coin CI [-0.011, +0.029] | DROP |
| direction2 (0.025) | h1_barrier | 4,280 | 51.0 %, vs coin CI [-0.012, +0.030] | DROP |
| direction2 | h4_dir, d1_gbm_macro:dir, d1_lstm_macro:dir | 1,219 / 346 / 346 | below 2,640 required | UNDERPOWERED — NO DECISION |

What this says, descriptively:

- **h1 ranges are well calibrated but not sharper than GARCH.** Adaptive calibration hit its coverage targets exactly
  where GARCH's static range over-covered (86.1 % at the 80 % level). A better-calibrated range is useful to read, but
  the registered test asked whether it is sharper, and it is not, at this alpha. The live ranges keep running and
  their coverage is tracked forward.
- **The macro drivers add nothing measurable to the daily models** over 346 validation days.
- **Abstaining and barrier labels do not create a direction edge** on H1. h1_selective covered 43.4 % of validation
  hours against its training-fold floor of 45 %. Its threshold was fixed on training folds, so a lower validation
  coverage is a distribution shift, not a tuning choice.
- Round one's registries, registration, config and trial log are byte-identical after the run.
