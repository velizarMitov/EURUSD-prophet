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

Owner decision, 2026-10-23: the four dropped round-two cells (`d1_lstm_macro`, `d1_gbm_macro`, `h1_selective`,
`h1_barrier`) are logged live for observation on the leaderboard (`predictions_v2.csv`, version `v2`). This changes
none of the verdicts above: forward data can confirm a cleared cell, never revive a dropped one.

## 2026-11-04 — strategy backtests after cost, descriptive only (no registry, no verdict)

Owner-requested simulated backtests, run from throwaway scripts outside the repo. They wrote
nothing here, and no registry, registration, config or artifact changed. Unlike the family's
verdicts, these figures are **net of cost**:

- Spread is MT5's per-bar spread or a fixed $0.25 / $0.49 (the window median and the
  2026-11-04 value).
- Overnight swap is the broker's terms on 2026-11-04 (points mode, long −66.88 and short
  +23.84 per lot per night, Wednesday triple, about −6.1 % a year for a long). Where noted,
  the swap was rescaled to the effective fed funds rate.

Several of these ideas were looked at on the same years, and the validation and test blocks
were already seen. **Nothing here is evidence of an edge, and nothing may be revived as a
hypothesis on these spans.**

| idea | span | result |
|---|---|---|
| External TSFM P(up) > 0.5, hold one H1 bar, as-of labels 14–20 Mon–Fri | clean replay year, 1,785 h | Chronos-2 52.0 %, +$222/oz gross, −$225 / −$653 net; TimesFM 51.4 %, +$96 gross, −$350 / −$778 net; mean gain per trade below the spread; gross CI includes 0 |
| D1 trend, 14 rules (20/60/120/250-day sign, SMA 50/100/200; long/short and long/flat) | chosen on 2014-08 .. 2022-10-29 | none positive with swap; best, 120-day long/short, had Sharpe 0.05 |
| The chosen 120-day rule | 2022-10-30 .. 2026-11-03, once | +77 % vs buy-and-hold +89 %; daily difference CI [−0.074, +0.055] % |
| d1_lstm long/flat at the D1 close, honest validation + test predictions | 1,039 days | +97 % vs buy-and-hold +83 %; max drawdown −17 % vs −30 %; difference CI [−0.037, +0.046] %; hit 55.1 % against 54.0 % up-days; 5.1 % of time-shift placebos as good |
| The same signal acted on inside the owner's window (16:00 Sofia, label 14 close) | 994 days | to 23:00 the same day: +0.2 % vs always-long −10.7 %. Held to the next 16:00, with rate-linked swap: +60 % vs +90 %. 23–35 % of placebos as good |
| Short 16:00 → 23:00 Sofia every weekday, fixed before looking, on data unseen for it | 2015-01-02 .. 2022-10-28, 1,964 days | +4.5 % gross over 8 years, −7.5 % to −21 % net; daily CI includes 0; 49 % winning days; the yearly sign flips |

What this says:

- **No tested direction strategy beats buy-and-hold after cost.** The d1_lstm timing gain
  sits in the hours before the owner's window and vanishes when the call is acted on at
  16:00 Sofia.
- **The in-window fall of 2022–2026 was that span's, not a rule.** It did not replicate
  on 2015–2022.
- MT5 reports a 0.00 spread for 2015–2017 H1 bars, so the "MT5 spread" net figure for
  those years is optimistic.
- The MT5 chart's D1 arrows come from the final models, which were fit on all history. They
  have no honest past, so they were not backtested.

## 2026-11-04 — window strategy backtest, descriptive only (no registry, no verdict)

This entry comes from openspec change `gold-window-strategy`, package `src/gold_window/`;
the full report is `results/gold/window/backtest.md`.

The backtest covers four fixed rules that open and close inside 15:30–23:00 Sofia:
- the models are refit each calendar year on the past only (h1_lstm from 2018;
  d1_lstm and the D1 GARCH from 2016);
- prices are MT5 bid closes plus the bar's spread, with a 0.0062 % per round-trip floor;
- nothing is held overnight, so no swap applies.

The rules were chosen after the 2015–2026 descriptive statistics were seen, and the owner
declined an independent history (Dukascopy blocks direct download). **These figures are not
evidence and must not be revived as a hypothesis on these years.**

| rule | span | trades | mean net %/day, 95 % interval | total net | gate always open | always long 16–23 | placebo share |
|---|---|---|---|---|---|---|---|
| `h1_gated_direction` | 2019-01 .. 2026-11 | 3,349 | −0.043 [−0.071, −0.019] | −58 % | −89 % | −24 % | 0.99 |
| `continuation` | 2016-12 .. 2026-11 | 1,095 | +0.009 [−0.007, +0.028] | +22 % | +38 %, [−0.008, +0.039] %/day | −31 % | 0.31 |
| `us_open_breakout` | 2022-08 .. 2026-11 | 534 | −0.006 [−0.038, +0.025] | −7 % | −1 % | −12 % | 0.75 |
| `d1_gated_direction` | 2016-12 .. 2026-11 | 1,312 | −0.015 [−0.036, +0.005] | −34 % | −44 % | −32 % | 0.72 |

What this says:

- **The h1_lstm direction loses inside the window**, even before cost (−30 % gross).
  Trading it hourly multiplies the cost. Its interval is wholly below zero.
- **None of the size gates helps.** Each placebo share is far from 0, so shifting the gate
  in time does as well or better. For `continuation`, the rule without its gate did better
  (+38 %) than with it (+22 %).
- **Continuation is the only rule above zero, and its interval includes 0 both with and
  without the gate.** Its result depends on the years 2020, 2022–2024 and 2026. Its losing
  years are 2017–2019, 2021 and 2025.
- **Always long in the window loses** about 3 % a year after cost, as `DESCRIPTIVE.md`'s
  "no drift in the window" predicts.

## 2026-11-04 — acceleration and next-hour direction, descriptive only (no registry, no verdict)

This entry comes from openspec change `gold-acceleration-direction`, package `src/gold_accel/`;
the full report is `results/gold/accel/backtest.md`.

The study adds four time-of-day acceleration features (`accel_1h`, `activity_rel`,
`accel_expected_next`, `accel_m15`) to shadow copies of h1_gbm and h1_lstm, and adds a new
GBM `h1_accel` that uses only acceleration and the calendar:
- every model is refit each calendar year on the past only, from 2018 (the M15 variants
  from 2024, because MT5 M15 history starts 2022-08);
- each model with acceleration is compared with its pair without it on exactly the same rows;
- the registered models, `gold_config.json` and the registries were not touched.

The acceleration statistics were looked at before this study and every year has been seen.
**These figures are not evidence and must not be revived as a hypothesis on these years.**

| model | pair | view | rows | hit | pair hit | difference [95 %] |
|---|---|---|---|---|---|---|
| `h1_gbm_accel` | `h1_gbm` | all | 47,620 | 51.21 % | 51.28 % | −0.07 pp [−0.43, +0.30] |
| `h1_gbm_accel` | `h1_gbm` | window | 15,797 | 50.81 % | 50.84 % | −0.03 pp [−0.69, +0.66] |
| `h1_lstm_accel` | `h1_lstm` | all | 47,620 | 51.01 % | 51.10 % | −0.09 pp [−0.40, +0.24] |
| `h1_lstm_accel` | `h1_lstm` | window | 15,797 | 50.29 % | 50.83 % | −0.54 pp [−1.01, −0.01] |
| `h1_accel` | `h1_gbm` | all | 47,620 | 50.34 % | 51.28 % | −0.94 pp [−1.48, −0.36] |
| `h1_accel` | `h1_gbm` | window | 15,797 | 49.45 % | 50.84 % | −1.39 pp [−2.39, −0.30] |
| `h1_gbm_accel_m15` | `h1_gbm` | all | 15,350 | 50.40 % | 51.13 % | −0.74 pp [−1.70, +0.25] |
| `h1_gbm_accel_m15` | `h1_gbm` | window | 5,102 | 50.53 % | 50.86 % | −0.33 pp [−1.71, +1.36] |
| `h1_accel_m15` | `h1_gbm` | all | 15,350 | 50.03 % | 51.13 % | −1.10 pp [−2.27, −0.01] |
| `h1_accel_m15` | `h1_gbm` | window | 5,102 | 49.49 % | 50.86 % | −1.37 pp [−2.91, +0.31] |

What this says:

- **Adding acceleration to h1_gbm or h1_lstm did not change how often the next-hour
  direction was called right.** Every difference is within ±0.1 pp over all hours, and every
  interval but one holds 0. The exception is h1_lstm inside the window: −0.54 pp.
- **A model built only on acceleration (`h1_accel`) called direction less often right than
  h1_gbm:** −0.94 pp over all hours and −1.39 pp inside the window. Inside the window its
  hit rate, 49.45 %, is below a coin.
- **The M15 acceleration did not help either.** Both M15 variants are below h1_gbm on the
  same 2024–2026 rows.
- **After the real spread, every model loses inside the window** when its sign is held for
  one hour. h1_gbm averages −0.0144 % per trade [−0.0189, −0.0105] on 15,797 trades; the
  mean cost, 0.0154 %, is fifteen times its mean gross, +0.0010 %.
- Acceleration describes how large the next hour is likely to be, not which way it goes.
