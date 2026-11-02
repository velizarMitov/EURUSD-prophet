# external_tsfm — pre-registration

Written by `python -m src.external.tsfm.prereg --write`. Rows are `registered` only once this file and `PRE_REGISTRATION.json` are committed unmodified; every earlier row is `dry_run` and is never scored.

## Cells

| cell | configuration | baseline | baseline digest |
|---|---|---|---|
| `external_tsfm:chronos2_gold` | `chronos2:fine_tuned:returns:8192` | h1_garch_hod | `752a367f3d696338` |
| `external_tsfm:chronos2_eurusd` | `chronos2:fine_tuned:returns:8192` | baseline_eurusd | `f544f74da0a59474` |
| `external_tsfm:timesfm25_gold` | `timesfm25:zero_shot:returns:2048` | h1_garch_hod | `752a367f3d696338` |
| `external_tsfm:timesfm25_eurusd` | `timesfm25:fine_tuned:returns:2048` | baseline_eurusd | `f544f74da0a59474` |

## Test

- Target: the next H1 bar's absolute log return in %; score: mean absolute error of `size_pct`.
- Statistic: mean of |baseline - realised| - |size - realised| on paired rows.
- Two-sided block bootstrap: block 24, 2000 resamples, seed 42.
- alpha = 0.05 / 4 = 0.0125. KEEP when the lower bound of the 98.75% interval > 0; DROP otherwise.
- Each cell is scored ONCE: at its first 1500 rows, or at the cap 2027-04-30 with what it has. Fewer than 500 rows at the cap: UNDERPOWERED — NO DECISION.
- Rows: phase registered, eligible, model and baseline both present, in as-of order.
- Direction: logged and shown, never registered, never judged.
- This family never changes: EUR/USD 0.05/9 feature bar, gold registries, the horizon study record.

## Power

At N = 1500 the standard error of the paired MAE difference is about 0.0032 pp for gold, so an edge of 0.0108 pp is detected with 80% power at alpha 0.0125. Gold h1_lstm's own edge over h1_garch_hod was 0.0113 pp: an external edge of that size would be borderline. About 110 eligible hours a week means N arrives in roughly 14 weeks.

## Frozen artifacts

- `external_tsfm:chronos2_gold`: chronos2 `ebb2ec865b31c448`, chronos2_gold_lora `38214c0081a62204`
- `external_tsfm:chronos2_eurusd`: chronos2 `ebb2ec865b31c448`, chronos2_eurusd_lora `7d49ab9268c8c2e1`
- `external_tsfm:timesfm25_gold`: timesfm25 `dac1d6da84b999d2`
- `external_tsfm:timesfm25_eurusd`: timesfm25 `dac1d6da84b999d2`, timesfm25_eurusd_lora `dfa65fdb11c1585a`

Replay (descriptive, chose the configurations): `results/external_tsfm/replay/replay.json`.

Label on every view until a verdict exists: "external model — research, no proven edge".
