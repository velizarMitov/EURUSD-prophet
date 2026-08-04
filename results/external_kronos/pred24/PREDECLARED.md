# KRONOS AT THE AUTHORS' OWN CONFIGURATION (pred_len=24) — pre-declared readings

**Written 2026-08-03T20:17Z, BEFORE the first window of this program was
generated.** Nothing below may be softened after a result is seen.

## Configuration (matched to the authors' demo, not tuned)

| | value | note |
|---|---|---|
| pred_len | **24** | the demo's horizon |
| sample_count | 30 | the demo's value, unchanged |
| T | 1.0 | |
| top_p | 0.9 | |
| context | **512** | Kronos-base max. The demo used 360 with Kronos-mini; 512 is used for BOTH models here so the mini/base comparison isolates the model, not the context. Reported as used. |
| instrument | EURUSD H1 | `results/eurusd_h1.csv`, read-only |
| window | 2024-07-01 onward, whole context inside | |
| sampling | **every 24th bar, non-overlapping** | 519 disjoint 24-hour forecasts; label uniqueness 1.0 by construction |

Models: **Kronos-base** (primary, already integrated) and **Kronos-mini**
(4.1M, Kronos-Tokenizer-2k — the model the demo actually uses).

## The two metrics, exactly as the demo publishes them

**A) Upside Probability (next 24h).** `p_up_24` = share of the 30 sampled paths
whose bar-24 close exceeds the last actual close. Scored for sharpness
(distribution, dead-band share), calibration (reliability table, Brier vs the
constant-base-rate forecast, ECE, MCE) and accuracy against a train-majority
baseline fitted **pre-2024-07-01**. Distinct endpoint count out of 30 reported
alongside — it was **7.7** at pred_len=1.

**B) Volatility Amplification (next 24h).** `p_vol_amp` = share of the 30 paths
whose realised volatility over their 24 generated bars exceeds the trailing
realised volatility of the context. Trailing window = **the preceding 24 bars**,
matching the horizon. Volatility is the paper's eq. 12 form, sqrt of the sum of
squared log returns, over exactly 24 returns on every side (the step from the
last actual close into the first forecast bar is included, so predicted,
realised and trailing all count 24 returns). Binary outcome: did realised
volatility over the next 24 **actual** bars exceed that same trailing measure?
Scored by reliability table, Brier vs constant, ECE.

Descriptive alongside: correlation between predicted 24-bar volatility and
realised 24-bar volatility, with GARCH(1,1) (fit train-only, pre-2024-07) and
persistence on identical rows.

## STEP 4 — how this will be read

* **Distinct endpoints rise materially above 7.7 AND calibration improves
  materially** → the previous failure was configuration, the harness is
  validated, and Kronos has a usable probabilistic output on FX at a 24-hour
  horizon. First positive external result in this project; the serving
  configuration should then be revisited.
* **Distinct endpoints rise but calibration stays flat** → the collapse
  diagnosis was right and irrelevant; the model has no directional information
  on FX at this horizon either. A cleaner negative than before, because it is
  measured at the authors' own configuration.
* **Volatility amplification calibrates while direction does not** → the
  expected outcome given the paper claims volatility skill and no directional
  skill. That would validate the harness and locate the model's usefulness
  precisely.

## What this does to the earlier rounds

Every previous Kronos evaluation in this project ran at pred_len=1: the
withdrawn EURUSD H1 directional result, the four-frequency dose-response, and
the Alibaba/BTC positive controls. They are **superseded by configuration, not
merely by new evidence.** No conclusion about tokenizer resolution or about
Kronos on FX can be drawn from them. The artifacts are annotated, not deleted.

The **served endpoint still generates at pred_len=1** and is deliberately NOT
changed by this program. A serving change follows the result, not the reverse.
