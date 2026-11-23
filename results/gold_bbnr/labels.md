# gold-bb-nyrange — labels and power, before any outcome

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

Config digest `03ecd5a5372c38cc732e4fa33c9233d038fd83a04f6f313fbb5e67783f849cb8`. No outcome, exit, target or model output was read.

## Data

- Gold H1: 70,527 bars, 2015-01-02 04:00:00+00:00 .. 2026-11-20 21:00:00+00:00; 3,068 valid NY-close days.
- DXY H1 (MT5): 53,046 bars from 2017-01-03 02:00:00+00:00; DXY daily (Yahoo): 3,140 days.
- DXY days opening > 1% from the previous close: 1 (2025-02-03).

## Part 1a — Bollinger reversion (alpha 0.05/2 per cell)

- First breaches 180 (107 up, 73 down); removed by the 3-day cooldown 6; events 174.
- Rule cell: 173 events on 173 dates (no outcome 1). SD of the 3-day ATR move over all valid days 1.208. **Minimum detectable mean r: 0.257 ATR.**
- Model cell: 106 events (warm-up 2, before 2018 38, thin fold 27) in 7 folds. **Minimum detectable accuracy 0.636.**

| Fold start | Train | Test |
|---|---|---|
| 2020-01-01 | 64 | 13 |
| 2021-01-01 | 78 | 20 |
| 2022-01-01 | 94 | 14 |
| 2023-01-01 | 111 | 16 |
| 2024-01-01 | 127 | 13 |
| 2025-01-01 | 139 | 17 |
| 2026-01-01 | 157 | 13 |

## Part 1b — the SMA20 trade (alpha 0.05/2 per cell)

- Candidate entries (every first breach) 180; no exit data 1. The one-open-trade rule removes more once exits are known, so these counts are **upper bounds**.
- Rule cell, at most 179 trades: **minimum detectable mean R ≥ 0.209 R** (declared SD 1.0 R).
- Model cell, at most 110 trades (warm-up 2, before 2018 38, thin fold 29, folds from the latest possible exit date): **minimum detectable accuracy ≥ 0.634**.

## Part 2 — the NY session range (alpha 0.05/3 per cell)

- Weekdays 3,069; usable 2,442 from 2017-01-31; reasons {'usable': 2442, 'warm-up': 534, 'incomplete session': 92, 'invalid day': 1}.
- Scored days from 2018-01-01: 2,211 in 18 folds; first training set 229 days. **Minimum detectable mean squared-error difference: 0.0632 SD of that difference.**
- Scored days per year: {2018: 250, 2019: 250, 2020: 251, 2021: 250, 2022: 250, 2023: 249, 2024: 243, 2025: 245, 2026: 223}.

One-sided alphas 0.025, 0.025, 0.0167; power 80%.

_Generated 2026-11-22T21:28:59+00:00._
