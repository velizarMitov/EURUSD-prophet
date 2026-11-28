# gold-nyrange-forward

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

A forward test of gold_bbnr's frozen HAR-X for the size of gold's New York session. The code is in
`src/gold_nyfwd/`, and the openspec change is `gold-nyrange-forward`. The daily steps are in
[RUNBOOK.md](RUNBOOK.md).

## The question

gold-bb-nyrange found, on 2 211 historical days, that the linear HAR-X forecast of
`log(range of the bars labelled 15:00–21:00 Berlin / 14:00 close)` beat HAR, at 0.847 × HAR's MSE. Does it
also beat HAR on live days that no model has seen?

## How

- **Model.** The frozen OLS coefficients are copied into `PRE_REGISTRATION.json` and never refitted.
  Features and target are gold_bbnr's own code, run on the family's live MT5 caches. The live features
  were checked equal to gold_bbnr's historical ones for 2026-11-16 .. 20.
- **Every weekday.** A forecast at 16:08 Sofia from bars labelled 14:00 Berlin or earlier, and a
  settlement at 23:08. Logs are append-only and never backfilled.
- **The cell.** `gold_harx_vs_har_forward` at alpha 0.05 in `gold_nyfwd_hypothesis_log.csv`. It is
  scored once on the first 60 forward days, or on 2027-04-30 as `UNDERPOWERED — NO DECISION` if fewer.
- **Power.** About 34 % if the true effect equals the historical one. **A DROP is not evidence that the
  model fails.**

## Files

| File | What |
|---|---|
| `PRE_REGISTRATION.md` / `.json` | the cell, horizon, power and the pinned coefficients (committed first) |
| `forward/forecasts.csv` | one row per weekday the pass ran: forecast or the reason there is none |
| `forward/settlements.csv` | one row per settled forecast: `y`, both squared errors, the realised high and low |
| `gold_nyfwd_hypothesis_log.csv` | the one verdict, written once |
| `report.md` | counts before scoring; verdict and monthly MSE after |

The live caches and `contracts.json` (dollar-index contracts and their lag checks) are in the gitignored
`research_models/gold_nyfwd/`.
