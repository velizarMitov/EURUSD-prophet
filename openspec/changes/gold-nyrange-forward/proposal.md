# Proposal

## Why

gold-bb-nyrange found one KEEP: on 2 211 historical days the linear HAR-X forecast of gold's
New York session range beat HAR, with 0.847 × HAR's MSE. That result comes from history the
model was designed on. Only live days that no model has seen can show whether it holds.
fx-ny-orb-hot already runs such a forward test for EURUSD and GBPUSD. Gold has none.

## What Changes

- **New family `gold_nyfwd`.** It lives in the package `src/gold_nyfwd/`.
  - It reads the frozen gold_bbnr HAR-X and HAR coefficients and gold_bbnr's feature and target code
    read-only. It never refits them.
  - The coefficients are copied into the pre-registration, so the registered model is pinned exactly.
- **Live data, read-only from MT5.** It keeps its own gold H1 and dollar-index H1 caches.
  - The clock is real time on the Berlin label clock, never the last MT5 tick.
  - `USDIndDec26` expires on 2026-12-11. The family then rolls to the listed `USDInd` contract with the
    nearest later expiry, checked against EURUSD for a lag of 0.
  - On a day with no usable contract, no forecast is logged and the day is marked `no dollar data`.
- **Two daily passes.** Both are scheduled tasks, run Mon–Fri.
  - Forecast at 16:08 Sofia, after the bar labelled 14:00 Berlin closes. It logs HAR-X and HAR for the
    day.
  - Settlement at 23:08 Sofia, after the bar labelled 21:00 closes. It records the realised range and
    both squared errors.
  - The logs are append-only. A missed day is never backfilled.
- **One forward cell, owner choice of horizon.**
  - The cell is `gold_harx_vs_har_forward` at `alpha = 0.05`, in its own registry
    `results/gold_nyfwd/gold_nyfwd_hypothesis_log.csv`.
  - It is scored once:
    - after 60 settled eligible days counted from the pre-registration commit; or
    - on 2027-04-30, labelled `UNDERPOWERED — NO DECISION` if fewer than 60.
  - **The test is weak by the owner's choice.** If the true effect equals the historical one, the
    chance of a KEEP is only about 35 %. A DROP therefore does not show the model fails. The
    pre-registration states this.
- **Pre-registration first.** Nothing is logged as a forward day before
  `results/gold_nyfwd/PRE_REGISTRATION.md` and `.json` are committed.
- **MT5 chart.** A new read-only indicator, `ProphetGoldNYRange.mq5`, reads a TSV the passes rebuild.
  - It draws the forecast size band from 16:00 Sofia and, after 23:00, the realised range.
  - It shows size only: no direction, no entry, and no trade word.
  - Its corner shows the forward day count and says that no verdict exists yet.
- **Gross of cost on bid prices.** No verdict from this family is a trading result. It never writes
  `models/`, another family's files or any other registry.

## Capabilities

### New Capabilities
- `gold-nyrange-forward/live-data`: the read-only MT5 gold and dollar-index H1 caches, the label clock,
  the contract roll and its lag check.
- `gold-nyrange-forward/daily-passes`: the forecast and settlement passes, their causality, eligibility
  and append-only logs, and the scheduled tasks.
- `gold-nyrange-forward/forward-evaluation`: the pinned coefficients, the pre-registration gate, the one
  forward cell, its scoring and the report.
- `gold-nyrange-forward/chart-display`: the TSV export and the read-only MT5 indicator.

### Modified Capabilities
- None. The `gold-bb-nyrange` specs and its registries are unchanged; this family only reads them.

## Impact

- New code: `src/gold_nyfwd/`, `tests/test_gold_nyfwd_*.py`, `mql5/Indicators/ProphetGoldNYRange.mq5`,
  and `scripts/gold_nyfwd/` with the task, indicator install and uninstall scripts.
- New results: `results/gold_nyfwd/`. It holds the pre-registration, logs, registry, report and
  RUNBOOK. The live caches go in the gitignored `research_models/gold_nyfwd/`.
- New scheduled tasks: `EURUSDProphet-GoldNYForecast` (16:08) and `EURUSDProphet-GoldNYSettle`
  (23:08). The other loggers run at :01, :03 and :05, so these do not collide with them.
- One file outside the repo: `Common\Files\EURUSDProphet\gold_ny_range.tsv`.
- Docs: a new `ARCHITECTURE_DOCS.md` section and a `CLAUDE.md` invariant.
- No change to production, `models/`, `config.json`, the gold family, gold_bbnr or fx_ny_orb.
