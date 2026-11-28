# Design

## Context

- **What exists.** See proposal.md for the motivation. gold_bbnr froze HAR-X (OLS on fourteen features)
  and HAR (OLS on the three HAR lags). Both were fit on all 2 442 usable days up to 2026-11-20 and
  stored in the gitignored `research_models/gold_bbnr/ols.json`. Its digest is in
  `results/gold_bbnr/frozen_manifest.csv`.
- **Historical inputs.** gold_bbnr's feature, target and eligibility code works on any H1 frame. Its
  loader `bars.load_h1` does not fit live use: it is pinned to the gold family's cache and to
  `PINNED_LAST`.
- **The live pattern to copy.** fx_ny_orb's [A] forward part:
  - live passes on the label clock from `forecast_eval.m15_data.label_now`;
  - append-only CSV logs, a git pre-registration gate and `fx_fib.score.test_cell`;
  - a TSV export with a read-only MQL5 indicator, and schtasks via a hidden VBS runner.
- **Where the dollar index comes from.** It exists only in the `mt5app` terminal, as `USDIndDec26`,
  which expires on 2026-12-11. No successor is listed yet.
- **Busy task minutes.** The other loggers run at :01, :03 and :05.

## Goals / Non-Goals

**Goals:**
- Live forecasts identical to what gold_bbnr's code would compute for the same bars.
- A forward cell decided only on days after the pre-registration commit, scored once.
- A chart that shows size only.

**Non-Goals:**
- Refitting, recalibrating or extending any model.
- LightGBM or the MLP. They were DROP against HAR-X and are not forecast live.
- Any direction, signal, trade, cost or P&L.
- Backfilling days, or reading the gold family's live cache.

## Decisions

**D1. A new package `src/gold_nyfwd/`, separate from gold_bbnr.**
- gold_bbnr is a closed historical study, with a pinned loader and scored registries.
- A forward family has a different lifecycle, with live caches, tasks and its own registry. Keeping
  them apart means a live bug can never touch the historical verdicts.
- It imports gold_bbnr's `bars.d1_ny`, `ny_target` and `ny_features` and calls them with gold_bbnr's
  own config, read-only. It never imports `bars.load_h1` or `inputs.load`.
- Alternative: add a `live.py` to gold_bbnr. Rejected because gold_bbnr's protection tests and
  invariants would then cover a living module.

**D2. Coefficients are copied into the pre-registration.**
- `PRE_REGISTRATION.json` holds the HAR-X and HAR coefficient lists, the column order and the
  `ols.json` SHA-256.
- Passes predict from the JSON. If `ols.json` still exists, they also refuse when its digest differs.
- Reason: `research_models/` is gitignored and could be lost. The committed JSON is then the only copy,
  and it is exactly what was registered.

**D3. The family keeps its own MT5 caches.**
- Caches: gold H1, one cache per `USDInd` contract, and EURUSD H1 for the lag check. They live under
  `research_models/gold_nyfwd/cache/`.
- Each pass tops them up read-only through market-data calls, the last 4 000 closed bars (about
  8 months). That covers the 20-day HAR lags and ratio windows.
- The terminal path is in the family's config (`C:\Users\test\Desktop\mt5app\terminal64.exe`).
- Alternative: read `results/gold/gold_h1.csv`. Rejected because that file belongs to another
  family's hourly task and its timing.

**D4. The forecast pass truncates explicitly.**
- The pass builds the frames from cached bars labelled ≤ 14:00 on the day. It cuts the dollar index the
  same way.
- It calls `ny_features.dataset(..., read_target=True)` on that frame. Today's row is therefore
  `incomplete session`, and its features use only earlier targets (HAR lags use
  `allow_exact_matches=False`).
- The forecast uses today's features and ignores today's eligibility reason. Only a missing feature
  (warm-up) or a missing pre-session bar blocks it.
- A late pass, say at 18:00, gives the same row as one at 16:08.

**D5. Settlement re-runs gold_bbnr's eligibility and target.**
- It runs on the cache through the bar labelled 21:00, at 23:08 Sofia (22:08 Berlin).
- The NY-close D1 validity counts bars only up to 21:00, not 22:00. It needs at least 12 H1 bars,
  and a normal day has about 22, so this differs only on near-empty days.
- It also records `realised_high` and `realised_low` over bars 15–21 for the chart.

**D6. The contract choice is a pure function of the day and the listed expiries.**
- Rule: the listed `USDInd` contract with the nearest expiry at least two trading days after the day.
- The same rule decides which contract's bars give each earlier day's pre-range in the 20-day mean. The
  feature is therefore reproducible, and a roll needs no special code.
- Before first use, a contract must pass the lag-0 check against EURUSD: the negative correlation of
  hourly returns must peak at lag 0 within ±3 h. Its result is recorded in
  `research_models/gold_nyfwd/contracts.json`.
- Alternative: build a synthetic dollar series from EURUSD. Rejected because it would be a different
  feature from the frozen model's.

**D7. Scoring.**
- One cell at `alpha = 0.05`, family size 1.
- The statistic is the mean of `se_harx − se_har` over forward days, tested with `fx_fib.score.test_cell`
  using a 5-day block, 10 000 resamples and seed 20261201.
- KEEP if the upper bound is below 0.
- The horizon is 60 forward days or 2027-04-30. Forward days start the day after the commit date on
  the label clock.
- Power from gold_bbnr's historical difference (mean −0.0276, block SE 0.0037 at 2 211 days, scaled to
  60 days): about 34 %. The pre-registration states this.

**D8. Clock and tasks.**
- `label_now()` is the clock. Passes refuse on weekends and before their window:
  - forecast after the close of the 14:00 label bar, i.e. 15:00 Berlin;
  - settlement after the close of the 21:00 label bar, i.e. 22:00 Berlin.
- Tasks run at 16:08 and 23:08 Sofia, avoiding the :01, :03 and :05 loggers.
- The scripts copy fx_ny_orb's: `run_hidden.vbs`, `run_module.cmd`, install, uninstall and `-DryRun`.
- Logs go to `research_models/gold_nyfwd/<task>.log`.

**D9. Export and indicator.**
- `gold_ny_range.tsv` follows the layout of `ny_orb.tsv`: a header line with the forward count and the
  verdict text, then one row per day.
- `ProphetGoldNYRange.mq5`:
  - copies ProphetNYOrb's file-reading and object code;
  - draws only on symbols whose name starts with `GOLD`, at H1 and below;
  - draws a band of ±F/2 around close14 for 15:00–22:00 Berlin and a dotted realised box;
  - uses no trade words in its strings.
- Python writes every visible string.

## Risks / Trade-offs

- **No `USDInd` successor is listed by 2026-12-10.** Every day is then `no dollar data` and the test
  stops counting. → The RUNBOOK says to check MT5 weekly in early December. The pre-registration
  states that such days never count and that the horizon is not extended.
- **Low power, about 34 % at 60 days.** This is the owner's choice. → It is stated in the
  pre-registration, the report and the chart corner. A DROP is labelled as not evidence of failure.
- **The frozen model was fit on all history.** The forward effect may be smaller than the
  walk-forward's. → That is what the forward test measures, and it is stated.
- **PC off during the session.** The day has no forecast, and the sample grows more slowly. → It is
  never backfilled. The report counts missed weekdays.
- **The broker changes the gold clock or the symbol.** → The clock check makes the day
  `unverified clock week`, and an unresolved symbol makes the pass refuse.
- **Two MT5 clients at the same minute.** → The 16:08 and 23:08 slots are free.

## Migration Plan

1. Implement and test with fake feeds. No real pass logs before step 3.
2. Write the pre-registration, which copies the coefficients, and show it to the owner.
3. The owner commits it. `prereg --check` must pass.
4. Install the tasks and the indicator.
5. Rollback: the uninstall scripts remove the tasks, indicator and TSV. The logs stay as a record.
