# Design

## Context

See `proposal.md` (Why) for the data facts. These are the constraints that shape the
approach:

- `src/forecast_eval/` already contains the methodology as pure functions:
  - `splits` (purged walk-forward, uniqueness);
  - `targets`;
  - `stats` (Pesaran–Timmermann, Clark–West, DM-HLN, block bootstrap, Bonferroni);
  - `benchmarks`;
  - `prereg` (`n_required`, `fixed_point_family`, `committed_and_clean(path)`);
  - `overfit` (`append_trials(path=…)`, Deflated Sharpe, PBO).

  Its `study`, `forward_logger`, `costs`, `m15_data` and feature adapters are hard-wired
  to EUR/USD, to spread-based mid prices and to the forward arbiter's paths. That
  arbiter is mid-pre-registration (tasks 11.1, 12.3 and 14.11 are open).
- `src/pooled_h1_model.py::FEATURE_COLUMNS` is a 15-column H1 set: log returns at lags
  1/2/3/6/12/24, ATR/close, RSI, SMA distances in ATR units, EWMA std, and hour and
  weekday encodings. It is already scale-free, volume-free and spread-free, and it fed
  the only direction edge the project has confirmed.
- `src/features.py` production features include raw levels. `config.json` is pinned by
  three protected-set fixtures, so it cannot gain a gold section.
- Measured broker clock for `GOLD` (H1, 2013-08 to 2026-10):
  - weekly opens are labelled Monday 00:00 (643 of 701) or Sunday 23:00 (54, the US/EU
    DST-mismatch weeks), 99.4 % in total;
  - the daily break is the 23:00 label, moving to 22:00 in mismatch weeks;
  - the median spread is 40 points from 00:00 to 06:00 and 25 points thereafter.

  The spread figures are used only for this design note, never at runtime.

## Goals / Non-Goals

**Goals:**
- One code path per cadence, shared by validation scoring, the one-shot test and the
  forward logger, so the forward models are the studied models.
- Every gold number traceable to an artifact digest and a pre-registered rule.

**Non-Goals:**
- Generalising `forecast_eval` to multiple symbols. That is worthwhile later, but not
  while its arbiter is being registered.
- Hyper-parameter search beyond one small declared GBM grid. Neural configs are frozen
  copies of the EUR/USD ones.
- Any cost model. Per the owner's decision, spread is used nowhere, not even in reports.

## Decisions

### D1. A separate `src/gold/` package that imports `forecast_eval` read-only
**Choice:**
- Modules `data`, `clock`, `features`, `challengers`, `study`, `prereg`, `forward_logger`
  and `summary`, plus `gold_config.json`.
- Imports `forecast_eval.{splits,targets,stats,benchmarks,overfit}` and the power
  functions of `forecast_eval.prereg`, all of which take explicit paths.
- Imports `pooled_h1_data.resolve_symbol` and the OHLC fetch, and
  `pooled_h1_model.compute_pooled_features`.

**Alternative:** add a `symbol` parameter to `forecast_eval`. Rejected for now: it edits
modules whose behaviour the EUR/USD pre-registration is about to freeze.

**Alternative:** copy the stats code. Rejected: duplicated statistics drift silently.

### D2. Data: MT5 only, OHLC only, cached under `results/gold/`
**Choice:**
- The symbol is resolved by trying the ordered alias list `["GOLD", "XAUUSD"]` from
  `gold_config.json` through `resolve_symbol` and using the first that resolves.
  `resolve_symbol('XAUUSD')` does not match the broker's `GOLD`, so a single base name
  is not enough. The resolved name is recorded in each run record.
- D1 and H1 are pulled with `copy_rates_from_pos`. The H1 request is capped below the
  terminal's 100,000-bar `maxbars` limit; requesting more fails outright, as measured.
- Rows go through the shared coverage checks (`find_gaps`, `find_thin_months`), with two
  gold-specific rules found during implementation:
  - **Good Friday is exempt.** Gold is closed on Good Friday and FX is not. Every one of
    the 12 gaps over 72 h in 2013–2026 spans a Good Friday (D1 96 h, H1 74–96 h), and no
    other gap exists. Only gaps that span Good Friday and are at most 100 h are exempt.
  - **Leading thin H1 months are trimmed.** Before 2014-12 the broker's "H1" history
    carries about 21 bars a month (one a day). The H1 series therefore starts on
    2015-01-02, with 70,015 bars, not 70,507 from 2013-08 as the proposal estimated. A
    thin month after the first full one is still refused.
- Only `time, open, high, low, close` are written, to `results/gold/gold_d1.csv` and
  `results/gold/gold_h1.csv`.

**Alternative:** yfinance `GC=F` for a longer daily history. Rejected: it is a futures
contract, not the broker's CFD, and mixing instruments contaminates the sample.

### D3. Clock guard
**Choice:** `clock.verify(h1)` measures the share of weekly opens labelled Monday 00:00
or Sunday 23:00. It refuses below **0.95**, a registered threshold set against the
measured 0.994. It returns the share for the run record. It runs before features in
every entry point (study, test, logger).

**Found during implementation:**
- **Method.** The check predicts each week's open hour from Sunday 18:00 New York read on
  the Berlin clock, as `m15_data.verify_clock` does for FX at 17:00, instead of accepting
  "Monday 00:00 or Sunday 23:00". That is stricter about DST weeks.
- **Live result: 97.6 % of 624 weeks.** The 15 misses are two holiday-delayed opens at
  01:00 and one contiguous anomaly: from 2016-11-13 to 2017-02-19 (14 weeks) the broker
  labelled opens at 23:00, so its clock ran one hour off Berlin that winter. Eligibility is
  gap-based and unaffected. Hour-of-day features in those training weeks are one hour
  off, and they are kept as recorded, never repaired.
- **The forward logger adds a live-window check.** Because broker clocks demonstrably
  change, the logger also verifies the last 26 weeks of its own window at 0.6, mirroring
  the EUR/USD logger's `LIVE_CLOCK_MATCH_MIN`. A clock change then stops logging within
  weeks, not years.

### D4. Features
**H1:** `compute_pooled_features` on the gold H1 frame, unchanged (15 columns).

**D1 (gold's own, scale-free):**
- `log_return` and lags 1–5;
- `range_pct = (high − low)/open` and lags 1–3;
- `vol_20` (std of log returns) and `ewma_vol_20`;
- `atr14_pct = ATR14/close`;
- `bb_width`;
- `dist_sma_{21,50,200} = close/SMA − 1`;
- `rsi_14`;
- `dow_sin/cos` and `month_sin/cos`.

A guard rejects any column named like a raw level (`open|high|low|close|SMA_n|ATR_n`
without a normalising suffix). A unit test multiplies OHLC by 3.0 and asserts identical
features.

**Alternative:** reuse `src/features.py` with levels dropped. Rejected: it would still
carry `SMA_*` and `ATR_14` levels, and its PCA-on-lags step is EUR/USD-specific
plumbing.

### D5. Targets and eligibility
**Targets:** from `forecast_eval.targets` at h = 1: `return_pct_target` (log return in
percent), `direction_target` and `volatility_pct_target`. The price is derived only in
reports and the view.

**H1 eligibility:** the time difference between the as-of and target labels is exactly
1 h, and the as-of bar itself follows its predecessor by exactly 1 h. This excludes the
last bar before every break and the first bar after it, whatever the clock does in
mismatch weeks.

**D1:** close-to-close. Every daily close sits at the same break, so its bid-side bias
largely cancels in the difference.

### D6. Challengers and baselines (cells)

| cell | cadence | model | produces | families |
|---|---|---|---|---|
| `d1_gbm` | D1 | XGB classifier + pseudo-Huber regressor | p_up, ret | direction*, return |
| `d1_lstm` | D1 | multi-task LSTM, 5 seeds, steps 20, units 64, dropout 0.3 | p_up, ret, vol | direction*, return, volatility |
| `d1_garch` | D1 | GARCH(1,1), train-only fit | vol | baseline |
| `h1_gbm` | H1 | XGB classifier + pseudo-Huber regressor | p_up, ret | direction, return |
| `h1_lstm` | H1 | multi-task LSTM, 5 seeds, steps 24, units 32, dropout 0.2 | p_up, ret, vol | direction, return, volatility |
| `h1_garch_hod` | H1 | GARCH(1,1) on returns ÷ train-only hour-of-day mean \|r\|, rescaled | vol | baseline |

`*` marks cells expected to be registered `UNDERPOWERED — NO DECISION` by the arithmetic
in D7.

- **GBM:** one declared grid (`n_estimators {200,400}`, `max_depth {3,5}`,
  `learning_rate {0.03,0.1}`), tuned by inner `TimeSeriesSplit(5)` on training rows
  only. Every grid point is appended to `results/gold/trial_log.csv`.
- **LSTM:**
  - heads: return (Huber), direction (BCE) and volatility (MAE, softplus output);
  - early stopping on the last 1/8 of training;
  - one `StandardScaler` fitted on training rows;
  - the ensemble is the mean of seeds 42–46, all or nothing.

**Alternatives:**
- Transformers / PatchTST / N-BEATS: rejected on sample size (D1 has about 2,400
  training rows), where no evidence shows them beating LSTM or GBM.
- Kronos zero-shot: deferred (see proposal, out of scope).
- EGARCH/GJR: kept out of the registered baselines to avoid baseline-shopping. They can
  be reported descriptively.

### D7. Families, alpha and verdict rules (registered in D9)
**Direction:**
- admitted cells are those whose eligible validation n ≥ `n_required(α_family, edge=0.03)`,
  solved by `fixed_point_family`;
- expected admission: `h1_gbm` and `h1_lstm` (about 6,400 eligible validation rows
  against roughly 2,700 required at α = 0.025);
- D1 cells (about 347 rows against more than 3,000 required) are UNDERPOWERED;
- verdict: the paired block-bootstrap CI (`block_length(1,'H1')`) of the accuracy minus
  the training-majority accuracy on the same rows must exclude 0 at 1 − α;
  Pesaran–Timmermann is reported alongside.

**Return:**
- all four return-producing cells, at α = 0.05/4;
- verdict: Clark–West against the zero forecast, one-sided, p < α.

**Volatility:**
- `d1_lstm` vs `d1_garch` and `h1_lstm` vs `h1_garch_hod`, at α = 0.05/2;
- verdict: the block-bootstrap CI of MAE_baseline − MAE_model must exclude 0.

**Shared rules:**
- Registries: `results/gold/gold_{direction,return,volatility}_hypothesis_log.csv`,
  append-only, with columns modelled on `volatility_hypothesis_log.csv`.
- Descriptive extras (never verdicts): accuracy by hour of day, flagged if more than
  half the excess accuracy comes from the 00:00–06:00 labels (the high-spread hours); DSR
  of a sign strategy, gross, over the trial log; and PBO across cells.

### D8. Stages and guards
1. **`validation`:** fit on [0:70], score [70:80].
2. **`test`:** refit on [0:80], score [80:100] once. The guard refuses if
   `results/gold/test_report.json` exists in HEAD. It also requires every validation
   registry row to be committed.
3. **`final`:** refit on [0:100] and save to `research_models/gold/<cell>/`, with
   SHA-256 digests in `results/gold/artifact_manifest.csv`. These frozen artifacts feed
   the forward logger.

All three stages share `study.run(stage)`. The stage only changes the split and the
output file.

### D9. Pre-registrations
- `results/gold/PRE_REGISTRATION.md` plus `.json` are committed before stage 1. They
  record cells, families, α, the threshold 0.95, the eligibility rule, the grid, configs,
  CI method (2,000 resamples) and expected outcomes.
- Expected outcomes:
  - D1 direction: UNDERPOWERED, about 50 %;
  - H1 direction: possible CLEARED at 51–53 %, from the EUR/USD/AUD/CHF analogue;
  - all return cells: DROP (shrinkage to zero);
  - both volatility cells: CLEARED.
- `results/gold/forward/PRE_REGISTRATION.md` is committed after a dry run of at least
  two weeks and before any forward scoring. It records the forward power table at the
  dry run's measured coverage, a 3-year cap, a minimum window of 6 months, frozen
  artifacts for the window, and "gross of cost".
- Both gates use `prereg.committed_and_clean(path)`.

### D10. Forward logger
- **Schedule:** scheduled task `EURUSDProphet-GoldLogger`, hourly at :03, installed by
  `scripts/gold/install_task.ps1`.
- **Each run:**
  1. take the lock `research_models/gold/forward.lock`;
  2. verify the clock;
  3. verify the manifest digests;
  4. append forecasts for the newest closed H1 bar and day whose target is still open;
  5. write gap records for missed as-of bars;
  6. settle matured rows with the D5 targets and eligibility.
- **Outputs:** `results/gold/forward/{predictions,settlements,gaps,failures}.csv`.
- **Phase:** a `phase` field (`dry_run` | `registered`) is stamped on every row from the
  forward pre-registration's presence.
- **Isolation:** no import of `forecast_eval.forward_logger`, and no shared file.

### D11. Read-only view
- `src/gold/summary.py::build()` reads the forward CSVs, the registries and the test
  report. It returns the newest row per cell taken whole, and drops rows whose target bar
  has closed (bar-label clock: `Europe/Berlin` now versus the label).
- It derives `implied_price = close·exp(ret/100)`.
- **Routes:** `api.py` gains `GET /api/gold` (JSON) and `GET /gold` (serves
  `static/gold.html`, bilingual BG/EN like the home page). `static/index.html` gains one
  link.
- A test asserts that building the payload changes no file and imports no model library.

## Risks / Trade-offs

- **[A trending test block flatters any up-biased model]** Gold rose from about $2,000 to
  $4,100 across the test block, so an "always up" model can score above 50 %.
  → Direction is judged against the training-majority accuracy on the same rows, and
  Pesaran–Timmermann (which conditions on both marginals) is reported.
- **[Bid-only prices without spread can manufacture intraday skill]** → The D5 gap
  eligibility, plus the hour-of-day concentration flag in D7. Measured exposure is
  modest: 40 against 25 points, about 1.6×, whereas EUR/USD's rollover was about 5×. This
  is weaker than the mid-price parity check and is stated as such in the
  pre-registration.
- **[Gross verdicts can be misread as tradable]** → Fixed wording in the specs. No P&L,
  pips or ledger anywhere. Any trading use needs a later change that adds a cost model.
- **[D1 sample is too small to decide direction]** → Registered UNDERPOWERED by
  arithmetic, never after seeing results.
- **[Frozen forward models age]** → The window is short (at least 6 months) and
  registered. Refitting requires a new registration, never a quiet retrain.
- **[An automatic committer has been active in this repo (2026-10-21)]** It could commit
  results or bundle files under a generic message. → Commit the pre-registrations and
  registries by hand with explicit messages. The guards check commit ancestry, not
  message text.
- **[CPU time for H1 LSTM ensembles]** About 49k training rows × 5 seeds × 3 stages,
  likely 1–2 h per stage on CPU. → Acceptable for a one-off study. The forward logger
  only runs inference.
- **[MT5 `maxbars` = 100,000]** H1 grows by about 6k bars a year and reaches the cap
  around 2031. → The cached CSV is append-only, so older bars are kept on disk.

## Migration Plan

The change is purely additive. To roll back: run `scripts/gold/uninstall_task.ps1`,
remove the two routes and the link, and delete `src/gold/` and `results/gold/`. Nothing
in production depends on any of it.
