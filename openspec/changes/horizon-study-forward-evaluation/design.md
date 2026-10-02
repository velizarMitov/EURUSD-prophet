# Design

## Context

See `proposal.md` (Why) for motivation, and the three specs under
`specs/forecast-evaluation/` for requirements. This section covers only the current
state that constrains *how* the change is built.

- **Pinned code.** `_train_pipeline.py`, `src/features.py`, `config.json`,
  `src/h1_direction_model.py`, `src/volatility.py` and `src/inference.py` are pinned by
  `tests/fixtures/*_protected_sha256.json`. The daily GBM/LSTM training lives inline
  in `train_variant()`, with a hard-coded 1-day target and output to `models/<variant>`.
  The H1→daily ensemble is trained inline in the same file. None of it can be called
  with another horizon. The owner chose to build the study as a new standalone
  system rather than modify or copy this code.
- **Feature code is importable.** Daily features come from
  `src/features.py::compute_features` (inference-safe, no target), with
  `PRICE_FEATURE_COLUMNS` (23) and `FEATURE_COLUMNS` (27). The H1 next-bar features
  come from `compute_pooled_features`, called by `build_direction_dataset`, which
  separates context from target. TI-LSTM indicators come from
  `enrich_h1_with_indicators`. The H1→daily features come from
  `src/h1_features.py::build_h1_datasets`. All of these can be imported read-only;
  where a builder also returns a target, the study discards it and builds its own.
- **Data on disk.** EURUSD H1 runs 2015-04-27 → 2026-07-28 in
  `results/pooled_h1/EURUSD_h1.csv` (70,000 bars), plus the rolling
  `results/eurusd_h1.csv` to the present. The daily euro-era matrix has 8,605 rows.
  M1 spreads per instrument are in `results/curl/m1_coverage.csv` (EURUSD median
  5 points = 0.5 pip).
- **Live access.** Every MT5 call in the repo is a bare `mt5.initialize()`. It attaches
  to whatever account the running terminal holds, today **real account 1007437,
  ActivTradesEU-Server**, with AutoTrading off. MT5 rates carry a per-bar `spread`
  field, but `src/live_data.py` drops it.
- **Forward logs today** are written only on API calls, have no price-source column,
  and cover a small fraction of bars.
- **Hardware.** The owner's Windows machine. `ti_lstm_h1_experimental.require_cuda()`
  shows that the existing TI-LSTM path assumes CUDA.

## Goals / Non-Goals

**Goals:**

- One scoring library used identically by the historical study and the forward
  arbiter, so that "how a forecast is scored" exists once.
- Horizon targets and cost breakevens defined in one place each, and tested.
- Every challenger runs on CPU. GPU is optional and never required.
- Unattended forward operation that degrades into recorded gaps, never into silent
  fallback data or retroactive rows.
- Full traceability: forward row → model version → artifact digest → training range →
  feature-code digest.

**Non-Goals:**

- Improving, retraining or replacing any production model, or serving a challenger.
- Forward evaluation of instruments other than EURUSD.
- Horizons shorter than one H1 bar.
- Any order, sizing, stop-loss or broker-write capability.
- Hyperparameter search beyond the grids declared in the study record.

## Decisions

### D1 — A new package, `src/forecast_eval/`, with its own challengers

The layout separates scoring (shared) from models (study) from operations (forward):

```
src/forecast_eval/
  splits.py        purged walk-forward, CPCV, label uniqueness
  targets.py       horizon direction/return/volatility targets, breakeven
  costs.py         spread/round-trip/swap cost model
  benchmarks.py    coin/majority, zero-return RW, random-sign matched turnover
  stats.py         Pesaran–Timmermann, Clark–West, DM-HLN, block bootstrap,
                   Bonferroni, Romano–Wolf
  overfit.py       PBO (CSCV), Deflated Sharpe, append-only trial log
  features.py      read-only adapters over the project's feature builders
  challengers/     one module per model type (daily_gbm, daily_lstm,
                   h1_daily_ensemble, ti_lstm, h1_gbm, vol_ensemble, kronos)
  study.py         runs the horizon study, writes descriptive curves
  mt5_reader.py    read-only MT5 bars + spread + swap + server name
  forward_logger.py  predict / settle / gap records
  refit.py         monthly expanding-window refits, manifest
  prereg.py        power table, fixed-point family, registry rows
```

Each challenger exposes the same narrow interface: `fit(X, y, cfg, seed)`, then
`predict(X)` returning a direction probability, plus a return or volatility estimate
where the type has one.

*Alternatives considered:* extracting `train_variant` into parameterised functions,
which touches pinned files and needs a fixture re-baseline; and copying
`train_variant` into the study, which breaks the repo's "never copy-paste it" rule and
invites silent drift. The owner chose new challengers. Findings therefore describe the
challengers, not the production models, and that is stated in every report.

### D2 — Challenger configurations start from production values, declared once

Each challenger's declared configuration starts from the production hyperparameters in
`config.json`, read-only, so challengers stay comparable with the production line.
Any departure (for example a CPU-sized TI-LSTM) is written into the study record
before the first fit. The record stores a configuration hash, and the same hash must
appear at every horizon.

### D3 — Targets on log returns, one function per target

Direction is `sign(log(close[t+h] / close[t]))`, and zero moves are excluded and
counted. The return target is the percent log return over h, and the volatility
target is `|log return over h| × 100`. All three live in `targets.py`, and a test
asserts that the study scales by 100 nowhere else. The production invariant ("`* 100`
only in `src/features.py`") governs the production pipeline and is not affected. Log
returns make multi-step targets additive, which keeps the purge and uniqueness
arithmetic exact.

### D4 — Splitting parameters

- **Primary development view:** expanding-window purged walk-forward. Purge = h,
  embargo = max(h, 1 bar). This mirrors how the forward window will refit.
- **Distributional view:** CPCV with N = 6 and k = 2, giving 15 splits and 5 paths per
  cell. These paths feed PBO.
- **Bootstrap block length:** max(h, 24) for H1 cadence and max(h, 5) for daily.
  Earlier work measured |return| autocorrelation at lag 24 on H1; direction sign is
  not autocorrelated.

### D5 — Development uses the full history, labelled descriptive

The study reads all EURUSD history, including the spent blocks, because the arbiter is
forward data and no study output adjudicates. This follows the precedent of
`src/walk_forward_validation.py`. *Alternative:* restricting development to `[0:70%]`
only makes the curves noisier and protects nothing, since no verdict is drawn from
them. Every output carries a `DESCRIPTIVE — not a verdict` banner and its date range.

### D6 — Costs

- **Development:** the measured instrument spread (EURUSD 0.5 pip) is the primary cost.
  The `config.json` 1.5-pip round trip is the sensitivity case.
- **Forward:** the entry bar's own MT5 `spread` field is recorded, and the registered
  cost level is applied at scoring.
- **Swap:** read from MT5 `symbol_info` at logging time (read-only). It is charged in
  net P&L only when a hold crosses the server rollover. It is excluded from the
  breakeven, because the admitted cells hold at most 4 h.
- **Commission:** defaults to 0 until confirmed (see Open Questions).

Breakeven per cell = `0.5 + c / (2 · E|r_h|)`, where `E|r_h|` comes from development
data at that horizon.

### D7 — Statistics written fresh, with parity tests

`stats.py` implements the tests itself rather than importing private helpers such as
`h1_direction_model._block_bootstrap_delta`. A parity test runs the new block bootstrap
and the existing helper on the same synthetic input and requires identical intervals,
so the math matches the repo's established method without depending on private
names.

### D8 — Overfitting accounting

The trial log is `results/horizon_study/trial_log.csv`. It is append-only, with one
row per model × horizon × configuration ever fitted, frozen and tuned variants
included. The Deflated Sharpe uses its cumulative row count. PBO uses CSCV with
S = 16 submatrices over each cell's candidate set.

### D9 — Tuning rule (one configuration per cell leaves the study)

The frozen declared configuration is the default. A tuned variant comes from nested
purged CV over a grid declared in the study record. It replaces the frozen one only if
the CPCV median net return per trade improves by at least the declared threshold
**and** PBO < 0.5. Both variants are logged as trials.

### D10 — Research artifacts outside git, manifest inside

Artifacts live at `research_models/horizon_study/<family>/h<h>/<version>/`, which is
gitignored. Monthly refits of about 34 cells, many of them LSTMs, would bloat the repo.
The manifest `results/horizon_study/artifact_manifest.csv` is tracked. It holds the
path, SHA-256, family, horizon, configuration hash, training range, and the digest of
every imported feature module. *Alternative:* tracking artifacts in git was rejected
for size. The risk of losing the local disk is listed below.

### D11 — Forward logger as a scheduled standalone process

`python -m src.forecast_eval.forward_logger` is run by Windows Task Scheduler every
hour, 60 s after the bar close. It touches neither `api.py` nor
`src/inference.py`, so serving and the additive-only contract are unaffected.

- **One run:** take a lock file, read MT5 through `mt5_reader`, then for each model
  whose cadence has a newly closed bar, load the latest manifest version with
  `train_end < as_of`, predict at every horizon, and append rows. Daily-cadence
  models act only when a new D1 bar has closed at the server's day boundary.
- **Idempotent:** the key is (model, h, as_of_bar). A repeat run is a no-op.
- **No backfill:** if bars were missed, append one `gap` record per missed bar and
  never predict them.
- **Settle step:** in the same run, append settlement rows for forecasts whose target
  bar has closed, using MT5 closes only.
- **Source rules:** any non-MT5 source, or a server other than the registered one, is
  logged and excluded from scoring. The account login is never written.
- **Feature-code guard:** if the digest of an imported feature module differs from the
  one the model version was trained with, write a failure record instead of a
  prediction.
- **Storage:** append-only CSVs under `results/forward_eval/`: `predictions.csv`,
  `settlements.csv`, `gaps.csv`, `failures.csv`.

### D12 — Monthly walk-forward refits

Refits run on the first weekend of each month, while the market is closed. They use an
expanding window to the last closed Friday bar and the frozen configuration, and append
new manifest versions. If a refit fails, the previous version stays live and a failure
is recorded. A refit can never see bars after its own fit time, because the fit time
falls on a weekend after the window's last bar.

### D13 — Pre-registration and the fixed-point family

`results/forward_eval/PRE_REGISTRATION.md` is committed before scoring mode is switched
on. The logger refuses to write settlements in scoring mode unless that file exists and
its commit is an ancestor of HEAD.

**Direction family composition.** Bonferroni, power 0.8, an edge of 3 pp over
breakeven, and a cap of 3 years time-to-decision. Admission is iterated to a fixed
point:

| step | family F | alpha | cells admitted |
|---|---:|---:|---:|
| 1 | 34 (every direction cell) | 0.00147 | 6 |
| 2 | 6 | 0.00833 | 6 — stable |

The **admitted cells** are the H1 GBM challenger and Kronos direction at h = 1, 2 and
4 bars. Assuming full bar coverage, they need 0.5, 1.1 and 2.2 years respectively. The
other 28 direction cells (H1 at h ≥ 6, which needs 3.3–65 years, and every daily-cadence
cell, which needs 13–65 years) are `UNDERPOWERED — NO DECISION` by design: logged,
reported, and spending zero alpha.

The coverage measured during the dry run (D14) scales these times before registration,
and the fixed point is recomputed with it.

The **volatility family** has its own power table: the challenger ensemble at
{1, 2, 5} days, plus Kronos volatility at 24 bars, each measured by ΔMAE against
GARCH(1,1) × day-of-week. Its variance comes from the development curves.

**Expected outcome, stated in advance.** H_dir.1 measured 52.96 % against a 53.32 %
breakeven at 0.5 pip. The H1 h = 1 cells are therefore expected to end
"predictive, not cost-viable" or undecided.

### D14 — Dry run before registration

The logger runs for at least two weeks in `phase = dry_run`. Those rows are never
scored. The dry run measures real coverage (the fraction of bar closes logged), feeds
it into the power table, and shakes out operational failures before any alpha exists.

## Risks / Trade-offs

- **[Machine sleeps or is offline → gaps]** → Gap records make coverage visible. The
  power table uses the measured coverage. The pre-registration states the coverage
  assumption, and the owner disables sleep or moves to an always-on host.
- **[Terminal logged into a different account or server]** → The server name is
  recorded per row. Rows from any server other than the registered one are excluded
  from scoring.
- **[The logger attaches to a real-money account]** → `mt5_reader` exposes only
  rate, symbol-info and account-server reads. An AST test fails if any module in
  `src/forecast_eval/` references `order_send`, `order_check`, `positions_*`,
  `orders_*` or any other trade function. Keeping terminal AutoTrading off is
  recommended in the runbook.
- **[Kronos direction was already retired as uninformative]** → Including it, at the
  owner's request for all models, takes 3 of the 6 family slots. That makes the bar
  for the H1 GBM challenger stricter than a 3-cell family would (0.00833 vs 0.0167).
  This is a deliberate cost of the "all models" decision, stated in the registration.
- **[A change to production feature code mid-window]** → The feature-module digest is
  checked at prediction time. A mismatch produces failure records, not silently
  shifted features.
- **[Challengers ≠ production models]** → Every report says that findings describe
  the challengers. Promotion is a separate decision.
- **[Overlapping horizons make cells correlated, so Bonferroni is conservative]** →
  Romano–Wolf is reported alongside. It is not adjudicated on, consistent with every
  other family in the repo.
- **[Interim figures invite early stopping]** → Verdicts are issued only at the
  registered n. The dashboard and CSVs label running numbers "interim, not
  adjudicating".
- **[Losing the local disk loses the artifacts]** → The manifest keeps the digests, so
  the record stays verifiable, and an off-machine backup is optional. Lost artifacts
  invalidate only future predictions, never logged ones.
- **[CPU compute for monthly LSTM refits]** → The first study run measures compute
  time. If refits exceed the weekend window, the refit cadence is lengthened before
  registration, never after.
- **[A single forward regime]** → This cannot be fixed. Results are also reported per
  quarter so that regime dependence is visible.

## Migration Plan

1. **Phase 0 — harness.** `splits`, `targets`, `costs`, `benchmarks`, `stats` and
   `overfit`, with tests. No data is read.
2. **Phase 1 — horizon study.** Declare the study record (grid, configs, tuning grids,
   cap), fit the challengers on history, and write the descriptive curves, PBO, DSR
   and manifest.
3. **Phase 2 — forward dry run.** Install the scheduled task and log for at least two
   weeks in `dry_run`. Measure coverage and failures.
4. **Phase 3 — registration.** Compute the power table with the measured coverage, fix
   the families, commit `PRE_REGISTRATION.md` and the empty registries, then switch
   the logger to scoring mode.
5. **Phase 4 — operation.** Monthly refits. Verdicts are issued at the registered n
   only.

**Rollback:** disable the scheduled task. Nothing in serving depends on any of this.
`research_models/` can be deleted, and the tracked CSVs remain as a historical record.

## Open Questions

- **Commission on the trading account.** If ActivTrades charges a per-lot commission on
  account 1007437, it enters the cost model before registration. It changes only a
  number, not the design.
- **Off-machine backup** of `research_models/`, and whether to move the logger to an
  always-on host later. Both are operational only.
