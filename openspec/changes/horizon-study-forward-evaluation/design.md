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
- **Sub-hourly data on disk.** `results/eurusd_m15.csv` holds 350,000 M15 bars,
  2012-06-25 → 2026-07-24, but it is byte-pinned and carries **no spread column**.
  `results/curl/raw/EURUSD_M1.parquet` holds 2,980,060 M1 bars,
  2018-08-08 → 2026-08-07, and does carry `spread_points` and `point` per bar. It is
  excluded from git by `DATA.md` §7 and regenerable from `src/curl_mt5_fetch.py`.
  `pyarrow` is present in the environment although it is not a declared dependency.
  Aggregating it to M15 gives 199,097 bars, of which 62,238 fall inside the owner's
  session over 2,075 session days, and 60,163 are eligible at a 15-minute horizon under
  the strict rule of D16.
- **The bar labels are not UTC.** They are Europe/Berlin wall clock, established by
  measurement in D16. Every session rule in this change is expressed against that
  clock.
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
- Horizons shorter than one M15 bar, and tick or order-book data.
- Sub-hourly evaluation outside the owner's declared session.
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
  m15_data.py      M1 -> M15 mid aggregation, session window, bid/mid pair
  challengers/     one module per model type (daily_gbm, daily_lstm,
                   h1_daily_ensemble, ti_lstm, h1_gbm, m15_session,
                   vol_ensemble, kronos)
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
  not autocorrelated. For M15 the block is max(h, 26) — one full session — because the
  session restriction makes consecutive rows neighbours within a session, so the
  session, not the calendar day, is the dependence unit.

### D5 — Development uses the full history, labelled descriptive

The study reads all EURUSD history, including the spent blocks, because the arbiter is
forward data and no study output adjudicates. This follows the precedent of
`src/walk_forward_validation.py`. *Alternative:* restricting development to `[0:70%]`
only makes the curves noisier and protects nothing, since no verdict is drawn from
them. Every output carries a `DESCRIPTIVE — not a verdict` banner and its date range.

### D6 — Costs

- **Development:** the measured instrument spread (EURUSD 0.5 pip) is the primary cost.
  The `config.json` 1.5-pip round trip is the sensitivity case. For M15 cells the
  measured level is the median spread of in-session bars only, which is also 0.50 pip
  (p90 0.60 pip) — the session is the calmest spread window of the day.
- **Forward:** the entry bar's own MT5 `spread` field is recorded, and the registered
  cost level is applied at scoring.
- **Swap:** read from MT5 `symbol_info` at logging time (read-only). It is charged in
  net P&L only when a hold crosses the server rollover. It is excluded from the
  breakeven, because the admitted cells hold at most 4 h.
- **Commission:** none. The owner confirmed on 2026-10-02 that ActivTrades charges no
  commission, so the spread is the only trading cost.

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

**Verdict criterion.** The owner chose on 2026-10-02 to judge direction against a
coin (50 %), not against the spread breakeven: the forecast is used for direction, with
holds far longer than the bars the spread would dominate. KEEP iff the one-sided lower
bound exceeds 50 % at the registered n, else DROP. The breakeven and the net result per
trade are reported beside every verdict and never change it.

**Direction family composition.** Bonferroni, power 0.8, an edge of 3 pp over 50 %,
and a cap of 3 years time-to-decision. The arithmetic is unchanged from the breakeven
version: n depends on the size of the edge, not on where it is measured from. With the
M15 session cells (D15–D17) the candidate set is 46 cells — the 34 original direction
cells plus 2 session challengers × 6 M15 horizons. The M15 rate is **measured per
horizon** rather than taken as rate/h, because the strict eligibility rule of D16 costs
more at longer horizons: 7,523, 3,632, 1,816, 778 and 259 non-overlapping trades a year
at h = 1, 2, 4, 8 and 16. Admission is iterated to a fixed point:

| step | family F | alpha | cells admitted |
|---|---:|---:|---:|
| 1 | 46 (every direction cell) | 0.001087 | 10 |
| 2 | 10 | 0.005000 | 12 |
| 3 | 12 | 0.004167 | 12 — stable |

The iteration is not monotone: at step 1 the alpha is tight enough to exclude the 4-hour
H1 cells, and relaxing it at step 2 lets them back in. Step 3 is self-consistent — with
12 cells at alpha = 0.05/12 = 0.004167, exactly those 12 decide within the cap. The
registered n is **3,817** forecasts per cell.

The **admitted cells** are:

| cell | horizon | trades / year | years to decide |
|---|---|---:|---:|
| `m15_session_gbm`, `m15_session_lstm` | 15 min | 7,523 | 0.51 |
| `m15_session_gbm`, `m15_session_lstm` | 30 min | 3,632 | 1.05 |
| `m15_session_gbm`, `m15_session_lstm` | 1 h | 1,816 | 2.10 |
| `h1_gbm`, `kronos_direction` | 1 h | 6,170 | 0.62 |
| `h1_gbm`, `kronos_direction` | 2 h | 3,085 | 1.24 |
| `h1_gbm`, `kronos_direction` | 4 h | 1,542 | 2.47 |

The other 34 candidate cells are `UNDERPOWERED — NO DECISION` by design: logged,
reported, and spending zero alpha. They are the M15 cells at 2 h (4.9 years), 4 h and
6.5 h (14.7 years each), H1 at h ≥ 6 (3.7–74 years) and every daily-cadence cell
(13–65 years).

Adding the two M15 challengers costs the existing cells very little, because the family
only grows from 6 to 12 and alpha from 0.00833 to 0.004167: the H1 1-hour cell goes from
0.5 to 0.62 years. Two session challengers rather than one costs 0.02 years at the
15-minute cell. That is why a second model type was affordable; a third would start to
matter.

The coverage measured during the dry run (D14) scales these times before registration,
and the fixed point is recomputed with it.

The **volatility family** has its own power table: the challenger ensemble at
{1, 2, 5} days, plus Kronos volatility at 24 bars, each measured by ΔMAE against
GARCH(1,1) × day-of-week. Its variance comes from the development curves.

**Expected outcome, stated in advance.** In the study the H1 GBM challenger scored
52.66 %, 52.25 % and 52.01 % at 1, 2 and 4 h, every interval above 50 %. Its three cells
are expected to end KEEP, with 1 h and 2 h labelled "predictive, not cost-viable" on the
cost side. Kronos direction is expected to end DROP: no interval of it excluded 50 %.
The M15 session study has now run, so the six session cells have a development prior
too, recorded here before the forward test begins:

| cell | accuracy | 95 % CI | breakeven | net / trade | PBO | DSR |
|---|---:|---|---:|---:|---:|---:|
| `m15_session_gbm` 15 min | 51.01 % | 50.46–51.56 | 56.02 % | −0.3 pip | 0.28 | 0.000 |
| `m15_session_lstm` 15 min | 51.12 % | 50.56–51.66 | 56.02 % | −0.2 pip | 0.77 | 0.000 |
| both at 30 min | 50.4–50.6 % | includes 50 | 54.27 % | negative | | ~0 |
| both at 1 h and longer | 48.4–50.1 % | includes 50 | 51.1–53.0 % | negative | | ~0 |

Only the two 15-minute cells have an interval above 50 %, by about one point. Both are
**five points below their spread breakeven**, every net figure at every cost level is
negative, and the Deflated Sharpe is zero everywhere. The LSTM's PBO of 0.77 says that
choosing its best horizon in-sample is mostly overfitting.

The expectation stated in advance is therefore: the two 15-minute cells end **KEEP**
under the owner's beat-a-coin criterion, labelled "predictive, not cost-viable"; the
four 1-hour cells end **DROP**. The bid-versus-mid parity check passed on all twelve
cells (worst gap 0.70 pp against a 1.0 pp tolerance), which is the measurement that the
session window is clean rather than merely assumed.

### D14 — Dry run before registration

The logger runs for at least two weeks in `phase = dry_run`. Those rows are never
scored. The dry run measures real coverage (the fraction of bar closes logged), feeds
it into the power table, and shakes out operational failures before any alpha exists.
The M15 cadence joins the dry run when it is built, and its own two weeks of coverage
are measured before registration — the H1 and D1 dry run already under way does not
cover it.

### D15 — M15 bars from M1, on mid prices

`results/eurusd_m15.csv` is byte-pinned and has no spread column, so it cannot support
a sub-hourly study: without a spread there is no mid price and no honest cost. The M15
series is therefore aggregated from `results/curl/raw/EURUSD_M1.parquet`, which carries
`spread_points` per bar. Each M15 bar keeps OHLC on the bid, the mid close
(`close + spread/2`), summed tick volume and the median spread of its minutes. The
pinned CSV is never written and never read for targets.

This costs history: the M1 file starts 2018-08-08, so the sub-hourly study spans 8 years
against the H1 study's 11. That is accepted, because an 8-year series with spreads is
worth more than a 14-year series that cannot distinguish a price move from a spread
move. A declared alternative — using the pinned CSV on bid closes only — was rejected by
the evidence in D17.

### D16 — The bar-label clock, measured before anything was built on it

The bar labels are **not UTC**, although every cache in this repository stores them as
if they were (`src/live_data.py` localises MT5's epoch to UTC as-is, so a label is the
broker server's wall clock). Measured on the M1 file, the weekly market open label is
`Sun 23:00` and the last bar before the weekend is `Fri 22:59`, year-round — and
exactly `22:00` in the March and late-October weeks when US and EU daylight saving
disagree. The forex week runs Sunday to Friday 17:00 New York, so:

| period | NY 17:00 in UTC | observed label | implied label offset |
|---|---|---|---|
| normal summer | 21:00 | 23:00 | UTC+2 |
| normal winter | 22:00 | 23:00 | UTC+1 |
| US on DST, EU not (March) | 21:00 | 22:00 | UTC+1 |
| EU off DST, US still on (late Oct) | 21:00 | 22:00 | UTC+1 |

Only one zone fits all four rows: **Europe/Berlin** (CET/CEST). The median spread
confirms it independently — it triples at label hour 23, which is exactly New York
17:00, the daily rollover. A first version of this design assumed the labels were UTC
and converted them to `America/New_York`, which put the session two hours off and
mislocated the rollover artifact at "19:00 New York". Every figure here is from the
corrected clock.

**The session is therefore expressed in label time, with no conversion at all.**
Europe/Sofia and Europe/Berlin share the EU daylight-saving rule, so Sofia is Berlin
plus one hour on every day of the year. The owner's 15:30 Sofia is label 14:30, always,
and 23:00 Sofia is label 22:00. The window is `14:30 <= label < 22:00` on weekdays.
Resolving through `America/New_York` would be worse, not better: it coincides with NY
08:30–16:00 for 47 weeks and drifts an hour in the other 5, and the owner sits down by
their own clock.

A trade is eligible only when **both** its as-of bar and its target bar fall inside the
session on the same label date. Restricting only the as-of bar is not enough: at a
1-hour horizon the label-21 hour alone carries a 3.0 pp bid-versus-mid gap, because its
target lands in the widening-spread hour. Under the strict rule the worst in-session
hour is 0.69 pp and the cell-level gap is 0.16 pp.

This conditioning cuts the sample from 24,861 M15 bars a year to 7,523 eligible at a
15-minute horizon, and that cost is the point: the remaining bars are the ones the owner
can act on. Session position is also given to the challengers as a feature, so a
within-session pattern can be learned rather than assumed.

### D17 — The bid-bounce guard, and the measurement that forced it

A model trained on bid closes can learn the broker's spread schedule instead of the
market. Measured on M15 bars built from M1, at a 1-hour horizon, by label hour:

| label hour | New York | up-rate on bid | up-rate on mid | gap | median spread |
|---|---|---:|---:|---:|---:|
| 14:00–21:00 (the session) | 08:00–15:00 | 48.0–50.5 % | 48.1–50.5 % | ≤ 0.69 pp | 0.5 pip |
| 22:00 | 16:00 | 46.60 % | 57.08 % | +10.48 pp | 0.9 pip |
| 23:00 | **17:00 — rollover** | **72.20 %** | 60.48 % | −11.72 pp | 2.4 pip |

The 72.20 % at label 23:00 is not skill. The spread quintuples at the New York 17:00
rollover; the bid falls because the spread opens, not because the market falls, and a
bid-only series records that as a reproducible move. A sign model fitted there would
backtest above 70 % and be unprofitable at any cost level: the mean move there is about
4 pips against a 2.4-pip spread, so its breakeven is near 80 %.

Under the strict eligibility rule of D16 the session is clean by measurement: the
worst in-session hour differs by 0.69 pp between the two price definitions and the
cell-level difference is 0.16 pp at a 1-hour horizon. Every sub-hourly cell is
nevertheless scored on both definitions and the difference reported; a gap beyond the
declared tolerance of **1.0 pp** labels the cell a spread artifact. That threshold sits
an order of magnitude above every in-session measurement and an order of magnitude
below the rollover hours, so the check flags the artifact and leaves the session alone.

The two runs are two separately fitted models, so their accuracies differ by sampling
noise as well as by the data. The tolerance is therefore applied against a **noise
floor** of twice the standard error of the difference, and both numbers are reported.
At the admitted cells the floor is well under the tolerance (n ≈ 54,000 at the 1-hour
cell gives about 0.6 pp), so the declared 1.0 pp binds. At the underpowered long
horizons the floor is larger than the tolerance, and without it those cells would be
called spread artifacts on noise alone — the opposite of what the check is for.

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
  owner's request for all models, takes 3 of the 12 family slots. That makes the bar
  for every other cell stricter than a 9-cell family would (0.004167 vs 0.00556).
  This is a deliberate cost of the "all models" decision, stated in the registration.
- **[Bid-only data manufactures a sub-hourly edge]** → The measurement in D17 shows it
  reaching 72 % at the rollover hour. Mitigated three ways: mid prices from the per-bar
  spread, a session window measured to be clean, and the bid-versus-mid parity check on
  every cell. This is the single largest methodological risk the M15 layer adds.
- **[The bar-label clock is assumed rather than measured]** → This already happened
  once: the first draft of this design read the labels as UTC, which put the session two
  hours off and mislocated the rollover artifact. Mitigated by establishing the clock
  from the weekly boundary, recording it in the study record, and refusing a run whose
  weekly open label stops matching it. Every intraday rule in the repository rests on
  this one fact.
- **[Short horizons have breakevens no model here approaches]** → 56.02 % at 15 minutes
  and 54.27 % at 30 minutes, against the 52.66 % best ever measured in this project. A
  KEEP at those cells will very likely still be labelled "predictive, not cost-viable".
  The owner chose the beat-a-coin criterion knowing this; the cost label carries it.
- **[The M1 parquet is not in git]** → `results/curl/raw/` is excluded by `DATA.md` §7,
  so the sub-hourly study is not reproducible from a fresh clone. Mitigated by
  regenerating with `src/curl_mt5_fetch.py` and by recording the source file's SHA-256
  and row count in the study record, so a rebuilt file can be proven identical.
- **[`pyarrow` is used but not declared]** → It is present in the environment but
  absent from `requirements.txt`. The M15 aggregation must fail with a clear message
  naming the install command, never with an import traceback.
- **[Four times the forward rows]** → M15 logging quadruples row volume against H1 and
  runs 30 times per session day. The logger's idempotent key and append-only CSVs
  already handle it; the runbook gains the expected row counts so drift is visible.
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
   `overfit`, with tests. No data is read. *(done)*
2. **Phase 1 — horizon study.** Declare the study record (grid, configs, tuning grids,
   cap), fit the challengers on history, and write the descriptive curves, PBO, DSR
   and manifest. *(done for the daily and H1 cadences)*
3. **Phase 1b — M15 session layer.** Build the mid-price aggregation and the session
   window, add the two session challengers and the M15 cadence to the splitting
   arithmetic, run the session study, and report every cell on both price definitions
   with the parity check.
4. **Phase 2 — forward dry run.** Install the scheduled task and log for at least two
   weeks in `dry_run`, per cadence. Measure coverage and failures. *(running for H1
   and D1 since 2026-10-01; M15 joins when Phase 1b lands)*
5. **Phase 3 — registration.** Compute the power table with the measured coverage, fix
   the families at the 12 cells from D13, commit `PRE_REGISTRATION.md` and the empty
   registries, then switch the logger to scoring mode.
6. **Phase 4 — operation.** Monthly refits. Verdicts are issued at the registered n
   only.

Phase 1b lands before Phase 3 deliberately. The registration fixes the family for
good, and a family chosen before the M15 cells exist would force them into a second
family with its own alpha. One family of 12 is stronger than two families of 6.

**Rollback:** disable the scheduled task. Nothing in serving depends on any of this.
`research_models/` can be deleted, and the tracked CSVs remain as a historical record.

## Open Questions

- **Off-machine backup** of `research_models/`, and whether to move the logger to an
  always-on host later. Both are operational only.
