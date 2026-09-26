# eurusdprophet — project handover

**Purpose of this document.** A complete description of the project as it stands, written
to be handed to a new agent that has no prior context. It is a *description*, not
instructions: it does not tell you what to build.

**Read the provenance markers. They matter.**

| marker | meaning |
|---|---|
| `[DOC]` | Stated in the project's own `CLAUDE.md`. Authoritative by the project's standards, but see §11 — some of it has been verified wrong. |
| `[VERIFIED]` | Established by direct inspection of the repository during a working session on 2026-09-19/20 (git history, file reads, test runs, computed numbers). |
| `[INFERRED]` | My reasoning from the above. Not checked. Treat as a hypothesis. |
| `[UNKNOWN]` | Not established. Listed in §12 so you know what to go find out. |

Everything below carries one of these. Nothing here is invented. Where a number appears
without a marker it belongs to the marker of the sentence or table it sits in.

---

## 1. What the system is

`[DOC]` A EUR/USD **next-day directional + return predictor**. A research notebook trains
the models; a shared inference service serves them behind two frontends. The deep
reference for data flow, artifacts and failure modes is `ARCHITECTURE_DOCS.md`.

`[DOC]` The whole design exists to stop **research-to-production drift**. Training
(notebook + `_train_pipeline.py`) and serving (`api.py` → `src/inference.py`) import the
**same** `src/features.py`, so the feature matrix is byte-identical on both sides.

`[DOC]` The project "now trades REAL money" and "a false-positive feature is now live
capital risk."

`[VERIFIED]` **That claim does not match the code.** `src/paper_trading.py`'s own docstring
says "This is SIMULATED ONLY. There is no broker, no order execution, no position…". The
ledgers have no `size` or `weight` column; every position is unit-size and
`net_return_pct` is unscaled. `src/vol_scaled_backtest.py` — the position-sizing overlay —
is imported by exactly two files: itself and `tests/test_unit.py`. It is not in the
serving path. **Resolve this contradiction before acting on anything framed as capital
risk** (see §12, Q1).

`[DOC]` Environment is **Windows / PowerShell**; a `.venv` is present; `MetaTrader5` is
Windows-only. Scripts printing non-ASCII need `PYTHONIOENCODING=utf-8` or the `cp1252`
console codec raises `UnicodeEncodeError`.

`[DOC]` Pre-trained artifacts in `models/` **and** `results/eurusd_features.csv` are
git-tracked, so the app runs immediately after `pip install -r requirements.txt` — no
training required.

### Commands `[DOC]`

```bash
python -m pytest -q                      # full suite
python -m uvicorn api:app --reload       # -> http://127.0.0.1:8000
python _train_pipeline.py                # retrain & regenerate models/
jupyter notebook notebooks/01_data_preparation.ipynb
```

`[VERIFIED]` Suite size as of 2026-09-20: **587 passed, 0 failed**, ~13 minutes.
(`README.md:17` and `HOW_TO_RUN.md:143` state the count; no test enforces it.)

---

## 2. Repository layout

`[VERIFIED]` Paths confirmed to exist during the session:

```
api.py                          FastAPI entry point; dashboard + all endpoints
_train_pipeline.py              sole producer of the real per-variant artifacts
config.json                     split fractions, variant definitions
CLAUDE.md                       agent-facing project rules
ARCHITECTURE_DOCS.md            deep reference (sections cited throughout)
DATA.md                         data inventory + pinning policy
IMPROVEMENT_LOG.md              history of methodology changes
HOW_TO_RUN.md
README.md
.githooks/commit-msg            refuses models/ commits without a RETRAIN: line
.githooks/check_retrain_declaration.py
.gitattributes                  *.keras/*.pkl/*.h5 binary; *.py/*.csv/*.md eol=lf

src/
  features.py                   FEATURE_COLUMNS (27), LAG_COLUMNS (6), the SSOT
  inference.py                  PredictionService — the single serving path
  live_data.py                  MT5 -> yfinance -> bundled CSV fallback chain
  macro_data.py                 FRED API -> FRED CSV -> disk cache -> None
  ablation.py                   validation-slice feature ablation, Bonferroni bar
  volatility.py                 5-seed multi-task LSTM volatility ensemble
  h1_direction_model.py         H1 next-bar direction GBM + LSTM, arbiter()
  h1_direction_final.py         the H_dir program; HARD BOUNDARY: never writes models/
  h1_features.py                H1 feature computation (SMA504, RSI_24, ...)
  density_model.py              MDN density family
  calendar_paired_bootstrap.py
  vol_scaled_backtest.py        §3.9 sizing overlay — NOT in the serving path
  paper_trading.py              forward simulated ledgers
  vix_features.py               ablation-only VIX features
  fibonacci_fractals.py         built but dormant (its gating hypothesis DROPped)
  metalabel.py                  added 2026-09-20; power analysis only, evaluates nothing
  external/kronos/              separate "Kronos" program

models/
  baseline/                     7 daily artifacts (price-only variant)
  with_macro/                   7 daily artifacts (full 27-col variant)
  volatility/                   5 seed .keras + own PCA/scalers + vol_metrics.json (10 files)
  h1_direction/                 the served H1 next-bar direction model + meta
  ti_lstm_h1/                   ti_lstm_h1.keras, ti_metrics.json, ti_scaler.pkl
  calendar/                     calendar artifact; density family loads it frozen
  (root)                        h1_xgb_regressor.pkl, h1_rf_regressor.pkl,
                                h1_svm_regressor.pkl, h1_lstm.keras,
                                h1_feature_scaler.pkl, h1_lstm_scaler.pkl,
                                h1_feature_columns, h1_lstm_config,
                                exploratory_gbm_*, randomforest_tuned, scaler,
                                xgboost_tuned

results/                        data, logs, registries, program outputs (see §3, §6)
tests/                          587 tests
tests/fixtures/*_protected_sha256.json   byte-pinning guards
notebooks/01_data_preparation.ipynb
static/index.html               the dashboard
```

---

## 3. Data

### 3.1 The pinning policy `[VERIFIED]`

`tests/test_input_data_provenance.py` splits inputs into two classes and this distinction
is load-bearing:

- **FROZEN** — research inputs no running code rewrites. Byte-pinned by SHA-256. A moved
  digest fails loudly. Re-baselining one is a deliberate, documented act.
- **ROLLING** — `results/eurusd_h1.csv` only. An *operational cache*:
  `src/live_data.py::fetch_h1_market_data` rewrites it on every successful MT5/yfinance
  pull, so byte-pinning it would assert only that nobody had run the app. It carries a
  **recorded provenance stamp** (digest, row count, span at stamp time) plus structural
  invariants, and `DATA.md`'s row count must match the file on disk.

### 3.2 The files `[VERIFIED]`

| file | rows | span | class |
|---|---:|---|---|
| `results/eurusd_features.csv` | 15,760 | 1971-01-11 → 2026-08-10 | FROZEN |
| `results/eurusd_h1.csv` | 60,136 | 2017-01-20 02:00 → 2026-09-18 22:00 UTC | ROLLING |
| `results/eurusd_m15.csv` | 350,000 | 2012-06-25 21:30 → 2026-07-24 22:45 UTC | FROZEN |
| `results/pooled_h1/EURUSD_h1.csv` | 70,000 | 2015-04-27 20:00 → 2026-07-28 15:00 UTC | FROZEN |
| `results/pooled_h1/GBPUSD_h1.csv` | 70,000 | same window | FROZEN |
| `results/pooled_h1/AUDUSD_h1.csv` | 70,000 | same window | FROZEN |
| `results/pooled_h1/EURUSD_h1_newyork.csv` | 70,000 | NY-session re-stamped | FROZEN |
| `results/pooled_h1/EURUSD_m15_newyork.csv` | 350,000 | NY-session re-stamped | FROZEN |
| `results/pooled_h1/retired/CHFUSD_h1.csv` | 70,000 | retired instrument | FROZEN |

`eurusd_features.csv` columns `[VERIFIED]`: `time, open, high, low, close, tick_volume,
log_return, lag_1..lag_5, bar_dynamics, bar_dynamics_lag_1, day_of_week, day_sin, day_cos,
target_return`.

**Critical distinction** `[VERIFIED]`: three of the four H1 families (`h1_direction`,
`h1_multiday`, `pooled_h1`) read the **frozen** `results/pooled_h1/EURUSD_h1.csv`, *not*
the rolling cache. That is what protects their committed numbers.

**Window slide** `[VERIFIED]`: the MT5 window is fixed-size, so the rolling cache both
gains recent bars and **loses its earliest ones**. Between 2026-08-18 and 2026-09-18 its
first row moved 2016-12-21 18:00 → 2017-01-20 02:00 — ~30 days of leading history gone.
This matters for `src/h1_features.py`'s trailing warm-ups (SMA504 = 504 H1 bars, RSI_24),
which consume leading rows.

### 3.3 Macro data `[VERIFIED]` / `[INFERRED]`

`[VERIFIED]` These CSVs exist and are maintained by the FRED framework:
`results/yield_differential.csv`, `results/inflation_differential.csv`,
`results/policy_rate_differential.csv`, `results/usd_index.csv`, `results/vix.csv`.

`[DOC]` The `with_macro` variant adds **4 FRED macro columns** over `baseline`.
`[DOC]` VIX rides the shared FRED framework (`macro.features.vix`) to maintain
`results/vix.csv`, but its level is kept **out of** `MACRO_MERGE_COLUMNS` and its features
out of `FEATURE_COLUMNS` — ablation-only.

`[INFERRED]` The 4 macro columns are therefore most likely yield differential, inflation
differential, policy-rate differential and USD index. **Not verified** — confirm against
`src/features.py` (§12, Q3).

---

## 4. Model families

This is the part that matters for strategy work. There are **eight** distinct families.

### 4.1 Daily dual-variant (the production predictor) `[DOC]`

- `baseline` = price-only, `PRICE_FEATURE_COLUMNS`, **23 columns**, no FRED at all.
- `with_macro` = the full **27-column** set whose 4 macro features are statistically
  **unproven (KEEP-provisional)**.
- Both train in ONE `_train_pipeline.py` run through ONE `train_variant()` body, on the
  **identical** euro-era row set — only the columns differ — into `models/<variant>/`.
- Both serve on every prediction. The response carries `baseline` + `with_macro` blocks and
  a `variant_agreement` flag.
- Each variant emits a direction (classifier) and a return (regressor), plus an LSTM
  multi-task head.

**7 artifacts per variant** `[DOC]`: `lag_scaler`, `lag_pca`, `global_scaler`,
`best_gbm_eurusd`, `best_gbm_regressor_eurusd`, `lstm_multitask_eurusd.keras`,
`lstm_time_steps`. 14 total. `test_smoke.py` asserts all of them exist.

**Consensus and the sentinel** `[DOC]`: `CONFIDENCE_THRESHOLD = 0.52` flags near-chance
agreement as `"MIXED / LOW CONFIDENCE"`. `[VERIFIED]` `src/paper_trading.py` books such a
row **flat, P&L 0** — it is *not a trade*. Any analysis that treats MIXED as a wrong
direction call is broken.

**Expected (not buggy) behaviour** `[DOC]`: daily EUR/USD is near-efficient —
**ROC-AUC ≈ 0.50** — and the return regressor **shrinks predictions toward the ~0% mean**
(Huber loss on a noisy zero-mean target → predict the conditional mean; trained MAE ≈ the
"predict the mean" baseline). Documented as mathematical honesty in `ARCHITECTURE_DOCS.md
§4.2.1` and notebook Section 21. Do **not** "fix" the low return magnitudes.

### 4.2 Volatility ensemble `[DOC]` + `[VERIFIED]`

`[DOC]` `src/volatility.py` → `models/volatility/`. Predicts `target_volatility_pct`
(= |next-day log return| · 100). A **5-seed multi-task LSTM ensemble** vs a train-only-fit
**GARCH(1,1)**; pre-registered gate in `results/volatility_seed_ensemble.csv`. ONE
price-only model, NOT per-variant. `vol_ready` is all-or-nothing — never serve a partial
ensemble. Own hypothesis family (`results/volatility_hypothesis_log.csv`), now **SPENT** on
the validation arbiter.

`[DOC]` CLAUDE.md §3.5 calls it "the ONLY neural family with a CI-confirmed edge over its
honest baseline."

`[VERIFIED]` **That claim is materially incomplete.** `volatility_hypothesis_log.csv`
row #3 (out-of-sample, n = 1712, verified 2026-08-07) says both of:

> CONFIRMED vs plain GARCH — dMAE +0.01333 CI95 [+0.01039, +0.01612] … ensemble ahead in
> every year 2021-2026

> BUT THE MECHANISM IS CALENDAR, NOT VOLATILITY SKILL: the entire aggregate edge comes from
> FRIDAY rows (dMAE +0.09112), while on Mon/Tue/Wed GARCH is slightly AHEAD. A GARCH(1,1)
> multiplied by a 6-number day-of-week table fitted on [0:80%] only BEATS the ensemble
> outright: MAE 0.20521 vs 0.21925 … dMAE −0.01403 CI95 [−0.01773, −0.01073].

`[VERIFIED]` The Friday target is the weekend gap and averages **0.0877** against
**0.31–0.38** on other days — i.e. the LSTM learned that Friday's realised move is small
because the market is shut. That is a calendar fact, and a six-number lookup encodes it
explicitly in ~10 parameters and wins.

`[VERIFIED]` Row #3's own verdict says "ship … with honest GARCH-relative framing." The
dashboard drops the qualifier. A caveat naming the better baseline was added to the card
body on 2026-09-20 (commit `9097f04`); the badge itself still reads "✓ validated vs
GARCH(1,1)", which is true as written.

`[VERIFIED]` `vol_metrics.json`'s `validation_decision` block (mt_ensemble_mae 0.18594,
`cleared_bar: true`, n_val = 856, arbiter `validation[70:80]`) is what the live badge
reads. `models/volatility/` was reverted on 2026-09-20 (`0419645`) so that block again
describes the artifacts actually serving.

### 4.3 H1 next-bar direction — **the only registered edge in the project** `[VERIFIED]`

`src/h1_direction_model.py` + `src/h1_direction_final.py`. Predicts the direction of the
**next H1 bar**. Registry: `results/h1_direction_hypothesis_log.csv`, alpha **0.007143**.

| # | hypothesis | CI95 (block) | McNemar p | verdict |
|---|---|---|---:|---|
| 1 | GBM vs train majority — **TEST BLOCK confirmation** | [+0.011404, +0.049667] | 1.70e-05 | **KEEP** |
| 1 | GBM vs train majority — validation | [+0.005533, +0.042205] | 0.00169 | **KEEP** |
| 2 | LSTM vs GBM | [−0.023704, +0.001831] | 0.0527 | DROP |
| 3 | replication GBPUSD | [−0.006459, +0.029590] | 0.120 | DROP |
| 4 | replication AUDUSD | [+0.003574, +0.038497] | 0.00340 | **KEEP** |
| 5 | replication CHFUSD | [+0.014085, +0.048236] | 6e-06 | **KEEP** |
| 6 | magnitude regressor vs train mean | nan | nan | DROP |
| 7 | cross-currency lags vs H_dir.1 | [−0.010958, +0.008479] | 0.818 | DROP |

`[VERIFIED]` So: the H1 direction GBM beats the train-majority baseline by roughly **+0.6
to +4.2 pp** on validation, confirmed on the test block at **+1.1 to +5.0 pp**, and it
**replicates on AUD/USD and CHF/USD but not GBP/USD**. The LSTM does not beat the GBM.
Cross-currency lags add nothing.

`[VERIFIED]` Registered reproduction gate: validation accuracy **0.527462 ± 0.003** from a
`[0:70%]` refit.

`[VERIFIED]` `models/h1_direction/` is the **served, full-history** model. Its own meta says
`"validated_out_of_sample": false`, `trained_at_utc: 2026-07-30T18:11:32`, trained through
2026-07-28. **It is in-sample on the validation slice** — scoring it there proves nothing.
The model that produced the registry rows was fit in memory and discarded, because
`h1_direction_final.py` is forbidden from writing to `models/` and has no save/dump call.
The registry's anchor is the behavioural reproduction gate, not a digest.

`[VERIFIED]` **NY-session breakdown** in `results/h1_direction_final/report.txt`:
NY-only **+7.98 pp**, London **+4.92 pp**, Asia **−0.26 pp**. This is *descriptive only* and
was computed on the **spent** test block. Using it to define a subpopulation for a new test
is a methodology violation and was explicitly declined in 2026-09.

`[VERIFIED]` **Autocorrelation structure**: direction has essentially none — Ljung–Box on
sign at lag 24, p = 0.115; ACF(sign, 1) = −0.035. The clustering lives in |return| —
ACF 0.216 at lag 1, **0.188 at lag 24** (the diurnal cycle that justifies block = 24 bars).
Measured design effect (block-24 vs i.i.d. CI half-widths across the registry's seven
accuracy rows): SE ratio 0.942–1.119, median 1.017 → **deff 0.89–1.25, median 1.035**.
Blocking barely widens a direction-accuracy interval.

### 4.4 H1→daily ensemble (§3.4) `[VERIFIED]`

`models/h1_xgb_regressor.pkl`, `h1_rf_regressor.pkl`, `h1_svm_regressor.pkl`,
`h1_lstm.keras`, plus `h1_feature_scaler.pkl`, `h1_lstm_scaler.pkl`.

**It does not predict the next H1 bar.** `build_h1_datasets` aggregates H1 bars into
**daily** statistics and predicts the **next-DAY** percent return on a **2,504-row daily
index**. It trains on `[0:80%]` of its own row set = through **2024-10-09**. Of the 437 of
its days that fall inside the H1 direction validation window, **408 (93.4%) are in-sample
for it**.

This family is easy to confuse with §4.3. They are different estimands at different
frequencies.

### 4.5 `ti_lstm_h1` `[VERIFIED]`

`models/ti_lstm_h1/` (`ti_lstm_h1.keras`, `ti_metrics.json`, `ti_scaler.pkl`). Its own
docstring says "NOT next-hour prediction" — it is **next-day**. Has its own forward ledger,
`results/paper_trading_log_ti_h1.csv`. `[UNKNOWN]` what "ti" stands for and what its
features are.

### 4.6 Density family `[VERIFIED]`

`src/density_model.py`, `results/density/PRE_REGISTRATION.md` + `RESULTS.md`. An MDN
(mixture density network) forecasting the return *distribution*, evaluated by CRPS against
GARCH+t and an empirical unconditional baseline. Row set n = **8603** (note: two edge rows
fewer than the daily families' 8605 — the row sets are **not** interchangeable).

Outcome: **H_den.1 DROP** — the MDN loses to the registered rival, on validation and again
on the test block. **H_den.2 NOT-RUN** — its precondition (H_den.1 clearing) failed, alpha
unspent. A test-block cell that excluded zero was recorded and explicitly **left unread**,
because H_den.2's precondition had failed and because the test block is not the arbiter.

Also recorded there: seed-to-seed CRPS range (0.00071) exceeds the effect under test
(0.00045), which is what makes the all-or-nothing seed ensemble load-bearing; but
run-to-run determinism held perfectly for this dense MDN (0.0 difference across two full
runs), so the TF/oneDNN nondeterminism seen elsewhere does not apply to it.

### 4.7 Kronos `[VERIFIED]` / `[UNKNOWN]`

`[VERIFIED]` `src/external/kronos/` exists as a self-contained package with its own results
directory `results/external_kronos/` (`kronos_vol_log.csv`, `kronos_vol_ledger.csv`), its
own fixture `tests/fixtures/kronos_protected_sha256.json`, its own test file
`tests/test_external_kronos.py`, and two endpoint pairs: `/api/kronos-volatility` +
`/kronos-volatility`, `/api/kronos-direction` + `/kronos-direction`.

`[VERIFIED]` It introduced an **additive-only contract**: files it touched outside its own
package (`src/inference.py`, `api.py`, `static/index.html`, `pyproject.toml`, `.gitignore`)
must **gain lines and lose none** against the pre-Kronos baseline commit `6319df2`. Two
documented exemptions exist in `LATER_PROGRAM_FILES`: `src/live_data.py` 3 deletions,
`api.py` 22 deletions, each with a justification comment. The comment warns the exemption
must not quietly widen.

`[UNKNOWN]` What the Kronos model actually is, what it forecasts, and whether it has a
registered verdict. **Go read `src/external/kronos/` first** (§12, Q4).

### 4.8 Calendar `[VERIFIED]`

`models/calendar/` exists as a frozen artifact and the density family loads it. Related:
`src/calendar_paired_bootstrap.py`. `[UNKNOWN]` its full scope. Note the connection to §4.2:
the volatility family's whole measured edge turned out to be a calendar effect, and a
GARCH × day-of-week lookup beat a five-model neural ensemble.

---

## 5. Splits — memorise these

`[VERIFIED]` Two frequencies, two conventions, and they are **not** interchangeable.

### Daily — `src/ablation.py::_canonical_split`, from `config.json`
(`train_fraction = 0.80`, `val_fraction = 0.10`), on the euro-era matrix n = **8605**:

| block | positions | n | dates |
|---|---|---:|---|
| train | `[0 : 6023]` | 6,023 | 1999-01-04 → 2018-05-01 |
| **validation (the arbiter)** | `[6023 : 6884]` | **861** | 2018-05-02 → 2021-02-01 |
| test | `[6884 : 8605]` | 1,721 | **SPENT** |

Arithmetic: `int(8605 × 0.70) = 6023`; `int(8605 × 0.80) = 6884`.

`[DOC]` Within that: the GBM trains `[0:80%]`; the LSTM trains `[0:70%]` with `[70%:80%]`
for early stopping; both test on the identical held-out `[80%:100%]`.

**Consequence** `[VERIFIED]`: the daily GBM's training block **contains the whole
validation slice**. Any evaluation of the served daily GBM on validation is in-sample. The
only honest scorer is a `[0:70%]` refit.

### H1 — `src/h1_direction_model.py::split_purge_embargo`
(`TRAIN_FRAC 0.70`, `VAL_FRAC 0.85` as an *end* position, purge 1, embargo 24), on
`results/pooled_h1/EURUSD_h1.csv`: 70,000 raw bars → **69,349 labelled** after a 200-bar
warm-up and 451 exact-zero-return drops:

| block | positions | n | dates (UTC) |
|---|---|---:|---|
| train | `[0 : 48544]`, last row purged | 48,543 | 2015-05-08 03:00 → 2023-03-15 02:00 |
| **validation (the arbiter)** | `[48544 : 58946]`, first 24 embargoed | **10,378** | 2023-03-16 04:00 → 2024-11-19 04:00 |
| test | `[58946 : 69349]` | 10,403 | **SPENT** (read once, 2026-07-30) |

Arithmetic: `int(69349 × 0.70) = 48544`; `int(69349 × 0.85) = 58946`.

---

## 6. Methodology — the rules that govern new claims

`[DOC]` These override the older test-block narrative wherever they conflict. Full detail
in `ARCHITECTURE_DOCS.md` → *Production Methodology* and `IMPROVEMENT_LOG.md`.

1. **The historical test block `[80%:100%]` is SPENT for feature search.** It was reused as
   a repeated KEEP/DROP criterion — data snooping. All feature ablation now runs on the
   **validation slice `[70%:80%]`** via `src/ablation.py`, fit on `[0:70%]` only, test block
   never indexed. The test block is a **one-shot final report** from `_train_pipeline.py`,
   never a search knob.
2. **Every KEEP clears a Bonferroni-corrected bar, not a flat 0.05.**
   `results/feature_hypothesis_log.csv` counts every feature hypothesis ever spent; the bar
   is `alpha = 0.05 / family_size`. Registering a genuinely new feature tightens the bar for
   everyone.
3. **The forward paper-trading ledgers are the primary production-worthiness signal.**
   `src/paper_trading.py` → `results/paper_trading_log_baseline.csv` and
   `_macro.csv`, at `/paper-trading` + `/api/paper-trading`. Whichever variant nets better
   over a **months**-long forward window is the honest winner. Do not decide by re-analysing
   the spent test block. Simulated only — **do not add broker / order-execution /
   position-sizing / stop-loss code**; real execution is a separate risk-management
   conversation, only after a ledger shows a supported edge.
4. **A commit touching `models/` must declare the retrain.** `.githooks/commit-msg`
   (enable once: `git config core.hooksPath .githooks`) refuses any commit staging a path
   under `models/` unless the message carries a `RETRAIN:` line. Never bundle a retrain into
   a commit about something else; never `--no-verify` past it.

### 6.1 Invariants that caused real bugs when broken `[DOC]`

Changing any of these requires updating `config.json`, `_train_pipeline.py`, notebook
**Section 19** and `src/inference.py` **together**:

1. **Single `global_scaler.pkl` per variant**, shared by that variant's GBM and LSTM. No
   per-model scalers. Each variant's PCA (`lag_pca` + `lag_scaler`) and `global_scaler` are
   fit **only on the 0–80% train block** and live under `models/<variant>/`. Never mix one
   variant's scaler/PCA with another's models.
2. **`target_return` is in PERCENT, natively.** The `× 100` lives **only** in
   `src/features.py::add_advanced_features`. No `× 100` anywhere else.
3. **Unified chronological split** from `config.json` (see §5).
4. **No look-ahead bias.** Targets via `shift(-1)`; `ffill` only carries a *past* value
   forward; scaler/PCA fit train-only; `TimeSeriesSplit` everywhere, never random K-fold.
   The FRED/no-look-ahead unit tests guard this.

### 6.2 Registries and what has been spent `[VERIFIED]`

| registry | rows | alpha | status |
|---|---:|---|---|
| `results/feature_hypothesis_log.csv` | 9 | a new feature faces 0.05/10 | standing bar |
| `results/h1_direction_hypothesis_log.csv` | 8 | 0.007143 | see §4.3 |
| `results/volatility_hypothesis_log.csv` | ≥3 | — | **SPENT** on the validation arbiter |
| `results/metalabel_hypothesis_log.csv` | 3 | 0.004167 | **zero comparisons spent** |
| density family | — | — | H_den.1 DROP, H_den.2 NOT-RUN (alpha unspent) |

`[DOC]` The five rejected feature ADD-tests: `fomc_calendar_block` (#5, 2026-07-17),
`cot_positioning_block` (#6, 2026-07-20), `fibonacci_retracement_block` (#7, 2026-07-26),
`vix_regime_block` (#8, 2026-07-26), `volatility_forecast_block` (#9, 2026-07-26). All
DROPs. All 4 macro features are **KEEP-provisional** — no proven edge.

`[VERIFIED]` **`arbiter()` defect**: `src/h1_direction_model.py:1004` calls `arbiter(...)`
without an `alpha` argument, so it defaults to `FAMILY_ALPHA = 0.025` — 3.5× looser than the
0.007143 the registry restates. The module constants `FAMILY_SIZE = 2`,
`FAMILY_SIZE_IF_TRIGGERED = 5`, `FAMILY_SIZE_FINAL = 6` are all stale. **Pass alpha
explicitly if you reuse this helper.** The existing rows were checked: the restatement was a
genuine partial re-adjudication (recomputed rows carry ~16 significant digits, carried-over
rows are rounded to 6), and the two exposed KEEPs survive the tighter bar
(+0.011404 → +0.011028 and +0.014085 → +0.013750, both still > 0). No recorded verdict
flips.

### 6.3 Bootstrap conventions `[VERIFIED]`

- **H1**: paired moving-block (circular), `block_len = 24` bars = one trading day,
  `n_boot = 2000`, `random_state = 42`, percentile CI at the family alpha.
- **Daily**: `block_len = 5`. Both `src/density_model.py:561` and
  `src/calendar_paired_bootstrap.py:57` attribute this to "the volatility family
  convention" — **that attribution is false**: `src/volatility.py:388 bootstrap_delta` is
  plain i.i.d. and that family never had a block length. Cite density/calendar directly.
- **Refusal**: if `n <= block_len`, return `nan` rather than clamping the block length —
  `src/vol_scaled_backtest.py::bootstrap_delta_sharpe` is the reference implementation. A
  block as long as the sample is a cyclic rotation; every resample gives the identical
  statistic and the CI collapses to a falsely confident point.

### 6.4 The metalabel family — a worked negative `[VERIFIED]`

Registered and closed on 2026-09-20 (`6082fe5` protocol-only, then `8e78369` code +
results). Question: *does disagreement between sub-models flag when a direction call is
reliable?* Three conditions: C0 daily variant disagreement, C1 H1 cross-model disagreement,
C2 H1 probability-margin terciles. `alpha = 0.05/12 = 0.004167`. Economic floor 3 pp.
80%-power MDE governs.

| cell | n₁ | n₀ | 80%-power MDE | × deff | outcome |
|---|---:|---:|---:|---:|---|
| C0 daily disagree/agree | 353 | 508 | **12.84 pp** | 13.06 | UNDERPOWERED — NO DECISION |
| C1 as registered | — | — | — | — | **INFEASIBLE-AS-SPECIFIED** |
| C1 *proposal* (not run) | 3,585 | 6,793 | 3.83 pp | 3.89 | would be underpowered |
| C2 tercile top/bottom | 3,459 | 3,459 | **4.46 pp** | 4.53 | UNDERPOWERED — NO DECISION |

3 pp at 80% power needs **7,634 rows per cell**. C0 has 5% of that; C2 has 45%.

**C0 is structurally unanswerable**: resolving 3 pp needs n_val 15,779 = 60 years in the
arbiter block = **605 years of total daily history**, against the 27 years of euro-era data
that exist (22× short). For 2 pp: 1,360 years (50× short). **Do not re-ask C0 at daily
frequency.**

**C1 was infeasible for three independent reasons**: different estimand (§4.4), 93.4%
in-sample contamination, and — at the time — no recorded provenance for the ensemble
artifacts. There is exactly **one** persisted model in the repository that emits an H1
next-bar direction, and it is the one under test; there is no second party.

**Zero comparisons were spent.** The standing bar for future feature claims remains
`0.05/9`. A registered-but-unevaluated family charges nothing — multiplicity counts
comparisons *made*, not hypotheses *considered*.

`src/metalabel.py` is worth reading as a template: it contains the split arithmetic as pure
functions, the two-proportion MDE, a `TestBlockGuard` that **raises** on any index reaching
a spent block, the forced two-model agreement rule, train-only tercile boundaries, the
guarded bootstrap, and a sub-pip `as_of_close` reconstruction. `tests/test_metalabel.py`
asserts on the **AST** that the module loads no model and calls no `.fit()`.

---

## 7. Serving

`[DOC]` `src/inference.py::PredictionService` loads every artifact once, with
graceful-degradation gates per variant (`baseline_ready` / `macro_ready` / `models_ready`;
per-family flags in `service.variants[name]`). A broken variant never crashes the other. It
is the single serving path behind `api.py`.

`[DOC]` **Two fallback chains, neither hard-fails**: `src/live_data.py`
(MT5 → yfinance → bundled history CSV) and `src/macro_data.py`
(FRED API → FRED public CSV → on-disk cache → `None`). The `baseline` variant consumes no
macro columns, so it is immune to FRED outages by construction.

`[VERIFIED]` Endpoints in `api.py`:

```
GET  /                        dashboard (static/index.html)
GET  /api/predict             the prediction (per CLAUDE.md)
GET  /history                 rendered prediction history
GET  /api/paper-trading   +   GET /paper-trading
GET  /api/h1-direction    +   GET /h1-direction
GET  /api/kronos-volatility + GET /kronos-volatility
GET  /api/kronos-direction  + GET /kronos-direction
POST /api/retrain             triggers _train_pipeline.py
GET  /api/retrain/status
```

`[DOC]` The forecast date is **weekend-aware** (Fri/Sat roll forward to Monday — the next
*trading* session).

`[VERIFIED]` Live logs written by serving: `results/prediction_log.csv` (with
`vol_pred_pct` at column 19), `results/prediction_history.html`,
`results/h1_direction_log.csv`, `results/h1_feed_offset.json` (sticky H1 feed offset,
currently +2 h, inferred at runtime — not a constant), plus the four ledgers
(`paper_trading_log_{baseline,macro,h1_direction,ti_h1}.csv`) and the Kronos logs.

### 7.1 The retrain path — a live systemic defect `[VERIFIED]`

- There is **no scheduler** anywhere in the repo. The only code path that writes
  `retrain_state.json` with a pid is `POST /api/retrain` — a **dashboard button**.
- `/api/retrain/status` **hot-reloads the new artifacts into the serving process on
  success**, so production begins serving new models within minutes, **with no commit
  involved**.
- `logf = open(RETRAIN_LOG, "w")` **truncates**, so each run erases the previous run's log.
- The commit hook only fires if someone later tries to commit. Nothing in the loop prompts
  that.

This is how the 2026-09-11 incident happened (§8).

---

## 8. What happened on 2026-09-11 → 2026-09-20 (recent history you need)

`[VERIFIED]`

On **2026-09-11 06:45:55** someone pressed the retrain button. A full `_train_pipeline.py`
run completed (rc = 0, 1639.7 s), moving **18 production artifacts**: both daily variants'
LSTMs, `with_macro/global_scaler`, the 5-seed volatility ensemble + `vol_metrics.json`, the
H1→daily ensemble (xgb/rf/svm/lstm + both scalers), and `ti_lstm_h1`. The GBM artifacts did
**not** move — tree training is deterministic on the same 8,605-row set, so only the
nondeterministic Keras models and the macro-dependent scaler changed.

The artifacts sat **staged and uncommitted for 8 days while the app served them**. One
commit was attempted over them, titled *"Refactor code structure for improved readability
and maintainability"* — and `.githooks/commit-msg` **refused it**. That is the guard firing
in anger for the first time, and it worked. But nothing escalated for eight days.

Cleanup, 2026-09-19/20, branch `density-forecasting`:

| commit | what |
|---|---|
| `4cc7920` | Declares the 2026-09-11 retrain (`RETRAIN:` line) + re-baselines four checksum fixtures. Old digests preserved in `tests/fixtures/PROTECTED_SET_REBASELINE_2026-09-19.json`. |
| `e77a4eb` | Serving-log churn, split out. Records that ledger rows settled 09-11 → 09-19 were produced by then-undeclared artifacts. |
| `f712694` | Re-stamps the rolling H1 cache (60,056 → 60,136) and documents the window slide. |
| `9c3aac0` → `9097f04` | Volatility caveat on the dashboard card. The first violated the additive-only contract (+1/−1); the second restored the original line verbatim and added the caveat as a new line (124 added, 0 removed) rather than widening the exemption. |
| `0419645` | Reverts `models/volatility/` so the registered gate again describes what serves. |
| `6082fe5`, `8e78369` | The metalabel family (§6.4). |

`[VERIFIED]` **The revert was informationally free**: `results/eurusd_features.csv` never
changed, and of the ten files in `models/volatility/` only the 5 `.keras` seeds and
`vol_metrics.json` moved — `global_scaler.pkl`, `lag_pca.pkl`, `lag_scaler.pkl`,
`lstm_time_steps.pkl` are bit-identical, exactly as deterministic sklearn fits on unchanged
data must be. Old and new seeds differ by TF/oneDNN nondeterminism only. The tempting
tiebreak — the new seeds score better on the test block (0.214985 vs 0.216033) — was
**refused**, because using the one-shot block to choose artifacts is the data snooping the
methodology forbids.

---

## 9. Guards and tests

`[VERIFIED]` 587 tests. The non-obvious ones:

- **Byte-pinning fixtures** (`tests/fixtures/*_protected_sha256.json`). Each is a *boundary
  guard* naming what a given program is forbidden to touch — e.g.
  `h_dir_final_protected_sha256.json` pins **58 paths** (`_train_pipeline.py`,
  `config.json`, `models/baseline/` ×7, `models/volatility/` ×10, `models/with_macro/` ×7,
  `models/ti_lstm_h1/` ×4, `models/` root ×13, `src/` ×4, `results/` ×11). The name refers
  to the *program* whose isolation it enforces, not to the model it is named after.
- **`tests/test_input_data_provenance.py`** — FROZEN digests, ROLLING stamp, structural
  invariants (UTC, strictly increasing unique timestamps, schema, sane OHLC), and
  `DATA.md`'s row counts must match the files on disk.
- **`tests/test_external_kronos.py::test_only_additive_changes_outside_the_new_package`** —
  the additive-only contract (§4.7).
- **`tests/test_metalabel.py`** — AST-level assertions that the module loads no model and
  calls no `.fit()`, plus three tests that `TestBlockGuard` actually raises.
- **`.githooks/commit-msg`** — refuses `models/` commits without `RETRAIN:`.

---

## 10. The notebook `[DOC]`

- Runs **from `notebooks/`**, so paths are `../` (e.g. `../config.json`, `../models/`). Any
  `subprocess` call to pytest must pass `cwd=os.path.abspath('..')` or pytest collects 0
  tests.
- **Two distinct tracks — do not conflate them**: the **exploratory baseline**
  (Sections 13–17: no FRED, no PCA, binary target, saves `exploratory_*.pkl`) and the
  **production pipeline** (Section 19: FRED + PCA + percent targets, saves the real
  artifacts).
- **Known drift**: Section 19 still trains the pre-dual single 27-column pipeline into
  root-level `models/*.pkl` paths that production no longer loads. `_train_pipeline.py` is
  the sole producer of the real per-variant artifacts. Treat the notebook training cells as
  historical/research until ported to `train_variant()`.

---

## 11. Known defects — open as of 2026-09-21

`[VERIFIED]` All confirmed by direct inspection. None fixed unless noted.

1. **`CLAUDE.md`'s hidden-retrain list is wrong in both directions.** There are **23–25**
   commits titled *"Refactor code structure for improved readability and maintainability"*.
   **Six** of them moved production artifacts — `d222ecc` (2026-06-21, 10 files),
   `0ece63c` (07-07, 10), `b30599f` (07-25, 18), `a73344e` (08-08, 12), `3def541` (08-08,
   17), `f2645a0` (08-15, 30) — **97 artifact modifications** in total. `CLAUDE.md` names
   three; of those, `82b45ee` and `d61d033` touched **zero** model artifacts (they are
   same-titled serving-log churn, dated to the two clean-ups `8a02c9b` and `5c0fe0d`).
   Only `f2645a0` of the three was correct. *Correction drafted, not yet written.*
2. **`retrain.log` truncates**, so it evidences exactly one run (2026-09-11). Every earlier
   retrain's log was overwritten. Git bounds `models/` movement back to 2026-06-20;
   2026-09-11 is the earliest we can evidence *from the log*, not the earliest that
   occurred.
3. **The retrain loop has no detection** (§7.1). Three fixes were proposed and **not
   implemented**: P1 append instead of truncate; P2 warn on a dirty `models/` before
   retraining; **P3** — a digest check at startup and after hot-reload surfacing a
   dashboard banner naming diverged paths and how long they have been undeclared. P3 is the
   only one that would have caught the actual failure.
4. **`arbiter()` default alpha is stale** (§6.2).
5. **`ARCHITECTURE_DOCS.md §4.2.2`'s Brier numbers are doc-only** — 0.25063 for the daily
   classifier's raw `predict_proba` against 0.25013 for a constant at the training base rate
   (*worse than a constant*), and calibration was evaluated and rejected because sigmoid
   scaling collapsed the range to [0.461, 0.513] and would have pinned the consensus at
   permanent MIXED. **These numbers are not traceable to any results CSV.**
6. **`block_len = 5` is misattributed** in two files (§6.3).
7. **`CLAUDE.md §3.5` / `ARCHITECTURE_DOCS §3.5` overstate the volatility claim** (§4.2).
   Partially mitigated by the card-body caveat; the badge text is unchanged by decision.
8. **`0.05/9 ≈ 0.00556` in `CLAUDE.md` and `ARCHITECTURE_DOCS.md:59, 612` is stale** — that
   is the bar as of the *last spent* feature hypothesis; a genuinely new feature faces
   `0.05/10`.
9. **Notebook Section 19 drift** (§10).
10. **The "trades real money" framing does not match the code** (§1).

---

## 12. What I could not verify — please answer these

These are the gaps. A new agent should get answers before planning anything.

**Q1 — Is real money actually at risk?** `CLAUDE.md` says the project trades real money;
the code has no broker, no order execution, and unit-size simulated ledgers. Which is true?
Is there execution happening outside this repository — manually, or through another tool?

**Q2 — What is in `ARCHITECTURE_DOCS.md`?** I have only ever seen quoted fragments
(§3.4, §3.5, §3.9, §4.2.1, §4.2.2, §4.3.2). Sections 1–2, 3.1–3.3, 3.6–3.8, 4.1 and 4.3 are
unread by me. That file is the project's deep reference and this handover is incomplete
without it.

**Q3 — Which 4 columns are the macro features?** I inferred yield / inflation / policy-rate
differential + USD index from the CSV filenames. Confirm against `src/features.py`'s
`FEATURE_COLUMNS` and `MACRO_MERGE_COLUMNS`. Also: what are the full 27 columns and the 23
price-only ones?

**Q4 — What is Kronos?** A whole program with its own package, endpoints, ledger, fixture
and an additive-only contract, and I know nothing about what it models or whether it has a
registered verdict.

**Q5 — What is `ti_lstm_h1`?** What does "ti" mean, what are its features, does it have a
registered verdict, and why does it have its own forward ledger?

**Q6 — What is `h1_multiday`?** Named once as one of the four H1 families. Nothing else
known.

**Q7 — What is the `models/calendar/` artifact?** Given that the volatility family's edge
turned out to be a calendar effect, this may be the most interesting thing in the repo.

**Q8 — Branch state.** All the 2026-09 work is on `density-forecasting`. Is that merged to
main? Is main what serves?

**Q9 — What are the "two frontends"?** `CLAUDE.md` says the inference service serves two.
I have only identified the dashboard at `/`.

**Q10 — How long have the forward ledgers been running, and what do they show?** Under
`CLAUDE.md` rule 3 these are the *primary* production-worthiness signal, and they are the
one source of genuinely new, un-mined data. I have never seen their contents. Note the
2026-09-11 → 09-19 contamination boundary recorded in `e77a4eb`.

---

## 13. If you are here to find a new edge — read this first

`[VERIFIED]` Three facts that should shape any new plan:

1. **The only registered, replicated edge in the project is the H1 next-bar direction GBM**
   (§4.3): ~+1.1 to +5.0 pp over the train-majority baseline on the test block, replicating
   on AUD/USD and CHF/USD. Everything else has either DROPped, been declared
   KEEP-provisional, or turned out to be a calendar effect.
2. **The historical data is heavily mined.** Nine feature hypotheses, eight H1 hypotheses,
   at least three volatility ones, two density ones and three metalabel ones have been
   registered and spent. Both test blocks are consumed. Every further search on the same
   data tightens the bar for everyone and raises the chance of a false positive.
3. **The power arithmetic is brutal and it is not bad luck.** Detecting a 3 pp effect at the
   current bar needs ~7,634 observations per cell. Daily EUR/USD cannot supply that — C0
   would need 605 years. An effect small enough to be plausible in a liquid FX market is too
   small to measure with this data; an effect large enough to measure is too large to be
   plausible. That squeeze is structural, not a modelling failure.

`[INFERRED]` Consequently, the directions that are *not* already exhausted are: higher
frequency (H1 and below, where n is 8× larger), forward data (the ledgers, which nobody has
mined), other instruments (the pooled_h1 replications suggest cross-sectional structure),
and explicit calendar/session structure (which is the one mechanism this project has
actually found twice — Friday volatility, and the NY-session direction concentration).
Each of those needs its own registration, its own family, and its own alpha.

---

*Compiled 2026-09-21 from `CLAUDE.md` as checked into the repository and from a working
session on 2026-09-19/20 in which every `[VERIFIED]` item was inspected directly. Nothing
in this document is invented; the unknowns are listed rather than filled in.*
