# EURUSD Predictor — System Architecture & Pipeline Report

> **Scope:** Complete architectural blueprint of the EURUSD Machine Learning
> project as it exists in the current tree. Documents data flow, prediction
> generation, training, validation, and every known failure point.
>
> **Audience:** MLOps / engineering. Every claim below is traceable to a
> specific file and line; nothing is aspirational.

---

## 0. Component Map (orientation)

| Layer | File | Role |
|---|---|---|
| Research / training | `notebooks/01_data_preparation.ipynb` | 61-cell research notebook (Sections 1–20). Mirrors the standalone trainer. |
| Standalone trainer | `_train_pipeline.py` | `train_variant()` trains BOTH model variants (baseline price-only + with_macro) in one run; writes `models/<variant>/` + MLflow. |
| Inference core | `src/inference.py` | `PredictionService` — loads BOTH variants' artifacts once, serves both committees + `variant_agreement` per prediction. **Shared by both frontends.** |
| Feature engineering | `src/features.py` | The 27-column `FEATURE_COLUMNS` contract + per-variant subsets (`variant_feature_columns`), PCA on lag block, macro merge. |
| Auxiliary intraday feature engineering | `src/h1_features.py` | H1→Daily feature module for the auxiliary ensemble (§3.4) — flattened daily stats + 24h tensor, both including `Trend_vs_SMA504`/`RSI_24`. Independent of `src/features.py`. |
| Live market data | `src/live_data.py` | MT5 → yfinance fallback chain for OHLCV (also `fetch_h1_market_data` for the H1 stream). |
| Macro data | `src/macro_data.py` | FRED API → FRED public CSV → on-disk cache fallback chain. |
| COT positioning data | `src/cot_data.py` | CFTC "Traders in Financial Futures" Socrata API → cache fallback chain; availability-date look-ahead logic + trailing-window z-scores. **Tested candidate — DROPPED in both families (§4.3.2), not served.** |
| Web app (single entry point) | `api.py` | FastAPI server: serves `static/index.html` at `/`, `POST /api/predict`, `GET /history`, `POST /api/retrain`. Port 8000. |
| Feature-ablation harness | `src/ablation.py` | Validation-only KEEP/DROP arbiter + Bonferroni-corrected significance bar (see *Production Methodology*). |
| Forward paper-trading | `src/paper_trading.py` | Simulated cost-net P&L ledger from the prediction log; the primary going-forward arbiter. |
| Config | `config.json` | All hyperparameters + paths. Single source of truth. |
| Artifacts | `models/`, `results/`, `mlruns/`, `mlflow.db` | Serialized models, diagnostics, experiment tracking. |

The web layer is a **single entry point** (`api.py`) on top of one shared
`PredictionService`. All prediction logic lives in `src/` — `api.py` is only the
HTTP/dashboard layer.

---

## Production Methodology (post-defense) — governs all new feature/model claims

The academic phase is over (defended, graded); the system now **trades real
money**. A false positive here is live capital risk, not a lost exam point, so
the methodology bar is raised for everything from **2026-07-06** forward. Three
rules override the older validation narrative below wherever they conflict:

**(a) The historical test block `[80%:100%]` is SPENT for feature search.**
Feature KEEP/DROP decisions were iteratively scored against that same fixed
~1,700-row block (the original `yield_differential` + the three macro features).
Reusing one held-out block as a repeated search criterion is data-snooping — with
enough features tried, one crosses a naive 0.05 bar by chance. From now on every
feature ablation is decided on the **validation slice `[70%:80%]`** via
`src/ablation.py` (PCA/scaler/model fit on `[0:70%]` only; the test block is
never indexed there). The test block reverts to a **one-shot final report**
produced by `_train_pipeline.py`, never a knob. Re-running the four
already-tested features on the clean validation arbiter
(`results/feature_ablation_validation.csv`) leaves **all four KEEP-provisional** —
the test-block "positive point estimates" do not survive the move.

**(b) Every KEEP must clear a Bonferroni-corrected bar, not a flat 0.05.**
`results/feature_hypothesis_log.csv` is the running family count of every feature
hypothesis ever spent (seeded retroactively with the 4 already tried). The bar in
force is `alpha = 0.05 / family_size` (currently **0.05 / 8 = 0.00625** — the 4
original macro features, the 2026-07-17 `fomc_calendar_block` ADD-test (rejected:
Δacc 95% CI [−0.0549, −0.0035], McNemar p=0.0261), the 2026-07-20
`cot_positioning_block` ADD-test (rejected: Δacc CI [−0.0234, +0.0175],
p=0.8264), the 2026-07-26 `fibonacci_retracement_block` ADD-test (rejected:
Δacc CI [−0.0187, +0.0257], p=0.8376), and the 2026-07-26 `vix_regime_block`
ADD-test (rejected: Δacc CI [−0.0327, +0.0105], p=0.3584)), printed in the header
of every `src/ablation.py` report so it can never be silently forgotten. A
genuinely new feature grows the family and tightens the bar for all. At the
current bar all four macro features stay **KEEP-provisional** (smallest McNemar
p = 0.56, ~90× the bar); the four ADD-test bundles are outright DROPs.

The Fibonacci/fractal work also **built but did not spend** a hypothesis #8
(`dist_to_nearest_fib_extension_pct`, a 3-point extension/projection in
`src/fibonacci_fractals.py`): its pre-registered contingency runs #8 only if #7
clears its bar, and #7 is DROP, so #8 stays a dormant, unit-tested module with no
Bonferroni slot spent.

**(c) The forward paper-trading ledger is the primary production-worthiness
signal — not historical ablation.** `src/paper_trading.py` accumulates a
simulated, cost-net position ledger (`results/paper_trading_log.csv`, surfaced at
`/paper-trading` and `/api/paper-trading`) from each day's live committee call as
new sessions settle. Whether the system deserves real capital should be judged by
this ledger showing a **genuine, cost-net edge over months** (win rate, Sharpe-
like, cumulative net pips, max drawdown), not by any further re-analysis of the
spent test block. It is **simulated only** — there is deliberately no broker,
order-execution, position-sizing, or stop-loss code anywhere; real execution is a
separate, larger risk-management conversation that should happen only *after* this
ledger demonstrates a supported edge.

See `IMPROVEMENT_LOG.md` → "Production methodology hardening" for the commit trail.

---

## Dual-Variant Architecture (baseline vs with_macro)

A direct consequence of the Production Methodology above: the four FRED macro
features are **statistically unproven** (all KEEP-provisional under the
Bonferroni-corrected validation bar), so instead of betting the single
production model on them, the system trains and serves **two complete model
families side by side** and lets forward evidence arbitrate:

| Variant | Feature set | FRED dependency | Artifacts |
|---|---|---|---|
| `baseline` | `PRICE_FEATURE_COLUMNS` — 23 price-derived columns, **zero macro features** | none (immune to FRED outages by construction) | `models/baseline/` |
| `with_macro` | full 27-column `FEATURE_COLUMNS` (adds `yield_differential_delta`, `usd_index_return`, `policy_rate_differential`, `inflation_differential`) | FRED API → public CSV → cache fallback | `models/with_macro/` |

Design rules (enforced by code + tests):

- **One training body.** `_train_pipeline.py::train_variant()` is the single
  parametrized trainer; the run loops over `config.json → variants`. Never
  duplicate the training logic per variant.
- **Identical rows, different columns.** Both variants train on the SAME
  euro-era engineered row set (the macro merge's 1999+ truncation), so the
  comparison isolates the feature set itself, not a training-span difference.
  Same unified 70/80/100 chronological split, same percent targets.
- **Fully self-contained artifact sets.** Each variant owns its own
  `lag_scaler`/`lag_pca`/`global_scaler`/GBM heads/LSTM under
  `models/<variant>/` — the lag block is currently identical across variants,
  but separate fits are kept deliberately to rule out any cross-variant
  coupling. Never mix one variant's scaler with another's model.
  (`test_smoke.py` asserts all 14 daily artifacts.)
- **Independent serving gates.** `PredictionService` exposes `baseline_ready` /
  `macro_ready`; a missing or corrupt variant degrades to an `error` note in
  its response block while the other variant keeps serving.
- **Every prediction returns both.** `POST /api/predict` →
  `{ baseline: {gbm,lstm,consensus}, with_macro: {...}, variant_agreement }`.
  `variant_agreement=false` is the most informative single signal on the
  dashboard: it means the unproven macro block is *actually changing the
  decision* that day. The UI labels the macro panel **experimental/unproven**
  (with the Bonferroni context in the tooltip) — the two variants are NOT
  presented as equally validated.
- **Two forward ledgers.** `results/paper_trading_log_baseline.csv` and
  `results/paper_trading_log_macro.csv` (config: `paper_trading.ledgers`)
  accumulate separately from the same prediction log — the macro ledger drives
  off the historical `pred_*` columns (continuous lineage back through the
  pre-dual era), the baseline ledger off the new `baseline_*` columns
  (accumulating from the first dual prediction). Whichever variant nets better
  cost-adjusted P&L over a meaningful forward window is the honest winner.
- The **H1→Daily ensemble** is price-only by construction, trains once, stays
  at `models/` root, and is shared by both variants' responses.

---

## 1. End-to-End Execution Cycle

### 1.1 Process initialization (once, at startup)

`api.py` constructs `PredictionService(BASE_DIR, CONFIG)` once at startup.
The constructor (`src/inference.py:21-71`) **eagerly deserializes every artifact
exactly once**, each in an independent `try/except` that appends to
`self.load_errors` rather than failing fast:

1. **PCA pair** — `lag_scaler.pkl`, `lag_pca.pkl`
2. **Global scaler** — `global_scaler.pkl` (a single StandardScaler shared by both model families)
3. **GBM pair** — `best_gbm_eurusd.pkl` (classifier), `best_gbm_regressor_eurusd.pkl`
4. **LSTM pair** — `lstm_multitask_eurusd.keras`, `lstm_time_steps.pkl`
5. **Historical context** — `results/eurusd_features.csv` via `load_history()`

Readiness gates are then computed:
- `pca_ready` = both PCA artifacts present
- `scaler_ready` = the single `global_scaler.pkl` present
- `gbm_ready` = `pca_ready` **and** `scaler_ready` **and** both GBM models present
- `lstm_ready` = `pca_ready` **and** `scaler_ready` **and** LSTM model + time_steps present
- `models_ready` = `(gbm_ready or lstm_ready)` **and** history loaded

> **Design consequence:** the service degrades gracefully. A missing LSTM file
> still leaves a servable GBM-only pipeline (and vice versa). `api.py` gates
> on `models_ready` (returns `503` if false).

### 1.2 Per-request lifecycle (the "today → t+1" cycle)

Triggered by `POST /api/predict` (`api.py`), which calls `service.predict()`
(`src/inference.py`):

```
predict()
  └─ _resolve_latest_window(time_steps=max(lstm_time_steps,1))   # the data pipeline
        ├─ fetch_live_market_data()      # OHLCV: MT5 → yfinance → history fallback
        ├─ fetch_yield_differential()    # macro: FRED api → FRED public → cache
        ├─ merge_macro_features()        # tz-align + ffill yield_differential
        ├─ compute_features()            # 24 FEATURE_COLUMNS
        ├─ dropna(subset=FEATURE_COLUMNS)# drop warm-up rows, KEEP latest bar
        └─ apply_lag_pca()               # 6 lag cols → k principal components
  ├─ _predict_gbm(window.iloc[-1])       # if gbm_ready  — single latest row
  ├─ _predict_lstm(window.tail(steps))   # if lstm_ready — (time_steps, n_feat) window
  └─ compute_consensus(predictions)      # committee aggregation
```

**"Today" is never supplied by the caller.** It is inferred as the most recent
bar returned by whichever live source answers first (`src/inference.py:73-91`).
The forecast target is mechanically `as_of_date + 1 day` (`src/inference.py:125`).

### 1.3 How the date drives the macro (FRED) fetch

After OHLCV is resolved, `_resolve_latest_window` calls
`fetch_yield_differential(ohlcv_df.index.min(), ohlcv_df.index.max(), ...)`
(`src/inference.py:92-97`). The **start/end of the FRED request are derived from
the live price index**, so the macro window always matches the price window of
the current run. The fallback chain is in §2.2.

---

## 2. Data Ingestion & Processing Flow

### 2.1 OHLCV ingestion — `src/live_data.py`

`fetch_live_market_data(mt5_symbol, yf_symbol, bars)` (`src/live_data.py:58`)
is a strict fallback chain:

| Tier | Function | Source | tz | Returns on failure |
|---|---|---|---|---|
| 1 | `_fetch_from_mt5` | MT5 `copy_rates_from_pos(... TIMEFRAME_D1 ...)` | tz-naive | `None` (import error, no terminal, empty) |
| 2 | `_fetch_from_yfinance` | `yf.Ticker(symbol).history(...)` | tz-stripped to naive | `None` |
| 3 | *(caller)* | `history_df.tail(bars)` bundled CSV | — | label `"history_fallback"` |

Tier 3 is applied **by the caller** in `src/inference.py:88-90` when no live
source returns ≥ `200 + time_steps` bars (the SMA_200 + LSTM-window warm-up
floor). `bars_needed = max(live_fetch_bars=250, 200 + time_steps)`.

> **Note on `tick_volume`:** loaded and surfaced to the UI for display, but
> **deliberately excluded from `FEATURE_COLUMNS`** (`src/features.py:14-19`).
> MT5 tick-volume is a broker tick count, not traded volume, and decades of
> placeholder `1`s in the 1971-era history contaminated the fitted scaler,
> causing the LSTM to extrapolate 8σ out on live volume. It never reaches a model.

### 2.2 Macro ingestion — `src/macro_data.py`

`fetch_yield_differential(start, end, series_ids, cache_path)`
(`src/macro_data.py:62`) computes the **US 10Y − DE 10Y bond-yield differential**
(`DGS10` − `IRLTLT01DEM156N`). Fallback chain:

| Tier | Function | Requires | Source label |
|---|---|---|---|
| 1 | `_fetch_via_fredapi` | `FRED_API_KEY` (≠ placeholder) | `FRED_api` |
| 2 | `_fetch_via_pandas_datareader` | nothing (public CSV) | `FRED_public` |
| 3 | on-disk `results/yield_differential.csv` | prior cache | `cache` |
| — | none reachable → `(None, None)` | — | `unavailable` → caller defaults to `0.0` |

**Series alignment (`_combine`, `src/macro_data.py:11-31`):** the two series are
concatenated, the index is coerced to **UTC**, sorted, and **forward-filled**
(the monthly German series is carried forward onto the daily US index). ffill
only ever carries a *past* value forward — never a future value backward — so
**no look-ahead** is introduced. The spread is `us10y − de10y`.

> **Cache-write behavior (recently hardened):** on a successful live fetch the
> result is now **merged onto the existing cache** and de-duplicated
> (`keep='last'`) before writing (`src/macro_data.py:80-101`), rather than
> overwriting it outright. See §4.4 — this fixed a defect that previously
> truncated 54 years of cached history down to whatever narrow window the
> current price fetch happened to cover.

### 2.3 Merge & timezone alignment — `merge_macro_features`

`src/features.py:123-142`: left-joins `yield_differential` (the raw US10Y−DE10Y
**level**) onto the OHLCV index. The OHLCV index is localized/converted to UTC for
the join, the differential is `ffill()`-ed across weekend FX bars and bond
holidays, then the **original index is restored**. Zero look-ahead regardless of
calendar offset. This function is unchanged by §4.3's feature revision below — it
still produces the raw level, used for the dashboard's human-readable display;
the model-facing transform (`yield_differential_delta`) is derived downstream in
`compute_features` (§2.4), not here.

### 2.4 Feature transformation — `compute_features`

`compute_features` produces the canonical **27 `FEATURE_COLUMNS`**
(`src/features.py`):

| Group | Columns | Math |
|---|---|---|
| Raw price | `open, high, low, close` | passthrough |
| Stationarity | `log_return` | `ln(close / close.shift(1))` |
| Trend | `SMA_21, SMA_50, SMA_100, SMA_200` | rolling means |
| Volatility | `volatility_20` | rolling std of log_return |
| Bar shape | `bar_dynamics` | `(high − low) / open` (0-open → NaN guard) |
| Autoregressive lags | `return_lag_1..3`, `dynamics_lag_1..3` | shifted log_return / bar_dynamics |
| Cyclical time | `day_sin, day_cos, month_sin, month_cos` | sin/cos encoding (wrap-around preserved) |
| Range | `ATR_14` | True Range, 14-period EWM (`com=13`) |
| Bands | `BB_width` | `4·std / mid` (normalized Bollinger width) |
| Exogenous macro — yield | `yield_differential_delta` | `yield_differential.diff(1)` — raw level (pre-merged) stationarized like `log_return`; see §4.3 |
| Exogenous macro — USD | `usd_index_return` | `ln(usd_index / usd_index.shift(1))` on the merged DTWEXBGS level. RETURN not level (the level is ~57% EUR-weighted, near-collinear with EUR/USD). DTWEXBGS starts 2006 → pre-2006 rows get a flat **0** (fillna) so they are not truncated; see §2.6 |
| Exogenous macro — rates | `policy_rate_differential` | DFF (effective fed funds) − ECBDFR (ECB deposit rate), passthrough level; see §2.6 |
| Exogenous macro — inflation | `inflation_differential` | US CPI YoY% (CPIAUCSL) − DE HICP YoY% (CP0000DEM086NEST), computed in `macro_data`, passthrough; see §2.6 |

**Critical live-edge property:** `compute_features` does **not** compute targets
and does **not** `dropna`, so the most-recent bar (which has no future bar to
form a target) **survives**. `add_advanced_features` (`src/features.py:108-120`)
is the *training-only* variant that additionally builds `target_return` /
`target_direction` and drops NaNs.

### 2.5 Where the deserialized `StandardScaler` + `PCA` are applied (unified 80% split)

Both preprocessing components are fit on the **identical unified train block —
the first 80%** of history (`train_fraction = 0.80`), and the held-out
`[80%:100%]` test block is seen by neither fit. This removes the prior
"non-obvious coupling" where the PCA was fit on 70% while the GBM split at 80%.

| Stage | Object | Fitted in training on | Applied at inference in |
|---|---|---|---|
| Lag dim-reduction | `lag_scaler` + `lag_pca` | unified **0–80%** train block | `apply_lag_pca()` |
| Global feature scaling | `global_scaler` (one StandardScaler) | unified **0–80%** train block | `_predict_gbm` **and** `_predict_lstm` |

`apply_lag_pca` (`src/features.py:181-194`) drops the 6 raw `LAG_COLUMNS` and
appends `lag_pca_1..k`. `model_input_columns` (`src/features.py:197-200`)
re-derives the exact post-PCA column order so training and inference never
diverge. **A single `global_scaler` now serves BOTH model families** — the
former separate `scaler_gb` / `scaler_lstm` are gone. The LSTM's early-stopping
validation slice `[70%:80%]` sits *inside* the scaler/PCA fit window, but the
final test block `[80%:100%]` does not, so reported test metrics stay
leakage-free.

### 2.6 Macro feature expansion (four FRED features, generalized fetcher)

`src/macro_data.py` was generalized so the API → public-CSV → cache fallback
chain (§2.2) is written **once** in `fetch_fred_feature` and reused by every macro
feature; `fetch_macro_features` fans out to all four and returns one UTC frame
(`fetch_yield_differential` is kept as a thin backward-compatible wrapper). Each
feature has its **own** cache file under `results/` so a thin live fetch of one
never truncates another's longer cached history.

| Model feature | FRED series | Live start | Notes |
|---|---|---|---|
| `yield_differential_delta` | DGS10, IRLTLT01DEM156N | 1970 | existing (§4.3) |
| `usd_index_return` | DTWEXBGS | 2006 | log-return; the Fed discontinued the pre-2006 trade-weighted indices (DTWEXM ends 2019), so no live series reaches earlier — pre-2006 rows get a flat 0 return rather than truncating history |
| `policy_rate_differential` | DFF − ECBDFR | 1999 | DFF (effective fed funds, from 1970) chosen over DFEDTARU (2008) so the binding floor is the euro's own 1999 start, not 2008 |
| `inflation_differential` | CPIAUCSL YoY − CP0000DEM086NEST YoY | 1997 | CP0000DEM086NEST (Eurostat HICP DE, live) replaces DEUCPIALLMINMEI (stale since 2025-03). YoY needs 12 prior months, so the fetcher pulls `yoy_lookback_days=420` extra so the feature is defined at the live edge |

**History consequence (deliberate).** Requiring the differentials non-NaN makes
`ECBDFR` (1999-01) the binding floor, so `add_advanced_features` now truncates
training to the **real euro era (1999+, ~8,560 rows)** and drops the ~28 years of
**synthetic pre-euro DEM-proxy** bars the 24-column model trained on (EUR/USD did
not exist before 1999; those bars are a backfilled proxy). This is a data-quality
improvement but shifts every split boundary, so 1999+ metrics are **not**
comparable row-for-row with the old 1971+ ones — the old 24-col/1971+ baseline is
preserved in `results/comparison_table.csv`. `add_advanced_features` drops NaN
only on `FEATURE_COLUMNS + targets` (not on the intermediate merged levels
`usd_index`/`yield_differential`), so `usd_index`'s 2006 start does not drag the
floor to 2006. LSTM splits stay healthy at the 1999+ size (train ≈ 5,830 seq,
val ≈ 816, test ≈ 1,653).

**No look-ahead.** `merge_macro_features` aligns each macro series onto the price
index by as-of forward-fill (reindex onto the union, `ffill`, select the OHLCV
dates), so a monthly CPI print propagates onto every later daily bar — even when
the month-start lands on a non-trading day — and never a future value backward.
Guarded by `test_*_no_lookahead_*` for all four macro features. See §4.3.1 for the
(provisional, not-significant) ablation of the three added features.

---

## 3. Prediction Logic & Model Architecture

### 3.1 GBM dual pipeline (tree ensemble)

`_predict_gbm` (`src/inference.py:129-146`) consumes **one flat PCA-reduced row**
(`window.iloc[-1]`), scales it with the single `global_scaler`, and runs two heads:

- **Classifier** `best_gbm_eurusd.pkl` — `xgb.XGBClassifier`, tuned for
  `roc_auc`. Emits `predict_proba` → `direction` (UP/DOWN) + `confidence`.
- **Regressor** `best_gbm_regressor_eurusd.pkl` — `XGBRegressor(objective='reg:pseudohubererror')`,
  tuned for MAE. Emits `predicted_return` **natively in percent** — the regressor
  is now trained on the percent target produced by `src/features.py`, so there is
  **no `*100` rescaling** at inference.

  **Why a Huber-family loss, not squared error (ESL §10.6):** daily EUR/USD
  returns are a long-tailed, occasionally outlier-prone target (quiet noise most
  days, occasional large jumps around macro releases) — exactly the scenario ESL
  cites as squared-error's weak point ("its performance severely degrades for
  long-tailed error distributions and especially for grossly mis-measured
  y-values"). A Huber-family loss trades squared-error's sensitivity near zero for
  a linear (robust) penalty on large residuals.

  **Known doc/config drift on `huber_alpha`:** `config.json → gbm.huber_alpha`
  (`0.9`) is logged to MLflow as a record of intent but is **never passed to the
  `XGBRegressor` constructor** — it does not control anything at training time.
  sklearn's `GradientBoostingRegressor(loss='huber', alpha=0.9)` (which the name
  and the `0.9` value evoke) would set its `δ` threshold *adaptively per boosting
  iteration* to the 90th-percentile of current absolute residuals — but this repo
  trains `xgb.XGBRegressor`, whose pseudo-Huber objective instead exposes a fixed
  `huber_slope` hyperparameter (default `1.0`, unset here), not an adaptive
  quantile. In short: the `huber_alpha` name and value currently describe a
  mechanism (sklearn's adaptive-quantile Huber) that is **not the one actually
  running** (XGBoost's fixed-slope pseudo-Huber). This is a documentation/naming
  gap, not a correctness bug — training still proceeds with a sensible default —
  but wiring `huber_slope=CONFIG['gbm']['huber_alpha']` explicitly (or renaming the
  config key to avoid the implied sklearn semantics) is an open follow-up.

### 3.2 Multi-Task LSTM (sequence model)

Built with the Keras Functional API (`_train_pipeline.py:230-243`,
mirrored in notebook Section 19b):

```
Input(time_steps=20, n_features)
   └─ LSTM(units=64, name="shared_lstm_trunk")     # shared trunk
        └─ Dropout(0.3)
             ├─ Dense(1, linear,  name="return_output")     # head 1: % return
             └─ Dense(1, sigmoid, name="direction_output")  # head 2: UP prob
```

`_predict_lstm` (`src/inference.py:148-170`) consumes a **`(20, n_features)`
sliding window**, scales with the **same `global_scaler`** the GBM uses,
reshapes to `(1, 20, n_features)`, and returns both heads. The return head
outputs **percent natively**, exactly like the GBM regressor — both heads are
trained on the percent target from `src/features.py`, so neither path applies a
`*100`. The former GBM-fraction / LSTM-percent asymmetry is **resolved**; see
§4.5.1.

> **Why percent units?** Fractional log-returns (std ≈ 0.006) give MSE ≈ 3e-5,
> five orders of magnitude below the direction head's BCE (≈ 0.69). At the old
> loss weights the shared trunk got almost no gradient for the return head
> (observed: −11% predicted returns with the wrong sign). Producing the target
> natively in percent (in `src/features.py`, the single source of truth)
> rebalances the two losses so `loss_weights` can stay `1.0 / 1.0`.

### 3.3 Committee Consensus (with low-confidence guard)

`compute_consensus` (`src/inference.py`), static method, gated by the class
constant `CONFIDENCE_THRESHOLD = 0.52`:

- **Agreement** (both heads same direction): average the two confidences and
  the two predicted returns — **unless** that averaged confidence is strictly
  below `CONFIDENCE_THRESHOLD`. In that case the unanimous-but-coin-flip call is
  **downgraded**: `agreement` is overridden to `False` and the consensus
  `direction` becomes the literal flag **`"MIXED / LOW CONFIDENCE"`**. Because
  the direction heads sit near chance (ROC-AUC ≈ 0.50), this stops a coin-flip
  agreement from being advertised as a confident ensemble call.
- **Disagreement:** defer to the **higher-confidence** model and set
  `agreement=False` so the UI can flag it — rather than silently averaging
  across opposite-signed predictions.

The response dict carries `as_of_date`, `forecasting_date`, `data_source`,
`bar_used` (incl. `macro_source`), the per-model blocks, and `consensus`.

### 3.4 Auxiliary H1→Daily Ensemble (XGBoost / RandomForest / SVM / LSTM)

A second, **fully independent** predictor sits alongside the daily GBM+LSTM
committee above. It answers the same next-day question from a different data
source — hourly (H1) OHLCV collapsed into daily statistics — rather than from the
daily bar history `src/features.py` consumes. It is additive: it never touches
the 7 canonical daily artifacts, has its own readiness gate, and degrades
independently if its data or models are unavailable.

**Data & features — `src/h1_features.py`:** two load paths share one UTC-indexed
H1 OHLCV stream. TRAINING uses the cache-first `load_h1_frame` (safe because
`_train_pipeline.py` explicitly refreshes the cache beforehand). INFERENCE uses
`refresh_h1_frame` — live-first with a **mandatory staleness gate**: the cache is
served untouched when its last COMPLETE session (the same `MIN_HOURS ≥ 12` rule
`aggregate_daily_features` applies) already is the expected latest weekday
session; only a genuinely behind cache triggers
`src/live_data.py::fetch_h1_market_data` (MT5 → yfinance → cache). A live pull
thin in history is merged onto the cached rows (dedup by index, live wins) and
the merged frame rewritten to the cache, so the SMA504/RSI trailing warm-ups
never silently truncate (the H1 analogue of the daily SMA_200 warm-up handling,
§4.5.1); a fully failed live chain degrades to the stale cache. The chosen path
is reported as `"live"` / `"cache"` / `"live+history_backfill"`. (Historical
bug: inference originally used cache-first `load_h1_frame`, so the served H1
day froze at the last retrain's cache write — pinned by
`test_h1_inference_refreshes_stale_cache_live_first` and
`test_h1_staleness_gate_skips_live_fetch_when_cache_current`.) Two aligned representations are built from it, sharing one daily
index and one `shift(-1)` next-day target (`build_daily_target`, percent, same
no-look-ahead contract as `src/features.py`):

| Representation | Shape | Consumers | Columns |
|---|---|---|---|
| Flattened daily stats | `(n_days, 11)` | XGBoost / RandomForest / SVM | `Intraday_Volatility`, `Intraday_Momentum`, `Daily_Range`, `H1_Moving_Average`, `H1_Volume_Mean`, `H1_Return_Skew`, `H1_Max_Abs_Return`, `First_Half_Return`, `Second_Half_Return`, `Trend_vs_SMA504`, `RSI_24` |
| 24h tensor | `(n_days, 24, 5)` | LSTM (seq2vec) | `log_return`, `hl_range`, `co_change`, `volume`, `rsi_24` per hour |

`Trend_vs_SMA504` (`close / SMA504 − 1`, a 504-H1-bar ≈ 21-trading-day trend
baseline) and `RSI_24`/`rsi_24` (a 24-period RSI, i.e. one trading day of hourly
momentum) are computed on the **continuous hourly stream** with trailing-only
windows (`_rsi`, `_enrich_hourly`), so they carry cross-day context without
violating the no-look-ahead invariant — verified by
`test_h1_features_do_not_depend_on_future_days`.

**Training — `_train_pipeline.py` Section 13:** additive to the daily pipeline;
refreshes the H1 cache (`fetch_h1_market_data`, live → existing-cache fallback) and
retrains all four models on a chronological 80/20 split of the flattened dataset,
scored with `TimeSeriesSplit`, mirroring the daily invariants (§ above) but on its
own split and its own scalers (`h1_feature_scaler.pkl`, `h1_lstm_scaler.pkl` — kept
fully separate from the daily `global_scaler.pkl`).

**Serving — `src/inference.py`:** `PredictionService` loads the 8 H1 artifacts
(`h1_xgb_regressor.pkl`, `h1_rf_regressor.pkl`, `h1_svm_regressor.pkl`,
`h1_feature_scaler.pkl`, `h1_lstm_scaler.pkl`, `h1_feature_columns.pkl`,
`h1_lstm_config.pkl`, `h1_lstm.keras`) independently of the daily 7, gated by
`h1_ready` (all 8 present). `predict()` only attempts `_predict_h1()` if
`h1_ready`; any failure there (thin H1 feed, feature-shape mismatch after a
feature-set change not yet retrained, etc.) is caught and surfaced as
`response['h1_error']` — it **never** fails the daily prediction. `_predict_h1`
runs on the latest **complete** trading day (`build_h1_inference_sample` drops the
still-forming current UTC day), and each of the four models is a **return-only
regressor** — direction is derived from the sign of the predicted return, there is
no calibrated probability.

`compute_h1_consensus` (static method) aggregates the four regressors
independently of the daily committee's `compute_consensus`, but with the SAME
vote-based design: direction is the **strict** majority sign; an exact 2–2 vote
has no majority and is labeled **`MIXED / TIE`** (mirroring the daily
`MIXED / LOW CONFIDENCE` honesty — never an arbitrarily crowned side).
`confidence` is the **fraction of models on the majority side** (a genuine
[0.5, 1.0] agreement measure — **not** a calibrated probability, unlike the daily
consensus's `confidence`; exactly 0.5 on a tie). `predicted_return_pct` is the
mean over the **majority-side models only**, so the number is sign-consistent
with the direction label by construction (an H1 model's direction IS the sign of
its return, so a full-panel mean can contradict a 3–1 vote whenever the
minority's magnitude dominates); on a tie it is the full-panel mean, reported as
context with no directional claim. `agreement=True` only on a unanimous sign.

> **Fixed bug (2026-07-07):** the original implementation broke 2–2 ties with
> `up >= down` (arbitrary "UP") while displaying the full-panel mean return —
> the live dashboard showed *"UP — 50% model agreement"* over a **negative**
> −0.0131% average. Regression-guarded by
> `test_compute_h1_consensus_exact_tie_is_mixed_not_arbitrary_up`.

```jsonc
"h1": {
  "as_of_date": "YYYY-MM-DD",
  "data_source": "live | cache | live+history_backfill",  // refresh_h1_frame's chosen path
  "predictions": {
    "h1_xgboost":       { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_random_forest":  { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_svm":            { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_lstm":           { "direction": "UP|DOWN", "predicted_return_pct": float }
  },
  "consensus": { "direction": "UP|DOWN|MIXED / TIE", "agreement": bool, "confidence": 0.5-1.0, "predicted_return_pct": float, "n_models": 4 }
}
// or, on any failure: "h1_error": "<message>"
// A MIXED / TIE consensus (exact 2-2 vote) is NOT scored by /history and takes
// no position anywhere -- same no-directional-claim handling as the daily MIXED.
```

`src/tracking.py::log_prediction` also logs the H1 consensus (`h1_direction`,
`h1_return_pct`, `h1_agreement`) alongside the daily forecast, and
`build_history_html` scores it against the same realised close in its own
"H1 ensemble" column with its own hit-rate — independent of the daily
committee's hit-rate. The static UI (`static/index.html`) renders the H1 block as
a separate "Auxiliary Intraday Ensemble" section below the daily cards.

### 3.5 Next-Day Realized Volatility (5-seed multi-task LSTM ensemble)

A **genuinely different prediction task** from direction/return: the target is
`target_volatility_pct = |next-day log return| × 100` (`src/features.py`, same
`shift(-1)` convention and percent unit as `target_return`). Unlike next-day
direction (near-efficient, ROC-AUC ≈ 0.50, §4.2.1), **volatility clustering is a
well-established FX stylized fact** — and this is, accordingly, the only neural
model family in the project with a CI-confirmed edge over its honest baseline.

**Methodology (`src/volatility.py`, entirely under the post-defense Production
Methodology):**

- **Mandatory baselines first.** A GARCH(1,1) (`arch` package) with parameters
  fit on the train block ONLY, rolled forward with FIXED parameters
  (`ARCHModel.fix`) so every validation/test forecast uses only past data —
  plus a naive persistence baseline (today's |return| = tomorrow's forecast).
  GARCH's conditional σ is converted to an E|r| point forecast via the
  folded-normal factor √(2/π). On validation: GARCH MAE 0.2038% / R² +0.009;
  persistence MAE 0.2611% / R² −0.842.
- **Validation-only arbiter.** All decisions on `[70%:80%]`; the experiment's
  LSTMs fit on `[0:63%]` with `[63%:70%]` as the early-stopping tail so the
  arbiter is genuinely held out of everything the experiment fits (stricter
  than the production LSTM convention, because here validation IS the arbiter).
- **Its own hypothesis family** (`results/volatility_hypothesis_log.csv`) —
  continuous R²/MAE metrics, separate from the 4-feature direction/return
  family; the Bonferroni bar tightened as it grew: 0.05 → 0.025 → 0.0167,
  then (family re-opened 2026-07-17 with two pre-declared candidate INPUT
  features, judged at the final-family bar 0.05/5 = 0.01) → 0.01.
  Hypotheses 4–5 — `RSI_14` (daily 14-period RSI, the H1 `_rsi` formula) and
  `BB_percent_b` (%B from the same 20-day mean/σ as `BB_width`), each added
  alone to the 5-seed ensemble's input set vs a same-seeds base ensemble —
  were both **null results (DROP)**: ΔMAE CI99 [−0.0022, +0.0001] for RSI_14
  and [−0.0025, −0.0004] for %B (the latter CI-confirmed *worse* on MAE).
  Hypothesis 6 (bar → 0.05/6 = 0.0083) — the `fomc_calendar_block` bundle
  (`is_fomc_day`/`days_to_next_fomc`/`days_since_last_fomc` from the
  scheduled-meetings calendar `results/fomc_dates.csv`, built by
  `src/fomc_calendar.py` from the official Fed pages; three views of one
  calendar fact = ONE hypothesis slot) — also a **null result (DROP)** despite
  the strong mechanistic prior: ΔMAE CI99.2 [−0.0024, +0.0016], ΔR²
  [−0.0207, +0.0174] (`results/volatility_candidate_fomc.csv`); the same
  bundle also failed the direction/return family's bar (its hypothesis #5).
  The candidate constructors live in
  `src/volatility.py::add_volatility_candidate_features` and
  `src/fomc_calendar.py::add_fomc_features` (candidate-only — guarded out of
  the direction/return `FEATURE_COLUMNS` by a unit test); the production
  input set remains unchanged. Any future volatility hypothesis faces
  0.05/7 ≈ 0.0071.
- **Training-noise honesty.** Single-seed runs exposed TF/oneDNN CPU
  nondeterminism of the same order as the deltas under test (identical-seed
  dedicated-model MAE moved 0.190→0.197 between runs). Bootstrap CIs capture
  row-sampling noise only, so the ship candidate was pre-registered as the
  **seed-ensemble** (mean over seeds 42–46) of the 3-head multi-task
  architecture (which beat the dedicated single-head model head-to-head at
  the tightened bar — sharing the trunk with return/direction HELPS the
  volatility head).
- **The pre-registered ship gate cleared decisively**
  (`results/volatility_seed_ensemble.csv`): MT 5-seed ensemble MAE 0.1859% /
  R² +0.144 vs GARCH 0.2038% / +0.009; ΔMAE CI98.33 [+0.0111, +0.0242],
  ΔR² CI [+0.080, +0.183], frac(ΔMAE>0)=1.000 — every individual seed also
  beat GARCH. One-shot test-block report (never a search knob): ensemble
  MAE 0.2188% / R² +0.110 vs GARCH 0.2326% / +0.036 — the edge generalizes.

**Production artifacts** (`models/volatility/`, produced by
`train_production_volatility_model` inside `_train_pipeline.py` §12B):
5 × `volatility_lstm_seed{42..46}.keras` + its own `lag_scaler/lag_pca/
global_scaler` (fit `[0:80%]`) + `lstm_time_steps.pkl` + `vol_metrics.json`
(one-shot test report + the validation ship-gate evidence, consumed by
serving for honest framing). **Price-only by nature** — ONE family, no
baseline/with_macro duplication; GARCH and volatility consume no macro columns.

**Serving:** `PredictionService` loads the family behind its own `vol_ready`
gate; because the VALIDATED object is the full 5-seed mean, a partial ensemble
refuses to serve (all-or-nothing load). `predict()` adds a
`volatility_forecast` block — `predicted_vol_pct` (the seed-averaged
volatility head; the ensemble's return/direction heads are training
scaffolding and are discarded) plus `vs_garch_baseline` /
`vs_persistence_baseline` / `test_report_one_shot` context from
`vol_metrics.json`. The UI renders it as a direction-free "expected movement
magnitude" card labeled `✓ validated vs GARCH(1,1)` — the framing matches
exactly what the rigorous test found, no more.

### 3.6 H1 TI-LSTM (observational — NOT validated; in production by owner override)

**Status warning for every future reader: this model's presence in production
is NOT validation.** The H1-native technical-indicator LSTM
(`src/ti_lstm_h1_experimental.py`: %B-20, MACD 13/34 with 8-SMA signal, trend
vs SMA-504/168, RSI-24, CCI-20, ADX-14 over the last complete session's 24
hourly bars; next-day direction/return heads) **FAILED its pre-registered
hypothesis bar** (`results/ti_lstm_h1_hypothesis_log.csv`: DROP — one-shot
test AUC 0.5128 vs the existing H1 ensemble's 0.5283, ΔAUC −0.015
CI95 [−0.072, +0.042], point estimate negative). By **explicit owner decision
(2026-07-18)** it was wired into serving anyway, for transparent forward
observation via its own paper-trading ledger — overriding, for this one
model, the "only ship what clears the bar" rule. Honesty contract: the
`ti_h1_forecast` response block carries `validated: false` + the verbatim
test numbers; the UI card is amber-warning-framed ("⚠ Not Validated — No
Demonstrated Edge"), deliberately NOT the volatility card's validated badge
nor the macro panel's "nominally positive" framing.

Mechanics: artifacts in `models/ti_lstm_h1/` (2×64, seed 42 — the Keras 3
**torch/CUDA** backend was verified bit-deterministic); `ti_h1_ready`
all-or-nothing gate mirrors `vol_ready`; the `.keras` file is
backend-portable, so serving loads it under tf.keras with **no torch
dependency**. Retraining runs as a SUBPROCESS (`_train_pipeline.py` §12C) —
mandatory, because KERAS_BACKEND freezes at the first keras import and the
pipeline process already imported tf.keras. Its forward ledger is
`results/paper_trading_log_ti_h1.csv` (config `paper_trading.ledgers.ti_h1`,
driven by the `ti_h1_direction` prediction-log column).

---

## 4. Testing, Validation & Error Diagnostics

### 4.1 Cross-validation strategy

**`TimeSeriesSplit`** (never random K-fold) is used everywhere a temporal model
is tuned:
- GBM: `TimeSeriesSplit(n_splits=cv_splits=5)` inside `GridSearchCV`
  (`_train_pipeline.py:106-126`), classifier scored on `roc_auc`, regressor on
  `neg_mean_absolute_error`.
- All train/val/test splits are **chronological fractions** from `config.json`
  (`train_fraction=0.80`, `val_fraction=0.10`): the GBM trains on `[0:80%]`, the
  LSTM on `[0:70%]` with `[70%:80%]` reserved for early-stopping, and **both**
  test on the identical held-out `[80%:100%]`. No shuffling.

### 4.2 Validation metrics & where they live

| Metric | Model | Logged to |
|---|---|---|
| `direction_accuracy`, `direction_roc_auc` | GBM & LSTM | MLflow (`_train_pipeline.py:154,287`) |
| `return_mse`, `return_mae` | GBM & LSTM | MLflow (both heads in **percent** units — directly comparable) |
| Learning curves, confusion matrices, residuals, ACF/PACF | notebook | `results/*.png` |
| Multi-model CV table | notebook | `results/comparison_table.csv` |
| FRED ablation | notebook §2C | `results/2C_fred_ablation.csv` |

**Honest performance reality (from committed artifacts):**

- `results/comparison_table.csv`: every model sits at **ROC-AUC ≈ 0.51–0.52**,
  accuracy ≈ 0.51 — i.e. **marginally above chance**. Tuned XGBoost hold-out
  accuracy is **0.499** (below chance). This is consistent with the efficient-
  market difficulty of daily FX direction and should be communicated as such,
  not oversold.
- The retrained production heads (unified 80% split, percent target) score on
  the held-out test block: **GBM** Direction Acc = 0.5011, ROC-AUC = 0.5024,
  Return MAE = 0.296%; **LSTM** Direction Acc = 0.5018, ROC-AUC = 0.4997,
  Return MAE = 0.304%. The two return heads are now on the **same percent scale**
  (MAE ≈ 0.30% each) and directly comparable, but the **direction heads still do
  not beat a coin flip** — the low-confidence consensus guard (§3.3) exists
  precisely for this.

### 4.2.1 The Efficient Market Reality

A recurring question is *"why are the predicted returns such tiny fractions of
a percent (e.g. `-0.0225%`)?"* The answer is that this is **mathematically
correct behaviour, not a defect** — and the test block proves it numerically.

**The "Predict the Mean" baseline.** EUR/USD daily returns are, to a very good
approximation, a **random walk**: their unconditional mean is ≈ 0 and a typical
day moves ± one standard deviation. On the held-out test block:

| Quantity | Value |
|---|---|
| Actual next-day return — mean | **+0.0060%** (≈ zero) |
| Actual next-day return — std | **0.5846%** (the typical daily move) |
| GBM regressor — predicted mean | **+0.0062%** (≈ the unconditional mean) |
| GBM regressor — predicted std | **0.0057%** (≈ **100× tighter** than reality) |
| `corr(prediction, actual)` | **≈ 0.000** (essentially no signal) |

The decisive comparison is the MAE:

| Predictor | Test MAE |
|---|---|
| Trivial **"predict the mean"** baseline (`ŷ = mean(y)`) | **0.2958%** |
| The trained **GBM regressor** | **0.2959%** |

The GBM is **indistinguishable from — in fact a hair worse than — a constant
that always predicts the historical average**. This is the empirical signature
of an efficient market: there is almost no day-ahead signal in the price/feature
history to extract, so no estimator can do materially better than the mean.

**Why Huber loss makes the predictions hug zero — by design.** The GBM regressor
uses `loss='huber'` (`alpha=0.9`), a robust loss that behaves like MSE near the
centre and like MAE in the tails. On a noisy target with no learnable signal,
the loss-minimising output is the conditional mean, and Huber's tail-robustness
**actively shrinks predictions toward that mean** to avoid overfitting to
individual noisy moves. The ~100× collapse in predicted std is exactly this
shrinkage working as intended — it is the model declining to fabricate
confident forecasts it cannot justify.

**Conclusion — a feature of mathematical honesty, not a bug.** The micro-percent
outputs are the system *correctly* reporting that day-ahead EUR/USD returns are
near-unpredictable. A model that emitted large, swinging return forecasts on
this target would be **overfitting noise and lying about its certainty**. The
practical implication (also noted in §4.5): `predicted_return_pct` should be read
as near-noise, not as a tradeable magnitude.

**Corroborating evidence — the Ch.11 train-vs-test capacity diagnostic
(2026-07-17, `results/train_vs_test_diagnostic.csv`).** The Practical
Methodology question "would more capacity (epochs/layers) help?" was answered
empirically against the served artifacts, so it does not need re-asking:

| model | variant | ROC-AUC train | ROC-AUC test | return R² train | return R² test |
|---|---|---|---|---|---|
| GBM | baseline | 0.6157 | 0.5220 | +0.023 | −0.002 |
| GBM | with_macro | 0.6166 | 0.5218 | +0.025 | −0.002 |
| LSTM | baseline | 0.5575 | 0.5046 | +0.024 | −0.007 |
| LSTM | with_macro | 0.5697 | 0.5302 | +0.031 | −0.032 |

The pattern is a **mild overfit above a Bayes floor at chance**: every model
already fits +0.04–0.09 AUC (and ~2–3% of return variance) of pure noise
in-sample that generalizes to exactly nothing out-of-sample. Three
independent facts close the question. (1) Larger capacity was already offered
and **rejected by validation**: all four GBM grid searches picked the
minimum-capacity corner (`n_estimators=100, max_depth=3, lr=0.01`), and LSTM
early stopping (patience=10 of a 100-epoch cap) halted at epoch 14/13 with
best weights from epoch ~4 — "more epochs" is mechanically a no-op. (2) The
Ch.11 tiny-dataset check passed: a fresh production-architecture LSTM drove
training loss to ~0 on a 5-row slice (direction 5/5, return MAE 0.037% vs
target scale ~0.37%), ruling out a training-loop/loss/scaling defect. (3) The
train-side gap means the Ch.11 remedy direction is *more regularization*, but
harder regularization can only converge train toward 0.50 too — it cannot
lift test above the floor. **Conclusion: scaling epochs/layers on the
direction/return models is evidence-refuted; only genuinely new information
(different features via the forward ledgers, or a different target as with
the volatility family §3.5) can move test performance.** (Note when comparing
splits: train MAE 0.39–0.41% vs test 0.30% reflects the train era's larger
target dispersion, not underfitting — R² is the comparable number.)

### 4.2.2 Probability calibration — evaluated, not adopted

A natural follow-up question: is the GBM classifier's `predict_proba` a genuinely
**calibrated** probability, or just a raw score `compute_consensus`'s
`CONFIDENCE_THRESHOLD=0.52` guard (§3.3) treats as one? This was tested directly —
wrapping the tuned classifier in `sklearn.calibration.CalibratedClassifierCV`
(Platt/sigmoid scaling, `TimeSeriesSplit` folds, same held-out test block) — and
**deliberately not adopted**, for the same reason §4.2.1's Huber shrinkage is a
feature rather than a defect: there is essentially no calibration gap to close on
this target.

| Predictor | Brier score (test) |
|---|---|
| Trivial **"always predict the base rate"** baseline (train-set mean, `ŷ = 0.4892`) | **0.25013** |
| Raw tuned GBM `predict_proba` (uncalibrated) | 0.25063 (**worse** than the trivial baseline) |
| + `CalibratedClassifierCV(method='sigmoid')` | 0.25025 |
| + `CalibratedClassifierCV(method='isotonic')` | 0.25038 |

The raw classifier's probabilities are, by Brier score, *indistinguishable from —
in fact microscopically worse than* — a constant that always predicts the
training-set base rate. Both calibration variants pull the score closer to that
same trivial baseline (which is literally what Platt/isotonic scaling does when a
model carries near-zero real signal), but the "improvement" is entirely calibration
correctly discounting a classifier that has nothing to calibrate. This is the
classification-side twin of §4.2.1's `MAE 0.2959% vs baseline 0.2958%` regression
finding — the same efficient-market conclusion, now confirmed via a second, unrelated
metric (Brier score vs baseline) and a second, unrelated model family test (binary
calibration vs regression shrinkage).

**Why it was not shipped despite the (tiny) Brier improvement:** sigmoid calibration
collapses the predicted-probability range from `[0.333, 0.672]` (raw) to
`[0.461, 0.513]` — under `CONFIDENCE_THRESHOLD=0.52`, this means the GBM head would
almost **never** cross the confidence guard again, silently changing
`compute_consensus`'s real-world behavior (near-permanent `MIXED / LOW CONFIDENCE`)
for a Brier gain of `0.00050 → 0.00012` (both already within noise of the trivial
baseline). Isotonic keeps a wider range (`[0.395, 0.610]`) but the same
near-baseline Brier finding holds. Adopting calibration here would trade a real,
visible behavior change for a statistically negligible accuracy-of-belief gain —
the same "honest shrinkage over confident-looking noise" principle §4.2.1 already
establishes, just evaluated and rejected on the classification side instead of
silently assumed. If a future retrain shows the raw classifier's ROC-AUC pull away
from chance, this decision should be revisited.

### 4.3 FRED feature — raw level was net-negative; the stationarized delta flips it positive

`results/2C_fred_ablation.csv` (methodology: notebook §2C's quick GBM classifier,
no grid search, same chronological 80/20 split, `WITHOUT` vs `WITH` the feature):

| Variant | Accuracy | ROC-AUC |
|---|---|---|
| WITHOUT the FRED feature | 0.5040 | 0.5071 |
| WITH `yield_differential` (raw level — **superseded**) | 0.5002 | 0.5050 |
| Δ (raw-level FRED effect) | **−0.0039** | **−0.0021** |
| WITH `yield_differential_delta` (diff(1) — **current production feature**) | 0.5069 | 0.5103 |
| Δ (delta-feature FRED effect) | **+0.0029** | **+0.0032** |

**Root cause and fix.** The raw level is a slow-trending, highly persistent
series (bond yields move in multi-month trends) — feeding it directly to a
next-day model is the same class of mistake the project already avoids
elsewhere: `log_return` is used instead of raw `close`, `bar_dynamics` instead
of raw `high`/`low`, because a next-day model should see *change*, not *level*.
Re-running the identical ablation with `yield_differential.diff(1)` (§2.4) in
place of the raw level flips the effect from net-negative to net-positive on
both metrics — small, consistent with everything else on this near-efficient
target (§4.2.1), but real and in the theory-predicted direction. The raw level
is still merged and displayed on the dashboard (`bar_used.yield_differential`,
unchanged) — only the **model-facing** feature changed.

**Follow-up not yet done:** the notebook's own §2C ablation cell computes this
same comparison dynamically (it imports `FEATURE_COLUMNS`/`merge_macro_features`/
`compute_features` from `src/features.py`, so it will pick up the new delta
feature automatically on its next execution) but its markdown narrative still
describes the old raw-level result and has not been re-run to confirm the
notebook environment reproduces the same numbers as this standalone check.

### 4.3.1 Three added macro features — kept provisionally, not statistically proven

> **⚠ Methodology superseded (2026-07-06).** The significance numbers in this
> section were computed on the **held-out test block** `[80%:100%]` — the same
> block used repeatedly for feature search, which is data-snooping (see
> *Production Methodology* at the top of this doc). The KEEP decisions have since
> been re-run on the clean **validation slice** `[70%:80%]` via `src/ablation.py`
> and judged against a **Bonferroni-corrected** bar
> (`results/feature_ablation_validation.csv`, `results/feature_hypothesis_log.csv`).
> The conclusion is unchanged — **all four features stay KEEP-provisional** — but
> the validation table there, not the test-block table below, is the governing
> record. The numbers below are retained only as the before/after audit trail.

Three more FRED macro features were added (`usd_index_return`,
`policy_rate_differential`, `inflation_differential` — see §2.4/§2.6). Each was
ablated individually (WITH vs WITHOUT, one feature toggled, all other columns and
rows held fixed) on the **euro-era 1999+ row set** — the addition of ECB/USD/HICP
series truncates the trainable history from the synthetic-1971 span to the real
euro era (§2.6), so this ablation runs on a different, shorter row set than §4.3's
1971+ numbers and the two are **not** directly comparable (the old 24-col/1971+
baseline row is preserved in `results/comparison_table.csv` as the permanent
before reference). Point-estimate deltas (`results/new_macro_ablation.csv`):

| Added feature | Δ accuracy | Δ ROC-AUC |
|---|---|---|
| `usd_index_return` | +0.0064 | +0.0073 |
| `policy_rate_differential` | +0.0047 | +0.0045 |
| `inflation_differential` | +0.0123 | +0.0041 |

All three point estimates are **positive**, so per the "keep only non-negative
features" rule they are kept — **but a proper significance test
(`results/new_macro_significance.csv`) shows none of the three is distinguishable
from noise:**

| Added feature | 95% bootstrap CI (Δacc) | frac(Δ>0) | McNemar p | Verdict |
|---|---|---|---|---|
| `usd_index_return` | [−0.0099, +0.0234] | 0.77 | 0.499 | KEEP — **provisional** |
| `policy_rate_differential` | [−0.0099, +0.0181] | 0.73 | 0.568 | KEEP — **provisional** |
| `inflation_differential` | [−0.0064, +0.0298] | 0.90 | 0.210 | KEEP — **provisional** |

Every 95% CI straddles 0 and every McNemar p-value is far above 0.05 (2000
paired bootstrap resamples of the 1,712-row test block; McNemar over the 2×2
correct/wrong flip table). So the features are retained on a nominally-positive
point estimate but carry **no proven edge** — exactly the efficient-market result
of §4.2. They should be revisited with a longer live test window before being
treated as real signal, and this is a live-money caveat, not an academic one.

**Out-of-scope flag (not acted on):** on this same 1999+ set, the existing
`yield_differential_delta` now shows a small *negative* ablation delta
(−0.0041 acc / −0.0045 auc) — the opposite sign from its +0.0029/+0.0032 on the
1971+ set in §4.3. Both are within noise; re-evaluating/removing the yield
feature is tracked as a separate `IMPROVEMENT_LOG.md` follow-up, deliberately not
changed in the same pass that added the three features (one change at a time).

### 4.3.2 COT positioning — genuinely new information, tested in both families, dropped

`src/cot_data.py` added weekly **net speculative positioning** (CFTC *Traders in
Financial Futures*, leveraged-funds long − short) for **EURO FX** and **ICE's USD
INDEX (DX)** futures as two candidate features (`cot_eur_zscore`,
`cot_usdindex_zscore`), z-scored over a trailing 3-year weekly window. Unlike the
FRED macro features this is a different *kind* of information — positioning /
sentiment, not price or rates — so it was worth a dedicated hypothesis in each
family. Raw contract counts are **not** used: open interest has grown structurally
for two decades, so raw levels are non-stationary; the trailing z-score removes
that drift.

**Look-ahead handling (the load-bearing part).** CFTC reports a Tuesday "as of"
date but only *publishes* the following Friday (~3-day structural lag), delayed
irregularly by holidays and government shutdowns. The daily join therefore uses
the **availability date**, never the as-of date. The live Socrata API exposes a
true system publish timestamp `:created_at`, but it is reliable only for rows
inserted after the dataset's **2022-09-13 bulk reload** (every earlier row carries
that one reload timestamp). `availability_date()` is hybrid: trust `:created_at`
when its lag over as-of is plausible (recent rows — captures real holiday/shutdown
delays exactly), else fall back to a conservative **`as_of + 10 days`** (a fixed
+3 would leak during exactly those gaps). Z-scores are computed on the native
weekly cadence and ffilled by availability date onto daily bars
(`add_cot_features`), so a bar can only ever see a reading already public — the
same as-of ffill discipline the monthly FRED series use. Guarded by
`test_add_cot_features_ffill_by_availability_date_no_lookahead` and
`test_cot_availability_date_trusts_recent_publish_but_buffers_bulk_reload`.

**Result — DROP in both families** (validation-only arbiter `[70%:80%]`, test block
never touched, each a single bundled Bonferroni hypothesis):

| Family | Point estimate | 95% CI | Test | Bar (Bonferroni) | Verdict |
|---|---|---|---|---|---|
| Direction/return (`src/ablation.py`) | Δacc **−0.0035**, Δauc −0.0099 | Δacc [−0.0234, +0.0175] | McNemar **p=0.83** | 0.05/6 = 0.0083 | **DROP** (`feature_hypothesis_log.csv` n=6) |
| Volatility 5-seed ensemble (`src/volatility.py`) | ΔMAE **−0.0067%**, ΔR² −0.025 | ΔMAE **[−0.0091, −0.0045]** (entirely < 0) | frac(dMAE>0)=0.000 | 0.05/7 = 0.0071 | **DROP** (`volatility_hypothesis_log.csv` n=7) |

In direction/return COT is indistinguishable from noise; in volatility it is
reliably **worse** than the validated base ensemble (the CI sits wholly below
zero, and the challenger's best of 5 seeds still lost to the base mean, so it is
not training-noise). Consistent with COT being documented for multi-week reversals
rather than next-day moves, and with the efficient-market result of §4.2. The
module, both ablation hooks, and the tests are kept so the finding is reproducible,
but COT is **not** in `FEATURE_COLUMNS`, **not** served, and triggered no retrain.
`cot_staleness_days()` exists as a module diagnostic for a future forward test; it
is wired into the live response only if COT ever ships.

**Weekly-horizon side-check (also DROP, own family).** Because COT is documented
for multi-week reversals rather than next-day moves, a separate exploratory check
(`src/cot_weekly_check.py`) asked whether the z-scores carry a **weekly** edge —
daily close resampled to W-TUE bars (CFTC's Tuesday cadence), target = forward
weekly log return, predictors joined by availability date (`merge_asof` backward,
same no-look-ahead discipline, guarded by
`test_weekly_cot_asof_join_backward_no_lookahead`). This is a **different target
horizon**, so it is *not* comparable to the daily direction/return or volatility
bars and is logged in its **own** family file `results/cot_weekly_hypothesis_log.csv`
(alpha=0.05 first test) — `feature_hypothesis_log.csv` and
`volatility_hypothesis_log.csv` are untouched. One pre-registered battery (992
analysis weeks, 99 validation): Spearman rho on validation was **+0.061** for both
z-scores with 95% CIs straddling 0, and a logistic direction model collapsed
exactly to the majority baseline (Δacc +0.000, McNemar p=1.0) → **DROP**. Honest
power caveat: ~100 validation weeks only detect |rho| ≳ 0.2, so this is weak
evidence of absence, and it remains **research-only** — no model, no variant, no
serving change regardless of outcome.

A **second** weekly hypothesis (`run_extremes()`, same family, so the bar tightens
to alpha=0.05/2=**0.025**) tested positioning **extremes as a contrarian signal**
rather than linear correlation: pre-registered crowded-long = z>+1.0 /
crowded-short = z<−1.0 (~16% tails, a priori), predicting crowded-long → negative
and crowded-short → positive forward weekly returns, with the PRIMARY test the
bootstrap 95% CI of `spread = mean(fwd|z>+1) − mean(fwd|z<−1)` (KEEP only if
entirely below 0 with the expected sign) and exact binomial sign tests as
context. **Outcome: INCONCLUSIVE / underpowered** — the validation window had
one-sided positioning (`cot_eur_zscore` crowded-long 30 weeks / crowded-short 0;
`cot_usdindex_zscore` crowded-short 27 / crowded-long 2), so neither z-score had
≥5 two-sided extreme weeks for a stable CI. Per the pre-registration the cutoff
was **not** loosened to manufacture rows; the thin tails were reported plainly
(`cot_weekly_hypothesis_log.csv` row #2, `cleared_bar=False`). Still research-only.

**Fibonacci retracement + Williams fractals (direction/return hypothesis #7, DROP).**
`src/fibonacci_fractals.py` adds swing-structure *geometry* from OHLC alone — a
different kind of information from momentum or the macro feeds. A Williams 5-bar
fractal (a strict extremum of high/low`[i-2:i+3]`) is the primitive, and the
**confirmation lag is the load-bearing look-ahead surface**: a fractal at bar i
reaches two bars into the future, so it is only knowable at bar i+2. The reveal
walk exposes the fractal at index t−2 exactly at step t, so a fractal forming at
i is invisible on bars i, i+1 and first usable on i+2 — asserted directly by
`test_fractal_confirmation_lag_no_lookahead` (mirroring the FRED/COT guards).
Every feature is neutral 0 / NaN-safe until a confirmed structure exists, so the
modeled row set is unchanged. The bundle (`fractal_breakout_up`,
`fractal_breakout_down`, `dist_to_nearest_fib_pct` — three views of one swing
fact, one Bonferroni slot) was ADD-tested exactly like the macro/COT/FOMC blocks:

| Family | Point estimate | 95% CI | Test | Bar (Bonferroni) | Verdict |
|---|---|---|---|---|---|
| Direction/return (`src/ablation.py fib`) | Δacc **+0.0035**, Δauc +0.0047 | Δacc [−0.0187, +0.0257] | McNemar **p=0.84** | 0.05/7 = 0.0071 | **DROP** (`feature_hypothesis_log.csv` n=7) |

A **hypothesis #8** (`dist_to_nearest_fib_extension_pct` — a 3-point
extension/projection off a confirmed A→B→C swing, chronology-guarded) was **built
and unit-tested but deliberately not spent**: a pre-registered contingency (in the
module docstring, `feature_hypothesis_log.csv` #7 notes, and IMPROVEMENT_LOG.md)
runs #8 only if #7 clears its bar. #7 is DROP, so #8 stays dormant — a more
discretionary feature is not worth a slot when the simpler 2-point version already
failed. Nothing is in `FEATURE_COLUMNS`, served, or retrained; the finding is
reproducible via `python -m src.ablation fib`.

**VIX regime features (direction/return hypothesis #8, DROP).**
`src/vix_features.py` adds broad equity risk sentiment (the "fear gauge") — a
different *kind* of information again. The raw VIXCLS **level** rides the shared
FRED framework (`config.json macro.features.vix` → `macro_data._combine_vix`,
cache `results/vix.csv`); the two stationary transforms live downstream, mirroring
how `usd_index_return` is derived from the merged `usd_index` level:
`vix_zscore` (trailing 756/252-day rolling z-score — VIX has real multi-year
regime drift, so a level is non-stationary, the COT z-score treatment) and
`vix_change_pct` (day-over-day shock). **STEP 0 verified, not assumed:** the FX D1
bar closes ~17:00 ET, VIXCLS is the ~16:15 ET CBOE close, and a live FRED probe
(2026-07-26) showed VIXCLS publishes with a business-day lag (Friday's print
absent two days later) — so a print dated D is treated as usable only on **D+1
business day** (conservative D-1 rule), and the z-score is computed on the
**native business-day cadence** (not the ffill-duplicated FX-daily series, whose
Sunday bars would corrupt the window), then as-of ffilled. Guarded by
`test_vix_availability_is_one_business_day_after_the_print_no_lookahead`,
`test_vix_value_never_usable_on_the_bar_it_would_leak_into`, and the graceful-
degradation / feature-exclusion tests.

| Family | Point estimate | 95% CI | Test | Bar (Bonferroni) | Verdict |
|---|---|---|---|---|---|
| Direction/return (`src/ablation.py vix`) | Δacc **−0.0117**, Δauc +0.0040 | Δacc [−0.0327, +0.0105] | McNemar **p=0.36** | 0.05/8 = 0.00625 | **DROP** (`feature_hypothesis_log.csv` n=8) |

The point estimate is actively negative — consistent with equity fear being a
*volatility* event, not a next-day *directional* EUR/USD signal. The raw VIX level
is deliberately kept OUT of `MACRO_MERGE_COLUMNS`, so nothing enters the served
model frame and predictions are byte-identical; reproducible via
`python -m src.ablation vix`.

### 4.4 Known defects — fixed in this branch

1. **Macro cache truncation (data-loss).** `fetch_yield_differential` previously
   wrote the live fetch over the cache unconditionally. Because the request
   window is derived from the (often short) live price index, a single run could
   shrink `results/yield_differential.csv` from ~14,600 rows (1971→) to ~200.
   **Fixed** by merging onto the existing cache + de-dup before writing
   (`src/macro_data.py:80-101`), plus restoring the `DATE` index name.
2. **Section 20 test cells (false failures).** Notebook cells 20a/20b ran
   `pytest` via `subprocess` **without `cwd`**; from `notebooks/` pytest
   collected 0 tests (exit 5) / hit a usage error (exit 4) and raised a
   misleading "tests failed". **Fixed** by passing `cwd=os.path.abspath('..')`.
   The suite itself is green (**19 passed**).

### 4.5 Live-edge / architecture risks (open, by design)

| Risk | Where | Mitigation in place | Residual exposure |
|---|---|---|---|
| **Stale macro at live edge** | `merge_macro_features` ffill | weekend/holiday gaps inherit last differential | If the live price index is newer than the newest FRED obs, the latest bars carry a *stale* differential (ffill cannot interpolate the future). Since §4.3, the model actually consumes `yield_differential_delta` (diff of the ffilled level) — a run of stale/repeated level values now correctly diffs to `0` ("no new information"), which is arguably a more honest signal than a stale level pretending to be current |
| **LSTM direction at chance** | model quality | low-confidence consensus guard (§3.3) downgrades a coin-flip agreement to `MIXED / LOW CONFIDENCE` | A near-chance head still contributes when averaged confidence ≥ 0.52 |
| **History CSV legacy schema** | `results/eurusd_features.csv` | `load_history` selects only OHLCV cols (`src/features.py:147-148`) | The CSV's precomputed feature columns are an *older* schema and are ignored — only raw OHLCV is consumed and features are recomputed fresh |

### 4.5.1 Resolved Architectural Risks

These risks, documented in earlier revisions, have been **eliminated** by the
unified-pipeline refactor:

| Former risk | Resolution | Where |
|---|---|---|
| **SMA_200 warm-up hard-fail** — a thin live fetch raised `RuntimeError` | `_resolve_latest_window` now **back-fills** the missing preceding rows from the bundled history (concat + de-dup by index) and proceeds; the data source is tagged `…+history_backfill` | `src/inference.py` `_resolve_latest_window` |
| **GBM vs LSTM unit asymmetry** — GBM trained on fractions (×100 at inference), LSTM on percent | `target_return` is produced **natively in percent** by `src/features.py`; **both** heads train on and output percent — no `*100` anywhere | `src/features.py`, `src/inference.py` |
| **Dual scalers / dual train fractions** — separate `scaler_gb` (80%) and `scaler_lstm` (70%); PCA fit on 70% but GBM split at 80% | **One** `global_scaler` **and** the PCA are both fit on the **unified 0–80%** block; serialized as `global_scaler.pkl` | `_train_pipeline.py`, `config.json` |

### 4.6 Test inventory

| File | Category | Coverage |
|---|---|---|
| `tests/test_smoke.py` | Smoke | All 7 production artifacts (incl. the single `global_scaler.pkl`) + `eurusd_features.csv` + `config.json` + `.env.example` exist |
| `tests/test_unit.py` | Unit (31 tests) | feature engineering, `build_live_features` (no mocks), **lag-PCA no-leakage**, **macro merge no-look-ahead** (both the raw level and the derived `yield_differential_delta`), FRED fallback chain (4 tests), live-data fallback chain (3 tests), consensus agree/disagree, edge cases, **H1 ensemble** (no-look-ahead, inference-sample forming-day drop, `compute_h1_consensus` majority/unanimous — 5 tests), prediction-log tracking incl. `worst_mistakes`, **backtest cost-on-flip-only logic** (2 tests) |
| `tests/test_integration.py` | Integration | `POST /api/predict` contract (schema, bounds `0≤conf≤1`, direction ∈ {UP,DOWN}, consensus presence), static UI route |

### 4.7 Backtest — does the direction edge survive transaction costs?

`src/backtest.py::simulate_strategy` runs a minimal daily long/short strategy
driven by the GBM direction signal, scored on the **same held-out test block**
against the **actually realised** `target_return` — no position sizing, no
compounding (returns are simply summed, not geometrically chained), no slippage
beyond a flat per-trade spread cost. The point is a sanity check on whether the
model's apparent statistical edge (§4.2) would survive contact with a real
market, not a trading system. Cost is charged only when the signal's sign
actually **changes** (a flat→long/short entry, or a long↔short flip) — holding
an unchanged position overnight incurs no fresh spread — via `EURUSD_PIP_TO_PCT`
(`0.0001 / 1.10 * 100 ≈ 0.0091%`, a representative EUR/USD level converting pips
to a round-trip percent cost).

`_train_pipeline.py` runs this immediately after GBM test evaluation and saves
`results/backtest_transaction_costs.csv`. On the current production artifacts
(test block: 3,103 days, ≈12 years):

| Scenario | Round-trip cost | Trades | Hit rate | Gross return (total) | Net return (total) |
|---|---|---|---|---|---|
| Gross (no costs) | 0 pips | 1,337 | 0.5021 | **+29.08%** | +29.08% |
| Realistic (tight) | 1 pip | 1,337 | 0.5021 | +29.08% | **+16.93%** |
| Realistic (typical retail) | 2 pips | 1,337 | 0.5021 | +29.08% | **+4.77%** |

**Reading this honestly:** the *gross* (frictionless, unrealistic) edge is already
razor-thin — a 0.5021 hit rate over ≈12 years compounds to only +29% *simple-summed*
return, not even geometric growth. A realistic 1-pip spread more than **halves**
it; a typical retail 2-pip spread leaves **+4.77% over 12 years** (≈0.4%/year) —
economically negligible once financing costs, occasional slippage beyond the quoted
spread, and any position-sizing risk are considered. This is the third independent
confirmation of the same efficient-market conclusion in this document (§4.2.1's
"predict the mean" regression finding, §4.2.2's Brier-vs-baseline classification
finding, and now a P&L-denominated backtest) — each using an unrelated metric on the
same near-chance direction signal, all converging on the same answer: whatever
weak edge exists is not *tradeable*, not a modeling defect to "fix".

### 5.1 Trained model artifacts — `models/`

**Production (loaded by `PredictionService`):**

| File | Type | Produced by |
|---|---|---|
| `lag_scaler.pkl` | joblib / StandardScaler (lag block, pre-PCA) | `_train_pipeline.py` |
| `lag_pca.pkl` | joblib / PCA | `_train_pipeline.py` |
| `global_scaler.pkl` | joblib / StandardScaler (**single, shared by both models**) | `_train_pipeline.py` |
| `best_gbm_eurusd.pkl` | joblib / `xgb.XGBClassifier` | `_train_pipeline.py` |
| `best_gbm_regressor_eurusd.pkl` | joblib / `xgb.XGBRegressor` (`reg:pseudohubererror`) | `_train_pipeline.py` |
| `lstm_multitask_eurusd.keras` | Keras native format | `_train_pipeline.py` |
| `lstm_time_steps.pkl` | joblib / int (20) | `_train_pipeline.py` |

**Auxiliary H1→Daily ensemble (§3.4, loaded independently — gated by `h1_ready`):**

| File | Type | Produced by |
|---|---|---|
| `h1_xgb_regressor.pkl` | joblib / `xgb.XGBRegressor` | `_train_pipeline.py` §13 |
| `h1_rf_regressor.pkl` | joblib / `RandomForestRegressor` | `_train_pipeline.py` §13 |
| `h1_svm_regressor.pkl` | joblib / `SVR` (RBF) | `_train_pipeline.py` §13 |
| `h1_feature_scaler.pkl` | joblib / StandardScaler (flat features) | `_train_pipeline.py` §13 |
| `h1_lstm_scaler.pkl` | joblib / StandardScaler (24h tensor) | `_train_pipeline.py` §13 |
| `h1_feature_columns.pkl` | joblib / list[str] (`FLAT_FEATURE_COLUMNS` order) | `_train_pipeline.py` §13 |
| `h1_lstm_config.pkl` | joblib / dict (`hours`, `seq_features`) | `_train_pipeline.py` §13 |
| `h1_lstm.keras` | Keras native format | `_train_pipeline.py` §13 |

**Exploratory (notebook baselines, not loaded in production):**
`exploratory_gbm_baseline.pkl`, `exploratory_gbm_scaler.pkl`,
`randomforest_tuned.pkl`, `xgboost_tuned.pkl`, `scaler.pkl`.

### 5.2 MLflow experiment tracking

- **Experiment name:** `EURUSD_Prediction` (`_train_pipeline.py:40`).
- **Runs:** `GBM_dual_pipeline`, `MultiTask_LSTM` (params + metrics + logged models).
- **Store:** file store under **`mlruns/`** (experiment id `1`; logged model
  blobs under `mlruns/1/models/m-*`) plus a SQLite DB **`mlflow.db`** at repo
  root. View with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (or the
  default `./mlruns` file store).
- **Note:** the notebook training cells (§19) persist `models/` artifacts but do
  **not** themselves wrap MLflow runs — MLflow logging lives in
  `_train_pipeline.py`. The two share identical feature/PCA/model code paths.

### 5.3 Diagnostic exports — `results/`

PNGs (`01_price_sma`, `02_learning_curves`, `03_tscv_folds`,
`04_confusion_matrix`, `05_residual_analysis`, `06_acf_pacf`,
`09_lstm_learning_curve`, `10_lstm_evaluation`, `GBM_*`, `2C_fred_*`) and CSVs
(`comparison_table.csv`, `2C_fred_ablation.csv`, `2C_fred_table.csv`,
`eurusd_features.csv` = bundled OHLCV history, `yield_differential.csv` = FRED
cache). Post-defense methodology exports (see *Production Methodology*):
`feature_ablation_validation.csv` = validation-slice KEEP/DROP re-run,
`feature_hypothesis_log.csv` = running Bonferroni family count,
`paper_trading_log.csv` = simulated forward P&L ledger.

### 5.4 Prediction output structure & UI/API routing

`service.predict()` returns a single dict (dual-variant shape):

```jsonc
{
  "as_of_date": "YYYY-MM-DD",
  "forecasting_date": "YYYY-MM-DD",      // as_of + 1 trading session (t+1)
  "data_source": "MT5|yfinance|history_fallback|<src>+history_backfill",
  "bar_used": { "date","open","high","low","close","tick_volume",
                "yield_differential","usd_index","policy_rate_differential",
                "inflation_differential","macro_source","macro_sources" },
  "baseline": {                          // price-only variant (23 cols, no FRED)
    "gbm":  { "direction":"UP|DOWN","confidence":0..1,"predicted_return_pct":float },  // percent
    "lstm": { "direction":"UP|DOWN","confidence":0..1,"predicted_return_pct":float },  // percent
    "consensus": { "direction":"UP|DOWN|MIXED / LOW CONFIDENCE","agreement":bool,"confidence","predicted_return_pct" }
  },
  "with_macro": { /* same committee shape — the experimental 27-col variant */ },
  "variant_agreement": true,             // bool when both consensuses exist; null when either is degraded
  "h1": { /* auxiliary intraday ensemble, shared by both variants */ }
}
```

A degraded variant replaces its block's committee with an `"error"` note; the
other variant keeps serving (`baseline_ready` / `macro_ready` gates).

Routing:

| Layer | Entry | Rendering |
|---|---|---|
| **FastAPI** `api.py` | `POST /api/predict` | Returns the raw dict as JSON; `503` if `models_ready` is false, `400` on pipeline error. |
| **Static UI** `static/index.html` | `fetch('/api/predict', {method:'POST'})` | Mounted at `/` by `api.py`; client-side JS renders the prediction, plus the retrain button and a link to `/history`. |
| **Prediction history** `api.py` | `GET /history` | `src/tracking.build_history_html` — every logged forecast scored against the realised close; cross-links `/paper-trading`. |
| **Forward paper-trading** `api.py` | `GET /paper-trading` (HTML), `GET /api/paper-trading` (JSON) | `src/paper_trading` — simulated cost-net ledger + scorecard (cum. net pips/%, win rate, Sharpe-like, max drawdown). Simulated only, no orders. |

### 5.5 Containerization note

`Dockerfile` builds the **FastAPI app** (`uvicorn api:app` on port 8000), strips
`MetaTrader5` (Windows-only) from requirements, and bakes in `api.py` + `src/` +
`static/` + `models/` + `results/eurusd_features.csv`. With no MT5 terminal the
container serves live prices from **yfinance** (falling back to the bundled
history), with FRED still reachable at runtime.

---

## Appendix A — Failure-Mode Quick Reference

| Symptom | Most likely cause | File |
|---|---|---|
| `models_ready == False` at startup | a `models/` artifact missing/corrupt (incl. `global_scaler.pkl`) | `src/inference.py` `__init__` |
| Data source tagged `…+history_backfill` | live fetch was thin; preceding rows were back-filled from history to satisfy the SMA_200/lag warm-up — **no longer a hard-fail** (replaces the former `RuntimeError`) | `src/inference.py` `_resolve_latest_window` |
| `yield_differential` looks frozen on newest bars | live price index newer than FRED cache; ffill can't see the future | `src/features.py:140` |
| Consensus shows `MIXED / LOW CONFIDENCE` | both heads agree but averaged confidence < 0.52 (low-confidence guard) | `src/inference.py` `compute_consensus` |
| Notebook §20 "tests failed" | (fixed) subprocess `cwd` not set to repo root | notebook cells 20a/20b |
| `yield_differential.csv` shrank dramatically | (fixed) cache overwrite instead of merge | `src/macro_data.py:80-101` |
| `h1_ready == False` / response carries `h1_error` instead of `h1` | one of the 8 H1 artifacts missing/corrupt, **or** a feature-set change (e.g. new flat/seq columns) not yet followed by a retrain — the saved scalers expect the old column count and raise a shape mismatch | `src/inference.py` `_predict_h1` (§3.4); never fails the daily prediction |
| H1 `feature_importances_`/scaler shape error right after editing `src/h1_features.py` | `FLAT_FEATURE_COLUMNS`/`SEQ_FEATURE_COLUMNS` changed but `models/h1_*` weren't regenerated | run `_train_pipeline.py` (§13 refreshes the H1 cache and retrains all four) |
