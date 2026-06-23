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
| Standalone trainer | `_train_pipeline.py` | Headless reproduction of the notebook's training sections; writes `models/` + MLflow. |
| Inference core | `src/inference.py` | `PredictionService` — loads artifacts once, serves t+1 predictions. **Shared by both frontends.** |
| Feature engineering | `src/features.py` | The 24-column `FEATURE_COLUMNS` contract, PCA on lag block, macro merge. |
| Live market data | `src/live_data.py` | MT5 → yfinance fallback chain for OHLCV. |
| Macro data | `src/macro_data.py` | FRED API → FRED public CSV → on-disk cache fallback chain. |
| Web app (single entry point) | `api.py` | FastAPI server: serves `static/index.html` at `/`, `POST /api/predict`, `GET /history`, `POST /api/retrain`. Port 8000. |
| Config | `config.json` | All hyperparameters + paths. Single source of truth. |
| Artifacts | `models/`, `results/`, `mlruns/`, `mlflow.db` | Serialized models, diagnostics, experiment tracking. |

The web layer is a **single entry point** (`api.py`) on top of one shared
`PredictionService`. All prediction logic lives in `src/` — `api.py` is only the
HTTP/dashboard layer.

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

`src/features.py:123-142`: left-joins `yield_differential` onto the OHLCV index.
The OHLCV index is localized/converted to UTC for the join, the differential is
`ffill()`-ed across weekend FX bars and bond holidays, then the **original index
is restored**. Zero look-ahead regardless of calendar offset.

### 2.4 Feature transformation — `compute_features`

`src/features.py:51-105` produces the canonical **24 `FEATURE_COLUMNS`**
(`src/features.py:26-32`):

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
| Exogenous macro | `yield_differential` | passthrough (must be pre-merged) |

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

---

## 3. Prediction Logic & Model Architecture

### 3.1 GBM dual pipeline (tree ensemble)

`_predict_gbm` (`src/inference.py:129-146`) consumes **one flat PCA-reduced row**
(`window.iloc[-1]`), scales it with the single `global_scaler`, and runs two heads:

- **Classifier** `best_gbm_eurusd.pkl` — `GradientBoostingClassifier`, tuned for
  `roc_auc`. Emits `predict_proba` → `direction` (UP/DOWN) + `confidence`.
- **Regressor** `best_gbm_regressor_eurusd.pkl` — `GradientBoostingRegressor`,
  `loss='huber'`, `alpha=0.9`, tuned for MAE. Emits `predicted_return`
  **natively in percent** — the regressor is now trained on the percent target
  produced by `src/features.py`, so there is **no `*100` rescaling** at inference.

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

### 4.3 FRED feature — ablation shows net-negative effect

`results/2C_fred_ablation.csv`:

| Variant | Accuracy | ROC-AUC |
|---|---|---|
| WITHOUT `yield_differential` | 0.5040 | 0.5071 |
| WITH `yield_differential` | 0.5002 | 0.5050 |
| **Δ (FRED effect)** | **−0.0039** | **−0.0021** |

The macro feature is fully wired through training and inference, but on this
target it **slightly hurts** test metrics. It is retained for architectural/
academic completeness; its production value is **not yet demonstrated**.

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
| **Stale macro at live edge** | `merge_macro_features` ffill | weekend/holiday gaps inherit last differential | If the live price index is newer than the newest FRED obs, the latest bars carry a *stale* differential (ffill cannot interpolate the future) |
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
| `tests/test_unit.py` | Unit (18 tests) | feature engineering, `build_live_features` (no mocks), **lag-PCA no-leakage**, **macro merge no-look-ahead**, FRED fallback chain (4 tests), live-data fallback chain (3 tests), consensus agree/disagree, edge cases |
| `tests/test_integration.py` | Integration | `POST /api/predict` contract (schema, bounds `0≤conf≤1`, direction ∈ {UP,DOWN}, consensus presence), static UI route |

---

## 5. Output & Artifact Locations

### 5.1 Trained model artifacts — `models/`

**Production (loaded by `PredictionService`):**

| File | Type | Produced by |
|---|---|---|
| `lag_scaler.pkl` | joblib / StandardScaler (lag block, pre-PCA) | `_train_pipeline.py` |
| `lag_pca.pkl` | joblib / PCA | `_train_pipeline.py` |
| `global_scaler.pkl` | joblib / StandardScaler (**single, shared by both models**) | `_train_pipeline.py` |
| `best_gbm_eurusd.pkl` | joblib / GBClassifier | `_train_pipeline.py` |
| `best_gbm_regressor_eurusd.pkl` | joblib / GBRegressor | `_train_pipeline.py` |
| `lstm_multitask_eurusd.keras` | Keras native format | `_train_pipeline.py` |
| `lstm_time_steps.pkl` | joblib / int (20) | `_train_pipeline.py` |

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
cache).

### 5.4 Prediction output structure & UI/API routing

`service.predict()` returns a single dict:

```jsonc
{
  "as_of_date": "YYYY-MM-DD",
  "forecasting_date": "YYYY-MM-DD",      // as_of + 1 day (t+1)
  "data_source": "MT5|yfinance|history_fallback|<src>+history_backfill",
  "bar_used": { "date","open","high","low","close","tick_volume",
                "yield_differential","macro_source" },
  "gbm":  { "direction":"UP|DOWN","confidence":0..1,"predicted_return_pct":float },  // percent
  "lstm": { "direction":"UP|DOWN","confidence":0..1,"predicted_return_pct":float },  // percent
  "consensus": { "direction":"UP|DOWN|MIXED / LOW CONFIDENCE","agreement":bool,"confidence","predicted_return_pct" }
}
```

Routing:

| Layer | Entry | Rendering |
|---|---|---|
| **FastAPI** `api.py` | `POST /api/predict` | Returns the raw dict as JSON; `503` if `models_ready` is false, `400` on pipeline error. |
| **Static UI** `static/index.html` | `fetch('/api/predict', {method:'POST'})` | Mounted at `/` by `api.py`; client-side JS renders the prediction, plus the retrain button and a link to `/history`. |

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
