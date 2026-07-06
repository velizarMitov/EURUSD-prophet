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
| Auxiliary intraday feature engineering | `src/h1_features.py` | H1→Daily feature module for the auxiliary ensemble (§3.4) — flattened daily stats + 24h tensor, both including `Trend_vs_SMA504`/`RSI_24`. Independent of `src/features.py`. |
| Live market data | `src/live_data.py` | MT5 → yfinance fallback chain for OHLCV (also `fetch_h1_market_data` for the H1 stream). |
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

**Data & features — `src/h1_features.py`:** `load_h1_frame` reads (or fetches, via
`src/live_data.py::fetch_h1_market_data`, MT5 → yfinance → cache) a UTC-indexed H1
OHLCV stream. Two aligned representations are built from it, sharing one daily
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
independently of the daily committee's `compute_consensus`: direction is the
majority sign, `confidence` is the **fraction of models agreeing** (a genuine
[0.5, 1.0] agreement measure — **not** a calibrated probability, unlike the daily
consensus's `confidence`), and `predicted_return_pct` is the mean across all four.
`agreement=True` only on a unanimous sign.

```jsonc
"h1": {
  "as_of_date": "YYYY-MM-DD",
  "predictions": {
    "h1_xgboost":       { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_random_forest":  { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_svm":            { "direction": "UP|DOWN", "predicted_return_pct": float },
    "h1_lstm":           { "direction": "UP|DOWN", "predicted_return_pct": float }
  },
  "consensus": { "direction": "UP|DOWN", "agreement": bool, "confidence": 0.5-1.0, "predicted_return_pct": float, "n_models": 4 }
}
// or, on any failure: "h1_error": "<message>"
```

`src/tracking.py::log_prediction` also logs the H1 consensus (`h1_direction`,
`h1_return_pct`, `h1_agreement`) alongside the daily forecast, and
`build_history_html` scores it against the same realised close in its own
"H1 ensemble" column with its own hit-rate — independent of the daily
committee's hit-rate. The static UI (`static/index.html`) renders the H1 block as
a separate "Auxiliary Intraday Ensemble" section below the daily cards.

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
| `h1_ready == False` / response carries `h1_error` instead of `h1` | one of the 8 H1 artifacts missing/corrupt, **or** a feature-set change (e.g. new flat/seq columns) not yet followed by a retrain — the saved scalers expect the old column count and raise a shape mismatch | `src/inference.py` `_predict_h1` (§3.4); never fails the daily prediction |
| H1 `feature_importances_`/scaler shape error right after editing `src/h1_features.py` | `FLAT_FEATURE_COLUMNS`/`SEQ_FEATURE_COLUMNS` changed but `models/h1_*` weren't regenerated | run `_train_pipeline.py` (§13 refreshes the H1 cache and retrains all four) |
