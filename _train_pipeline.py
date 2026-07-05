"""
Standalone training run mirroring notebooks/01_data_preparation.ipynb
Sections 3/4 (features), 14 (PCA on lag features), 15 (Multi-Task LSTM) and
16 (GBM dual pipeline), sourcing raw OHLCV from results/eurusd_features.csv
since no live MT5 terminal is available in this environment, and reading
all hyperparameters from config.json. Produces real artifacts under models/.

Unified-pipeline refactor:
  * ONE chronological split shared by every model (train_fraction / val_fraction
    from config.json) -- GBM trains on [0:80%], the LSTM on [0:70%] with
    [70%:80%] as its early-stopping validation set, and BOTH evaluate on the
    identical held-out [80%:100%] test block.
  * ONE global StandardScaler (global_scaler.pkl) fit exclusively on the 0-80%
    block, replacing the former separate scaler_gb / scaler_lstm.
  * PCA on the lag block fit strictly on the same 0-80% slice (resolves the
    prior coupling where PCA saw 70% but GBM split at 80%).
  * target_return arrives natively in PERCENT units from src/features.py, so
    there is no longer any *100 rescaling anywhere in training or inference.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
)

from src.features import (
    add_advanced_features, merge_macro_features, TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN,
    FEATURE_COLUMNS, LAG_COLUMNS, fit_lag_pca, apply_lag_pca, model_input_columns,
)
from src.macro_data import fetch_yield_differential

load_dotenv('.env')

with open('config.json') as f:
    CONFIG = json.load(f)

RANDOM_STATE = CONFIG['random_state']
np.random.seed(RANDOM_STATE)

_XGB_DEVICE = 'cpu'
try:
    _probe = xgb.XGBClassifier(device='cuda', n_estimators=1, verbosity=0)
    _probe.fit(np.array([[1.0, 0.0]]), [0])
    _XGB_DEVICE = 'cuda'
    print("XGBoost GPU (CUDA): available — tree training will run on RTX 4070.")
except Exception as _e:
    print(f"XGBoost GPU: not available ({_e}). Falling back to CPU.")

mlflow.set_experiment("EURUSD_Prediction")


def _safe_log_model(log_fn, model, artifact_path):
    """MLflow model-artifact logging is observability only -- the real serving
    artifacts are written via joblib/keras.save below. A tracking-layer failure
    (e.g. the skops 'untrusted types' guard rejecting XGBoost boosters on some
    mlflow versions) must never abort training, so swallow it with a warning."""
    try:
        log_fn(model, artifact_path=artifact_path)
    except Exception as _e:
        print(f"[mlflow] skipped logging '{artifact_path}': {_e}")


print("=== 1. Loading historical OHLCV ===")
raw_df = pd.read_csv(CONFIG['data']['history_csv_path'], index_col='time', parse_dates=True)
raw_df = raw_df[['open', 'high', 'low', 'close', 'tick_volume']]
print(f"Loaded {len(raw_df):,} bars ({raw_df.index[0].date()} -> {raw_df.index[-1].date()})")

print("\n=== 1B. Macro Feature Ingestion (FRED: US 10Y - DE 10Y Yield Differential) ===")
macro_cfg = CONFIG.get('macro', {})
macro_df, macro_source = fetch_yield_differential(
    raw_df.index.min(), raw_df.index.max(),
    series_ids=macro_cfg.get('fred_series'),
    cache_path=macro_cfg.get('cache_path', 'results/yield_differential.csv'),
)
if macro_df is not None:
    raw_df = merge_macro_features(raw_df, macro_df)
    print(f"Merged yield_differential via {macro_source}: {len(macro_df):,} macro observations "
          f"({macro_df.index[0].date()} -> {macro_df.index[-1].date()})")
else:
    raw_df = raw_df.assign(yield_differential=0.0)
    macro_source = "unavailable"
    print("WARNING: no live or cached FRED data reachable -- yield_differential defaulted to 0.0")

print("\n=== 2. Feature Engineering (Multi-Task targets, target_return in PERCENT) ===")
basic_advanced_df = add_advanced_features(raw_df)
assert list(basic_advanced_df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
print(f"Shape: {basic_advanced_df.shape}")

# ---------------------------------------------------------------------------
# Section 14 — Unified chronological splits + PCA on autoregressive lags
# ---------------------------------------------------------------------------
print("\n=== 3. Unified Chronological Splits & PCA on Lag Features ===")
n_total = len(basic_advanced_df)
train_fraction = CONFIG['split']['train_fraction']    # 0.80
val_fraction = CONFIG['split']['val_fraction']        # 0.10

# One set of chronological boundaries shared by EVERY model, so the PCA, the
# global scaler, the GBM and the LSTM can never drift apart. The held-out TEST
# block [train_end:] is identical for both models and is never seen by the PCA
# fit or the global-scaler fit below -- that is the leakage boundary that matters.
train_end = int(n_total * train_fraction)                        # 80% -> test starts here (GBM train = [0:train_end])
lstm_train_end = int(n_total * (train_fraction - val_fraction))  # 70% -> LSTM val starts here
print(f"n={n_total}  LSTM-train [0:{lstm_train_end}]  LSTM-val [{lstm_train_end}:{train_end}]  "
      f"TEST [{train_end}:{n_total}]   (GBM-train = [0:{train_end}])")

# PCA on the lag block, fit STRICTLY on the unified 0-80% train slice.
lag_scaler, lag_pca = fit_lag_pca(
    basic_advanced_df.iloc[:train_end],
    lag_columns=LAG_COLUMNS,
    variance_threshold=CONFIG['pca']['variance_threshold']
)
print(f"Lag columns in: {len(LAG_COLUMNS)}  ->  PCA components out: {lag_pca.n_components_}")
print(f"Cumulative variance explained: {lag_pca.explained_variance_ratio_.sum():.4f}")

basic_advanced_df_reduced = apply_lag_pca(basic_advanced_df, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
# Use the canonical FEATURE_COLUMNS constant (not basic_advanced_df.columns, which also
# carries tick_volume/target columns as passthrough) so training and live inference can
# never silently diverge on which columns actually feed the models.
MODEL_INPUT_COLUMNS = model_input_columns(lag_pca, base_columns=FEATURE_COLUMNS, lag_columns=LAG_COLUMNS)
print(f"Model input columns ({len(MODEL_INPUT_COLUMNS)}): {MODEL_INPUT_COLUMNS}")

# ---------------------------------------------------------------------------
# Unified global StandardScaler -- ONE scaler for BOTH models. Fit exclusively
# on the 0-80% block, then used to transform the entire matrix (the 80-100%
# test rows are scaled with train-only statistics, so there is no test leakage).
# This replaces the former separate scaler_gb / scaler_lstm entirely.
# ---------------------------------------------------------------------------
X_all = basic_advanced_df_reduced[MODEL_INPUT_COLUMNS]
global_scaler = StandardScaler()
global_scaler.fit(X_all.iloc[:train_end])
X_all_scaled = global_scaler.transform(X_all)
print(f"Global scaler fit on [0:{train_end}] ({train_fraction:.0%}); transformed full matrix {X_all_scaled.shape}.")

# ---------------------------------------------------------------------------
# Section 16 — GBM Dual Pipeline (train [0:train_end] / test [train_end:])
# ---------------------------------------------------------------------------
print("\n=== 4. GBM Dual Pipeline: Chronological Split & Global Scaling ===")
y_direction = basic_advanced_df_reduced[TARGET_DIRECTION_COLUMN].values
y_return = basic_advanced_df_reduced[TARGET_RETURN_COLUMN].values    # already in PERCENT

X_gb_train_s, X_gb_test_s = X_all_scaled[:train_end], X_all_scaled[train_end:]
y_dir_train, y_dir_test = y_direction[:train_end], y_direction[train_end:]
y_ret_train, y_ret_test = y_return[:train_end], y_return[train_end:]

print("=== 5. GBM Hyperparameter Tuning ===")
tscv_gb = TimeSeriesSplit(n_splits=CONFIG['gbm']['cv_splits'])
param_grid = CONFIG['gbm']['param_grid']
# With GPU, parallelism is inside XGBoost (CUDA streams). n_jobs=1 avoids
# spawning competing processes that would each try to claim the same GPU.
_cv_n_jobs = 1 if _XGB_DEVICE == 'cuda' else -1

with mlflow.start_run(run_name="GBM_dual_pipeline") as gbm_run:
    print(f"--- Classification head (target_direction) [device={_XGB_DEVICE}] ---")
    grid_search = GridSearchCV(
        xgb.XGBClassifier(device=_XGB_DEVICE, eval_metric='auc',
                          random_state=RANDOM_STATE, verbosity=0),
        param_grid=param_grid, cv=tscv_gb, scoring='roc_auc', n_jobs=_cv_n_jobs
    )
    grid_search.fit(X_gb_train_s, y_dir_train)
    best_gbm = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}  CV ROC-AUC: {grid_search.best_score_:.4f}")

    print(f"--- Regression head (target_return [percent], pseudo-Huber) [device={_XGB_DEVICE}] ---")
    grid_search_reg = GridSearchCV(
        xgb.XGBRegressor(device=_XGB_DEVICE, objective='reg:pseudohubererror',
                         random_state=RANDOM_STATE, verbosity=0),
        param_grid=param_grid, cv=tscv_gb, scoring='neg_mean_absolute_error', n_jobs=_cv_n_jobs
    )
    grid_search_reg.fit(X_gb_train_s, y_ret_train)
    best_gbm_reg = grid_search_reg.best_estimator_
    print(f"Best params: {grid_search_reg.best_params_}  CV MAE: {-grid_search_reg.best_score_:.6f} (percent)")

    print("\n=== 6. GBM Evaluation (held-out test) ===")
    y_pred_dir = best_gbm.predict(X_gb_test_s)
    y_prob_dir = best_gbm.predict_proba(X_gb_test_s)[:, 1]
    acc_gb = accuracy_score(y_dir_test, y_pred_dir)
    auc_gb = roc_auc_score(y_dir_test, y_prob_dir)
    print(f"[Direction] Accuracy={acc_gb:.4f}  ROC-AUC={auc_gb:.4f}")

    y_pred_ret = best_gbm_reg.predict(X_gb_test_s)
    # Both heads are now natively in PERCENT units, so these errors are directly
    # comparable with the LSTM's below -- no /100 normalization required anywhere.
    mse_gb = mean_squared_error(y_ret_test, y_pred_ret)
    mae_gb = mean_absolute_error(y_ret_test, y_pred_ret)
    print(f"[Return]    MSE={mse_gb:.6f}  MAE={mae_gb:.6f}  (percent units)")

    # Train-set direction metrics -- NOT for model selection (that stays purely on
    # the held-out test block), but a diagnostic control. Per ml-practical-methodology
    # (Goodfellow Ch.11.3): if train_roc_auc is ALSO ~0.50, that is independent
    # confirmation of a Bayes-error floor (efficient market), not a bug; if train is
    # noticeably higher (e.g. 0.65+) while test stays at chance, that is classic
    # overfitting and reopens the bias-variance investigation instead.
    y_prob_dir_train = best_gbm.predict_proba(X_gb_train_s)[:, 1]
    acc_gb_train = accuracy_score(y_dir_train, best_gbm.predict(X_gb_train_s))
    auc_gb_train = roc_auc_score(y_dir_train, y_prob_dir_train)
    print(f"[Direction:TRAIN] Accuracy={acc_gb_train:.4f}  ROC-AUC={auc_gb_train:.4f}  "
          f"(train-test ROC-AUC gap={auc_gb_train - auc_gb:+.4f})")

    mlflow.log_params({
        "model_family": "XGBoost_DualPipeline",
        "device": _XGB_DEVICE,
        "direction_n_estimators": grid_search.best_params_['n_estimators'],
        "direction_learning_rate": grid_search.best_params_['learning_rate'],
        "direction_max_depth": grid_search.best_params_['max_depth'],
        "return_n_estimators": grid_search_reg.best_params_['n_estimators'],
        "return_learning_rate": grid_search_reg.best_params_['learning_rate'],
        "return_max_depth": grid_search_reg.best_params_['max_depth'],
        "huber_alpha": CONFIG['gbm']['huber_alpha'],
        "cv_splits": CONFIG['gbm']['cv_splits'],
        "pca_variance_threshold": CONFIG['pca']['variance_threshold'],
        "n_model_input_features": len(MODEL_INPUT_COLUMNS),
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "target_unit": "percent",
        "scaler": "global_StandardScaler",
        "macro_yield_differential_source": macro_source,
    })
    mlflow.log_metrics({
        "direction_accuracy": acc_gb,
        "direction_roc_auc": auc_gb,
        "return_mse": mse_gb,
        "return_mae": mae_gb,
        "train_direction_accuracy": acc_gb_train,
        "train_direction_roc_auc": auc_gb_train,
    })
    _safe_log_model(mlflow.sklearn.log_model, best_gbm, "gbm_direction_classifier")
    _safe_log_model(mlflow.sklearn.log_model, best_gbm_reg, "gbm_return_regressor")
    print(f"MLflow run logged: run_id={gbm_run.info.run_id}")

print("\n=== 7. Persisting GBM + PCA + global scaler artifacts ===")
os.makedirs('models', exist_ok=True)
joblib.dump(lag_scaler, 'models/lag_scaler.pkl')
joblib.dump(lag_pca, 'models/lag_pca.pkl')
joblib.dump(global_scaler, 'models/global_scaler.pkl')
joblib.dump(best_gbm, 'models/best_gbm_eurusd.pkl')
joblib.dump(best_gbm_reg, 'models/best_gbm_regressor_eurusd.pkl')
print("Saved: lag_scaler.pkl, lag_pca.pkl, global_scaler.pkl, best_gbm_eurusd.pkl, best_gbm_regressor_eurusd.pkl")

# Feature importance (ESL §10.13.1, eq. 10.43 — importance averaged over all M
# trees, robust to correlated inputs thanks to shrinkage). Never extracted before;
# directly actionable alongside the FRED ablation (results/2C_fred_ablation.csv)
# to see whether other features are similarly dead weight before the next retrain.
feature_importance = pd.DataFrame({
    "feature": MODEL_INPUT_COLUMNS,
    "importance_direction_classifier": best_gbm.feature_importances_,
    "importance_return_regressor": best_gbm_reg.feature_importances_,
}).sort_values("importance_direction_classifier", ascending=False)
feature_importance.to_csv("results/gbm_feature_importance.csv", index=False)
print(f"Saved: results/gbm_feature_importance.csv\n{feature_importance.to_string(index=False)}")

# The former per-model scalers are now superseded by the single global_scaler.
# Remove any stale copies so inference can never silently load an out-of-date,
# wrong-unit scaler alongside the freshly retrained artifacts.
for _stale in ('models/scaler_gb_eurusd.pkl', 'models/scaler_lstm_multitask.pkl'):
    if os.path.exists(_stale):
        os.remove(_stale)
        print(f"Removed superseded scaler: {_stale}")

# ---------------------------------------------------------------------------
# Section 15 — Multi-Task LSTM (Functional API, shared trunk, dual heads)
# ---------------------------------------------------------------------------
print("\n=== 8. Multi-Task LSTM: Sliding-Window Data Preparation ===")
# target_return is ALREADY in percent (src/features.py), which keeps the return
# head's MSE on the same order of magnitude as the direction head's BCE so the
# shared trunk gets real gradient for both tasks at equal loss_weights. There is
# therefore no *100 rescaling here anymore.
data_y_return = basic_advanced_df_reduced[TARGET_RETURN_COLUMN].values
data_y_direction = basic_advanced_df_reduced[TARGET_DIRECTION_COLUMN].values

# Reuse the SAME global-scaler-transformed matrix -- the LSTM no longer owns a
# scaler. Slice it into the unified 0-70 / 70-80 / 80-100 chronological blocks.
X_train_s = X_all_scaled[:lstm_train_end]
X_val_s = X_all_scaled[lstm_train_end:train_end]
X_test_s = X_all_scaled[train_end:]

y_ret_train_raw, y_dir_train_raw = data_y_return[:lstm_train_end], data_y_direction[:lstm_train_end]
y_ret_val_raw, y_dir_val_raw = data_y_return[lstm_train_end:train_end], data_y_direction[lstm_train_end:train_end]
y_ret_test_raw, y_dir_test_raw = data_y_return[train_end:], data_y_direction[train_end:]


def create_mt_sequences(X, y_ret, y_dir, time_steps):
    Xs, ys_ret, ys_dir = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys_ret.append(y_ret[i + time_steps - 1])
        ys_dir.append(y_dir[i + time_steps - 1])
    return np.array(Xs), np.array(ys_ret), np.array(ys_dir)


TIME_STEPS = CONFIG['lstm']['time_steps']
X_train_seq, y_ret_train_seq, y_dir_train_seq = create_mt_sequences(X_train_s, y_ret_train_raw, y_dir_train_raw, TIME_STEPS)
X_val_seq, y_ret_val_seq, y_dir_val_seq = create_mt_sequences(X_val_s, y_ret_val_raw, y_dir_val_raw, TIME_STEPS)
X_test_seq, y_ret_test_seq, y_dir_test_seq = create_mt_sequences(X_test_s, y_ret_test_raw, y_dir_test_raw, TIME_STEPS)

print(f"Train seq: {X_train_seq.shape}  Val seq: {X_val_seq.shape}  Test seq: {X_test_seq.shape}")

print("\n=== 9. Multi-Task LSTM Model Construction (Functional API) ===")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

_tf_gpus = tf.config.list_physical_devices('GPU')
if _tf_gpus:
    for _gpu in _tf_gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"TensorFlow GPU: {len(_tf_gpus)} device(s) detected — mixed_float16 enabled.")
else:
    print("TensorFlow: no GPU on native Windows (TF >=2.11 requires WSL2 for CUDA). Training on CPU.")

tf.random.set_seed(RANDOM_STATE)

inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name="ohlcv_window")
shared = LSTM(CONFIG['lstm']['units'], name="shared_lstm_trunk")(inputs)
shared = Dropout(CONFIG['lstm']['dropout'], name="shared_dropout")(shared)
return_output = Dense(1, activation='linear', name="return_output", dtype='float32')(shared)
direction_output = Dense(1, activation='sigmoid', name="direction_output", dtype='float32')(shared)

mt_lstm_model = Model(inputs=inputs, outputs=[return_output, direction_output], name="multitask_lstm_eurusd")
mt_lstm_model.compile(
    optimizer=Adam(learning_rate=CONFIG['lstm']['learning_rate']),
    loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
    loss_weights=CONFIG['lstm']['loss_weights'],
    metrics={'return_output': 'mae', 'direction_output': 'accuracy'}
)
mt_lstm_model.summary()

with mlflow.start_run(run_name="MultiTask_LSTM") as lstm_run:
    print("\n=== 10. Training (early-stopping on the 70-80% validation block) ===")
    early_stop = EarlyStopping(monitor='val_loss', patience=CONFIG['lstm']['patience'], restore_best_weights=True, verbose=1)
    history = mt_lstm_model.fit(
        X_train_seq,
        {'return_output': y_ret_train_seq, 'direction_output': y_dir_train_seq},
        validation_data=(X_val_seq, {'return_output': y_ret_val_seq, 'direction_output': y_dir_val_seq}),
        epochs=CONFIG['lstm']['epochs'], batch_size=CONFIG['lstm']['batch_size'], callbacks=[early_stop], verbose=2
    )
    print(f"Stopped after {len(history.history['loss'])} epochs.")

    print("\n=== 11. Evaluation (held-out test) ===")
    y_pred_ret_lstm, y_prob_dir_lstm = mt_lstm_model.predict(X_test_seq, verbose=0)
    y_pred_ret_lstm = y_pred_ret_lstm.ravel()
    y_prob_dir_lstm = y_prob_dir_lstm.ravel()
    y_pred_dir_lstm = (y_prob_dir_lstm >= 0.5).astype(int)

    # Both target and prediction are in percent units (same as the GBM regressor
    # above), so MSE/MAE are reported directly and are head-to-head comparable.
    mse_lstm = mean_squared_error(y_ret_test_seq, y_pred_ret_lstm)
    mae_lstm = mean_absolute_error(y_ret_test_seq, y_pred_ret_lstm)
    acc_lstm = accuracy_score(y_dir_test_seq, y_pred_dir_lstm)
    auc_lstm = roc_auc_score(y_dir_test_seq, y_prob_dir_lstm)
    print(f"[Return]    MSE={mse_lstm:.6f}  MAE={mae_lstm:.6f}  (percent units, comparable to GBM)")
    print(f"[Direction] Accuracy={acc_lstm:.4f}  ROC-AUC={auc_lstm:.4f}")

    # Multi-task trunk justification check (ml-practical-methodology Part B /
    # Goodfellow Ch.7.7): the shared_lstm_trunk only earns its generalization
    # benefit if the factors predicting return magnitude and direction actually
    # overlap. Measure how often the return head's sign agrees with the direction
    # head's call on the SAME held-out test block -- barely above 50% would mean
    # the two heads are not pulling from a shared signal, and two single-task
    # models should be tried instead.
    sign_agreement_lstm = (np.sign(y_pred_ret_lstm) == np.where(y_pred_dir_lstm == 1, 1, -1)).mean()
    print(f"[Multi-task check] Return-sign vs direction-head agreement: {sign_agreement_lstm:.4f} "
          f"({'weak -- shared trunk may not be justified' if sign_agreement_lstm < 0.55 else 'shared signal present'})")

    # Train-set control (see ml-practical-methodology / GBM section above for the
    # rationale): lets us tell a Bayes-error floor (train≈test≈0.50) apart from a
    # bug or genuine overfitting (train materially above test).
    _, y_prob_dir_lstm_train = mt_lstm_model.predict(X_train_seq, verbose=0)
    y_prob_dir_lstm_train = y_prob_dir_lstm_train.ravel()
    y_pred_dir_lstm_train = (y_prob_dir_lstm_train >= 0.5).astype(int)
    acc_lstm_train = accuracy_score(y_dir_train_seq, y_pred_dir_lstm_train)
    auc_lstm_train = roc_auc_score(y_dir_train_seq, y_prob_dir_lstm_train)
    print(f"[Direction:TRAIN] Accuracy={acc_lstm_train:.4f}  ROC-AUC={auc_lstm_train:.4f}  "
          f"(train-test ROC-AUC gap={auc_lstm_train - auc_lstm:+.4f})")

    mlflow.log_params({
        "model_family": "MultiTask_LSTM_FunctionalAPI",
        "units": CONFIG['lstm']['units'],
        "dropout": CONFIG['lstm']['dropout'],
        "learning_rate": CONFIG['lstm']['learning_rate'],
        "epochs_configured": CONFIG['lstm']['epochs'],
        "epochs_trained": len(history.history['loss']),
        "batch_size": CONFIG['lstm']['batch_size'],
        "patience": CONFIG['lstm']['patience'],
        "time_steps": TIME_STEPS,
        "loss_weight_return": CONFIG['lstm']['loss_weights']['return_output'],
        "loss_weight_direction": CONFIG['lstm']['loss_weights']['direction_output'],
        "n_model_input_features": len(MODEL_INPUT_COLUMNS),
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "target_unit": "percent",
        "scaler": "global_StandardScaler",
        "macro_yield_differential_source": macro_source,
    })
    mlflow.log_metrics({
        "return_mse": mse_lstm,
        "return_mae": mae_lstm,
        "direction_accuracy": acc_lstm,
        "direction_roc_auc": auc_lstm,
        "train_direction_accuracy": acc_lstm_train,
        "train_direction_roc_auc": auc_lstm_train,
        "multitask_sign_agreement": sign_agreement_lstm,
    })
    _safe_log_model(mlflow.keras.log_model, mt_lstm_model, "multitask_lstm")
    print(f"MLflow run logged: run_id={lstm_run.info.run_id}")

print("\n=== 12. Persisting LSTM artifacts ===")
mt_lstm_model.save('models/lstm_multitask_eurusd.keras')
joblib.dump(TIME_STEPS, 'models/lstm_time_steps.pkl')
print("Saved: lstm_multitask_eurusd.keras, lstm_time_steps.pkl")

# ===========================================================================
# H1 -> Daily Predictor (auxiliary multi-model ensemble)
# ===========================================================================
# ADDITIVE to the daily pipeline above -- it never touches the 7 canonical daily
# artifacts. It trains XGBoost / RandomForest / SVM on FLATTENED intraday
# statistics and a sequence-to-vector LSTM on the raw (samples, 24, features)
# tensor, all forecasting the next-day (t+1) return. Both metrics required by the
# task are logged: MAE (regression quality) and a directional ROC-AUC derived by
# scoring the predicted return against the realised up/down label. The whole
# block is guarded so that H1 data being unavailable degrades gracefully to a
# warning and leaves production's daily models intact.
print("\n=== 13. H1 -> Daily Predictor (XGBoost / RandomForest / SVM / LSTM) ===")
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from src.h1_features import (
        build_h1_datasets, FLAT_FEATURE_COLUMNS, SEQ_FEATURE_COLUMNS, HOURS_PER_DAY,
    )

    h1_cfg = CONFIG.get('h1', {})
    h1_cache = h1_cfg.get('cache_path', 'results/eurusd_h1.csv')

    # Refresh the H1 cache with fresh bars (MT5 -> yfinance) before training so a
    # retrain actually learns from new data. fetch_h1_market_data writes the CSV
    # itself; if every live source is unreachable it falls back to the existing
    # cache, so an offline retrain degrades to the last-known data rather than
    # aborting the whole H1 section.
    try:
        from src.live_data import fetch_h1_market_data
        _df, _src = fetch_h1_market_data(cache_path=h1_cache)
        print(f"H1 cache refreshed from {_src}: {0 if _df is None else len(_df)} bars.")
    except Exception as e:
        print(f"H1 cache refresh skipped ({e}); training on existing cache.")

    X_flat_df, X_seq, y_ret_h1, y_dir_h1, h1_index = build_h1_datasets(cache_path=h1_cache)
    print(f"H1 datasets: flat {X_flat_df.shape}, seq {X_seq.shape}, "
          f"days {h1_index.min().date()} -> {h1_index.max().date()}")

    # Chronological 80/20 split shared by every H1 model -- NO shuffling.
    n_h1 = len(X_flat_df)
    h1_train_end = int(n_h1 * CONFIG['split']['train_fraction'])
    X_flat = X_flat_df.values
    X_flat_tr, X_flat_te = X_flat[:h1_train_end], X_flat[h1_train_end:]
    y_ret_tr, y_ret_te = y_ret_h1[:h1_train_end], y_ret_h1[h1_train_end:]
    y_dir_te = y_dir_h1[h1_train_end:]

    # StandardScaler fit on the TRAIN flattened features only. It is required by
    # the RBF SVM (distance-based); the tree models are scale-invariant and so
    # receive the raw features, but the scaler is still serialized for inference.
    h1_scaler = StandardScaler().fit(X_flat_tr)
    X_flat_tr_s = h1_scaler.transform(X_flat_tr)
    X_flat_te_s = h1_scaler.transform(X_flat_te)

    tscv_h1 = TimeSeriesSplit(n_splits=h1_cfg.get('cv_splits', 5))

    def _directional_auc(y_true_dir, pred_return):
        """ROC-AUC of the predicted return as a score for up(1)/down(0) days.
        Returns NaN on the degenerate single-class test window."""
        if len(np.unique(y_true_dir)) < 2:
            return float('nan')
        return roc_auc_score(y_true_dir, pred_return)

    h1_results = {}

    with mlflow.start_run(run_name="H1_to_Daily_Ensemble") as h1_run:
        # --- 1) XGBoost regressor on flattened features (GPU-aware) ---
        print(f"--- XGBoost [device={_XGB_DEVICE}] ---")
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(device=_XGB_DEVICE, objective='reg:squarederror',
                             random_state=RANDOM_STATE, verbosity=0),
            param_grid=CONFIG['gbm']['param_grid'], cv=tscv_h1,
            scoring='neg_mean_absolute_error', n_jobs=_cv_n_jobs)
        xgb_grid.fit(X_flat_tr, y_ret_tr)
        h1_xgb = xgb_grid.best_estimator_
        _p = h1_xgb.predict(X_flat_te)
        h1_results['xgboost'] = (mean_absolute_error(y_ret_te, _p), _directional_auc(y_dir_te, _p))

        # --- 2) Random Forest: bagging + random feature subsets -> low variance ---
        print("--- RandomForest ---")
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=h1_cfg.get('rf_param_grid',
                                  {"n_estimators": [200, 400], "max_depth": [4, 8, None]}),
            cv=tscv_h1, scoring='neg_mean_absolute_error', n_jobs=-1)
        rf_grid.fit(X_flat_tr, y_ret_tr)
        h1_rf = rf_grid.best_estimator_
        _p = h1_rf.predict(X_flat_te)
        h1_results['random_forest'] = (mean_absolute_error(y_ret_te, _p), _directional_auc(y_dir_te, _p))

        # --- 3) SVM (RBF kernel) on SCALED flattened features ---
        print("--- SVM (RBF) ---")
        svm_grid = GridSearchCV(
            SVR(kernel='rbf'),
            param_grid=h1_cfg.get('svm_param_grid', {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1]}),
            cv=tscv_h1, scoring='neg_mean_absolute_error', n_jobs=-1)
        svm_grid.fit(X_flat_tr_s, y_ret_tr)
        h1_svm = svm_grid.best_estimator_
        _p = h1_svm.predict(X_flat_te_s)
        h1_results['svm_rbf'] = (mean_absolute_error(y_ret_te, _p), _directional_auc(y_dir_te, _p))

        # --- 4) Sequence-to-vector LSTM on the (samples, 24, n_features) tensor ---
        print("--- Sequence-to-Vector LSTM ---")
        n_seq_feat = X_seq.shape[2]
        # Per-hour scaler fit on TRAIN timesteps only, applied across the tensor.
        seq_scaler = StandardScaler().fit(X_seq[:h1_train_end].reshape(-1, n_seq_feat))

        def _scale_seq(t):
            return seq_scaler.transform(t.reshape(-1, n_seq_feat)).reshape(t.shape).astype('float32')

        X_seq_s = _scale_seq(X_seq)
        # Inner chronological validation slice for early stopping (tail of train).
        val_frac_of_train = CONFIG['split']['val_fraction'] / CONFIG['split']['train_fraction']
        lstm_val_start = int(h1_train_end * (1 - val_frac_of_train))
        Xs_tr, Xs_val, Xs_te = X_seq_s[:lstm_val_start], X_seq_s[lstm_val_start:h1_train_end], X_seq_s[h1_train_end:]
        ys_tr, ys_val = y_ret_h1[:lstm_val_start], y_ret_h1[lstm_val_start:h1_train_end]

        seq_inputs = Input(shape=(HOURS_PER_DAY, n_seq_feat), name="h1_hourly_window")
        xh = LSTM(h1_cfg.get('lstm_units', 32), name="h1_lstm_trunk")(seq_inputs)
        xh = Dropout(h1_cfg.get('lstm_dropout', 0.2), name="h1_dropout")(xh)
        seq_out = Dense(1, activation='linear', name="h1_return", dtype='float32')(xh)
        h1_lstm = Model(seq_inputs, seq_out, name="h1_seq2vec_lstm")
        h1_lstm.compile(optimizer=Adam(learning_rate=h1_cfg.get('lstm_lr', 0.001)),
                        loss='mse', metrics=['mae'])
        _es = EarlyStopping(monitor='val_loss', patience=h1_cfg.get('lstm_patience', 8),
                            restore_best_weights=True, verbose=0)
        h1_lstm.fit(Xs_tr, ys_tr, validation_data=(Xs_val, ys_val),
                    epochs=h1_cfg.get('lstm_epochs', 60), batch_size=h1_cfg.get('lstm_batch', 32),
                    callbacks=[_es], verbose=2)
        _p = h1_lstm.predict(Xs_te, verbose=0).ravel()
        h1_results['lstm_seq2vec'] = (mean_absolute_error(y_ret_te, _p), _directional_auc(y_dir_te, _p))

        # --- MLflow: log MAE + directional ROC-AUC for every model ---
        print("\n=== 14. H1 Evaluation (held-out test) ===")
        for _name, (_mae, _auc) in h1_results.items():
            print(f"[H1:{_name:14s}] MAE={_mae:.4f}%  directional ROC-AUC={_auc:.4f}")
            mlflow.log_metric(f"{_name}_mae", _mae)
            if not np.isnan(_auc):
                mlflow.log_metric(f"{_name}_roc_auc", _auc)
        mlflow.log_params({
            "predictor": "H1_to_Daily",
            "models": "xgboost,random_forest,svm_rbf,lstm_seq2vec",
            "n_flat_features": len(FLAT_FEATURE_COLUMNS),
            "seq_shape": f"({HOURS_PER_DAY},{n_seq_feat})",
            "seq_features": ",".join(SEQ_FEATURE_COLUMNS),
            "n_days": n_h1,
            "train_fraction": CONFIG['split']['train_fraction'],
            "target_unit": "percent",
            "device": _XGB_DEVICE,
        })
        _safe_log_model(mlflow.sklearn.log_model, h1_xgb, "h1_xgboost")
        _safe_log_model(mlflow.sklearn.log_model, h1_rf, "h1_random_forest")
        _safe_log_model(mlflow.sklearn.log_model, h1_svm, "h1_svm")
        _safe_log_model(mlflow.keras.log_model, h1_lstm, "h1_lstm")
        print(f"MLflow run logged: run_id={h1_run.info.run_id}")

    print("\n=== 15. Persisting H1 -> Daily artifacts ===")
    joblib.dump(h1_xgb, 'models/h1_xgb_regressor.pkl')
    joblib.dump(h1_rf, 'models/h1_rf_regressor.pkl')
    joblib.dump(h1_svm, 'models/h1_svm_regressor.pkl')
    joblib.dump(h1_scaler, 'models/h1_feature_scaler.pkl')
    joblib.dump(seq_scaler, 'models/h1_lstm_scaler.pkl')
    joblib.dump(list(FLAT_FEATURE_COLUMNS), 'models/h1_feature_columns.pkl')
    joblib.dump({"hours": HOURS_PER_DAY, "seq_features": list(SEQ_FEATURE_COLUMNS)},
                'models/h1_lstm_config.pkl')
    h1_lstm.save('models/h1_lstm.keras')
    print("Saved H1 artifacts: h1_xgb_regressor.pkl, h1_rf_regressor.pkl, h1_svm_regressor.pkl, "
          "h1_feature_scaler.pkl, h1_lstm_scaler.pkl, h1_lstm.keras, h1_feature_columns.pkl, h1_lstm_config.pkl")
except Exception as _h1_err:
    import traceback
    print(f"WARNING: H1 -> Daily predictor section skipped ({_h1_err}). "
          f"Daily production artifacts are unaffected.")
    traceback.print_exc()

print("\n=== DONE ===")
