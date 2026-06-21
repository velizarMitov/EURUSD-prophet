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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

mlflow.set_experiment("EURUSD_Prediction")

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

with mlflow.start_run(run_name="GBM_dual_pipeline") as gbm_run:
    print("--- Classification head (target_direction) ---")
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid, cv=tscv_gb, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_gb_train_s, y_dir_train)
    best_gbm = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}  CV ROC-AUC: {grid_search.best_score_:.4f}")

    print("--- Regression head (target_return [percent], Huber loss) ---")
    grid_search_reg = GridSearchCV(
        GradientBoostingRegressor(loss='huber', alpha=CONFIG['gbm']['huber_alpha'], random_state=RANDOM_STATE),
        param_grid=param_grid, cv=tscv_gb, scoring='neg_mean_absolute_error', n_jobs=-1
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

    mlflow.log_params({
        "model_family": "GradientBoosting_DualPipeline",
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
    })
    mlflow.sklearn.log_model(best_gbm, artifact_path="gbm_direction_classifier")
    mlflow.sklearn.log_model(best_gbm_reg, artifact_path="gbm_return_regressor")
    print(f"MLflow run logged: run_id={gbm_run.info.run_id}")

print("\n=== 7. Persisting GBM + PCA + global scaler artifacts ===")
os.makedirs('models', exist_ok=True)
joblib.dump(lag_scaler, 'models/lag_scaler.pkl')
joblib.dump(lag_pca, 'models/lag_pca.pkl')
joblib.dump(global_scaler, 'models/global_scaler.pkl')
joblib.dump(best_gbm, 'models/best_gbm_eurusd.pkl')
joblib.dump(best_gbm_reg, 'models/best_gbm_regressor_eurusd.pkl')
print("Saved: lag_scaler.pkl, lag_pca.pkl, global_scaler.pkl, best_gbm_eurusd.pkl, best_gbm_regressor_eurusd.pkl")

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

tf.random.set_seed(RANDOM_STATE)

inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name="ohlcv_window")
shared = LSTM(CONFIG['lstm']['units'], name="shared_lstm_trunk")(inputs)
shared = Dropout(CONFIG['lstm']['dropout'], name="shared_dropout")(shared)
return_output = Dense(1, activation='linear', name="return_output")(shared)
direction_output = Dense(1, activation='sigmoid', name="direction_output")(shared)

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
    })
    mlflow.keras.log_model(mt_lstm_model, artifact_path="multitask_lstm")
    print(f"MLflow run logged: run_id={lstm_run.info.run_id}")

print("\n=== 12. Persisting LSTM artifacts ===")
mt_lstm_model.save('models/lstm_multitask_eurusd.keras')
joblib.dump(TIME_STEPS, 'models/lstm_time_steps.pkl')
print("Saved: lstm_multitask_eurusd.keras, lstm_time_steps.pkl")

print("\n=== DONE ===")
