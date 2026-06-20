"""
Standalone training run mirroring notebooks/01_data_preparation.ipynb
Sections 3/4 (features), 15 (GBM dual pipeline) and 14+16 (Multi-Task LSTM),
sourcing raw OHLCV from results/eurusd_features.csv since no live MT5
terminal is available in this environment. Produces real artifacts under
models/ to replace the previous mocked inference path.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
)

from src.features import add_advanced_features, TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN, FEATURE_COLUMNS

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=== 1. Loading historical OHLCV (results/eurusd_features.csv) ===")
raw_df = pd.read_csv('results/eurusd_features.csv', index_col='time', parse_dates=True)
raw_df = raw_df[['open', 'high', 'low', 'close', 'tick_volume']]
print(f"Loaded {len(raw_df):,} bars ({raw_df.index[0].date()} -> {raw_df.index[-1].date()})")

print("\n=== 2. Feature Engineering (Multi-Task targets) ===")
basic_advanced_df = add_advanced_features(raw_df)
assert list(basic_advanced_df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
print(f"Shape: {basic_advanced_df.shape}")

# ---------------------------------------------------------------------------
# Section 15 — GBM Dual Pipeline
# ---------------------------------------------------------------------------
print("\n=== 3. GBM Dual Pipeline: Chronological Split & Scaling ===")
y_direction = basic_advanced_df[TARGET_DIRECTION_COLUMN]
y_return = basic_advanced_df[TARGET_RETURN_COLUMN]
X = basic_advanced_df[FEATURE_COLUMNS]

n_split = int(len(basic_advanced_df) * 0.8)
X_gb_train, X_gb_test = X.iloc[:n_split], X.iloc[n_split:]
y_dir_train, y_dir_test = y_direction.iloc[:n_split], y_direction.iloc[n_split:]
y_ret_train, y_ret_test = y_return.iloc[:n_split], y_return.iloc[n_split:]

scaler_gb = StandardScaler()
X_gb_train_s = scaler_gb.fit_transform(X_gb_train)
X_gb_test_s = scaler_gb.transform(X_gb_test)

print("=== 4. GBM Hyperparameter Tuning ===")
tscv_gb = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

print("--- Classification head (target_direction) ---")
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    param_grid=param_grid, cv=tscv_gb, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_gb_train_s, y_dir_train)
best_gbm = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}  CV ROC-AUC: {grid_search.best_score_:.4f}")

print("--- Regression head (target_return, Huber loss) ---")
grid_search_reg = GridSearchCV(
    GradientBoostingRegressor(loss='huber', alpha=0.9, random_state=RANDOM_STATE),
    param_grid=param_grid, cv=tscv_gb, scoring='neg_mean_absolute_error', n_jobs=-1
)
grid_search_reg.fit(X_gb_train_s, y_ret_train)
best_gbm_reg = grid_search_reg.best_estimator_
print(f"Best params: {grid_search_reg.best_params_}  CV MAE: {-grid_search_reg.best_score_:.6f}")

print("\n=== 5. GBM Evaluation ===")
y_pred_dir = best_gbm.predict(X_gb_test_s)
y_prob_dir = best_gbm.predict_proba(X_gb_test_s)[:, 1]
acc_gb = accuracy_score(y_dir_test, y_pred_dir)
auc_gb = roc_auc_score(y_dir_test, y_prob_dir)
print(f"[Direction] Accuracy={acc_gb:.4f}  ROC-AUC={auc_gb:.4f}")

y_pred_ret = best_gbm_reg.predict(X_gb_test_s)
mse_gb = mean_squared_error(y_ret_test, y_pred_ret)
mae_gb = mean_absolute_error(y_ret_test, y_pred_ret)
print(f"[Return]    MSE={mse_gb:.8f}  MAE={mae_gb:.6f}")

print("\n=== 6. Persisting GBM artifacts ===")
os.makedirs('models', exist_ok=True)
joblib.dump(best_gbm, 'models/best_gbm_eurusd.pkl')
joblib.dump(best_gbm_reg, 'models/best_gbm_regressor_eurusd.pkl')
joblib.dump(scaler_gb, 'models/scaler_gb_eurusd.pkl')
print("Saved: best_gbm_eurusd.pkl, best_gbm_regressor_eurusd.pkl, scaler_gb_eurusd.pkl")

# ---------------------------------------------------------------------------
# Sections 14+16 — Multi-Task LSTM (Functional API, shared trunk, dual heads)
# ---------------------------------------------------------------------------
print("\n=== 7. Multi-Task LSTM: Sliding-Window Data Preparation ===")
df_dl = basic_advanced_df.dropna().copy()
data_x = df_dl[FEATURE_COLUMNS].values
data_y_return = df_dl[TARGET_RETURN_COLUMN].values
data_y_direction = df_dl[TARGET_DIRECTION_COLUMN].values

n_total = len(df_dl)
train_end = int(n_total * 0.70)
val_end = int(n_total * 0.85)

X_train_raw, y_ret_train_raw, y_dir_train_raw = data_x[:train_end], data_y_return[:train_end], data_y_direction[:train_end]
X_val_raw, y_ret_val_raw, y_dir_val_raw = data_x[train_end:val_end], data_y_return[train_end:val_end], data_y_direction[train_end:val_end]
X_test_raw, y_ret_test_raw, y_dir_test_raw = data_x[val_end:], data_y_return[val_end:], data_y_direction[val_end:]

scaler_lstm = StandardScaler()
X_train_s = scaler_lstm.fit_transform(X_train_raw)
X_val_s = scaler_lstm.transform(X_val_raw)
X_test_s = scaler_lstm.transform(X_test_raw)


def create_mt_sequences(X, y_ret, y_dir, time_steps=20):
    Xs, ys_ret, ys_dir = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys_ret.append(y_ret[i + time_steps - 1])
        ys_dir.append(y_dir[i + time_steps - 1])
    return np.array(Xs), np.array(ys_ret), np.array(ys_dir)


TIME_STEPS = 20
X_train_seq, y_ret_train_seq, y_dir_train_seq = create_mt_sequences(X_train_s, y_ret_train_raw, y_dir_train_raw, TIME_STEPS)
X_val_seq, y_ret_val_seq, y_dir_val_seq = create_mt_sequences(X_val_s, y_ret_val_raw, y_dir_val_raw, TIME_STEPS)
X_test_seq, y_ret_test_seq, y_dir_test_seq = create_mt_sequences(X_test_s, y_ret_test_raw, y_dir_test_raw, TIME_STEPS)

print(f"Train seq: {X_train_seq.shape}  Val seq: {X_val_seq.shape}  Test seq: {X_test_seq.shape}")

print("\n=== 8. Multi-Task LSTM Model Construction (Functional API) ===")
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(RANDOM_STATE)

inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name="ohlcv_window")
shared = LSTM(64, name="shared_lstm_trunk")(inputs)
shared = Dropout(0.3, name="shared_dropout")(shared)
return_output = Dense(1, activation='linear', name="return_output")(shared)
direction_output = Dense(1, activation='sigmoid', name="direction_output")(shared)

mt_lstm_model = Model(inputs=inputs, outputs=[return_output, direction_output], name="multitask_lstm_eurusd")
mt_lstm_model.compile(
    optimizer='adam',
    loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
    loss_weights={'return_output': 1.0, 'direction_output': 50.0},
    metrics={'return_output': 'mae', 'direction_output': 'accuracy'}
)
mt_lstm_model.summary()

print("\n=== 9. Training ===")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
history = mt_lstm_model.fit(
    X_train_seq,
    {'return_output': y_ret_train_seq, 'direction_output': y_dir_train_seq},
    validation_data=(X_val_seq, {'return_output': y_ret_val_seq, 'direction_output': y_dir_val_seq}),
    epochs=100, batch_size=64, callbacks=[early_stop], verbose=2
)
print(f"Stopped after {len(history.history['loss'])} epochs.")

print("\n=== 10. Evaluation ===")
y_pred_ret_lstm, y_prob_dir_lstm = mt_lstm_model.predict(X_test_seq, verbose=0)
y_pred_ret_lstm = y_pred_ret_lstm.ravel()
y_prob_dir_lstm = y_prob_dir_lstm.ravel()
y_pred_dir_lstm = (y_prob_dir_lstm >= 0.5).astype(int)

print(f"[Return]    MSE={mean_squared_error(y_ret_test_seq, y_pred_ret_lstm):.8f}  MAE={mean_absolute_error(y_ret_test_seq, y_pred_ret_lstm):.6f}")
print(f"[Direction] Accuracy={accuracy_score(y_dir_test_seq, y_pred_dir_lstm):.4f}  ROC-AUC={roc_auc_score(y_dir_test_seq, y_prob_dir_lstm):.4f}")

print("\n=== 11. Persisting LSTM artifacts ===")
mt_lstm_model.save('models/lstm_multitask_eurusd.keras')
joblib.dump(scaler_lstm, 'models/scaler_lstm_multitask.pkl')
joblib.dump(TIME_STEPS, 'models/lstm_time_steps.pkl')
print("Saved: lstm_multitask_eurusd.keras, scaler_lstm_multitask.pkl, lstm_time_steps.pkl")

print("\n=== DONE ===")
