import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Canonical 20-column feature order consumed by the trained GBM/LSTM artifacts.
# Shared by the notebook (training) and api.py/app.py (inference) so the two
# can never silently drift apart.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'tick_volume', 'log_return',
    'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200', 'volatility_20',
    'bar_dynamics', 'return_lag_1', 'dynamics_lag_1', 'return_lag_2',
    'dynamics_lag_2', 'return_lag_3', 'dynamics_lag_3', 'day_sin', 'day_cos'
]

# The 6 autoregressive lag columns are the most mutually correlated block in
# FEATURE_COLUMNS (each is a shifted copy of log_return/bar_dynamics) and are
# the dimensionality-reduction target described in the project's PCA section.
LAG_COLUMNS = [
    'return_lag_1', 'dynamics_lag_1',
    'return_lag_2', 'dynamics_lag_2',
    'return_lag_3', 'dynamics_lag_3',
]

TARGET_RETURN_COLUMN = 'target_return'
TARGET_DIRECTION_COLUMN = 'target_direction'

DEFAULT_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'eurusd_features.csv'
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 20 FEATURE_COLUMNS from raw OHLCV. Unlike add_advanced_features,
    this does NOT compute targets or drop rows, so the most recent bar (which
    has no future bar to derive a target from) survives — required for live
    inference on the latest available row.
    """
    data = df.copy()

    # 1. Log return (Stationarity)
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 2. Simple Moving Averages
    for period in [21, 50, 100, 200]:
        data[f'SMA_{period}'] = data['close'].rolling(period).mean()

    # 3. Rolling Volatility
    data['volatility_20'] = data['log_return'].rolling(20).std()

    # 4. Bar Dynamics
    # To prevent DivisionByZero logically inside pandas, replace 0 open with NaN temporarily
    safe_open = data['open'].replace(0, np.nan)
    data['bar_dynamics'] = (data['high'] - data['low']) / safe_open

    # 5. Autoregressive Lags
    for lag in range(1, 4):
        data[f'return_lag_{lag}'] = data['log_return'].shift(lag)
        data[f'dynamics_lag_{lag}'] = data['bar_dynamics'].shift(lag)

    # 6. Cyclical Encoding
    day_of_week = data.index.dayofweek
    data['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    data['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    return data


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracted purely functional processing logic for production API and Unit Testing.
    """
    data = compute_features(df)

    # Multi-Task Targets: continuous next-period return + its derived sign
    data[TARGET_RETURN_COLUMN] = (data['close'].shift(-1) - data['close']) / data['close']
    data[TARGET_DIRECTION_COLUMN] = (data[TARGET_RETURN_COLUMN] > 0).astype(int)

    # Clean missing values systematically
    data.dropna(inplace=True)
    return data


def load_history(path: str = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Load the real historical OHLCV bars used as live feature context."""
    history = pd.read_csv(path, index_col='time', parse_dates=True)
    return history[['open', 'high', 'low', 'close', 'tick_volume']]


def build_live_features(history: pd.DataFrame, new_bar: dict, time_steps: int) -> pd.DataFrame:
    """
    Append `new_bar` (a dict with open/high/low/close/tick_volume) as the next
    chronological bar after `history`, recompute the real rolling indicators
    over genuine price history (no hardcoded/mocked SMA or lag values), and
    return the last `time_steps` engineered rows ending at the new bar.
    """
    next_date = history.index[-1] + pd.Timedelta(days=1)
    new_row = pd.DataFrame([new_bar], index=pd.DatetimeIndex([next_date], name=history.index.name))
    combined = pd.concat([history, new_row])
    engineered = compute_features(combined)
    return engineered[FEATURE_COLUMNS].tail(time_steps)


def fit_lag_pca(df: pd.DataFrame, lag_columns=LAG_COLUMNS, variance_threshold: float = 0.95):
    """
    Fit a StandardScaler + PCA on the autoregressive lag columns only. Must be
    fit on a chronological TRAIN slice (never the full dataset) to avoid leakage.
    Returns (lag_scaler, lag_pca); lag_pca.n_components_ is chosen automatically
    to explain >= variance_threshold of the lag block's variance.
    """
    lag_scaler = StandardScaler()
    scaled_lags = lag_scaler.fit_transform(df[lag_columns])
    lag_pca = PCA(n_components=variance_threshold)
    lag_pca.fit(scaled_lags)
    return lag_scaler, lag_pca


def apply_lag_pca(df: pd.DataFrame, lag_scaler: StandardScaler, lag_pca: PCA, lag_columns=LAG_COLUMNS) -> pd.DataFrame:
    """
    Replace the raw lag columns with their principal components. All other
    columns are preserved in their original order; PCA components are
    appended as lag_pca_1..lag_pca_k. Used identically at training time and
    at live inference time so the model always sees the same input shape.
    """
    scaled_lags = lag_scaler.transform(df[lag_columns])
    components = lag_pca.transform(scaled_lags)

    reduced = df.drop(columns=lag_columns).copy()
    for i in range(components.shape[1]):
        reduced[f'lag_pca_{i + 1}'] = components[:, i]
    return reduced


def model_input_columns(lag_pca: PCA, base_columns=FEATURE_COLUMNS, lag_columns=LAG_COLUMNS) -> list:
    """The exact column order produced by apply_lag_pca, for a given fitted PCA."""
    remaining = [c for c in base_columns if c not in lag_columns]
    return remaining + [f'lag_pca_{i + 1}' for i in range(lag_pca.n_components_)]