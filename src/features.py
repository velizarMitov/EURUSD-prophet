import os

import numpy as np
import pandas as pd

# Canonical 20-column feature order consumed by the trained GBM/LSTM artifacts.
# Shared by the notebook (training) and api.py/app.py (inference) so the two
# can never silently drift apart.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'tick_volume', 'log_return',
    'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200', 'volatility_20',
    'bar_dynamics', 'return_lag_1', 'dynamics_lag_1', 'return_lag_2',
    'dynamics_lag_2', 'return_lag_3', 'dynamics_lag_3', 'day_sin', 'day_cos'
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