import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Canonical 24-column feature order consumed by the trained GBM/LSTM artifacts.
# Shared by the notebook (training) and api.py/app.py (inference) so the two
# can never silently drift apart.
#
# tick_volume is deliberately excluded: MT5's "tick_volume" is a broker-specific
# tick count, not genuine traded volume, and the bundled 1971-2026 training
# history carries decades of degenerate placeholder values (e.g. 1) from its
# earliest synthetic era. That contaminated the fitted StandardScaler's
# mean/std for the column, which then caused the LSTM to extrapolate wildly
# on genuine live volumes (8+ standard deviations out). It is still loaded
# and surfaced to the UI for display purposes (see load_history/build_live_features
# callers' bar_used dicts), it just never reaches the models.
#
# yield_differential (US 10Y - DE 10Y bond yield, via src/macro_data.py) is a
# passthrough column like tick_volume: it must already be present on the
# DataFrame passed into compute_features()/add_advanced_features() -- see
# merge_macro_features() below -- since it is exogenous macro data, not
# something derivable from OHLCV alone.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'log_return',
    'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200', 'volatility_20',
    'bar_dynamics', 'return_lag_1', 'dynamics_lag_1', 'return_lag_2',
    'dynamics_lag_2', 'return_lag_3', 'dynamics_lag_3', 'day_sin', 'day_cos',
    'month_sin', 'month_cos', 'ATR_14', 'BB_width', 'yield_differential',
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
    Compute the 24 FEATURE_COLUMNS from raw OHLCV (+ yield_differential, if
    already merged in via merge_macro_features). Unlike add_advanced_features,
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

    # 6. Cyclical Encoding (day of week + month, both periodic -- sin/cos
    # preserves the wrap-around geometry, e.g. Friday->Monday, Dec->Jan)
    day_of_week = data.index.dayofweek
    data['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    data['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    month = data.index.month
    data['month_sin'] = np.sin(2 * np.pi * month / 12)
    data['month_cos'] = np.cos(2 * np.pi * month / 12)

    # 7. Average True Range (vectorized, 14-period EWM smoothing)
    prev_close = data['close'].shift(1)
    true_range = pd.concat([
        data['high'] - data['low'],
        (data['high'] - prev_close).abs(),
        (data['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    data['ATR_14'] = true_range.ewm(com=13, adjust=False).mean()

    # 8. Normalized Bollinger Band Width: (upper - lower) / mid = 4*std/mid
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['BB_width'] = (4 * bb_std) / bb_mid

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


def merge_macro_features(ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the (already UTC-indexed) yield_differential series from
    src/macro_data.py onto the OHLCV index, then forward-fill any remaining
    gaps -- weekend FX bars and bond-market holidays inherit the last known
    differential rather than NaN. ffill only ever carries a *past* value
    forward (never a future one backward), so this introduces zero look-ahead
    bias regardless of how the two calendars are offset.
    """
    original_index = ohlcv_df.index
    utc_index = (
        original_index.tz_localize('UTC') if original_index.tz is None
        else original_index.tz_convert('UTC')
    )
    merged = ohlcv_df.copy()
    merged.index = utc_index
    merged = merged.join(macro_df[['yield_differential']], how='left')
    merged['yield_differential'] = merged['yield_differential'].ffill()
    merged.index = original_index
    return merged


def load_history(path: str = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Load the real historical OHLCV bars used as live feature context."""
    history = pd.read_csv(path, index_col='time', parse_dates=True)
    return history[['open', 'high', 'low', 'close', 'tick_volume']]


def build_live_features(history: pd.DataFrame, new_bar: dict, time_steps: int) -> pd.DataFrame:
    """
    Append `new_bar` (a dict with open/high/low/close/tick_volume/
    yield_differential) as the next chronological bar after `history` (which
    must carry the same columns, including a pre-merged yield_differential --
    see merge_macro_features), recompute the real rolling indicators over
    genuine price history (no hardcoded/mocked SMA or lag values), and return
    the last `time_steps` engineered rows ending at the new bar.
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