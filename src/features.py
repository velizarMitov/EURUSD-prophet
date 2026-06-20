import numpy as np
import pandas as pd

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracted purely functional processing logic for production API and Unit Testing.
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

    # 7. Target Mapping (for training only; production ignores this target col post-processing)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Clean missing values systematically
    data.dropna(inplace=True)
    return data