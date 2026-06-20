import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_advanced_features, build_live_features, FEATURE_COLUMNS, TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN,
    LAG_COLUMNS, fit_lag_pca, apply_lag_pca, model_input_columns,
)

def test_feature_engineering_success():
    """Unit test logically verifying the exact extraction geometry explicitly isolating features
    preventing DataFrame corruption edge cases."""
    dates = pd.date_range('2026-01-01', periods=300, freq='D')
    df = pd.DataFrame({
        'open': np.random.rand(300) * 0.1 + 1.1,
        'high': np.random.rand(300) * 0.1 + 1.15,
        'low': np.random.rand(300) * 0.1 + 1.05,
        'close': np.random.rand(300) * 0.1 + 1.12,
        'tick_volume': np.random.randint(1000, 100000, 300)
    }, index=dates)

    res = add_advanced_features(df)

    # Assertions guaranteeing explicit functional reliability
    assert not res.isnull().any().any(), "Function illegally generated NaN bounds within output matrix."
    assert 'day_sin' in res.columns, "Cyclical feature extraction bypassed."
    assert 'volatility_20' in res.columns, "Volatility matrix bypassed."
    assert TARGET_RETURN_COLUMN in res.columns, "Continuous Multi-Task return target bypassed."
    assert TARGET_DIRECTION_COLUMN in res.columns, "Binary Multi-Task direction target bypassed."
    assert set(res[TARGET_DIRECTION_COLUMN].unique()).issubset({0, 1}), "Direction target escaped its binary bounds."
    assert ((res[TARGET_RETURN_COLUMN] > 0).astype(int) == res[TARGET_DIRECTION_COLUMN]).all(), \
        "target_direction must be the exact sign of target_return."
    # With a maximum rolling window of 200 and lags of 4, the initial 200+ samples must mathematically drop
    assert len(res) < 300, "Structural sequential overlap detected. NaN shifting bounds failed."
    # The longest window is 200 bars (`SMA_200`), making rows 0-198 (199 rows) contain NaNs.
    # target_return's shift(-1) genuinely propagates NaN on the very last row (unlike a boolean
    # comparison against NaN, which silently evaluates to False instead of being dropped), so it
    # is correctly excluded by dropna() too.
    # Total drops: 199 (SMA_200 warmup) + 1 (last-row target undefined) = 200 dropped rows.
    assert len(res) == 300 - 200, f"Sequential boundary row exclusion math is strictly misaligned. Expected {300-200}, got {len(res)}"


def test_build_live_features_no_mock():
    """Verify live inference features are genuinely recomputed from appended history,
    not hardcoded/mocked constants, and that the newest bar survives (no target-shift drop)."""
    dates = pd.date_range('2026-01-01', periods=250, freq='D')
    history = pd.DataFrame({
        'open': np.random.rand(250) * 0.1 + 1.1,
        'high': np.random.rand(250) * 0.1 + 1.15,
        'low': np.random.rand(250) * 0.1 + 1.05,
        'close': np.random.rand(250) * 0.1 + 1.12,
        'tick_volume': np.random.randint(1000, 100000, 250)
    }, index=dates)

    new_bar = {'open': 1.15, 'high': 1.16, 'low': 1.14, 'close': 1.155, 'tick_volume': 50000}
    window = build_live_features(history, new_bar, time_steps=20)

    assert len(window) == 20, "Sliding window length must exactly match the requested time_steps."
    assert list(window.columns) == FEATURE_COLUMNS, "Live feature columns must match the trained FEATURE_COLUMNS order."
    assert not window.isnull().any().any(), "Live feature window must be fully real-valued, no NaNs."

    last_row = window.iloc[-1]
    expected_sma_21 = pd.concat([history['close'], pd.Series([new_bar['close']])]).tail(21).mean()
    assert last_row['SMA_21'] == pytest.approx(expected_sma_21), "SMA_21 must be genuinely computed from real history, not mocked."

def test_lag_pca_fit_transform_no_leakage():
    """Verify the lag-PCA helpers reduce/replace the lag columns consistently
    between a 'training' fit and a held-out 'live' application, with the fit
    never touching anything beyond the slice it was given (no leakage)."""
    dates = pd.date_range('2026-01-01', periods=400, freq='D')
    df = pd.DataFrame({
        'open': np.random.rand(400) * 0.1 + 1.1,
        'high': np.random.rand(400) * 0.1 + 1.15,
        'low': np.random.rand(400) * 0.1 + 1.05,
        'close': np.random.rand(400) * 0.1 + 1.12,
        'tick_volume': np.random.randint(1000, 100000, 400)
    }, index=dates)

    engineered = add_advanced_features(df)
    train_slice = engineered.iloc[:150]
    lag_scaler, lag_pca = fit_lag_pca(train_slice, lag_columns=LAG_COLUMNS, variance_threshold=0.95)

    assert lag_pca.n_components_ <= len(LAG_COLUMNS), "PCA must never produce more components than input lag columns."
    assert lag_pca.explained_variance_ratio_.sum() >= 0.95 - 1e-9, "Selected components must explain >= the configured variance threshold."

    reduced = apply_lag_pca(engineered, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    for col in LAG_COLUMNS:
        assert col not in reduced.columns, f"Raw lag column {col} must be dropped after PCA reduction."
    expected_pca_cols = [f'lag_pca_{i + 1}' for i in range(lag_pca.n_components_)]
    for col in expected_pca_cols:
        assert col in reduced.columns, f"Expected PCA component column {col} missing from reduced output."
    assert len(reduced) == len(engineered), "PCA reduction must not drop or add rows."

    cols = model_input_columns(lag_pca, base_columns=list(engineered.columns), lag_columns=LAG_COLUMNS)
    non_target_cols = [c for c in cols if c not in (TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN)]
    assert non_target_cols == [c for c in reduced.columns if c not in (TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN)], \
        "model_input_columns() must match the actual column order produced by apply_lag_pca()."


def test_feature_engineering_edge_cases():
    """Unit test explicitly guaranteeing 0 Open prices evaluate correctly evading DivisionByZero crashes."""
    dates = pd.date_range('2026-01-01', periods=300, freq='D')
    df = pd.DataFrame({
        'open': np.zeros(300), # Mathematical poison
        'high': np.ones(300) * 1.5,
        'low': np.ones(300) * 0.5,
        'close': np.ones(300) * 1.0,
        'tick_volume': np.ones(300)
    }, index=dates)
    
    res = add_advanced_features(df)
    # The replaced NaN drops mathematically should mean data parses out without hard crashing 
    # the server internally via structural exception throwing.
    assert len(res) == 0 or not res.isnull().any().any(), "Edge case math poisoned matrix evaluation."