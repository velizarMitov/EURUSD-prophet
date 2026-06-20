import sys

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


def test_compute_consensus_agreement_averages():
    """When both models agree on direction, the consensus must average their
    confidence/return rather than just picking one arbitrarily."""
    from src.inference import PredictionService

    predictions = {
        'gbm': {'direction': 'UP', 'confidence': 0.60, 'predicted_return_pct': 0.10},
        'lstm': {'direction': 'UP', 'confidence': 0.70, 'predicted_return_pct': 0.20},
    }
    consensus = PredictionService.compute_consensus(predictions)

    assert consensus['direction'] == 'UP'
    assert consensus['agreement'] is True
    assert consensus['confidence'] == pytest.approx(0.65)
    assert consensus['predicted_return_pct'] == pytest.approx(0.15)


def test_compute_consensus_disagreement_defers_to_confident_model():
    """When models disagree on direction, the consensus must flag the
    disagreement and defer to whichever model is more confident, rather than
    silently averaging across opposite-signed predictions."""
    from src.inference import PredictionService

    predictions = {
        'gbm': {'direction': 'DOWN', 'confidence': 0.55, 'predicted_return_pct': -0.05},
        'lstm': {'direction': 'UP', 'confidence': 0.80, 'predicted_return_pct': 0.30},
    }
    consensus = PredictionService.compute_consensus(predictions)

    assert consensus['agreement'] is False
    assert consensus['direction'] == 'UP'
    assert consensus['confidence'] == pytest.approx(0.80)
    assert consensus['predicted_return_pct'] == pytest.approx(0.30)


def test_fetch_live_market_data_prefers_mt5(monkeypatch):
    """When a live MT5 terminal session is reachable, it must be used in
    preference to Yahoo Finance, per the requested MT5 -> yfinance fallback order."""
    import src.live_data as live_data

    rates = np.array(
        [(1781481600 + i * 86400, 1.10 + i * 0.001, 1.11 + i * 0.001, 1.09 + i * 0.001, 1.105 + i * 0.001, 100000 + i, 5, 0) for i in range(5)],
        dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'),
               ('tick_volume', '<i8'), ('spread', '<i4'), ('real_volume', '<i8')]
    )

    class _FakeMT5:
        TIMEFRAME_D1 = 1

        @staticmethod
        def initialize():
            return True

        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start, count):
            return rates

        @staticmethod
        def shutdown():
            pass

    monkeypatch.setitem(sys.modules, 'MetaTrader5', _FakeMT5)

    def _boom_yfinance(*args, **kwargs):
        raise AssertionError("yfinance must not be called when MT5 succeeds.")

    monkeypatch.setattr(live_data, "_fetch_from_yfinance", _boom_yfinance)

    df, source = live_data.fetch_live_market_data(bars=5)
    assert source == "MT5"
    assert len(df) == 5
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'tick_volume']


def test_fetch_live_market_data_falls_back_to_yfinance(monkeypatch):
    """When MT5 is unreachable, the pipeline must fall back to Yahoo Finance
    rather than returning nothing."""
    import src.live_data as live_data

    monkeypatch.setattr(live_data, "_fetch_from_mt5", lambda symbol, bars: None)

    fake_df = pd.DataFrame({
        'open': [1.10], 'high': [1.11], 'low': [1.09], 'close': [1.105], 'tick_volume': [0.0],
    }, index=pd.date_range('2026-06-19', periods=1))
    monkeypatch.setattr(live_data, "_fetch_from_yfinance", lambda symbol, bars: fake_df)

    df, source = live_data.fetch_live_market_data(bars=5)
    assert source == "yfinance"
    assert len(df) == 1


def test_fetch_live_market_data_returns_none_when_both_unreachable(monkeypatch):
    """When neither live source is reachable, the caller must get (None, None)
    so it can fall back to its own bundled historical data."""
    import src.live_data as live_data

    monkeypatch.setattr(live_data, "_fetch_from_mt5", lambda symbol, bars: None)
    monkeypatch.setattr(live_data, "_fetch_from_yfinance", lambda symbol, bars: None)

    df, source = live_data.fetch_live_market_data(bars=5)
    assert df is None
    assert source is None


def test_fetch_latest_bar_returns_none_on_failure(monkeypatch):
    """The automated fetch must degrade gracefully (return None, not raise)
    so callers can fall back to historical data when live data is unreachable."""
    import src.live_data as live_data

    class _BoomTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            raise ConnectionError("simulated network failure")

    monkeypatch.setattr(live_data.yf, "Ticker", _BoomTicker)
    assert live_data.fetch_latest_bar("EURUSD=X") is None


def test_fetch_latest_bar_parses_successful_response(monkeypatch):
    """Verify a successful fetch is parsed into the expected dict shape without hitting the network."""
    import src.live_data as live_data

    fake_history = pd.DataFrame({
        'Open': [1.10, 1.11],
        'High': [1.12, 1.13],
        'Low': [1.09, 1.10],
        'Close': [1.105, 1.115],
        'Volume': [0, 0],
    }, index=pd.date_range('2026-06-18', periods=2, freq='D'))

    class _FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return fake_history

    monkeypatch.setattr(live_data.yf, "Ticker", _FakeTicker)
    result = live_data.fetch_latest_bar("EURUSD=X")

    assert result['date'] == '2026-06-19'
    assert result['close'] == pytest.approx(1.115)
    assert result['tick_volume'] == 0.0


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