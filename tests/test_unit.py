import sys

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_advanced_features, build_live_features, merge_macro_features, FEATURE_COLUMNS, TARGET_RETURN_COLUMN,
    TARGET_DIRECTION_COLUMN, LAG_COLUMNS, fit_lag_pca, apply_lag_pca, model_input_columns,
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
        'tick_volume': np.random.randint(1000, 100000, 300),
        'yield_differential': np.random.uniform(-1.0, 3.0, 300),
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
        'tick_volume': np.random.randint(1000, 100000, 250),
        'yield_differential': np.random.uniform(-1.0, 3.0, 250),
    }, index=dates)

    new_bar = {'open': 1.15, 'high': 1.16, 'low': 1.14, 'close': 1.155, 'tick_volume': 50000, 'yield_differential': 1.8}
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
        'tick_volume': np.random.randint(1000, 100000, 400),
        'yield_differential': np.random.uniform(-1.0, 3.0, 400),
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


def test_merge_macro_features_no_lookahead_on_weekend_gap():
    """A Saturday/Sunday FX bar must inherit Friday's differential via ffill,
    never a future Monday value -- this is the look-ahead guard the
    macro merge is built around."""
    ohlcv_dates = pd.to_datetime(['2026-06-19', '2026-06-20', '2026-06-21', '2026-06-22'])  # Fri, Sat, Sun, Mon
    ohlcv = pd.DataFrame({
        'open': [1.1] * 4, 'high': [1.1] * 4, 'low': [1.1] * 4, 'close': [1.1] * 4, 'tick_volume': [1] * 4,
    }, index=ohlcv_dates)

    macro = pd.DataFrame(
        {'yield_differential': [1.50, 1.80]},
        index=pd.DatetimeIndex(['2026-06-19', '2026-06-22'], tz='UTC'),
    )

    merged = merge_macro_features(ohlcv, macro)
    assert merged.loc[ohlcv_dates[0], 'yield_differential'] == pytest.approx(1.50)
    assert merged.loc[ohlcv_dates[1], 'yield_differential'] == pytest.approx(1.50), "Saturday must inherit Friday's value."
    assert merged.loc[ohlcv_dates[2], 'yield_differential'] == pytest.approx(1.50), "Sunday must inherit Friday's value."
    assert merged.loc[ohlcv_dates[3], 'yield_differential'] == pytest.approx(1.80), "Monday must use its own value, not Friday's."
    assert list(merged.index) == list(ohlcv_dates), "merge_macro_features must preserve the caller's original index labels."


def test_fetch_yield_differential_prefers_fred_api(monkeypatch, tmp_path):
    """When the official FRED API (via FRED_API_KEY) succeeds, the free
    public-CSV fallback must not be called at all."""
    import src.macro_data as macro_data

    fake_df = pd.DataFrame({'yield_differential': [1.5, 1.6]}, index=pd.date_range('2026-06-01', periods=2, tz='UTC'))
    monkeypatch.setattr(macro_data, '_fetch_via_fredapi', lambda series_ids, start, end: fake_df)

    def _boom(*args, **kwargs):
        raise AssertionError("pandas_datareader must not be called when the FRED API succeeds.")
    monkeypatch.setattr(macro_data, '_fetch_via_pandas_datareader', _boom)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-02', cache_path=str(tmp_path / 'cache.csv'))
    assert source == 'FRED_api'
    assert len(df) == 2


def test_fetch_yield_differential_falls_back_to_public_endpoint(monkeypatch, tmp_path):
    """When no FRED_API_KEY is configured (or the official API fails), fall
    back to FRED's public CSV endpoint, which needs no key."""
    import src.macro_data as macro_data

    monkeypatch.setattr(macro_data, '_fetch_via_fredapi', lambda *args, **kwargs: None)
    fake_df = pd.DataFrame({'yield_differential': [2.0]}, index=pd.date_range('2026-06-01', periods=1, tz='UTC'))
    monkeypatch.setattr(macro_data, '_fetch_via_pandas_datareader', lambda *args, **kwargs: fake_df)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=str(tmp_path / 'cache.csv'))
    assert source == 'FRED_public'
    assert len(df) == 1


def test_fetch_yield_differential_falls_back_to_cache_when_unreachable(monkeypatch, tmp_path):
    """When neither live FRED source is reachable, reuse the last cached
    snapshot on disk rather than failing the whole prediction pipeline."""
    import src.macro_data as macro_data

    cache_path = str(tmp_path / 'cache.csv')
    cached_df = pd.DataFrame({'yield_differential': [0.9]}, index=pd.date_range('2026-05-01', periods=1, tz='UTC'))
    cached_df.to_csv(cache_path)

    monkeypatch.setattr(macro_data, '_fetch_via_fredapi', lambda *args, **kwargs: None)
    monkeypatch.setattr(macro_data, '_fetch_via_pandas_datareader', lambda *args, **kwargs: None)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=cache_path)
    assert source == 'cache'
    assert df['yield_differential'].iloc[0] == pytest.approx(0.9)


def test_fetch_yield_differential_returns_none_when_nothing_reachable(monkeypatch, tmp_path):
    """With no live source and no cache file at all, the caller must get
    (None, None) so it can apply its own constant-default fallback."""
    import src.macro_data as macro_data

    monkeypatch.setattr(macro_data, '_fetch_via_fredapi', lambda *args, **kwargs: None)
    monkeypatch.setattr(macro_data, '_fetch_via_pandas_datareader', lambda *args, **kwargs: None)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=str(tmp_path / 'missing.csv'))
    assert df is None
    assert source is None


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
        'tick_volume': np.ones(300),
        'yield_differential': np.ones(300) * 1.5,
    }, index=dates)
    
    res = add_advanced_features(df)
    # The replaced NaN drops mathematically should mean data parses out without hard crashing
    # the server internally via structural exception throwing.
    assert len(res) == 0 or not res.isnull().any().any(), "Edge case math poisoned matrix evaluation."


def _fake_predict_result(as_of, forecast, close, direction, ret):
    """Minimal PredictionService.predict()-shaped dict for tracking tests."""
    return {
        'as_of_date': as_of, 'forecasting_date': forecast,
        'bar_used': {'close': close},
        'gbm': {'direction': direction}, 'lstm': {'direction': direction},
        'consensus': {'direction': direction, 'predicted_return_pct': ret, 'confidence': 0.55},
    }


def test_log_prediction_is_idempotent_per_day(tmp_path):
    """Re-logging the same as_of_date must replace that day's row, not duplicate
    it, so the comparison log carries exactly one forecast per trading day."""
    from src.tracking import log_prediction
    log = str(tmp_path / 'log.csv')

    log_prediction(_fake_predict_result('2026-06-19', '2026-06-22', 1.146, 'DOWN', -0.02), log)
    log_prediction(_fake_predict_result('2026-06-19', '2026-06-22', 1.146, 'UP', +0.03), log)  # same day, re-run

    rows = pd.read_csv(log)
    assert len(rows) == 1, "Same-day re-prediction must overwrite, not append."
    assert rows.iloc[0]['pred_direction'] == 'UP', "Latest forecast for the day must win."


def test_build_history_html_scores_against_actual(tmp_path, monkeypatch):
    """A logged forecast whose forecast date has closed must be scored UP/DOWN
    against the realised return and marked correct/wrong; an unresolved one
    stays pending."""
    import src.tracking as tracking
    log = str(tmp_path / 'log.csv')
    # Predicted UP from a 1.1000 close, forecasting 2026-06-22.
    tracking.log_prediction(_fake_predict_result('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05), log)
    # ...and a future-dated one that cannot be resolved yet.
    tracking.log_prediction(_fake_predict_result('2026-06-22', '2026-06-23', 1.1050, 'DOWN', -0.05), log)

    # Realised market: 2026-06-22 actually closed UP at 1.1080 (so the UP call was correct).
    actual = pd.DataFrame({'close': [1.1080]}, index=pd.DatetimeIndex(['2026-06-22']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    html = tracking.build_history_html(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'})
    assert 'correct' in html and "class='hit'" in html, "Correct UP call must be scored as a hit."
    assert 'pending' in html, "An unresolved future forecast must render as pending."
    assert '1/1 resolved' in html or '100%' in html, "Hit-rate summary must reflect the single resolved row."