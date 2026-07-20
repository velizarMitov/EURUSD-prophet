import sys

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_advanced_features, build_live_features, merge_macro_features, compute_features, FEATURE_COLUMNS,
    TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN, TARGET_VOLATILITY_COLUMN,
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


def test_yield_differential_delta_no_lookahead_on_weekend_gap():
    """The MODEL-facing feature is yield_differential_delta = diff(1) of the
    (already ffilled) level, computed in compute_features. A Saturday/Sunday
    delta must be derived only from Friday's ffilled level -- never from
    Monday's future print -- exactly the same no-look-ahead guarantee the raw
    level enjoys via merge_macro_features's ffill."""
    ohlcv_dates = pd.to_datetime(['2026-06-18', '2026-06-19', '2026-06-20', '2026-06-21', '2026-06-22'])  # Thu..Mon
    ohlcv = pd.DataFrame({
        'open': [1.1] * 5, 'high': [1.1] * 5, 'low': [1.1] * 5, 'close': [1.1] * 5, 'tick_volume': [1] * 5,
    }, index=ohlcv_dates)

    macro = pd.DataFrame(
        {'yield_differential': [1.40, 1.50, 1.80]},
        index=pd.DatetimeIndex(['2026-06-18', '2026-06-19', '2026-06-22'], tz='UTC'),
    )

    merged = merge_macro_features(ohlcv, macro)
    engineered = compute_features(merged)
    delta = engineered['yield_differential_delta']

    assert delta.loc[ohlcv_dates[1]] == pytest.approx(0.10), "Friday: 1.50 - 1.40 (own value vs prior day)."
    assert delta.loc[ohlcv_dates[2]] == pytest.approx(0.0), \
        "Saturday must diff against Friday's ffilled level (1.50-1.50=0), never Monday's future 1.80."
    assert delta.loc[ohlcv_dates[3]] == pytest.approx(0.0), \
        "Sunday must diff against Saturday's ffilled level (1.50-1.50=0), never Monday's future 1.80."
    assert delta.loc[ohlcv_dates[4]] == pytest.approx(0.30), "Monday: 1.80 - 1.50 (own new value vs weekend ffill)."


def _raw_yield_levels(vals_us, vals_de, start='2026-06-01'):
    """A raw-level (us10y/de10y) frame shaped exactly like the generalized
    low-level FRED fetchers now return -- combine derives yield_differential."""
    idx = pd.date_range(start, periods=len(vals_us), tz='UTC')
    return pd.DataFrame({'us10y': vals_us, 'de10y': vals_de}, index=idx)


def _weekday_ohlcv(dates):
    return pd.DataFrame(
        {'open': [1.1] * len(dates), 'high': [1.1] * len(dates), 'low': [1.1] * len(dates),
         'close': [1.1] * len(dates), 'tick_volume': [1] * len(dates)},
        index=pd.to_datetime(dates),
    )


def test_usd_index_return_no_lookahead_and_flat_on_weekend():
    """usd_index_return = log-return of the ffilled USD index level. A weekend
    bar inherits Friday's level (flat 0 return), never Monday's future level;
    Monday's return is measured against the ffilled Friday value."""
    dates = ['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06']  # Thu..Mon (post-2006)
    ohlcv = _weekday_ohlcv(dates)
    macro = pd.DataFrame({'usd_index': [100.0, 110.0, 121.0]},
                         index=pd.DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-06'], tz='UTC'))

    eng = compute_features(merge_macro_features(ohlcv, macro))
    r = eng['usd_index_return']
    d = pd.to_datetime(dates)
    assert r.loc[d[1]] == pytest.approx(np.log(110 / 100)), "Fri: log(110/100) vs Thu."
    assert r.loc[d[2]] == pytest.approx(0.0), "Sat inherits Fri's level -> flat 0, no Monday leak."
    assert r.loc[d[3]] == pytest.approx(0.0), "Sun inherits Fri's level -> flat 0."
    assert r.loc[d[4]] == pytest.approx(np.log(121 / 110)), "Mon: log(121/110) vs ffilled Fri, not vs Sun."


def test_usd_index_return_flat_zero_before_series_start():
    """DTWEXBGS only exists from 2006; earlier rows (level NaN) must get a flat 0
    return via fillna, NOT be dropped -- that is what preserves the 1999+ history
    the other macro features cover."""
    dates = ['1999-06-01', '1999-06-02', '1999-06-03']  # pre-2006, no USD level
    ohlcv = _weekday_ohlcv(dates)
    macro = pd.DataFrame({'usd_index': [np.nan]}, index=pd.DatetimeIndex(['1999-06-01'], tz='UTC'))
    eng = compute_features(merge_macro_features(ohlcv, macro))
    assert (eng['usd_index_return'] == 0.0).all(), "Pre-series-start USD return must be a flat 0, not NaN."


def test_policy_rate_differential_ffill_no_lookahead():
    """policy_rate_differential passes through as a level; a weekend/holiday bar
    inherits the last known (Friday) rate differential, never Monday's future one."""
    dates = ['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06']  # Thu..Mon
    ohlcv = _weekday_ohlcv(dates)
    macro = pd.DataFrame({'policy_rate_differential': [1.0, 1.5, 2.0]},
                         index=pd.DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-06'], tz='UTC'))
    eng = compute_features(merge_macro_features(ohlcv, macro))
    p, d = eng['policy_rate_differential'], pd.to_datetime(dates)
    assert p.loc[d[1]] == pytest.approx(1.5), "Fri: own value."
    assert p.loc[d[2]] == pytest.approx(1.5), "Sat inherits Fri, never Monday's future 2.0."
    assert p.loc[d[3]] == pytest.approx(1.5), "Sun inherits Fri."
    assert p.loc[d[4]] == pytest.approx(2.0), "Mon: own new value."


def test_inflation_differential_monthly_ffill_onto_daily_no_lookahead():
    """A monthly CPI-YoY differential must propagate onto EVERY later daily bar
    via as-of ffill -- including when the month-start print lands on a non-trading
    day -- and a mid-month day must carry THIS month's value, never next month's."""
    dates = ['2020-01-15', '2020-01-31', '2020-02-03', '2020-02-14']  # daily business days
    ohlcv = _weekday_ohlcv(dates)
    # Month-start prints; 2020-02-01 is a Saturday (non-trading) on purpose.
    macro = pd.DataFrame({'inflation_differential': [2.0, 2.5]},
                         index=pd.DatetimeIndex(['2020-01-01', '2020-02-01'], tz='UTC'))
    eng = compute_features(merge_macro_features(ohlcv, macro))
    infl, d = eng['inflation_differential'], pd.to_datetime(dates)
    assert infl.loc[d[0]] == pytest.approx(2.0), "Jan 15 carries Jan's 2.0 (not future Feb 2.5)."
    assert infl.loc[d[1]] == pytest.approx(2.0), "Jan 31 still Jan's 2.0."
    assert infl.loc[d[2]] == pytest.approx(2.5), "Feb 3 carries Feb's 2.5 even though Feb 1 was a Saturday."
    assert infl.loc[d[3]] == pytest.approx(2.5), "Feb 14 still Feb's 2.5."


def test_fetch_yield_differential_prefers_fred_api(monkeypatch, tmp_path):
    """When the official FRED API (via FRED_API_KEY) succeeds, the free
    public-CSV fallback must not be called at all. The low-level fetcher now
    returns RAW LEVELS; fetch_yield_differential's combine derives the spread."""
    import src.macro_data as macro_data

    raw = _raw_yield_levels([4.5, 4.6], [3.0, 3.0])  # -> yield_differential [1.5, 1.6]
    monkeypatch.setattr(macro_data, '_fetch_raw_via_api', lambda series_ids, start, end: raw)

    def _boom(*args, **kwargs):
        raise AssertionError("public endpoint must not be called when the FRED API succeeds.")
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public', _boom)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-02', cache_path=str(tmp_path / 'cache.csv'))
    assert source == 'FRED_api'
    assert len(df) == 2
    assert df['yield_differential'].tolist() == pytest.approx([1.5, 1.6])


def test_fetch_yield_differential_falls_back_to_public_endpoint(monkeypatch, tmp_path):
    """When no FRED_API_KEY is configured (or the official API fails), fall
    back to FRED's public CSV endpoint, which needs no key."""
    import src.macro_data as macro_data

    monkeypatch.setattr(macro_data, '_fetch_raw_via_api', lambda *args, **kwargs: None)
    raw = _raw_yield_levels([5.0], [3.0])  # -> yield_differential [2.0]
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public', lambda *args, **kwargs: raw)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=str(tmp_path / 'cache.csv'))
    assert source == 'FRED_public'
    assert len(df) == 1
    assert df['yield_differential'].iloc[0] == pytest.approx(2.0)


def test_fetch_yield_differential_falls_back_to_cache_when_unreachable(monkeypatch, tmp_path):
    """When neither live FRED source is reachable, reuse the last cached
    snapshot on disk rather than failing the whole prediction pipeline. The
    cache stores the already-combined frame, so it is returned as-is."""
    import src.macro_data as macro_data

    cache_path = str(tmp_path / 'cache.csv')
    cached_df = pd.DataFrame({'us10y': [3.9], 'de10y': [3.0], 'yield_differential': [0.9]},
                             index=pd.date_range('2026-05-01', periods=1, tz='UTC'))
    cached_df.to_csv(cache_path)

    monkeypatch.setattr(macro_data, '_fetch_raw_via_api', lambda *args, **kwargs: None)
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public', lambda *args, **kwargs: None)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=cache_path)
    assert source == 'cache'
    assert df['yield_differential'].iloc[0] == pytest.approx(0.9)


def test_fetch_yield_differential_returns_none_when_nothing_reachable(monkeypatch, tmp_path):
    """With no live source and no cache file at all, the caller must get
    (None, None) so it can apply its own constant-default fallback."""
    import src.macro_data as macro_data

    monkeypatch.setattr(macro_data, '_fetch_raw_via_api', lambda *args, **kwargs: None)
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public', lambda *args, **kwargs: None)

    df, source = macro_data.fetch_yield_differential('2026-06-01', '2026-06-01', cache_path=str(tmp_path / 'missing.csv'))
    assert df is None
    assert source is None


def test_fetch_fred_feature_generic_fallback_chain(monkeypatch, tmp_path):
    """The generalized fetch_fred_feature drives the SAME API -> public -> cache
    chain for ANY feature, not just the yield differential -- parametrized here
    over a 2-series 'policy' combine to prove the reuse, so the chain logic is
    tested once instead of duplicated per feature."""
    import src.macro_data as macro_data

    def _raw_policy(vals_fed, vals_ecb, start='2026-06-01'):
        idx = pd.date_range(start, periods=len(vals_fed), tz='UTC')
        return pd.DataFrame({'fed': vals_fed, 'ecb': vals_ecb}, index=idx)

    # Tier 1: API succeeds -> public must not run.
    monkeypatch.setattr(macro_data, '_fetch_raw_via_api',
                        lambda *a, **k: _raw_policy([4.5, 4.5], [2.25, 2.5]))
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("public must not run")))
    df, source = macro_data.fetch_fred_feature(
        {'fed': 'DFF', 'ecb': 'ECBDFR'}, '2026-06-01', '2026-06-02',
        combine=macro_data._combine_policy, cache_path=str(tmp_path / 'pol.csv'))
    assert source == 'FRED_api'
    assert df['policy_rate_differential'].tolist() == pytest.approx([2.25, 2.0])

    # Tier 3: both live tiers fail -> the just-written cache answers.
    monkeypatch.setattr(macro_data, '_fetch_raw_via_api', lambda *a, **k: None)
    monkeypatch.setattr(macro_data, '_fetch_raw_via_public', lambda *a, **k: None)
    df2, source2 = macro_data.fetch_fred_feature(
        {'fed': 'DFF', 'ecb': 'ECBDFR'}, '2026-06-01', '2026-06-02',
        combine=macro_data._combine_policy, cache_path=str(tmp_path / 'pol.csv'))
    assert source2 == 'cache'
    assert df2['policy_rate_differential'].iloc[0] == pytest.approx(2.25)


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


def test_fetch_bar_count_counts_actual_sundays_in_window():
    """The over-fetch margin must be an exact count of the Sundays inside the
    real lookback window (plus 1 for today's forming bar), not a flat
    empirical percentage -- so it must match a hand-computed reference count."""
    from src.inference import PredictionService

    bars_needed = 250
    now = pd.Timestamp('2026-06-25 11:00')
    fetch_bars = PredictionService._fetch_bar_count(bars_needed, now=now)

    HOLIDAY_PAD_DAYS = 10
    lookback_days = -(-bars_needed * 7 // 5) + HOLIDAY_PAD_DAYS
    window = pd.date_range(now.normalize() - pd.Timedelta(days=lookback_days), now.normalize(), freq='D')
    expected_sundays = int((window.weekday == 6).sum())

    assert fetch_bars == bars_needed + expected_sundays + 1
    # Sanity bound: must be enough to actually contain bars_needed weekday
    # bars once Sundays and today's forming bar are stripped, with no excess
    # waste from a flat multiplier.
    assert bars_needed < fetch_bars < bars_needed * 1.3


def test_drop_incomplete_bars_strips_forming_day_and_weekend():
    """The live feed must be reduced to fully-closed weekday sessions before it
    can set as_of_date: today's still-forming bar and MT5's partial weekend bar
    are both out-of-distribution (training history skips weekends and never
    contains a forming bar) and must be dropped."""
    from src.inference import PredictionService

    # Fri 19.06, Sat 20.06, Sun 21.06 (MT5 partial), Mon 22.06 (forming "now").
    idx = pd.to_datetime(['2026-06-19', '2026-06-20', '2026-06-21', '2026-06-22'])
    df = pd.DataFrame({
        'open': [1.10, 1.11, 1.115, 1.12],
        'high': [1.12, 1.12, 1.117, 1.13],
        'low': [1.09, 1.10, 1.113, 1.11],
        'close': [1.115, 1.118, 1.116, 1.125],
        'tick_volume': [50000, 1000, 800, 30000],
    }, index=idx)

    # Asking on Monday morning -> last completed bar must be Friday 19.06,
    # not Monday's forming bar nor the partial Sunday bar.
    trimmed = PredictionService._drop_incomplete_bars(df, now=pd.Timestamp('2026-06-22 09:00'))

    assert list(trimmed.index) == [pd.Timestamp('2026-06-19')], \
        "Must keep only the last completed weekday bar (Friday), dropping Sat/Sun/forming-Monday."


def test_drop_incomplete_bars_strips_only_current_intraday_session():
    """A weekday fetch at 11:00 must drop the still-forming current-day bar and
    settle on the previous trading day as the t+1 base (so the forecast targets
    the current, not the next, day)."""
    from src.inference import PredictionService

    # Tue..Thu, with Thu (23.04) still forming at 11:00.
    idx = pd.to_datetime(['2026-04-21', '2026-04-22', '2026-04-23'])
    df = pd.DataFrame({
        'open': [1.10, 1.11, 1.12],
        'high': [1.12, 1.12, 1.13],
        'low': [1.09, 1.10, 1.11],
        'close': [1.115, 1.118, 1.125],
        'tick_volume': [50000, 40000, 12000],
    }, index=idx)

    trimmed = PredictionService._drop_incomplete_bars(df, now=pd.Timestamp('2026-04-23 11:00'))

    assert list(trimmed.index) == [pd.Timestamp('2026-04-21'), pd.Timestamp('2026-04-22')], \
        "The forming current-day (23.04) bar must be excluded; 22.04 is the last completed bar."


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


def _synthetic_h1(day_closes, hours=24, start='2026-03-02'):
    """Build a UTC-indexed H1 OHLCV frame of len(day_closes) days x `hours`
    bars, where each day's LAST hourly close is exactly day_closes[i] -- so the
    daily close series is known and the shift(-1) target can be checked by hand."""
    frames = []
    days = pd.date_range(start, periods=len(day_closes), freq='D', tz='UTC')
    for d, close in zip(days, day_closes):
        idx = pd.date_range(d, periods=hours, freq='h', tz='UTC')
        c = np.linspace(close - 0.002, close, hours)  # ramp that ends exactly at `close`
        frames.append(pd.DataFrame({
            'open': c, 'high': c + 0.001, 'low': c - 0.001, 'close': c,
            'tick_volume': np.arange(1, hours + 1) * 10.0,
        }, index=idx))
    return pd.concat(frames)


def test_h1_target_is_strict_next_day_shift_no_lookahead():
    """The H1->Daily target must be exactly the t+1 daily return via shift(-1):
    row t carries the return realised on t+1, the final (unresolved) day is NaN,
    and build_h1_datasets drops it from every aligned representation."""
    from src.h1_features import aggregate_daily_features, build_daily_target, build_h1_datasets

    closes = [1.10, 1.11, 1.09, 1.12, 1.13]
    h1 = _synthetic_h1(closes)
    feats, daily_close = aggregate_daily_features(h1)
    target = build_daily_target(daily_close)

    # The recovered daily close series must equal the known per-day last closes.
    assert list(np.round(daily_close.values, 5)) == closes

    # target[t] == 100*ln(close[t+1]/close[t]) -- strictly the *next* day's return.
    for t in range(len(closes) - 1):
        expected = 100 * np.log(closes[t + 1] / closes[t])
        assert target.iloc[t] == pytest.approx(expected), f"target[{t}] must be the t+1 return."
    # No future beyond the last day -> its shifted target must be undefined.
    assert np.isnan(target.iloc[-1]), "Last day's shift(-1) target must be NaN (nothing to leak)."

    # The unresolved last row must be dropped from ALL representations, aligned.
    Xf, Xs, yr, yd, idx = build_h1_datasets(h1=h1)
    assert len(Xf) == len(closes) - 1 == len(yr) == Xs.shape[0] == len(idx)
    assert not np.isnan(yr).any(), "Resolved targets must contain no NaN after the last-day drop."


def test_target_volatility_is_next_day_abs_log_return_no_lookahead():
    """target_volatility_pct[t] must be exactly |ln(close[t+1]/close[t])| * 100 --
    strictly the NEXT day's realized absolute log return via the same shift(-1)
    geometry as target_return, with the final (unresolved) bar dropped."""
    dates = pd.date_range('2026-01-01', periods=300, freq='D')
    rng = np.random.default_rng(7)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.004, 300)), index=dates)
    df = pd.DataFrame({
        'open': close.values, 'high': close.values + 0.002,
        'low': close.values - 0.002, 'close': close.values,
        'tick_volume': np.full(300, 1000),
        'yield_differential': np.full(300, 1.5),
    }, index=dates)

    res = add_advanced_features(df)
    assert TARGET_VOLATILITY_COLUMN in res.columns, "Volatility target bypassed."

    # Hand-check every surviving row against the raw close series: the value at
    # date d must use ONLY d and d+1 -- nothing further into the future.
    for d in res.index:
        pos = dates.get_loc(d)
        expected = abs(np.log(close.iloc[pos + 1] / close.iloc[pos])) * 100
        assert res.loc[d, TARGET_VOLATILITY_COLUMN] == pytest.approx(expected), \
            f"target_volatility_pct at {d} must be the |t+1 log return| * 100."

    # Nonnegative by construction, and the last raw bar (undefined target) must
    # have been dropped -- same boundary discipline as target_return.
    assert (res[TARGET_VOLATILITY_COLUMN] >= 0).all(), "Realized volatility can never be negative."
    assert dates[-1] not in res.index, "Last bar's shift(-1) volatility target is undefined and must be dropped."


def test_volatility_sequence_alignment_no_lookahead():
    """src/volatility.py::make_sequences must end each window AT the target row
    (rows [t-time_steps+1 : t] predict y[t]) -- the window may never include
    row t+1, whose return IS the target."""
    from src.volatility import make_sequences

    n, time_steps = 30, 5
    X = np.arange(n, dtype=float).reshape(-1, 1)  # row t has value t
    y = np.arange(n, dtype=float) * 10

    windows, targets, target_rows = make_sequences(X, y, time_steps)
    assert windows.shape == (n - time_steps + 1, time_steps, 1)
    for k, t in enumerate(target_rows):
        assert targets[k] == y[t], "Target must be the window's own end row's y."
        assert windows[k, -1, 0] == float(t), "Window must END at the target row t."
        assert windows[k, 0, 0] == float(t - time_steps + 1), "Window must start time_steps-1 rows back."
        assert (windows[k, :, 0] <= t).all(), "No window row may lie beyond t (look-ahead)."


def test_h1_features_do_not_depend_on_future_days():
    """A day's flattened feature row must be identical whether or not later days
    exist in the frame -- a strong guarantee that features never peek forward."""
    from src.h1_features import aggregate_daily_features

    closes = [1.10, 1.11, 1.09, 1.12, 1.13]
    h1_full = _synthetic_h1(closes)
    feats_full, _ = aggregate_daily_features(h1_full)

    cutoff = feats_full.index[2] + pd.Timedelta(days=1)   # keep days 0..2 only
    feats_trunc, _ = aggregate_daily_features(h1_full[h1_full.index < cutoff])

    pd.testing.assert_series_equal(feats_full.iloc[1], feats_trunc.iloc[1])


def _synthetic_daily_ohlcv(n=120, seed=11):
    """Random-walk daily OHLCV frame for the volatility candidate-feature tests."""
    dates = pd.date_range('2026-01-01', periods=n, freq='D')
    rng = np.random.default_rng(seed)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.004, n)), index=dates)
    return pd.DataFrame({
        'open': close.values, 'high': close.values + 0.002,
        'low': close.values - 0.002, 'close': close.values,
        'tick_volume': np.full(n, 1000),
    }, index=dates), close


def test_volatility_candidate_rsi14_formula_and_no_lookahead():
    """RSI_14 (volatility-model candidate ONLY) must be literally the H1
    module's _rsi formula applied to the DAILY close with period=14, and a
    trailing-window computation: truncating the future must not change the
    past values."""
    from src.volatility import add_volatility_candidate_features
    from src.h1_features import _rsi

    df, close = _synthetic_daily_ohlcv()
    res = add_volatility_candidate_features(df)

    expected = _rsi(close, 14)
    pd.testing.assert_series_equal(res['RSI_14'], expected,
                                   check_names=False)
    assert res['RSI_14'].dropna().between(0.0, 100.0).all(), \
        "RSI must live in [0, 100]."

    # Past-only windows: dropping the future leaves earlier values untouched.
    res_trunc = add_volatility_candidate_features(df.iloc[:80])
    pd.testing.assert_series_equal(res['RSI_14'].iloc[:80], res_trunc['RSI_14'],
                                   check_names=False)

    # Flat window -> the same neutral-50 convention as the H1 rsi_24.
    flat = df.copy()
    flat[['open', 'high', 'low', 'close']] = 1.10
    res_flat = add_volatility_candidate_features(flat)
    assert (res_flat['RSI_14'].iloc[20:] == 50.0).all(), \
        "A flat window must map to neutral RSI 50, mirroring _rsi's convention."


def test_volatility_candidate_bb_percent_b_consistent_with_bb_width_no_lookahead():
    """BB_percent_b must be built from the SAME 20-day rolling mean/std that
    BB_width uses (upper/lower = mid ± 2σ) -- never a second, inconsistent
    Bollinger computation -- and must be a strictly trailing-window feature."""
    from src.volatility import add_volatility_candidate_features

    df, close = _synthetic_daily_ohlcv()
    res = add_volatility_candidate_features(df)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    expected_pb = (close - (mid - 2 * std)) / (4 * std)
    pd.testing.assert_series_equal(res['BB_percent_b'], expected_pb,
                                   check_names=False)

    # Consistency with the shipped BB_width on identical rows: BB_width is
    # (upper - lower) / mid over the same components, so reconstructing the
    # band range two ways must agree: BB_width * mid == 4σ (the %B denominator).
    feats = compute_features(df)
    valid = std.notna() & (std > 0)
    np.testing.assert_allclose((feats['BB_width'] * mid)[valid], (4 * std)[valid],
                               rtol=1e-12)

    # Past-only windows: truncating the future leaves earlier values untouched.
    res_trunc = add_volatility_candidate_features(df.iloc[:80])
    pd.testing.assert_series_equal(res['BB_percent_b'].iloc[:80],
                                   res_trunc['BB_percent_b'], check_names=False)

    # Degenerate flat window (σ=0) -> neutral mid-band 0.5, not NaN/inf.
    flat = df.copy()
    flat[['open', 'high', 'low', 'close']] = 1.10
    res_flat = add_volatility_candidate_features(flat)
    assert (res_flat['BB_percent_b'].iloc[20:] == 0.5).all(), \
        "A zero-width band must map to the neutral mid-band 0.5."


def test_volatility_candidates_stay_out_of_direction_return_models():
    """The Ch.11 diagnostic closed the direction/return capacity question:
    RSI_14 / BB_percent_b are volatility-model candidates ONLY and must never
    leak into FEATURE_COLUMNS (which both daily variants derive from). The
    FOMC calendar trio is likewise candidate-only until a hypothesis test
    clears its family bar."""
    from src.volatility import CANDIDATE_VOL_FEATURES
    from src.fomc_calendar import FOMC_FEATURE_COLUMNS

    overlap = set(CANDIDATE_VOL_FEATURES + FOMC_FEATURE_COLUMNS) & set(FEATURE_COLUMNS)
    assert not overlap, \
        f"Candidate features must not enter the direction/return feature set: {overlap}"


def test_ti_adx_matches_wilder_closed_form_on_monotone_series():
    """ADX(14) hand-check (experimental TI module): on a strictly monotone
    series with constant bar geometry, Wilder's recursion has CLOSED-FORM
    values — +DM=1, -DM=0, TR=1.25 from bar 1, so with w=13/14:
        plus_smooth_t = 1 - w^t,   atr_t = 1.25 - 0.75 * w^t,
        +DI_t = 100 * (1 - w^t) / (1.25 - 0.75 w^t)  ->  80,
        DX_t = 100 for every t >= 1 (since -DI = 0),
        ADX_t = 100 * (1 - w^t)  ->  100.
    This verifies the actual smoothing recursion, not a formula transcription."""
    from src.ti_lstm_h1_experimental import adx

    n = 41
    t = np.arange(n, dtype=float)
    close = pd.Series(t)
    high = pd.Series(t + 0.25)
    low = pd.Series(t - 0.25)

    adx_line, plus_di, minus_di = adx(high, low, close, period=14)
    w = 13.0 / 14.0
    for k in (1, 5, 14, 40):
        expected_plus_di = 100.0 * (1 - w ** k) / (1.25 - 0.75 * w ** k)
        assert plus_di.iloc[k] == pytest.approx(expected_plus_di, rel=1e-9), \
            f"+DI closed form mismatch at t={k}"
        assert minus_di.iloc[k] == pytest.approx(0.0, abs=1e-12), \
            "downward DM must be zero on a pure uptrend"
        assert adx_line.iloc[k] == pytest.approx(100.0 * (1 - w ** k), rel=1e-9), \
            f"ADX closed form mismatch at t={k}"
    # Mirror-symmetric downtrend: -DI takes the same closed form, ADX identical.
    # (Negation swaps the high/low roles, so the mirrored HIGH is -low.)
    adx_dn, plus_dn, minus_dn = adx(-low, -high, -close, period=14)
    assert np.allclose(minus_dn.iloc[1:], plus_di.iloc[1:], rtol=1e-9)
    assert np.allclose(adx_dn.iloc[1:], adx_line.iloc[1:], rtol=1e-9)
    # Flat market: no directional movement, no blow-ups.
    flat = pd.Series(np.ones(n))
    adx_f, p_f, m_f = adx(flat, flat, flat, period=14)
    assert (adx_f == 0).all() and (p_f == 0).all() and (m_f == 0).all()


def test_ti_indicators_closed_forms_and_specified_parameters():
    """Analytic spot-checks of the experimental TI module's indicators on a
    linear ramp close_t = t (window statistics have closed forms), pinning the
    SPECIFIED parameters (MACD 13/34 with an 8-period SMA signal, CCI 20 with
    Lambert's 0.015, Bollinger 20/2.0)."""
    from src.ti_lstm_h1_experimental import (
        bollinger_percent_b, macd_features, cci, trend_vs_sma)

    n = 400
    t = np.arange(n, dtype=float)
    close = pd.Series(t + 100.0)
    high, low = close + 0.5, close - 0.5

    # CCI on a ramp: TP - SMA20 = 9.5, meandev(0..19) = 5.0 -> 9.5/(0.015*5).
    cci_v = cci(high, low, close, period=20, c=0.015)
    assert cci_v.iloc[-1] == pytest.approx(9.5 / (0.015 * 5.0), rel=1e-9)

    # %B on a ramp: (9.5 + 2*std)/(4*std), std = std(0..19, ddof=1) = sqrt(35).
    s = np.sqrt(35.0)
    pb = bollinger_percent_b(close, period=20, ndev=2.0)
    assert pb.iloc[-1] == pytest.approx((9.5 + 2 * s) / (4 * s), rel=1e-9)
    # Flat window -> neutral mid-band.
    pb_flat = bollinger_percent_b(pd.Series(np.ones(50)), period=20)
    assert (pb_flat.iloc[20:] == 0.5).all()

    # MACD on a ramp: EMA(span) lags a linear ramp by (span-1)/2 asymptotically,
    # so macd -> (34-1)/2 - (13-1)/2 = 10.5 and the SMA-8 histogram -> 0.
    macd_line, hist = macd_features(close, fast=13, slow=34, signal=8)
    assert macd_line.iloc[-1] == pytest.approx(10.5, rel=1e-3)
    assert hist.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    # trend_vs_sma on a ramp: SMA20 = close - 9.5 -> ratio has a closed form;
    # and the undefined warm-up is neutral 0, the h1_features convention.
    tr504 = trend_vs_sma(close, 20)
    assert tr504.iloc[-1] == pytest.approx((close.iloc[-1] / (close.iloc[-1] - 9.5)) - 1, rel=1e-12)
    assert (trend_vs_sma(close, 20).iloc[:19] == 0.0).all()


def test_ti_indicators_no_lookahead_via_future_truncation():
    """Every experimental TI indicator must be a pure trailing-window function:
    dropping the future must leave earlier enriched rows byte-identical."""
    from src.ti_lstm_h1_experimental import enrich_h1_with_indicators, TI_FEATURE_COLUMNS

    h1 = _synthetic_h1([1.10, 1.11, 1.09, 1.12, 1.13, 1.115, 1.12])
    full = enrich_h1_with_indicators(h1)
    cut = len(h1) // 2
    trunc = enrich_h1_with_indicators(h1.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut][TI_FEATURE_COLUMNS],
                                  trunc[TI_FEATURE_COLUMNS])


def test_ti_dataset_target_is_next_day_return_same_convention_as_h1():
    """The experimental TI dataset must inherit the H1->Daily target contract:
    row t's target = next-day percent log return (shift(-1)), final day
    dropped, tensor right-aligned to the day's last 24 bars."""
    from src.ti_lstm_h1_experimental import build_ti_datasets

    closes = [1.10, 1.11, 1.09, 1.12, 1.13]
    X, y_ret, y_dir, index = build_ti_datasets(h1=_synthetic_h1(closes))

    assert len(index) == len(closes) - 1, "final (unresolved-target) day must drop"
    for i in range(len(index)):
        expected = np.log(closes[i + 1] / closes[i]) * 100.0
        assert y_ret[i] == pytest.approx(expected, rel=1e-5)
        assert y_dir[i] == int(expected > 0)
    assert X.shape[1:] == (24, 8), "24 hourly steps x 8 TI features"


def test_history_scores_volatility_as_magnitude_mae_not_hitmiss(tmp_path, monkeypatch):
    """The /history page must score the validated volatility forecast as a
    MAGNITUDE error (|predicted - realized| against |ln(close_{t+1}/close_t)|*100,
    the exact target it was validated on) and report a running MAE — never a
    directional hit/miss. A pending row shows the forecast with an hourglass and
    is not counted; a pre-volatility row (no vol_pred_pct) shows an em-dash."""
    import src.tracking as tracking

    log = pd.DataFrame([
        # resolved row: as_of close 1.1000 -> realized close 1.1050
        {'as_of_date': '2026-06-01', 'forecasting_date': '2026-06-02',
         'as_of_close': 1.1000, 'pred_direction': 'UP', 'pred_return_pct': 0.05,
         'pred_confidence': 0.55, 'vol_pred_pct': 0.20},
        # pending row: no realized close for its forecast day
        {'as_of_date': '2026-06-03', 'forecasting_date': '2026-06-30',
         'as_of_close': 1.1100, 'pred_direction': 'DOWN', 'pred_return_pct': -0.03,
         'pred_confidence': 0.53, 'vol_pred_pct': 0.18},
        # pre-volatility row: vol_pred_pct is NaN -> shown as em-dash, uncounted
        {'as_of_date': '2026-06-04', 'forecasting_date': '2026-06-05',
         'as_of_close': 1.1200, 'pred_direction': 'UP', 'pred_return_pct': 0.02,
         'pred_confidence': 0.54, 'vol_pred_pct': float('nan')},
    ])
    log_path = tmp_path / 'log.csv'
    log.to_csv(log_path, index=False)

    # Realized closes injected (no network): only the first & third forecast days
    # have settled.
    monkeypatch.setattr(tracking, '_actual_closes',
                        lambda *a, **k: {'2026-06-02': 1.1050, '2026-06-05': 1.1180})

    html = tracking.build_history_html(str(log_path), {'symbol': 'EURUSD'})

    realized = abs(np.log(1.1050 / 1.1000)) * 100          # ~0.4535
    err = abs(realized - 0.20)                              # ~0.2535
    assert f"{realized:.4f}%" in html, "realized volatility (|log return|*100) must be shown"
    assert f"err {err:.4f}" in html, "per-row |predicted - realized| volatility error must be shown"
    assert "±0.2000%" in html and "±0.1800% ⏳" in html, \
        "predicted magnitude must render; a pending row keeps the hourglass"
    # Running MAE over the ONE settled volatility row, labeled as validated.
    assert f"<b>{err:.4f}%</b> MAE over 1" in html
    assert "validated vs GARCH(1,1)" in html
    # The magnitude call is NOT dressed up as a direction hit: the volatility
    # cell carries no ✅/❌ (those belong to the daily/H1 directional columns).
    assert "±0.2000% → " in html and "✅" not in html.split("±0.2000%")[1][:40]


def test_fomc_calendar_join_resolves_known_dates_correctly():
    """The FOMC calendar join must resolve is_fomc_day / days_to_next /
    days_since_last exactly for hand-checkable dates. (No look-ahead test by
    design: scheduled FOMC dates are public knowledge months-to-years ahead —
    there is no publish-lag surface to guard, unlike the FRED series.)"""
    from src.fomc_calendar import add_fomc_features

    calendar = ['2026-01-28', '2026-03-18']
    dates = pd.date_range('2026-01-26', '2026-01-31', freq='D')
    df = pd.DataFrame({'close': np.ones(len(dates))}, index=dates)

    res = add_fomc_features(df, fomc_dates=calendar)

    # is_fomc_day: exactly the statement day.
    assert res['is_fomc_day'].tolist() == [0, 0, 1, 0, 0, 0]
    # Countdown to the next statement day (0 ON the day, then to March 18).
    assert res.loc['2026-01-26', 'days_to_next_fomc'] == 2
    assert res.loc['2026-01-28', 'days_to_next_fomc'] == 0
    assert res.loc['2026-01-29', 'days_to_next_fomc'] == \
        (pd.Timestamp('2026-03-18') - pd.Timestamp('2026-01-29')).days
    # Distance since the last statement day (0 ON the day).
    assert res.loc['2026-01-28', 'days_since_last_fomc'] == 0
    assert res.loc['2026-01-31', 'days_since_last_fomc'] == 3
    # Before the first calendar date there IS no last meeting -> NaN, so a
    # truncated calendar can never silently masquerade as a long quiet period.
    assert np.isnan(res.loc['2026-01-26', 'days_since_last_fomc'])


def test_fomc_dates_csv_covers_known_statement_days():
    """The committed reference CSV must contain hand-known statement days,
    exclude the non-scheduled actions (surprise moves are not knowable in
    advance -> including them would leak the future into the countdown), and
    span the whole euro-era modeling window."""
    from src.fomc_calendar import load_fomc_dates

    dates = set(load_fomc_dates())
    for known in ['1999-02-03', '2008-12-16', '2015-12-16', '2019-05-01',
                  '2022-06-15', '2026-01-28']:
        assert pd.Timestamp(known) in dates, f"missing known FOMC statement day {known}"
    for excluded in ['2020-03-03', '2020-03-15', '2020-03-18', '2025-08-22']:
        assert pd.Timestamp(excluded) not in dates, \
            f"non-scheduled/cancelled action {excluded} must be excluded"
    idx = pd.DatetimeIndex(sorted(dates))
    assert idx.min() <= pd.Timestamp('1999-01-04') + pd.Timedelta(days=60)
    assert idx.max() >= pd.Timestamp('2026-12-01')
    assert (idx.dayofweek < 5).all(), "FOMC statement days are always weekdays"


def test_h1_inference_sample_drops_forming_current_day():
    """Live H1 inference must ignore the still-forming current UTC session and
    settle on the previous completed day, mirroring drop_incomplete_bars."""
    from src.h1_features import build_h1_inference_sample, SEQ_FEATURE_COLUMNS, HOURS_PER_DAY

    closes = [1.10, 1.11, 1.12]   # 03-02, 03-03, 03-04
    h1 = _synthetic_h1(closes, start='2026-03-02')

    # Mid-session on the last day (03-04) -> it is forming and must be excluded.
    flat_row, seq, as_of, source = build_h1_inference_sample(h1=h1, now=pd.Timestamp('2026-03-04 11:00', tz='UTC'))

    assert as_of == pd.Timestamp('2026-03-03', tz='UTC'), "Must settle on the last COMPLETED day."
    assert flat_row.shape[0] == 1
    assert seq.shape == (1, HOURS_PER_DAY, len(SEQ_FEATURE_COLUMNS))
    assert source == "preloaded", "A directly passed frame must be labeled as such, not as cache/live."


def test_h1_inference_refreshes_stale_cache_live_first(tmp_path, monkeypatch):
    """REGRESSION (frozen H1 'as of' bug): the old cache-first load served
    whatever day the last retrain happened to leave in results/eurusd_h1.csv.
    With a live source available, inference must serve the latest complete
    trading session — never a stale cached one."""
    from src import live_data
    from src.h1_features import build_h1_inference_sample

    # Cache ends Wed 2026-03-04; by Friday it is two sessions behind.
    stale = _synthetic_h1([1.10, 1.11, 1.12], start='2026-03-02')
    cache = tmp_path / 'eurusd_h1.csv'
    stale.to_csv(cache)

    # The live chain knows through Thu 2026-03-05 (the expected latest session).
    fresh = _synthetic_h1([1.10, 1.11, 1.12, 1.13], start='2026-03-02')
    monkeypatch.setattr(live_data, 'fetch_h1_market_data', lambda **kw: (fresh, 'MT5'))

    now = pd.Timestamp('2026-03-06 09:00', tz='UTC')  # Friday morning
    flat_row, seq, as_of, source = build_h1_inference_sample(cache_path=str(cache), now=now)

    assert as_of == pd.Timestamp('2026-03-05', tz='UTC'), \
        "Must serve the latest complete session from the live fetch, not the stale cache day."
    assert source == "live"


def test_h1_staleness_gate_skips_live_fetch_when_cache_current(tmp_path, monkeypatch):
    """The staleness gate is mandatory: when the cache already holds the
    expected latest complete session, a dashboard load must NOT hit the live
    chain at all (no fetch-on-every-call regression)."""
    from src import live_data
    from src.h1_features import build_h1_inference_sample

    current = _synthetic_h1([1.10, 1.11, 1.12, 1.13], start='2026-03-02')  # ends Thu 03-05
    cache = tmp_path / 'eurusd_h1.csv'
    current.to_csv(cache)

    def _must_not_fetch(**kw):
        raise AssertionError("Cache is current — the staleness gate must not trigger a live fetch.")
    monkeypatch.setattr(live_data, 'fetch_h1_market_data', _must_not_fetch)

    now = pd.Timestamp('2026-03-06 09:00', tz='UTC')  # Friday; expected latest = Thu 03-05
    flat_row, seq, as_of, source = build_h1_inference_sample(cache_path=str(cache), now=now)

    assert as_of == pd.Timestamp('2026-03-05', tz='UTC')
    assert source == "cache"


def test_h1_thin_live_fetch_backfills_history_from_cache(tmp_path, monkeypatch):
    """A live pull rich in NEW bars but thin in history must be merged onto the
    cached rows (dedup by index, live wins) and the merged frame rewritten to
    the cache — a shallow fetch may never truncate the SMA504/RSI warm-up
    history (the H1 analogue of the old daily SMA_200 warm-up bug)."""
    from src import live_data
    from src.h1_features import refresh_h1_frame

    old = _synthetic_h1([1.10, 1.11, 1.12], start='2026-03-02')  # Mon..Wed history
    cache = tmp_path / 'eurusd_h1.csv'
    old.to_csv(cache)

    thin_live = _synthetic_h1([1.13], start='2026-03-05')  # Thu only — no history

    def _fake_fetch(**kw):
        thin_live.to_csv(kw.get('cache_path'))  # mimic fetch's own live-only cache write
        return thin_live, 'yfinance'
    monkeypatch.setattr(live_data, 'fetch_h1_market_data', _fake_fetch)

    now = pd.Timestamp('2026-03-06 09:00', tz='UTC')
    frame, source = refresh_h1_frame(cache_path=str(cache), now=now)

    assert source == "live+history_backfill"
    assert frame.index.min() == old.index.min(), "Cached history rows must survive the merge."
    assert frame.index.max() == thin_live.index.max(), "The fresh live bars must be present."
    assert len(frame) == len(old) + len(thin_live)

    # The cache on disk must now hold the MERGED frame, not the thin live-only write.
    reread = pd.read_csv(cache, index_col=0, parse_dates=True)
    assert len(reread) == len(frame), "Merged frame must be persisted back to the cache."


def test_compute_h1_consensus_majority_and_agreement():
    """A 3-1 split must yield the majority direction, confidence = agreeing
    fraction (0.75), agreement=False, and the mean return over the MAJORITY-side
    models only — a full-panel mean here would be NEGATIVE (-0.025) under an
    'UP' label, the exact vote-vs-magnitude inconsistency the tie bug exposed."""
    from src.inference import PredictionService

    per_model = {
        'h1_xgboost':      {'direction': 'UP',   'predicted_return_pct': 0.10},
        'h1_random_forest': {'direction': 'UP',   'predicted_return_pct': 0.20},
        'h1_svm':          {'direction': 'UP',   'predicted_return_pct': 0.00},
        'h1_lstm':         {'direction': 'DOWN', 'predicted_return_pct': -0.40},
    }
    c = PredictionService.compute_h1_consensus(per_model)

    assert c['direction'] == 'UP'
    assert c['agreement'] is False
    assert c['confidence'] == pytest.approx(0.75)
    assert c['predicted_return_pct'] == pytest.approx((0.10 + 0.20 + 0.00) / 3), \
        "return must average the majority side only, staying sign-consistent with the label"
    assert c['predicted_return_pct'] >= 0, "an UP label must never ship with a negative consensus return"
    assert c['n_models'] == 4


def test_compute_h1_consensus_exact_tie_is_mixed_not_arbitrary_up():
    """REGRESSION (live dashboard bug, 2026-07-07): a 2-2 vote split was labeled
    'UP — 50% model agreement' (via the old `up >= down` tie-break) while the
    displayed averaged return was NEGATIVE (-0.0131%). An exact tie has NO
    majority: it must be labeled MIXED / TIE, mirroring the daily committee's
    MIXED honesty, with the full-panel mean only as unclaiming context.
    Exact numbers from the report."""
    from src.inference import PredictionService

    per_model = {
        'h1_xgboost':       {'direction': 'UP',   'predicted_return_pct':  0.0043},
        'h1_random_forest': {'direction': 'UP',   'predicted_return_pct':  0.0048},
        'h1_svm':           {'direction': 'DOWN', 'predicted_return_pct': -0.0035},
        'h1_lstm':          {'direction': 'DOWN', 'predicted_return_pct': -0.0581},
    }
    c = PredictionService.compute_h1_consensus(per_model)

    assert c['direction'] == 'MIXED / TIE', "a 2-2 split must never be crowned UP or DOWN"
    assert c['agreement'] is False
    assert c['confidence'] == pytest.approx(0.5)
    assert c['predicted_return_pct'] == pytest.approx((0.0043 + 0.0048 - 0.0035 - 0.0581) / 4)
    assert c['n_models'] == 4


def test_compute_h1_consensus_unanimous_sets_agreement_true():
    """When all four regressors agree on sign, agreement is True and confidence
    saturates at 1.0."""
    from src.inference import PredictionService

    per_model = {k: {'direction': 'DOWN', 'predicted_return_pct': v}
                 for k, v in {'h1_xgboost': -0.10, 'h1_random_forest': -0.05,
                              'h1_svm': -0.20, 'h1_lstm': -0.15}.items()}
    c = PredictionService.compute_h1_consensus(per_model)

    assert c['direction'] == 'DOWN'
    assert c['agreement'] is True
    assert c['confidence'] == pytest.approx(1.0)


def _fake_predict_result(as_of, forecast, close, direction, ret,
                         baseline_direction=None, baseline_ret=None):
    """Minimal PredictionService.predict()-shaped dict (dual-variant schema)
    for tracking tests. The with_macro block carries the primary `direction`;
    pass baseline_direction to also populate the price-only variant's block
    (omitting it mimics a degraded/pre-dual prediction with no baseline call)."""
    result = {
        'as_of_date': as_of, 'forecasting_date': forecast,
        'bar_used': {'close': close},
        'with_macro': {
            'gbm': {'direction': direction}, 'lstm': {'direction': direction},
            'consensus': {'direction': direction, 'predicted_return_pct': ret, 'confidence': 0.55},
        },
        'baseline': {},
        'variant_agreement': None,
    }
    if baseline_direction is not None:
        result['baseline'] = {
            'gbm': {'direction': baseline_direction}, 'lstm': {'direction': baseline_direction},
            'consensus': {'direction': baseline_direction,
                          'predicted_return_pct': ret if baseline_ret is None else baseline_ret,
                          'confidence': 0.55},
        }
        result['variant_agreement'] = baseline_direction == direction
    return result


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

    html = tracking.build_history_html(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                                        now=pd.Timestamp('2026-06-25 11:00'))
    assert 'correct' in html and "class='hit'" in html, "Correct UP call must be scored as a hit."
    assert 'pending' in html, "An unresolved future forecast must render as pending."


def test_build_history_html_does_not_score_against_still_forming_today_bar(tmp_path, monkeypatch):
    """A forecast whose forecast_date is the current, still-open trading
    session must stay pending -- it must never be scored against today's
    mid-session price, which can still move before the session actually
    closes (the bug reported live: a forecast for 2026-06-25 was marked
    correct/wrong while 25.06 hadn't finished yet)."""
    import src.tracking as tracking
    log = str(tmp_path / 'log.csv')
    # Predicted DOWN from yesterday's close, forecasting TODAY (2026-06-25).
    tracking.log_prediction(_fake_predict_result('2026-06-24', '2026-06-25', 1.1356, 'DOWN', -0.02), log)

    # The live feed's most recent bar is today's still-forming session --
    # a real intraday price, but not a closed one.
    actual = pd.DataFrame({'close': [1.1369]}, index=pd.DatetimeIndex(['2026-06-25']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    html = tracking.build_history_html(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                                        now=pd.Timestamp('2026-06-25 11:00'))

    assert 'pending' in html, "Today's not-yet-closed session must render as pending."
    assert "class='hit'" not in html and "class='miss'" not in html, \
        "Must not score a forecast against an intraday price that hasn't closed yet."
    assert '1/1 resolved' in html or '100%' in html, "Hit-rate summary must reflect the single resolved row."


def test_worst_mistakes_ranks_by_absolute_error_and_excludes_pending(tmp_path, monkeypatch):
    """worst_mistakes must join each RESOLVED row against the realised return,
    rank by descending abs_error, and drop any row whose forecast date hasn't
    closed yet (abs_error is undefined for a pending prediction)."""
    import src.tracking as tracking
    log = str(tmp_path / 'log.csv')
    # Small miss: predicted +0.05%, actual +0.08% (1.1000 -> 1.10088) -> abs_error ~0.03.
    tracking.log_prediction(_fake_predict_result('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05), log)
    # Big miss: predicted -0.02%, actual +0.727% (1.1050 -> 1.1130) -> abs_error ~0.747.
    tracking.log_prediction(_fake_predict_result('2026-06-20', '2026-06-23', 1.1050, 'DOWN', -0.02), log)
    # Still pending -- no realised close available for this forecast date.
    tracking.log_prediction(_fake_predict_result('2026-06-23', '2026-06-24', 1.1130, 'UP', +0.01), log)

    actual = pd.DataFrame(
        {'close': [1.10088, 1.1130]},
        index=pd.DatetimeIndex(['2026-06-22', '2026-06-23']),
    )
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    worst = tracking.worst_mistakes(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                                     now=pd.Timestamp('2026-06-25 11:00'))

    assert len(worst) == 2, "Only the two resolved rows should be scored; the pending one is excluded."
    assert worst.iloc[0]['forecasting_date'] == '2026-06-23', "The larger miss must rank first."
    assert worst.iloc[0]['abs_error'] > worst.iloc[1]['abs_error'], "Must be sorted descending by abs_error."


def test_simulate_strategy_charges_cost_only_on_position_change():
    """Transaction costs must be charged only when the signal actually flips
    (or on day 0, entering from flat) -- not on every day a position is held
    -- and gross/net returns must match a hand-computed reference exactly."""
    from src.backtest import simulate_strategy

    y_true_return_pct = [0.5, -0.3, 0.2, 0.1]   # actual realised next-day returns
    y_pred_direction = [1, 0, 1, 1]              # UP, DOWN, UP, UP -> signal [+1,-1,+1,+1]

    result = simulate_strategy(y_true_return_pct, y_pred_direction, cost_pct_per_trade=0.05)

    assert result['n_trades'] == 3, "Day 0 (flat->long) + day 1 (long->short) + day 2 (short->long) flip; day 3 holds."
    assert result['gross_return_pct_total'] == pytest.approx(1.1)
    assert result['net_return_pct_total'] == pytest.approx(1.1 - 3 * 0.05)
    assert result['hit_rate'] == pytest.approx(1.0), "Every day's signal matches the realised direction here."


def test_simulate_strategy_wrong_calls_produce_negative_gross_return():
    """A signal that's wrong every day must show a negative gross return and a
    0% hit rate -- confirms sign(y_true) vs signal comparison isn't inverted."""
    from src.backtest import simulate_strategy

    result = simulate_strategy(y_true_return_pct=[0.4, 0.3], y_pred_direction=[0, 0], cost_pct_per_trade=0.0)

    assert result['hit_rate'] == pytest.approx(0.0)
    assert result['gross_return_pct_total'] == pytest.approx(-0.7)


def test_ablation_arbiter_is_validation_never_test():
    """The feature-ablation harness must decide KEEP/DROP on the validation
    slice [70%:80%] ONLY -- the final test block [80%:100%] must never be read
    during feature search (post-defense anti-data-snooping invariant). Guard the
    split math directly so a future refactor can't silently push the arbiter
    onto the test rows."""
    from src.ablation import _canonical_split

    # 1000 rows, config's 0.80 / 0.10 -> train [0:700], val [700:800], test [800:1000].
    split = _canonical_split(1000, train_fraction=0.80, val_fraction=0.10)
    assert split['train_end'] == 700, "ablation fit block must end at the 70% mark"
    assert split['val_end'] == 800, "validation arbiter must end at the 80% mark"
    # The validation arbiter is exactly the 70-80% band, disjoint from and strictly
    # BELOW the test block [val_end:] which this harness never indexes into.
    val_rows = set(range(split['train_end'], split['val_end']))
    test_rows = set(range(split['val_end'], split['n']))
    assert val_rows.isdisjoint(test_rows)
    assert max(val_rows) < min(test_rows), "validation must lie entirely before the untouched test block"


def test_ablation_mcnemar_exact_symmetric_and_significant_cases():
    """The exact-binomial McNemar helper: perfectly balanced discordant pairs
    give p=1.0 (no evidence of a difference); a lopsided split gives a small
    two-sided p. Guards the b/c discordant-pair accounting the KEEP decision
    rests on."""
    import numpy as np
    from src.ablation import _mcnemar_exact

    # Balanced: WITHOUT-wrong/WITH-correct on 5 rows, the reverse on 5 rows.
    wo = np.array([False] * 5 + [True] * 5 + [True] * 3)
    w = np.array([True] * 5 + [False] * 5 + [True] * 3)
    b, c, p = _mcnemar_exact(wo, w)
    assert (b, c) == (5, 5) and p == pytest.approx(1.0)

    # Lopsided 8 vs 1: two-sided exact p must be well under 0.05.
    wo2 = np.array([False] * 8 + [True] * 1)
    w2 = np.array([True] * 8 + [False] * 1)
    b2, c2, p2 = _mcnemar_exact(wo2, w2)
    assert (b2, c2) == (8, 1) and p2 < 0.05


def test_bonferroni_bar_tightens_with_family_size():
    """The corrected significance bar must shrink as more feature hypotheses are
    spent (0.05/N), so a KEEP gets harder to earn the more features are tried --
    the whole point of the multiple-comparisons guard."""
    from src.ablation import bonferroni_alpha

    assert bonferroni_alpha(1) == pytest.approx(0.05)
    assert bonferroni_alpha(4) == pytest.approx(0.0125)
    assert bonferroni_alpha(10) == pytest.approx(0.005)
    assert bonferroni_alpha(0) == pytest.approx(0.05), "degenerate family size 0 falls back to uncorrected bar"


def test_register_hypothesis_is_idempotent_and_counts_family(tmp_path, monkeypatch):
    """Registering a new feature grows the family by one and records the bar it
    faced; registering the SAME feature again must not double-count it (the log
    is the source of truth for the multiple-comparisons denominator)."""
    import src.ablation as ablation

    log_path = tmp_path / "feature_hypothesis_log.csv"
    monkeypatch.setattr(ablation, "HYPOTHESIS_LOG", str(log_path))

    res = {
        'feature': 'brand_new_feature', 'arbiter': 'validation[70:80]',
        'point_delta_acc': 0.02, 'ci95_dacc_low': 0.005, 'ci95_dacc_high': 0.035,
        'mcnemar_p': 0.001, 'verdict': 'KEEP (CI excludes 0 and McNemar p<0.05)',
    }
    ablation.register_hypothesis(res)
    ablation.register_hypothesis(res)   # second call must be a no-op

    log = ablation.load_hypothesis_log()
    assert list(log['feature']).count('brand_new_feature') == 1, "must not double-count"
    row = log[log['feature'] == 'brand_new_feature'].iloc[0]
    assert row['alpha_bonferroni'] == pytest.approx(0.05)   # first hypothesis -> family size 1
    assert bool(row['cleared_bar']) is True                 # CI excludes 0 and p < 0.05


def _paper_log(tmp_path, rows):
    """Write a minimal prediction-log CSV for the paper-trading tests."""
    from src.tracking import log_prediction
    log = str(tmp_path / 'log.csv')
    for as_of, forecast, close, direction, ret in rows:
        log_prediction(_fake_predict_result(as_of, forecast, close, direction, ret), log)
    return log


def test_paper_trading_ledger_scores_costs_and_direction(tmp_path, monkeypatch):
    """A correct LONG call nets (move − spread) pips and counts a win; a wrong
    call is a loss; a MIXED row takes no position (flat, excluded from the win
    rate). Guards the direction sign and the per-position spread charge."""
    import src.tracking as tracking
    from src.paper_trading import build_ledger, summarize

    log = _paper_log(tmp_path, [
        ('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05),   # correct long: +50 pips gross
        ('2026-06-22', '2026-06-23', 1.1050, 'UP', +0.05),   # wrong long: market falls
        ('2026-06-23', '2026-06-24', 1.1000, 'MIXED / LOW CONFIDENCE', 0.0),  # no position
    ])
    # Realised closes: 22nd up to 1.1050 (+50p), 23rd down to 1.1000 (−50p), 24th anything.
    actual = pd.DataFrame({'close': [1.1050, 1.1000, 1.1010]},
                          index=pd.DatetimeIndex(['2026-06-22', '2026-06-23', '2026-06-24']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    ledger = build_ledger(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                          spread_pips=2.0, now=pd.Timestamp('2026-06-27 11:00'))

    assert len(ledger) == 3
    win_row = ledger.iloc[0]
    assert win_row['direction'] == 'LONG'
    assert win_row['gross_pips'] == pytest.approx(50.0, abs=0.5)
    assert win_row['net_pips'] == pytest.approx(48.0, abs=0.5), "gross 50p minus 2p spread"
    assert win_row['outcome'] == 'win'
    assert ledger.iloc[1]['outcome'] == 'loss', "wrong long into a falling market is a loss"
    assert ledger.iloc[2]['direction'] == 'FLAT' and ledger.iloc[2]['spread_pips'] == 0.0, \
        "MIXED row takes no position and is charged no spread"

    summary = summarize(ledger)
    assert summary['n_positions'] == 2, "flat day excluded from taken positions"
    assert summary['n_wins'] == 1 and summary['win_rate'] == pytest.approx(0.5)


def test_paper_trading_excludes_unsettled_forecasts(tmp_path, monkeypatch):
    """A forecast whose day hasn't closed has undefined P&L and must not appear
    in the ledger -- the forward ledger only ever scores settled sessions."""
    import src.tracking as tracking
    from src.paper_trading import build_ledger

    log = _paper_log(tmp_path, [
        ('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05),   # settled
        ('2026-06-22', '2026-06-23', 1.1050, 'DOWN', -0.05),  # 23rd not in closes -> pending
    ])
    actual = pd.DataFrame({'close': [1.1080]}, index=pd.DatetimeIndex(['2026-06-22']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    ledger = build_ledger(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                          now=pd.Timestamp('2026-06-25 11:00'))
    assert len(ledger) == 1, "only the settled forecast is scored"
    assert str(ledger.iloc[0]['forecasting_date']) == '2026-06-22'


def test_paper_trading_summary_max_drawdown_and_cumulative(tmp_path, monkeypatch):
    """Max drawdown is measured on the cumulative net-return curve seeded at
    flat, and cumulative pips accumulate in chronological order. A win then an
    equal-and-opposite loss must leave a drawdown equal to the loss leg."""
    import src.tracking as tracking
    from src.paper_trading import build_ledger, summarize

    log = _paper_log(tmp_path, [
        ('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05),   # +move
        ('2026-06-22', '2026-06-23', 1.1100, 'UP', +0.05),   # −move (drawdown leg)
    ])
    actual = pd.DataFrame({'close': [1.1100, 1.1000]},
                          index=pd.DatetimeIndex(['2026-06-22', '2026-06-23']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))

    ledger = build_ledger(log, {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'},
                          spread_pips=0.0, now=pd.Timestamp('2026-06-27 11:00'))
    summary = summarize(ledger)

    assert ledger['cum_net_pips'].iloc[-1] == pytest.approx(0.0, abs=1.0), \
        "a +100p win then a −100p loss nets ~0 cumulative"
    assert summary['max_drawdown_pct'] > 0, "the losing leg must register as a drawdown"


# ── Dual model variants (baseline price-only vs with_macro) ────────────────

def test_variant_feature_columns_price_only_vs_full():
    """The 'baseline' variant must be strictly price-only (no macro-derived
    feature reaches its models), and 'with_macro' must be the full canonical
    FEATURE_COLUMNS set. Both must still carry the PCA lag block. Guards the
    contract that the two served variants differ ONLY in the macro features."""
    from src.features import (
        variant_feature_columns, FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS, LAG_COLUMNS,
    )

    baseline = variant_feature_columns('baseline')
    with_macro = variant_feature_columns('with_macro')

    assert with_macro == FEATURE_COLUMNS, "with_macro must be the full canonical column set"
    assert set(MACRO_FEATURE_COLUMNS) == {'yield_differential_delta', 'usd_index_return',
                                          'policy_rate_differential', 'inflation_differential'}
    assert not set(baseline) & set(MACRO_FEATURE_COLUMNS), "baseline must contain NO macro-derived feature"
    assert set(with_macro) - set(baseline) == set(MACRO_FEATURE_COLUMNS), \
        "the two variants must differ exactly by the macro block"
    for col in LAG_COLUMNS:
        assert col in baseline and col in with_macro, "both variants keep the PCA lag block"
    # baseline preserves the canonical relative order (a scrambled order would
    # silently break the trained artifacts' column contract)
    assert baseline == [c for c in FEATURE_COLUMNS if c not in MACRO_FEATURE_COLUMNS]

    with pytest.raises(KeyError):
        variant_feature_columns('nonexistent_variant')


def test_log_prediction_records_both_variants(tmp_path):
    """A dual-variant prediction must land the with_macro consensus in the
    legacy pred_* columns (ledger continuity) AND the baseline consensus in the
    baseline_* columns, plus the variant_agreement flag."""
    from src.tracking import log_prediction
    log = str(tmp_path / 'log.csv')

    log_prediction(_fake_predict_result('2026-07-06', '2026-07-07', 1.1400, 'DOWN', -0.02,
                                        baseline_direction='UP', baseline_ret=+0.01), log)

    row = pd.read_csv(log).iloc[0]
    assert row['pred_direction'] == 'DOWN', "pred_* columns must carry the with_macro committee"
    assert row['baseline_direction'] == 'UP', "baseline_* columns must carry the price-only committee"
    assert row['baseline_return_pct'] == pytest.approx(0.01)
    assert bool(row['variant_agreement']) is False, "UP vs DOWN must record disagreement"


def test_paper_trading_baseline_ledger_skips_rows_without_baseline_forecast(tmp_path, monkeypatch):
    """The baseline ledger must be driven by baseline_direction and must SKIP
    (not flat-log) rows where no baseline forecast exists — e.g. every pre-dual
    historical row — while the macro ledger still scores them via pred_direction."""
    import src.tracking as tracking
    from src.paper_trading import build_ledger

    from src.tracking import log_prediction
    log = str(tmp_path / 'log.csv')
    # Pre-dual row: with_macro call only, no baseline block.
    log_prediction(_fake_predict_result('2026-06-19', '2026-06-22', 1.1000, 'UP', +0.05), log)
    # Dual row: both variants called (they disagree).
    log_prediction(_fake_predict_result('2026-06-22', '2026-06-23', 1.1050, 'UP', +0.05,
                                        baseline_direction='DOWN'), log)

    actual = pd.DataFrame({'close': [1.1050, 1.1000]},
                          index=pd.DatetimeIndex(['2026-06-22', '2026-06-23']))
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (actual, 'stub'))
    data_cfg = {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X'}
    now = pd.Timestamp('2026-06-27 11:00')

    macro_ledger = build_ledger(log, data_cfg, spread_pips=0.0, now=now,
                                direction_column='pred_direction')
    baseline_ledger = build_ledger(log, data_cfg, spread_pips=0.0, now=now,
                                   direction_column='baseline_direction')

    assert len(macro_ledger) == 2, "macro ledger scores both rows (full pred_direction lineage)"
    assert len(baseline_ledger) == 1, "baseline ledger must skip the pre-dual row entirely"
    assert str(baseline_ledger.iloc[0]['forecasting_date']) == '2026-06-23'
    assert baseline_ledger.iloc[0]['direction'] == 'SHORT', "baseline DOWN call -> short position"
    # Same day, opposite calls: the two ledgers must book opposite-signed P&L.
    macro_23 = macro_ledger[macro_ledger['forecasting_date'] == '2026-06-23'].iloc[0]
    assert macro_23['net_pips'] == pytest.approx(-baseline_ledger.iloc[0]['net_pips'], abs=0.01), \
        "opposite positions on the same move must mirror each other's P&L at zero cost"