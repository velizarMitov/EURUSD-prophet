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


# ── COT (CFTC positioning) features ─────────────────────────────────────────

def test_cot_availability_date_trusts_recent_publish_but_buffers_bulk_reload():
    """The look-ahead heart of src/cot_data.py: the join date must be when the
    weekly report was actually PUBLISHED, not its Tuesday 'as of' date.

    * A plausible recent publish timestamp (:created_at a few days after as_of,
      even a holiday-delayed 6 days) is trusted -> availability = publish + 1d.
    * The 2022-09-13 bulk-reload artifact (created_at YEARS after as_of) is NOT
      trusted -> availability falls back to the conservative as_of + buffer.
    * A missing timestamp -> the conservative buffer too.
    Never earlier than reality (erring late), for every branch."""
    from src.cot_data import availability_date, PUBLISH_BUFFER_DAYS, CREATED_AT_TRUST_MAX_LAG_DAYS

    as_of = pd.Timestamp('2026-06-30', tz='UTC')          # a Tuesday
    # Typical 3-day (Fri) publish -> trusted, +1d safety.
    assert availability_date(as_of, pd.Timestamp('2026-07-03T19:30Z')) == pd.Timestamp('2026-07-04', tz='UTC')
    # Holiday-delayed 6-day publish -> still trusted (this is the whole point of
    # using created_at over a fixed +3 offset).
    assert availability_date(as_of, pd.Timestamp('2026-07-06T19:30Z')) == pd.Timestamp('2026-07-07', tz='UTC')
    # Bulk-reload artifact (created 2022 for a 2006 report) -> buffer, NOT the
    # absurd reload date, and NOT the leaky as_of.
    old_as_of = pd.Timestamp('2006-06-13', tz='UTC')
    got = availability_date(old_as_of, pd.Timestamp('2022-09-13T14:16:09Z'))
    assert got == old_as_of + pd.Timedelta(days=PUBLISH_BUFFER_DAYS)
    # Missing timestamp -> conservative buffer.
    assert availability_date(as_of, pd.NaT) == as_of + pd.Timedelta(days=PUBLISH_BUFFER_DAYS)
    # The trust window is bounded (a lag beyond it is treated as an artifact).
    stale = as_of + pd.Timedelta(days=CREATED_AT_TRUST_MAX_LAG_DAYS + 5)
    assert availability_date(as_of, stale) == as_of + pd.Timedelta(days=PUBLISH_BUFFER_DAYS)


def test_add_cot_features_ffill_by_availability_date_no_lookahead():
    """add_cot_features joins the z-scores by AVAILABILITY date with as-of ffill:
    a daily bar must carry the last COT reading ALREADY PUBLIC on that date, and
    must NOT see a reading whose publish date is still in the future."""
    from src.cot_data import add_cot_features, COT_FEATURE_COLUMNS

    # Two weekly readings, availability-dated (as the module would stamp them).
    cot_frame = pd.DataFrame(
        {'cot_eur_zscore': [1.0, 2.0], 'cot_usdindex_zscore': [-1.0, -2.0]},
        index=pd.DatetimeIndex(['2020-01-03', '2020-01-10'], tz='UTC'),  # Fri publishes
    )
    # Daily bars straddling the second publish date.
    days = pd.DatetimeIndex(['2020-01-06', '2020-01-09', '2020-01-10', '2020-01-13'])  # Mon..Mon
    df = pd.DataFrame({'close': [1.1] * len(days)}, index=days)

    out = add_cot_features(df, cot_frame=cot_frame)
    e = out['cot_eur_zscore']
    assert e.loc[days[0]] == pytest.approx(1.0), "Jan 6: only the Jan 3 reading is public yet."
    assert e.loc[days[1]] == pytest.approx(1.0), "Jan 9: Jan 10 reading is still in the FUTURE -> no leak."
    assert e.loc[days[2]] == pytest.approx(2.0), "Jan 10: the second reading is now public (its availability date)."
    assert e.loc[days[3]] == pytest.approx(2.0), "Jan 13: carries the latest public reading."
    assert set(COT_FEATURE_COLUMNS).issubset(out.columns)


def test_add_cot_features_neutral_zero_before_cot_exists():
    """Bars before any COT reading is available (pre-2006, or z-score warm-up)
    must get a NEUTRAL z-score of 0 -- never NaN (which would drop the whole
    euro-era row set) and never back-filled from the first future reading."""
    from src.cot_data import add_cot_features

    cot_frame = pd.DataFrame(
        {'cot_eur_zscore': [1.5], 'cot_usdindex_zscore': [0.5]},
        index=pd.DatetimeIndex(['2006-06-23'], tz='UTC'),
    )
    days = pd.DatetimeIndex(['1999-06-01', '2005-01-03', '2006-06-30'])
    df = pd.DataFrame({'close': [1.1] * len(days)}, index=days)
    out = add_cot_features(df, cot_frame=cot_frame)
    assert out['cot_eur_zscore'].loc[days[0]] == 0.0, "Pre-2006 must be neutral 0, not back-filled."
    assert out['cot_eur_zscore'].loc[days[1]] == 0.0, "Still before first reading -> neutral 0."
    assert out['cot_eur_zscore'].loc[days[2]] == pytest.approx(1.5), "After the first reading is public."


def test_cot_zscore_is_trailing_only_no_lookahead():
    """The z-score at week t must standardize against the trailing window ending
    AT t (only data public by t), never a window that peeks at future weeks."""
    from src.cot_data import _compute_cot_frame, ZSCORE_MIN_WEEKS

    n = ZSCORE_MIN_WEEKS + 5
    as_of = pd.date_range('2015-01-06', periods=n, freq='7D', tz='UTC')  # weekly Tuesdays
    net = np.arange(n, dtype=float)                                       # strictly increasing
    raw = pd.DataFrame({'cot_eur_net': net, 'cot_usdindex_net': net[::-1],
                        'created_at': pd.NaT}, index=as_of)
    frame = _compute_cot_frame(raw)
    # A strictly increasing net that only ever rises means the LAST reading is
    # the max of its trailing window -> its z-score must be POSITIVE. A
    # look-ahead window (including nothing future here, but the sign is the
    # tell) computed on trailing data gives a clean positive z at the top.
    assert frame['cot_eur_zscore'].iloc[-1] > 0, "Rising series: latest week sits above its trailing mean."
    # Early weeks below the min-periods threshold are undefined (-> neutral 0
    # downstream), never computed from later weeks.
    assert frame['cot_eur_zscore'].iloc[:ZSCORE_MIN_WEEKS - 1].isna().all(), \
        "No z-score may exist before the trailing window has min_periods of history."


def test_fetch_cot_features_falls_back_to_cache_when_api_down(monkeypatch, tmp_path):
    """When the CFTC API is unreachable, fetch_cot_features must reuse the cached
    raw snapshot on disk (recomputing z-scores from it) rather than failing the
    pipeline -- the same graceful-degradation contract as the FRED chain."""
    import src.cot_data as cot_data

    # A tiny cached raw frame (as_of index, per-market net, created_at).
    cache = str(tmp_path / 'cot.csv')
    idx = pd.date_range('2015-01-06', periods=cot_data.ZSCORE_MIN_WEEKS + 3, freq='7D', tz='UTC')
    pd.DataFrame({'cot_eur_net': np.arange(len(idx), dtype=float),
                  'cot_usdindex_net': np.arange(len(idx), dtype=float),
                  'created_at': pd.NaT}, index=idx).to_csv(cache)

    monkeypatch.setattr(cot_data, '_fetch_all_markets_via_api', lambda *a, **k: None)
    frame, source = cot_data.fetch_cot_features(cache_path=cache)
    assert source == 'cache'
    assert list(frame.columns) == cot_data.COT_FEATURE_COLUMNS
    assert frame['cot_eur_zscore'].notna().any(), "z-scores must be recomputed from the cached raw net series."


def test_fetch_cot_features_returns_none_when_nothing_reachable(monkeypatch, tmp_path):
    """No API and no cache at all -> (None, 'unavailable'), so add_cot_features
    neutralizes every COT column to 0 and the pipeline degrades, never crashes."""
    import src.cot_data as cot_data

    monkeypatch.setattr(cot_data, '_fetch_all_markets_via_api', lambda *a, **k: None)
    frame, source = cot_data.fetch_cot_features(cache_path=str(tmp_path / 'missing.csv'))
    assert frame is None and source == 'unavailable'

    # With the feed entirely unreachable (fetch -> None), add_cot_features must
    # still neutralize every COT column to 0 so the pipeline degrades, not dies.
    monkeypatch.setattr(cot_data, 'fetch_cot_features', lambda *a, **k: (None, 'unavailable'))
    df = pd.DataFrame({'close': [1.1, 1.1]}, index=pd.date_range('2020-01-01', periods=2))
    out = cot_data.add_cot_features(df, cot_frame=None)  # None frame -> all-neutral
    assert (out['cot_eur_zscore'] == 0.0).all() and (out['cot_usdindex_zscore'] == 0.0).all()


def test_weekly_cot_asof_join_backward_no_lookahead():
    """The weekly-horizon side-check (src/cot_weekly_check.py) joins COT by
    AVAILABILITY date with merge_asof(direction='backward'): a week-ending Tuesday
    must carry the last COT reading ALREADY PUBLIC by that Tuesday, never one
    whose availability date is still in the future; and the target must be the
    strictly-forward weekly return (last week, with no next, dropped)."""
    from src.cot_weekly_check import weekly_cot_target_frame

    tuesdays = pd.DatetimeIndex(['2020-01-07', '2020-01-14', '2020-01-21', '2020-01-28'])
    weekly_close = pd.Series([1.10, 1.11, 1.12, 1.13], index=tuesdays)
    cot_frame = pd.DataFrame(
        {'cot_eur_zscore': [0.5, 1.0, 2.0], 'cot_usdindex_zscore': [-0.5, -1.0, -2.0]},
        index=pd.DatetimeIndex(['2020-01-03', '2020-01-10', '2020-01-17'], tz='UTC'),  # Friday publishes
    )
    out = weekly_cot_target_frame(weekly_close, cot_frame)
    e = out['cot_eur_zscore']
    assert e.loc[pd.Timestamp('2020-01-07')] == pytest.approx(0.5), "Jan 7: only the Jan 3 reading is public."
    assert e.loc[pd.Timestamp('2020-01-14')] == pytest.approx(1.0), \
        "Jan 14: the Jan 17 reading is still FUTURE -> must use Jan 10, no look-ahead."
    assert e.loc[pd.Timestamp('2020-01-21')] == pytest.approx(2.0), "Jan 21: Jan 17 reading now public."
    assert pd.Timestamp('2020-01-28') not in out.index, "Final week has no forward return -> dropped."
    assert out['fwd_weekly_ret'].loc[pd.Timestamp('2020-01-07')] == pytest.approx(np.log(1.11 / 1.10)), \
        "Target is the strictly-forward (next-week) weekly log return."


def test_cot_weekly_contrarian_spread_signal_and_underpowered_guard():
    """Hypothesis-#2 core (src/cot_weekly_check._extreme_contrarian_spread,
    driving run_extremes): a clean crowded-long→negative / crowded-short→positive
    setup yields spread < 0 with a 95% CI entirely below 0 (KEEP-signal); and a
    tail with fewer than MIN_EXTREME_PER_SIDE extreme weeks is flagged
    non-computable (INCONCLUSIVE), never silently scored — the pre-registered
    underpowered guard that forbids loosening the |z|>1 cutoff to manufacture
    rows."""
    from src.cot_weekly_check import _extreme_contrarian_spread, MIN_EXTREME_PER_SIDE

    # Case A: strong contrarian signal, both tails well-populated (>= min/side).
    z = np.array([2.0] * 8 + [-2.0] * 8 + [0.0] * 20)
    y = np.array([-0.02] * 8 + [0.02] * 8 + [0.0] * 20)   # crowded-long -> neg, crowded-short -> pos
    a = _extreme_contrarian_spread(z, y, n_boot=500, random_state=0)
    assert a['n_long'] == 8 and a['n_short'] == 8 and a['computable']
    assert a['point_spread'] == pytest.approx(-0.04), "mean(-0.02) - mean(+0.02) = -0.04."
    assert a['ci_high'] < 0, "95% CI entirely below 0 -> contrarian signal."
    assert a['signal'] is True

    # Case B: one tail too thin (< MIN_EXTREME_PER_SIDE) -> inconclusive, not scored.
    z2 = np.array([2.0] * 2 + [-2.0] * 10 + [0.0] * 20)
    y2 = np.array([-0.02] * 2 + [0.02] * 10 + [0.0] * 20)
    b = _extreme_contrarian_spread(z2, y2, n_boot=500, random_state=0)
    assert b['n_long'] == 2 and b['n_long'] < MIN_EXTREME_PER_SIDE
    assert b['computable'] is False and b['signal'] is False, \
        "Too few crowded-long weeks -> INCONCLUSIVE, never a scored KEEP/DROP."


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


@pytest.mark.parametrize("tz_label", [None, 'UTC', 'Etc/GMT-2', 'Etc/GMT-3'])
def test_drop_incomplete_bars_strips_real_sunday_bar_any_server_offset(tz_label):
    """REGRESSION (data-integrity bug): pins the EXACT genuine Sunday D1 bar
    the owner saw live on their MT5 chart (2026-07-26 00:00 server time,
    O=1.13895 H=1.13954 L=1.13848 C=1.13880, MT5's characteristic thin
    partial-session tick_volume). Investigation (IMPROVEMENT_LOG.md) found
    MT5's raw epoch already bakes in the broker SERVER's own wall-clock date
    -- verified live against a real ActivTrades session and across four years
    of EU DST transitions with zero deviation -- so drop_incomplete_bars must
    strip this bar regardless of what tz label (if any) a caller's index
    happens to carry. Parametrized over a couple of plausible server-offset
    labels to prove that label is never load-bearing."""
    from src.live_data import drop_incomplete_bars

    idx = pd.to_datetime(['2026-07-23', '2026-07-24', '2026-07-26', '2026-07-27'])
    if tz_label is not None:
        idx = idx.tz_localize(tz_label)
    df = pd.DataFrame({
        'open':  [1.14083, 1.13774, 1.13895, 1.13809],
        'high':  [1.14354, 1.14004, 1.13954, 1.14177],
        'low':   [1.13633, 1.13641, 1.13848, 1.13616],
        'close': [1.13723, 1.13696, 1.13880, 1.13679],
        'tick_volume': [159409, 145453, 586, 138244],
    }, index=idx)

    trimmed = drop_incomplete_bars(df, now=pd.Timestamp('2026-07-28 09:00'))

    assert len(trimmed) == 3, "The Sunday bar must be dropped; Thu/Fri/Mon must survive."
    assert (trimmed.index.weekday < 5).all()
    assert 1.13880 not in trimmed['close'].values, \
        "The owner's exact Sunday close must never appear in a live-feed frame handed to the model."


def test_drop_incomplete_bars_never_raises_on_naive_aware_mismatch():
    """REGRESSION: D1 (`_fetch_from_mt5`, tz-naive) and H1/M15
    (`_fetch_h1_from_mt5`/`_fetch_m15_from_mt5`, tz-aware UTC) tag the
    identical MT5 wall-clock encoding differently. drop_incomplete_bars is
    the one shared choke point both PredictionService._resolve_latest_window
    and tracking._actual_closes call through, so it must never raise a
    naive-vs-aware TypeError -- and must still correctly drop the weekend bar
    -- regardless of which convention a caller's index/now happens to use."""
    from src.live_data import drop_incomplete_bars

    naive_idx = pd.to_datetime(['2026-07-23', '2026-07-24', '2026-07-26', '2026-07-27'])
    df_naive = pd.DataFrame({
        'open': [1.1] * 4, 'high': [1.1] * 4, 'low': [1.1] * 4, 'close': [1.1] * 4,
        'tick_volume': [1] * 4,
    }, index=naive_idx)

    # tz-aware `now` against a tz-naive index must not raise.
    out1 = drop_incomplete_bars(df_naive, now=pd.Timestamp('2026-07-28 09:00', tz='UTC'))
    assert len(out1) == 3
    assert '2026-07-26' not in [str(d.date()) for d in out1.index]

    # tz-aware index against a tz-naive `now` must not raise.
    df_aware = df_naive.copy()
    df_aware.index = naive_idx.tz_localize('UTC')
    out2 = drop_incomplete_bars(df_aware, now=pd.Timestamp('2026-07-28 09:00'))
    assert len(out2) == 3
    assert '2026-07-26' not in [str(d.date()) for d in out2.index]


def test_resolve_latest_window_as_of_date_never_lands_on_weekend(monkeypatch):
    """REGRESSION: PredictionService._resolve_latest_window must never set
    as_of_date to a Saturday/Sunday, even when the live feed's raw tail
    (as MT5 genuinely does) includes a trailing weekend bar. Builds a full
    weekday+weekend daily history (so SMA_200/lag warm-up is satisfied) that
    ends on a Sunday, mimicking the owner's live feed shape, and drives the
    exact production code path (fetch -> drop_incomplete_bars -> compute_features)
    with no network/model dependency."""
    import src.inference as inference
    from src.inference import PredictionService

    dates = pd.date_range('2024-01-01', periods=400, freq='D')
    last_sunday = dates[dates.weekday == 6][-1]
    dates = dates[dates <= last_sunday]
    assert dates[-1].weekday() == 6 and dates[-2].weekday() == 5, "fixture must end Sat/Sun like a real MT5 tail"

    rng = np.random.default_rng(7)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.003, len(dates))), index=dates)
    df = pd.DataFrame({
        'open': close.values, 'high': close.values + 0.002,
        'low': close.values - 0.002, 'close': close.values,
        'tick_volume': np.where(dates.weekday < 5, 100000, 500),
    }, index=dates)

    monkeypatch.setattr(inference, 'fetch_live_market_data', lambda *a, **k: (df, 'MT5'))
    monkeypatch.setattr(inference, 'fetch_macro_features', lambda *a, **k: (None, {}))

    svc = PredictionService.__new__(PredictionService)
    svc.config = {'data': {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X', 'live_fetch_bars': 250}, 'macro': {}}
    svc.base_dir = '.'
    svc.history_df = None

    _, data_source, bar_used, as_of_date, forecasting_date = svc._resolve_latest_window(20)

    expected_as_of = dates[dates.weekday < 5][-1]
    assert data_source == 'MT5'
    assert pd.Timestamp(as_of_date).weekday() < 5, "as_of_date must never fall on a Saturday/Sunday."
    assert as_of_date == expected_as_of.date().isoformat(), \
        "Must resolve to the last completed WEEKDAY session, not the trailing weekend bar."
    assert bar_used['date'] == expected_as_of.date().isoformat()


def test_resolve_latest_window_as_of_close_anchors_on_last_weekday_not_phantom_weekend_bar(monkeypatch):
    """SHARPER REGRESSION for the Sunday-bar bug, at its actual point of
    corruption. forecasting_date's own label math is NOT a reliable witness
    here: Sunday's weekday (6) is not in the `{Fri:3, Sat:2}` lookup so it
    falls through to the generic `+1 day` default and lands on Monday
    2026-07-27 -- the EXACT SAME date the correct Friday(+3 days) path also
    produces. A test that only checks forecasting_date would pass whether or
    not as_of_date/as_of_close were silently corrupted to the phantom bar.
    The real risk is as_of_close (`bar_used['close']`, the reference price
    both the live prediction AND the later scored "actual return %" are
    anchored to) and every feature computed relative to "the latest bar"
    (lag returns, ATR, SMAs, ...) silently switching to the weekend bar's
    numbers. Mocks a live feed shaped exactly like MT5's real output (no
    Saturday row at all -- MT5 skips straight from Friday to a single
    partial Sunday bar, confirmed live against ActivTrades this session)
    with the owner's exact reported Sunday OHLC (O=1.13895 H=1.13954
    L=1.13848 C=1.13880) as the MOST RECENT raw bar, and pins as_of_date,
    bar_used['close'], AND the feature_window's own last 'close' value to
    Friday's real close -- not Sunday's."""
    import src.inference as inference
    from src.inference import PredictionService

    FRIDAY_CLOSE = 1.13696    # real 2026-07-24 close (live-reproduced this session)
    SUNDAY_CLOSE = 1.13880    # owner's exact reported phantom-bar close

    weekdays = pd.bdate_range(end='2026-07-24', periods=260)
    rng = np.random.default_rng(3)
    close = 1.10 + np.cumsum(rng.normal(0, 0.003, len(weekdays)))
    close[-1] = FRIDAY_CLOSE
    df = pd.DataFrame({
        'open': close, 'high': close + 0.002, 'low': close - 0.002, 'close': close,
        'tick_volume': np.full(len(weekdays), 100000),
    }, index=weekdays)

    sunday = pd.Timestamp('2026-07-26')
    df.loc[sunday] = {
        'open': 1.13895, 'high': 1.13954, 'low': 1.13848, 'close': SUNDAY_CLOSE,
        'tick_volume': 586,
    }
    df = df.sort_index()
    assert df.index[-1] == sunday and df.index[-1].weekday() == 6
    assert df.index[-2] == pd.Timestamp('2026-07-24') and df.index[-2].weekday() == 4, \
        "fixture must skip Saturday entirely, exactly like the real MT5 feed"

    monkeypatch.setattr(inference, 'fetch_live_market_data', lambda *a, **k: (df, 'MT5'))
    monkeypatch.setattr(inference, 'fetch_macro_features', lambda *a, **k: (None, {}))

    svc = PredictionService.__new__(PredictionService)
    svc.config = {'data': {'symbol': 'EURUSD', 'live_symbol': 'EURUSD=X', 'live_fetch_bars': 250}, 'macro': {}}
    svc.base_dir = '.'
    svc.history_df = None

    feature_window, data_source, bar_used, as_of_date, forecasting_date = svc._resolve_latest_window(20)

    # The actual point of corruption: as_of_date/as_of_close must anchor on
    # Friday, never on the phantom Sunday bar.
    assert as_of_date == '2026-07-24'
    assert bar_used['close'] == pytest.approx(FRIDAY_CLOSE)
    assert bar_used['close'] != pytest.approx(SUNDAY_CLOSE)

    # The model's actual INPUT (not just response metadata) must be built off
    # Friday's row -- every lag/SMA/ATR feature is computed relative to this.
    assert feature_window['close'].iloc[-1] == pytest.approx(FRIDAY_CLOSE)

    # The label-only symptom this bug would NOT have shown: Sunday(weekday=6)
    # isn't in the Fri/Sat lookup, so it falls through to the generic +1-day
    # default and lands on the SAME Monday the correct Friday(+3) path
    # produces -- forecasting_date alone cannot distinguish a correct run
    # from one silently anchored on the phantom bar.
    assert forecasting_date == '2026-07-27'
    would_be_corrupted_forecasting_date = (sunday + pd.Timedelta(days=1)).date().isoformat()
    assert forecasting_date == would_be_corrupted_forecasting_date, \
        "proves forecasting_date's label math looks plausible either way -- " \
        "only as_of_close/feature_window pin the actual corruption point"


def test_actual_closes_never_returns_a_weekend_date_key(monkeypatch):
    """REGRESSION: the realised-close lookup used to score a logged forecast
    (tracking._actual_closes) must never key an entry off the owner's real
    Sunday bar (2026-07-26, O=1.13895 H=1.13954 L=1.13848 C=1.13880) -- that
    bar must be dropped by drop_incomplete_bars before the date->close dict
    is built, exactly like the live prediction path."""
    from src import tracking

    idx = pd.to_datetime(['2026-07-23', '2026-07-24', '2026-07-26', '2026-07-27', '2026-07-28'])
    df = pd.DataFrame({
        'open':  [1.14083, 1.13774, 1.13895, 1.13809, 1.13687],
        'high':  [1.14354, 1.14004, 1.13954, 1.14177, 1.13798],
        'low':   [1.13633, 1.13641, 1.13848, 1.13616, 1.13610],
        'close': [1.13723, 1.13696, 1.13880, 1.13679, 1.13619],
        'tick_volume': [159409, 145453, 586, 138244, 27491],
    }, index=idx)
    monkeypatch.setattr(tracking, 'fetch_live_market_data', lambda *a, **k: (df, 'stub'))

    closes = tracking._actual_closes({'symbol': 'EURUSD'}, now=pd.Timestamp('2026-07-28 09:00'))

    assert all(pd.Timestamp(d).weekday() < 5 for d in closes), \
        "No Saturday/Sunday key may ever appear in the realised-close lookup."
    assert '2026-07-26' not in closes
    assert closes == {'2026-07-23': 1.13723, '2026-07-24': 1.13696, '2026-07-27': 1.13679}


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


# ── Fibonacci / Williams-fractal features (hypothesis #7 + built-only #8) ─────

def test_detect_fractals_strict_5bar_extremum():
    """A Williams high fractal is a STRICT max of the 5-bar window high[i-2:i+3];
    a low fractal a strict min of low[i-2:i+3]. Flat runs are NOT fractals."""
    from src.fibonacci_fractals import detect_fractals

    #                0    1    2    3    4    5    6
    high = np.array([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])   # strict peak at 3
    low = np.array([9.0, 8.0, 7.0, 4.0, 7.0, 8.0, 9.0])    # strict trough at 3
    hf, lf = detect_fractals(high, low)
    assert hf.tolist() == [False, False, False, True, False, False, False]
    assert lf.tolist() == [False, False, False, True, False, False, False]

    # A flat top (tie) is not a strict maximum -> no fractal.
    flat = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0])
    hf2, _ = detect_fractals(flat, np.zeros(7))
    assert not hf2.any(), "A tie at the top is not a STRICT max -> not a fractal."


def test_fractal_confirmation_lag_no_lookahead():
    """CONFIRMATION LAG is the whole risk: a fractal forming at bar i is only
    knowable at bar i+2. It must be INVISIBLE on bars i and i+1 and first appear
    on bar i+2 -- mirrors the FRED/COT no-look-ahead guards."""
    from src.fibonacci_fractals import confirmed_high_low_levels

    #                0    1    2    3    4    5    6   -> high fractal forms at i=3
    high = np.array([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])
    low = np.array([1.0] * 7)
    conf_high, _conf_low = confirmed_high_low_levels(high, low)
    assert np.isnan(conf_high[3]), "fractal at bar i is invisible ON bar i (not yet confirmed)"
    assert np.isnan(conf_high[4]), "still invisible on bar i+1 (needs i+2 to confirm)"
    assert conf_high[5] == pytest.approx(5.0), "confirmed and first usable exactly on bar i+2"
    assert conf_high[6] == pytest.approx(5.0), "carries forward as the most recent confirmed high"


def test_fractal_breakout_features_semantics_and_no_lookahead():
    """fractal_breakout_up/down compare close to the most recent CONFIRMED
    fractal level (revealed at i+2). Before confirmation the breakout cannot
    fire off a still-forming fractal."""
    from src.fibonacci_fractals import add_fibonacci_features, FIBONACCI_FEATURE_COLUMNS

    idx = pd.date_range('2020-01-01', periods=7, freq='D')
    df = pd.DataFrame({
        'open': [1.0] * 7,
        'high': [1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 6.0],   # high fractal at i=3 (level 5)
        'low': [1.0] * 7,
        'close': [1.0, 1.0, 1.0, 4.9, 1.0, 1.0, 5.5],  # bar 6 closes above the confirmed 5
    }, index=idx)
    out = add_fibonacci_features(df)
    assert set(FIBONACCI_FEATURE_COLUMNS).issubset(out.columns)
    # Bar 3 (fractal-forming bar) closes 4.9 < 5, and the fractal is not even
    # confirmed yet -> no breakout regardless.
    assert out['fractal_breakout_up'].iloc[3] == 0.0
    # Bar 6 closes 5.5 > the confirmed high-fractal level 5 -> breakout up.
    assert out['fractal_breakout_up'].iloc[6] == 1.0
    assert out['fractal_breakout_down'].iloc[6] == 0.0


def test_fibonacci_features_are_nan_safe_neutral_zero():
    """Every hypothesis-#7 column must be fully defined (neutral 0 where no
    confirmed structure exists yet) -- never NaN -- so the ablation harness's
    'extra candidate columns must be fully defined' assert always holds."""
    from src.fibonacci_fractals import add_fibonacci_features, FIBONACCI_FEATURE_COLUMNS

    idx = pd.date_range('2020-01-01', periods=40, freq='D')
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 0.5, 40)) + 100
    df = pd.DataFrame({'open': base, 'high': base + 1, 'low': base - 1, 'close': base}, index=idx)
    out = add_fibonacci_features(df)
    assert not out[FIBONACCI_FEATURE_COLUMNS].isna().any().any(), "no NaN allowed in the bundle"
    # No swing has completed on the first two bars -> retracement distance is the
    # neutral 0, never back-filled from a later swing.
    assert out['dist_to_nearest_fib_pct'].iloc[0] == 0.0


def test_dist_to_nearest_fib_pct_known_swing():
    """dist_to_nearest_fib_pct sits at 0 when close lands exactly on a Fibonacci
    retracement level of the confirmed 2-point swing, and is normalized by the
    swing range in percent."""
    from src.fibonacci_fractals import _retracement_dist

    # Uptrend swing L=0 -> H=100. 0.5 retracement level = 50.
    a = {'idx': 0, 'kind': 'L', 'level': 0.0}
    b = {'idx': 5, 'kind': 'H', 'level': 100.0}
    assert _retracement_dist(a, b, 50.0) == pytest.approx(0.0), "close on the 0.5 level -> 0"
    # close 60 -> nearest level is 0.382 retracement = 100 - 38.2 = 61.8;
    # distance (60 - 61.8)/100*100 = -1.8.
    assert _retracement_dist(a, b, 60.0) == pytest.approx(-1.8, abs=1e-6)
    # Degenerate swing (H == L) -> neutral 0, no division blow-up.
    flat = {'idx': 5, 'kind': 'H', 'level': 0.0}
    assert _retracement_dist(a, flat, 10.0) == 0.0


def test_fibonacci_extension_3point_chronology_and_confirmation_lag():
    """Hypothesis #8 (built, not tested this pass): the extension projects off a
    confirmed 3-point swing A->B->C in CHRONOLOGICAL order, and inherits the same
    confirmation-lag guard -- it stays neutral 0 until the THIRD alternating
    fractal is itself confirmed (i+2)."""
    from src.fibonacci_fractals import (
        add_fibonacci_extension_features, _extension_dist,
        FIBONACCI_EXTENSION_FEATURE_COLUMNS,
    )

    # Direct formula check: A=0 -> B=10 -> C=4, span=B-A=10, ext grid off C.
    # r=1.13 level = 4 + 11.3 = 15.3; close 15.3 lands on it -> distance 0.
    A = {'idx': 0, 'kind': 'L', 'level': 0.0}
    B = {'idx': 5, 'kind': 'H', 'level': 10.0}
    C = {'idx': 10, 'kind': 'L', 'level': 4.0}
    assert _extension_dist(A, B, C, 15.3) == pytest.approx(0.0, abs=1e-9)
    # Reversing the chronology (C,B,A) would project the OTHER direction and give
    # a different nearest level -> the order genuinely matters.
    assert _extension_dist(C, B, A, 15.3) != pytest.approx(0.0, abs=1e-6)

    # Confirmation-lag / warm-up: three alternating fractals must all be
    # confirmed before any extension distance is emitted; early bars are 0.
    idx = pd.date_range('2020-01-01', periods=25, freq='D')
    # Craft a low@2, high@7, low@12 fractal sequence via a zigzag price path.
    h = np.full(25, 1.0)
    lo = np.full(25, 5.0)
    lo[2] = 0.0                       # low fractal at i=2
    h[7] = 10.0                       # high fractal at i=7
    lo[12] = -1.0                     # low fractal at i=12
    close = np.linspace(1.0, 3.0, 25)
    df = pd.DataFrame({'open': close, 'high': np.maximum(h, close + 0.01),
                       'low': np.minimum(lo, close - 0.01), 'close': close}, index=idx)
    out = add_fibonacci_extension_features(df)
    assert set(FIBONACCI_EXTENSION_FEATURE_COLUMNS).issubset(out.columns)
    assert not out['dist_to_nearest_fib_extension_pct'].isna().any(), "NaN-safe"
    # The third fractal forms at i=12 and is confirmed at i=14; nothing before it.
    assert (out['dist_to_nearest_fib_extension_pct'].iloc[:14] == 0.0).all(), \
        "no 3-point extension before the third alternating fractal is confirmed (i+2)"


def test_fibonacci_features_stay_out_of_direction_return_models():
    """Ablation-first discipline: the Fibonacci/fractal candidates (both the
    tested #7 bundle and the built-only #8 extension) must NEVER leak into
    FEATURE_COLUMNS until a hypothesis test clears the family bar -- same guard
    as the volatility and FOMC candidates."""
    from src.fibonacci_fractals import (
        FIBONACCI_FEATURE_COLUMNS, FIBONACCI_EXTENSION_FEATURE_COLUMNS,
    )

    overlap = set(FIBONACCI_FEATURE_COLUMNS + FIBONACCI_EXTENSION_FEATURE_COLUMNS) & set(FEATURE_COLUMNS)
    assert not overlap, f"Fibonacci candidates must stay out of the model feature set: {overlap}"


# ── VIX (CBOE Volatility Index) regime features (hypothesis #8) ───────────────

def test_vix_availability_is_one_business_day_after_the_print_no_lookahead():
    """STEP 0 pin: FRED publishes VIXCLS with a business-day lag and the VIX
    close (~16:15 ET) is only ~45 min before the FX rollover (~17:00 ET), so a
    print dated D is treated as available on D+1 BUSINESS day (conservative D-1
    rule). `_compute_vix_frame` must stamp the availability index accordingly."""
    from src.vix_features import _compute_vix_frame

    level = pd.DataFrame(
        {'vix': [15.0, 16.0, 17.0]},
        index=pd.DatetimeIndex(['2020-01-06', '2020-01-07', '2020-01-08'], tz='UTC'),  # Mon,Tue,Wed
    )
    frame = _compute_vix_frame(level)
    # Availability = print date + 1 business day (Mon->Tue, Tue->Wed, Wed->Thu).
    assert frame.index[0] == pd.Timestamp('2020-01-07', tz='UTC')
    assert frame.index[1] == pd.Timestamp('2020-01-08', tz='UTC')
    assert frame.index[2] == pd.Timestamp('2020-01-09', tz='UTC')
    # The Tuesday print's day-over-day change (16/15-1)*100 is stamped at its
    # availability date, Wednesday.
    assert frame.loc[pd.Timestamp('2020-01-08', tz='UTC'), 'vix_change_pct'] == \
        pytest.approx((16.0 / 15.0 - 1.0) * 100.0)


def test_add_vix_features_ffill_by_availability_date_no_lookahead():
    """add_vix_features joins by AVAILABILITY date with as-of ffill: a daily bar
    must carry the last VIX reading ALREADY USABLE on that date, and must NOT see
    a reading whose availability date is still in the future."""
    from src.vix_features import add_vix_features, VIX_FEATURE_COLUMNS

    # Availability-dated readings (as _compute_vix_frame would stamp them).
    vix_frame = pd.DataFrame(
        {'vix_zscore': [1.0, 2.0], 'vix_change_pct': [10.0, 20.0]},
        index=pd.DatetimeIndex(['2020-01-07', '2020-01-08'], tz='UTC'),
    )
    days = pd.DatetimeIndex(['2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09'])
    df = pd.DataFrame({'close': [1.1] * len(days)}, index=days)

    out = add_vix_features(df, vix_frame=vix_frame)
    z = out['vix_zscore']
    assert z.loc[days[0]] == 0.0, "Jan 6: no reading usable yet -> neutral 0, never back-filled."
    assert z.loc[days[1]] == pytest.approx(1.0), "Jan 7: first reading now usable (its availability date)."
    assert z.loc[days[2]] == pytest.approx(2.0), "Jan 8: second reading usable."
    assert z.loc[days[3]] == pytest.approx(2.0), "Jan 9: carries the latest usable reading."
    assert set(VIX_FEATURE_COLUMNS).issubset(out.columns)


def test_vix_value_never_usable_on_the_bar_it_would_leak_into():
    """End-to-end conservative D-1 guard: build a native business-day VIX series,
    compute the availability-stamped frame, and join it onto same-dated FX bars.
    An FX bar on date D must reflect only prints dated <= D-1 (available by D)."""
    from src.vix_features import _compute_vix_frame, add_vix_features

    nat = pd.bdate_range('2020-01-06', periods=5, tz='UTC')          # Mon..Fri
    level = pd.DataFrame({'vix': [10.0, 11.0, 12.0, 13.0, 14.0]}, index=nat)
    frame = _compute_vix_frame(level)

    fx_days = pd.bdate_range('2020-01-06', periods=5)                # same dates, tz-naive FX bars
    fx = pd.DataFrame({'close': [1.1] * 5}, index=fx_days)
    out = add_vix_features(fx, vix_frame=frame)

    # Wed 01-08 may use at most Tue 01-07's print, whose change is Mon->Tue =
    # (11/10-1)*100 = 10.0 -- NOT Wed's own (12/11-1) print, which is not usable
    # until Thu. This is the whole no-look-ahead point.
    assert out['vix_change_pct'].loc[pd.Timestamp('2020-01-08')] == pytest.approx(10.0)
    assert out['vix_change_pct'].loc[pd.Timestamp('2020-01-08')] != \
        pytest.approx((12.0 / 11.0 - 1.0) * 100.0), "Wed must NOT see Wed's own print."
    # Tue 01-07: Mon's print (change NaN, first bar) is the only thing dated <= Mon,
    # so the bar is neutral 0 -- never a future-dated reading.
    assert out['vix_change_pct'].loc[pd.Timestamp('2020-01-07')] == 0.0


def test_vix_zscore_is_trailing_only_and_neutral_until_warmup():
    """The z-score at day t standardizes against the trailing window ending AT t
    (only past data), and stays undefined (-> neutral 0 downstream) before the
    min-periods history exists -- never computed from later days."""
    from src.vix_features import _compute_vix_frame, VIX_ZSCORE_MIN

    n = VIX_ZSCORE_MIN + 5
    nat = pd.bdate_range('2015-01-01', periods=n, tz='UTC')
    level = pd.DataFrame({'vix': np.arange(1.0, n + 1.0)}, index=nat)   # strictly rising
    frame = _compute_vix_frame(level)
    z = frame['vix_zscore']
    # Rising series -> the latest day sits above its trailing mean -> positive z.
    assert z.iloc[-1] > 0, "A rising series' latest day is above its trailing mean."
    # No z-score may exist before the trailing window has min_periods of history.
    assert z.iloc[:VIX_ZSCORE_MIN - 1].isna().all(), \
        "No z-score before the trailing window has min_periods of history."


def test_add_vix_features_neutral_zero_when_feed_unavailable(monkeypatch):
    """A completely unreachable VIX feed (no API/public/cache) must neutralize
    every VIX column to 0 so the pipeline degrades, never hard-fails -- the same
    contract as every other macro/COT feature."""
    import src.vix_features as vix_features

    monkeypatch.setattr(vix_features, 'fetch_vix_features', lambda *a, **k: (None, 'unavailable'))
    days = pd.DatetimeIndex(['2020-01-06', '2020-01-07', '2020-01-08'])
    df = pd.DataFrame({'close': [1.1] * len(days)}, index=days)
    out = vix_features.add_vix_features(df, vix_frame=None)
    assert (out['vix_zscore'] == 0.0).all() and (out['vix_change_pct'] == 0.0).all()


def test_vix_features_stay_out_of_direction_return_models():
    """Ablation-first discipline: the VIX candidates must NEVER leak into
    FEATURE_COLUMNS until a hypothesis test clears the family bar. The raw VIX
    LEVEL is also deliberately kept out of the always-merged macro columns."""
    from src.vix_features import VIX_FEATURE_COLUMNS
    from src.features import MACRO_MERGE_COLUMNS

    assert not set(VIX_FEATURE_COLUMNS) & set(FEATURE_COLUMNS), \
        "VIX candidate features must stay out of the model feature set."
    assert 'vix' not in MACRO_MERGE_COLUMNS, \
        "The raw VIX level must not be merged into the served model frame (ablation-only)."


# ── Cross-family reuse: FROZEN volatility ensemble -> direction/return
# candidate feature (hypothesis #9) ────────────────────────────────────────

def test_frozen_volatility_ensemble_batch_inference_never_fits(monkeypatch):
    """src/volatility.py::add_volatility_forecast_feature /
    batch_predict_frozen_ensemble_vol_pct (direction/return hypothesis #9's
    cross-family reuse of the FROZEN production volatility ensemble) must
    perform PURE INFERENCE -- no fitting step may ever be triggered by this
    code path, regardless of which rows (train/val/test) are passed in `feat`.
    This is what makes reusing the once-fit-on-train-block ensemble across the
    WHOLE historical row set look-ahead-safe: it mirrors the SAME idiom
    src/ablation.py::build_matrix already uses (a PCA fit once on a train
    slice, then apply_lag_pca'd across the full dataset including val/test) --
    nothing here re-fits or peeks at any row to produce another row's
    prediction."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from src.features import PRICE_FEATURE_COLUMNS, LAG_COLUMNS
    from src.volatility import (
        add_volatility_forecast_feature, batch_predict_frozen_ensemble_vol_pct,
    )

    n_lag = len(LAG_COLUMNS)
    rng = np.random.default_rng(0)
    # Fit tiny stand-in artifacts BEFORE the fit-guard below -- this setup
    # fitting is not part of the code path under test (it stands in for the
    # real models/volatility/ artifacts, already fit once historically).
    lag_scaler = StandardScaler().fit(rng.normal(size=(30, n_lag)))
    lag_pca = PCA(n_components=3).fit(lag_scaler.transform(rng.normal(size=(30, n_lag))))
    n_cols_after_pca = len(PRICE_FEATURE_COLUMNS) - n_lag + lag_pca.n_components_
    global_scaler = StandardScaler().fit(rng.normal(size=(30, n_cols_after_pca)))

    class _FakeSeedModel:
        """Mimics the 3-head Keras multi-task model's .predict() interface
        (return_output, direction_output, volatility_output) with no real
        model load -- keeps this test fast and dependency-light."""
        def predict(self, windows, verbose=0):
            n = windows.shape[0]
            zeros = np.zeros((n, 1), dtype='float32')
            vol = np.full((n, 1), 0.42, dtype='float32')
            return zeros, zeros, vol

    artifacts = {
        'lag_scaler': lag_scaler, 'lag_pca': lag_pca, 'global_scaler': global_scaler,
        'time_steps': 5, 'models': [_FakeSeedModel(), _FakeSeedModel()],
    }

    # NOW guard fit/fit_transform on BOTH classes -- proves the code path
    # under test never re-fits, whichever instance it touches.
    def _explode(self, *a, **k):
        raise AssertionError('fit() must never be called by frozen-ensemble batch inference')
    monkeypatch.setattr(StandardScaler, 'fit', _explode)
    monkeypatch.setattr(StandardScaler, 'fit_transform', _explode)
    monkeypatch.setattr(PCA, 'fit', _explode)
    monkeypatch.setattr(PCA, 'fit_transform', _explode)

    idx = pd.date_range('2020-01-01', periods=20, freq='D')
    feat = pd.DataFrame({c: rng.normal(size=20) for c in PRICE_FEATURE_COLUMNS}, index=idx)

    # Exercised across the FULL synthetic row range (stands in for
    # train+val+test all at once) -- must not raise despite spanning "test" rows.
    pred = batch_predict_frozen_ensemble_vol_pct(feat, artifacts)
    assert len(pred) == len(feat)
    time_steps = artifacts['time_steps']
    assert np.isnan(pred[:time_steps - 1]).all(), "no prediction before the first full window"
    assert pred[time_steps - 1:] == pytest.approx(0.42)

    out = add_volatility_forecast_feature(feat, artifacts=artifacts)
    assert 'predicted_vol_pct' in out.columns
    assert not out['predicted_vol_pct'].isna().any(), \
        "warm-up rows must be neutral-filled (0.0), never NaN"
    assert out['predicted_vol_pct'].iloc[0] == 0.0
    assert out['predicted_vol_pct'].iloc[-1] == pytest.approx(0.42)


def test_volatility_forecast_feature_stays_out_of_direction_return_feature_columns():
    """Ablation-first discipline: predicted_vol_pct must NEVER leak into
    FEATURE_COLUMNS until a hypothesis test clears the family bar -- same
    guard as every other candidate module."""
    from src.volatility import VOLATILITY_FORECAST_FEATURE_COLUMNS

    assert not set(VOLATILITY_FORECAST_FEATURE_COLUMNS) & set(FEATURE_COLUMNS), \
        "predicted_vol_pct must stay out of the model feature set."


# ── Harmonic patterns (src.harmonic_patterns) + triple-barrier labeling
# (src.triple_barrier) + the event-conditional model pipeline
# (src.harmonic_event_check) -- own hypothesis family, first budget ─────────

def _zigzag(points, seg_len=6):
    """Strictly-monotonic zigzag path through `points` (each an exact
    turning-point level): every interior point is a genuine local extremum
    (no flat plateaus, which would break detect_fractals' STRICT inequality),
    with `seg_len` bars per leg -- enough for the 5-bar fractal window plus
    the 2-bar confirmation lag to resolve cleanly."""
    path = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        path.extend(np.linspace(a, b, seg_len)[1:].tolist())
    return np.array(path)


# Exact Gartley by construction: X=0(L) -> A=100(H) -> B=38.2(L) [r_AB=0.618
# of XA] -> C=76.39(H) [r_BC=0.618 of AB] -> D=21.4(L) [r_AD=0.786 of XA].
_GARTLEY_POINTS = [5.0, 0.0, 100.0, 38.2, 76.39, 21.4, 30.0]


def test_score_xabcd_exact_gartley_scores_1_and_bullish_direction():
    """A hand-computed exact-Gartley 5-point swing must score best_fit_score
    == 1.0 against the 'gartley' template and report the correct ratios and
    bullish (+1) direction (D is a LOW -> upward reversal)."""
    from src.harmonic_patterns import score_xabcd

    X = {'idx': 0, 'kind': 'L', 'level': 0.0}
    A = {'idx': 5, 'kind': 'H', 'level': 100.0}
    B = {'idx': 10, 'kind': 'L', 'level': 38.2}
    C = {'idx': 15, 'kind': 'H', 'level': 76.39}
    D = {'idx': 20, 'kind': 'L', 'level': 21.4}
    res = score_xabcd(X, A, B, C, D)
    assert res['pattern'] == 'gartley'
    assert res['best_fit_score'] == pytest.approx(1.0)
    assert res['r_AB'] == pytest.approx(0.618)
    assert res['r_AD'] == pytest.approx(0.786)
    assert res['direction'] == 1, "D is a LOW -> bullish reversal up"

    # Mirror image (bearish): swap H/L kinds and negate levels -> same ratios,
    # opposite sign direction.
    Xb = {'idx': 0, 'kind': 'H', 'level': 0.0}
    Ab = {'idx': 5, 'kind': 'L', 'level': -100.0}
    Bb = {'idx': 10, 'kind': 'H', 'level': -38.2}
    Cb = {'idx': 15, 'kind': 'L', 'level': -76.39}
    Db = {'idx': 20, 'kind': 'H', 'level': -21.4}
    res_b = score_xabcd(Xb, Ab, Bb, Cb, Db)
    assert res_b['best_fit_score'] == pytest.approx(1.0)
    assert res_b['direction'] == -1, "D is a HIGH -> bearish reversal down"


def test_score_xabcd_degenerate_swing_returns_none():
    """A zero-length leg (XA, AB, or BC) makes every ratio undefined -- must
    return None rather than divide by zero."""
    from src.harmonic_patterns import score_xabcd

    X = {'idx': 0, 'kind': 'L', 'level': 50.0}
    A = {'idx': 5, 'kind': 'H', 'level': 50.0}   # XA == 0
    B = {'idx': 10, 'kind': 'L', 'level': 40.0}
    C = {'idx': 15, 'kind': 'H', 'level': 45.0}
    D = {'idx': 20, 'kind': 'L', 'level': 41.0}
    assert score_xabcd(X, A, B, C, D) is None


def test_detect_harmonic_events_confirmation_lag_and_end_to_end_geometry():
    """End-to-end: a synthetic exact-Gartley zigzag must be detected with the
    correct X/A/B/C/D bar indices, scored 1.0, and -- the confirmation-lag
    guard, mirroring src.fibonacci_fractals' CONFIRMATION_LAG -- INVISIBLE
    until confirmed_at_idx == D_idx + 2, never one bar earlier."""
    from src.harmonic_patterns import detect_harmonic_events, CONFIRMATION_LAG

    path = _zigzag(_GARTLEY_POINTS, seg_len=6)
    df = pd.DataFrame({'high': path, 'low': path, 'close': path})
    events = detect_harmonic_events(df)

    assert len(events) == 1
    ev = events.iloc[0]
    assert (ev['X_idx'], ev['A_idx'], ev['B_idx'], ev['C_idx'], ev['D_idx']) == (5, 10, 15, 20, 25)
    assert ev['confirmed_at_idx'] == ev['D_idx'] + CONFIRMATION_LAG == 27
    assert ev['pattern'] == 'gartley'
    assert ev['best_fit_score'] == pytest.approx(1.0)
    assert ev['direction'] == 1
    assert ev['harmonic_pattern_score_signed'] == pytest.approx(1.0)   # direction(+1) * score(1.0)

    # INVISIBLE one bar before confirmation (truncate to D_idx+1 inclusive).
    trunc_before = detect_harmonic_events(df.iloc[:ev['D_idx'] + 2])   # rows 0..D_idx+1
    assert len(trunc_before) == 0, "event must not exist before its own confirmed_at_idx"

    # Present exactly AT confirmation (truncate to confirmed_at_idx inclusive).
    trunc_at = detect_harmonic_events(df.iloc[:ev['confirmed_at_idx'] + 1])
    assert len(trunc_at) == 1
    assert trunc_at.iloc[0]['D_idx'] == ev['D_idx']


def test_detect_harmonic_events_no_lookahead_via_future_truncation():
    """Explicit no-look-ahead check: re-running detection on a series
    truncated well PAST an event's confirmation must reproduce the identical
    event geometry -- nothing about an already-confirmed event may depend on
    bars beyond its own confirmed_at_idx. Mirrors
    test_ti_indicators_no_lookahead_via_future_truncation's truncation-
    equivalence convention."""
    from src.harmonic_patterns import detect_harmonic_events

    # Extra trailing bars (a further leg) AFTER the Gartley completes+confirms.
    path = _zigzag(_GARTLEY_POINTS + [90.0, 15.0], seg_len=6)
    df = pd.DataFrame({'high': path, 'low': path, 'close': path})
    full = detect_harmonic_events(df)
    gartley_row = full[full['pattern'] == 'gartley'].iloc[0]

    cut = int(gartley_row['confirmed_at_idx']) + 1   # just past confirmation, well before the extra legs
    trunc = detect_harmonic_events(df.iloc[:cut])
    trunc_gartley = trunc[trunc['pattern'] == 'gartley'].iloc[0]
    for col in ['X_idx', 'A_idx', 'B_idx', 'C_idx', 'D_idx', 'confirmed_at_idx',
               'pattern', 'best_fit_score', 'r_AB', 'r_BC', 'r_CD', 'r_AD', 'direction']:
        assert trunc_gartley[col] == pytest.approx(gartley_row[col]) if isinstance(gartley_row[col], float) \
            else trunc_gartley[col] == gartley_row[col]


# ── triple-barrier labeling correctness ───────────────────────────────────

def _flat_series(entry, n=10):
    return np.full(n, entry), np.full(n, entry), np.full(n, entry)


def test_triple_barrier_target_touched_first():
    from src.triple_barrier import triple_barrier_label

    entry, hv = 100.0, 0.01
    target = entry * np.exp(1.5 * hv)
    high, low, close = _flat_series(entry)
    high[3] = target + 0.001
    close[3] = target
    label, outcome = triple_barrier_label(high, low, close, 0, 1, hv, 5, cost_price=0.00015)
    assert (label, outcome) == (1, 'target')


def test_triple_barrier_stop_touched_first():
    from src.triple_barrier import triple_barrier_label

    entry, hv = 100.0, 0.01
    stop = entry * np.exp(-1.0 * hv)
    high, low, close = _flat_series(entry)
    low[2] = stop - 0.001
    label, outcome = triple_barrier_label(high, low, close, 0, 1, hv, 5, cost_price=0.00015)
    assert (label, outcome) == (0, 'stop')


def test_triple_barrier_time_barrier_above_cost_threshold_wins():
    """Neither barrier touched within the horizon; the signed move at the
    time barrier EXCEEDS the transaction-cost threshold -> label 1."""
    from src.triple_barrier import triple_barrier_label

    entry, hv, cost = 100.0, 0.01, 0.00015
    high, low, close = _flat_series(entry)
    close[5] = entry + 0.0003   # > cost
    label, outcome = triple_barrier_label(high, low, close, 0, 1, hv, 5, cost_price=cost)
    assert (label, outcome) == (1, 'time_win')


def test_triple_barrier_time_barrier_below_cost_threshold_loses():
    """Neither barrier touched; the signed move at the time barrier is
    POSITIVE but smaller than the transaction-cost threshold -> label 0 (a
    move smaller than the spread is not a realizable win, mirroring
    src.paper_trading's cost-netting -- never a bare sign(>0))."""
    from src.triple_barrier import triple_barrier_label

    entry, hv, cost = 100.0, 0.01, 0.00015
    high, low, close = _flat_series(entry)
    close[5] = entry + 0.00005   # positive but < cost
    label, outcome = triple_barrier_label(high, low, close, 0, 1, hv, 5, cost_price=cost)
    assert (label, outcome) == (0, 'time_loss')


def test_triple_barrier_insufficient_history_excluded_not_padded():
    from src.triple_barrier import triple_barrier_label

    entry, hv = 100.0, 0.01
    high, low, close = _flat_series(entry, n=10)
    label, outcome = triple_barrier_label(high, low, close, 6, 1, hv, 5, cost_price=0.00015)
    assert (label, outcome) == (None, 'insufficient_history')


def test_triple_barrier_short_direction_mirrors_long():
    """direction=-1 flips which side is target vs stop: the barrier BELOW
    entry is now the (aligned) target, the barrier ABOVE is the (opposite)
    stop."""
    from src.triple_barrier import triple_barrier_label

    entry, hv = 100.0, 0.01
    target = entry * np.exp(-1.5 * hv)   # aligned with direction=-1 -> below entry
    stop = entry * np.exp(1.0 * hv)      # opposite direction=-1 -> above entry
    high, low, close = _flat_series(entry)
    low[3] = target - 0.001
    close[3] = target
    label, outcome = triple_barrier_label(high, low, close, 0, -1, hv, 5, cost_price=0.00015)
    assert (label, outcome) == (1, 'target')

    high2, low2, close2 = _flat_series(entry)
    high2[2] = stop + 0.001
    label2, outcome2 = triple_barrier_label(high2, low2, close2, 0, -1, hv, 5, cost_price=0.00015)
    assert (label2, outcome2) == (0, 'stop')


# ── horizon-matched volatility: sqrt(120)-scaled EWMA std ─────────────────

def test_horizon_vol_is_ewma_std_times_sqrt_horizon():
    """Mathematical check: horizon_vol = r_ewma_std * sqrt(horizon_bars),
    computed BEFORE any exponential price translation -- exactly
    np.sqrt(120), not some other scaling."""
    from src.triple_barrier import horizon_vol_from_ewma_std

    r_ewma_std = np.array([0.001, 0.002, np.nan, 0.0005])
    result = horizon_vol_from_ewma_std(r_ewma_std, 120)
    expected = r_ewma_std * np.sqrt(120)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    # Hand-check one value explicitly (guards against a stray sqrt(horizon-1)
    # or missing sqrt entirely).
    assert result[0] == pytest.approx(0.001 * (120 ** 0.5))


def test_ewma_log_return_std_is_causal_and_scales_barriers_correctly():
    """ewma_log_return_std must be a pure trailing/causal function (identical
    on the surviving rows after truncating the future -- pandas .ewm is
    inherently causal, but this pins it explicitly), and the resulting
    horizon_vol must be EXACTLY what triple_barrier_label uses to place its
    exponential target/stop barriers (end-to-end math check)."""
    from src.triple_barrier import (
        ewma_log_return_std, horizon_vol_from_ewma_std, triple_barrier_label,
    )

    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 40)))
    full = ewma_log_return_std(close, span=24)
    cut = 25
    trunc = ewma_log_return_std(close[:cut], span=24)
    np.testing.assert_allclose(full[:cut], trunc, equal_nan=True)

    horizon_vol = horizon_vol_from_ewma_std(full, 120)
    entry_idx = 10
    hv = horizon_vol[entry_idx]
    high = close.copy() * 1.0000001
    low = close.copy() * 0.9999999
    expected_target = close[entry_idx] * np.exp(1.5 * hv)
    expected_stop = close[entry_idx] * np.exp(-1.0 * hv)
    # Force a target touch at the expected (correctly-scaled) level and
    # confirm the label fires -- if the sqrt(120) scaling were wrong, this
    # exact price would not trigger it.
    high[entry_idx + 3] = expected_target + 1e-6
    label, outcome = triple_barrier_label(high, low, close, entry_idx, 1, hv,
                                          20, cost_price=0.00015)
    assert (label, outcome) == (1, 'target')
    assert expected_stop < close[entry_idx] < expected_target


# ── the event pipeline's generic paired-bootstrap comparison ─────────────

def test_bootstrap_delta_and_mcnemar_is_a_direct_paired_comparison():
    """H1.2's PRIMARY decision path must compare against H1.1's OWN
    predictions on IDENTICAL validation rows -- not an independently
    resampled/regenerated baseline. Verify bootstrap_delta_and_mcnemar's
    McNemar b/c are computed directly from the two prediction arrays passed
    in (hand-computable), so swapping in H1.1's actual predictions as
    `pred_reference` is a genuine row-for-row comparison, not a proxy."""
    from src.harmonic_event_check import bootstrap_delta_and_mcnemar

    y_val = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    pred_reference = np.array([1, 0, 0, 0, 0, 0, 1, 1])   # (H1.1-style) 6/8 correct
    pred_challenger = np.array([1, 0, 1, 0, 1, 1, 1, 1])  # (H1.2-style) 6/8 correct, different rows right

    res = bootstrap_delta_and_mcnemar(y_val, pred_reference, pred_challenger,
                                      alpha=0.025, random_state=42)
    correct_ref = (pred_reference == y_val)
    correct_chg = (pred_challenger == y_val)
    # Hand-computed discordant pairs: reference wrong & challenger right (b),
    # reference right & challenger wrong (c).
    expected_b = int(np.sum((~correct_ref) & correct_chg))
    expected_c = int(np.sum(correct_ref & (~correct_chg)))
    assert res['mcnemar_b'] == expected_b
    assert res['mcnemar_c'] == expected_c
    assert res['acc_reference'] == pytest.approx(correct_ref.mean())
    assert res['acc_challenger'] == pytest.approx(correct_chg.mean())

    # Calling it again with pred_reference and pred_challenger SWAPPED must
    # flip which array is "reference" -- proving there is no hidden internal
    # baseline independent of the two arrays actually supplied.
    swapped = bootstrap_delta_and_mcnemar(y_val, pred_challenger, pred_reference,
                                          alpha=0.025, random_state=42)
    assert swapped['mcnemar_b'] == expected_c
    assert swapped['mcnemar_c'] == expected_b
    assert swapped['acc_reference'] == pytest.approx(correct_chg.mean())
    assert swapped['acc_challenger'] == pytest.approx(correct_ref.mean())


def test_harmonic_features_stay_out_of_daily_feature_columns():
    """Ablation-first discipline, applied even though this is a different
    event universe/target entirely: nothing from the harmonic-pattern module
    may leak into the daily direction/return FEATURE_COLUMNS."""
    from src.harmonic_patterns import HARMONIC_EVENT_COLUMNS
    from src.harmonic_event_check import MODEL_FEATURE_COLUMNS

    assert not (set(HARMONIC_EVENT_COLUMNS) | set(MODEL_FEATURE_COLUMNS)) & set(FEATURE_COLUMNS)


# ── ZigZag swings (src.zigzag_swings) -- the alternative, causal ATR-scaled
# swing basis for the harmonic-pattern hypotheses H1.3/H1.4. Highest-priority
# test set in this whole family given the ELEVATED look-ahead risk of a
# variable (unbounded) confirmation lag vs. the fractal path's trivial fixed
# one -- these tests exist specifically to rule out repainting. ────────────

def _zigzag_test_path():
    """A synthetic price path built to produce ONE confirmed HIGH pivot whose
    reveal_bar is DEMONSTRABLY later than its own pivot bar: a quiet warm-up
    (stable small ATR), a clean rise to a peak, several bars that wobble just
    under the ATR-scaled threshold (the peak stays an UNCONFIRMED candidate),
    then a genuine drop that finally exceeds the threshold and confirms it."""
    warmup = 100.0 + np.sin(np.linspace(0, 3, 20)) * 0.05   # tiny noise -> small, stable ATR
    rise = np.linspace(warmup[-1], 110.0, 10)                # clean rise to a peak (last point)
    wobble = np.array([109.9, 109.95, 109.8, 109.9])         # stays under threshold -> no confirm yet
    drop = np.linspace(109.8, 95.0, 10)                      # eventually exceeds threshold -> confirms
    return np.concatenate([warmup, rise, wobble, drop])


def test_zigzag_pivot_invisible_before_reveal_bar():
    """The elevated-risk-specific look-ahead guard: a pivot must be
    PROVABLY INVISIBLE to any query truncated to end before its own
    reveal_bar (including the pivot_bar itself and every bar up to
    reveal_bar-1), and present exactly once the query reaches reveal_bar."""
    from src.zigzag_swings import zigzag_swings

    path = _zigzag_test_path()
    pivots_full = zigzag_swings(path, path, path)
    high_pivots = [p for p in pivots_full if p['kind'] == 'H'
                  and p['level'] == pytest.approx(110.0, abs=0.01)]
    assert len(high_pivots) == 1, "expected exactly one confirmed high pivot at the constructed peak"
    peak = high_pivots[0]
    assert peak['reveal_bar'] > peak['idx'], \
        "reveal_bar must be strictly LATER than the pivot's own bar (a genuine reveal lag)"

    # Invisible for every bar up to reveal_bar - 1 inclusive.
    cut = peak['reveal_bar']
    trunc_before = zigzag_swings(path[:cut], path[:cut], path[:cut])
    assert not any(p['idx'] == peak['idx'] and p['kind'] == 'H' for p in trunc_before), \
        "pivot must not exist (even unconfirmed) in any query ending before its own reveal_bar"

    # Present exactly AT reveal_bar.
    trunc_at = zigzag_swings(path[:cut + 1], path[:cut + 1], path[:cut + 1])
    matches = [p for p in trunc_at if p['idx'] == peak['idx'] and p['kind'] == 'H']
    assert len(matches) == 1
    assert matches[0]['reveal_bar'] == peak['reveal_bar']
    assert matches[0]['level'] == pytest.approx(peak['level'])


def test_zigzag_confirmed_pivot_is_stable_under_future_extension():
    """THE core repainting guard: a CONFIRMED pivot's (idx, level, reveal_bar)
    must be IDENTICAL whether computed causally up to its own reveal_bar, or
    with arbitrarily more future bars (a further whipsaw and a new trend)
    appended afterward. This is the property a naive 'scan the whole series
    for local extrema' implementation would violate."""
    from src.zigzag_swings import zigzag_swings

    path = _zigzag_test_path()
    extra = np.concatenate([np.linspace(95.0, 130.0, 15), np.linspace(130.0, 80.0, 15)])
    extended = np.concatenate([path, extra])

    pivots_short = zigzag_swings(path, path, path)
    pivots_long = zigzag_swings(extended, extended, extended)

    peak_short = [p for p in pivots_short if p['kind'] == 'H'
                 and p['level'] == pytest.approx(110.0, abs=0.01)][0]
    matches_long = [p for p in pivots_long if p['idx'] == peak_short['idx'] and p['kind'] == 'H']
    assert len(matches_long) == 1, "the same pivot bar must still be confirmed as a high in the extended series"
    peak_long = matches_long[0]

    assert peak_long['level'] == pytest.approx(peak_short['level'])
    assert peak_long['reveal_bar'] == peak_short['reveal_bar']
    assert peak_long['idx'] == peak_short['idx']


def test_zigzag_no_reversal_produces_zero_confirmed_pivots():
    """Threshold-sensitivity sanity check: a price path with NO reversal
    ever exceeding the ATR-scaled threshold (a strictly monotonic rise, no
    pullback at all) must confirm ZERO pivots -- the running candidate keeps
    extending forever and is correctly never confirmed."""
    from src.zigzag_swings import zigzag_swings

    path = np.linspace(100.0, 200.0, 50)   # strictly increasing, no reversal whatsoever
    pivots = zigzag_swings(path, path, path)
    assert pivots == []


def test_zigzag_atr_matches_features_py_atr14_formula():
    """The pre-registered threshold (1.5 * ATR(14)) must reuse this project's
    EXISTING ATR convention (src.features.compute_features's ATR_14 column:
    same true-range definition, same ewm(com=13, adjust=False) Wilder
    smoothing) -- not a second, subtly different ATR formula."""
    from src.zigzag_swings import _atr14

    rng = np.random.default_rng(0)
    n = 30
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))

    data = pd.DataFrame({'high': high, 'low': low, 'close': close})
    prev_close = data['close'].shift(1)
    true_range = pd.concat([
        data['high'] - data['low'],
        (data['high'] - prev_close).abs(),
        (data['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    expected = true_range.ewm(com=13, adjust=False).mean().to_numpy()

    np.testing.assert_allclose(_atr14(high, low, close), expected, equal_nan=True)


def test_zigzag_pivots_strictly_alternate_kind():
    """ZigZag pivots must strictly alternate H/L by construction (each
    confirmation immediately flips the search direction) -- unlike the
    independent per-bar fractal detector, no separate 'collapse a same-kind
    run' step should ever be necessary."""
    from src.zigzag_swings import zigzag_swings

    rng = np.random.default_rng(1)
    path = 100.0 + np.cumsum(rng.normal(0, 0.4, 400))
    pivots = zigzag_swings(path, path, path)
    assert len(pivots) >= 4, "need a handful of pivots for the alternation check to be meaningful"
    kinds = [p['kind'] for p in pivots]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "consecutive pivots must never share a kind"


def test_detect_harmonic_events_from_pivots_reuses_score_xabcd_unchanged():
    """The zigzag event path must produce byte-identical scoring to the
    fractal event path for the SAME geometric swing -- only confirmed_at_idx
    differs (D's own reveal_bar, not D_idx + fixed CONFIRMATION_LAG)."""
    from src.harmonic_patterns import detect_harmonic_events_from_pivots

    pivots = [
        {'idx': 5, 'kind': 'L', 'level': 0.0, 'reveal_bar': 9},      # variable lag, NOT idx+2
        {'idx': 10, 'kind': 'H', 'level': 100.0, 'reveal_bar': 11},
        {'idx': 15, 'kind': 'L', 'level': 38.2, 'reveal_bar': 20},
        {'idx': 20, 'kind': 'H', 'level': 76.39, 'reveal_bar': 21},
        {'idx': 25, 'kind': 'L', 'level': 21.4, 'reveal_bar': 33},   # e.g. an 8-bar reveal lag
    ]
    events = detect_harmonic_events_from_pivots(pivots)
    assert len(events) == 1
    ev = events.iloc[0]
    assert ev['pattern'] == 'gartley'
    assert ev['best_fit_score'] == pytest.approx(1.0)
    assert ev['direction'] == 1
    assert ev['confirmed_at_idx'] == 33, "confirmed_at_idx must be D's OWN reveal_bar, not D_idx+2"
    assert ev['confirmed_at_idx'] != ev['D_idx'] + 2, \
        "must NOT silently reuse the fractal path's fixed-lag convention"


def test_harmonic_event_check_swing_source_selects_zigzag_pivots(monkeypatch):
    """build_event_dataset(swing_source='zigzag') must route through
    zigzag_swings, NOT the fractal detector -- verified by monkeypatching
    each to a call-counting sentinel and confirming only the requested one
    fires."""
    import src.harmonic_event_check as hec
    from src.harmonic_patterns import HARMONIC_EVENT_COLUMNS

    calls = {'fractal': 0, 'zigzag': 0}

    def fake_fractal(*a, **k):
        calls['fractal'] += 1
        return pd.DataFrame(columns=HARMONIC_EVENT_COLUMNS)

    def fake_zigzag(*a, **k):
        calls['zigzag'] += 1
        return []   # empty pivot list -> detect_harmonic_events_from_pivots short-circuits

    monkeypatch.setattr(hec, 'detect_harmonic_events', fake_fractal)
    monkeypatch.setattr(hec, 'zigzag_swings', fake_zigzag)

    idx = pd.date_range('2020-01-01', periods=30, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    close = 1.1 + np.cumsum(rng.normal(0, 0.0005, 30))
    h1 = pd.DataFrame({'open': close, 'high': close + 0.0003, 'low': close - 0.0003,
                       'close': close, 'tick_volume': 1.0}, index=idx)

    hec.build_event_dataset(h1=h1, swing_source='zigzag')
    assert calls == {'fractal': 0, 'zigzag': 1}, \
        "swing_source='zigzag' must call zigzag_swings and NOT the fractal detector"

    calls['fractal'] = calls['zigzag'] = 0
    hec.build_event_dataset(h1=h1, swing_source='fractal')
    assert calls == {'fractal': 1, 'zigzag': 0}, \
        "swing_source='fractal' must call the fractal detector and NOT zigzag_swings"


# ── fractal-breakout drift/continuation event-study (own family) ────────────

def test_signed_continuation_direction_sign_construction():
    """A breakout_up day with a positive forward return must give POSITIVE
    signed_continuation (momentum confirmed); a breakout_down day with the SAME
    positive forward return must give NEGATIVE signed_continuation (momentum
    against the breakout direction), since direction flips the sign."""
    from src.fractal_breakout_driftcheck import compute_signed_continuation

    # up-breakout at idx=0, close rises by t+2 -> positive continuation.
    close_up = np.array([1.00, 1.00, 1.02, 1.00, 1.00])
    up = np.array([1, 0, 0, 0, 0])
    down = np.array([0, 0, 0, 0, 0])
    events_up, both_up = compute_signed_continuation(close_up, up, down, horizons=(2,))
    assert both_up == 0
    assert events_up.loc[0, 'signed_continuation_2'] > 0, \
        "up-breakout + positive forward return must be POSITIVE signed_continuation"

    # down-breakout at idx=0, close rises by t+2 (same forward move) -> negative continuation.
    close_down = np.array([1.00, 1.00, 1.02, 1.00, 1.00])
    up2 = np.array([0, 0, 0, 0, 0])
    down2 = np.array([1, 0, 0, 0, 0])
    events_down, both_down = compute_signed_continuation(close_down, up2, down2, horizons=(2,))
    assert both_down == 0
    assert events_down.loc[0, 'signed_continuation_2'] < 0, \
        "down-breakout + positive forward return must be NEGATIVE signed_continuation"


def test_signed_continuation_both_flags_excluded_from_direction():
    """A day where breakout_up AND breakout_down both fire has an undefined
    direction and must be excluded entirely (not defaulted to either sign),
    counted separately as both_excluded."""
    from src.fractal_breakout_driftcheck import compute_signed_continuation

    close = np.array([1.00, 1.00, 1.02, 1.00, 1.00])
    up = np.array([0, 1, 0, 0, 0])
    down = np.array([0, 1, 0, 0, 0])   # idx=1: both flags fire
    events, both = compute_signed_continuation(close, up, down, horizons=(2,))
    assert both == 1
    assert 1 not in events.index


def test_signed_continuation_insufficient_forward_history_excluded_not_padded():
    """An event too close to the end of the array (t+N would run off the end)
    must be excluded (NaN) for that horizon, never padded or estimated -- the
    default max_idx=len(close) is the plain 'insufficient history' case."""
    from src.fractal_breakout_driftcheck import compute_signed_continuation

    close = np.array([1.00, 1.00, 1.00, 1.00, 1.02])
    up = np.array([0, 0, 0, 1, 0])     # event at idx=3; idx+2=5 is OFF THE END (len=5)
    down = np.array([0, 0, 0, 0, 0])
    events, both = compute_signed_continuation(close, up, down, horizons=(2,))
    assert both == 0
    assert np.isnan(events.loc[3, 'signed_continuation_2']), \
        "insufficient forward history must be excluded (NaN), not padded"


def test_signed_continuation_validation_test_boundary_excludes_crossing_events():
    """A validation-slice event whose forward window would cross INTO the
    reserved test block must be excluded for that horizon even though the
    underlying array physically has more rows there (max_idx enforces the
    split boundary, not just the true end of the series) -- mirrors the
    look-ahead discipline of every other hypothesis family in this project."""
    from src.fractal_breakout_driftcheck import compute_signed_continuation

    # 10 bars total; pretend val_end=6 (bars 6..9 are the "reserved test block").
    close = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.02, 1.00, 1.00, 1.00])
    val_end = 6
    up = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])   # event at idx=4; idx+2=6 == val_end -> crosses boundary
    down = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    events_bounded, _ = compute_signed_continuation(close, up, down, horizons=(2,), max_idx=val_end)
    events_unbounded, _ = compute_signed_continuation(close, up, down, horizons=(2,), max_idx=None)

    assert np.isnan(events_bounded.loc[4, 'signed_continuation_2']), \
        "forward window crossing into the reserved test block must be excluded"
    assert not np.isnan(events_unbounded.loc[4, 'signed_continuation_2']), \
        "sanity check: the data DOES exist beyond val_end (this is a boundary rule, not missing data)"


def test_fractal_breakout_driftcheck_split_matches_ablation_convention():
    """The split boundaries must match src.ablation's canonical 70/80 formula
    exactly (same train_fraction=0.80/val_fraction=0.10 convention used by
    every other family in this project)."""
    from src.fractal_breakout_driftcheck import _canonical_split

    train_end, val_end = _canonical_split(1000, train_fraction=0.80, val_fraction=0.10)
    assert train_end == 700
    assert val_end == 800


# ── volatility-scaled sizing overlay (research-only retrospective backtest) ──

def test_trailing_ref_vol_is_causal_via_future_truncation():
    """trailing_ref_vol[t] must be a purely CAUSAL rolling/expanding median --
    appending more data AFTER day t must never change trailing_ref_vol AT or
    BEFORE t (the standard truncation-equivalence look-ahead guard used
    throughout this project)."""
    from src.vol_scaled_backtest import compute_trailing_ref_vol

    rng = np.random.default_rng(3)
    idx = pd.date_range('2020-01-01', periods=400, freq='D')
    vol_series = pd.Series(rng.uniform(0.1, 1.0, 400), index=idx)

    full = compute_trailing_ref_vol(vol_series)
    truncated = compute_trailing_ref_vol(vol_series.iloc[:150])

    pd.testing.assert_series_equal(full.iloc[:150], truncated, check_names=False)


def test_trailing_ref_vol_expanding_then_rolling_window():
    """Before 252 observations exist, trailing_ref_vol must be an EXPANDING
    median (using however many rows are available so far); once >= 252 exist,
    it must be a genuine ROLLING 252-period median (older-than-252 history
    dropped) -- both halves of the pre-registered formula."""
    from src.vol_scaled_backtest import compute_trailing_ref_vol

    idx = pd.date_range('2020-01-01', periods=300, freq='D')
    vol_series = pd.Series(np.arange(300, dtype=float), index=idx)  # strictly increasing
    trailing = compute_trailing_ref_vol(vol_series, window=252)

    # Expanding phase: trailing[t] = median(vol_series[0:t+1]) = t/2.
    assert trailing.iloc[10] == pytest.approx(np.median(np.arange(11)))
    # Rolling phase (t=299, window=252): must use ONLY the trailing 252 obs
    # (rows 48..299), not the full expanding history from row 0.
    expected_rolling = np.median(np.arange(299 - 252 + 1, 300))
    assert trailing.iloc[-1] == pytest.approx(expected_rolling)
    assert trailing.iloc[-1] != pytest.approx(np.median(np.arange(300))), \
        "must NOT still be an expanding median once the 252-window is full"


def test_compute_vol_weight_clips_at_both_bounds():
    """vol_weight = trailing_ref_vol / predicted_vol_pct must clip to
    [0.25, 4.0] at BOTH ends: a near-zero predicted vol (relative to the
    trailing reference) must saturate at the HIGH clip (4.0), and a
    predicted vol far above the trailing reference must saturate at the LOW
    clip (0.25) -- a mid-range ratio must pass through UNCLIPPED."""
    from src.vol_scaled_backtest import compute_vol_weight, VOL_WEIGHT_MIN, VOL_WEIGHT_MAX

    trailing = np.array([1.0, 1.0, 1.0])
    predicted = np.array([0.01, 100.0, 1.0])   # ratios: 100 (clip high), 0.01 (clip low), 1.0 (pass through)
    weight = compute_vol_weight(predicted, trailing)

    assert weight[0] == pytest.approx(VOL_WEIGHT_MAX), "near-zero predicted vol must clip to the HIGH bound"
    assert weight[1] == pytest.approx(VOL_WEIGHT_MIN), "predicted vol far above trailing must clip to the LOW bound"
    assert weight[2] == pytest.approx(1.0), "a mid-range ratio must pass through unclipped"


def test_compute_vol_weight_handles_zero_predicted_vol_defensively():
    """A defensive edge case never expected from the real ensemble output:
    predicted_vol_pct == 0 would otherwise divide by zero -- must saturate
    to the HIGH clip, never raise or propagate inf/NaN."""
    from src.vol_scaled_backtest import compute_vol_weight, VOL_WEIGHT_MAX

    weight = compute_vol_weight(np.array([0.0]), np.array([1.0]))
    assert weight[0] == pytest.approx(VOL_WEIGHT_MAX)
    assert np.isfinite(weight).all()


def test_moving_block_bootstrap_refuses_degenerate_short_sample():
    """When the settled-position sample is no longer than the pre-registered
    block length (this project's actual paper-trading ledgers are this thin
    right now), a 'block' as long as the whole series is just a rotation of
    every value once -- Sharpe (mean/std) is invariant to that rotation, so
    every resample would give an IDENTICAL delta and the 'CI' would be a
    single point. This must be refused (NaN), never silently reported as a
    razor-thin, falsely confident interval."""
    from src.vol_scaled_backtest import bootstrap_delta_sharpe

    rng = np.random.default_rng(1)
    original = rng.normal(0, 0.1, 10)
    weighted = original * 1.5
    lo, hi, degenerate = bootstrap_delta_sharpe(original, weighted, block_len=20, n_boot=200)
    assert lo != lo and hi != hi, "n <= block_len must refuse with NaN, not a degenerate point CI"


def test_compute_predicted_vol_series_never_fits(monkeypatch, tmp_path):
    """src.vol_scaled_backtest.compute_predicted_vol_series must perform PURE
    INFERENCE via the frozen volatility ensemble -- mirrors
    test_frozen_volatility_ensemble_batch_inference_never_fits (hypothesis
    #9's own reuse guard) but exercises THIS module's own call path
    (build_daily_ohlcv_from_h1 -> compute_features -> dropna ->
    batch_predict_frozen_ensemble_vol_pct), proving the sizing-overlay
    backtest re-uses the validated artifacts without ever re-fitting them."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from src.features import PRICE_FEATURE_COLUMNS, LAG_COLUMNS
    from src.vol_scaled_backtest import compute_predicted_vol_series

    n_lag = len(LAG_COLUMNS)
    rng = np.random.default_rng(0)
    lag_scaler = StandardScaler().fit(rng.normal(size=(30, n_lag)))
    lag_pca = PCA(n_components=3).fit(lag_scaler.transform(rng.normal(size=(30, n_lag))))
    n_cols_after_pca = len(PRICE_FEATURE_COLUMNS) - n_lag + lag_pca.n_components_
    global_scaler = StandardScaler().fit(rng.normal(size=(30, n_cols_after_pca)))

    class _FakeSeedModel:
        def predict(self, windows, verbose=0):
            n = windows.shape[0]
            zeros = np.zeros((n, 1), dtype='float32')
            vol = np.full((n, 1), 0.42, dtype='float32')
            return zeros, zeros, vol

    artifacts = {
        'lag_scaler': lag_scaler, 'lag_pca': lag_pca, 'global_scaler': global_scaler,
        'time_steps': 5, 'models': [_FakeSeedModel(), _FakeSeedModel()],
    }

    def _explode(self, *a, **k):
        raise AssertionError('fit() must never be called by the vol-scaled backtest')
    monkeypatch.setattr(StandardScaler, 'fit', _explode)
    monkeypatch.setattr(StandardScaler, 'fit_transform', _explode)
    monkeypatch.setattr(PCA, 'fit', _explode)
    monkeypatch.setattr(PCA, 'fit_transform', _explode)

    closes = 1.10 + np.cumsum(rng.normal(0, 0.002, 260))
    h1 = _synthetic_h1(closes)
    h1_csv = tmp_path / 'h1.csv'
    h1.to_csv(h1_csv)

    vol_series = compute_predicted_vol_series(h1_csv=str(h1_csv), artifacts=artifacts)
    assert len(vol_series) > 0
    assert vol_series.dropna().iloc[-1] == pytest.approx(0.42)


def test_paper_trading_module_left_untouched_by_vol_scaled_backtest():
    """HARD BOUNDARY check: src/paper_trading.py (build_ledger/summarize/
    build_all_ledgers and the live logging path) must be byte-for-byte
    unmodified by the new research-only sizing-overlay backtest -- verified
    directly against the git-tracked HEAD version of the file."""
    import os
    import subprocess

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    result = subprocess.run(
        ['git', 'diff', '--quiet', 'HEAD', '--', 'src/paper_trading.py'],
        cwd=repo_root,
    )
    assert result.returncode == 0, \
        "src/paper_trading.py must be untouched by the vol-scaled backtest work"


# ── M15 fetch chain (harmonic-pattern hypotheses H1.5/H1.6, STEP 0) ─────────

def test_fetch_m15_market_data_prefers_mt5(monkeypatch):
    """MT5 is the sole LIVE source for M15 -- when a terminal session is
    reachable it must be used, via the same bar-count copy_rates_from_pos
    API as the existing H1 fetch (just TIMEFRAME_M15 instead of H1)."""
    import sys
    import src.live_data as live_data

    rates = np.array(
        [(1781481600 + i * 900, 1.10 + i * 0.0001, 1.11 + i * 0.0001, 1.09 + i * 0.0001,
          1.105 + i * 0.0001, 100 + i, 5, 0) for i in range(5)],
        dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'), ('close', '<f8'),
               ('tick_volume', '<i8'), ('spread', '<i4'), ('real_volume', '<i8')]
    )

    class _FakeMT5:
        TIMEFRAME_M15 = 15

        @staticmethod
        def initialize():
            return True

        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start, count):
            assert timeframe == _FakeMT5.TIMEFRAME_M15, "must request TIMEFRAME_M15, not another timeframe"
            return rates

        @staticmethod
        def shutdown():
            pass

    monkeypatch.setitem(sys.modules, 'MetaTrader5', _FakeMT5)
    df, source = live_data.fetch_m15_market_data(bars=5, cache_path=None)
    assert source == "MT5"
    assert len(df) == 5
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'tick_volume']


def test_fetch_m15_market_data_never_falls_back_to_yfinance(monkeypatch):
    """UNLIKE fetch_live_market_data/fetch_h1_market_data, M15 must NEVER
    fall back to Yahoo Finance -- if MT5 is unreachable and no cache exists,
    the result must be (None, None), not a yfinance call."""
    import src.live_data as live_data

    monkeypatch.setattr(live_data, '_fetch_m15_from_mt5', lambda symbol, bars: None)

    def _boom_yfinance(*args, **kwargs):
        raise AssertionError("yfinance must never be used for the M15 fetch chain.")
    monkeypatch.setattr(live_data.yf, "Ticker", _boom_yfinance)

    df, source = live_data.fetch_m15_market_data(bars=5, cache_path=None)
    assert df is None and source is None


def test_fetch_m15_market_data_falls_back_to_cache_when_mt5_unreachable(monkeypatch, tmp_path):
    """When MT5 is unreachable, the on-disk cache of previously-fetched M15
    bars is an acceptable OFFLINE fallback (same 'never hard-fail'
    convention as every other fetch chain), even though no other LIVE
    source is ever tried."""
    import src.live_data as live_data

    cache_path = str(tmp_path / 'm15_cache.csv')
    cached = pd.DataFrame({
        'open': [1.10], 'high': [1.11], 'low': [1.09], 'close': [1.105], 'tick_volume': [100.0],
    }, index=pd.date_range('2026-06-19', periods=1, freq='15min', tz='UTC'))
    cached.to_csv(cache_path)

    monkeypatch.setattr(live_data, '_fetch_m15_from_mt5', lambda symbol, bars: None)
    df, source = live_data.fetch_m15_market_data(bars=5, cache_path=cache_path)
    assert source == "cache"
    assert len(df) == 1


# ── M15 harmonic-pattern hypotheses (H1.5/H1.6, SAME family, block bootstrap) ──

def test_m15_ewma_span_and_horizon_are_the_documented_rederivation():
    """The M15 constants must be the PRE-REGISTERED re-derivation (24h*4,
    120 H1 bars*4) -- NOT H1's literal numbers copy-pasted -- preserving the
    same real-world '~1 day' / '~5 trading days' meaning at M15 granularity."""
    from src.harmonic_m15_check import M15_EWMA_SPAN, M15_HORIZON_BARS

    assert M15_EWMA_SPAN == 24 * 4, "EWMA span must preserve the '~1 day of memory' meaning"
    assert M15_HORIZON_BARS == 120 * 4, "time horizon must preserve the '~5 trading days' meaning"


def test_event_gap_diagnostics_median_and_iqr():
    """Median/IQR of the gap between consecutive (already-sorted) event entry
    indices must be a plain, direct percentile computation on the diffs --
    the honest clustering diagnostic the block-bootstrap decision leans on."""
    from src.harmonic_m15_check import event_gap_diagnostics

    entry_idx = np.array([100, 40, 0, 25, 10])   # unsorted on purpose
    gaps = event_gap_diagnostics(entry_idx)
    # sorted: [0, 10, 25, 40, 100] -> diffs [10, 15, 15, 60]
    expected_q1, expected_med, expected_q3 = np.percentile([10, 15, 15, 60], [25, 50, 75])
    assert gaps['median_gap_bars'] == pytest.approx(expected_med)
    assert gaps['iqr_low_bars'] == pytest.approx(expected_q1)
    assert gaps['iqr_high_bars'] == pytest.approx(expected_q3)


def test_event_gap_diagnostics_undefined_with_fewer_than_two_events():
    from src.harmonic_m15_check import event_gap_diagnostics

    gaps = event_gap_diagnostics(np.array([5]))
    assert gaps['median_gap_bars'] != gaps['median_gap_bars']   # NaN


def test_circular_block_bootstrap_indices_are_contiguous_within_a_block():
    """Each block of `block_len` resampled indices must be CONSECUTIVE
    original positions (wrapping modulo n) -- this is what makes it a
    MOVING-BLOCK bootstrap rather than an i.i.d. one; within a block,
    successive resampled indices must differ by exactly 1 (mod n)."""
    from src.harmonic_m15_check import _circular_block_bootstrap_indices

    n, block_len = 50, 10
    rng = np.random.default_rng(0)
    idx = _circular_block_bootstrap_indices(n, block_len, rng)
    assert len(idx) == n
    assert idx.min() >= 0 and idx.max() < n
    # Check contiguity within each block boundary (blocks start at multiples
    # of block_len in the OUTPUT array, per the function's own construction).
    for block_start in range(0, n, block_len):
        block = idx[block_start:block_start + block_len]
        for k in range(1, len(block)):
            assert block[k] == (block[0] + k) % n, \
                "resampled indices within one block must be consecutive original positions"


def test_bootstrap_delta_and_mcnemar_block_runs_and_detects_a_clear_edge():
    """Sanity check on the block-bootstrap primitive itself: with a
    challenger that is CLEARLY and consistently more accurate than the
    reference over the whole validation set, the block bootstrap must still
    detect it (CI entirely > 0), even though it resamples in blocks rather
    than i.i.d. rows."""
    from src.harmonic_m15_check import bootstrap_delta_and_mcnemar_block

    rng = np.random.default_rng(1)
    n = 200
    y_val = rng.integers(0, 2, n)
    pred_reference = rng.integers(0, 2, n)          # ~50% accuracy, uncorrelated with y
    pred_challenger = y_val.copy()                   # 100% accuracy
    res = bootstrap_delta_and_mcnemar_block(y_val, pred_reference, pred_challenger,
                                            alpha=0.05, block_len=20, n_boot=500, random_state=0)
    assert res['acc_challenger'] == pytest.approx(1.0)
    assert res['ci_low'] > 0
    assert res['cleared'] is True


def test_cost_drag_diagnostic_computes_atr_pip_ratio(tmp_path, monkeypatch):
    """cost_drag_diagnostic must convert mean ATR(14) to PIPS via the
    project's own PIP_SIZE and report the fixed round-trip cost as a
    fraction of that -- the empirical grounding for the 'proportionally
    larger drag on M15' claim, using a KNOWN constant true range so the
    expected ATR is exactly computable by hand."""
    from src.harmonic_m15_check import cost_drag_diagnostic
    from src.paper_trading import PIP_SIZE

    n = 60
    # Constant true range of 0.0010 (10 pips): high-low=0.0010 every bar,
    # close held flat so (high-prev_close)/(low-prev_close) never exceed it.
    idx_m15 = pd.date_range('2026-01-01', periods=n, freq='15min', tz='UTC')
    m15 = pd.DataFrame({'open': 1.10, 'high': 1.1005, 'low': 1.0995, 'close': 1.10,
                        'tick_volume': 1.0}, index=idx_m15)

    idx_h1 = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    h1 = pd.DataFrame({'open': 1.10, 'high': 1.1010, 'low': 1.0990, 'close': 1.10,
                       'tick_volume': 1.0}, index=idx_h1)
    h1_dir = tmp_path / 'results'
    h1_dir.mkdir()
    h1.to_csv(h1_dir / 'eurusd_h1.csv')

    res = cost_drag_diagnostic(m15, base_dir=str(tmp_path))
    assert res['mean_atr14_m15_pips'] == pytest.approx(0.0010 / PIP_SIZE, rel=0.05)
    assert res['mean_atr14_h1_pips'] == pytest.approx(0.0020 / PIP_SIZE, rel=0.05)
    assert res['cost_frac_of_atr_m15'] > res['cost_frac_of_atr_h1'], \
        "the same fixed pip cost must be a LARGER fraction of the smaller M15 typical range"


# ── walk-forward validation (research-only robustness report) ──────────────

def test_compute_windows_rolling_schedule_matches_spec():
    """3yr trailing train / 1yr step+test, sliding forward across the FULL
    available history -- pure calendar arithmetic, no row-count dependence."""
    from src.walk_forward_validation import compute_windows

    idx = pd.date_range('1999-01-04', '2026-06-17', freq='D')
    windows = compute_windows(idx, train_years=3, test_years=1)
    assert len(windows) == 24, "must match the real euro-era span's actual window count"
    assert windows[0]['train_start'] == pd.Timestamp('1999-01-04')
    assert windows[0]['train_end'] == pd.Timestamp('2002-01-04')
    assert windows[0]['test_end'] == pd.Timestamp('2003-01-04')
    # Each subsequent window steps forward by exactly test_years (1 year).
    assert windows[1]['train_start'] == windows[0]['train_start'] + pd.DateOffset(years=1)
    # No window's test_end may exceed the available history.
    assert all(w['test_end'] <= idx.max() for w in windows)


def test_slice_window_strict_causality():
    """A window's training rows must NEVER include any date at or after that
    SAME window's own test period start -- strict chronological causality at
    every single window, no exceptions."""
    from src.walk_forward_validation import compute_windows, _slice_window

    idx = pd.date_range('1999-01-04', '2010-01-04', freq='D')
    feat = pd.DataFrame({'close': np.arange(len(idx), dtype=float)}, index=idx)
    windows = compute_windows(idx, train_years=3, test_years=1)

    for window in windows:
        window_df, train_end_local = _slice_window(feat, window)
        train_rows = window_df.iloc[:train_end_local]
        test_rows = window_df.iloc[train_end_local:]
        assert (train_rows.index < window['train_end']).all(), \
            "training rows must never reach the test period's own start date"
        assert (test_rows.index >= window['train_end']).all()
        assert (test_rows.index < window['test_end']).all()


def test_windows_train_never_crosses_into_a_later_window_test_period():
    """Scoped-reuse check: walking across the full history (including years
    that are a LATER window's own test period) must never let an EARLIER
    window's training data reach into that later test period -- still no
    future leakage anywhere, just a different overall sweep. Concretely: for
    every pair of windows, if window A's train block overlaps calendar time
    with window B's test block, that overlap must never include window B's
    test dates themselves being used as window A's training input for a
    row dated at/after A's OWN test start (i.e. each window is causal in
    isolation, which composes safely across the whole walk)."""
    from src.walk_forward_validation import compute_windows, _slice_window

    idx = pd.date_range('1999-01-04', '2026-06-17', freq='D')
    feat = pd.DataFrame({'close': np.arange(len(idx), dtype=float)}, index=idx)
    windows = compute_windows(idx, train_years=3, test_years=1)

    for k, window in enumerate(windows):
        window_df, train_end_local = _slice_window(feat, window)
        train_rows = window_df.iloc[:train_end_local]
        # This window's own training data must never include ANY date from
        # its own (or by construction, any later window's) test period.
        assert (train_rows.index < window['train_end']).all()
        # And a LATER window's test period starts even further in the future,
        # so this window's train data (bounded above by its OWN train_end)
        # can never reach into it either.
        if k + 1 < len(windows):
            later = windows[k + 1]
            assert window['train_end'] <= later['train_end']
            assert (train_rows.index < later['train_end']).all()


def test_walk_forward_never_writes_to_models_or_production_files(tmp_path):
    """HARD BOUNDARY check: running the walk-forward report must NEVER modify
    any file under models/, nor _train_pipeline.py, src/inference.py, or
    config.json -- verified dynamically by hashing every such file before and
    after a real (small) run, rather than relying on git history (which may
    already carry unrelated pre-existing local modifications)."""
    import hashlib
    import os
    from src.walk_forward_validation import run

    protected_paths = ['_train_pipeline.py', 'src/inference.py', 'config.json']
    for root, _dirs, files in os.walk('models'):
        for fname in files:
            protected_paths.append(os.path.join(root, fname))

    def _hash(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    before = {p: _hash(p) for p in protected_paths}
    run(register=False, max_windows=1)
    after = {p: _hash(p) for p in protected_paths}

    changed = [p for p in protected_paths if before[p] != after[p]]
    assert changed == [], f"walk-forward run must never modify: {changed}"