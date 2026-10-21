"""Gold features (task 3.1; spec "Gold features are scale-free and price-only")."""
import numpy as np
import pandas as pd
import pytest

from src.gold import features as F


def _ohlc(n, freq):
    rng = np.random.default_rng(7)
    idx = pd.date_range('2020-01-01', periods=n, freq=freq, tz='UTC')
    c = 1500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.002, n))
    return pd.DataFrame({'open': o, 'high': np.maximum(o, c) * 1.004,
                         'low': np.minimum(o, c) * 0.996, 'close': c}, index=idx)


@pytest.mark.parametrize('cadence,freq', [('D1', 'D'), ('H1', 'h')])
def test_features_are_invariant_to_the_price_level(cadence, freq):
    df = _ohlc(600, freq)
    a = F.build(df, cadence)
    b = F.build(df * 3.0, cadence)
    pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('col', ['close', 'SMA_50', 'open', 'ATR_14', 'tick_volume', 'spread_rel'])
def test_level_volume_and_spread_columns_are_refused(col):
    with pytest.raises(F.ScaleDependentFeature):
        F.assert_scale_free(['log_return', col])


def test_no_column_names_volume_or_spread():
    for cols in (F.D1_FEATURE_COLUMNS, F.h1_feature_columns()):
        assert not [c for c in cols if 'volume' in c.lower() or 'spread' in c.lower()]
        F.assert_scale_free(cols)


@pytest.mark.parametrize('cadence,freq', [('D1', 'D'), ('H1', 'h')])
def test_features_at_t_ignore_later_bars(cadence, freq):
    df = _ohlc(600, freq)
    t = 450
    altered = df.copy()
    altered.iloc[t + 1:] *= 1.37
    a, b = F.build(df, cadence), F.build(altered, cadence)
    pd.testing.assert_frame_equal(a.iloc[:t + 1], b.iloc[:t + 1])
