"""fx-rates-2y, group 3: the twelve features."""
import numpy as np
import pandas as pd
import pytest

from src.fx_rates import features as FE
from src.fx_rates import load_config
from src.fx_rates.align import align_yields

CFG = load_config()
DAYS = pd.bdate_range('2012-01-02', '2014-12-31', name='date')


def synth_d1(seed=0, days=DAYS):
    rng = np.random.default_rng(seed)
    c = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.005, len(days))))
    return pd.DataFrame({'mid_close': c, 'close': c}, index=days)


def synth_yields(seed=0, holidays=('2013-07-04', '2013-12-25', '2014-01-20')):
    rng = np.random.default_rng(seed + 100)
    days = pd.bdate_range('2011-06-01', '2014-12-31')
    out = {}
    for k, name in enumerate(('us2y', 'us10y', 'de2y', 'uk2y')):
        s = pd.Series(1 + k + np.cumsum(rng.normal(0, 0.02, len(days))), index=days)
        out[name] = s.drop(pd.DatetimeIndex(holidays)) if name == 'us2y' else s
    return out


@pytest.fixture(scope='module')
def built():
    d1 = {'EURUSD': synth_d1(1), 'GBPUSD': synth_d1(2)}
    y = synth_yields()
    return d1, y, FE.build_features(d1, y, CFG)


def test_columns_and_warm_up(built):
    d1, y, feats = built
    f = feats['EURUSD']
    assert list(f.columns) == CFG['features']['columns'] + ['reason']
    assert (f.iloc[:250]['reason'] != '').all()                     # divergence needs 252 + 20 days
    ok = f[f['reason'] == '']
    assert len(ok) > 400 and not ok[CFG['features']['columns']].isna().any().any()


def test_hand_computed_values(built):
    d1, y, feats = built
    f = feats['EURUSD']
    t = f[f['reason'] == ''].index[50]
    i = d1['EURUSD'].index.get_loc(t)
    days = d1['EURUSD'].index
    lagged = lambda name, j: y[name][y[name].index <= days[j - 1]].iloc[-1]
    spread = lambda j: lagged('de2y', j) - lagged('us2y', j)
    assert f.loc[t, 'spread_2y_level'] == pytest.approx(spread(i))
    assert f.loc[t, 'spread_2y_chg_5d'] == pytest.approx(spread(i) - spread(i - 5))
    win = np.array([spread(j) for j in range(i - 59, i + 1)])
    assert f.loc[t, 'spread_2y_zscore_60d'] == pytest.approx((win[-1] - win.mean()) / win.std(ddof=1))
    assert f.loc[t, 'us_2s10s_slope'] == pytest.approx(lagged('us10y', i) - lagged('us2y', i))
    c = d1['EURUSD']['mid_close'].to_numpy()
    r = np.diff(np.log(c))
    vol20 = r[i - 20:i].std(ddof=1)
    assert f.loc[t, 'fx_ret_5d'] == pytest.approx(np.log(c[i] / c[i - 5]) / (vol20 * np.sqrt(5)))
    assert f.loc[t, 'fx_vol_20d_ratio'] == pytest.approx(vol20 / r[i - 60:i].std(ddof=1))
    fx20 = np.log(c[20:] / c[:-20])                                   # fx20[k] ends at day k + 20
    k = i - 20
    zf = (fx20[k] - fx20[k - 252:k].mean()) / fx20[k - 252:k].std(ddof=1)
    sc = np.array([spread(j) - spread(j - 20) for j in range(i - 252, i + 1)])
    zs = (sc[-1] - sc[:-1].mean()) / sc[:-1].std(ddof=1)
    assert f.loc[t, 'fx_vs_spread_div_20d'] == pytest.approx(zf - zs)


def test_us_holiday_sets_the_stale_flag(built):
    d1, y, feats = built
    t = pd.Timestamp('2013-07-05')                                    # lagged date 07-04, a US holiday
    assert feats['EURUSD'].loc[t, 'yield_stale_flag'] == 1.0
    assert feats['EURUSD'].loc[pd.Timestamp('2013-07-10'), 'yield_stale_flag'] == 0.0


def test_scaling_fx_prices_leaves_every_feature_unchanged(built):
    d1, y, feats = built
    doubled = {p: d * 2 for p, d in d1.items()}
    again = FE.build_features(doubled, y, CFG)
    cols = CFG['features']['columns']
    pd.testing.assert_frame_equal(again['GBPUSD'][cols], feats['GBPUSD'][cols], rtol=1e-9)


def test_missing_or_extra_column_fails_and_names_it():
    df = pd.DataFrame(columns=CFG['features']['columns'])
    with pytest.raises(FE.FeatureColumnError, match='us_2s10s_slope'):
        FE.assemble(df.drop(columns=['us_2s10s_slope']), CFG)
    with pytest.raises(FE.FeatureColumnError, match='extra'):
        FE.assemble(df.assign(x=1), CFG)


def test_truncating_inputs_after_t_leaves_the_row_unchanged(built):
    d1, y, feats = built
    cols = CFG['features']['columns']
    days = feats['EURUSD'][feats['EURUSD']['reason'] == ''].index
    for t in np.random.default_rng(3).choice(days, size=20, replace=False):
        t = pd.Timestamp(t)
        short_d1 = {p: d[d.index <= t] for p, d in d1.items()}
        short_y = {k: s[s.index < t] for k, s in y.items()}
        again = FE.build_features(short_d1, short_y, CFG)
        for p in CFG['pairs']:
            pd.testing.assert_series_equal(again[p].loc[t, cols], feats[p].loc[t, cols], check_names=False)
