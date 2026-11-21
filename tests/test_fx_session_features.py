"""fx-session-ny, group 3: pre-session features."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_fib.mtf_aligner import atr, build_d1
from src.fx_session import features as FE
from src.fx_session import load_config
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()


def make_frames(n_weeks=12, first=date(2018, 1, 7), same=False):
    out = {}
    for i, p in enumerate(CFG['factor_pairs']):
        f = synthetic_weeks(first, n_weeks, seed=0 if same else i + 1)
        f['spread'] = 5.0
        out[p] = f
    return out


@pytest.fixture(scope='module')
def built():
    fr = make_frames()
    return fr, FE.build_features(fr, CFG)


def _complete_dates(feats):
    ok = None
    for f in feats.values():
        m = f['reason'] == ''
        ok = m if ok is None else ok & m
    return ok[ok].index


def test_rows_have_exactly_the_config_columns_and_warm_up_is_excluded(built):
    fr, feats = built
    f = feats['EURUSD']
    assert list(f.columns) == CFG['features']['columns'] + ['reason']
    assert f.iloc[:10]['reason'].ne('').all()                 # no ATR / range history yet
    assert len(_complete_dates(feats)) > 20
    assert not f.loc[f['reason'] == '', CFG['features']['columns']].isna().any().any()


def test_hand_computed_moves_and_flags(built):
    fr, feats = built
    d = _complete_dates(feats)[5]
    h = fr['EURUSD']
    d1 = build_d1(h)
    d1 = d1[d1['n_h1'] >= 12].copy()
    d1['atr'] = atr(d1, 14)
    prev = d1[d1.index < d].iloc[-1]
    bar = lambda hr, col: h.loc[d + pd.Timedelta(hours=hr), col]
    row = feats['EURUSD'].loc[d]
    assert row['asia_move'] == pytest.approx((bar(8, 'close') - bar(0, 'open')) / prev['atr'])
    assert row['london_move'] == pytest.approx((bar(13, 'close') - bar(9, 'open')) / prev['atr'])
    assert row['data_hour_move'] == pytest.approx((bar(14, 'close') - bar(14, 'open')) / prev['atr'])
    assert row['move_since_prev_close'] == pytest.approx((bar(14, 'close') - prev['close']) / prev['atr'])
    day_high = max(bar(k, 'high') for k in range(15))
    assert row['broke_prev_high'] == float(day_high > prev['high'])
    lo = min(bar(k, 'low') for k in range(9, 14))
    hi = max(bar(k, 'high') for k in range(9, 14))
    assert row['london_position'] == pytest.approx((bar(14, 'close') - lo) / (hi - lo))
    assert row['atr_d1_frac'] == pytest.approx(prev['atr'] / bar(14, 'close'))


def test_scaling_every_price_leaves_every_feature_unchanged(built):
    fr, feats = built
    doubled = {p: f.assign(**{c: f[c] * 2 for c in ('open', 'high', 'low', 'close')}) for p, f in fr.items()}
    again = FE.build_features(doubled, CFG)
    cols = CFG['features']['columns']
    for p in CFG['pairs']:
        pd.testing.assert_frame_equal(again[p][cols], feats[p][cols], rtol=1e-9)
        assert (again[p]['reason'] == feats[p]['reason']).all()


def test_common_dollar_move_has_zero_residual():
    fr = make_frames(same=True)
    feats = FE.build_features(fr, CFG)
    f = feats['EURUSD'][feats['EURUSD']['reason'] == '']
    assert np.allclose(f['pair_residual'], 0.0)
    d = f.index[3]
    h = fr['EURUSD']
    d1 = build_d1(h)
    d1['atr'] = atr(d1, 14)
    a = d1[d1.index < d]['atr'].iloc[-1]
    own = (h.loc[d + pd.Timedelta(hours=14), 'close'] - h.loc[d, 'open']) / a
    assert f.loc[d, 'usd_factor'] == pytest.approx(own)


def test_dollar_factor_is_the_mean_of_the_three_pairs(built):
    fr, feats = built
    e, g = feats['EURUSD'], feats['GBPUSD']
    d = _complete_dates(feats)
    assert np.allclose(e.loc[d, 'usd_factor'], g.loc[d, 'usd_factor'])
    assert not np.allclose(e.loc[d, 'pair_residual'], 0.0)


def test_calendar_flags():
    dates = pd.DatetimeIndex(['2026-03-10', '2026-06-15', '2026-11-06', '2026-11-13', '2024-01-31'], tz='UTC')
    cal = FE.calendar_block(dates, CFG)
    assert list(cal['dst_mismatch']) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert list(cal['is_nfp_day']) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert cal['is_fomc_day'].iloc[-1] == 1.0 and cal['is_fomc_day'].iloc[1] == 0.0
    assert list(cal['day_of_week']) == [1.0, 0.0, 4.0, 4.0, 2.0]
    assert FE.dst_mismatch(pd.DatetimeIndex(['2026-10-27'], tz='UTC'))[0] == 1.0   # EU back, US not yet


def test_unscheduled_fomc_action_is_not_flagged():
    cal = FE.calendar_block(pd.DatetimeIndex(['2020-03-03', '2020-03-15'], tz='UTC'), CFG)
    assert (cal['is_fomc_day'] == 0).all()


def test_extra_or_missing_column_fails_and_names_it():
    df = pd.DataFrame(columns=CFG['features']['columns'] + ['sneaky'])
    with pytest.raises(FE.FeatureColumnError, match='sneaky'):
        FE.assemble(df, CFG)
    with pytest.raises(FE.FeatureColumnError, match='dst_mismatch'):
        FE.assemble(df.drop(columns=['dst_mismatch', 'sneaky']), CFG)


def test_a_missing_pre_session_bar_excludes_the_day_with_its_reason(built):
    fr, feats = built
    d = _complete_dates(feats)[4]
    holed = {p: f.copy() for p, f in fr.items()}
    holed['AUDUSD'] = holed['AUDUSD'].drop(d + pd.Timedelta(hours=10))
    again = FE.build_features(holed, CFG)
    assert again['EURUSD'].loc[d, 'reason'] == 'missing factor-pair input'
    holed['EURUSD'] = holed['EURUSD'].drop(d + pd.Timedelta(hours=3))
    assert FE.build_features(holed, CFG)['EURUSD'].loc[d, 'reason'] == 'missing pre-session bar'


def test_one_bar_per_day_d1_counts_as_missing():
    fr = make_frames()
    g = fr['GBPUSD']
    d = pd.Timestamp('2018-02-07', tz='UTC')
    fr['GBPUSD'] = g[(g.index.normalize() != d) | (g.index.hour == 0)]
    p = FE.prev_d1(fr['GBPUSD'], pd.DatetimeIndex([d + pd.Timedelta(days=1)]), CFG)
    assert p.iloc[0]['d1_date'] == d - pd.Timedelta(days=1)     # the thin day is skipped


def test_truncating_after_the_entry_bar_leaves_the_row_unchanged(built):
    fr, feats = built
    rng = np.random.default_rng(7)
    days = _complete_dates(feats)
    cols = CFG['features']['columns']
    for d in rng.choice(days, size=min(20, len(days)), replace=False):
        d = pd.Timestamp(d)
        cut = d + pd.Timedelta(hours=CFG['session']['entry_hour'])
        short = {p: f[f.index <= cut] for p, f in fr.items()}
        again = FE.build_features(short, CFG)
        for p in CFG['pairs']:
            pd.testing.assert_series_equal(again[p].loc[d, cols], feats[p].loc[d, cols], check_names=False)
