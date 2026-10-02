"""M15 session challengers and their features (task 14.6).

Synthetic M15 bars throughout, so these run without the ungitted M1 parquet.
"""
import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import m15_data as MD
from src.forecast_eval import record as R
from src.forecast_eval import targets as T
from src.forecast_eval.challengers import make


def _m15(days=40, seed=0, spread_points=5.0):
    """A contiguous weekday M15 grid with a random walk and a clean spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp('2026-01-05', tz='UTC'), periods=96 * days, freq='15min')
    idx = idx[idx.dayofweek < 5]
    close = 1.10 + np.cumsum(rng.normal(0, 2e-5, len(idx)))
    spread = spread_points * 1e-5
    return pd.DataFrame({'open': close, 'high': close + 3e-5, 'low': close - 3e-5,
                         'close': close, 'mid': close + spread / 2,
                         'tick_volume': rng.integers(20, 200, len(idx)).astype(float),
                         'spread_price': spread, 'point': 1e-5}, index=idx)


@pytest.fixture(scope='module')
def m15():
    return _m15()


def _cfg(name):
    return R.new_record('m15t')['challengers'][name]


# ── features ───────────────────────────────────────────────────────────────

def test_features_carry_the_base_builder_and_the_session_columns(m15):
    from src.h1_features import DIRECTION_FEATURE_COLUMNS
    f = F.m15_bar_features(m15)
    assert set(DIRECTION_FEATURE_COLUMNS) <= set(f.columns), 'the shared H1 builder'
    assert set(F.M15_EXTRA_COLUMNS) <= set(f.columns)
    assert 'close' in f.columns and not f.isna().to_numpy().any()


def test_feature_rows_are_a_contiguous_suffix_of_the_bars(m15):
    """An interior drop would renumber positions and break the purge."""
    f = F.m15_bar_features(m15)
    assert f.index.equals(m15.index[len(m15) - len(f):])


def test_mid_and_bid_definitions_differ_by_half_the_spread(m15):
    mid = F.m15_bar_features(m15, price='mid')
    bid = F.m15_bar_features(m15, price='bid')
    assert np.allclose(mid['close'] - bid['close'], 0.5 * 5e-5)
    assert mid.index.equals(bid.index)
    with pytest.raises(KeyError, match='price definition'):
        F.m15_bar_features(m15, price='ask')


def test_features_are_trailing_only(m15):
    """Replacing every bar AFTER t must not change any feature at t."""
    f_full = F.m15_bar_features(m15)
    cut = len(m15) - 40
    tampered = m15.copy()
    tampered.iloc[cut:, :] = tampered.iloc[cut:, :] * 1.02
    f_tam = F.m15_bar_features(tampered)
    common = f_full.index[:np.searchsorted(f_full.index, m15.index[cut])]
    pd.testing.assert_frame_equal(f_full.loc[common], f_tam.loc[common])


def test_session_position_is_a_feature_and_marks_out_of_session_rows(m15):
    f = F.m15_bar_features(m15)
    inside = MD.session_mask(f.index)
    assert set(np.unique(f['in_session'])) == {0.0, 1.0}
    assert (f.loc[inside, 'session_pos'] >= 0).all()
    assert (f.loc[~inside, 'session_pos'] == -1.0).all()
    assert np.isfinite(f['session_pos']).all(), 'must stay finite or the row is dropped'


def test_features_refuse_a_frame_without_a_spread(m15):
    with pytest.raises(KeyError, match='spread_price'):
        F.m15_bar_features(m15.drop(columns=['spread_price']))


def test_features_refuse_a_frame_shorter_than_the_warm_up():
    with pytest.raises(ValueError, match='warm-up'):
        F.m15_bar_features(_m15(days=1))


# ── the two challengers ────────────────────────────────────────────────────

def _fit_one(name, m15, h, price='mid'):
    m = make(name, _cfg(name), price=price)
    inputs = m.build_inputs(m15)
    d, _ = T.direction_target(pd.Series(inputs.close), h)
    tg = {'dir': d.to_numpy()}
    elig = MD.eligible_positions(inputs.index, h)
    train = elig[elig < int(len(inputs) * 0.7)]
    return m.fit(inputs, train, tg, 42, h), inputs, tg, elig


@pytest.mark.parametrize('name', ['m15_session_gbm', 'm15_session_lstm'])
@pytest.mark.parametrize('h', [1, 4])
def test_fit_and_predict_on_the_session_grid(name, m15, h):
    m, inputs, _tg, elig = _fit_one(name, m15, h)
    p = m.predict(inputs)['p_up']
    assert len(p) == len(inputs)
    scored = p[elig]
    assert np.isfinite(scored).all() and ((scored >= 0) & (scored <= 1)).all()


@pytest.mark.parametrize('name', ['m15_session_gbm', 'm15_session_lstm'])
def test_configuration_hash_is_identical_across_horizons(name, m15):
    hashes = {make(name, _cfg(name)).config_hash for _ in (1, 2, 4)}
    assert len(hashes) == 1
    assert make(name, _cfg(name), price='mid').config_hash == \
        make(name, _cfg(name), price='bid').config_hash, \
        'the price definition must not change the declared configuration'


@pytest.mark.parametrize('name', ['m15_session_gbm', 'm15_session_lstm'])
def test_save_and_load_round_trips(name, m15, tmp_path):
    m, inputs, _tg, elig = _fit_one(name, m15, 1)
    before = m.predict(inputs)['p_up'][elig]
    files = m.save(str(tmp_path / name))
    assert any(f.endswith('challenger.json') for f in files)
    again = make(name, _cfg(name)).load(str(tmp_path / name))
    assert np.allclose(again.predict(inputs)['p_up'][elig], before, equal_nan=True)


def test_the_lstm_only_predicts_rows_with_a_full_window(m15):
    m, inputs, _tg, _elig = _fit_one('m15_session_lstm', m15, 1)
    steps = m.config['lstm']['time_steps']
    p = m.predict(inputs)['p_up']
    assert np.isnan(p[:steps - 1]).all()
    assert np.isfinite(p[steps - 1:]).all()


def test_the_lstm_windows_are_a_view_not_a_dense_tensor(m15):
    """A dense (n, steps, f) tensor over the real 199k-bar grid would be
    hundreds of megabytes per fit, so only requested rows are materialised."""
    m = make('m15_session_lstm', _cfg('m15_session_lstm'))
    steps = m.config['lstm']['time_steps']
    inputs = m.build_inputs(m15)
    Xs = inputs.X.astype('float32')
    rows = np.array([steps - 1, 100, 250])
    win, ok = m._windows(Xs, rows)
    assert win.shape == (3, steps, Xs.shape[1]) and ok.all()
    for i, r in enumerate(rows):
        assert np.allclose(win[i, -1], Xs[r]), 'the window ENDS at its row'
        assert np.allclose(win[i, 0], Xs[r - steps + 1])
    # a row without enough history is reported through the mask, never guessed
    win2, ok2 = m._windows(Xs, np.array([steps - 2, 100]))
    assert list(ok2) == [False, True] and len(win2) == 1


def test_a_seed_reproduces_the_lstm(m15):
    a, inputs, tg, elig = _fit_one('m15_session_lstm', m15, 1)
    b = make('m15_session_lstm', _cfg('m15_session_lstm'))
    train = elig[elig < int(len(inputs) * 0.7)]
    b.fit(inputs, train, tg, 42, 1)
    assert np.allclose(a.predict(inputs)['p_up'][elig], b.predict(inputs)['p_up'][elig],
                       equal_nan=True)


def test_fit_refuses_too_few_labelled_rows(m15):
    m = make('m15_session_gbm', _cfg('m15_session_gbm'))
    inputs = m.build_inputs(m15)
    d, _ = T.direction_target(pd.Series(inputs.close), 1)
    with pytest.raises(ValueError, match='labelled training rows'):
        m.fit(inputs, np.arange(10), {'dir': d.to_numpy()}, 42, 1)


def test_make_rejects_an_unknown_price():
    with pytest.raises(KeyError, match='price definition'):
        make('m15_session_gbm', _cfg('m15_session_gbm'), price='ask')
