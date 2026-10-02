"""forecast_eval challengers (tasks 7.2-7.8). Small real-data fixtures and
reduced epochs keep these fast; the declared study configurations are tested
for identity, not trained at full size here."""
import os

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import record as R
from src.forecast_eval import targets as T
from src.forecast_eval.challengers import make
from src.forecast_eval.challengers.base import ChallengerUnavailable

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

SMALL_LSTM = {'time_steps': 5, 'units': 8, 'dropout': 0.1, 'learning_rate': 0.01,
              'epochs': 2, 'batch_size': 64, 'patience': 1}


def _targets(close, h):
    s = pd.Series(close)
    d, _ = T.direction_target(s, h)
    return {'dir': d.to_numpy(), 'ret': T.return_pct_target(s, h).to_numpy(),
            'vol': T.volatility_pct_target(s, h).to_numpy()}


@pytest.fixture(scope='module')
def daily_df():
    raw = F.load_daily_ohlcv().iloc[-1100:]
    return F.daily_features(raw, None, 'price')


@pytest.fixture(scope='module')
def h1():
    return F.load_h1().iloc[-24 * 160:]


@pytest.fixture(scope='module')
def rec():
    return R.new_record('test')


def _deterministic():
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()


# ── 7.2 daily GBM ───────────────────────────────────────────────────────────

def test_daily_gbm_fit_predict_and_roundtrip(daily_df, rec, tmp_path):
    m = make('daily_gbm_price', rec['challengers']['daily_gbm_price'])
    inp = m.build_inputs(daily_df)
    tg = _targets(inp.close, 1)
    m.fit(inp, np.arange(800), tg, seed=42, h=1)
    out = m.predict(inp)
    assert out['p_up'].shape == (len(inp),) and np.all((out['p_up'] >= 0) & (out['p_up'] <= 1))
    assert np.isfinite(out['ret_pct']).all()
    m.save(str(tmp_path / 'm'))
    m2 = make('daily_gbm_price', rec['challengers']['daily_gbm_price']).load(str(tmp_path / 'm'))
    assert np.array_equal(m2.predict(inp)['p_up'], out['p_up'])


def test_daily_gbm_config_identical_across_horizons(daily_df, rec, tmp_path):
    path = str(tmp_path / 'rec.json')
    R.save(rec, path)
    cfg = rec['challengers']['daily_gbm_macro']
    hashes = []
    for h in (1, 2, 5):
        m = make('daily_gbm_macro', cfg)
        R.assert_fit_allowed('daily_gbm_macro', m.config, h, 'D1', path)
        hashes.append(m.config_hash)
    assert len(set(hashes)) == 1


def test_daily_preprocessing_fits_on_train_only(daily_df, rec):
    m = make('daily_gbm_price', rec['challengers']['daily_gbm_price'])
    inp = m.build_inputs(daily_df)
    m.fit(inp, np.arange(600), _targets(inp.close, 1), seed=1, h=1)
    mu1 = m.prep.scaler.mean_.copy()
    X2 = inp.X.copy()
    X2[600:] *= 50                                     # wreck the non-training rows
    inp2 = type(inp)(inp.index, inp.close, X2, inp.extra)
    m.fit(inp2, np.arange(600), _targets(inp.close, 1), seed=1, h=1)
    assert np.allclose(m.prep.scaler.mean_, mu1)


# ── 7.3 daily LSTM ──────────────────────────────────────────────────────────

def test_daily_lstm_is_reproducible_with_a_fixed_seed(daily_df):
    _deterministic()
    cfg = {'cadence': 'D1', 'kind': 'direction', 'feature_set': 'price',
           'lag_pca_variance': 0.95, 'lstm': SMALL_LSTM}
    preds = []
    for _ in range(2):
        m = make('daily_lstm_price', cfg)
        inp = m.build_inputs(daily_df)
        m.fit(inp, np.arange(700), _targets(inp.close, 2), seed=7, h=2)
        preds.append(m.predict(inp))
    a, b = preds
    assert np.allclose(a['p_up'], b['p_up'], equal_nan=True)
    assert np.isnan(a['p_up'][:SMALL_LSTM['time_steps'] - 1]).all()   # no full window yet
    assert np.isfinite(a['p_up'][SMALL_LSTM['time_steps'] - 1:]).all()


# ── 7.4 H1 -> daily ensemble ────────────────────────────────────────────────

def test_h1_daily_ensemble_scalers_fit_on_train_only(h1, rec):
    cfg = dict(rec['challengers']['h1_daily_ensemble'])
    cfg['lstm'] = {'lstm_units': 8, 'lstm_dropout': 0.1, 'lstm_lr': 0.01, 'lstm_epochs': 2,
                   'lstm_batch': 32, 'lstm_patience': 1}
    cfg['rf'] = {'n_estimators': 20, 'max_depth': 4}
    m = make('h1_daily_ensemble', cfg)
    inp = m.build_inputs(h1)
    train = np.arange(90)
    m.fit(inp, train, _targets(inp.close, 1), seed=3, h=1)
    assert np.allclose(m.flat_scaler.mean_, inp.X[train].mean(axis=0))
    flat_seq = inp.extra['seq'][train].reshape(-1, inp.extra['seq'].shape[2])
    assert np.allclose(m.seq_scaler.mu, flat_seq.mean(axis=0))
    p = m.predict(inp)['p_up']
    assert p.shape == (len(inp),) and np.all((p >= 0) & (p <= 1))


# ── 7.5 TI-LSTM on CPU ──────────────────────────────────────────────────────

def test_ti_lstm_runs_without_cuda(h1, monkeypatch):
    import src.ti_lstm_h1_experimental as ti
    def no_cuda():
        raise RuntimeError('CUDA required')
    monkeypatch.setattr(ti, 'require_cuda', no_cuda)
    cfg = {'cadence': 'D1', 'kind': 'direction', 'n_layers': 1, 'units': 8, 'dropout': 0.1,
           'lr': 0.01, 'epochs': 2, 'batch_size': 32, 'patience': 1}
    m = make('ti_lstm', cfg)
    inp = m.build_inputs(h1)
    m.fit(inp, np.arange(100), _targets(inp.close, 1), seed=5, h=1)
    out = m.predict(inp)
    assert np.isfinite(out['p_up']).all() and np.isfinite(out['ret_pct']).all()


# ── 7.6 H1 next-bar GBM ─────────────────────────────────────────────────────

@pytest.mark.parametrize('h', [1, 4])
def test_h1_gbm_fit_predict(h1, rec, h):
    m = make('h1_gbm', rec['challengers']['h1_gbm'])
    inp = m.build_inputs(h1)
    m.fit(inp, np.arange(2500), _targets(inp.close, h), seed=42, h=h)
    p = m.predict(inp)['p_up']
    assert p.shape == (len(inp),) and np.all((p > 0) & (p < 1))


# ── 7.7 volatility ensemble and baselines ───────────────────────────────────

def test_vol_ensemble_uses_every_seed_and_refuses_partial(daily_df):
    cfg = {'cadence': 'D1', 'kind': 'volatility', 'feature_set': 'price',
           'lstm': SMALL_LSTM, 'seeds': [1, 2]}
    m = make('vol_ensemble', cfg)
    inp = m.build_inputs(daily_df)
    m.fit(inp, np.arange(700), _targets(inp.close, 1), seed=0, h=1)
    out = m.predict(inp)
    assert len(m.models) == 2
    assert np.all(out['vol_pct'][SMALL_LSTM['time_steps'] - 1:] >= 0)
    m.models = m.models[:1]
    with pytest.raises(RuntimeError, match='partial ensemble'):
        m.predict(inp)


def test_garch_dow_baseline_sees_no_test_rows(daily_df):
    from src.forecast_eval.challengers.volatility import GarchDow
    close = daily_df['close'].to_numpy()
    dow = pd.DatetimeIndex(daily_df.index).dayofweek.to_numpy()
    y = _targets(close, 2)['vol']
    train = np.zeros(len(close), dtype=bool)
    train[:800] = True
    a = GarchDow().fit(close, dow, y, train)
    y2 = y.copy()
    y2[800:] = 99.0
    b = GarchDow().fit(close, dow, y2, train)
    assert a.model.params == b.model.params
    pred = a.predict(close, dow)
    assert pred.shape == close.shape and np.all(pred[1:] > 0)


def test_random_walk_vol_is_the_past_h_bar_move():
    from src.forecast_eval.challengers.volatility import RandomWalkVol
    close = np.array([1.0, 1.1, 1.0, 1.2, 1.2])
    p = RandomWalkVol().predict(close, 2)
    assert np.isnan(p[:2]).all()
    assert p[2] == pytest.approx(abs(100 * np.log(1.0 / 1.0)))
    assert p[3] == pytest.approx(abs(100 * np.log(1.2 / 1.1)))


# ── 7.8 Kronos ──────────────────────────────────────────────────────────────

def test_kronos_unavailable_is_a_recorded_failure_not_a_crash(h1, rec, monkeypatch):
    from src.external.kronos import loader
    monkeypatch.setattr(loader, 'probe', lambda: (False, 'torch not installed'))
    m = make('kronos_direction', rec['challengers']['kronos_direction'])
    inp = m.build_inputs(h1)
    with pytest.raises(ChallengerUnavailable, match='torch not installed'):
        m.predict(inp, as_of_pos=[len(inp) - 1], h=1)


def test_kronos_reads_horizon_steps_from_one_sampling():
    from src.forecast_eval.challengers.kronos import Kronos
    paths = {3: np.array([[1.0, 2.0, 0.5], [1.0, 0.5, 0.5], [1.0, 2.0, 2.0]])}
    last = np.full(5, 1.0)
    p1 = Kronos.from_paths(paths, last, 1, 5, 'direction')
    p2 = Kronos.from_paths(paths, last, 2, 5, 'direction')
    assert np.isnan(p1[[0, 1, 2, 4]]).all()
    assert p1[3] == 0.0 and p2[3] == pytest.approx(2 / 3)


def test_kronos_skips_bars_before_its_clean_window(h1, rec):
    from src.external.kronos import loader
    ok, _ = loader.probe()
    if not ok:
        pytest.skip('Kronos weights not available')
    m = make('kronos_direction', {**rec['challengers']['kronos_direction'], 'n_paths': 4})
    inp = m.build_inputs(h1)
    last = len(inp) - 1
    paths = m.sample(inp, [10, last], max_h=4)
    assert 10 not in paths                          # not enough context
    assert paths[last].shape == (4, 4)
    p = m.predict(inp, h=4, paths=paths)['p_up']
    assert 0.0 <= p[last] <= 1.0 and np.isnan(p[10])
