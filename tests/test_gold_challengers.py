"""Gold challengers and baselines (tasks 4.1-4.3)."""
import copy
import os
import time

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import overfit
from src.forecast_eval.challengers.base import ChallengerUnavailable
from src.forecast_eval.tuning import grid_points
from src.gold import challengers as C
from src.gold import features as F
from src.gold import load_config
from src.gold import splits as S
from src.gold import targets as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config()
SMALL_GRID = {'gbm': {'grid': {'n_estimators': [50, 100], 'max_depth': [3], 'learning_rate': [0.1]},
                      'cv_splits': 2}}


def _random_walk(n, freq='D', seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('1950-01-02', periods=n, freq=freq, tz='UTC')
    c = 1500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.002, n))
    return pd.DataFrame({'open': o, 'high': np.maximum(o, c) * 1.004,
                         'low': np.minimum(o, c) * 0.996, 'close': c}, index=idx)


# ── 4.1 GBM ──────────────────────────────────────────────────────────────────

def test_gbm_on_shuffled_labels_scores_chance():
    df = _random_walk(30000)
    inp = C.build_inputs(df, 'D1')
    tg = G.build_targets(df['close'])
    p = S.stage_positions(len(df), 'validation', CFG)
    shuffled = dict(tg)
    rng = np.random.default_rng(0)
    d = tg['dir'].copy()
    d[p['fit']] = rng.permutation(d[p['fit']])
    shuffled['dir'] = d
    m = C.make('d1_gbm', CFG, overrides=SMALL_GRID).fit(inp, p['fit'], shuffled, seed=42)
    val = C.usable(inp, p['score'], tg, ('dir',))
    acc = float(((m.predict(inp)['p_up'][val] > 0.5) == (tg['dir'][val] == 1)).mean())
    assert 0.48 <= acc <= 0.52


def test_trial_log_gains_one_row_per_grid_point(tmp_path):
    df = _random_walk(3000)
    inp = C.build_inputs(df, 'D1')
    tg = G.build_targets(df['close'])
    p = S.stage_positions(len(df), 'validation', CFG)
    m = C.make('d1_gbm', CFG, overrides=SMALL_GRID).fit(inp, p['fit'], tg, seed=42)
    path = str(tmp_path / 'trial_log.csv')
    n = overfit.append_trials(m.trial_rows('test'), path=path)
    assert n == len(grid_points(SMALL_GRID['gbm']['grid'])) == 2
    assert sum('SELECTED' in r['notes'] for r in overfit.read_trials(path)) == 1


def test_declared_grid_has_eight_points():
    assert len(grid_points(CFG['gbm']['grid'])) == 8


def test_challenger_refuses_a_level_feature():
    df = _random_walk(600)
    inp = C.build_inputs(df, 'D1')
    m = C.make('d1_gbm', CFG, overrides={**SMALL_GRID, 'columns': ['close'] + F.D1_FEATURE_COLUMNS[1:]})
    with pytest.raises(F.ScaleDependentFeature):
        m.fit(inp, np.arange(400), G.build_targets(df['close']), seed=42)


# ── 4.2 LSTM ensemble ────────────────────────────────────────────────────────

def _two_seed_lstm():
    return C.make('d1_lstm', CFG, overrides={'seeds': [42, 43],
                                             'lstm': {'epochs': 2, 'units': 8, 'batch_size': 128}})


def test_lstm_smoke_fit_outputs_and_runtime():
    df = _random_walk(1500)
    inp = C.build_inputs(df, 'D1')
    tg = G.build_targets(df['close'])
    t0 = time.time()
    m = _two_seed_lstm().fit(inp, np.arange(1000), tg)
    out = m.predict(inp)
    assert time.time() - t0 < 120
    assert set(out) == {'p_up', 'ret_pct', 'vol_pct'}
    tail = slice(400, 1499)
    assert np.isfinite(out['p_up'][tail]).all() and (out['vol_pct'][tail] > 0).all()
    assert np.isnan(out['p_up'][:200]).all()            # warm-up windows are not zero-filled


def test_lstm_missing_seed_emits_no_prediction(tmp_path):
    df = _random_walk(800)
    inp = C.build_inputs(df, 'D1')
    m = _two_seed_lstm().fit(inp, np.arange(600), G.build_targets(df['close']))
    m.save(str(tmp_path))
    os.remove(tmp_path / 'seed43.keras')
    with pytest.raises(ChallengerUnavailable):
        _two_seed_lstm().load(str(tmp_path))
    m.models = m.models[:1]
    with pytest.raises(ChallengerUnavailable):
        m.predict(inp)


# ── 4.3 GARCH baselines ──────────────────────────────────────────────────────

@pytest.mark.parametrize('cell,freq', [('d1_garch', 'D'), ('h1_garch_hod', 'h')])
def test_garch_parameters_ignore_validation_rows(cell, freq):
    df = _random_walk(4000, freq=freq)
    p = S.stage_positions(len(df), 'validation', CFG)
    a = C.make(cell, CFG).fit(C.build_inputs(df, cell[:2].upper()), p['fit'])
    altered = df.copy()
    altered.iloc[p['fit_end']:] *= np.exp(np.random.default_rng(5).normal(0, 0.05, len(df) - p['fit_end']))[:, None]
    b = C.make(cell, CFG).fit(C.build_inputs(altered, cell[:2].upper()), p['fit'])
    assert a.params == b.params
    if cell == 'h1_garch_hod':
        assert a.hod == b.hod


@pytest.mark.parametrize('cell,cache', [('d1_garch', 'gold_d1.csv'), ('h1_garch_hod', 'gold_h1.csv')])
def test_garch_forecasts_are_positive_and_finite_on_live_caches(cell, cache):
    path = os.path.join(REPO, 'results', 'gold', cache)
    if not os.path.exists(path):
        pytest.skip('no live gold cache')
    from src.gold import data as D
    df = D.read_cache(path)
    inp = C.build_inputs(df, cell[:2].upper())
    p = S.stage_positions(len(df), 'validation', CFG)
    vol = C.make(cell, CFG).fit(inp, p['fit']).predict(inp)['vol_pct'][1:]
    assert np.isfinite(vol).all() and (vol > 0).all()
