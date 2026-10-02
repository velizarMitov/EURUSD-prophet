"""Horizon-study orchestration end to end on a small dataset (task 8.1)."""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import record as R
from src.forecast_eval import study as ST
from src.forecast_eval.challengers.base import config_hash

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL_LSTM = {'time_steps': 5, 'units': 8, 'dropout': 0.1, 'learning_rate': 0.01,
              'epochs': 2, 'batch_size': 64, 'patience': 1}


def _macro(index):
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    t = np.arange(len(idx), dtype=float)
    return pd.DataFrame({'yield_differential': 1 + 0.01 * np.sin(t / 7), 'usd_index': 100 + 0.1 * t,
                         'policy_rate_differential': 0.5 + 0 * t,
                         'inflation_differential': 0.2 + 0.001 * t}, index=idx)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    out, art = tmp_path / 'results', tmp_path / 'research_models'
    monkeypatch.setattr(ST, 'ART', str(art))
    raw = F.load_daily_ohlcv().iloc[-700:]
    m = _macro(raw.index)
    sources = {'daily': {fs: F.daily_features(raw, m, fs) for fs in F.FEATURE_SETS},
               'h1': F.load_h1().iloc[-24 * 70:], 'macro_sources': {'all': 'synthetic'}}
    rec = R.new_record('tiny')
    rec['grid'] = {'H1': [1, 4], 'D1': [1]}
    keep = {'daily_gbm_price', 'h1_gbm', 'vol_ensemble', 'daily_lstm_price'}
    rec['challengers'] = {k: v for k, v in rec['challengers'].items() if k in keep}
    rec['challengers']['vol_ensemble'] = {**rec['challengers']['vol_ensemble'], 'lstm': SMALL_LSTM,
                                          'seeds': [1, 2]}
    rec['challengers']['daily_lstm_price'] = {**rec['challengers']['daily_lstm_price'], 'lstm': SMALL_LSTM}
    rec['config_hashes'] = {k: config_hash(v) for k, v in rec['challengers'].items()}
    rec['tuning']['grids'] = {'daily_gbm_price': {'max_depth': [2, 3]}, 'h1_gbm': {'max_depth': [3, 4]}}
    rec['tuning']['rule']['inner_folds'] = 2
    rec['development'] = {**rec['development'], 'walk_forward_splits': 2,
                          'cpcv': {'default': {'n_groups': 4, 'k': 2}}}
    path = str(out / 'study_record.json')
    R.save(rec, path)
    return {'out': str(out), 'art': str(art), 'sources': sources, 'record': path}


def test_end_to_end_with_failure_isolation_and_resume(setup, monkeypatch):
    import src.forecast_eval.study as mod
    real_make = mod.make

    def flaky_make(name, cfg):
        if name == 'daily_lstm_price':
            raise RuntimeError('deliberate failure for the isolation test')
        return real_make(name, cfg)
    monkeypatch.setattr(mod, 'make', flaky_make)

    ST.run_study('tiny', out=setup['out'], record_path=setup['record'], sources=setup['sources'])

    curves = pd.read_csv(os.path.join(setup['out'], 'horizon_curves.csv'))
    cells = set(zip(curves['model'], curves['horizon']))
    assert cells == {('daily_gbm_price', 1), ('h1_gbm', 1), ('h1_gbm', 4), ('vol_ensemble', 1)}
    assert (curves['banner'].str.startswith('DESCRIPTIVE')).all()
    assert curves.loc[curves['model'] == 'h1_gbm', 'label'].notna().all()
    assert {'breakeven_measured', 'breakeven_config', 'pt_p', 'auc', 'dsr', 'cpcv_net_median'} <= set(curves.columns)

    fails = pd.read_csv(os.path.join(setup['out'], 'failures.csv'))
    assert set(fails['model']) == {'daily_lstm_price'} and 'deliberate failure' in fails['error'].iloc[0]

    man = pd.read_csv(os.path.join(setup['out'], 'artifact_manifest.csv'))
    assert len(man) == 4
    for _, r in man.iterrows():
        assert r['path'].startswith('research_models/') or 'research_models' in r['path']
        for f, digest in json.loads(r['files_sha256']).items():
            full = f if os.path.isabs(f) else os.path.join(REPO, f)
            if not os.path.exists(full):
                full = os.path.join(os.path.dirname(setup['art']), f[f.index('research_models'):])
            assert hashlib.sha256(open(full, 'rb').read()).hexdigest() == digest
        assert r['train_start'] < r['train_end'] and isinstance(r['feature_digest'], str)

    trials_before = len(pd.read_csv(os.path.join(setup['out'], 'trial_log.csv')))
    assert trials_before >= 4
    assert R.load(setup['record'])['first_fit_at'] is not None

    ST.run_study('tiny', out=setup['out'], record_path=setup['record'], sources=setup['sources'])
    again = pd.read_csv(os.path.join(setup['out'], 'horizon_curves.csv'))
    assert len(again) == len(curves), 'a finished cell is never re-run'
    assert len(pd.read_csv(os.path.join(setup['out'], 'trial_log.csv'))) == trials_before


def test_study_refuses_without_its_record(setup):
    with pytest.raises(R.StudyRecordLocked):
        ST.run_study('other-id', out=setup['out'], record_path=setup['record'], sources=setup['sources'])


def test_non_overlapping_trades():
    assert list(ST.non_overlapping(np.arange(10), 3)) == [0, 3, 6, 9]
    assert list(ST.non_overlapping(np.array([0, 1, 5, 6, 7]), 2)) == [0, 5, 7]
