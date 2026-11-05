"""Walk-forward of the acceleration study (openspec change gold-acceleration-direction, task 2.1)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.gold_accel import load_config
from src.gold_accel.walkforward import _guard, model_years, run

GOLD_CFG = {'seeds': [42, 43, 44, 45, 46]}


def _frames(start='2015-01-01', end='2026-03-01'):
    idx = pd.date_range(start, end, freq='h', tz='UTC', inclusive='left')
    idx = idx[idx.weekday < 5]
    c = 1200 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, len(idx))))
    h1 = pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close': c}, index=idx)
    m_idx = pd.date_range('2022-08-12', end, freq='15min', tz='UTC', inclusive='left')
    m_idx = m_idx[m_idx.weekday < 5]
    mc = 1700 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.001, len(m_idx))))
    m15 = pd.DataFrame({'open': mc, 'high': mc, 'low': mc, 'close': mc}, index=m_idx)
    return {'h1': h1, 'm15': m15}


class FakeModel:
    calls = []

    def __init__(self, base, overrides, fail_year=None):
        self.base, self.cols, self.fail_year = base, overrides['columns'], fail_year
        self.config_hash = 'fake'

    def fit(self, inputs, pos, targets, seed=None):
        self.last = inputs.index[pos[-1]]
        FakeModel.calls.append((self.base, len(self.cols), self.last))
        if self.fail_year and self.last.year + 1 == self.fail_year:
            raise RuntimeError('seed 44 diverged')
        return self

    def predict(self, inputs):
        p = np.where(inputs.extra['complete'], 0.6, np.nan)
        return {'p_up': p}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        f = os.path.join(path, 'model.txt')
        with open(f, 'w') as fh:
            fh.write(self.base)
        return [f]


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = {**c['paths'], 'models_dir': str(tmp_path / 'research_models' / 'gold' / 'accel'),
                  'inputs_dir': str(tmp_path / 'research_models' / 'gold' / 'accel' / 'inputs'),
                  'results': str(tmp_path / 'results' / 'gold' / 'accel'),
                  'artifact_manifest': str(tmp_path / 'results' / 'gold' / 'accel' / 'artifact_manifest.csv')}
    return c


def _make(fail_year=None):
    return lambda base, gold_cfg, overrides: FakeModel(base, overrides, fail_year)


def test_fit_ends_before_each_year_and_m15_models_start_in_2024(cfg):
    FakeModel.calls = []
    out = run(cfg, models=['h1_gbm_accel', 'h1_accel_m15'], frames=_frames(), make=_make(), gold_cfg=GOLD_CFG,
              log=lambda *_: None)
    years = sorted({t.year + 1 for _, _, t in FakeModel.calls})
    assert all(t < pd.Timestamp(f'{t.year + 1}-01-01', tz='UTC') for _, _, t in FakeModel.calls)
    assert years[0] == 2018 and years[-1] == 2026
    m15_fits = [t for b, n, t in FakeModel.calls if n == 8]       # 3 + accel_m15 + 4 calendar
    assert min(t.year + 1 for t in m15_fits) == 2024
    assert out['h1_accel_m15']['time'].min().year == 2024
    assert model_years('h1_gbm', _frames()['h1'].index, cfg)[0] == 2018


def test_a_resumed_year_is_not_refit(cfg):
    frames = _frames()
    run(cfg, models=['h1_accel'], frames=frames, make=_make(), gold_cfg=GOLD_CFG, log=lambda *_: None)
    FakeModel.calls = []
    out = run(cfg, models=['h1_accel'], frames=frames, make=_make(), gold_cfg=GOLD_CFG, log=lambda *_: None)
    assert FakeModel.calls == [] and len(out['h1_accel']) > 0


def test_a_failed_year_has_no_predictions(cfg):
    out = run(cfg, models=['h1_accel'], frames=_frames(), make=_make(fail_year=2020), gold_cfg=GOLD_CFG,
              log=lambda *_: None)
    assert 2020 not in set(out['h1_accel']['time'].dt.year)
    with open(os.path.join(cfg['paths']['models_dir'], 'h1_accel', '2020', 'done.json')) as fh:
        status = json.load(fh)
    assert status['ok'] is False and 'diverged' in status['error']


def test_nothing_is_written_outside_the_accel_folders(cfg, tmp_path):
    run(cfg, models=['h1_accel'], frames=_frames('2015-01-01', '2019-03-01'), make=_make(), gold_cfg=GOLD_CFG,
        log=lambda *_: None)
    roots = [os.path.normcase(cfg['paths']['models_dir']), os.path.normcase(cfg['paths']['results'])]
    written = [os.path.normcase(os.path.join(r, f)) for r, _, fs in os.walk(tmp_path) for f in fs]
    assert written and all(any(p.startswith(r) for r in roots) for p in written), written
    with pytest.raises(RuntimeError):
        _guard(str(tmp_path / 'models' / 'x.pkl'), cfg)
    reuse = json.load(open(os.path.join(cfg['paths']['models_dir'], 'h1_lstm_reuse.json')))
    assert reuse['reused'] is False and reuse['reason']
