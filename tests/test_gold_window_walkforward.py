"""Walk-forward predictions (openspec change gold-window-strategy, task 4.1)."""
import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import gold_window as W
from src.gold_window import walkforward as WF
from src.forecast_eval.challengers.base import ChallengerUnavailable


class FakeModel:
    fits = []

    def __init__(self, name, fail_years=()):
        self.name, self.fail_years = name, fail_years

    def fit(self, inputs, pos, targets, seed=None):
        # the year being predicted is the one after the last fit row (fits end in December)
        self.year = inputs.index[pos[-1]].year + 1
        FakeModel.fits.append((self.name, self.year, inputs.index[pos[-1] + 1]))
        return self

    def predict(self, inputs):
        if self.year in self.fail_years:
            raise ChallengerUnavailable('1 of 5 seed models available -- a partial ensemble is never served')
        n = len(inputs.index)
        return {'p_up': np.full(n, 0.6), 'vol_pct': np.arange(n, dtype=float)}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        f = os.path.join(path, 'challenger.json')
        with open(f, 'w') as fh:
            fh.write(self.name)
        return [f]


class FakeInputs:
    def __init__(self, df):
        self.index = df.index


def _frames():
    h1 = pd.DataFrame({'close': 1.0}, index=pd.date_range('2015-01-02', '2019-12-31 23:00', freq='6h', tz='UTC'))
    d1 = pd.DataFrame({'close': 1.0}, index=pd.date_range('2013-08-08', '2019-12-31', freq='D', tz='UTC'))
    return {'H1': h1, 'D1': d1}


def _cfg(tmp_path):
    cfg = copy.deepcopy(W.load_config())
    cfg['paths']['models_dir'] = str(tmp_path / 'models')
    cfg['paths']['artifact_manifest'] = str(tmp_path / 'artifact_manifest.csv')
    return cfg


def _run(tmp_path, fail_years=()):
    FakeModel.fits = []
    return WF.run(_cfg(tmp_path), frames=_frames(), make=lambda name, gc: FakeModel(name, fail_years),
                  build_inputs=lambda df, cad: FakeInputs(df), build_targets=lambda close: {},
                  gold_cfg={'seeds': [42, 43, 44, 45, 46]})


def test_year_positions_purge_the_row_whose_target_is_in_the_year():
    idx = pd.date_range('2019-12-30', '2020-01-02', freq='D', tz='UTC')
    fit, pred = WF.year_positions(idx, 2020)
    assert list(fit) == [0] and list(pred) == [2, 3]            # 12-31's target is 01-01


def test_every_fit_target_is_labelled_before_its_prediction_year(tmp_path):
    out = _run(tmp_path)
    assert FakeModel.fits
    for _name, year, last_target in FakeModel.fits:
        assert last_target < pd.Timestamp(f'{year}-01-01', tz='UTC')
    for model, frame in out.items():
        fitted = {y for n, y, _t in FakeModel.fits if n == model}
        assert set(frame['time'].dt.year) == fitted
    # warm-up: H1 from 2015 predicts 2018 on, D1 from 2013 predicts 2016 on
    assert out['h1_lstm']['time'].dt.year.min() == 2018
    assert out['d1_lstm']['time'].dt.year.min() == 2016 and out['d1_garch']['time'].dt.year.min() == 2016


def test_missing_seed_year_has_no_neural_predictions(tmp_path):
    out = _run(tmp_path, fail_years=(2019,))
    assert 2019 not in set(out['h1_lstm']['time'].dt.year)
    with open(os.path.join(tmp_path, 'models', 'h1_lstm', '2019', 'done.json')) as fh:
        st = json.load(fh)
    assert st['ok'] is False and 'partial ensemble' in st['error']


def test_resume_skips_completed_years(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert FakeModel.fits == []
    for k in first:
        pd.testing.assert_frame_equal(first[k].reset_index(drop=True), second[k].reset_index(drop=True),
                                      check_dtype=False)


def test_writes_only_inside_its_own_folders(tmp_path):
    repo_manifest = W.repo_path(W.load_config()['paths']['artifact_manifest'])
    before = os.path.getmtime(repo_manifest) if os.path.exists(repo_manifest) else None
    _run(tmp_path)
    after = os.path.getmtime(repo_manifest) if os.path.exists(repo_manifest) else None
    assert before == after
    written = {os.path.relpath(os.path.join(r, f), tmp_path).split(os.sep)[0]
               for r, _d, fs in os.walk(tmp_path) for f in fs}
    assert written == {'models', 'artifact_manifest.csv'}
