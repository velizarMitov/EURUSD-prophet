"""fx-ny-volatility, group 5: the walk-forward stages, one-shot scoring, the registry and the guard."""
import copy
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_nyvol import REPO
from src.fx_nyvol import labels as LB
from src.fx_nyvol import score as SC
from src.fx_nyvol import target as TG
from src.fx_nyvol import walk_forward as WF
from src.fx_nyvol.registry import AlreadyScored, ConfigChanged, StageOrderError, read_registry, registry_path
from tests.fx_nyvol_synth import POINTS, tmp_cfg, walk_frames

QUIET = {'log': lambda *a: None}


@pytest.fixture(scope='module')
def data():
    return TG.dataset(walk_frames(n_weeks=75), POINTS, tmp_cfg())


def _sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    return {p: _sha(p) for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))}


def _cfg(tmp_path):
    c = tmp_cfg(tmp_path)
    c['scoring']['resamples'] = 1000
    c['mlp']['max_epochs'] = 15
    return c


def _verdicts(monkeypatch, verdicts):
    """Force the bootstrap verdict per call (pair order), keep every other field real."""
    real = SC.test_cell
    it = iter(verdicts)

    def fake(diff, *a, **k):
        out = real(diff, *a, **k)
        out['verdict'] = next(it)
        return out
    monkeypatch.setattr(SC, 'test_cell', fake)


def test_stage_1_refuses_before_the_label_report(tmp_path, data):
    with pytest.raises(StageOrderError, match='label'):
        WF.run_lgbm(_cfg(tmp_path), 'T', data, **QUIET)


def test_full_stage_1_and_one_shot_lightgbm_scoring(tmp_path, data):
    cfg = _cfg(tmp_path)
    before = _registries()
    LB.write_report(cfg, 'T0', data)
    run = WF.run_lgbm(cfg, 'T1', data, digests={'x': 'y'}, **QUIET)
    oos = WF.load_oos(cfg, 'lgbm')
    assert len(oos) == sum(r['n_eval'] for r in run['pairs'].values()) > 50
    assert not oos[['y', 'p_har', 'p_harx', 'p_lightgbm', 'hot_threshold']].isna().any().any()
    assert not oos.duplicated(['pair', 'date']).any() and oos['date'].min() >= pd.Timestamp('2013-01-01', tz='UTC')
    rec = json.load(open(os.path.join(cfg['paths']['results'], 'run_lgbm.json'), encoding='utf-8'))
    assert rec['input_digests'] == {'x': 'y'} and all(len(f['har_coef']) == 4 for f in rec['pairs']['EURUSD']['folds'])
    rows = SC.score_lgbm(cfg, 'T2')
    assert [r['hypothesis'] for r in rows] == ['EURUSD_lightgbm_vs_har', 'GBPUSD_lightgbm_vs_har']
    for r in rows:
        assert r['rel_mse_reduction'] == pytest.approx(1 - r['mse_model'] / r['mse_reference'])
        assert 'Gross of spread and slippage' in r['notes'] and 'fx-session-ny' in r['notes']
    frozen = _sha(registry_path(cfg))
    with pytest.raises(AlreadyScored):
        SC.score_lgbm(cfg, 'T3')
    assert _sha(registry_path(cfg)) == frozen and _registries() == before


def test_keep_on_a_planted_improvement_and_drop_on_none(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(0)
    y = rng.normal(-5.5, 0.5, 2000)
    d = pd.DataFrame({'y': y, 'fold': 0, 'date': pd.bdate_range('2013-01-01', periods=2000, tz='UTC')})
    good = SC._row(cfg, 1, 'T', 'EURUSD', 'lightgbm', 'har', d, y + rng.normal(0, 0.3, 2000),
                   y + rng.normal(0, 0.4, 2000), {})
    same = SC._row(cfg, 1, 'T', 'EURUSD', 'lightgbm', 'har', d, y + rng.normal(0, 0.4, 2000),
                   y + rng.normal(0, 0.4, 2000), {})
    assert good['verdict'] == 'KEEP' and good['rel_mse_reduction'] > 0.3
    assert same['verdict'] == 'DROP'


def test_relative_reduction_of_ten_percent():
    assert 1 - 0.180 / 0.200 == pytest.approx(0.10)


def test_changed_config_is_refused_by_labels_walk_forward_and_score(tmp_path, data):
    cfg = _cfg(tmp_path)
    LB.write_report(cfg, 'T0', data)
    WF.run_lgbm(cfg, 'T1', data, **QUIET)
    other = copy.deepcopy(cfg)
    other['session']['target_hours'] = [15, 22]
    for call in (lambda: LB.write_report(other, 'T', data), lambda: WF.run_lgbm(other, 'T', data, **QUIET),
                 lambda: SC.score_lgbm(other, 'T'), lambda: WF.run_mlp(other, 'T', data, **QUIET)):
        with pytest.raises(ConfigChanged):
            call()
    assert not os.path.exists(registry_path(cfg))


def test_mlp_stage_refuses_without_both_lightgbm_rows(tmp_path, data):
    cfg = _cfg(tmp_path)
    LB.write_report(cfg, 'T0', data)
    WF.run_lgbm(cfg, 'T1', data, **QUIET)
    with pytest.raises(StageOrderError, match='LightGBM verdicts'):
        WF.run_mlp(cfg, 'T2', data, **QUIET)


def test_both_drop_trains_nothing_and_records_not_triggered(tmp_path, data, monkeypatch):
    cfg = _cfg(tmp_path)
    LB.write_report(cfg, 'T0', data)
    WF.run_lgbm(cfg, 'T1', data, **QUIET)
    _verdicts(monkeypatch, ['DROP', 'DROP'])
    SC.score_lgbm(cfg, 'T2')
    run = WF.run_mlp(cfg, 'T3', data, **QUIET)
    assert run['trained_pairs'] == [] and len(WF.load_oos(cfg, 'mlp')) == 0
    rows = SC.score_mlp(cfg, 'T4')
    assert [r['verdict'] for r in rows] == ['NOT TRIGGERED', 'NOT TRIGGERED']
    reg = read_registry(cfg)
    assert len(reg) == 4 and 'not reassigned' in reg[2]['notes']
    with pytest.raises(AlreadyScored):
        SC.score_mlp(cfg, 'T5')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_mlp_trains_only_the_keep_pair_on_the_gpu(tmp_path, data, monkeypatch):
    cfg = _cfg(tmp_path)
    LB.write_report(cfg, 'T0', data)
    WF.run_lgbm(cfg, 'T1', data, **QUIET)
    _verdicts(monkeypatch, ['KEEP', 'DROP'])
    SC.score_lgbm(cfg, 'T2')
    monkeypatch.undo()
    run = WF.run_mlp(cfg, 'T3', data, **QUIET)
    assert run['trained_pairs'] == ['EURUSD'] and ('NVIDIA' in run['device'] or 'GeForce' in run['device'])
    m = WF.load_oos(cfg, 'mlp')
    assert set(m['pair']) == {'EURUSD'} and not m['p_mlp'].isna().any()
    rows = SC.score_mlp(cfg, 'T4')
    assert rows[0]['verdict'] in ('KEEP', 'DROP') and rows[1]['verdict'] == 'NOT TRIGGERED'
