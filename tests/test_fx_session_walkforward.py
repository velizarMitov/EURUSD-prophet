"""fx-session-ny, group 6: the two-stage walk-forward, one-shot scoring and the registry."""
import copy
import hashlib
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_session import config_digest, feature_list_digest, load_config
from src.fx_session import labeling as LB
from src.fx_session import registry as RG
from src.fx_session import score as SC
from src.fx_session import walk_forward as WF
from tests.test_fx_session_features import make_frames
from tests.test_fx_session_protected import REPO, other_registries

CFG = load_config()
POINTS = {'EURUSD': 1e-5, 'GBPUSD': 1e-5, 'AUDUSD': 1e-5}
quiet = lambda *a: None


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'd'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    c['folds'].update(first_eval_start='2018-06-01', fold_months=2)
    c['scoring']['resamples'] = 2000
    c['mlp']['max_epochs'] = 3
    os.makedirs(c['paths']['results'])
    return c


@pytest.fixture(scope='module')
def inputs():
    data = LB.dataset(make_frames(n_weeks=44, first=date(2018, 1, 7)), POINTS, CFG)
    return {p: d[d['reason'] == ''].sort_index() for p, d in data.items()}


def synth_oos(cfg, edge_e=0.1, edge_g=0.1, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for pair, edge in (('EURUSD', edge_e), ('GBPUSD', edge_g)):
        y = rng.integers(0, 2, n).astype(float)
        dates = pd.bdate_range('2013-01-01', periods=n, tz='UTC')
        d = pd.DataFrame({'pair': pair, 'date': dates, 'fold': np.arange(n) // 125, 'y': y, 'y_bid': y,
                          'entry_mid': 1.1, 'exit_mid': 1.1 + 0.001 * (2 * y - 1), 'entry_bid': 1.1,
                          'exit_bid': 1.1 + 0.001 * (2 * y - 1), 'dst_mismatch': 0.0, 'p_prior': 0.5,
                          'p_lightgbm': 0.5 + edge * (2 * y - 1) * (rng.random(n) < 0.8)})
        frames.append(d)
    pd.concat(frames).to_csv(os.path.join(cfg['paths']['results'], WF.OOS['lgbm']), index=False, compression='gzip')
    with open(os.path.join(cfg['paths']['results'], WF.RUN['lgbm']), 'w') as fh:
        json.dump({'run_at': 'T', 'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
                   'device': 'cpu (lightgbm)'}, fh)


def _digests():
    return {p: hashlib.sha256(open(os.path.join(REPO, p), 'rb').read()).hexdigest() for p in other_registries()}


# -- 6.1 stage 1 ----------------------------------------------------------------------------------

def test_stage_one_writes_predictions_for_every_fold(cfg, inputs):
    run = WF.run_lgbm(cfg, 'T', inputs=inputs, log=quiet)
    oos = WF.load_oos(cfg, 'lgbm')
    assert set(oos['pair']) == {'EURUSD', 'GBPUSD'}
    assert not oos[['p_prior', 'p_lightgbm']].isna().any().any()
    assert not oos.duplicated(['pair', 'date']).any()
    assert (oos['date'] >= pd.Timestamp('2018-06-01', tz='UTC')).all()
    e, g = run['pairs']['EURUSD']['folds'], run['pairs']['GBPUSD']['folds']
    assert [(f['start'], f['end']) for f in e] == [(f['start'], f['end']) for f in g] and len(e) >= 2
    assert run['config_digest'] == config_digest(cfg)
    assert os.path.exists(os.path.join(cfg['paths']['results'], 'artifact_manifest.csv'))


# -- 6.2 scoring --------------------------------------------------------------------------------------

def test_clear_edge_is_keep_and_no_edge_is_drop(cfg):
    synth_oos(cfg, edge_e=0.1, edge_g=0.0)
    rows = SC.score_lgbm(cfg, '2026-11-21T00:00:00+00:00')
    assert [r['hypothesis'] for r in rows] == ['EURUSD_lightgbm_vs_prior', 'GBPUSD_lightgbm_vs_prior']
    assert [r['verdict'] for r in rows] == ['KEEP', 'DROP']
    reg = RG.read_registry(cfg)
    assert all(r['alpha'] == '0.0125' and r['block_days'] == '5' for r in reg)
    assert all('not a trading edge' in r['notes'] and 'fx-fibonacci' in r['notes'] for r in reg)


def test_second_scoring_is_refused_and_the_registry_is_byte_identical(cfg):
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'first')
    before = open(RG.registry_path(cfg), 'rb').read()
    with pytest.raises(RG.AlreadyScored):
        SC.score_lgbm(cfg, 'second')
    with pytest.raises(RG.AlreadyScored):
        RG.append_rows(cfg, [{'hypothesis': 'EURUSD_lightgbm_vs_prior'}])
    assert open(RG.registry_path(cfg), 'rb').read() == before


def test_a_changed_config_is_refused_by_every_stage(cfg, inputs):
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'T')
    changed = copy.deepcopy(cfg)
    changed['session']['entry_hour'] = 15
    for stage in (lambda: LB.write_report(changed, 'T', frames=make_frames(), points=POINTS),
                  lambda: WF.run_lgbm(changed, 'T', inputs=inputs, log=quiet),
                  lambda: WF.run_mlp(changed, 'T', inputs=inputs, log=quiet),
                  lambda: SC.score_mlp(changed, 'T')):
        with pytest.raises(RG.ConfigChanged):
            stage()


def test_predictions_from_another_config_are_refused(cfg):
    synth_oos(cfg)
    other = copy.deepcopy(cfg)
    other['lightgbm']['num_leaves'] = 15
    with pytest.raises(RG.ConfigChanged):
        SC.score_lgbm(other, 'T')


def test_both_drop_trains_nothing_and_records_not_triggered(cfg, inputs):
    synth_oos(cfg, 0.0, 0.0)
    SC.score_lgbm(cfg, 'T')
    run = WF.run_mlp(cfg, 'T', inputs=inputs, log=quiet)
    assert run['trained_pairs'] == []
    assert not os.path.exists(os.path.join(cfg['paths']['artifacts']))
    rows = SC.score_mlp(cfg, 'T')
    assert [r['verdict'] for r in rows] == ['NOT TRIGGERED', 'NOT TRIGGERED']
    reg = RG.read_registry(cfg)
    assert len(reg) == 4 and {r['alpha'] for r in reg} == {'0.0125'}
    assert 'not reassigned' in reg[2]['notes']


def test_mlp_stage_refuses_before_the_lightgbm_verdicts(cfg, inputs):
    with pytest.raises(WF.StageOrderError):
        WF.run_mlp(cfg, 'T', inputs=inputs, log=quiet)
    with pytest.raises(WF.StageOrderError):
        SC.score_mlp(cfg, 'T')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_mlp_trains_only_the_keep_pair_and_scores_it(cfg, inputs):
    WF.run_lgbm(cfg, 'T', inputs=inputs, log=quiet)
    oos = WF.load_oos(cfg, 'lgbm')
    rows = [{'hypothesis': f'{p}_lightgbm_vs_prior', 'verdict': v, 'config_digest': config_digest(cfg),
             'feature_list_digest': feature_list_digest(cfg)} for p, v in (('EURUSD', 'KEEP'), ('GBPUSD', 'DROP'))]
    RG.append_rows(cfg, rows)
    run = WF.run_mlp(cfg, 'T', inputs=inputs, log=quiet)
    assert run['trained_pairs'] == ['EURUSD'] and run['device'] != 'cpu'
    m = WF.load_oos(cfg, 'mlp')
    assert set(m['pair']) == {'EURUSD'} and len(m) == (oos['pair'] == 'EURUSD').sum()
    out = SC.score_mlp(cfg, 'T')
    assert out[0]['hypothesis'] == 'EURUSD_mlp_vs_lightgbm' and out[0]['verdict'] in ('KEEP', 'DROP')
    assert out[1]['verdict'] == 'NOT TRIGGERED'


def test_scoring_leaves_other_families_registries_byte_identical(cfg):
    before = _digests()
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'T')
    assert _digests() == before and len(before) >= 5
