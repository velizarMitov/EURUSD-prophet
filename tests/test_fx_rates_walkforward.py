"""fx-rates-2y, group 6: the two-stage walk-forward, one-shot scoring and the registry."""
import copy
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.fx_rates import config_digest, feature_list_digest, load_config
from src.fx_rates import labeling as LB
from src.fx_rates import registry as RG
from src.fx_rates import score as SC
from src.fx_rates import walk_forward as WF
from tests.test_fx_rates_features import synth_d1, synth_yields
from tests.test_fx_rates_protected import REPO, other_registries

CFG = load_config()
quiet = lambda *a: None


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = dict(c['paths'], data=str(tmp_path / 'd'), results=str(tmp_path / 'r'), artifacts=str(tmp_path / 'a'))
    c['folds']['first_eval_start'] = '2014-01-01'
    c['scoring']['resamples'] = 2000
    c['mlp']['max_epochs'] = 3
    os.makedirs(c['paths']['results'])
    return c


@pytest.fixture(scope='module')
def inputs():
    data = LB.dataset({'EURUSD': synth_d1(1), 'GBPUSD': synth_d1(2)}, synth_yields(), CFG)
    return {p: d[d['reason'] == ''].sort_index() for p, d in data.items()}


def synth_oos(cfg, edge_e=0.1, edge_g=0.1, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for pair, edge in (('EURUSD', edge_e), ('GBPUSD', edge_g)):
        y = rng.integers(0, 2, n).astype(float)
        dates = pd.bdate_range('2015-01-01', periods=n)
        frames.append(pd.DataFrame({'pair': pair, 'date': dates, 'fold': np.arange(n) // 125, 'y': y,
                                    'entry_mid': 1.1, 'exit_mid': 1.1 + 0.002 * (2 * y - 1),
                                    'exit_date': dates + pd.offsets.BDay(5), 'yield_stale_flag': 0.0,
                                    'p_prior': 0.5, 'p_lightgbm': 0.5 + edge * (2 * y - 1) * (rng.random(n) < 0.8)}))
    pd.concat(frames).to_csv(os.path.join(cfg['paths']['results'], WF.OOS['lgbm']), index=False, compression='gzip')
    with open(os.path.join(cfg['paths']['results'], WF.RUN['lgbm']), 'w') as fh:
        json.dump({'run_at': 'T', 'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
                   'device': 'cpu (lightgbm)'}, fh)


def _digests():
    return {p: hashlib.sha256(open(os.path.join(REPO, p), 'rb').read()).hexdigest() for p in other_registries()}


def test_stage_one_writes_predictions_for_every_fold(cfg, inputs):
    run = WF.run_lgbm(cfg, 'T', inputs=inputs, log=quiet, digests={'x': 'y'})
    oos = WF.load_oos(cfg, 'lgbm')
    assert set(oos['pair']) == {'EURUSD', 'GBPUSD'} and not oos[['p_prior', 'p_lightgbm', 'y']].isna().any().any()
    assert not oos.duplicated(['pair', 'date']).any() and (oos['date'] >= pd.Timestamp('2014-01-01')).all()
    e, g = run['pairs']['EURUSD']['folds'], run['pairs']['GBPUSD']['folds']
    assert [(f['start'], f['end']) for f in e] == [(f['start'], f['end']) for f in g] and len(e) == 2
    assert run['input_digests'] == {'x': 'y'} and run['config_digest'] == config_digest(cfg)


def test_clear_edge_is_keep_and_no_edge_is_drop(cfg):
    synth_oos(cfg, 0.1, 0.0)
    rows = SC.score_lgbm(cfg, '2026-11-21T00:00:00+00:00')
    assert [r['verdict'] for r in rows] == ['KEEP', 'DROP']
    reg = RG.read_registry(cfg)
    assert all(r['alpha'] == '0.0125' and r['block_days'] == '10' for r in reg)
    assert all('not a trading edge' in r['notes'] and '#1' in r['notes'] and 'fx-session' in r['notes'] for r in reg)
    assert float(reg[0]['n_eff']) <= 1500 and float(reg[0]['accuracy_lo']) < float(reg[0]['accuracy_model'])


def test_second_scoring_is_refused_and_the_registry_is_byte_identical(cfg):
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'first')
    before = open(RG.registry_path(cfg), 'rb').read()
    with pytest.raises(RG.AlreadyScored):
        SC.score_lgbm(cfg, 'second')
    assert open(RG.registry_path(cfg), 'rb').read() == before


def test_a_changed_config_is_refused_by_every_stage(cfg, inputs):
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'T')
    changed = copy.deepcopy(cfg)
    changed['target']['horizon_days'] = 10
    for stage in (lambda: LB.write_report(changed, 'T', d1={'EURUSD': synth_d1(1), 'GBPUSD': synth_d1(2)},
                                          yields=synth_yields()),
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
    assert run['trained_pairs'] == [] and not os.path.exists(cfg['paths']['artifacts'])
    assert [r['verdict'] for r in SC.score_mlp(cfg, 'T')] == ['NOT TRIGGERED', 'NOT TRIGGERED']
    reg = RG.read_registry(cfg)
    assert len(reg) == 4 and {r['alpha'] for r in reg} == {'0.0125'} and 'not reassigned' in reg[3]['notes']


def test_mlp_stage_refuses_before_the_lightgbm_verdicts(cfg, inputs):
    with pytest.raises(WF.StageOrderError):
        WF.run_mlp(cfg, 'T', inputs=inputs, log=quiet)
    with pytest.raises(WF.StageOrderError):
        SC.score_mlp(cfg, 'T')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_mlp_trains_only_the_keep_pair_and_scores_it(cfg, inputs):
    WF.run_lgbm(cfg, 'T', inputs=inputs, log=quiet)
    oos = WF.load_oos(cfg, 'lgbm')
    RG.append_rows(cfg, [{'hypothesis': f'{p}_lightgbm_vs_prior', 'verdict': v, 'config_digest': config_digest(cfg),
                          'feature_list_digest': feature_list_digest(cfg)}
                         for p, v in (('EURUSD', 'DROP'), ('GBPUSD', 'KEEP'))])
    run = WF.run_mlp(cfg, 'T', inputs=inputs, log=quiet)
    assert run['trained_pairs'] == ['GBPUSD'] and run['device'] != 'cpu'
    m = WF.load_oos(cfg, 'mlp')
    assert set(m['pair']) == {'GBPUSD'} and len(m) == (oos['pair'] == 'GBPUSD').sum()
    out = SC.score_mlp(cfg, 'T')
    assert out[0]['verdict'] == 'NOT TRIGGERED' and out[1]['verdict'] in ('KEEP', 'DROP')


def test_scoring_leaves_other_registries_byte_identical(cfg):
    before = _digests()
    synth_oos(cfg)
    SC.score_lgbm(cfg, 'T')
    assert _digests() == before and len(before) >= 5
