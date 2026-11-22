"""fx-bb-reversion, group 6: the walk-forward, one-shot scoring, the registry and the config guard."""
import copy
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import REPO, load_config
from src.fx_bb import labels as LB
from src.fx_bb import score as SC
from src.fx_bb import walk_forward as WF
from src.fx_bb.registry import AlreadyScored, ConfigChanged, StageOrderError, read_registry, registry_path
from tests.test_fx_bb_features import walk

CFG = load_config()


def synthetic_d1(n=1300, seed=0):
    return {p: walk(n, seed + i, start='2012-01-02') for i, p in enumerate(CFG['pairs'])}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {**c['paths'], 'results': str(tmp_path / 'results'), 'artifacts': str(tmp_path / 'artifacts')}
    c['data']['start']['USDCHF'] = '2012-01-01'
    c['scoring']['resamples'] = 2000
    return c


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _real_registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    return {p: _sha(p) for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))}


def test_walk_forward_refuses_before_the_label_report(cfg):
    with pytest.raises(StageOrderError, match='label'):
        WF.run(cfg, 'T', synthetic_d1(), log=lambda *a: None)


def test_full_stage_writes_every_out_of_sample_event_once_and_scores_both_cells_once(cfg):
    d1 = synthetic_d1()
    before = _real_registries()
    LB.write_report(cfg, 'T0', d1)
    rec = WF.run(cfg, 'T1', d1, log=lambda *a: None, digests={'EURUSD': 'x'})
    oos = WF.load_oos(cfg)
    assert len(oos) == rec['n_oos'] > 50 and not oos[['p_prior', 'p_lightgbm', 'y', 'r']].isna().any().any()
    assert not oos.duplicated(['pair', 'date']).any()
    assert oos['date'].min() >= pd.Timestamp('2015-01-01') and set(oos['fold']) == {f['k'] for f in rec['folds']}
    run = json.load(open(os.path.join(cfg['paths']['results'], 'run.json'), encoding='utf-8'))
    assert run['config_digest'] and run['input_digests'] == {'EURUSD': 'x'}
    for f in run['folds']:
        assert np.busday_count(np.datetime64(f['val_last_t3']), np.datetime64(f['start'])) > 10
    man = pd.read_csv(os.path.join(cfg['paths']['results'], 'artifact_manifest.csv'))
    assert len(man) == len(rec['folds']) and man['path'].str.contains('POOLED/fold_').all()

    r1 = SC.score_rule(cfg, 'T2', d1)
    r2 = SC.score_model(cfg, 'T3')
    assert r1['verdict'] in ('KEEP', 'DROP') and r2['verdict'] in ('KEEP', 'DROP')
    reg = read_registry(cfg)
    assert [r['hypothesis'] for r in reg] == ['bb_rule_mean_reversion', 'bb_lightgbm_vs_base_rate']
    for r in reg:
        assert 'gross of spread and slippage, not a trading edge' in r['notes']
        assert all(f in r['notes'] for f in ('fx-fibonacci', 'fx-session', 'fx-rates'))
        assert float(r['alpha']) == 0.025 and r['block_days'] == '5'
    assert reg[0]['first_event'] and reg[0]['bid_statistic'] != '' and reg[0]['parity_flag'] in ('', 'FLAG')
    assert int(reg[1]['n_events']) == len(oos)

    frozen = _sha(registry_path(cfg))
    with pytest.raises(AlreadyScored):
        SC.score_rule(cfg, 'T4', d1)
    with pytest.raises(AlreadyScored):
        SC.score_model(cfg, 'T5')
    assert _sha(registry_path(cfg)) == frozen
    assert _real_registries() == before


def test_changed_config_is_refused_by_labels_walk_forward_and_score(cfg):
    d1 = synthetic_d1()
    LB.write_report(cfg, 'T0', d1)
    WF.run(cfg, 'T1', d1, log=lambda *a: None)
    other = copy.deepcopy(cfg)
    other['bands']['width'] = 2.5
    with pytest.raises(ConfigChanged):
        LB.write_report(other, 'T2', d1)
    with pytest.raises(ConfigChanged):
        WF.run(other, 'T3', d1, log=lambda *a: None)
    with pytest.raises(ConfigChanged):
        SC.score_rule(other, 'T4', d1)
    with pytest.raises(ConfigChanged):
        SC.score_model(other, 'T5')
    assert not os.path.exists(registry_path(cfg))


def test_parity_flag_is_recorded_without_changing_the_verdict(cfg, monkeypatch):
    d1 = synthetic_d1()
    LB.write_report(cfg, 'T0', d1)
    mid, bid = SC.rule_inputs(cfg, d1)
    bid = bid.assign(r=bid['r'] - 0.04 + (mid['r'].mean() - bid['r'].mean()))   # bid statistic 0.04 below mid
    monkeypatch.setattr(SC, 'rule_inputs', lambda c, d=None: (mid, bid))
    row = SC.score_rule(cfg, 'T1', d1)
    assert row['parity_flag'] == 'FLAG' and row['parity_gap'] == pytest.approx(0.04)
    S, N = SC.day_sums(mid['date'], mid['r'])
    sc = cfg['scoring']
    assert row['verdict'] == SC.test_ratio(S, N, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'],
                                           'greater')['verdict']
