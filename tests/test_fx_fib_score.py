"""fx-fibonacci-mlp, group 9: one-shot scoring, the registry, the config guard and the report."""
import copy
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_fib import config_digest, feature_list_digest, load_config
from src.fx_fib import labeling as LB
from src.fx_fib import report as RP
from src.fx_fib import score as SC
from src.fx_fib import walk_forward_backtest as WF

CFG = load_config()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synthetic_oos(n=6000, mlp_edge=0.15, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for pair in ('EURUSD', 'GBPUSD'):
        t = pd.date_range('2019-07-01', periods=n, freq='h', tz='UTC')
        y = rng.choice(3, size=n, p=[0.78, 0.11, 0.11])
        prior = np.tile([0.78, 0.11, 0.11], (n, 1))
        onehot = np.eye(3)[y]
        mlp = (1 - mlp_edge) * prior + mlp_edge * onehot
        lgb = (1 - mlp_edge / 2) * prior + (mlp_edge / 2) * onehot
        d = pd.DataFrame({'pair': pair, 'time': t, 'pos': np.arange(n), 'fold': np.arange(n) // 1000,
                          'close': 1.1, 'label': y, 'exit_offset': 24, 'upper': 1.11, 'lower': 1.09,
                          'exit_close': 1.1, 'exit_day': t.normalize()})
        for name, p in (('prior', prior), ('mlp', mlp), ('lightgbm', lgb), ('logistic', prior), ('mlp_small', prior)):
            for c in range(3):
                d[f'p_{name}_{c}'] = p[:, c]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'd'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    c['scoring']['resamples'] = 2000
    os.makedirs(c['paths']['results'])
    return c


def _write_run(cfg, oos):
    oos.to_csv(os.path.join(cfg['paths']['results'], WF.OOS_FILE), index=False, compression='gzip')
    with open(os.path.join(cfg['paths']['results'], WF.RUN_FILE), 'w') as fh:
        json.dump({'run_at': 'T', 'config_digest': config_digest(cfg), 'device': 'NVIDIA test',
                   'feature_list_digest': feature_list_digest(cfg), 'torch': 'x', 'cuda': 'y', 'pairs': {}}, fh)


# -- the test --------------------------------------------------------------------

def test_block_bootstrap_centres_on_the_sample_mean_and_is_seeded():
    x = np.random.default_rng(1).normal(0.3, 1, 5000)
    a = SC.circular_block_means(x, 24, 3000, 7)
    assert a.mean() == pytest.approx(x.mean(), abs=0.02)
    assert np.array_equal(a, SC.circular_block_means(x, 24, 3000, 7))


def test_clear_improvement_is_keep():
    diff = -0.02 + np.random.default_rng(2).normal(0, 0.05, 20000)
    assert SC.test_cell(diff, 0.0125, 24, 3000, 5)['verdict'] == 'KEEP'


def test_interval_reaching_zero_is_drop():
    diff = np.random.default_rng(3).normal(0, 0.05, 20000)
    r = SC.test_cell(diff, 0.0125, 24, 3000, 5)
    assert r['verdict'] == 'DROP' and r['ci_upper'] > 0


def test_same_inputs_give_the_same_interval():
    diff = np.random.default_rng(4).normal(-0.001, 0.05, 8000)
    assert SC.test_cell(diff, 0.0125, 24, 2000, 9) == SC.test_cell(diff, 0.0125, 24, 2000, 9)


# -- the registry --------------------------------------------------------------------

def test_scoring_writes_four_cells_with_the_required_notes(cfg):
    _write_run(cfg, synthetic_oos())
    rows = SC.score_all(cfg, '2026-11-21T12:00:00+00:00')
    assert [r['hypothesis'] for r in rows] == ['EURUSD_mlp_vs_prior', 'EURUSD_mlp_vs_lightgbm',
                                              'GBPUSD_mlp_vs_prior', 'GBPUSD_mlp_vs_lightgbm']
    reg = SC.read_registry(cfg)
    assert len(reg) == 4 and all(r['alpha'] == '0.0125' for r in reg)
    assert all('Gross predictive result, not a trading edge' in r['notes'] for r in reg)
    assert '2024-11-19..2026-07-28' in reg[0]['notes'] and 'H_dir.3' in reg[2]['notes']
    assert {r['verdict'] for r in reg} == {'KEEP'}          # the synthetic MLP is clearly better


def test_second_scoring_is_refused_and_changes_nothing(cfg):
    _write_run(cfg, synthetic_oos())
    SC.score_all(cfg, 'first')
    before = open(SC.registry_path(cfg), 'rb').read()
    with pytest.raises(SC.AlreadyScored):
        SC.score_all(cfg, 'second')
    assert open(SC.registry_path(cfg), 'rb').read() == before


def test_predictions_from_another_config_are_refused(cfg):
    _write_run(cfg, synthetic_oos())
    other = copy.deepcopy(cfg)
    other['labels']['m_atr_d1'] = 1.5
    with pytest.raises(SC.ConfigChanged):
        SC.score_all(other, 'T')


def test_other_families_registries_are_untouched(cfg):
    paths = ['results/feature_hypothesis_log.csv', 'results/h1_direction_hypothesis_log.csv',
             'results/gold/gold_direction_hypothesis_log.csv', 'results/gold/gold_return_hypothesis_log.csv',
             'results/gold/gold_volatility_hypothesis_log.csv']
    paths = [os.path.join(REPO, *p.split('/')) for p in paths if os.path.exists(os.path.join(REPO, *p.split('/')))]
    digest = lambda: {p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}
    before = digest()
    _write_run(cfg, synthetic_oos())
    SC.score_all(cfg, 'T')
    assert digest() == before and len(paths) >= 2


# -- the guard ------------------------------------------------------------------------

def test_a_changed_label_parameter_is_refused_by_every_stage(cfg):
    _write_run(cfg, synthetic_oos())
    SC.score_all(cfg, 'T')
    changed = copy.deepcopy(cfg)
    changed['labels']['m_atr_d1'] = 1.5
    for stage in (lambda: SC.assert_config_matches_verdicts(changed),
                  lambda: LB.write_report(changed, 'T'),
                  lambda: WF.run_walk_forward(changed, 'T', log=lambda *a: None)):
        with pytest.raises(SC.ConfigChanged):
            stage()
    SC.assert_config_matches_verdicts(cfg)                 # the recorded config still passes


# -- the report -------------------------------------------------------------------------

def test_report_is_reproducible_and_carries_the_gross_header_and_verdicts(cfg):
    _write_run(cfg, synthetic_oos())
    SC.score_all(cfg, 'T')
    a = open(RP.build(cfg, 'A'), encoding='utf-8').read()
    b = open(RP.build(cfg, 'B'), encoding='utf-8').read()
    assert a.replace('_Generated A._', '') == b.replace('_Generated B._', '')
    assert RP.HEADER in a and 'not a trading result' in a
    assert '| EURUSD_mlp_vs_prior |' in a and '**KEEP**' in a
    assert '## Fixed rule — gross of spread and slippage' in a


def test_report_before_scoring_says_not_scored(cfg):
    _write_run(cfg, synthetic_oos())
    assert 'Not scored yet' in open(RP.build(cfg, 'T'), encoding='utf-8').read()
