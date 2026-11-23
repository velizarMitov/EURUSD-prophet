"""gold-bb-nyrange, tasks 6.1-6.2: the label report reads no outcome; the pre-registration gate."""
import copy
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import bars as B
from src.gold_bbnr import bb_events as E
from src.gold_bbnr import bb_trades as T
from src.gold_bbnr import labels as LB
from src.gold_bbnr import load_config
from src.gold_bbnr import ny_target as NT
from src.gold_bbnr import prereg as PR
from src.gold_bbnr.inputs import Inputs
from src.gold_bbnr.registry import ConfigChanged, read_json
from tests.test_gold_bbnr_ny import dxy_h1, gold_h1

CFG = load_config()
DIG = {'gold_h1': 'a' * 64, 'dxy_h1': 'b' * 64, 'dxy_d1': 'c' * 64}


def inputs(n_days=420):
    h1 = gold_h1(n_days, seed=4)
    dd = pd.bdate_range('2018-06-01', '2021-12-31')
    v = 95 + np.cumsum(np.random.default_rng(2).normal(0, 0.3, len(dd)))
    dxy_d1 = pd.DataFrame({'open': v, 'high': v, 'low': v, 'close': v}, index=pd.DatetimeIndex(dd, name='date'))
    return Inputs(h1, B.d1_ny(h1, CFG), dxy_d1, dxy_h1(h1))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results' / 'gold_bbnr')
    monkeypatch.setattr(LB, 'cache_digests', lambda cfg: dict(DIG))
    monkeypatch.setattr(PR, 'cache_digests', lambda cfg: dict(DIG))
    return c


def test_label_report_reads_no_outcome_exit_or_target(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('no outcome may be read before the pre-registration')
    for mod, name in ((E, 'outcomes'), (E, 'outcome_r'), (T, 'build_stream'), (T, 'scan'), (NT, 'target')):
        monkeypatch.setattr(mod, name, boom)
    p = LB.write_report(cfg, 'T', inputs())
    text = open(p, encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Gross of cost') and 'Minimum detectable' in text
    for word in ('Win rate', 'mean R:', 'take profit', 'KEEP', 'DROP'):
        assert word not in text
    rec = read_json(cfg, 'labels.json')
    assert rec['cache_digests'] == DIG and rec['part_1a']['counts']['first_breaches'] > 0


def test_label_report_refuses_a_changed_config(cfg):
    inp = inputs(200)
    LB.write_report(cfg, 'T', inp)
    other = copy.deepcopy(cfg)
    other['cooldown_days'] = 4
    with pytest.raises(ConfigChanged):
        LB.write_report(other, 'T', inp)


def git(repo, *args):
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, cfg):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'test@example.invalid')
    git(tmp_path, 'config', 'user.name', 'test')
    LB.write_report(cfg, 'T', inputs(200))
    PR.write(cfg)
    return str(tmp_path)


def test_gate_refuses_untracked_modified_config_and_cache(cfg, repo):
    with pytest.raises(PR.PreregNotCommitted, match='not committed'):
        PR.gate(cfg, repo, DIG)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'prereg')
    assert PR.gate(cfg, repo, DIG)
    with pytest.raises(PR.PreregNotCommitted, match='input cache changed'):
        PR.gate(cfg, repo, {**DIG, 'dxy_d1': 'd' * 64})
    other = copy.deepcopy(cfg)
    other['trade']['time_stop_days'] = 11
    with pytest.raises(PR.PreregNotCommitted, match='another config'):
        PR.gate(other, repo, DIG)
    md = os.path.join(cfg['paths']['results'], 'PRE_REGISTRATION.md')
    with open(md, 'a', encoding='utf-8') as fh:
        fh.write('edit\n')
    with pytest.raises(PR.PreregNotCommitted, match='differs'):
        PR.gate(cfg, repo, DIG)


def test_prereg_lists_the_seven_cells_and_the_digests(cfg, repo):
    text = open(os.path.join(cfg['paths']['results'], 'PRE_REGISTRATION.md'), encoding='utf-8').read()
    for c in ('gold_bb_rule_mean_reversion', 'gold_bb_lightgbm_vs_base_rate', 'gold_bb_sma_rule_mean_r',
              'gold_bb_sma_lightgbm_vs_base_rate', 'gold_harx_vs_har', 'gold_lgbm_vs_harx', 'gold_mlp_vs_harx'):
        assert f'`{c}`' in text
    assert all(v in text for v in DIG.values())
