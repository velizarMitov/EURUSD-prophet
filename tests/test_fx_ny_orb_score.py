"""fx-ny-orb-hot, group 5.1: the historical scoring, its bootstrap and the gate."""
import copy
import glob
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb.score import ratio_block_means
from src.fx_ny_orb import REPO, load_config
from src.fx_ny_orb import labels as LB
from src.fx_ny_orb import prereg as PR
from src.fx_ny_orb import score as SC
from src.fx_ny_orb.registry import HIST, AlreadyScored, path
from tests.test_fx_ny_orb_prereg import synth

CFG = load_config()


def test_no_cold_trades_reproduces_the_fx_bb_ratio_bootstrap_exactly():
    rng = np.random.default_rng(0)
    S = rng.normal(0.1, 1.5, 1203) * (rng.random(1203) < 0.3)
    N = (S != 0).astype(float)
    hot, diff = SC.pair_block_means(S, N, np.zeros_like(S), np.zeros_like(S), 5, 4000, 20261127)
    assert np.array_equal(hot, ratio_block_means(S, N, 5, 4000, 20261127)) and np.isnan(diff).all()


def _bound_for(mean_hot, mean_cold, seed=1):
    rng = np.random.default_rng(seed)
    n = 2000
    nh = (rng.random(n) < 0.15).astype(float)
    nc = (rng.random(n) < 0.6).astype(float)
    sh = nh * (mean_hot + rng.normal(0, 1.5, n))
    sc = nc * (mean_cold + rng.normal(0, 1.5, n))
    hot, diff = SC.pair_block_means(sh, nh, sc, nc, 5, 2000, 7)
    return SC._bound(hot, 0.025, 0.01), SC._bound(diff, 0.025, 0.01)


def test_planted_edge_is_keep_and_none_is_drop():
    h, d = _bound_for(0.6, 0.0)
    assert h['verdict'] == 'KEEP' and d['verdict'] == 'KEEP'
    h, d = _bound_for(0.0, 0.0, seed=3)
    assert h['verdict'] == 'DROP' and d['verdict'] == 'DROP'


def test_too_many_empty_resamples_refuse():
    nh = np.zeros(400)
    nh[3] = 1.0
    hot, _ = SC.pair_block_means(nh * 2.0, nh, np.zeros(400), np.zeros(400), 5, 500, 1)
    with pytest.raises(SC.TooManyDropped):
        SC._bound(hot, 0.025, 0.01)


def _sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _registries():
    return {p: _sha(p) for p in glob.glob(os.path.join(REPO, 'results', '**', '*hypothesis_log.csv'), recursive=True)}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    c['data']['window_start'], c['data']['window_end'] = '2019-01-02', '2020-02-28'
    c['scoring']['resamples'] = 1000
    return c


def test_gate_refusal_leaves_no_outcome_on_disk(cfg):
    data = synth(250)
    LB.write_report(cfg, 'T', data)
    PR.write(cfg)

    def closed(c):
        raise PR.PreregNotCommitted('PRE_REGISTRATION.json is not committed')
    with pytest.raises(PR.PreregNotCommitted):
        SC.score_historical(cfg, 'T1', data, gate=closed)
    assert not os.path.exists(path(cfg, SC.HIST_FILE)) and not os.path.exists(path(cfg, HIST))


def test_full_historical_scoring_once(cfg):
    data = synth(250)
    before = _registries()
    LB.write_report(cfg, 'T', data)
    PR.write(cfg)
    rows = SC.score_historical(cfg, 'T1', data, gate=lambda c: '2026-11-22T10:00:00+00:00')
    assert [r['hypothesis'] for r in rows] == ['orb_hot_net_r', 'orb_hot_minus_cold']
    for r in rows:
        assert r['verdict'] in ('KEEP', 'DROP') and 'not a trading result' in r['notes'] and 'HAR-X' in r['notes']
        assert r['prereg_sha256'] and r['n_trades_hot'] > 10
    assert rows[1]['statistic'] == pytest.approx(rows[0]['hot_mean'] - rows[0]['cold_mean'])
    sim = pd.read_csv(path(cfg, SC.HIST_FILE))
    assert {'r_net', 'exit_reason', 'hot', 'reason'} <= set(sim.columns)
    frozen = _sha(path(cfg, HIST))
    with pytest.raises(AlreadyScored):
        SC.score_historical(cfg, 'T2', data, gate=lambda c: 'x')
    assert _sha(path(cfg, HIST)) == frozen and _registries() == before
