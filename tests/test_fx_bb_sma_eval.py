"""fx-bb-sma-exit, groups 4-5: folds from exit dates, 10-day day-block bootstrap, LightGBM wiring,
the walk-forward, one-shot scoring and the config guard."""
import copy
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb_sma import REPO
from src.fx_bb_sma import labels as LB
from src.fx_bb_sma import score as SC
from src.fx_bb_sma import splits as SP
from src.fx_bb_sma import walk_forward as WF
from src.fx_bb_sma.registry import AlreadyScored, ConfigChanged, StageOrderError, read_registry, registry_path
from src.fx_fib.score import circular_block_means
from src.fx_session.splits import Fold
from tests.fx_bb_sma_synth import synth_inputs, tmp_cfg_like

CFG = tmp_cfg_like()


# -- 4.1 folds ------------------------------------------------------------------------------------

def _trades(seed=0, n=1100):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2012-03-01', '2026-11-06')
    d = pd.DatetimeIndex(np.sort(rng.choice(days, size=n)))
    ex = pd.DatetimeIndex([x + pd.offsets.BDay(int(k)) for x, k in zip(d, rng.integers(1, 11, n))])
    return d, ex


def test_no_training_exit_within_10_business_days_of_the_next_slice():
    d, ex = _trades()
    folds = SP.make_folds(d, ex, CFG)
    for f in folds:
        v0 = d[f.val].min()
        assert (SP.bdays_between(ex[f.fit], v0) > 10).all() and (SP.bdays_between(ex[f.val], f.start) > 10).all()


def test_december_trade_exiting_on_the_24th_is_not_in_the_2020_training_data():
    d, ex = _trades(seed=1)
    d = d.append(pd.DatetimeIndex(['2019-12-10'])).sort_values()
    pos = int(np.flatnonzero(d == pd.Timestamp('2019-12-10'))[0])
    ex = ex.insert(pos, pd.Timestamp('2019-12-24'))
    f = [x for x in SP.make_folds(d, ex, CFG) if x.start == pd.Timestamp('2020-01-01')][0]
    assert pos not in set(f.fit) | set(f.val)


def test_every_out_of_sample_trade_once_and_contiguous_folds():
    d, ex = _trades(seed=2)
    folds = SP.make_folds(d, ex, CFG)
    tested = np.concatenate([f.eval for f in folds])
    assert len(tested) == len(set(tested)) == int((d >= pd.Timestamp('2015-01-01')).sum())
    assert all(a.end == b.start for a, b in zip(folds, folds[1:]))


# -- 4.2 bootstrap ---------------------------------------------------------------------------------

def test_one_trade_a_day_reproduces_the_project_bootstrap_with_10_day_blocks():
    x = np.random.default_rng(3).normal(0, 1, 1203)
    assert np.array_equal(SC.ratio_block_means(x, np.ones_like(x), 10, 5000, 20261125),
                          circular_block_means(x, 10, 5000, 20261125))


def _events(mean, seed=0, n_days=1500, sd=1.0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2012-03-01', periods=n_days)
    dates = days[np.sort(rng.choice(n_days, n_days // 2))]
    r = rng.normal(0, sd, len(dates))
    return dates, r - r.mean() + mean


def test_planted_positive_mean_is_keep_zero_is_drop_and_cell_2_keeps_only_below_zero():
    for mean, verdict in ((0.3, 'KEEP'), (0.0, 'DROP')):
        S, N = SC.day_sums(*_events(mean))
        assert SC.test_ratio(S, N, 0.025, 10, 2000, 1, 'greater')['verdict'] == verdict
    S, N = SC.day_sums(*_events(-0.05, sd=0.1))
    assert SC.test_ratio(S, N, 0.025, 10, 2000, 1, 'less')['verdict'] == 'KEEP'
    S, N = SC.day_sums(*_events(0.01, sd=0.1))
    assert SC.test_ratio(S, N, 0.025, 10, 2000, 1, 'less')['verdict'] == 'DROP'


# -- 4.3 LightGBM ---------------------------------------------------------------------------------

def test_same_seed_lightgbm_is_identical_and_the_base_rate_is_the_fit_win_rate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10))
    y = (X[:, 0] + rng.normal(0, 1.5, 600) > 0).astype(float)
    f = Fold(0, pd.Timestamp('2015-01-01'), pd.Timestamp('2016-01-01'), np.arange(400), np.arange(410, 480),
             np.arange(500, 600))
    a = WF.fit_fold(X, y, f, CFG)
    b = WF.fit_fold(X, y, f, CFG)
    assert np.array_equal(a[2], b[2]) and a[1] == pytest.approx(y[:400].mean())


# -- 5.1 / 5.2 the full stage on synthetic data ---------------------------------------------------

def _sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    return {p: _sha(p) for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))}


@pytest.fixture(scope='module')
def inputs():
    return synth_inputs(1300)


def test_walk_forward_refuses_before_the_label_report(tmp_path, inputs):
    with pytest.raises(StageOrderError, match='label'):
        WF.run(tmp_cfg_like(tmp_path), 'T', inputs, log=lambda *a: None)


def test_full_stage_and_one_shot_scoring(tmp_path, inputs):
    cfg = tmp_cfg_like(tmp_path)
    before = _registries()
    LB.write_report(cfg, 'T0', inputs)
    rec = WF.run(cfg, 'T1', inputs, log=lambda *a: None, digests={'x': 'y'})
    oos = WF.load_oos(cfg)
    assert len(oos) == rec['n_oos'] > 30 and not oos[['p_prior', 'p_lightgbm', 'y', 'r_net']].isna().any().any()
    assert not oos.duplicated(['pair', 'date']).any() and oos['date'].min() >= pd.Timestamp('2015-01-01')
    assert ((oos['r_net'] > 0).astype(float) == oos['y']).all()
    run = json.load(open(os.path.join(cfg['paths']['results'], 'run.json'), encoding='utf-8'))
    for f in run['folds']:
        assert np.busday_count(np.datetime64(f['val_last_exit']), np.datetime64(f['start'])) > 10
    r1 = SC.score_rule(cfg, 'T2', inputs)
    r2 = SC.score_model(cfg, 'T3')
    reg = read_registry(cfg)
    assert [r['hypothesis'] for r in reg] == ['sma_exit_rule_net_r', 'sma_exit_lightgbm_vs_base_rate']
    for r in reg:
        assert 'net of the recorded bar spread only' in r['notes'] and 'not a trading result' in r['notes']
        assert 'fx-bb-reversion' in r['notes'] and r['block_days'] == '10' and float(r['alpha']) == 0.025
    shares = sum(float(reg[0][k]) for k in ('share_take_profit', 'share_stop_loss', 'share_time_stop'))
    assert shares == pytest.approx(1.0) and reg[0]['ctrl_statistic'] != ''
    assert r1['verdict'] in ('KEEP', 'DROP') and r2['verdict'] in ('KEEP', 'DROP')
    frozen = _sha(registry_path(cfg))
    with pytest.raises(AlreadyScored):
        SC.score_rule(cfg, 'T4', inputs)
    with pytest.raises(AlreadyScored):
        SC.score_model(cfg, 'T5')
    assert _sha(registry_path(cfg)) == frozen and _registries() == before


def test_changed_config_is_refused_by_labels_walk_forward_and_score(tmp_path, inputs):
    cfg = tmp_cfg_like(tmp_path)
    LB.write_report(cfg, 'T0', inputs)
    WF.run(cfg, 'T1', inputs, log=lambda *a: None)
    other = copy.deepcopy(cfg)
    other['trade']['time_stop_days'] = 15
    for call in (lambda: LB.write_report(other, 'T', inputs), lambda: WF.run(other, 'T', inputs, log=lambda *a: None),
                 lambda: SC.score_rule(other, 'T', inputs), lambda: SC.score_model(other, 'T')):
        with pytest.raises(ConfigChanged):
            call()
    assert not os.path.exists(registry_path(cfg))


def test_parity_flag_never_changes_the_verdict(tmp_path, inputs, monkeypatch):
    cfg = tmp_cfg_like(tmp_path)
    LB.write_report(cfg, 'T0', inputs)
    tr = SC.scored_trades(cfg, inputs)
    shifted = tr.assign(r_ctrl=tr['r_ctrl'] - 0.2)
    monkeypatch.setattr(SC, 'scored_trades', lambda c, i=None: shifted)
    row = SC.score_rule(cfg, 'T1', inputs)
    assert row['parity_flag'] == 'FLAG'
    S, N = SC.day_sums(tr['date'], tr['r_net'])
    sc = cfg['scoring']
    assert row['verdict'] == SC.test_ratio(S, N, sc['alpha'], 10, sc['resamples'], sc['seed'], 'greater')['verdict']
