"""gold-bb-nyrange, tasks 8.1-8.2: one-shot scoring, the report and the frozen models."""
import copy
import glob
import os

import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import REPO, load_config
from src.gold_bbnr import frozen as FZ
from src.gold_bbnr import report as RP
from src.gold_bbnr import score as S
from src.gold_bbnr.registry import AlreadyScored, read_rows
from src.gold_bbnr.walk_forward import FILES
from tests.test_fx_bb_protected import _digest

CFG = load_config()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    c['paths']['artifacts'] = str(tmp_path / 'art')
    c['scoring']['resamples'] = 2000
    os.makedirs(c['paths']['results'])
    return c


def no_gate(cfg):
    return 'T'


def plant(cfg, effect: float, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-02', periods=400)
    pick = np.sort(rng.choice(len(dates), 150, replace=False))
    d = dates[pick]
    r = rng.normal(effect, 1.0, len(d))
    y = (r > 0).astype(float)
    ev = pd.DataFrame({'date': d, 'dir': rng.choice([-1, 1], len(d)), 'reason': '', 'r': r, 'y': y})
    p_good = np.clip(np.where(y == 1, 0.5 + effect / 2, 0.5 - effect / 2), 0.01, 0.99)
    pr = pd.DataFrame({'date': d, 'y': y, 'fold': 1, 'p_prior': 0.5, 'p_lightgbm': p_good})
    tr = ev.rename(columns={'r': 'R'}).assign(exit_reason='stop')
    days = pd.bdate_range('2018-01-02', periods=600, tz='UTC')
    yy = rng.normal(-4.5, 0.4, len(days))
    noise = rng.normal(0, 0.4, len(days))
    ny = pd.DataFrame({'date': days, 'fold': 1, 'y': yy, 'p_har': yy + noise,
                       'p_harx': yy + noise * (1 - effect / 2), 'p_lightgbm': yy + noise * (1 - effect),
                       'p_mlp': yy + noise * (1 - effect)})
    out = cfg['paths']['results']
    ev.to_csv(os.path.join(out, FILES['bb'][0]), index=False, compression='gzip')
    pr.to_csv(os.path.join(out, FILES['bb'][1]), index=False, compression='gzip')
    tr.to_csv(os.path.join(out, FILES['bb_sma'][0]), index=False, compression='gzip')
    pr.to_csv(os.path.join(out, FILES['bb_sma'][1]), index=False, compression='gzip')
    ny.to_csv(os.path.join(out, FILES['nyvol'][1]), index=False, compression='gzip')


def other_registries():
    found = glob.glob(os.path.join(REPO, 'results', '**', '*hypothesis_log.csv'), recursive=True)
    return {os.path.relpath(p, REPO).replace('\\', '/'): _digest(os.path.relpath(p, REPO).replace('\\', '/'))
            for p in found if 'gold_bbnr' not in p}


def test_planted_effect_keeps_and_zero_drops(cfg, tmp_path):
    before = other_registries()
    plant(cfg, 0.6)
    for fam in ('bb', 'bb_sma', 'nyvol'):
        rows = S.score(fam, cfg, gate=no_gate)
        assert all(r['verdict'] == 'KEEP' for r in rows), [(r['hypothesis'], r['bound']) for r in rows]
    assert [len(read_rows(cfg, f)) for f in ('bb', 'bb_sma', 'nyvol')] == [2, 2, 3]
    assert other_registries() == before
    c2 = copy.deepcopy(cfg)
    c2['paths']['results'] = str(tmp_path / 'zero')
    os.makedirs(c2['paths']['results'])
    plant(c2, 0.0, seed=3)
    for fam in ('bb', 'bb_sma', 'nyvol'):
        assert all(r['verdict'] == 'DROP' for r in S.score(fam, c2, gate=no_gate))


def test_rescoring_is_refused_and_the_registry_unchanged(cfg):
    plant(cfg, 0.2)
    S.score('nyvol', cfg, gate=no_gate)
    p = os.path.join(cfg['paths']['results'], 'gold_nyvol_hypothesis_log.csv')
    before = open(p, 'rb').read()
    with pytest.raises(AlreadyScored):
        S.score('nyvol', cfg, gate=no_gate)
    assert open(p, 'rb').read() == before


def test_scoring_refuses_without_the_gate(cfg):
    plant(cfg, 0.2)

    def refuse(c):
        raise RuntimeError('not committed')
    with pytest.raises(RuntimeError, match='not committed'):
        S.score('bb', cfg, gate=refuse)
    assert not read_rows(cfg, 'bb')


def test_report_before_scoring_shows_no_outcome_and_is_deterministic(cfg):
    plant(cfg, 0.3)
    p = RP.write(cfg, 'A')
    t1 = open(p, encoding='utf-8').read()
    assert 'not scored yet' in t1 and 'Win' not in t1 and 'mean r' not in t1
    S.score('bb', cfg, gate=no_gate)
    S.score('nyvol', cfg, gate=no_gate)
    a = open(RP.write(cfg, 'A'), encoding='utf-8').read()
    b = open(RP.write(cfg, 'B'), encoding='utf-8').read()
    assert a.replace('_Generated A._', '') == b.replace('_Generated B._', '')
    assert 'HAR-X / HAR' in a and '| Year | N | Win | Mean |' in a and a.splitlines()[2].startswith('> **Gross of cost')


def test_frozen_models_refuse_an_altered_file(cfg):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for the MLP')
    c = copy.deepcopy(cfg)
    c['mlp'].update(seeds=[11], max_epochs=3, patience=2)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2019-01-01', periods=300, tz='UTC')
    rows = pd.DataFrame(rng.normal(size=(300, 14)), index=idx, columns=c['features_ny']['columns'])
    rows['y'] = rows.iloc[:, 0] * 0.3 + rng.normal(0, 0.1, 300)
    dig = FZ.fit(c, rows, log=lambda *a: None)
    assert set(dig) == {'ols.json', 'lightgbm.txt', 'mlp.pt'} and FZ.verify(c) == dig
    assert len(FZ.load_ols(c)['harx']) == 15
    with open(os.path.join(c['paths']['artifacts'], 'ols.json'), 'a', encoding='utf-8') as fh:
        fh.write(' ')
    with pytest.raises(FZ.FrozenChanged):
        FZ.verify(c)
