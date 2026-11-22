"""fx-ny-orb-hot, group 6.4: the forward HAR-X cells, scored once."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import live as LV
from src.fx_ny_orb import load_config
from src.fx_ny_orb import score as SC
from src.fx_ny_orb.registry import FWD, AlreadyScored, StageOrderError, read_rows

CFG = load_config()
COMMIT = '2026-11-23T10:00:00+00:00'


def settle_log(cfg, n, gain, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2026-11-24', periods=n, tz='UTC')
    rows = []
    for p in cfg['pairs']:
        y = rng.normal(-5.5, 0.45, n)
        harx = y + rng.normal(0, 0.40 * (1 - gain), n)
        har = y + rng.normal(0, 0.40, n)
        rows += [{'date': d, 'pair': p, 'status': 'ok', 'y': a, 'p_harx': b, 'p_har': c}
                 for d, a, b, c in zip(days, y, harx, har)]
    old = pd.DataFrame([{'date': pd.Timestamp('2026-11-20', tz='UTC'), 'pair': 'EURUSD', 'status': 'ok', 'y': -5.0,
                         'p_harx': 9.0, 'p_har': -5.0}])      # before the commit: must be ignored
    LV.append_log(cfg, 'settlements', old.to_dict('records') + rows)
    return days


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    c['scoring']['resamples'] = 2000
    return c


def test_early_scoring_is_refused_and_writes_nothing(cfg):
    settle_log(cfg, 150, 0.3)
    with pytest.raises(StageOrderError, match='270'):
        SC.score_forward(cfg, pd.Timestamp('2027-07-01', tz='UTC'), gate=lambda c: COMMIT)
    assert read_rows(cfg, FWD) == []


def test_keep_on_a_real_gain_and_scored_once(cfg):
    settle_log(cfg, 300, 0.3)
    rows = SC.score_forward(cfg, pd.Timestamp('2028-01-15', tz='UTC'), gate=lambda c: COMMIT)
    assert [r['verdict'] for r in rows] == ['KEEP', 'KEEP'] and all(r['n_days'] == 300 for r in rows)
    assert rows[0]['first_day'] == '2026-11-24'
    with pytest.raises(AlreadyScored):
        SC.score_forward(cfg, pd.Timestamp('2028-01-20', tz='UTC'), gate=lambda c: COMMIT)


def test_drop_without_a_gain(cfg):
    settle_log(cfg, 300, 0.0, seed=4)
    rows = SC.score_forward(cfg, pd.Timestamp('2028-01-15', tz='UTC'), gate=lambda c: COMMIT)
    assert {r['verdict'] for r in rows} == {'DROP'}


def test_underpowered_at_the_deadline(cfg):
    settle_log(cfg, 120, 0.3)
    rows = SC.score_forward(cfg, pd.Timestamp('2028-01-31', tz='UTC'), gate=lambda c: COMMIT)
    assert {r['verdict'] for r in rows} == {'UNDERPOWERED — NO DECISION'}
