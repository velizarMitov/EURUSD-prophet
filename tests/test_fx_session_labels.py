"""fx-session-ny, group 4: the binary target, the label report and the power table."""
import copy
import csv
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_session import labeling as LB
from src.fx_session import load_config
from src.fx_session import registry as RG
from tests.test_fx_session_features import make_frames

CFG = load_config()
POINTS = {'EURUSD': 1e-5, 'GBPUSD': 1e-5, 'AUDUSD': 1e-5}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'd'), 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    c['folds']['first_eval_start'] = '2018-02-01'
    return c


def _set(frames, pair, d, hour, close, spread=6.0):
    t = d + pd.Timedelta(hours=hour)
    frames[pair].loc[t, ['close', 'spread']] = [close, spread]


def test_up_day_is_one_and_down_day_is_zero():
    fr = make_frames()
    d = pd.Timestamp('2018-02-07', tz='UTC')
    _set(fr, 'EURUSD', d, 14, 1.08000)
    _set(fr, 'EURUSD', d, 21, 1.08247)
    _set(fr, 'GBPUSD', d, 14, 1.30000)
    _set(fr, 'GBPUSD', d, 21, 1.29900)
    t = LB.build_targets(fr, POINTS, CFG)
    assert t['EURUSD'].loc[d, 'entry_mid'] == pytest.approx(1.08003)
    assert t['EURUSD'].loc[d, 'y'] == 1.0 and t['GBPUSD'].loc[d, 'y'] == 0.0


def test_flat_session_is_a_counted_tie_with_no_target():
    fr = make_frames()
    d = pd.Timestamp('2018-02-07', tz='UTC')
    _set(fr, 'EURUSD', d, 14, 1.08000)
    _set(fr, 'EURUSD', d, 21, 1.08000)
    t = LB.build_targets(fr, POINTS, CFG)['EURUSD']
    assert t.loc[d, 'tie'] and np.isnan(t.loc[d, 'y'])
    data = LB.dataset(fr, POINTS, CFG)['EURUSD']
    assert data.loc[d, 'reason'] == 'tie'


def test_mid_and_bid_targets_can_differ_by_the_spread():
    fr = make_frames()
    d = pd.Timestamp('2018-02-07', tz='UTC')
    _set(fr, 'EURUSD', d, 14, 1.08000, spread=2.0)
    _set(fr, 'EURUSD', d, 21, 1.07999, spread=6.0)       # bid down 1 point, mid up 1 point
    t = LB.build_targets(fr, POINTS, CFG)['EURUSD']
    assert t.loc[d, 'y'] == 1.0 and t.loc[d, 'y_bid'] == 0.0


def test_consecutive_days_share_no_bar():
    d = pd.DatetimeIndex(['2018-02-06', '2018-02-07', '2018-02-09'], tz='UTC')
    sp = LB.target_spans(d, CFG)
    assert (sp['first'].iloc[1:].to_numpy() > sp['last'].iloc[:-1].to_numpy()).all()
    assert (sp['first'].dt.hour == 15).all() and (sp['last'].dt.hour == 21).all()


def test_power_table_for_3450_days():
    r = LB.power_row(3450, 0.0125, 0.8)
    assert r['mde_accuracy'] == pytest.approx(0.526, abs=0.001)
    assert 0 < r['mde_logloss_gain'] < 0.002


def test_label_report_sections_and_no_model_output(cfg):
    path = LB.write_report(cfg, 'T', frames=make_frames(n_weeks=16), points=POINTS)
    text = open(path, encoding='utf-8').read()
    assert 'not a trading result' in text.splitlines()[2]
    for section in ('## Up-rate per year', '## Excluded days', '## Power', 'Ties', 'DST-mismatch days'):
        assert section in text
    assert 'warm-up' in text
    assert os.listdir(cfg['paths']['results']) == ['labels.md']       # nothing but the report


def test_label_stage_refuses_a_config_that_differs_from_the_verdicts(cfg):
    os.makedirs(cfg['paths']['results'], exist_ok=True)
    with open(RG.registry_path(cfg), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=RG.FIELDS)
        w.writeheader()
        w.writerow({'hypothesis': 'EURUSD_lightgbm_vs_prior', 'config_digest': 'other', 'feature_list_digest': 'x'})
    with pytest.raises(RG.ConfigChanged):
        LB.write_report(cfg, 'T', frames=make_frames(), points=POINTS)
