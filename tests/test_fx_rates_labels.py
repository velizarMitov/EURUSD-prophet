"""fx-rates-2y, group 4: the 5-day target, the label report and the power table."""
import copy
import csv
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_rates import labeling as LB
from src.fx_rates import load_config
from src.fx_rates import registry as RG
from tests.test_fx_rates_features import synth_d1, synth_yields

CFG = load_config()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = dict(c['paths'], data=str(tmp_path / 'd'), results=str(tmp_path / 'r'), artifacts=str(tmp_path / 'a'))
    c['folds']['first_eval_start'] = '2014-01-01'
    return c


def test_up_week_is_one_down_week_is_zero_and_ties_are_dropped():
    days = pd.bdate_range('2024-01-01', periods=12)
    c = np.array([1.08, 1.0805, 1.07, 1.09, 1.10, 1.11, 1.0850, 1.06, 1.07, 1.08, 1.12, 1.13])
    t = LB.build_target(pd.DataFrame({'mid_close': c}, index=days), 5)
    assert t['y'].iloc[0] == 1.0 and t['exit_mid'].iloc[0] == 1.11            # 1.08 -> 1.11
    assert t['y'].iloc[1] == 1.0 and t['y'].iloc[3] == 0.0                     # 1.09 -> 1.06
    flat = LB.build_target(pd.DataFrame({'mid_close': [1.1] * 7}, index=days[:7]), 5)
    assert flat['tie'].iloc[0] and np.isnan(flat['y'].iloc[0])


def test_no_target_looks_past_t_plus_five():
    d1 = synth_d1(4)
    t = LB.build_target(d1, 5)
    assert t['y'].iloc[-5:].isna().all() and t['y'].iloc[:-5].notna().all()
    assert (t['exit_date'].iloc[:-5].to_numpy() == d1.index[5:].to_numpy()).all()
    changed = d1.copy()
    changed.iloc[-1] = 99.0                                                    # only affects t = last - 5
    t2 = LB.build_target(changed, 5)
    assert t2['y'].iloc[:-6].equals(t['y'].iloc[:-6])


def test_power_for_the_probe_sized_sample():
    r = LB.power_rows(3080, 3.7, 5, 0.0125, 0.8)
    assert r['mde_eff'] == pytest.approx(0.553, abs=0.003) and r['mde_weeks'] == pytest.approx(0.562, abs=0.003)


def test_overlapping_signs_have_vif_near_the_theory():
    rng = np.random.default_rng(0)
    c = pd.Series(np.cumsum(rng.normal(size=20000)))
    y = (c.shift(-5) > c).astype(float)[:-5]
    vif, rho = LB.sign_vif(y, 4)
    assert rho[0] == pytest.approx(0.59, abs=0.03) and vif == pytest.approx(3.78, abs=0.2)


def test_label_report_sections_and_no_model_output(cfg):
    d1 = {'EURUSD': synth_d1(1), 'GBPUSD': synth_d1(2)}
    path = LB.write_report(cfg, 'T', d1=d1, yields=synth_yields())
    text = open(path, encoding='utf-8').read()
    assert 'not a trading result' in text.splitlines()[2]
    for s in ('## Sources', '## Days', '## Up-rate per year', '## Excluded days', '## Power', 'Effective n',
              'MDE on weeks', 'Stale-filled usable days', 'warm-up'):
        assert s in text
    assert os.listdir(cfg['paths']['results']) == ['labels.md']


def test_label_stage_refuses_a_config_that_differs_from_the_verdicts(cfg):
    os.makedirs(cfg['paths']['results'], exist_ok=True)
    with open(RG.registry_path(cfg), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=RG.FIELDS)
        w.writeheader()
        w.writerow({'hypothesis': 'EURUSD_lightgbm_vs_prior', 'config_digest': 'other', 'feature_list_digest': 'x'})
    with pytest.raises(RG.ConfigChanged):
        LB.write_report(cfg, 'T', d1={'EURUSD': synth_d1(1), 'GBPUSD': synth_d1(2)}, yields=synth_yields())
