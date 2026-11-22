"""fx-bb-reversion, group 7.1: the descriptive report."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import labels as LB
from src.fx_bb import load_config
from src.fx_bb import report as RP
from src.fx_bb import score as SC
from src.fx_bb import walk_forward as WF
from tests.test_fx_bb_walkforward import synthetic_d1

CFG = load_config()


def test_pips_for_usdjpy_and_eurusd_on_a_toy_event_set():
    d = pd.DataFrame({'pair': ['USDJPY', 'EURUSD', 'EURUSD'], 'dir': [-1, 1, -1],
                      'close': [150.00, 1.1000, 1.1000], 'close_t3': [149.55, 1.1012, 1.1005],
                      'r': [0.5, 0.2, -0.1], 'date': pd.to_datetime(['2020-01-06', '2020-01-06', '2020-01-08']),
                      'spread_pips': [1.0, 0.5, 0.5]})
    np.testing.assert_allclose(RP.pips(d, CFG), [45.0, 12.0, -5.0])
    s = RP.stats(d, CFG)
    assert s['events'] == 3 and s['dates'] == 2 and s['total_pips'] == pytest.approx(52.0)
    assert s['pf'] == pytest.approx(57.0 / 5.0) and s['total_net'] == pytest.approx(50.0)
    assert s['pf_net'] == pytest.approx((44.0 + 11.5) / 5.5)


def test_report_before_scoring_shows_no_outcome(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths'] = {**cfg['paths'], 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    text = open(RP.build(cfg, 'T'), encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Gross of spread and slippage')
    assert 'Not scored yet' in text and 'Not shown until Cell 1 is scored.' in text


def test_two_builds_differ_only_in_the_generation_line(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths'] = {**cfg['paths'], 'results': str(tmp_path / 'r'), 'artifacts': str(tmp_path / 'a')}
    cfg['data']['start']['USDCHF'] = '2012-01-01'
    cfg['scoring']['resamples'] = 1000
    d1 = synthetic_d1()
    LB.write_report(cfg, 'T0', d1)
    WF.run(cfg, 'T1', d1, log=lambda *a: None)
    SC.score_rule(cfg, 'T2', d1)
    SC.score_model(cfg, 'T3')
    a = open(RP.build(cfg, 'first'), encoding='utf-8').read().splitlines()
    b = open(RP.build(cfg, 'second'), encoding='utf-8').read().splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and diff == [('_Generated first._', '_Generated second._')]
    text = '\n'.join(a)
    for needle in ('Mid versus bid', 'Reliability', 'P(win) > 0.5', 'upper-band breach', 'Mean pips after spread',
                   'bb_rule_mean_reversion', 'bb_lightgbm_vs_base_rate', 'USDJPY'):
        assert needle in text
