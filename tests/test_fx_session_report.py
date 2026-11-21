"""fx-session-ny, group 7: the walk-forward report."""
import numpy as np
import pandas as pd

from src.fx_session import report as RP
from src.fx_session import score as SC
from tests.test_fx_session_walkforward import cfg, synth_oos  # noqa: F401  (fixture)


def test_report_is_reproducible_and_carries_header_verdicts_and_sections(cfg):
    synth_oos(cfg, 0.1, 0.0)
    SC.score_lgbm(cfg, 'T')
    a = open(RP.build(cfg, 'A'), encoding='utf-8').read()
    b = open(RP.build(cfg, 'B'), encoding='utf-8').read()
    assert a.replace('_Generated A._', '') == b.replace('_Generated B._', '')
    assert a.splitlines()[2] == f'> **{RP.GROSS_HEADER}**'
    assert '| EURUSD_lightgbm_vs_prior |' in a and '**KEEP**' in a and '**DROP**' in a
    for s in ('## Power', '## Log-loss per fold', '## Reliability', '## Bid-versus-mid parity',
              '## Sign rule — gross of spread and slippage', 'Wilson 95 %', 'DST-mismatch days'):
        assert s in a


def test_report_before_scoring_says_not_scored(cfg):
    synth_oos(cfg)
    assert 'Not scored yet' in open(RP.build(cfg, 'T'), encoding='utf-8').read()


def test_parity_flag_triggers_above_one_point():
    y = np.r_[np.ones(510), np.zeros(490)]
    y_bid = y.copy()
    y_bid[:14] = 0                                 # 51.0 % on mid, 49.6 % on bid
    pr = RP.parity(pd.DataFrame({'y': y, 'y_bid': y_bid}), 1.0)
    assert pr['gap'] == 1.4000000000000012 or abs(pr['gap'] - 1.4) < 1e-9
    assert pr['flag']
    assert not RP.parity(pd.DataFrame({'y': y, 'y_bid': y}), 1.0)['flag']


def test_wilson_interval_and_sign_rule():
    lo, hi = RP.wilson(53, 100)
    assert 0.43 < lo < 0.53 < hi < 0.63
    d = pd.DataFrame({'entry_mid': [1.1, 1.1], 'exit_mid': [1.101, 1.099]})
    s = RP.sign_rule(d, np.array([0.6, 0.4]))
    assert s['win'] == 1.0 and abs(s['total_pips'] - 20) < 1e-6 and s['pf'] == float('inf')
