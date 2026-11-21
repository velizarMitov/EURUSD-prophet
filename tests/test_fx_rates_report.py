"""fx-rates-2y, group 7: the walk-forward report."""
import numpy as np
import pandas as pd
import pytest

from src.fx_rates import report as RP
from src.fx_rates import score as SC
from tests.test_fx_rates_walkforward import cfg, synth_oos  # noqa: F401  (fixture)


def test_report_is_reproducible_and_carries_header_verdicts_and_sections(cfg):
    synth_oos(cfg, 0.1, 0.0)
    SC.score_lgbm(cfg, 'T')
    a = open(RP.build(cfg, 'A'), encoding='utf-8').read()
    b = open(RP.build(cfg, 'B'), encoding='utf-8').read()
    assert a.replace('_Generated A._', '') == b.replace('_Generated B._', '')
    assert a.splitlines()[2] == f'> **{RP.GROSS_HEADER}**' and '**KEEP**' in a and '**DROP**' in a
    for s in ('## Power', 'Effective n', '## Log-loss per fold', '## Reliability', 'stale-yield days',
              '## Weekly rule — gross of spread and slippage', 'mean 0–4', 'always long'):
        assert s in a


def test_report_before_scoring_says_not_scored(cfg):
    synth_oos(cfg)
    assert 'Not scored yet' in open(RP.build(cfg, 'T'), encoding='utf-8').read()


def test_weekly_rule_takes_every_fifth_day_and_counts_pips():
    n = 12
    d = pd.DataFrame({'entry_mid': np.full(n, 1.10), 'exit_mid': 1.10 + 0.001 * np.arange(n)})
    p = np.where(np.arange(n) % 2 == 0, 0.6, 0.4)
    r0 = RP.weekly_rule(d, p, 0, 5, 0.0001)            # days 0, 5, 10
    assert r0['weeks'] == 3
    assert r0['total_pips'] == pytest.approx(0 - 50 + 100)     # long 0, short 5 (-50), long 10 (+100)
    r1 = RP.weekly_rule(d, p, 1, 5, 0.0001)            # days 1, 6, 11: short 10, long 60, short 110
    assert r1['weeks'] == 3 and r1['total_pips'] == pytest.approx(-10 + 60 - 110)
