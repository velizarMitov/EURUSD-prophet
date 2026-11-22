"""fx-bb-sma-exit, group 6.1: the report and its plain-language section."""
import numpy as np
import pandas as pd
import pytest

from src.fx_bb_sma import labels as LB
from src.fx_bb_sma import load_config
from src.fx_bb_sma import report as RP
from src.fx_bb_sma import score as SC
from src.fx_bb_sma import walk_forward as WF
from tests.fx_bb_sma_synth import synth_inputs, tmp_cfg_like

CFG = load_config()


def toy():
    return pd.DataFrame({
        'pair': ['USDJPY', 'EURUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD'],
        'dir': [-1, 1, 1, -1, 1, -1],
        'date': pd.to_datetime(['2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10', '2020-01-13']),
        'close': [150.40, 1.1000, 1.1000, 1.3000, 0.7000, 1.3000],
        'exit_price': [149.90, 1.0950, 1.1030, 1.2990, 0.6990, 1.3010],
        'exit_reason': ['take profit', 'stop loss', 'time stop', 'take profit', 'stop loss', 'time stop'],
    })


def test_pips_for_usdjpy_and_eurusd():
    np.testing.assert_allclose(RP.pips(toy(), CFG), [50.0, -50.0, 30.0, 10.0, -10.0, -10.0])


def test_exit_shares_add_up_and_profit_factor():
    d = toy()
    d['gross_pips'] = RP.pips(d, CFG)
    d['net_pips'] = d['gross_pips'] - 1.0
    d['r_gross'] = d['gross_pips'] / 50
    d['r_net'] = d['net_pips'] / 50
    s = RP.summary(d)
    assert s['take profit'] + s['stop loss'] + s['time stop'] == pytest.approx(1.0)
    assert s['pf_gross'] == pytest.approx(90 / 70) and s['pf_net'] == pytest.approx(87 / 73)
    assert s['win'] == pytest.approx(3 / 6)


def test_example_rule_picks_every_exit_reason_and_both_directions():
    d = toy()
    ex = RP.examples(d)
    assert len(ex) == 5 and set(ex['exit_reason']) == set(RP.REASONS) and set(ex['dir']) == {1, -1}
    assert ex.iloc[0]['date'] == pd.Timestamp('2020-01-09')        # the latest take profit, never the best one


def test_before_scoring_nothing_is_shown(tmp_path):
    text = open(RP.build(tmp_cfg_like(tmp_path), 'T'), encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Net of the recorded bar spread only')
    assert 'Not scored yet' in text and text.count('Not shown until') == 2


def test_two_builds_differ_only_in_the_generation_line(tmp_path):
    cfg = tmp_cfg_like(tmp_path)
    inputs = synth_inputs(1300)
    LB.write_report(cfg, 'T0', inputs)
    WF.run(cfg, 'T1', inputs, log=lambda *a: None)
    SC.score_rule(cfg, 'T2', inputs)
    SC.score_model(cfg, 'T3')
    a = open(RP.build(cfg, 'first'), encoding='utf-8').read().splitlines()
    b = open(RP.build(cfg, 'second'), encoding='utf-8').read().splitlines()
    assert [(x, y) for x, y in zip(a, b) if x != y] == [('_Generated first._', '_Generated second._')]
    text = '\n'.join(a)
    for needle in ('На прост език', 'Затворени на целта (SMA20)', 'Profit Factor след спреда', 'Пет примерни сделки',
                   'Bid/ask control', 'Reliability', 'P(win) > 0.5', 'Net R by exit reason'):
        assert needle in text, needle
