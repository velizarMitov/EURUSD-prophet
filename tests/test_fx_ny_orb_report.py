"""fx-ny-orb-hot, group 5.2: the report."""
import copy

import pandas as pd

from src.fx_ny_orb import labels as LB
from src.fx_ny_orb import load_config
from src.fx_ny_orb import prereg as PR
from src.fx_ny_orb import report as RP
from src.fx_ny_orb import score as SC
from tests.test_fx_ny_orb_prereg import synth

CFG = load_config()


def test_nothing_before_scoring_and_two_identical_builds(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['results'] = str(tmp_path / 'results')
    cfg['data']['window_start'], cfg['data']['window_end'] = '2019-01-02', '2020-02-28'
    cfg['scoring']['resamples'] = 500
    data = synth(250)
    LB.write_report(cfg, 'T', data)
    text = open(RP.build(cfg, 'T0'), encoding='utf-8').read()
    assert 'Not scored yet' in text and '## Trades' not in text
    PR.write(cfg)
    SC.score_historical(cfg, 'T1', data, gate=lambda c: 'x')
    a = open(RP.build(cfg, 'first'), encoding='utf-8').read().splitlines()
    b = open(RP.build(cfg, 'second'), encoding='utf-8').read().splitlines()
    assert [(x, y) for x, y in zip(a, b) if x != y] == [('_Generated first._', '_Generated second._')]
    text = '\n'.join(a)
    for needle in ('orb_hot_net_r', 'orb_hot_minus_cold', 'hot, pooled', 'non-hot, pooled', 'Five example',
                   'DST-mismatch', 'Conflicts'):
        assert needle in text, needle


def test_examples_follow_the_fixed_rule():
    d = pd.DataFrame({'date': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08',
                                              '2020-01-09']),
                      'pair': 'EURUSD', 'direction': [1, -1, 1, 1, -1, 1],
                      'exit_reason': ['take profit', 'stop loss', 'forced exit', 'take profit', 'stop loss', 'stop loss']})
    ex = RP.examples(d)
    assert len(ex) == 5 and set(ex['exit_reason']) == set(RP.REASONS) and set(ex['direction']) == {1, -1}
    assert ex.iloc[0]['date'] == pd.Timestamp('2020-01-07')
