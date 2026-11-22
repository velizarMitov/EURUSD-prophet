"""fx-bb-sma-exit, group 3: the label and power report, before any model."""
import copy
import json
import os

import pytest

from src.fx_bb_sma import labels as LB
from src.fx_bb_sma import trades as T
from src.fx_bb_sma.registry import ConfigChanged
from tests.fx_bb_sma_synth import synth_inputs, tmp_cfg_like


@pytest.fixture(scope='module')
def inputs():
    return synth_inputs(600)


def test_thousand_trades_on_750_dates():
    assert LB.mde_mean(1.0, 1000, 0.025, 0.8) == pytest.approx(0.0886, abs=5e-4)
    assert LB.mde_mean(1.0, 750, 0.025, 0.8) == pytest.approx(0.1023, abs=5e-4)


def test_report_succeeds_without_reading_any_result(tmp_path, inputs, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('the label report must not read an exit reason or result')
    monkeypatch.setattr(T, 'results', boom)
    monkeypatch.setattr(T, 'control_r', boom)
    cfg = tmp_cfg_like(tmp_path)
    text = open(LB.write_report(cfg, 'T', inputs), encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Net of the recorded bar spread only')
    for header in ('Win rate', 'Profit factor', 'Exit reason', 'Stop loss', 'Mean net R |', 'pips'):
        assert header not in text, header
    rec = json.load(open(os.path.join(cfg['paths']['results'], 'labels.json'), encoding='utf-8'))
    p = rec['power']['pooled']['cell1']
    assert p['eligible'] > 20 and p['trades'] == p['eligible'] + p['pair gap'] + p['corrupt spread'] + p['open at data end']


def test_a_changed_config_is_refused_after_the_label_report(tmp_path, inputs):
    cfg = tmp_cfg_like(tmp_path)
    LB.write_report(cfg, 'T', inputs)
    other = copy.deepcopy(cfg)
    other['trade']['time_stop_days'] = 15
    with pytest.raises(ConfigChanged, match='labels.json'):
        LB.write_report(other, 'T2', inputs)
