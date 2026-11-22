"""fx-bb-reversion, group 3.2: the label and power report, before any model."""
import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import events as E
from src.fx_bb import labels as LB
from src.fx_bb import load_config
from src.fx_bb.registry import ConfigChanged
from tests.test_fx_bb_events import d1_from_closes

CFG = load_config()


def synthetic_d1(n=600, start='2013-06-03', seed=0):
    out = {}
    for i, p in enumerate(CFG['pairs']):
        rng = np.random.default_rng(seed + i)
        out[p] = d1_from_closes(1.1 * np.exp(np.cumsum(rng.normal(0, 0.006, n))), start=start)
    return out


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {**c['paths'], 'results': str(tmp_path / 'results')}
    return c


def test_probe_sized_inputs_give_the_registered_detectable_accuracy():
    assert LB.mde_accuracy(1054, 0.025, 0.8) == pytest.approx(0.5431, abs=5e-4)
    assert LB.mde_accuracy(751, 0.025, 0.8) == pytest.approx(0.5511, abs=5e-4)
    assert LB.mde_mean(1.2, 1260, 0.025, 0.8) == pytest.approx(2.8016 * 1.2 / np.sqrt(1260), rel=1e-4)


def test_report_reads_no_outcome_and_succeeds_without_predictions(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('the label report must not read an event outcome')
    monkeypatch.setattr(E, 'outcomes', boom)
    path = LB.write_report(cfg, 'T', synthetic_d1())
    text = open(path, encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Gross of spread and slippage')
    assert '## Power — before any model' in text and 'Cell 2' in text and '_Generated T._' in text
    assert not os.path.exists(os.path.join(cfg['paths']['results'], 'oos_events.csv.gz'))
    rec = json.load(open(os.path.join(cfg['paths']['results'], 'labels.json'), encoding='utf-8'))
    assert rec['power']['pooled']['cell1']['eligible'] > 0


def test_counts_add_up_per_pair_and_pooled(cfg):
    d1 = synthetic_d1()
    data = LB.label_data(d1, cfg)
    pw = LB.power(data, cfg)
    pooled = pw['pooled']['cell1']
    assert pooled['events'] == sum(pw[p]['cell1']['events'] for p in CFG['pairs']) == len(data['events'])
    assert pooled['events'] == pooled['eligible'] + pooled['pair_gap'] + pooled['no_outcome']
    assert pooled['dates'] <= pooled['eligible']
    assert pw['pooled']['cell2']['events'] <= pooled['events']
    assert data['sd_all']['EURUSD'] > 0


def test_a_changed_config_is_refused_after_the_label_report(cfg):
    d1 = synthetic_d1()
    LB.write_report(cfg, 'T', d1)
    LB.write_report(cfg, 'T2', d1)                       # same config: a rerun is fine
    other = copy.deepcopy(cfg)
    other['bands']['width'] = 2.5
    with pytest.raises(ConfigChanged, match='labels.json'):
        LB.write_report(other, 'T3', d1)


def test_dispersion_is_over_all_valid_days_not_events(cfg):
    d1 = synthetic_d1()
    v = d1['EURUSD']
    ind = E.indicators(v, cfg)
    x = ((ind['close'].shift(-3) - ind['close']) / ind['atr']).dropna()
    assert E.dispersion(d1, cfg)['EURUSD'] == pytest.approx(float(x.std(ddof=1)))
    assert len(x) > 10 * len(E.detect(v, 'EURUSD', cfg)[0])
    assert isinstance(pd.Timestamp(cfg['folds']['first_eval_start']), pd.Timestamp)
