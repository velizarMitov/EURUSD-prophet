"""fx-ny-orb-hot, group 7.1: the chart file and its gate."""
import copy
import os
import re

import pandas as pd
import pytest

from src.fx_ny_orb import export as EX
from src.fx_ny_orb import live as LV
from src.fx_ny_orb import load_config
from src.fx_ny_orb.registry import HIST, HIST_FIELDS, append
from tests.test_fx_ny_orb_live import MODELS, FakeFeed, at

CFG = load_config()
FORBIDDEN = re.compile(r'\b(buy|sell|order\w*|lots?|guarantee\w*|купи|купувай|продай|продавай|поръчк\w*|лот|лота|'
                       r'лотове|гарант\w*)\b|сигурна печалба', re.IGNORECASE)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    monkeypatch.setenv('MT5_COMMON_FILES', str(tmp_path / 'common'))
    monkeypatch.setattr(LV, '_rebuild_export', lambda cc, n: 'skipped')
    return c


def rows(text):
    lines = [l for l in text.splitlines() if l and not l.startswith('#')]
    head = lines[0].split('\t')
    return [dict(zip(head, l.split('\t'))) for l in lines[1:]]


def run_day(cfg, models=MODELS, signal=True):
    LV.run_forecast(cfg, FakeFeed(at('15:05')), models)
    if signal:
        LV.run_signal(cfg, FakeFeed(at('15:52')))


def set_verdict(cfg, verdict):
    from src.fx_ny_orb import config_digest
    append(cfg, HIST, HIST_FIELDS, [{'hypothesis': 'orb_hot_net_r', 'verdict': verdict,
                                     'config_digest': config_digest(cfg)}])


def test_gate_closed_when_not_scored(cfg):
    run_day(cfg)
    r = rows(EX.build(cfg, at('16:00')))
    assert len(r) == 2 and all(x['kind'] == '' and x['entry'] == '' for x in r)
    assert all(x['or_high'] and x['band_lo'] and x['band_hi'] for x in r)


def test_gate_closed_after_a_drop(cfg):
    run_day(cfg)
    set_verdict(cfg, 'DROP')
    assert all(x['kind'] == '' for x in rows(EX.build(cfg, at('16:00'))))


def test_gate_closed_on_a_non_hot_day_even_after_keep(cfg):
    cold = {p: {'coef_harx': [-5.9] + [0.0] * 14, 'coef_har': [-5.9, 0, 0, 0]} for p in cfg['pairs']}
    run_day(cfg, cold)
    set_verdict(cfg, 'KEEP')
    assert all(x['kind'] == '' and x['hot'] == '0' for x in rows(EX.build(cfg, at('16:00'))))


def test_gate_closed_without_a_logged_signal(cfg):
    run_day(cfg, signal=False)
    set_verdict(cfg, 'KEEP')
    assert all(x['kind'] == '' for x in rows(EX.build(cfg, at('16:00'))))


def test_gate_open_shows_the_live_signal(cfg):
    run_day(cfg)
    set_verdict(cfg, 'KEEP')
    text = EX.build(cfg, at('16:00'))
    r = rows(text)
    assert all(x['kind'] == 'live' and x['direction'] == '1' and x['entry'] and x['stop'] and x['target'] for x in r)
    assert all(x['exit'] == '' for x in r)
    assert all('исторически KEEP' in x['label_bg'] and 'forward' in x['label_en'] for x in r)
    assert 'gate=open' in text.splitlines()[0]


def test_past_days_are_simulations_and_times_are_berlin_labels(cfg):
    run_day(cfg)
    LV.run_settle(cfg, FakeFeed(at('22:10')))
    r = rows(EX.build(cfg, at('22:15') + pd.Timedelta(days=1)))
    assert all(x['kind'] == 'sim' and x['label_bg'].startswith('Симулация') and
               x['label_en'].startswith('Simulation') for x in r)
    assert r[0]['or_start'] == '2026.11.04 14:00' and r[0]['band_end'] == '2026.11.04 22:00'
    assert r[0]['line_end'] == '2026.11.04 21:00' and r[0]['entry_time'] == '2026.11.04 15:45'
    assert r[0]['real_high'] and r[0]['real_low']


def test_no_text_is_an_order_lot_or_promise(cfg):
    run_day(cfg)
    set_verdict(cfg, 'KEEP')
    LV.run_settle(cfg, FakeFeed(at('22:10')))
    for text in (EX.build(cfg, at('16:00')), EX.build(cfg, at('23:00'))):
        words = ' '.join(l for l in text.splitlines() if not l.startswith('date\t'))
        assert not FORBIDDEN.search(words), FORBIDDEN.search(words)


def test_atomic_replace_retries_and_never_leaves_a_partial_file(tmp_path):
    target = tmp_path / 'f.tsv'
    target.write_text('old', encoding='utf-8')
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('held by the indicator')
        os.replace(src, dst)
    EX.write_atomic(str(target), 'new', replace=flaky, sleep=lambda s: None)
    assert target.read_text(encoding='utf-8') == 'new' and len(calls) == 3


def test_rebuild_writes_to_the_common_folder_and_nothing_reads_it_back(cfg):
    run_day(cfg)
    p = EX.rebuild(cfg, at('16:00'))
    assert p.endswith(os.path.join('EURUSDProphet', 'ny_orb.tsv')) and os.path.exists(p)
    src = os.path.join(os.path.dirname(EX.__file__))
    readers = [n for n in os.listdir(src) if n.endswith('.py') and n != 'export.py' and
               'export_path(' in open(os.path.join(src, n), encoding='utf-8').read()]
    assert readers == []                      # only export.py ever locates (and only writes) the file
