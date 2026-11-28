"""gold-nyrange-forward, group 6: the chart file and the indicator's wording."""
import os
import re

import pandas as pd
import pytest

from src.gold_nyfwd import REPO
from src.gold_nyfwd import export as X
from src.gold_nyfwd import live as LV
from src.gold_nyfwd import prereg as PR
from tests._gold_nyfwd_fakes import FakeFeed, cfg_in, common_files_in_tmp, listed, market, no_gate  # noqa: F401

TRADE_WORDS = re.compile(r'\b(buy|sell|long|short|entry|stop|target|lot|lots|profit|take|order)\b', re.I)
DAY = pd.Timestamp('2026-07-15', tz='UTC')
MQ5 = os.path.join(REPO, 'mql5', 'Indicators', 'ProphetGoldNYRange.mq5')


@pytest.fixture
def cfg(tmp_path):
    c = cfg_in(tmp_path)
    PR.write(c)
    return c


def run_day(cfg, bars):
    LV.run_forecast(cfg, FakeFeed(bars, listed(), DAY + pd.Timedelta(hours=15, minutes=8)), gate=no_gate)
    return LV.run_settle(cfg, FakeFeed(bars, listed(), DAY + pd.Timedelta(hours=22, minutes=8)), gate=no_gate)


def rows(text):
    lines = [l for l in text.splitlines() if not l.startswith('#')]
    head = lines[0].split('\t')
    return [dict(zip(head, l.split('\t'))) for l in lines[1:]]


def test_rows_after_settlement_carry_the_band_and_the_realised_range(cfg, common_files_in_tmp):
    out = run_day(cfg, market())
    assert out['export'] == 'export ok'
    p = X.export_path(cfg)
    assert p.startswith(str(common_files_in_tmp)) and p.endswith(os.path.join('EURUSDProphet', 'gold_ny_range.tsv'))
    text = open(p, encoding='utf-8').read()
    r = rows(text)[-1]
    f = LV.read_log(cfg, 'forecasts').iloc[0]
    s = LV.read_log(cfg, 'settlements').iloc[0]
    assert r['date'] == '2026.07.15' and r['status'] == 'ok'
    assert r['band_start'] == '2026.07.15 15:00' and r['band_end'] == '2026.07.15 22:00'
    assert float(r['band_hi']) - float(r['band_lo']) == pytest.approx(f['F'], abs=0.011)
    assert float(r['real_high']) == pytest.approx(s['realised_high'], abs=0.006)
    assert float(r['real_low']) == pytest.approx(s['realised_low'], abs=0.006)
    assert '#text\tstatus\t' in text and '/ 60' in text and 'no verdict yet' in text
    assert 'forward_days=1' in text and '1 / 60' in text


def test_export_failure_reports_and_leaves_the_logs(cfg, monkeypatch):
    def boom(*a, **k):
        raise PermissionError('Common Files folder is not writable')
    monkeypatch.setattr(X, 'write_atomic', boom)
    out = LV.run_forecast(cfg, FakeFeed(market(), listed(), DAY + pd.Timedelta(hours=15, minutes=8)), gate=no_gate)
    assert out['status'] == 'logged' and out['export'].startswith('export failed')
    assert LV.read_log(cfg, 'forecasts').shape[0] == 1


def test_no_trade_word_in_the_chart_file(cfg):
    run_day(cfg, market())
    LV.append_log(cfg, 'forecasts', [{'date': DAY + pd.Timedelta(days=1), 'status': 'no dollar data'}])
    text = X.build(cfg, committed=pd.Timestamp('2026-07-01', tz='UTC'))
    body = '\n'.join(l for l in text.splitlines() if not l.startswith('date\t'))
    assert not TRADE_WORDS.findall(body), TRADE_WORDS.findall(body)


def test_indicator_strings_have_no_trade_word_and_the_source_no_order_function():
    src = open(MQ5, encoding='utf-8').read()
    code = re.sub(r'//[^\n]*', '', src)
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', code)
    hits = [s for s in literals if TRADE_WORDS.search(s)]
    assert not hits, hits
    for fn in ('OrderSend', 'OrderSendAsync', 'PositionOpen', 'PositionClose', 'CTrade', 'OrderModify',
               'TRADE_ACTION', 'Trade.mqh'):
        assert fn not in code, fn
    assert '"GOLD"' in code and 'PERIOD_H1' in code and 'gold_ny_range.tsv' in code
