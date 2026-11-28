"""rts-signals task 6.1: the chart file."""

import os

import pandas as pd
import pytest

from src.rts_signals import load_config
from src.rts_signals.export import FIELDS, build, safe_rebuild
from src.rts_signals.ingest import RECS, ingest_bytes, read_csv, rpath

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'rts_signals')


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = dict(c['paths'], results=str(tmp_path / 'results'), artifacts=str(tmp_path / 'art'))
    for name in ('spot_112326.txt', 'spot_112526.txt'):
        with open(os.path.join(FIX, name), encoding='utf-8') as fh:
            ingest_bytes(c, name.encode(), 'drop', text=fh.read())
    return c


def rows(text):
    lines = [l for l in text.splitlines() if not l.startswith('#')]
    hdr = lines[0].split('\t')
    assert hdr == FIELDS
    return [dict(zip(hdr, l.split('\t'))) for l in lines[1:]], text


def recs(cfg):
    d = pd.DataFrame(read_csv(rpath(cfg, RECS)))
    return d[d['kind'] == 'spot']


def test_today_rows_for_the_latest_report(cfg):
    before_open = pd.Timestamp('2026-11-25 23:30', tz='UTC')            # label clock, before the session
    r, text = rows(build(cfg, recs(cfg), pd.DataFrame(), pd.DataFrame(), 40, before_open))
    jpy = next(x for x in r if x['symbol'] == 'USDJPY' and x['kind'] == 'today')
    assert (jpy['direction'], jpy['entry'], jpy['stop'], jpy['target']) == ('BUY', '158.98', '158.58', '159.42')
    assert jpy['start'] == '2026.11.26 00:00' and jpy['end'] == '2026.11.26 22:00'
    eur = next(x for x in r if x['symbol'] == 'EURUSD' and x['kind'] == 'today')
    assert eur['label_bg'] == 'RTS: SELL 1.1009 (партизански) · стоп 1.1044 · цел 1.0968 · непроверено'
    assert all(x['label_bg'].startswith('RTS:') and x['label_en'].startswith('RTS:') for x in r)
    assert all(x['label_bg'].endswith('непроверено') for x in r if x['kind'] == 'today')
    assert '40 / 300 · няма присъда' in text
    assert sum(x['kind'] == 'trade' for x in r) == 6                    # the 11/23 report's rows are past
    assert 'lot' not in text.lower() and 'лот' not in text.lower()


def test_after_the_session_rows_are_past_trades(cfg):
    after = pd.Timestamp('2026-11-27 12:00', tz='UTC')
    r, _ = rows(build(cfg, recs(cfg), pd.DataFrame(), pd.DataFrame(), 0, after))
    assert not any(x['kind'] == 'today' for x in r)


def test_failed_write_reports_export_failed(cfg, tmp_path):
    blocker = tmp_path / 'not_a_folder'
    blocker.write_text('x')
    assert safe_rebuild(cfg, env={'MT5_COMMON_FILES': str(blocker)}).startswith('export failed')
    assert len(read_csv(rpath(cfg, RECS))) == 12                         # no stored row changed


def test_rebuild_writes_the_file(cfg, tmp_path):
    common = tmp_path / 'common'
    assert safe_rebuild(cfg, env={'MT5_COMMON_FILES': str(common)},
                        now_label=pd.Timestamp('2026-11-25 23:30', tz='UTC')) == 'exported'
    assert (common / 'EURUSDProphet' / 'rts_signals.tsv').exists()
