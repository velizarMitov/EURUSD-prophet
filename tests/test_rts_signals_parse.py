"""rts-signals task 2.1: kind detection, the spot and morning parsers (futures are never read), conversion, flags and the
append-only ingest. Fixtures are trimmed text excerpts and OCR cells, never PDFs."""

import json
import os

import pytest

from src.rts_signals import load_config
from src.rts_signals.ingest import FLAGS, MORNING, RECS, REPORTS, TRACK, ingest_bytes, read_csv, rpath
from src.rts_signals.parse import (UnknownReport, _split_entry, kind_of, morning_rows, parse_text, report_date,
                                   track_rows)

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'rts_signals')
MORNING_TEXT = ('Rockefeller Treasury Services Strategic Currency Briefing Wednesday, November 25, 2026\n'
                'Summary: the dollar index ...\n')


def fx(name):
    with open(os.path.join(FIX, name), encoding='utf-8') as fh:
        return fh.read()


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = dict(c['paths'], results=str(tmp_path / 'results'), artifacts=str(tmp_path / 'art'))
    return c


def recs(text, cfg):
    return {r['pair']: r for r in parse_text(cfg, text)['recommendations']}


def test_kinds_and_dates():
    assert kind_of(fx('spot_112526.txt')) == 'spot' and report_date(fx('spot_112526.txt'), 'spot') == '2026-11-25'
    assert kind_of(fx('futures_header.txt')) == 'futures'
    assert kind_of(MORNING_TEXT) == 'morning' and report_date(MORNING_TEXT, 'morning') == '2026-11-25'
    with pytest.raises(UnknownReport):
        kind_of('Some newsletter\n')


def test_yen_row_converts():
    r = recs(fx('spot_112526.txt'), load_config())['YEN']
    assert (r['symbol'], r['direction'], r['entry'], r['stop'], r['target']) == ('USDJPY', 'BUY', 158.98, 158.58, 159.42)
    assert r['entry_type'] == 'guerilla'


def test_swiss_and_cad_convert():
    r = recs(fx('spot_112326.txt'), load_config())
    assert (r['SWISS']['symbol'], r['SWISS']['close'], r['SWISS']['entry_type']) == ('USDCHF', 0.8415, 'rule0')
    assert (r['CAD']['symbol'], r['CAD']['close'], r['CAD']['stop'], r['CAD']['target']) == ('USDCAD', 1.4374, 1.4339, 1.4402)


def test_euro_guerilla():
    r = recs(fx('spot_112526.txt'), load_config())['EURO']
    assert r['entry_type'] == 'guerilla' and r['entry'] == 1.1009 and r['close'] == 1.0975
    assert (r['position'], r['pos_entry_raw'], r['pos_stop_raw']) == ('SHORT', 10992, 10975)


def test_npr_rows_have_no_levels():
    rs = recs(fx('spot_112726.txt'), load_config())
    assert len(rs) == 6 and all(r['npr'] for r in rs.values())
    assert all(r['entry'] is None and r['stop'] is None and r['target'] is None for r in rs.values())


def test_merged_entry_and_rule_split():
    cfg = load_config()
    rows, un = track_rows(cfg, 'spot', ['11/03/26 USD/CHF Sold 0.81881 11/03/26 0.8152 36 43.97 164.10'])
    assert not un
    r = rows[0]
    assert (r['symbol'], r['side'], r['entry'], r['rule'], r['exit'], r['points']) == ('USDCHF', 'sell', 0.8188, 1, 0.8152, 36)
    assert not r['flags']
    assert _split_entry('1.3382', '0', 4) == ('1.3382', 0)


def test_points_correction_is_flagged():
    rows, _ = track_rows(load_config(), 'spot', ['11/23/26 USD/CAD Bought 1.4340 0 11/23/26 1.4382 -42 -29.29 204.91'])
    assert any('points differ' in f for f in rows[0]['flags'])


def test_cad_narrative_typo_flagged_no_trade():
    res = parse_text(load_config(), fx('spot_112526.txt'))
    f = [x for x in res['flags'] if x['flag'].startswith('narrative price off by more than 500 points')]
    assert [(x['pair'], x['value']) for x in f] == [('CAD', 13202)]
    assert all(r['entry_raw'] != 13202 for r in res['recommendations'])


def test_futures_report_adds_no_row(cfg):
    res = parse_text(load_config(), fx('futures_header.txt'))
    assert (res['kind'], res['report_date']) == ('futures', '2026-11-24')
    assert res['recommendations'] == res['track'] == res['flags'] == []
    r = ingest_bytes(cfg, b'futures-pdf', 'drop', 'f.pdf', text=fx('futures_header.txt'))
    assert r['status'] == 'futures — not read'
    assert read_csv(rpath(cfg, RECS)) == [] and read_csv(rpath(cfg, TRACK)) == []
    assert ingest_bytes(cfg, b'futures-pdf', 'drop', 'f.pdf', text=fx('futures_header.txt'))['status'] == 'already stored'


def test_morning_rows_checked():
    cells = json.loads(fx('morning_112526_cells.json'))
    rows = {r['pair']: r for r in morning_rows(load_config(), cells)}
    m = rows['USDMXN']
    assert (m['spot'], m['position'], m['signal_date'], m['signal_rate'], m['gain_pct']) == \
        (19.6834, 'LONG USD', '2026-09-20', 17.196, 14.46)
    assert all(r['checked'] for r in rows.values()) and len(rows) == 11
    bad = morning_rows(load_config(), [['USD/MXN', '19.6834', 'LONG USD', '09/20/26', '17.1960', '41.46%']])
    assert not bad[0]['checked'] and 'gain does not match' in bad[0]['flags'][0]


def test_ingest_stores_once_and_reparse_adds_nothing(cfg):
    spot = fx('spot_112526.txt')
    r1 = ingest_bytes(cfg, b'pdf-spot-1125', 'drop', 'a.pdf', text=spot, now='2026-11-28T10:00:00+00:00')
    assert r1['status'] == 'stored'
    snap = {n: open(rpath(cfg, n), encoding='utf-8').read() for n in (REPORTS, RECS, TRACK, FLAGS)}
    assert ingest_bytes(cfg, b'pdf-spot-1125', 'gmail', 'b.pdf', text=spot)['status'] == 'already stored'
    assert {n: open(rpath(cfg, n), encoding='utf-8').read() for n in snap} == snap
    assert os.path.exists(os.path.join(cfg['paths']['artifacts'], 'inbox', f"{r1['sha256']}.pdf"))
    assert len(read_csv(rpath(cfg, RECS))) == 6


def test_duplicate_morning_stored_once(cfg):
    cells = json.loads(fx('morning_112526_cells.json'))
    a = ingest_bytes(cfg, b'same-morning-bytes', 'drop', '112326 Morning Report ALL.pdf', text=MORNING_TEXT,
                     ocr=lambda d, p: cells)
    b = ingest_bytes(cfg, b'same-morning-bytes', 'drop', '112326 Morning Report ALL (1).pdf', text=MORNING_TEXT,
                     ocr=lambda d, p: cells)
    assert (a['status'], a['kind'], b['status']) == ('stored', 'morning', 'already stored')
    assert len(read_csv(rpath(cfg, REPORTS))) == 1 and len(read_csv(rpath(cfg, MORNING))) == 11
    c = ingest_bytes(cfg, b'other-bytes', 'drop', 'x.pdf', text=MORNING_TEXT, ocr=lambda d, p: cells)
    assert c['status'] == 'second version' and len(read_csv(rpath(cfg, MORNING))) == 11


def test_track_record_lines_deduplicated_across_reports(cfg):
    ingest_bytes(cfg, b'a', 'drop', text=fx('spot_112526.txt'))
    n = len(read_csv(rpath(cfg, TRACK)))
    ingest_bytes(cfg, b'b', 'drop', text=fx('spot_112726.txt'))
    tr = read_csv(rpath(cfg, TRACK))
    assert len(tr) > n
    keys = [(t['pair'], t['entry_date'], t['side'], t['entry'], t['rule'], t['status'], t['exit']) for t in tr]
    assert len(keys) == len(set(keys))


def test_ocr_position_without_space():
    r = morning_rows(load_config(), [['EUR/GBP', '0.8362', 'SHORTEUR', '10/05/26', '0.8473', '1.31%']])[0]
    assert r['checked'] and r['position'] == 'SHORT EUR'
