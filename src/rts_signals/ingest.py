"""Ingesting RTS PDFs into stored rows (spec rts-signals/report-ingest; design D3).

    python -m src.rts_signals.ingest          ingest every new PDF in drop/ and inbox/

Sources: the drop folder research_models/rts_signals/drop/ (files saved by hand) and the Gmail fetch
(gmail.py), which both end here. A file is identified by its SHA-256: one already in reports.csv is never
ingested twice. A canonical copy goes to the gitignored inbox/<sha256>.pdf.

Stored rows, append-only, under results/rts_signals/ (extracted numbers and digests only, never report
text, which is paid third-party content):
    reports.csv            one row per distinct file: kind, report date, source, received time, status
    recommendations.csv    the six table rows of each spot report (futures reports are logged, never read)
    track_record.csv       each distinct track-record line (a correction is a new line), first report seen
    morning_positions.csv  the Morning report's positions table (OCR, checked)
    flags.csv              every inconsistency, never silently fixed
The text of unparsed lines goes to the gitignored research_models/rts_signals/unparsed.csv; reports.csv
carries their count.

A second file with the same kind and report date but different bytes is recorded `second version` and
adds no rows: the first stored version stands.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
from datetime import datetime, timezone

from . import art_dir, load_config, results_dir, sha256_bytes, sub
from .parse import UnknownReport, parse_text

REPORTS = 'reports.csv'
RECS = 'recommendations.csv'
TRACK = 'track_record.csv'
MORNING = 'morning_positions.csv'
FLAGS = 'flags.csv'
UNPARSED = 'unparsed.csv'

F_REPORTS = ['sha256', 'kind', 'report_date', 'source', 'file_name', 'received_utc', 'ingested_utc',
             'parser_version', 'status', 'n_rows', 'n_flags', 'n_unparsed']
F_RECS = ['sha256', 'kind', 'report_date', 'pair', 'symbol', 'direction', 'position', 'npr', 'entry_type', 'close',
          'entry', 'stop', 'target', 'close_raw', 'entry_raw', 'stop_raw', 'target_raw', 'pos_stop_raw',
          'pos_target_raw', 'pos_entry_raw', 'pos_entry_date', 'mtm', 'points_line', 'flags', 'parser_version']
F_TRACK = ['kind', 'pair', 'symbol', 'entry_date', 'side', 'entry', 'rule', 'status', 'exit_date', 'exit', 'points',
           'pl', 'flags', 'first_report_date', 'first_sha256', 'parser_version']
TRACK_KEY = ['kind', 'pair', 'entry_date', 'side', 'entry', 'rule', 'status', 'exit_date', 'exit', 'points']
F_MORNING = ['sha256', 'report_date', 'pair', 'spot', 'position', 'signal_date', 'signal_rate', 'gain_pct', 'checked',
             'flags', 'parser_version']
F_FLAGS = ['sha256', 'kind', 'report_date', 'pair', 'value', 'flag']


def rpath(cfg: dict, name: str) -> str:
    return os.path.join(results_dir(cfg), name)


def read_csv(p: str) -> list:
    if not os.path.exists(p):
        return []
    with open(p, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def append_csv(p: str, fields: list, rows: list) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in fields})


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, list):
        return '; '.join(str(x) for x in v)
    return v


def extract_text(data: bytes) -> str:
    from pypdf import PdfReader
    return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(data)).pages)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _track_key(r: dict) -> tuple:
    return tuple(str(_cell(r.get(k))) for k in TRACK_KEY)


def ingest_bytes(cfg: dict, data: bytes, source: str, file_name: str = '', received_utc: str | None = None,
                 now: str | None = None, text: str | None = None, ocr=None) -> dict:
    """Store one PDF's rows. Returns {'sha256', 'status', ...}. `text` and `ocr` are injectable for tests."""
    sha = sha256_bytes(data)
    if any(r['sha256'] == sha for r in read_csv(rpath(cfg, REPORTS))):
        return {'sha256': sha, 'status': 'already stored'}
    now = now or _now()
    pv = cfg['parse']['parser_version']
    inbox = sub(cfg, 'inbox')
    os.makedirs(inbox, exist_ok=True)
    canon = os.path.join(inbox, f'{sha}.pdf')
    if not os.path.exists(canon):
        with open(canon, 'wb') as fh:
            fh.write(data)
    rep = {'sha256': sha, 'source': source, 'file_name': os.path.basename(file_name), 'received_utc': received_utc or now,
           'ingested_utc': now, 'parser_version': pv, 'kind': '', 'report_date': '', 'n_rows': 0, 'n_flags': 0,
           'n_unparsed': 0}
    try:
        text = text if text is not None else extract_text(data)
        from .parse import kind_of
        cells = None
        if kind_of(text) == 'morning':
            from .ocr import read_positions_table
            cells = (ocr or read_positions_table)(data, cfg['parse']['morning_pages'])
        res = parse_text(cfg, text, cells)
    except UnknownReport as e:
        rep['status'] = str(e)
        append_csv(rpath(cfg, REPORTS), F_REPORTS, [rep])
        return {'sha256': sha, 'status': rep['status']}
    rep.update(kind=res['kind'], report_date=res['report_date'])
    if res['kind'] == 'futures':                 # owner decision 2026-11-28: spot only
        rep['status'] = 'futures — not read'
        append_csv(rpath(cfg, REPORTS), F_REPORTS, [rep])
        return {'sha256': sha, 'status': rep['status'], 'kind': 'futures', 'report_date': res['report_date']}
    same = [r for r in read_csv(rpath(cfg, REPORTS)) if r['kind'] == res['kind'] and
            r['report_date'] == res['report_date'] and r['status'] == 'stored']
    if same:
        rep['status'] = 'second version'
        append_csv(rpath(cfg, REPORTS), F_REPORTS, [rep])
        append_csv(rpath(cfg, FLAGS), F_FLAGS, [{'sha256': sha, 'kind': res['kind'], 'report_date': res['report_date'],
                                                 'pair': '', 'value': same[0]['sha256'][:12],
                                                 'flag': 'second version of a stored report; rows not added'}])
        return {'sha256': sha, 'status': rep['status'], 'kind': res['kind'], 'report_date': res['report_date']}
    base = {'sha256': sha, 'kind': res['kind'], 'report_date': res['report_date'], 'parser_version': pv}
    recs = [dict(base, **r) for r in res['recommendations']]
    have = {_track_key(r) for r in read_csv(rpath(cfg, TRACK))}
    track = []
    for t in res['track']:
        row = dict(t, first_report_date=res['report_date'], first_sha256=sha, parser_version=pv)
        k = _track_key(row)
        if k not in have:
            have.add(k)
            track.append(row)
    morning = [dict(base, **m) for m in res['morning']]
    flags = [dict(base, **f) for f in res['flags']]
    append_csv(rpath(cfg, RECS), F_RECS, recs)
    append_csv(rpath(cfg, TRACK), F_TRACK, track)
    append_csv(rpath(cfg, MORNING), F_MORNING, morning)
    append_csv(rpath(cfg, FLAGS), F_FLAGS, flags)
    append_csv(os.path.join(art_dir(cfg), UNPARSED), ['sha256', 'line'],
               [{'sha256': sha, 'line': u} for u in res['unparsed']])
    rep.update(status='stored', n_rows=len(recs) + len(track) + len(morning), n_flags=len(flags),
               n_unparsed=len(res['unparsed']))
    append_csv(rpath(cfg, REPORTS), F_REPORTS, [rep])
    return {'sha256': sha, 'status': 'stored', 'kind': res['kind'], 'report_date': res['report_date'],
            'n_rows': rep['n_rows'], 'n_flags': rep['n_flags'], 'n_unparsed': rep['n_unparsed']}


def ingest_folder(cfg: dict, folder: str, source: str, now: str | None = None, ocr=None) -> list:
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith('.pdf'):
            continue
        with open(os.path.join(folder, name), 'rb') as fh:
            data = fh.read()
        r = ingest_bytes(cfg, data, source, name, now=now, ocr=ocr)
        if r['status'] != 'already stored':
            out.append(dict(r, file=name))
    return out


def ingest_all(cfg: dict | None = None, now: str | None = None, ocr=None) -> list:
    cfg = cfg or load_config()
    return ingest_folder(cfg, sub(cfg, 'drop'), 'drop', now, ocr) + ingest_folder(cfg, sub(cfg, 'inbox'), 'inbox', now, ocr)


def main(argv=None):
    argparse.ArgumentParser(description='ingest new RTS PDFs from the drop folder and the inbox').parse_args(argv)
    for r in ingest_all():
        print({k: r.get(k) for k in ('file', 'status', 'kind', 'report_date', 'n_rows', 'n_flags', 'n_unparsed')})


if __name__ == '__main__':
    main()
