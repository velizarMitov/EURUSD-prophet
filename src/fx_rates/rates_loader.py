"""Official daily yield caches (spec fx-rates/rates-data; design D3).

    us2y   FRED DGS2              us10y  FRED DGS10
    de2y   Bundesbank Svensson term structure, residual maturity 2.0 years
    uk2y   Bank of England nominal spot curve, 2.0 years (sheet "4. spot curve")

Each cache is results/fx_rates/data/<name>.csv (date,value), append-only: overlapping dates
are compared and revisions counted, never rewritten. Every series fails on its own: a failed
source leaves its cache unchanged and is logged; no other vendor is ever tried. Requests carry
a generic User-Agent ("research data client") and nothing about the user. FRED rejects a fake browser one.

The Bank of England publishes .xlsx files; they are read with zipfile + xml.etree (no openpyxl).
The current-month file wins over the archive where both cover a date.

    python -m src.fx_rates.rates_loader
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import load_config, repo_path

SERIES = ('us2y', 'us10y', 'de2y', 'uk2y')
LOG_FIELDS = ['run_at', 'series', 'status', 'appended', 'revisions', 'rows', 'first', 'last', 'detail']
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R_ID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
REL = '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'
EXCEL_EPOCH = datetime(1899, 12, 30)


class SourceFormatError(ValueError):
    """The source answered, but not in the expected layout. Nothing is guessed."""


def http_get(url: str, cfg: dict) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': cfg['sources']['user_agent']})
    with urllib.request.urlopen(req, timeout=int(cfg['sources']['timeout_s'])) as r:
        return r.read()


# -- parsers ------------------------------------------------------------------------------------

def _series(dates, values) -> pd.Series:
    s = pd.Series(pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)), name='date'))
    s = s.dropna()
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s


def parse_fred(blob: bytes) -> pd.Series:
    d = pd.read_csv(io.BytesIO(blob))
    if d.shape[1] != 2:
        raise SourceFormatError(f'FRED CSV has {d.shape[1]} columns, expected 2')
    return _series(d.iloc[:, 0], d.iloc[:, 1].replace('.', None))


def parse_bundesbank(blob: bytes) -> pd.Series:
    text = blob.decode('utf-8-sig')
    rows = [r for r in csv.reader(io.StringIO(text)) if r and re.fullmatch(r'\d{4}-\d{2}-\d{2}', r[0])]
    if not rows:
        raise SourceFormatError('Bundesbank CSV has no dated rows')
    return _series([r[0] for r in rows], [None if r[1] in ('.', '') else r[1] for r in rows])


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_sheet_rows(blob: bytes, name_contains: str) -> list:
    """Rows of the first sheet whose name contains `name_contains` (case-insensitive), as
    {column index: text}. Shared strings resolved; numbers stay text."""
    x = zipfile.ZipFile(io.BytesIO(blob))
    wb = ET.fromstring(x.read('xl/workbook.xml'))
    rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(x.read('xl/_rels/workbook.xml.rels')).iter(REL)}
    sheets = [(s.get('name'), s.get(R_ID)) for s in wb.find('m:sheets', NS)]
    match = [(n, i) for n, i in sheets if name_contains.lower() in n.lower()]
    if not match:
        raise SourceFormatError(f'no sheet containing "{name_contains}" in {[n for n, _ in sheets]}')
    target = rels[match[0][1]].lstrip('/')
    path = target if target.startswith('xl/') else 'xl/' + target
    shared = []
    if 'xl/sharedStrings.xml' in x.namelist():
        for si in ET.fromstring(x.read('xl/sharedStrings.xml')).findall('m:si', NS):
            shared.append(''.join(t.text or '' for t in si.iter(f"{{{NS['m']}}}t")))
    rows = []
    for row in ET.fromstring(x.read(path)).iter(f"{{{NS['m']}}}row"):
        cells = {}
        for c in row.findall('m:c', NS):
            v = c.find('m:v', NS)
            if v is not None:
                cells[_col(c.get('r'))] = shared[int(v.text)] if c.get('t') == 's' else v.text
        rows.append(cells)
    return rows


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_boe_xlsx(blob: bytes, maturity: float = 2.0) -> pd.Series:
    """The `maturity`-year column of the nominal "spot curve" sheet. The column is located by a
    header row whose cell equals `maturity` and whose next cell equals maturity + 0.5."""
    rows = xlsx_sheet_rows(blob, 'spot curve')
    for h, r in enumerate(rows):
        col = next((c for c, v in r.items() if c > 0 and _num(v) == maturity
                    and _num(r.get(c + 1)) == maturity + 0.5), None)
        if col is not None:
            break
    else:
        raise SourceFormatError(f'no maturity header {maturity} -> {maturity + 0.5} in the spot curve sheet')
    dates, vals = [], []
    for r in rows[h + 1:]:
        d, v = _num(r.get(0)), _num(r.get(col))
        if d is not None and v is not None:
            dates.append(EXCEL_EPOCH + timedelta(days=int(d)))
            vals.append(v)
    return _series(dates, vals)


def parse_boe_zip(blob: bytes, maturity: float = 2.0, from_year: int | None = None) -> pd.Series:
    """Concatenate every nominal daily file in the zip (archive spans ending before `from_year`
    are skipped)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    parts = []
    for fn in sorted(z.namelist()):
        if 'nominal daily' not in fn.lower():
            continue
        m = re.search(r'(\d{4}) to (\d{4}|present)', fn)
        if from_year and m and m.group(2) != 'present' and int(m.group(2)) < from_year:
            continue
        parts.append(parse_boe_xlsx(z.read(fn), maturity))
    if not parts:
        raise SourceFormatError('no nominal daily file in the Bank of England zip')
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep='last')].sort_index()


# -- caches -------------------------------------------------------------------------------------

def cache_path(name: str, cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['data']), f'{name}.csv')


def load_series(name: str, cfg: dict | None = None) -> pd.Series:
    cfg = cfg or load_config()
    d = pd.read_csv(cache_path(name, cfg), parse_dates=['date'])
    return pd.Series(d['value'].to_numpy(float), index=pd.DatetimeIndex(d['date'], name='date'), name=name)


def _merge(name: str, new: pd.Series, cfg: dict) -> dict:
    new = new[new.index >= pd.Timestamp(cfg['data']['yield_start'])]
    if new.empty:
        raise SourceFormatError('the source returned no values in range')
    path = cache_path(name, cfg)
    revisions = 0
    if os.path.exists(path):
        old = load_series(name, cfg)
        shared = old.index.intersection(new.index)
        revisions = int(((old.loc[shared] - new.loc[shared]).abs() > 1e-9).sum())
        add = new[new.index > old.index.max()]
    else:
        add = new
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = pd.DataFrame({'date': add.index.strftime('%Y-%m-%d'), 'value': add.to_numpy()})
    out.to_csv(path, mode='a', header=not os.path.exists(path), index=False, lineterminator='\n')
    full = load_series(name, cfg)
    return {'appended': int(len(add)), 'revisions': revisions, 'rows': int(len(full)),
            'first': str(full.index.min().date()), 'last': str(full.index.max().date())}


def fetch_series(name: str, cfg: dict, get) -> pd.Series:
    src = cfg['sources']
    if name in src['fred_series']:
        return parse_fred(get(src['fred_csv'].format(series=src['fred_series'][name])))
    if name == 'de2y':
        return parse_bundesbank(get(src['bundesbank_csv']))
    if name == 'uk2y':
        mat = float(src['boe_maturity_years'])
        current = parse_boe_zip(get(src['boe_current_zip']), mat)
        path = cache_path(name, cfg)
        need_archive = True
        if os.path.exists(path):
            last = load_series(name, cfg).index.max()
            need_archive = current.index.min() > last + pd.offsets.BDay(1)
        if not need_archive:
            return current
        archive = parse_boe_zip(get(src['boe_archive_zip']), mat, int(src['boe_archive_from_year']))
        both = pd.concat([archive, current])
        return both[~both.index.duplicated(keep='last')].sort_index()
    raise KeyError(name)


def refresh(cfg: dict | None = None, get=None, now_iso: str | None = None) -> dict:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    get = get or (lambda url: http_get(url, cfg))
    report, rows = {}, []
    for name in SERIES:
        row = {'run_at': now_iso, 'series': name}
        try:
            row.update(status='ok', **_merge(name, fetch_series(name, cfg, get), cfg))
        except Exception as e:                      # each series fails on its own
            row.update(status='failed', appended=0, detail=f'{type(e).__name__}: {str(e)[:250]}')
        rows.append(row)
        report[name] = {k: v for k, v in row.items() if k not in ('run_at', 'series')}
    path = os.path.join(repo_path(cfg['paths']['data']), 'refresh_log.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return report


def main(argv=None):
    argparse.ArgumentParser(description='refresh the fx-rates yield caches').parse_args(argv)
    print(json.dumps(refresh(), indent=1))


if __name__ == '__main__':
    main()
