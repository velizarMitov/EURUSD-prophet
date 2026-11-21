"""fx-rates-2y, group 2: yield sources, caches and the causal alignment."""
import copy
import io
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_rates import load_config
from src.fx_rates import rates_loader as RL

CFG = load_config()


# -- fixture builders -------------------------------------------------------------------------------

def _cell_ref(c, r):
    s = ''
    c += 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(65 + rem) + s
    return f'{s}{r}'


def make_xlsx(sheets: dict) -> bytes:
    """Minimal workbook: {sheet name: [[cell, ...], ...]}; str cells go to sharedStrings."""
    shared, sheet_xml = [], []
    for rows in sheets.values():
        out = []
        for i, row in enumerate(rows, start=1):
            cells = []
            for j, v in enumerate(row):
                if v is None:
                    continue
                ref = _cell_ref(j, i)
                if isinstance(v, str):
                    shared.append(v)
                    cells.append(f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>')
                else:
                    cells.append(f'<c r="{ref}"><v>{v}</v></c>')
            out.append(f'<row r="{i}">{"".join(cells)}</row>')
        sheet_xml.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                         f'<sheetData>{"".join(out)}</sheetData></worksheet>')
    wb = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
          + ''.join(f'<sheet name="{n}" sheetId="{k + 1}" r:id="rId{k + 1}"/>' for k, n in enumerate(sheets))
          + '</sheets></workbook>')
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(f'<Relationship Id="rId{k + 1}" Target="worksheets/sheet{k + 1}.xml"/>' for k in range(len(sheets)))
            + '</Relationships>')
    sst = ('<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
           + ''.join(f'<si><t>{s}</t></si>' for s in shared) + '</sst>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('xl/workbook.xml', wb)
        z.writestr('xl/_rels/workbook.xml.rels', rels)
        z.writestr('xl/sharedStrings.xml', sst)
        for k, x in enumerate(sheet_xml):
            z.writestr(f'xl/worksheets/sheet{k + 1}.xml', x)
    return buf.getvalue()


def excel_serial(d):
    return (pd.Timestamp(d) - pd.Timestamp('1899-12-30')).days


def boe_book(days, two_year_values):
    """A GLC-like book: a short-end sheet whose 24-month column holds decoys, and the spot curve
    with maturities 0.5, 1, 1.5, 2, 2.5."""
    short = [[None, 'UK nominal spot curve, short end'], ['Maturity '], ['months:', '23', '24', '25'],
             ['years:', 1.9166, 2.0, 2.0833]] + [[excel_serial(d), 9.9, 9.9, 9.9] for d in days]
    spot = [[None, 'UK nominal spot curve'], ['Maturity '], ['years:', 0.5, 1.0, 1.5, 2.0, 2.5]] + \
           [[excel_serial(d), 1.0, 1.1, 1.2, v, 1.3] for d, v in zip(days, two_year_values)]
    return make_xlsx({'3. spot, short end': short, '4. spot curve': spot})


def boe_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name, blob in files.items():
            z.writestr(name, blob)
    return buf.getvalue()


FRED = b'observation_date,DGS2\n2024-01-02,4.33\n2024-01-03,4.33\n2024-01-04,.\n2024-01-05,4.39\n'
BBK = ('﻿"",BBSIS.D...,BBSIS.D..._FLAGS\n"",Term structure ... 2.0 years,\nComment (in english),,\n'
       'Decimals,2,\nlast update,2024-01-06 12:00:00,\n2024-01-02,2.41,\n2024-01-03,2.38,\n'
       '2024-01-06,.,No value available\n2024-01-05,2.44,\n').encode('utf-8')
ARCH_DAYS = pd.bdate_range('2023-12-01', '2023-12-29')
CUR_DAYS = pd.bdate_range('2024-01-02', '2024-01-05')


def fake_get(fail=()):
    archive = boe_zip({'GLC Nominal daily data_2016 to 2023.xlsx': boe_book(ARCH_DAYS, np.linspace(4.0, 4.2, len(ARCH_DAYS))),
                       'GLC Nominal daily data_1990 to 1994.xlsx': b'not read'})
    current = boe_zip({'GLC Nominal daily data current month.xlsx': boe_book(CUR_DAYS, [4.1, 4.15, 4.2, 4.25]),
                       'GLC Real daily data current month.xlsx': b'not read'})

    def get(url):
        for key in fail:
            if key in url:
                raise OSError(f'{key} down')
        if 'fred' in url:
            return FRED.replace(b'DGS2', b'DGS10') if 'DGS10' in url else FRED
        if 'bundesbank' in url:
            return BBK
        return current if 'latest' in url else archive
    return get


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = dict(c['paths'], data=str(tmp_path / 'd'), results=str(tmp_path / 'r'), artifacts=str(tmp_path / 'a'))
    c['data']['yield_start'] = '2020-01-01'
    return c


# -- 2.1 parsing and caches ------------------------------------------------------------------------

def test_parsers_read_fred_and_bundesbank_and_skip_missing():
    f = RL.parse_fred(FRED)
    assert list(f.index.strftime('%Y-%m-%d')) == ['2024-01-02', '2024-01-03', '2024-01-05'] and f.iloc[-1] == 4.39
    b = RL.parse_bundesbank(BBK)
    assert list(b.index.strftime('%Y-%m-%d')) == ['2024-01-02', '2024-01-03', '2024-01-05'] and b.iloc[0] == 2.41


def test_boe_reader_takes_the_spot_curve_2y_not_the_short_end():
    s = RL.parse_boe_xlsx(boe_book(CUR_DAYS, [4.1, 4.15, 4.2, 4.25]))
    assert list(s.to_numpy()) == [4.1, 4.15, 4.2, 4.25] and s.index[0] == pd.Timestamp('2024-01-02')
    with pytest.raises(RL.SourceFormatError):
        RL.parse_boe_xlsx(make_xlsx({'3. spot, short end': [['years:', 2.0, 2.5]]}))
    with pytest.raises(RL.SourceFormatError, match='2.0 -> 2.5'):
        RL.parse_boe_xlsx(make_xlsx({'4. spot curve': [['years:', 1.0, 3.0]]}))


def test_first_refresh_fills_every_cache_and_uk_joins_archive_and_current(cfg):
    rep = RL.refresh(cfg, fake_get(), 'T')
    assert {k: v['status'] for k, v in rep.items()} == dict.fromkeys(RL.SERIES, 'ok')
    uk = RL.load_series('uk2y', cfg)
    assert uk.index.min() == ARCH_DAYS[0] and uk.index.max() == CUR_DAYS[-1]
    assert uk.loc['2024-01-05'] == 4.25
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert list(log['series']) == list(RL.SERIES)


def test_refresh_appends_and_counts_revisions_without_rewriting(cfg):
    RL.refresh(cfg, fake_get(), 'first')
    before = open(RL.cache_path('us2y', cfg), 'rb').read()
    revised = FRED.replace(b'2024-01-03,4.33', b'2024-01-03,4.30') + b'2024-01-08,4.41\n'
    get = fake_get()
    rep = RL.refresh(cfg, lambda url: revised if 'DGS2' in url else get(url), 'second')
    assert rep['us2y']['appended'] == 1 and rep['us2y']['revisions'] == 1
    assert open(RL.cache_path('us2y', cfg), 'rb').read().startswith(before)
    assert RL.load_series('us2y', cfg).loc['2024-01-03'] == 4.33


def test_a_failing_source_leaves_its_cache_and_the_others_refresh(cfg):
    rep = RL.refresh(cfg, fake_get(fail=('bundesbank',)), 'T')
    assert rep['de2y']['status'] == 'failed' and 'down' in rep['de2y']['detail']
    assert not os.path.exists(RL.cache_path('de2y', cfg))
    assert all(rep[s]['status'] == 'ok' for s in ('us2y', 'us10y', 'uk2y'))


def test_later_uk_refresh_reads_only_the_current_month(cfg):
    RL.refresh(cfg, fake_get(), 'first')
    seen = []
    get = fake_get()
    RL.refresh(cfg, lambda url: (seen.append(url), get(url))[1], 'second')
    boe = [u for u in seen if 'bankofengland' in u]
    assert len(boe) == 1 and 'latest' in boe[0]


# -- 2.2 the causal alignment --------------------------------------------------------------------

from src.fx_rates import align as AL  # noqa: E402


def ys(dates, start=1.0):
    d = pd.DatetimeIndex(dates)
    return pd.Series(start + 0.01 * np.arange(len(d)), index=d)


def yields_on(dates):
    return {s: ys(dates, k + 1.0) for k, s in enumerate(('us2y', 'us10y', 'de2y', 'uk2y'))}


FX = pd.bdate_range('2024-03-04', '2024-03-29')


def test_a_same_day_yield_is_never_used():
    y = yields_on(FX)
    a = AL.align_yields(FX, y, CFG)
    t = pd.Timestamp('2024-03-13')
    assert a.loc[t, 'us2y'] == y['us2y'].loc['2024-03-12']          # previous FX day, not t
    assert a.loc[t, 'yield_stale_flag'] == 0 and a.loc[t, 'yield_reason'] == ''
    y2 = {k: v.copy() for k, v in y.items()}
    y2['us2y'].loc['2024-03-13'] = 99.0                              # changing day t's own value
    assert AL.align_yields(FX, y2, CFG).loc[t, 'us2y'] == a.loc[t, 'us2y']


def test_a_us_holiday_fills_from_the_day_before_and_flags():
    y = yields_on(FX)
    y['us2y'] = y['us2y'].drop(pd.Timestamp('2024-03-12'))
    a = AL.align_yields(FX, y, CFG)
    t = pd.Timestamp('2024-03-13')
    assert a.loc[t, 'us2y'] == y['us2y'].loc['2024-03-11'] and a.loc[t, 'us2y_age'] == 1
    assert a.loc[t, 'yield_stale_flag'] == 1 and a.loc[t, 'yield_reason'] == ''


def test_a_five_business_day_gap_excludes_the_day():
    y = yields_on(FX)
    y['uk2y'] = y['uk2y'].drop(pd.bdate_range('2024-03-12', '2024-03-18'))
    a = AL.align_yields(FX, y, CFG)
    assert a.loc[pd.Timestamp('2024-03-19'), 'yield_reason'] == 'stale yields'     # lag 03-18, last obs 03-11
    assert a.loc[pd.Timestamp('2024-03-15'), 'yield_reason'] == ''                 # lag 03-14, age 3: allowed


def test_no_back_fill_before_the_first_observation():
    y = yields_on(pd.bdate_range('2024-03-11', '2024-03-29'))
    a = AL.align_yields(FX, y, CFG)
    assert a.loc[FX[0], 'yield_reason'] == 'no lagged date'
    early = a.loc[:'2024-03-11']
    assert (early['yield_reason'] != '').all() and early['us2y'].isna().all()


def test_one_bar_day_has_no_d1_bar_and_mid_close_uses_the_spread():
    from tests.fx_fib_synth import synthetic_weeks
    h1 = synthetic_weeks(date(2018, 1, 7), 2, seed=3)
    h1['spread'] = 6.0
    d = pd.Timestamp('2018-01-10', tz='UTC')
    h1 = h1[(h1.index.normalize() != d) | (h1.index.hour == 0)]
    c = copy.deepcopy(CFG)
    c['data']['fx_history_start'] = '2018-01-01'
    d1 = AL.d1_from_h1(h1, 1e-5, c)
    assert pd.Timestamp('2018-01-10') not in d1.index and pd.Timestamp('2018-01-09') in d1.index
    last = h1[h1.index.normalize() == pd.Timestamp('2018-01-09', tz='UTC')].iloc[-1]
    assert d1.loc['2018-01-09', 'mid_close'] == pytest.approx(last['close'] + 3e-5)
