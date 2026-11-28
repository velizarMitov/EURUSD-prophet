"""Parsing RTS report text into rows (spec rts-signals/report-ingest; design D3, D4).

Kinds, recognised by their title text:
    spot     "Daily Currency Trading Recommendation SPOT Market"   (4 pm New York, the simulated kind)
    morning  "Strategic Currency Briefing"                          (information only, positions table)
    futures  "... CME/Globex"   recognised only so it is not `unknown report`; never read (owner decision
                                2026-11-28: the owner trades spot only and has no futures on MT5)

The parser never invents a trade. Any doubt becomes a flag; a line it cannot read inside the table or the
track record is kept as `unparsed` (its text goes only to the gitignored artifacts folder).

The Morning positions table is an image inside the PDF. Its cells come from `ocr.read_positions_table`
and every OCR'd row is checked: Gain/Loss must equal +-(Spot - Rate) / Rate, or the row is flagged and
never shown.
"""

from __future__ import annotations

import re

PAIR_WORDS = r'(POUND|EURO|AUD|YEN|JPY|SWISS|CAD)'
TABLE_RE = re.compile(rf'^{PAIR_WORDS} (BUY|SELL|NPR) (LONG|SHORT|SQUARE|NPR|FLAT)\b(.*)$')
DATE = r'\d{1,2}/\d{2}/\d{2}'
DATE_RE = re.compile(rf'^{DATE}$')
NUM = r'-?[\d,]*\.\d+'
SPOT_TRACK_RE = re.compile(
    rf'^({DATE}) ([A-Z]{{3}}/[A-Z]{{3}}) (Bought|Sold) (\d+\.\d+)(?: ([0-9]))? ({DATE}) (\d+\.\d+) (-?\d+) ({NUM}) ({NUM})$')
SPOT_OPEN_RE = re.compile(rf'^({DATE}) ([A-Z]{{3}}/[A-Z]{{3}}) (Bought|Sold) (\d+\.\d+)(?: ([0-9]))?$')
NARR_PAIR = [('POUND', r'\b(GBP|POUND)\b'), ('EURO', r'\b(EURO|EUR)\b'), ('AUD', r'(\bAUD\b|A\$)'),
             ('YEN', r'\b(YEN|JPY)\b'), ('SWISS', r'\b(CHF|SF|SWISS)\b'), ('CAD', r'(\bCAD\b|C\$)')]
MONTHS = {m: i + 1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                          'September', 'October', 'November', 'December'])}


class UnknownReport(ValueError):
    pass


def iso(mmddyy: str) -> str:
    m, d, y = mmddyy.split('/')
    return f'20{y}-{int(m):02d}-{int(d):02d}'


def kind_of(text: str) -> str:
    flat = ' '.join(text.split())
    if 'Daily Currency Trading Recommendation SPOT Market' in flat:
        return 'spot'
    if 'Daily Currency Trading Recommendation' in flat and 'CME/Globex' in flat:
        return 'futures'
    if 'Strategic Currency Briefing' in flat:
        return 'morning'
    raise UnknownReport('unknown report')


def report_date(text: str, kind: str) -> str:
    if kind == 'futures':
        for line in text.splitlines()[:8]:          # dated only for the report log
            if DATE_RE.match(line.strip()):
                return iso(line.strip())
    elif kind == 'spot':
        for line in text.splitlines()[:8]:
            if DATE_RE.match(line.strip()):
                return iso(line.strip())
    else:
        flat = ' '.join(text.split())
        m = re.search(r'Strategic Currency Briefing\s+[A-Z][a-z]+day,\s*([A-Z][a-z]+)\s+(\d{1,2}),\s*(20\d\d)', flat)
        if m and m.group(1) in MONTHS:
            return f'{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}'
    raise UnknownReport('report date not found in the document')


def _int(tok: str):
    return None if tok in ('--', '') else int(tok.replace(',', ''))


def _canon(word: str) -> str:
    return 'YEN' if word == 'JPY' else word


def table_rows(cfg: dict, kind: str, lines: list) -> tuple:
    """The six recommendation rows of a spot report, and the table lines it could not read."""
    rows, unparsed = [], []
    pc = cfg['parse']
    for i, line in enumerate(lines):
        s = line.strip()
        if not re.match(rf'^{PAIR_WORDS} ', s):
            continue
        m = TABLE_RE.match(s)
        toks = m.group(4).split() if m else []
        di = next((j for j, t in enumerate(toks) if DATE_RE.match(t)), None)
        if not m or di is None or di < 2:
            unparsed.append(s)
            continue
        name = _canon(m.group(1))
        before, after = toks[:di], toks[di + 1:]
        r = {'pair': name, 'direction': m.group(2), 'position': m.group(3), 'close_raw': _int(before[0]),
             'pos_stop_raw': None, 'pos_target_raw': None, 'pos_entry_raw': _int(before[-1]),
             'pos_entry_date': iso(toks[di]), 'mtm': after[0] if after else '', 'npr': False,
             'entry_raw': None, 'stop_raw': None, 'target_raw': None, 'entry_type': '', 'points_line': '',
             'flags': []}
        mid = before[1:-1]
        if len(mid) >= 1:
            r['pos_stop_raw'] = _int(mid[0])
        if len(mid) >= 2:
            r['pos_target_raw'] = _int(mid[1])
        new = after[1:]
        if new[:1] == ['NPR'] or m.group(2) == 'NPR':
            r['npr'] = True
        elif len(new) == 3 and all(re.fullmatch(r'\d+', t) for t in new):
            r['entry_raw'], r['stop_raw'], r['target_raw'] = (int(t) for t in new)
            r['entry_type'] = 'rule0' if r['entry_raw'] == r['close_raw'] else 'guerilla'
        else:
            unparsed.append(s)
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
        if nxt.startswith('Points'):
            r['points_line'] = ' '.join(nxt.split()[1:])
        _check_levels(r, pc)
        rows.append(r)
    return rows, unparsed


def _check_levels(r: dict, pc: dict) -> None:
    if r['npr']:
        return
    e, s, t = r['entry_raw'], r['stop_raw'], r['target_raw']
    buy = r['direction'] == 'BUY'
    if (buy and not s < e < t) or (not buy and not t < e < s):
        r['flags'].append('levels out of order for the direction')
    if r['close_raw'] is not None and abs(e - r['close_raw']) > pc['entry_flag_points']:
        r['flags'].append(f"entry far from the 4 pm price ({abs(e - r['close_raw'])} points)")
    pts = [int(x) for x in re.findall(r'-?\d+', r['points_line'])]
    if len(pts) >= 2:
        ps, pt = abs(pts[-2]), abs(pts[-1])
        if abs(ps - abs(e - s)) > pc['points_tolerance'] or abs(pt - abs(t - e)) > pc['points_tolerance']:
            r['flags'].append(f"points line {ps}/{pt} differs from the levels {abs(e - s)}/{abs(t - e)}")


def convert(cfg: dict, r: dict) -> dict:
    """Spot integer prices to MT5 prices (design D4)."""
    p = cfg['pairs'][r['pair']]
    f = p['factor']
    nd = 2 if f == 100 else 4
    out = dict(r, symbol=p['symbol'])
    for k in ('close', 'entry', 'stop', 'target', 'pos_stop', 'pos_target', 'pos_entry'):
        v = r.get(f'{k}_raw')
        out[k] = None if v is None else round(v / f, nd)
    return out


def narrative_flags(cfg: dict, lines: list, rows: list) -> list:
    """Narrative notes whose price is far from every listed level of their pair (e.g. CAD 'RULE 2 AT 13202')."""
    by = {r['pair']: r for r in rows}
    lim = cfg['parse']['narrative_flag_points']
    out = []
    for line in lines:
        s = line.strip()
        if not s.startswith('WE ') and 'RULE' not in s:
            continue
        pair = next((name for name, rx in NARR_PAIR if re.search(rx, s)), None)
        if pair is None or pair not in by:
            continue
        r = by[pair]
        levels = [v for v in (r['close_raw'], r['entry_raw'], r['stop_raw'], r['target_raw'], r['pos_entry_raw'],
                              r['pos_stop_raw'], r['pos_target_raw']) if v is not None]
        for tok in re.findall(r'(?<![\d/.])\d{4,5}(?!\d|/|\.\d)', s):
            n = int(tok)
            if levels and min(abs(n - v) for v in levels) > lim:
                out.append({'pair': pair, 'value': n,
                            'flag': f'narrative price off by more than {lim} points'})
    return out


def _split_entry(entry: str, rule: str | None, decimals: int) -> tuple:
    """`0.81881` (no rule column) -> ('0.8188', 1) when the entry has one digit more than the pair's decimals."""
    if rule is not None:
        return entry, int(rule)
    if '.' in entry and len(entry.split('.')[1]) == decimals + 1:
        return entry[:-1], int(entry[-1])
    return entry, None


def track_rows(cfg: dict, kind: str, lines: list) -> tuple:
    """Closed and open trades of the spot track record, and the date lines it could not read."""
    pairs = {v['track']: (k, v) for k, v in cfg['pairs'].items()}
    tol = cfg['parse']['points_tolerance']
    rows, unparsed = [], []
    for line in lines:
        s = line.strip()
        if not re.match(rf'^{DATE} ', s):
            continue
        m, o = SPOT_TRACK_RE.match(s), SPOT_OPEN_RE.match(s)
        if not (m or o):
            if re.match(rf'^{DATE} [A-Z]{{3}}/[A-Z]{{3}} ', s):
                unparsed.append(s)
            continue
        g = (m or o).groups()
        if g[1] not in pairs:
            unparsed.append(s)
            continue
        name, p = pairs[g[1]]
        entry, rule = _split_entry(g[3], g[4], 4)
        side = 'buy' if g[2] == 'Bought' else 'sell'
        r = {'kind': kind, 'pair': name, 'symbol': p['symbol'], 'entry_date': iso(g[0]), 'side': side,
             'entry': float(entry), 'rule': rule, 'status': 'open' if o and not m else 'closed',
             'exit_date': '', 'exit': None, 'points': None, 'pl': '', 'flags': []}
        if r['rule'] is None:
            r['flags'].append('rule number missing')
        if m:
            r.update(exit_date=iso(g[5]), exit=float(g[6]), points=int(g[7]), pl=g[8].replace(',', ''))
            exp = (r['exit'] - r['entry']) / p['pip'] * (1 if side == 'buy' else -1)
            if abs(exp - r['points']) > tol:
                r['flags'].append(f"points differ from entry minus exit ({r['points']} vs {exp:+.0f})")
        rows.append(r)
    return rows, unparsed


# -- Morning positions ---------------------------------------------------------------------------

POS_RE = re.compile(r'^(LONG|SHORT)\s*([A-Z]{3})$')


def _f(tok: str):
    try:
        return float(tok.replace(',', '').replace('%', ''))
    except ValueError:
        return None


def morning_rows(cfg: dict, cells: list) -> list:
    """Positions from the OCR'd table: [pair, spot, 'LONG USD', 'MM/DD/YY', rate, 'x.xx%'] per row.
    Each row is checked against Gain/Loss = sign x (spot - rate) / rate; a failed check is flagged."""
    tol = cfg['parse']['morning_gain_tolerance_pp']
    out = []
    for c in cells:
        c = [t.strip() for t in c if t and t.strip()]
        j = 0
        while j < len(c) - 1:                       # OCR may split 'LONG USD' into two words
            if c[j] in ('LONG', 'SHORT') and re.fullmatch(r'[A-Z]{3}', c[j + 1]):
                c[j:j + 2] = [f'{c[j]} {c[j + 1]}']
            j += 1
        if not c or not re.fullmatch(r'[A-Z]{3}/[A-Z]{3}', c[0]):
            continue
        pair = c[0]
        r = {'pair': pair.replace('/', ''), 'spot': None, 'position': '', 'signal_date': '', 'signal_rate': None,
             'gain_pct': None, 'checked': False, 'flags': []}
        rest = c[1:]
        pos = next((t for t in rest if POS_RE.match(t)), None)
        date = next((t for t in rest if DATE_RE.match(t)), None)
        nums = [t for t in rest if t not in (pos, date)]
        if pos is None or date is None or len(nums) != 3 or not nums[2].endswith('%'):
            r['flags'].append('table row not read: ' + ' | '.join(c))
            out.append(r)
            continue
        spot, rate, gain = _f(nums[0]), _f(nums[1]), _f(nums[2])
        r.update(spot=spot, position=' '.join(POS_RE.match(pos).groups()), signal_date=iso(date), signal_rate=rate, gain_pct=gain)
        side, ccy = POS_RE.match(pos).groups()
        base, quote = pair.split('/')
        if None in (spot, rate, gain) or rate == 0 or ccy not in (base, quote):
            r['flags'].append('table row not read')
        else:
            sign = (1 if side == 'LONG' else -1) * (1 if ccy == base else -1)
            calc = sign * (spot - rate) / rate * 100
            if abs(calc - gain) <= tol:
                r['checked'] = True
            else:
                r['flags'].append(f'gain does not match spot and rate ({gain:.2f}% vs {calc:.2f}%)')
        out.append(r)
    return out


# -- whole document -------------------------------------------------------------------------------

def parse_text(cfg: dict, text: str, morning_cells: list | None = None) -> dict:
    """Everything one report yields. `morning_cells` are the OCR'd table rows for a Morning report."""
    kind = kind_of(text)
    date = report_date(text, kind)
    lines = text.splitlines()
    out = {'kind': kind, 'report_date': date, 'recommendations': [], 'track': [], 'morning': [], 'flags': [],
           'unparsed': []}
    if kind == 'futures':
        return out                                  # spot only: a futures report adds no row
    if kind == 'spot':
        rows, un = table_rows(cfg, kind, lines)
        rows = [convert(cfg, r) for r in rows]
        tr, un2 = track_rows(cfg, kind, lines)
        out.update(recommendations=rows, track=tr, unparsed=un + un2)
        for r in rows:
            out['flags'] += [{'pair': r['pair'], 'value': '', 'flag': f} for f in r['flags']]
        out['flags'] += narrative_flags(cfg, lines, rows)
        for t in tr:
            out['flags'] += [{'pair': t['pair'], 'value': t['entry_date'], 'flag': f} for f in t['flags']]
        if len(rows) != 6:
            out['flags'].append({'pair': '', 'value': len(rows), 'flag': 'table does not hold six pairs'})
    else:
        out['morning'] = morning_rows(cfg, morning_cells or [])
        for r in out['morning']:
            out['flags'] += [{'pair': r['pair'], 'value': '', 'flag': f} for f in r['flags']]
        if not out['morning']:
            out['flags'].append({'pair': '', 'value': '', 'flag': 'positions table not read'})
    return out
