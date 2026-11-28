"""The chart file for mql5/Indicators/ProphetRTS.mq5 (spec rts-signals/chart-display; design D10).

    python -m src.rts_signals.export      rebuild the file now

Rebuilds ONE file, %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\EURUSDProphet\\rts_signals.tsv (override the
folder with MT5_COMMON_FILES), through a temporary file and os.replace. It is the family's only write
outside the repository, and nothing reads it back. A failed write never stops a pass: `safe_rebuild`
returns `export failed`.

Rows (`kind`):
  today    the latest spot report's six rows while its session is still to come or running: entry, stop and
           target lines, labelled as RTS's unverified suggestion (owner decision 2026-11-28)
  trade    the base trades of the last `display_days` report days, with exit and net result, or the reason
           there was no trade
  morning  the latest Morning report's checked positions (corner)
Every time is a Berlin bar label ("YYYY.MM.DD HH:MM"); every visible text is built here in Bulgarian and
English. Trade words are RTS's own levels. There is no lot size and no promise.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd

from . import load_config
from .registry import verdict

COMMON_ENV = 'MT5_COMMON_FILES'
FIELDS = ['kind', 'symbol', 'report_date', 'direction', 'entry_type', 'start', 'end', 'entry', 'stop', 'target',
          'status', 'entry_time', 'exit_time', 'exit', 'net_pips', 'net_r', 'label_bg', 'label_en']
REASON = {'target': ('цел', 'target'), 'stop': ('стоп', 'stop'), 'time': ('изход по време', 'time exit')}
STATUS = {'entry not reached': 'входът не е достигнат', 'no price data': 'няма ценови данни',
          'open beyond stop': 'отваряне отвъд стопа', 'price mismatch': 'цената не съвпада с MT5',
          'levels out of order': 'нивата са объркани', 'npr': 'без препоръка (NPR)', 'pending': 'предстои'}


def export_path(cfg: dict, env=None) -> str:
    env = os.environ if env is None else env
    base = env.get(COMMON_ENV)
    if not base:
        appdata = env.get('APPDATA')
        if not appdata:
            raise RuntimeError(f'no APPDATA and no {COMMON_ENV}: cannot find the MT5 Common\\Files folder')
        base = os.path.join(appdata, 'MetaQuotes', 'Terminal', 'Common', 'Files')
    return os.path.join(base, cfg['export']['subdir'], cfg['export']['file'])


def lab(t) -> str:
    if t is None or (isinstance(t, float) and pd.isna(t)) or t == '':
        return ''
    return pd.Timestamp(t).strftime('%Y.%m.%d %H:%M')


def px(v, nd) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ''
    return '' if pd.isna(f) else f'{f:.{nd}f}'


def digits(cfg: dict, pair: str) -> int:
    return 2 if cfg['pairs'][pair]['factor'] == 100 else 4


def texts(cfg: dict, n: int, v: str) -> dict:
    need = cfg['forward']['min_trades']
    syms = ', '.join(p['symbol'] for p in cfg['pairs'].values())
    st = ((f'RTS тест напред: {n} / {need} · присъда: {v}', f'RTS forward test: {n} / {need} · verdict: {v}') if v else
          (f'RTS тест напред: {n} / {need} · няма присъда', f'RTS forward test: {n} / {need} · no verdict'))
    return {'status': st,
            'note': ('Нивата са предложения на RTS, не сигнал на този проект. Резултатите са нето от спреда.',
                     'Levels are RTS\'s suggestions, not this project\'s signal. Results are net of spread.'),
            'morning': ('RTS сутрешни позиции (само информация)', 'RTS morning positions (information only)'),
            'other_symbol': (f'Този индикатор е за {syms}.', f'This indicator is for {syms}.')}


def _today_label(r: dict, nd: int) -> tuple:
    if r['status'] == 'npr':
        return 'RTS: NPR · без препоръка · непроверено', 'RTS: NPR · no position recommended · unverified'
    e, s, t = px(r['entry'], nd), px(r['stop'], nd), px(r['target'], nd)
    g = (' (партизански)', ' (guerilla)') if r['entry_type'] == 'guerilla' else ('', '')
    return (f"RTS: {r['direction']} {e}{g[0]} · стоп {s} · цел {t} · непроверено",
            f"RTS: {r['direction']} {e}{g[1]} · stop {s} · target {t} · unverified")


def _trade_label(r: dict, nd: int) -> tuple:
    if r['status'] != 'filled':
        bg = STATUS.get(r['status'], r['status'])
        return f"RTS: {r['direction']} {px(r['entry'], nd)} · {bg}", f"RTS: {r['direction']} {px(r['entry'], nd)} · {r['status']}"
    rb, re_ = REASON.get(r['exit_reason'], (r['exit_reason'], r['exit_reason']))
    np_, nr = float(r['net_pips']), float(r['net_r'])
    return (f"RTS: {r['direction']} · {rb} · {np_:+.1f} пипса нето · {nr:+.2f} R",
            f"RTS: {r['direction']} · {re_} · {np_:+.1f} pips net · {nr:+.2f} R")


def build(cfg: dict, recs: pd.DataFrame, base: pd.DataFrame, morning: pd.DataFrame, n_forward: int,
          now_label: pd.Timestamp) -> str:
    from .rules import session
    v = verdict(cfg)
    lines = ['\t'.join(['#meta', 'version=1', f'generated_utc={datetime.now(timezone.utc):%Y.%m.%d %H:%M}',
                        f'forward_trades={n_forward}', f"min_trades={cfg['forward']['min_trades']}",
                        f"verdict={v or 'none'}"])]
    for k, (bg, en) in texts(cfg, n_forward, v).items():
        lines.append('\t'.join(['#text', k, bg, en]))
    lines.append('\t'.join(FIELDS))
    rows = []
    if len(recs):
        recs = recs.copy()
        recs['report_date'] = recs['report_date'].astype(str)
        b = {(str(r['report_date']), r['pair']): r for r in base.to_dict('records')} if len(base) else {}
        dates = sorted(recs['report_date'].unique())
        latest = dates[-1]
        s_latest = session(latest)
        for d in dates[-int(cfg['export']['display_days']):]:
            s = session(d)
            for rec in recs[recs['report_date'] == d].to_dict('records'):
                pair, nd = rec['pair'], digits(cfg, rec['pair'])
                t = b.get((d, pair), {})
                row = {k: '' for k in FIELDS}
                row.update(symbol=cfg['pairs'][pair]['symbol'], report_date=d.replace('-', '.'),
                           direction=rec['direction'], entry_type=rec.get('entry_type', ''), start=lab(s['open']),
                           end=lab(s['end']), entry=px(rec.get('entry'), nd), stop=px(rec.get('stop'), nd),
                           target=px(rec.get('target'), nd))
                if d == latest and now_label < s_latest['end']:
                    row.update(kind='today', status='npr' if str(rec.get('npr')).lower() == 'true' else 'unverified')
                    row['label_bg'], row['label_en'] = _today_label(dict(rec, status=row['status']), nd)
                else:
                    st = t.get('status', 'pending') if t else 'pending'
                    row.update(kind='trade', status=st)
                    if st == 'filled':
                        row.update(entry=px(t['entry'], nd), stop=px(t['stop'], nd), target=px(t['target'], nd),
                                   entry_time=lab(t['entry_time']), exit_time=lab(t['exit_time']),
                                   exit=px(t['exit'], nd + 1), net_pips=f"{float(t['net_pips']):.1f}",
                                   net_r=f"{float(t['net_r']):.2f}")
                    lb = dict(rec, **{k: row[k] for k in ('entry',)}, status=st,
                              exit_reason=t.get('exit_reason', ''), net_pips=t.get('net_pips', 0),
                              net_r=t.get('net_r', 0))
                    row['label_bg'], row['label_en'] = _trade_label(lb, nd)
                rows.append(row)
    if len(morning):
        m = morning[morning['report_date'] == morning['report_date'].max()]
        m = m[m['checked'] == 'true']
        for r in m.to_dict('records'):
            row = {k: '' for k in FIELDS}
            row.update(kind='morning', symbol=r['pair'], report_date=str(r['report_date']).replace('-', '.'),
                       label_bg=f"{r['pair']} {r['position'].replace('LONG', 'ДЪЛГА').replace('SHORT', 'КЪСА')} от "
                                f"{r['signal_date']} при {r['signal_rate']} ({float(r['gain_pct']):+.2f}%)",
                       label_en=f"{r['pair']} {r['position']} since {r['signal_date']} at {r['signal_rate']} "
                                f"({float(r['gain_pct']):+.2f}%)")
            rows.append(row)
    for row in rows:
        lines.append('\t'.join(str(row[k]).replace('\t', ' ') for k in FIELDS))
    return '\n'.join(lines) + '\n'


def write_atomic(path: str, text: str, replace=os.replace, sleep=time.sleep, tries: int = 5) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    for attempt in range(tries):
        try:
            replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            sleep(0.2)


def gather(cfg: dict, now_label=None) -> tuple:
    from .bars import label_now
    from .ingest import MORNING, read_csv, rpath
    from .ledger import forward_eligible
    from .report import committed_day
    from .simulate import load_recs, simulate_all
    now = label_now() if now_label is None else pd.Timestamp(now_label)
    recs = load_recs(cfg)
    base = simulate_all(cfg, now, recs=recs) if len(recs) else pd.DataFrame()
    morning = pd.DataFrame(read_csv(rpath(cfg, MORNING)))
    c = committed_day(cfg)
    n = len(forward_eligible(cfg, c)) if c is not None else 0
    return recs, base, morning, n, now


def rebuild(cfg: dict | None = None, env=None, now_label=None) -> str:
    cfg = cfg or load_config()
    p = export_path(cfg, env)
    write_atomic(p, build(cfg, *gather(cfg, now_label)))
    return p


def safe_rebuild(cfg: dict | None = None, env=None, now_label=None) -> str:
    try:
        rebuild(cfg, env, now_label)
        return 'exported'
    except Exception as e:                       # never stops a pass, never changes a stored row
        return f'export failed ({type(e).__name__})'


def main(argv=None):
    argparse.ArgumentParser(description='rebuild the rts_signals chart file').parse_args(argv)
    print(rebuild())


if __name__ == '__main__':
    main()
