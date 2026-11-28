"""The forward ledger (spec rts-signals/evaluation "Forward ledger"; design D8).

    results/rts_signals/forward_ledger.csv      append-only, one row per (report date, pair)

A spot recommendation enters the ledger once its session has ended, if its report is dated after the
pre-registration's commit day on the Berlin label clock. The row holds the base-trade outcome or the
reason there is none, and `received_late` when the report arrived (Gmail arrival time, or drop-folder
ingest time) after its session ended. Only `eligible` rows (filled and on time) enter the cell. A row is
never changed and a missed report is never backfilled: a later email for an already-ledgered date adds
nothing, and a first-seen late report is ledgered late.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from . import load_config
from .ingest import REPORTS, append_csv, read_csv, rpath
from .rules import session

LEDGER = 'forward_ledger.csv'
FIELDS = ['report_date', 'trade_date', 'pair', 'symbol', 'direction', 'entry_type', 'received_utc', 'received_late',
          'status', 'exit_reason', 'entry', 'stop', 'target', 'exit', 'gross_pips', 'net_pips', 'gross_r', 'net_r',
          'eligible', 'sha256', 'ledgered_utc']


def read(cfg: dict) -> pd.DataFrame:
    rows = read_csv(rpath(cfg, LEDGER))
    return pd.DataFrame(rows, columns=FIELDS) if rows else pd.DataFrame(columns=FIELDS)


def committed_label_day(iso: str) -> pd.Timestamp:
    from .bars import label_now
    return label_now(pd.Timestamp(iso).tz_convert('UTC')).normalize()


def update(cfg: dict, recs: pd.DataFrame, base: pd.DataFrame, committed_day: pd.Timestamp,
           ledgered_utc: str | None = None) -> int:
    """Append the settled forward rows not yet in the ledger. Returns the number appended."""
    from .bars import label_now
    if not len(recs) or not len(base):
        return 0
    have = {(r['report_date'], r['pair']) for r in read_csv(rpath(cfg, LEDGER))}
    reports = {r['sha256']: r for r in read_csv(rpath(cfg, REPORTS))}
    b = {(str(r['report_date']), r['pair']): r for r in base.to_dict('records')}
    stamp = ledgered_utc or datetime.now(timezone.utc).isoformat(timespec='seconds')
    out = []
    for rec in recs.sort_values(['report_date', 'pair']).to_dict('records'):
        key = (str(rec['report_date']), rec['pair'])
        if key in have or pd.Timestamp(rec['report_date'], tz='UTC') <= committed_day:
            continue
        t = b.get(key)
        if t is None or t['status'] == 'pending':
            continue
        rep = reports.get(rec['sha256'], {})
        rcv = rep.get('received_utc', '')
        end = session(rec['report_date'])['end']
        late = bool(rcv) and label_now(pd.Timestamp(rcv).tz_convert('UTC')) > end
        row = {k: t.get(k, '') for k in FIELDS}
        row.update(received_utc=rcv, received_late=late, sha256=rec['sha256'], ledgered_utc=stamp,
                   eligible=(t['status'] == 'filled') and not late)
        if late:
            row['status'] = f"received late ({t['status']})"
        out.append(row)
        have.add(key)
    append_csv(rpath(cfg, LEDGER), FIELDS, out)
    return len(out)


def update_from_files(cfg: dict | None = None, now_label=None) -> dict:
    """Ledger pass used by run.py: needs the committed pre-registration, else appends nothing."""
    from .prereg import PreregNotCommitted, gate
    from .simulate import load_recs, simulate_all
    cfg = cfg or load_config()
    try:
        committed = committed_label_day(gate(cfg))
    except PreregNotCommitted as e:
        return {'ledger': f'not counting: {e}', 'appended': 0}
    recs = load_recs(cfg)
    base = simulate_all(cfg, now_label, recs=recs) if len(recs) else pd.DataFrame()
    return {'ledger': 'counting', 'appended': update(cfg, recs, base, committed)}


def forward_eligible(cfg: dict, committed_day: pd.Timestamp) -> pd.DataFrame:
    d = read(cfg)
    if not len(d):
        return d
    d = d[(d['eligible'] == 'true') & (pd.to_datetime(d['report_date']).dt.tz_localize('UTC') > committed_day)].copy()
    order = {k: i for i, k in enumerate(cfg['pairs'])}
    d['_o'] = d['pair'].map(order)
    return d.sort_values(['report_date', '_o']).drop(columns='_o')


def ledger_path(cfg: dict) -> str:
    return os.path.join(rpath(cfg, LEDGER))
