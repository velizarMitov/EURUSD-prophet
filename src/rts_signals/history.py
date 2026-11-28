"""The descriptive historical report (spec rts-signals/evaluation "Historical check is descriptive").

    python -m src.rts_signals.history      rebuild results/rts_signals/history.md

RTS's November 2026 track record and the reports already received were seen before this check was
designed, so nothing here is a verdict. Per pair and in total it shows RTS's stated points, those points
less the broker's recorded spread at the entry, the reconciliation counts, and the base trades of the
received reports gross and net.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from . import COST_HEADER, load_config, results_dir
from .reconcile import RECON
from .simulate import BASE

MARK = 'descriptive — no verdict'


def _read(cfg, name) -> pd.DataFrame:
    p = os.path.join(results_dir(cfg), name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def build(cfg: dict | None = None, generated: str | None = None) -> str:
    cfg = cfg or load_config()
    rc, bt = _read(cfg, RECON), _read(cfg, BASE)
    order = list(cfg['pairs'])
    L = ['# RTS signals — history / история', '',
         f'> **{MARK}.** The results below were seen before this check was built; they cannot decide '
         'anything. The decision is the forward cell `rts_spot_net_r` (see report.md). / Резултатите са видени '
         'преди проверката; само тестът напред решава.', '',
         f'> {COST_HEADER}', '',
         f"_Generated {generated or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_", '']
    L += ['## 1. RTS\'s own spot track record on our prices / Отчетът на RTS върху нашите цени', '']
    if len(rc):
        first, last = rc['entry_date'].min(), rc['entry_date'].max()
        L += [f'{len(rc)} closed trades {first} .. {last} (latest version of each line). `spread` = the broker\'s '
              'recorded spread at the minute our prices first trade the entry.', '',
              '| pair | trades | RTS points | spread (pips) | RTS points − spread | confirmed | different exit | '
              'not reproducible |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in order + ['total']:
            d = rc if name == 'total' else rc[rc['pair'] == name]
            if not len(d):
                continue
            c = d['label'].value_counts()
            pts, spd = d['rts_points'].astype(float).sum(), d['spread_pips'].astype(float).sum()
            lab = cfg['pairs'][name]['symbol'] if name != 'total' else '**total**'
            L.append(f"| {lab} | {len(d)} | {pts:+.0f} | {spd:.1f} | {pts - spd:+.1f} | {c.get('confirmed', 0)} | "
                     f"{c.get('different exit', 0)} | {c.get('not reproducible', 0)} |")
        L += ['', '- `confirmed`: after the entry, our prices trade RTS\'s exit price (±3 points) by its exit date, '
              'and before the stop when the rule fixes it. It shows the trade was possible on our prices, not that '
              'every RTS decision is reproduced.',
              '- RTS enters Rule 0 trades at 18:00 New York, the broker\'s daily rollover, when the spread is widest. '
              'That is why the spread column is large.', '']
        bad = rc[rc['label'] != 'confirmed']
        if len(bad):
            L += ['Not confirmed:', '', '| entry date | pair | side | entry | rule | exit | RTS points | label | note |',
                  '|---|---|---|---:|---:|---:|---:|---|---|']
            for r in bad.itertuples():
                L.append(f'| {r.entry_date} | {r.symbol} | {r.side} | {r.entry} | {r.rule} | {r.exit} | {r.rts_points} | '
                         f'{r.label} | {r.note} |')
            L.append('')
    else:
        L += ['No track record stored yet.', '']
    L += ['## 2. Base trades of the received reports / Базови сделки от получените отчети', '']
    if len(bt):
        f = bt[bt['status'] == 'filled']
        L += [f'Reports {", ".join(sorted(bt["report_date"].astype(str).unique()))}. Status counts: ' +
              ', '.join(f'{k} {v}' for k, v in bt['status'].value_counts().sort_index().items()) + '.', '',
              '| pair | filled | target | stop | time | gross pips | net pips | gross R | net R |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name in order + ['total']:
            d = f if name == 'total' else f[f['pair'] == name]
            if not len(d):
                continue
            c = d['exit_reason'].value_counts()
            lab = cfg['pairs'][name]['symbol'] if name != 'total' else '**total**'
            L.append(f"| {lab} | {len(d)} | {c.get('target', 0)} | {c.get('stop', 0)} | {c.get('time', 0)} | "
                     f"{d['gross_pips'].astype(float).sum():+.1f} | {d['net_pips'].astype(float).sum():+.1f} | "
                     f"{d['gross_r'].astype(float).sum():+.2f} | {d['net_r'].astype(float).sum():+.2f} |")
        L += ['', f'Too few trades to mean anything. {MARK}.', '']
    else:
        L += ['No base trades yet.', '']
    return '\n'.join(L)


def write(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    p = os.path.join(results_dir(cfg), 'history.md')
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(build(cfg))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='rebuild results/rts_signals/history.md (descriptive)').parse_args(argv)
    print(write())


if __name__ == '__main__':
    main()
