"""The plain forward report (spec rts-signals/evaluation "Plain report").

    python -m src.rts_signals.report      rebuild results/rts_signals/report.md

Before scoring it shows only the forward trade count out of 300, the running exit counts and the days
without a usable report by reason: no mean, total or test result, and `not scored yet`. After scoring it
shows the verdict row.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from . import CELL, COST_HEADER, load_config, results_dir
from .ledger import committed_label_day, forward_eligible, read
from .registry import read_rows


def committed_day(cfg: dict):
    try:
        from .prereg import gate
        return committed_label_day(gate(cfg))
    except Exception:
        return None


def build(cfg: dict | None = None, committed=None, today=None, generated: str | None = None) -> str:
    cfg = cfg or load_config()
    need = int(cfg['forward']['min_trades'])
    committed = committed if committed is not None else committed_day(cfg)
    L = ['# RTS signals — forward report / тест напред', '', f'> {COST_HEADER}', '',
         f"_Generated {generated or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_", '']
    if committed is None:
        return '\n'.join(L + ['**The pre-registration is not committed yet: no forward trade counts.** / '
                              'Предварителната регистрация още не е в git — нищо не се брои.', ''])
    led = read(cfg)
    fwd = led[pd.to_datetime(led['report_date']).dt.tz_localize('UTC') > committed] if len(led) else led
    elig = forward_eligible(cfg, committed)
    L += [f"Counting from report dates after {committed.date()} (pre-registration commit, Berlin label clock).", '',
          f'## Forward trades: {len(elig)} / {need}', '']
    if len(elig):
        c = elig['exit_reason'].value_counts()
        L += ['| target | stop | time |', '|---:|---:|---:|',
              f"| {c.get('target', 0)} | {c.get('stop', 0)} | {c.get('time', 0)} |", '']
    if len(fwd):
        no = fwd[fwd['eligible'] != 'true']['status'].value_counts().sort_index()
        if len(no):
            L += ['Recommendations without a counted trade, by reason:', '', '| reason | count |', '|---|---:|']
            L += [f'| {k} | {v} |' for k, v in no.items()] + ['']
    today = pd.Timestamp(today or pd.Timestamp.now(tz='UTC')).normalize()
    if today.tzinfo is None:
        today = today.tz_localize('UTC')
    days = pd.bdate_range(committed + pd.Timedelta(days=1), today - pd.Timedelta(days=1))
    seen = set(fwd['report_date']) if len(fwd) else set()
    missing = [d for d in days if str(d.date()) not in seen]
    L += [f'Business days without a settled spot report: {len(missing)}.', '']
    rows = [r for r in read_rows(cfg) if r['hypothesis'] == CELL]
    if rows:
        r = rows[0]
        L += ['## Verdict', '', f"**{r['verdict']}** on {r['date']}: {r['n_trades']} trades, mean net R "
              f"{r['mean_net_r'] or '—'}, lower bound {r['bound'] or '—'} (alpha {r['alpha']}).", '', r['notes'], '']
    else:
        L += ['**not scored yet** — results stay hidden until the one scoring. / още не е оценено.', '']
    return '\n'.join(L)


def write(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    p = os.path.join(results_dir(cfg), 'report.md')
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(build(cfg))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='rebuild results/rts_signals/report.md').parse_args(argv)
    print(write())


if __name__ == '__main__':
    main()
