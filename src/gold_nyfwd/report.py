"""The plain report (spec gold-nyrange-forward/forward-evaluation).

    python -m src.gold_nyfwd.report      write results/gold_nyfwd/report.md

Before scoring it shows only counts: forward days, days without a forecast by reason, weekdays with no
forecast row at all (PC off or MT5 unreachable), and the dollar-index contracts used. It shows no
squared error and no MSE, and says `not scored yet`. After scoring it shows the registry verdict, the MSE
by month and HAR-X's MSE relative to HAR's. Two builds differ only in the generated-at line.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from . import CELL, GROSS_HEADER, load_config, results_dir
from .live import read_log
from .registry import read_rows

FILE = 'report.md'


def build(cfg: dict, committed=None, today=None, generated=None) -> str:
    fc, st = read_log(cfg, 'forecasts'), read_log(cfg, 'settlements')
    scored = next((r for r in read_rows(cfg) if r['hypothesis'] == CELL), None)
    need = int(cfg['forward']['min_days'])
    ok = st[st['status'] == 'ok'] if len(st) else st
    fwd = ok[ok['date'] > committed] if committed is not None and len(ok) else ok
    generated = generated or datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    L = ['# gold-nyrange-forward — report', '', f'> **{GROSS_HEADER}**', '', f'Generated {generated}.', '',
         '## Forward days', '',
         f"- Forward days (settled ok, after the pre-registration commit): **{len(fwd)} / {need}**; "
         f"deadline {cfg['forward']['deadline']}.",
         f'- Forecast rows: {len(fc)}; settlement rows: {len(st)}.']
    if committed is not None:
        L.append(f'- Pre-registration commit day (label clock): {committed.date()}.')
    L += ['', '## Days without a forecast, by reason', '']
    no = fc[fc['status'] != 'ok'] if len(fc) else fc
    if len(no):
        L += [f'- {k}: {v}' for k, v in sorted(no['status'].value_counts().items())]
    else:
        L.append('- none')
    if len(st):
        bad = st[st['status'] != 'ok']
        if len(bad):
            L += ['', 'Settled but not scorable:', ''] + [f'- {k}: {v}' for k, v in sorted(bad['status'].value_counts().items())]
    L += ['', '## Weekdays with no forecast row', '']
    if len(fc):
        start = fc['date'].min()
        end = pd.Timestamp(today) if today is not None else fc['date'].max()
        end = (end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC')).normalize()
        days = pd.bdate_range(start, end)
        missed = [d for d in days if d not in set(fc['date'])]
        L.append(f'- {len(missed)}' + (f": {', '.join(str(d.date()) for d in missed)}" if missed else ''))
    else:
        L.append('- no forecast rows yet')
    L += ['', '## Dollar-index contracts used', '']
    used = fc[fc['contract'].notna() & (fc['contract'].astype(str) != '')] if len(fc) else fc
    if len(used):
        for c, g in used.groupby('contract'):
            L.append(f"- {c}: {len(g)} days, {g['date'].min().date()} .. {g['date'].max().date()}")
    else:
        L.append('- none yet')
    L += ['', '## Verdict', '']
    if scored is None:
        L += ['- not scored yet. No error figure is shown before the one scoring.',
              '- Expected power at 60 days is about 34 %: a DROP would not be evidence that the model fails.']
    else:
        L.append(f"- `{CELL}`: **{scored['verdict']}** on {scored['n_days']} forward days (scored {scored['date']}).")
        if scored.get('mse_ratio'):
            L.append(f"- HAR-X MSE / HAR MSE: {float(scored['mse_ratio']):.3f} (HAR-X {float(scored['mse_harx']):.4f}, "
                     f"HAR {float(scored['mse_har']):.4f}); upper bound of the mean difference "
                     f"{float(scored['ci_upper']):.5f}.")
            d = fwd.sort_values('date').iloc[:need] if len(fwd) else fwd
            if len(d):
                L += ['', '| month | days | MSE HAR-X | MSE HAR |', '|---|---:|---:|---:|']
                for mth, g in d.groupby(d['date'].dt.strftime('%Y-%m')):
                    L.append(f"| {mth} | {len(g)} | {g['se_harx'].astype(float).mean():.4f} | "
                             f"{g['se_har'].astype(float).mean():.4f} |")
        L.append('- A DROP is not evidence that the model fails; the test was weak by design.')
    return '\n'.join(L) + '\n'


def write(cfg: dict | None = None, committed=None, today=None, generated=None) -> str:
    cfg = cfg or load_config()
    if committed is None:
        from .export import committed_day
        committed = committed_day(cfg)
    p = os.path.join(results_dir(cfg), FILE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(build(cfg, committed, today, generated))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='write the gold-nyrange-forward report').parse_args(argv)
    print(write())


if __name__ == '__main__':
    main()
