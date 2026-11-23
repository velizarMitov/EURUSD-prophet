"""The plain report (spec gold-bb-nyrange/evaluation).

    python -m src.gold_bbnr.report      write results/gold_bbnr/report.md

Before scoring: the power table and "not scored yet", with no outcome. After a family is scored: its
verdicts and descriptive tables (Parts 1a/1b: counts, win rates, mean outcome by year and by side, exit
reasons; Part 2: each model's MSE relative to HAR by year). Only the registry rows are verdicts.
Building it twice gives the same content apart from the generated-at line.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import GROSS_HEADER, load_config
from .registry import CELLS, LABELS_JSON, path, read_json, scored
from .walk_forward import FILES, load_csv

TITLES = {'bb': 'Part 1a — Bollinger reversion (3-day, ATR units)', 'bb_sma': 'Part 1b — the SMA20 trade (R)',
          'nyvol': 'Part 2 — the NY session range'}


def _f(x, n=3, sign=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 'n/a'
    return 'n/a' if np.isnan(x) else (f'{x:+.{n}f}' if sign else f'{x:.{n}f}')


def verdicts(cfg) -> list:
    L = ['## Verdicts', '', '| Family | Cell | Statistic | Bound | MDE (80 %) | Rows | Verdict |', '|---|---|---|---|---|---|---|']
    for fam, cells in CELLS.items():
        have = scored(cfg, fam)
        for c in cells:
            r = have.get(c)
            if r is None:
                L.append(f'| {fam} | `{c}` | — | — | — | — | not scored yet |')
            else:
                L.append(f"| {fam} | `{c}` | {_f(r['statistic'], 5, True)} | {_f(r['bound'], 5, True)} | "
                         f"{_f(r['mde_80'], 4)} | {r['n_rows']} | **{r['verdict']}** |")
    return L + ['']


def _group_table(df, value, by, label, win_col=None) -> list:
    L = [f'| {label} | N | Win | Mean |', '|---|---|---|---|']
    for k, g in df.groupby(by, sort=True):
        win = (g[value] > 0).mean()
        L.append(f'| {k} | {len(g)} | {win:.1%} | {_f(g[value].mean(), 3, True)} |')
    return L + ['']


def part_bb(cfg) -> list:
    ev = load_csv(cfg, FILES['bb'][0])
    rule = ev[ev['reason'].fillna('') == ''].copy()
    rule['year'] = rule['date'].dt.year
    rule['side_name'] = np.where(rule['dir'] > 0, 'below band (long)', 'above band (short)')
    L = [f"Events {len(rule)}; win (r > 0) {(rule['r'] > 0).mean():.1%}; mean r {_f(rule['r'].mean(), 3, True)} ATR.", '']
    return L + _group_table(rule, 'r', 'side_name', 'Side') + _group_table(rule, 'r', 'year', 'Year')


def part_bb_sma(cfg) -> list:
    tr = load_csv(cfg, FILES['bb_sma'][0])
    run = read_json(cfg, 'run_bb_sma.json') or {}
    rule = tr[tr['R'].notna()].copy()
    rule['year'] = rule['date'].dt.year
    rule['side_name'] = np.where(rule['dir'] > 0, 'below band (long)', 'above band (short)')
    ex = rule['exit_reason'].value_counts(normalize=True)
    L = [f"Opened trades {len(tr)} (blocked by an open trade {run.get('blocked', 'n/a')}); with an exit {len(rule)}; "
         f"win (R > 0) {(rule['R'] > 0).mean():.1%}; mean R {_f(rule['R'].mean(), 3, True)}.",
         'Exits: ' + ', '.join(f'{k} {v:.0%}' for k, v in ex.sort_index().items()) + '.', '']
    return L + _group_table(rule, 'R', 'side_name', 'Side') + _group_table(rule, 'R', 'year', 'Year')


def part_nyvol(cfg) -> list:
    d = load_csv(cfg, FILES['nyvol'][1])
    y = d['y']
    d['year'] = d['date'].dt.year
    L = ['| Year | Days | HAR-X / HAR | LightGBM / HAR | MLP / HAR |', '|---|---|---|---|---|']

    def rel(g, k):
        return ((g['y'] - g[f'p_{k}']) ** 2).mean() / ((g['y'] - g['p_har']) ** 2).mean()
    for yr, g in list(d.groupby('year')) + [('all', d)]:
        L.append(f"| {yr} | {len(g)} | {_f(rel(g, 'harx'))} | {_f(rel(g, 'lightgbm'))} | {_f(rel(g, 'mlp'))} |")
    return [f'Scored days {len(d)}; mean y {_f(y.mean(), 3)}.', ''] + L + ['']


PARTS = {'bb': part_bb, 'bb_sma': part_bb_sma, 'nyvol': part_nyvol}


def render(cfg, now_iso) -> str:
    lab = read_json(cfg, LABELS_JSON) or {}
    L = ['# gold-bb-nyrange report', '', f'> **{GROSS_HEADER}**', '',
         'Only the registry rows are verdicts; every table below is descriptive and none is a trading result.', '']
    L += verdicts(cfg)
    if lab:
        a, b, c = lab['part_1a'], lab['part_1b'], lab['part_2']
        L += ['## Power (labels.md, before any outcome)', '', '| Cell | Sample | Minimum detectable effect |', '|---|---|---|',
              f"| 1a rule | {a['rule_events']} events | {_f(a['mde_rule'])} ATR |",
              f"| 1a model | {a['model_events']} events | accuracy {_f(a['mde_model_accuracy'])} |",
              f"| 1b rule | ≤ {b['rule_upper_bound']} trades | ≥ {_f(b['mde_rule_at_upper_bound'])} R |",
              f"| 1b model | ≤ {b['model_upper_bound']} trades | accuracy ≥ {_f(b['mde_model_accuracy_at_upper_bound'])} |",
              f"| 2 (each) | {c['scored_days']} days | {_f(c['mde_sd_units'], 4)} SD |", '']
    for fam, fn in PARTS.items():
        L += [f'## {TITLES[fam]}', '']
        if not scored(cfg, fam):
            L += ['Not scored yet: no outcome is shown.', '']
            continue
        L += fn(cfg)
    L += [f'_Generated {now_iso}._', '']
    return '\n'.join(L)


def write(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    p = path(cfg, 'report.md')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(render(cfg, now_iso))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='write the gold-bb-nyrange report').parse_args(argv)
    print(write())


if __name__ == '__main__':
    main()
