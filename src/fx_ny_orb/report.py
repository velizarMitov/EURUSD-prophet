"""results/fx_ny_orb/report.md (spec fx-ny-orb-hot/evaluation).

    python -m src.fx_ny_orb.report

Built only from saved files (registries, labels.json, hist_trades.csv.gz); two builds differ only in
the generation line. Trade numbers appear only after the historical cells are scored. Everything but
the registry rows is descriptive; net of the recorded spread plus 0.4 bps; not a trading result.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_bb.report import wilson
from . import COST_HEADER, load_config
from .registry import CELL_DIFF, CELL_HOT, FWD, HIST, LABELS_JSON, path, read_json, read_rows, results_dir

REASONS = ('take profit', 'stop loss', 'forced exit')


def _pf(x) -> float:
    x = np.asarray(x, dtype=float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float('inf')


def _f(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def group_row(name: str, g: pd.DataFrame) -> str:
    if not len(g):
        return f'| {name} | 0 |' + ' — |' * 10
    share = g['exit_reason'].value_counts(normalize=True)
    win = (g['r_net'] > 0).mean()
    nd = int(g['date'].nunique())
    lo, hi = wilson(win * nd, nd)
    return (f"| {name} | {len(g)} | {share.get('take profit', 0):.0%} | {share.get('stop loss', 0):.0%} | "
            f"{share.get('forced exit', 0):.0%} | {win:.1%} | {lo:.1%}–{hi:.1%} | {g['r_net'].mean():+.3f} | "
            f"{g['gross_pips'].mean():+.1f} / {g['net_pips'].mean():+.1f} | {_f(_pf(g['gross_pips']))} / "
            f"{_f(_pf(g['net_pips']))} | {int(g['conflict'].astype(bool).sum())} |")


HEAD = ('| Group | Trades | TP | SL | Forced | Win (net) | Wilson 95 % (dates) | Mean net R | Pips gross / net | '
        'PF gross / net | Conflicts |')
SEP = '|---|---|---|---|---|---|---|---|---|---|---|'


def examples(tr: pd.DataFrame) -> pd.DataFrame:
    """Five hot-day trades by a fixed rule, never by result."""
    d = tr.sort_values(['date', 'pair'], kind='mergesort')
    picked = []
    for r in REASONS:
        g = d[(d['exit_reason'] == r) & ~d.index.isin(picked)]
        if len(g):
            picked.append(g.index[-1])
    for side in (1, -1):
        g = d[(d['direction'] == side) & ~d.index.isin(picked)]
        if len(g) and len(picked) < 5:
            picked.append(g.index[-1])
    for i in d.index[::-1]:
        if len(picked) >= 5:
            break
        if i not in picked:
            picked.append(i)
    return d.loc[picked]


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    reg = {r['hypothesis']: r for r in read_rows(cfg, HIST)}
    lab = read_json(cfg, LABELS_JSON)
    L = ['# fx-ny-orb-hot report', '', f'> **{COST_HEADER}**', '',
         'New York opening-range breakout (M15 bars 14:00–14:45 Berlin), taken on days the walk-forward HAR-X calls '
         'hot; non-hot days run the same rule as the control.', '', '## Verdicts', '']
    if reg:
        L += ['| Cell | Statistic | Hot trades | Non-hot trades | Value | Bound (α-quantile) | mde_80 | Verdict |',
              '|---|---|---|---|---|---|---|---|']
        for h in (CELL_HOT, CELL_DIFF):
            if h in reg:
                r = reg[h]
                L.append(f"| {h} | {r['metric']} | {r['n_trades_hot']} | {r['n_trades_cold']} | "
                         f"{float(r['statistic']):+.4f} R | {float(r['bound']):+.4f} | {float(r['mde_80']):.4f} | "
                         f"**{r['verdict']}** |")
    else:
        L.append('Not scored yet: the historical cells are scored once, after the pre-registration is committed.')
    fwd = read_rows(cfg, FWD)
    L += ['', '## Forward HAR-X check', '']
    L += ([f"| {r['hypothesis']} | {r.get('n_days', '')} days | {r['verdict']} |" for r in fwd] if fwd else
          ['Running: scored once after 270 settled forward days per pair or on 2028-01-31.'])
    L += ['', '## Power (labels.md, before any outcome)', '']
    if lab:
        p = lab['power']['pooled']
        L += ['| Side | Trades | Dates | MDE on trades | MDE on dates |', '|---|---|---|---|---|']
        L += [f"| {s} | {p[s]['trades']} | {p[s]['dates']} | {p[s]['mde_trades']:.3f} R | {p[s]['mde_dates']:.3f} R |"
              for s in ('hot', 'cold')]
    hp = path(cfg, 'hist_trades.csv.gz')
    if reg and os.path.exists(hp):
        sim = pd.read_csv(hp)
        sim['date'] = pd.to_datetime(sim['date'], utc=True)
        tr = sim[sim['reason'].fillna('') == ''].copy()
        tr['hot'] = tr['hot'].astype(bool)
        hot, cold = tr[tr['hot']], tr[~tr['hot']]
        L += ['', '## Trades (descriptive)', '',
              'Win = net R > 0. Wilson intervals count distinct dates. A bar touching both SL and TP counts as SL '
              '(conflicts).', '', HEAD, SEP, group_row('hot, pooled', hot), group_row('non-hot, pooled', cold)]
        for pair in cfg['pairs']:
            L += [group_row(f'hot, {pair}', hot[hot['pair'] == pair]),
                  group_row(f'non-hot, {pair}', cold[cold['pair'] == pair])]
        L += [group_row('hot, long', hot[hot['direction'] == 1]), group_row('hot, short', hot[hot['direction'] == -1]),
              group_row('hot, DST-mismatch days', hot[hot['dst_mismatch'] == 1])]
        L += [group_row(f'hot, {yr}', g) for yr, g in hot.groupby(hot['date'].dt.year)]
        L += ['', '### Five example hot-day trades (fixed rule: latest TP, SL, forced exit, then latest long/short)', '',
              '| Pair | Dir | Entry (Berlin label) | Entry | SL | TP | Exit (label) | Reason | Pips gross / net | Net R |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for _, e in examples(hot).iterrows():
            L.append(f"| {e['pair']} | {'long' if e['direction'] > 0 else 'short'} | {e['entry_time']} | "
                     f"{e['entry']:.5f} | {e['stop']:.5f} | {e['target']:.5f} | {e['exit_time']} | {e['exit_reason']} | "
                     f"{e['gross_pips']:+.1f} / {e['net_pips']:+.1f} | {e['r_net']:+.2f} |")
    L += ['', 'Only the registry rows are verdicts; everything else is descriptive and none is a trading result.',
          '', f'_Generated {now_iso}._', '']
    out = os.path.join(results_dir(cfg), 'report.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return out


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-ny-orb-hot report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
