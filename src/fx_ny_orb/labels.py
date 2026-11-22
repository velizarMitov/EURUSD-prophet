"""The label and power report, before any outcome (spec fx-ny-orb-hot/evaluation; design D5).

    python -m src.fx_ny_orb.labels      write results/fx_ny_orb/labels.md and labels.json

Runs the rule only up to the entry decision (`rule.entries`); `rule.scan_exit` is never called, so no
exit, price after the entry or result is read. Per pair and pooled, hot and non-hot: eligible days,
trades, non-trade reasons and distinct dates, and the minimum detectable mean net R at alpha 0.025
one-sided, 80 % power, with the declared SD of 1.5 R.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ..fx_bb.labels import mde_mean, z_sum
from . import COST_HEADER, config_digest, load_config
from . import rule as R
from .registry import LABELS_JSON, guard, results_dir

REASONS = ['no range', 'range too wide', 'no breakout', 'breakout too far', 'too late', 'no forecast']


def decisions(days: pd.DataFrame, day_bars: dict, cfg: dict) -> pd.DataFrame:
    """Entry decisions for every simulable day: `decision` '' (trade) or the reason."""
    out = days.copy()
    dec = []
    for _, r in out.iterrows():
        if r['day_reason']:
            dec.append(r['day_reason'])
            continue
        dec.append(R.entries(day_bars[r['pair']][r['date']], r['F'], cfg)['reason'])
    out['decision'] = dec
    return out


def power(dec: pd.DataFrame, cfg: dict) -> dict:
    sc = cfg['scoring']
    out = {}
    for key in list(cfg['pairs']) + ['pooled']:
        sub = dec if key == 'pooled' else dec[dec['pair'] == key]
        for side, mask in (('hot', sub['hot']), ('cold', ~sub['hot'])):
            g = sub[mask]
            tr = g[g['decision'] == '']
            n_dates = int(tr['date'].nunique())
            out.setdefault(key, {})[side] = {
                'days': len(g), 'simulable': int((g['day_reason'] == '').sum()), 'trades': len(tr), 'dates': n_dates,
                **{r: int((g['decision'] == r).sum()) for r in REASONS + ['no M15 data', 'corrupt spread']},
                'mde_trades': mde_mean(float(sc['declared_sd_r']), len(tr), sc['alpha'], sc['power']),
                'mde_dates': mde_mean(float(sc['declared_sd_r']), n_dates, sc['alpha'], sc['power'])}
    return out


def write_report(cfg: dict | None = None, now_iso: str | None = None, assembled: tuple | None = None) -> str:
    from .history import assemble
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    days, day_bars = assembled if assembled is not None else assemble(cfg)
    dec = decisions(days, day_bars, cfg)
    pw = power(dec, cfg)
    sc, r = cfg['scoring'], cfg['rule']
    L = ['# fx-ny-orb-hot labels and power', '', f'> **{COST_HEADER}**', '',
         f"NY opening range = M15 bars {', '.join(r['or_bars'])} Berlin; first M15 close outside it from "
         f"{r['breakout_from']}; entry at the next open before {r['entry_before']}; range ≤ {r['or_width_max_f']} F, "
         f"breakout ≤ {r['breakout_max_f']} F; SL {r['stop_f']} F, TP {r['target_f']} F, forced exit {r['last_bar']}. "
         f"Window {cfg['data']['window_start']} .. {cfg['data']['window_end']}. This report runs the rule only up to "
         'the entry decision: no exit, later price or result is read.', '',
         '## Days and entry decisions', '',
         '| Pair | Side | Days | Simulable | Trades | Dates | No range | Too wide | No breakout | Too far | Too late | '
         'No M15 | Corrupt |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key, sides in pw.items():
        for side, b in sides.items():
            L.append(f"| {key} | {side} | {b['days']} | {b['simulable']} | {b['trades']} | {b['dates']} | "
                     f"{b['no range']} | {b['range too wide']} | {b['no breakout']} | {b['breakout too far']} | "
                     f"{b['too late']} | {b['no M15 data']} | {b['corrupt spread']} |")
    L += ['', '## Hot-day trades per year', '', '| Year | ' + ' | '.join(cfg['pairs']) + ' | pooled |',
          '|---|' + '---|' * (len(cfg['pairs']) + 1)]
    tr = dec[(dec['decision'] == '') & dec['hot']]
    for yr, g in tr.groupby(tr['date'].dt.year):
        L.append(f'| {yr} | ' + ' | '.join(str(int((g['pair'] == p).sum())) for p in cfg['pairs']) + f' | {len(g)} |')
    p = pw['pooled']
    L += ['', '## Power — before any outcome', '',
          f"One-sided alpha {sc['alpha']} (0.05/{sc['family_size']}), power {sc['power']:.0%}, "
          f"z(1−α) + z(power) = {z_sum(sc['alpha'], sc['power']):.4f}, declared SD {sc['declared_sd_r']} R (a stop is "
          '−1 R and a target +2.4 R; forced exits shrink the spread).', '',
          '| Cell | Trades | Dates | MDE on trades (R) | MDE on dates (R) |', '|---|---|---|---|---|',
          f"| orb_hot_net_r (hot, pooled) | {p['hot']['trades']} | {p['hot']['dates']} | {p['hot']['mde_trades']:.3f} | "
          f"{p['hot']['mde_dates']:.3f} |",
          f"| non-hot control (pooled) | {p['cold']['trades']} | {p['cold']['dates']} | {p['cold']['mde_trades']:.3f} | "
          f"{p['cold']['mde_dates']:.3f} |", '',
          'The difference cell is limited by the hot side. A DROP means "no effect of this size", not "no effect".',
          '', f'_Generated {now_iso}._', '']
    os.makedirs(results_dir(cfg), exist_ok=True)
    out = os.path.join(results_dir(cfg), 'labels.md')
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    rec = {'stage': 'labels', 'generated': now_iso, 'config_digest': config_digest(cfg),
           'power': {k: {s: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in b.items()}
                         for s, b in v.items()} for k, v in pw.items()}}
    with open(os.path.join(results_dir(cfg), LABELS_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1)
    return out


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-ny-orb-hot label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
