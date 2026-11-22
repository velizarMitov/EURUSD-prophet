"""The label report and the power table, before any model (spec fx-bb-sma-exit/evaluation; design D7).

    python -m src.fx_bb_sma.labels      write results/fx_bb_sma/labels.md and labels.json

Counts first breaches, trades, breaches skipped while a trade was open, exclusions and distinct
entry dates. The no-overlap rule needs each trade's exit TIME, so this calls trades.build_stream;
it never calls trades.results, so no exit reason, exit price or result is read or shown.

    Cell 1 MDE (mean net R)   = (z(1 - alpha) + z(power)) * 1.0 R (declared SD) / sqrt(n)
    Cell 2 MDE (accuracy)     = 0.5 + (z(1 - alpha) + z(power)) * 0.5 / sqrt(n)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from ..fx_bb.labels import mde_accuracy, mde_mean, z_sum
from . import COST_HEADER, config_digest, feature_list_digest, load_config
from . import trades as T
from .registry import LABELS_JSON, guard, results_dir


def _block(tr: pd.DataFrame) -> dict:
    use = tr[tr['reason'] == '']
    out = {'trades': len(tr), 'eligible': len(use), 'dates': int(use['date'].nunique())}
    for r in ('pair gap', 'corrupt spread', 'open at data end'):
        out[r] = int((tr['reason'] == r).sum())
    return out


def power(tr: pd.DataFrame, cfg: dict) -> dict:
    sc = cfg['scoring']
    start = pd.Timestamp(cfg['folds']['first_eval_start'])
    out = {}
    for key in list(cfg['pairs']) + ['pooled']:
        sub = tr if key == 'pooled' else tr[tr['pair'] == key]
        b1, b2 = _block(sub), _block(sub[sub['date'] >= start])
        sd = float(sc['declared_sd_r'])
        out[key] = {'cell1': {**b1, 'mde_trades': mde_mean(sd, b1['eligible'], sc['alpha'], sc['power']),
                              'mde_dates': mde_mean(sd, b1['dates'], sc['alpha'], sc['power'])},
                    'cell2': {**b2, 'mde_trades': mde_accuracy(b2['eligible'], sc['alpha'], sc['power']),
                              'mde_dates': mde_accuracy(b2['dates'], sc['alpha'], sc['power'])}}
    return out


def write_report(cfg: dict | None = None, now_iso: str | None = None, inputs: tuple | None = None) -> str:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    ctxs, d1_all = inputs if inputs is not None else T.load_inputs(cfg)
    tr, counts = T.build_stream(ctxs, d1_all, cfg)
    pw = power(tr, cfg)
    sc, start = cfg['scoring'], cfg['folds']['first_eval_start']
    td = cfg['trade']
    L = ['# fx-bb-sma-exit labels and power', '', f'> **{COST_HEADER}**', '',
         f"Trades: entry at the D1 mid close of a first breach of the {cfg['bands']['length']}-day, "
         f"{cfg['bands']['width']:g}-SD Bollinger Bands; take profit at SMA20[t], stop at the same distance beyond the "
         f"entry, time stop at the close of valid day t+{td['time_stop_days']}; one open trade per pair. Exits are "
         'scanned hour by hour on H1 bars. This report reads each trade\'s exit TIME only (for the one-trade rule): no '
         'exit reason, exit price or result is read or shown.', '',
         '## Trades', '',
         '| Pair | Start | Valid D1 days | First breaches | Skipped (trade open) | Trades | Pair gap | Corrupt spread | '
         'Open at data end | Eligible | Distinct dates |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for p in cfg['pairs']:
        c, b = counts[p], pw[p]['cell1']
        L.append(f"| {p} | {cfg['data']['start'][p]} | {c['valid_days']:,} | {c['first_breaches']} | {c['skipped_open']} | "
                 f"{b['trades']} | {b['pair gap']} | {b['corrupt spread']} | {b['open at data end']} | {b['eligible']} | "
                 f"{b['dates']} |")
    b = pw['pooled']['cell1']
    L.append(f"| pooled | — | {sum(c['valid_days'] for c in counts.values()):,} | "
             f"{sum(c['first_breaches'] for c in counts.values())} | {sum(c['skipped_open'] for c in counts.values())} | "
             f"{b['trades']} | {b['pair gap']} | {b['corrupt spread']} | {b['open at data end']} | {b['eligible']} | "
             f"{b['dates']} |")
    use = tr[tr['reason'] == '']
    L += ['', '## Eligible trades per year', '', '| Year | ' + ' | '.join(cfg['pairs']) + ' | pooled | distinct dates |',
          '|---|' + '---|' * (len(cfg['pairs']) + 2)]
    for yr, g in use.groupby(use['date'].dt.year):
        L.append(f'| {yr} | ' + ' | '.join(str(int((g['pair'] == p).sum())) for p in cfg['pairs']) +
                 f" | {len(g)} | {g['date'].nunique()} |")
    L += ['', '## Power — before any model', '',
          f"One-sided alpha {sc['alpha']} (0.05/{sc['family_size']}), power {sc['power']:.0%}, "
          f"z(1−α) + z(power) = {z_sum(sc['alpha'], sc['power']):.4f}. Bootstrap blocks of {sc['block_days']} business "
          'days (a trade lives up to 10), so the effective size is nearer the date count than the trade count.', '',
          f"**Cell 1** (mean net R > 0, every eligible trade): minimum detectable mean = 2.80 · {sc['declared_sd_r']:g} R / √n. "
          'The SD of 1.0 R is declared, not estimated: a take profit is about +1 R and a stop about −1 R gross.', '',
          '| Pair | Eligible trades | MDE on trades (R) | Distinct dates | MDE on dates (R) |', '|---|---|---|---|---|']
    for key, r in pw.items():
        b = r['cell1']
        L.append(f"| {key} | {b['eligible']} | {b['mde_trades']:.3f} | {b['dates']} | {b['mde_dates']:.3f} |")
    L += ['', f'**Cell 2** (pooled LightGBM vs the base rate, from {start}): minimum detectable accuracy = '
          '0.5 + 2.80 · 0.5 / √n. Only the pooled row is a registered test.', '',
          '| Pair | Eligible trades | MDE on trades | Distinct dates | MDE on dates |', '|---|---|---|---|---|']
    for key, r in pw.items():
        b = r['cell2']
        L.append(f"| {key} | {b['eligible']} | {b['mde_trades']:.2%} | {b['dates']} | {b['mde_dates']:.2%} |")
    L += ['', 'Only an effect of about this size can be seen. A DROP means "no effect of this size", not "no effect".',
          '', f'_Generated {now_iso}._', '']
    os.makedirs(results_dir(cfg), exist_ok=True)
    path = os.path.join(results_dir(cfg), 'labels.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    rec = {'stage': 'labels', 'generated': now_iso, 'config_digest': config_digest(cfg),
           'feature_list_digest': feature_list_digest(cfg),
           'power': {k: {c: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in b.items()}
                         for c, b in r.items()} for k, r in pw.items()}}
    with open(os.path.join(results_dir(cfg), LABELS_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1)
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-bb-sma label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
