"""The label report and the power table, before any model (spec fx-bb-reversion/band-events).

    python -m src.fx_bb.labels      write results/fx_bb/labels.md and labels.json

Counts events, cooldown removals, exclusions, ties and distinct dates per pair and pooled,
overall, from the first fold (2015) and per year. Uses NO event outcome and no model output:
ties are found by equality alone, and the Cell 1 dispersion is the SD of the 3-day move in ATR
units over ALL valid days, never over events.

    Cell 1 MDE (mean outcome)  = (z(1 - alpha) + z(power)) * sd_all_days / sqrt(n)
    Cell 2 MDE (accuracy)      = 0.5 + (z(1 - alpha) + z(power)) * 0.5 / sqrt(n)

each on the event count and on the distinct-date count (events on one date are not independent).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

import pandas as pd

from . import GROSS_HEADER, config_digest, feature_list_digest, load_config
from . import events as E
from .registry import LABELS_JSON, guard, results_dir


def z_sum(alpha: float, power: float) -> float:
    z = NormalDist().inv_cdf
    return z(1 - alpha) + z(power)


def mde_mean(sd: float, n: int, alpha: float, power: float) -> float:
    return z_sum(alpha, power) * sd / math.sqrt(n) if n else float('nan')


def mde_accuracy(n: int, alpha: float, power: float) -> float:
    return 0.5 + z_sum(alpha, power) * 0.5 / math.sqrt(n) if n else float('nan')


def label_data(d1_all: dict, cfg: dict) -> dict:
    ev, counts = E.build_events(d1_all, cfg)
    ev['tie'] = E.ties(ev, d1_all)
    return {'events': ev, 'counts': counts, 'sd_all': E.dispersion(d1_all, cfg),
            'sd_oos': E.dispersion(d1_all, cfg, cfg['folds']['first_eval_start'])}


def _block(ev: pd.DataFrame) -> dict:
    use = ev[ev['reason'] == '']
    return {'events': len(ev), 'eligible': len(use), 'pair_gap': int((ev['reason'] == 'pair gap').sum()),
            'no_outcome': int((ev['reason'] == 'no outcome').sum()), 'ties': int(use['tie'].sum()),
            'dates': int(use['date'].nunique()), 'targets': int((~use['tie']).sum()),
            'target_dates': int(use.loc[~use['tie'], 'date'].nunique())}


def power(data: dict, cfg: dict) -> dict:
    sc, start = cfg['scoring'], pd.Timestamp(cfg['folds']['first_eval_start'])
    ev = data['events']
    out = {}
    for key in list(cfg['pairs']) + ['pooled']:
        sub = ev if key == 'pooled' else ev[ev['pair'] == key]
        b1, b2 = _block(sub), _block(sub[sub['date'] >= start])
        sd = data['sd_all'][key]
        out[key] = {'cell1': {**b1, 'sd': sd, 'mde_events': mde_mean(sd, b1['eligible'], sc['alpha'], sc['power']),
                              'mde_dates': mde_mean(sd, b1['dates'], sc['alpha'], sc['power'])},
                    'cell2': {**b2, 'mde_events': mde_accuracy(b2['targets'], sc['alpha'], sc['power']),
                              'mde_dates': mde_accuracy(b2['target_dates'], sc['alpha'], sc['power'])}}
    return out


def write_report(cfg: dict | None = None, now_iso: str | None = None, d1_all: dict | None = None) -> str:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if d1_all is None:
        from .daily import build_all
        from .data_loader import load_all
        d1_all = build_all(*load_all(cfg), cfg)
    data = label_data(d1_all, cfg)
    pw = power(data, cfg)
    ev, counts, sc = data['events'], data['counts'], cfg['scoring']
    start = cfg['folds']['first_eval_start']
    zs = z_sum(sc['alpha'], sc['power'])
    L = ['# fx-bb-reversion labels and power', '', f'> **{GROSS_HEADER}**', '',
         f"Events: the first D1 mid close outside the {cfg['bands']['length']}-day, {cfg['bands']['width']:g}-SD "
         f"Bollinger Bands (D1 closes at {cfg['data']['day_close_hour']}:00 New York), counter-trend direction, "
         f"{cfg['cooldown_days']}-day cooldown per pair. Outcome over {cfg['horizon_days']} valid days in ATR"
         f"({cfg['atr_period']}) units. This report reads no event outcome and no model output.", '',
         '## D1 days and breaches', '',
         '| Pair | Start | Valid D1 days | Closes outside | First breaches | Removed by cooldown | Accepted events |',
         '|---|---|---|---|---|---|---|']
    for p in cfg['pairs']:
        c = counts[p]
        L.append(f"| {p} | {cfg['data']['start'][p]} | {c['valid_days']:,} | {c['outside_days']} | {c['first_breaches']} | "
                 f"{c['cooldown_removed']} | {int((ev['pair'] == p).sum())} |")
    tot = {k: sum(c[k] for c in counts.values()) for k in ('valid_days', 'outside_days', 'first_breaches', 'cooldown_removed')}
    L.append(f"| pooled | — | {tot['valid_days']:,} | {tot['outside_days']} | {tot['first_breaches']} | "
             f"{tot['cooldown_removed']} | {len(ev)} |")
    for title, cell in (('Cell 1 — every event from each pair\'s start', 'cell1'),
                        (f'Cell 2 — out-of-sample events from {start}', 'cell2')):
        L += ['', f'## {title}', '',
              '| Pair | Accepted | Pair gap | No outcome yet | Eligible | Ties | Distinct dates |', '|---|---|---|---|---|---|---|']
        for key, r in pw.items():
            b = r[cell]
            L.append(f"| {key} | {b['events']} | {b['pair_gap']} | {b['no_outcome']} | {b['eligible']} | {b['ties']} | "
                     f"{b['dates']} |")
    L += ['', '## Eligible events per year', '', '| Year | ' + ' | '.join(cfg['pairs']) + ' | pooled | distinct dates |',
          '|---|' + '---|' * (len(cfg['pairs']) + 2)]
    use = ev[ev['reason'] == '']
    for yr, g in use.groupby(use['date'].dt.year):
        L.append(f'| {yr} | ' + ' | '.join(str(int((g['pair'] == p).sum())) for p in cfg['pairs']) +
                 f" | {len(g)} | {g['date'].nunique()} |")
    L += ['', '## Events per date (eligible, pooled)', '', '| Events on the date | Dates overall | Dates from ' + start + ' |',
          '|---|---|---|']
    per = use.groupby('date').size()
    per15 = use[use['date'] >= pd.Timestamp(start)].groupby('date').size()
    for k in sorted(set(per.values)):
        L.append(f'| {k} | {int((per == k).sum())} | {int((per15 == k).sum())} |')
    L += ['', '## Power — before any model', '',
          f"One-sided alpha {sc['alpha']} (0.05/{sc['family_size']}), power {sc['power']:.0%}, "
          f"z(1−α) + z(power) = {zs:.4f}.", '',
          '**Cell 1** (mean outcome > 0): minimum detectable mean = 2.80 · sd / √n, with sd the standard deviation '
          'of (close[t+3] − close[t]) / ATR20[t] over ALL valid days, never over events.', '',
          '| Pair | sd, all days | sd, all days from ' + start + ' | Eligible events | MDE on events (ATR) | Distinct dates | MDE on dates (ATR) |',
          '|---|---|---|---|---|---|---|']
    for key, r in pw.items():
        b = r['cell1']
        L.append(f"| {key} | {b['sd']:.3f} | {data['sd_oos'][key]:.3f} | {b['eligible']} | {b['mde_events']:.3f} | "
                 f"{b['dates']} | {b['mde_dates']:.3f} |")
    L += ['', f'**Cell 2** (pooled LightGBM vs the base rate, from {start}): minimum detectable accuracy = '
          '0.5 + 2.80 · 0.5 / √n, on the non-tie events and on their distinct dates.', '',
          '| Pair | Events with a target | MDE on events | Distinct dates | MDE on dates |', '|---|---|---|---|---|']
    for key, r in pw.items():
        b = r['cell2']
        L.append(f"| {key} | {b['targets']} | {b['mde_events']:.2%} | {b['target_dates']} | {b['mde_dates']:.2%} |")
    L += ['', 'Cell 2 is pooled, so only the pooled row is a registered test; the per-pair rows show scale. Only an '
          'effect of about this size can be seen. A DROP means "no effect of this size", not "no effect".',
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
    argparse.ArgumentParser(description='write the fx-bb label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
