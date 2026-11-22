"""The label report and the power table, before any model (spec fx-ny-volatility/range-target).

    python -m src.fx_nyvol.labels      write results/fx_nyvol/labels.md and labels.json

Per pair: weekdays, eligible days, out-of-sample days from 2013-01-01, exclusions by reason, eligible
days per year, the mean and SD of y, and the detectable mean squared-error difference at alpha 0.0125
one-sided and 80 % power in units of that difference's SD: (z(1 - alpha) + z(power)) / sqrt(n).
No model output is used.
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
from .registry import LABELS_JSON, guard, results_dir
from .target import REASONS, dataset, load_frames


def mde_sd(n: int, alpha: float, power: float) -> float:
    z = NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) / math.sqrt(n) if n else float('nan')


def write_report(cfg: dict | None = None, now_iso: str | None = None, data: dict | None = None) -> str:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if data is None:
        data = dataset(*load_frames(cfg), cfg)
    sc, start = cfg['scoring'], pd.Timestamp(cfg['folds']['first_eval_start'], tz='UTC')
    s = cfg['session']
    L = ['# fx-ny-volatility labels and power', '', f'> **{GROSS_HEADER}**', '',
         f"Target: y = log((mid high − mid low) / mid close14) over the bars labelled {s['target_hours'][0]}:00–"
         f"{s['target_hours'][1]}:00 Berlin (16:00–23:00 Sofia); features at the close of the bar labelled "
         f"{s['data_hour']}:00. Out of sample from {cfg['folds']['first_eval_start']}. No model output is used.", '',
         '## Days', '', '| Pair | Weekdays | Eligible | Eligible from 2013 | Mean y | SD y | Mean y_bid | corr(y, y_bid) |',
         '|---|---|---|---|---|---|---|---|']
    rec = {}
    for p, d in data.items():
        use = d[d['reason'] == '']
        oos = use[use.index >= start]
        rec[p] = {'weekdays': len(d), 'eligible': len(use), 'oos': len(oos),
                  'mde_sd': mde_sd(len(oos), sc['alpha'], sc['power']),
                  'exclusions': {r: int((d['reason'] == r).sum()) for r in REASONS}}
        L.append(f"| {p} | {len(d):,} | {len(use):,} | {len(oos):,} | {use['y'].mean():.4f} | {use['y'].std():.4f} | "
                 f"{use['y_bid'].mean():.4f} | {use['y'].corr(use['y_bid']):.4f} |")
    L += ['', '## Exclusions (one reason per day, shared by both pairs)', '', '| Reason | Days |', '|---|---|']
    first = next(iter(data.values()))
    L += [f"| {r} | {int((first['reason'] == r).sum())} |" for r in REASONS]
    L += ['', '## Eligible days per year', '', '| Year | Eligible |', '|---|---|']
    use = first[first['reason'] == '']
    L += [f'| {yr} | {len(g)} |' for yr, g in use.groupby(use.index.year)]
    L += ['', '## Power — before any model', '',
          f"One-sided alpha {sc['alpha']} (0.05/{sc['family_size']}), power {sc['power']:.0%}. The detectable mean of "
          'the per-day squared-error difference, in units of that difference\'s own standard deviation, is '
          '(z(1−α) + z(power)) / √n. Five-day blocks barely change it for a same-day target.', '',
          '| Pair | Out-of-sample days | Detectable difference (σ units) |', '|---|---|---|']
    L += [f"| {p} | {r['oos']:,} | {r['mde_sd']:.4f} |" for p, r in rec.items()]
    L += ['', 'A KEEP of this size can be practically tiny; the report gives the relative MSE reduction next to it.',
          '', f'_Generated {now_iso}._', '']
    os.makedirs(results_dir(cfg), exist_ok=True)
    path = os.path.join(results_dir(cfg), 'labels.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    with open(os.path.join(results_dir(cfg), LABELS_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'stage': 'labels', 'generated': now_iso, 'config_digest': config_digest(cfg),
                   'feature_list_digest': feature_list_digest(cfg), 'pairs': rec}, fh, indent=1)
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-ny-volatility label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
