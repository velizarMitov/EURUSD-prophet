"""The binary session target, the label report and the power table (spec fx-session/session-target).

For eligible day D:  y = 1 if mid_close(21:00 bar) > mid_close(14:00 bar), 0 if below; equal -> tie,
no target. The same sign on bid closes feeds the parity check only. No barrier, no class weights;
each target spans bars 15:00-21:00 of D only, so no two days share a bar. GROSS of cost.

    python -m src.fx_session.labeling      write results/fx_session/labels.md
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import GROSS_HEADER, load_config, repo_path
from .session import bar_at, mid, session_calendar


def build_targets(frames: dict, points: dict, cfg: dict) -> dict:
    """{pair: DataFrame by date: entry/exit mid and bid closes, y, y_bid, tie}."""
    s = cfg['session']
    dates = session_calendar(frames, cfg).index
    out = {}
    for pair in cfg['pairs']:
        f = frames[pair].copy()
        f['mid_close'] = mid(f, points[pair])
        t = pd.DataFrame({'entry_mid': bar_at(f, dates, s['entry_hour'], 'mid_close'),
                          'exit_mid': bar_at(f, dates, s['exit_hour'], 'mid_close'),
                          'entry_bid': bar_at(f, dates, s['entry_hour'], 'close'),
                          'exit_bid': bar_at(f, dates, s['exit_hour'], 'close')})
        diff = t['exit_mid'] - t['entry_mid']
        t['tie'] = diff == 0
        t['y'] = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, np.nan))
        bd = t['exit_bid'] - t['entry_bid']
        t['y_bid'] = np.where(bd > 0, 1.0, np.where(bd < 0, 0.0, np.nan))
        out[pair] = t
    return out


def target_spans(dates, cfg: dict) -> pd.DataFrame:
    """First and last bar label each day's target covers (the bars after the entry bar)."""
    d = pd.DatetimeIndex(dates)
    s = cfg['session']
    return pd.DataFrame({'first': d + pd.Timedelta(hours=s['entry_hour'] + 1),
                         'last': d + pd.Timedelta(hours=s['exit_hour'])}, index=d)


def dataset(frames: dict, points: dict, cfg: dict) -> dict:
    """{pair: rows by date with every feature, the targets and `reason` ('' = usable)}.
    Usable = calendar-eligible, complete features, and not a tie."""
    from .features import build_features
    cal = session_calendar(frames, cfg)
    feats = build_features(frames, cfg, dates=cal.index)
    tg = build_targets(frames, points, cfg)
    out = {}
    for pair in cfg['pairs']:
        d = feats[pair].join(tg[pair])
        reason = cal['reason'].copy()
        reason[(reason == '') & (d['reason'] != '')] = d['reason']
        reason[(reason == '') & d['tie']] = 'tie'
        d['reason'] = reason
        out[pair] = d
    return out


# -- power ------------------------------------------------------------------------------------

def binary_entropy(p: float) -> float:
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def power_row(n: int, alpha: float, power: float) -> dict:
    z = NormalDist().inv_cdf
    mde = 0.5 + (z(1 - alpha) + z(power)) * 0.5 / math.sqrt(n)
    return {'n': n, 'mde_accuracy': mde, 'mde_logloss_gain': math.log(2) - binary_entropy(mde)}


# -- report -----------------------------------------------------------------------------------

def label_summary(d: pd.DataFrame, cfg: dict) -> dict:
    use = d[d['reason'] == '']
    oos = use.loc[cfg['folds']['first_eval_start']:]
    up_bid = use['y_bid'].dropna()
    return {'days': int(len(d)), 'usable': int(len(use)), 'oos': int(len(oos)),
            'ties': int(d['tie'].sum()), 'up_rate': float(use['y'].mean()),
            'up_rate_bid': float(up_bid.mean()),
            'gap_pp': 100 * abs(float(use['y'].mean()) - float(up_bid.mean())),
            'per_year': use.groupby(use.index.year)['y'].agg(['size', 'mean']),
            'excluded': d.loc[d['reason'] != '', 'reason'].value_counts(),
            'dst_days': int(use['dst_mismatch'].sum())}


def write_report(cfg: dict | None = None, now_iso: str | None = None, frames=None, points=None) -> str:
    from .data_loader import load_h1, load_points
    from .registry import assert_config_matches_verdicts
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    frames = frames or {p: load_h1(p, cfg) for p in cfg['factor_pairs']}
    points = points or load_points(cfg)
    data = dataset(frames, points, cfg)
    s, sc = cfg['session'], cfg['scoring']
    tol = cfg['target']['parity_tolerance_pp']
    L = ['# fx-session labels and power', '', f'> **{GROSS_HEADER}**', '',
         f"Target: sign of the mid close of the bar labelled {s['exit_hour']}:00 Berlin minus the mid close of the "
         f"bar labelled {s['entry_hour']}:00 Berlin, same day. Ties dropped. Evaluation from "
         f"{cfg['folds']['first_eval_start']}.", '',
         '| Pair | Weekdays | Usable days | Usable from first fold | Ties | Up-rate (mid) | Up-rate (bid) | '
         'Gap (pp) | DST-mismatch days |', '|---|---|---|---|---|---|---|---|---|']
    sums = {p: label_summary(d, cfg) for p, d in data.items()}
    for p, m in sums.items():
        flag = ' ⚠' if m['gap_pp'] > tol else ''
        L.append(f"| {p} | {m['days']:,} | {m['usable']:,} | {m['oos']:,} | {m['ties']} | {m['up_rate']:.2%} | "
                 f"{m['up_rate_bid']:.2%} | {m['gap_pp']:.2f}{flag} | {m['dst_days']} |")
    L += ['', f'A bid-versus-mid gap above {tol} pp would be flagged ⚠ as a possible spread artifact.', '',
          '## Up-rate per year (usable days)', '',
          '| Year | ' + ' | '.join(f'{p} days | {p} up' for p in sums) + ' |',
          '|---|' + '---|---|' * len(sums)]
    years = sorted(set().union(*[m['per_year'].index for m in sums.values()]))
    for y in years:
        cells = []
        for m in sums.values():
            r = m['per_year'].loc[y] if y in m['per_year'].index else None
            cells += ([f"{int(r['size'])}", f"{r['mean']:.1%}"] if r is not None else ['0', '—'])
        L.append(f'| {y} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Excluded days', '', '| Pair | Reason | Days |', '|---|---|---|']
    for p, m in sums.items():
        L += [f'| {p} | {k} | {v} |' for k, v in m['excluded'].items()]
    L += ['', '## Power — computed from day counts only, before any model', '',
          f"One-sided alpha {sc['alpha']} (0.05/4), power {sc['power']:.0%}. The minimum detectable accuracy is "
          '`0.5 + (z(1−α) + z(power)) · 0.5 / √n`; the log-loss gain is `ln 2 − H(MDE)`.', '',
          '| Pair | Out-of-sample days | Minimum detectable accuracy | Matching log-loss gain |', '|---|---|---|---|']
    for p, m in sums.items():
        r = power_row(m['oos'], sc['alpha'], sc['power'])
        L.append(f"| {p} | {r['n']:,} | {r['mde_accuracy']:.2%} | {r['mde_logloss_gain']:.5f} |")
    L += ['', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'labels.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-session label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
