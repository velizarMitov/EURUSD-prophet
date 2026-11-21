"""The 5-day sign target, the label report and the overlap-aware power table
(spec fx-rates/weekly-target).

For day t: y = 1 if mid_close[t+5] > mid_close[t], 0 if below (t+5 counted in FX trading days,
i.e. D1 rows); equal -> tie, no target; no D1 bar 5 rows ahead -> no target. GROSS of cost.

Overlapping 5-day targets are dependent. The power table uses the effective size n / VIF with
VIF = 1 + 2 * sum(rho_k, k = 1..4) from the target's own sign autocorrelation on TRAINING-ERA
days (before the first fold), and the non-overlapping size n / 5.

    python -m src.fx_rates.labeling      write results/fx_rates/labels.md
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


def build_target(d1: pd.DataFrame, horizon: int) -> pd.DataFrame:
    c = d1['mid_close']
    fwd = c.shift(-horizon)
    diff = fwd - c
    t = pd.DataFrame({'entry_mid': c, 'exit_mid': fwd, 'exit_date': pd.Series(d1.index, index=d1.index).shift(-horizon)})
    t['tie'] = diff == 0
    t['y'] = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, np.nan))
    return t


def dataset(d1: dict, yields: dict, cfg: dict) -> dict:
    """{pair: features + target by date, `reason` '' when usable}."""
    from .features import build_features
    feats = build_features(d1, yields, cfg)
    out = {}
    for pair in cfg['pairs']:
        d = feats[pair].join(build_target(d1[pair], int(cfg['target']['horizon_days'])))
        reason = d['reason'].copy()
        reason[(reason == '') & d['tie']] = 'tie'
        reason[(reason == '') & d['exit_mid'].isna()] = 'no target yet'
        d['reason'] = reason
        out[pair] = d
    return out


# -- power ------------------------------------------------------------------------------------

def binary_entropy(p: float) -> float:
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def mde(n: float, alpha: float, power: float) -> float:
    z = NormalDist().inv_cdf
    return 0.5 + (z(1 - alpha) + z(power)) * 0.5 / math.sqrt(n)


def sign_vif(y: pd.Series, lags: int) -> tuple:
    s = 2 * pd.Series(y).dropna() - 1
    rho = [float(s.autocorr(k)) for k in range(1, lags + 1)]
    return 1 + 2 * sum(rho), rho


def power_rows(n_oos: int, vif: float, horizon: int, alpha: float, power: float) -> dict:
    n_eff = n_oos / vif
    n_weeks = n_oos // horizon
    return {'n_oos': n_oos, 'vif': vif, 'n_eff': n_eff, 'mde_eff': mde(n_eff, alpha, power),
            'n_weeks': n_weeks, 'mde_weeks': mde(n_weeks, alpha, power),
            'logloss_gain_eff': math.log(2) - binary_entropy(mde(n_eff, alpha, power))}


# -- report -----------------------------------------------------------------------------------

def write_report(cfg: dict | None = None, now_iso: str | None = None, d1=None, yields=None) -> str:
    from .align import load_fx_d1, load_yields
    from .registry import assert_config_matches_verdicts
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    d1 = d1 if d1 is not None else load_fx_d1(cfg)
    yields = yields if yields is not None else load_yields(cfg)
    data = dataset(d1, yields, cfg)
    sc, h = cfg['scoring'], int(cfg['target']['horizon_days'])
    start = pd.Timestamp(cfg['folds']['first_eval_start'])
    L = ['# fx-rates labels and power', '', f'> **{GROSS_HEADER}**', '',
         f'Target: sign of the D1 mid close {h} FX trading days after `t` minus the mid close of `t`. Yields enter '
         f"as of the previous FX trading day (fill limit {cfg['alignment']['fill_limit_business_days']} business days). "
         f"Evaluation from {cfg['folds']['first_eval_start']}.", '',
         '## Sources', '', '| Series | First | Last | Rows |', '|---|---|---|---|']
    for name, s in yields.items():
        L.append(f'| {name} | {s.index.min().date()} | {s.index.max().date()} | {len(s):,} |')
    L += ['', '## Days', '',
          '| Pair | D1 days | Usable | Usable from first fold | Ties | Up-rate | Stale-filled usable days | Excluded as stale yields |',
          '|---|---|---|---|---|---|---|---|']
    sums = {}
    for pair, d in data.items():
        use = d[d['reason'] == '']
        oos = use[use.index >= start]
        train = use[use.index < start]
        vif, rho = sign_vif(train['y'], int(sc['vif_lags']))
        sums[pair] = (use, oos, vif, rho, d)
        L.append(f"| {pair} | {len(d):,} | {len(use):,} | {len(oos):,} | {int(d['tie'].sum())} | {use['y'].mean():.2%} | "
                 f"{int(use['yield_stale_flag'].sum())} | {int((d['reason'] == 'stale yields').sum())} |")
    L += ['', '## Up-rate per year (usable days)', '', '| Year | ' + ' | '.join(f'{p} days | {p} up' for p in sums) + ' |',
          '|---|' + '---|---|' * len(sums)]
    years = sorted(set().union(*[set(v[0].index.year) for v in sums.values()]))
    for yr in years:
        cells = []
        for use, *_ in sums.values():
            g = use[use.index.year == yr]['y']
            cells += [f'{len(g)}', f'{g.mean():.1%}' if len(g) else '—']
        L.append(f'| {yr} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Excluded days', '', '| Pair | Reason | Days |', '|---|---|---|']
    for pair, (*_, d) in sums.items():
        L += [f'| {pair} | {k} | {v} |' for k, v in d.loc[d['reason'] != '', 'reason'].value_counts().items()]
    L += ['', '## Power — from day counts and the target\'s own overlap, before any model', '',
          f"One-sided alpha {sc['alpha']} (0.05/4), power {sc['power']:.0%}. Overlapping {h}-day targets are "
          'dependent: VIF = 1 + 2·Σρ_k over the target sign autocorrelation at lags 1–'
          f"{sc['vif_lags']} on training-era days (before {cfg['folds']['first_eval_start']}). "
          'Minimum detectable accuracy = 0.5 + (z(1−α) + z(power)) · 0.5 / √n.', '',
          '| Pair | Out-of-sample days | ρ lags 1–4 | VIF | Effective n | MDE on effective n | Independent weeks | MDE on weeks | Log-loss gain at MDE |',
          '|---|---|---|---|---|---|---|---|---|']
    for pair, (use, oos, vif, rho, d) in sums.items():
        r = power_rows(len(oos), vif, h, sc['alpha'], sc['power'])
        L.append(f"| {pair} | {r['n_oos']:,} | {', '.join(f'{x:.2f}' for x in rho)} | {vif:.2f} | {r['n_eff']:.0f} | "
                 f"{r['mde_eff']:.2%} | {r['n_weeks']} | {r['mde_weeks']:.2%} | {r['logloss_gain_eff']:.5f} |")
    L += ['', 'Only an edge of about this size can be seen. A DROP means "no edge of this size", not "no edge".',
          '', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'labels.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-rates label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
