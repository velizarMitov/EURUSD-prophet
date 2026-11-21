"""results/fx_rates/walk_forward_report.md (spec fx-rates/evaluation; design D9).

    python -m src.fx_rates.report

Built only from the saved predictions, the run logs and the registry; two builds differ only in
the generation line. Everything but the verdict table is descriptive and GROSS of cost.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_session.score import row_logloss
from . import GROSS_HEADER, load_config, repo_path
from .labeling import power_rows, sign_vif
from .registry import read_registry
from .score import wilson
from .walk_forward import OOS, load_oos, load_run, results_dir

LABEL = {'prior': 'class prior (fit slice)', 'lightgbm': 'LightGBM', 'mlp': 'MLP 5-seed ensemble'}


def _f(x, nd=4):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def weekly_rule(d: pd.DataFrame, p: np.ndarray, offset: int, h: int, pip: float) -> dict:
    """Every h-th out-of-sample day from `offset` (date order): long if P(up) > 0.5 else short,
    held h days, mid to mid. No overlap, no cost."""
    idx = np.arange(offset, len(d), h)
    side = np.where(p[idx] > 0.5, 1.0, -1.0)
    pips = side * (d['exit_mid'].to_numpy()[idx] - d['entry_mid'].to_numpy()[idx]) / pip
    wins = int((pips > 0).sum())
    lo, hi = wilson(wins, len(pips)) if len(pips) else (np.nan, np.nan)
    gains, losses = pips[pips > 0].sum(), -pips[pips < 0].sum()
    sd = pips.std(ddof=1) if len(pips) > 1 else np.nan
    return {'weeks': len(pips), 'win': wins / len(pips) if len(pips) else np.nan, 'lo': lo, 'hi': hi,
            'mean_pips': float(pips.mean()) if len(pips) else np.nan, 'total_pips': float(pips.sum()),
            'sharpe': float(pips.mean() / sd * np.sqrt(52)) if sd and sd > 0 else np.nan,
            'pf': float(gains / losses) if losses > 0 else float('inf')}


def reliability(y, p, bins=(0, .4, .45, .48, .5, .52, .55, .6, 1.0)) -> list:
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a)
        out.append((f'{a:.2f}-{b:.2f}', int(m.sum()), float(p[m].mean()) if m.any() else np.nan,
                    float(y[m].mean()) if m.any() else np.nan))
    return out


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    sc, h, pip = cfg['scoring'], int(cfg['target']['horizon_days']), float(cfg['target']['pip'])
    oos = load_oos(cfg, 'lgbm')
    run = load_run(cfg, 'lgbm')
    models = ['prior', 'lightgbm']
    if os.path.exists(os.path.join(results_dir(cfg), OOS['mlp'])):
        m = load_oos(cfg, 'mlp')
        if len(m):
            oos = oos.merge(m[['pair', 'date', 'p_mlp']], on=['pair', 'date'], how='left')
            models.append('mlp')
    reg = read_registry(cfg)
    L = ['# fx-rates walk-forward', '', f'> **{GROSS_HEADER}**', '',
         f"Stage-1 run {run['run_at']}, config digest `{run['config_digest'][:12]}`, feature list "
         f"`{run['feature_list_digest'][:12]}`. Decision at the D1 close of `t`, yields from `t−1`, target = sign of "
         f"the mid move to `t+{h}`. Out-of-sample days {oos['date'].min().date()}..{oos['date'].max().date()} in "
         f"{oos['fold'].nunique()} folds of {cfg['folds']['fold_months']} months; purge {cfg['folds']['purge_days']} + "
         f"embargo {cfg['folds']['embargo_days']} days between slices.", '', '## Verdicts', '']
    if reg:
        L += [f"alpha = {sc['alpha']} (0.05/4), one-sided circular block bootstrap, block {sc['block_days']} days, "
              f"{sc['resamples']} resamples, on the per-day log-loss difference. KEEP needs the whole interval below "
              'zero. An untriggered cell keeps its alpha unspent.', '',
              '| Cell | Days | Model log-loss | Reference log-loss | Accuracy | Wilson 95 % on n_eff | Mean diff | Upper bound | Verdict |',
              '|---|---|---|---|---|---|---|---|---|']
        for r in reg:
            if r['verdict'] == 'NOT TRIGGERED':
                L.append(f"| {r['hypothesis']} | — | — | — | — | — | — | — | **NOT TRIGGERED** |")
            else:
                L.append(f"| {r['hypothesis']} | {int(r['n_days']):,} | {float(r['logloss_model']):.5f} | "
                         f"{float(r['logloss_reference']):.5f} | {float(r['accuracy_model']):.2%} | "
                         f"{float(r['accuracy_lo']):.1%}–{float(r['accuracy_hi']):.1%} (n_eff {float(r['n_eff']):.0f}) | "
                         f"{float(r['mean_diff']):+.5f} | {float(r['ci_upper']):+.5f} | **{r['verdict']}** |")
    else:
        L.append('Not scored yet: run `python -m src.fx_rates.score --stage lgbm` once.')

    L += ['', '## Power (from day counts and the target overlap)', '',
          '| Pair | Out-of-sample days | VIF | Effective n | MDE on effective n | Weeks | MDE on weeks |',
          '|---|---|---|---|---|---|---|']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date')
        vif, _ = sign_vif(d['y'], int(sc['vif_lags']))
        r = power_rows(len(d), vif, h, sc['alpha'], sc['power'])
        L.append(f"| {pair} | {r['n_oos']:,} | {vif:.2f} | {r['n_eff']:.0f} | {r['mde_eff']:.2%} | {r['n_weeks']} | "
                 f"{r['mde_weeks']:.2%} |")

    L += ['', '## Pooled out-of-sample metrics (descriptive)', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date')
        y = d['y'].to_numpy(float)
        vif, _ = sign_vif(d['y'], int(sc['vif_lags']))
        stale = d['yield_stale_flag'].to_numpy() == 1
        L += [f'### {pair} ({len(d):,} days, up-rate {y.mean():.2%}, effective n {len(d) / vif:.0f})', '',
              '| Model | Log-loss | Brier | Accuracy | Wilson 95 % on n_eff | Accuracy on stale-yield days |',
              '|---|---|---|---|---|---|']
        for m in models:
            p = d[f'p_{m}'].to_numpy(float)
            hit = (p > 0.5) == (y == 1)
            lo, hi = wilson(hit.mean() * len(y) / vif, len(y) / vif)
            L.append(f"| {LABEL[m]} | {row_logloss(y, p).mean():.5f} | {np.mean((p - y) ** 2):.5f} | {hit.mean():.2%} | "
                     f"{lo:.1%}–{hi:.1%} | {hit[stale].mean():.1%} (n = {int(stale.sum())}) |")
        L += ['', '| Year | Days | ' + ' | '.join(f'{LABEL[m]} accuracy' for m in models) + ' |',
              '|---|---|' + '---|' * len(models)]
        for yr, g in d.groupby(d['date'].dt.year):
            yy = g['y'].to_numpy(float)
            L.append(f'| {yr} | {len(g)} | ' + ' | '.join(
                f"{np.mean((g[f'p_{m}'].to_numpy(float) > 0.5) == (yy == 1)):.1%}" for m in models) + ' |')
        L.append('')

    L += ['## Log-loss per fold', '', '| Fold | Start | ' + ' | '.join(f'{p} {m}' for p in cfg['pairs'] for m in models)
          + ' |', '|---|---|' + '---|' * (len(models) * len(cfg['pairs']))]
    for k in sorted(oos['fold'].unique()):
        cells = []
        for pair in cfg['pairs']:
            d = oos[(oos['pair'] == pair) & (oos['fold'] == k)]
            y = d['y'].to_numpy(float)
            cells += [_f(float(row_logloss(y, d[f'p_{m}'].to_numpy(float)).mean()) if len(d) else None, 5) for m in models]
        L.append(f"| {k} | {oos.loc[oos['fold'] == k, 'date'].min().date()} | " + ' | '.join(cells) + ' |')

    L += ['', '## Reliability (predicted P(up) vs observed up-rate)', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair]
        for m in [x for x in models if x != 'prior']:
            L += [f'**{pair} — {LABEL[m]}**', '', '| P(up) bin | Days | Mean P(up) | Observed up-rate |', '|---|---|---|---|']
            L += [f'| {b} | {n:,} | {_f(mp)} | {_f(o)} |' for b, n, mp, o in
                  reliability(d['y'].to_numpy(float), d[f'p_{m}'].to_numpy(float))]
            L.append('')

    L += ['## Weekly rule — gross of spread and slippage, descriptive, not a trading result', '',
          f'Every {h}th out-of-sample day from a start offset: long when P(up) > 0.5, else short; held {h} FX days, '
          'mid to mid, no overlap, no cost. Offset 0 and the mean over offsets 0–4.', '',
          '| Pair | Model | Offset | Weeks | Win rate | Wilson 95 % | Mean pips | Total pips | Sharpe (√52) | Profit factor |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date').reset_index(drop=True)
        rules = [('always long', np.ones(len(d)))] + [(LABEL[m], d[f'p_{m}'].to_numpy(float))
                                                     for m in models if m != 'prior']
        for name, p in rules:
            per = [weekly_rule(d, p, o, h, pip) for o in range(h)]
            s0 = per[0]
            L.append(f"| {pair} | {name} | 0 | {s0['weeks']} | {s0['win']:.1%} | {s0['lo']:.1%}–{s0['hi']:.1%} | "
                     f"{s0['mean_pips']:+.1f} | {s0['total_pips']:+.0f} | {_f(s0['sharpe'], 2)} | {_f(s0['pf'], 3)} |")
            mean = {k: float(np.mean([x[k] for x in per])) for k in ('win', 'mean_pips', 'total_pips', 'sharpe', 'pf')}
            L.append(f"| {pair} | {name} | mean 0–4 | {s0['weeks']} | {mean['win']:.1%} | — | {mean['mean_pips']:+.1f} | "
                     f"{mean['total_pips']:+.0f} | {_f(mean['sharpe'], 2)} | {_f(mean['pf'], 3)} |")
    L += ['', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'walk_forward_report.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-rates walk-forward report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
