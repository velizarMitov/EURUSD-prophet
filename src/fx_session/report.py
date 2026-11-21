"""results/fx_session/walk_forward_report.md (spec fx-session/evaluation).

    python -m src.fx_session.report

Built only from the saved out-of-sample predictions, the run logs and the registry, so two
builds differ only in the generation line. Everything except the verdict table is descriptive.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import GROSS_HEADER, load_config, repo_path
from .labeling import power_row
from .registry import read_registry
from .score import row_logloss
from .walk_forward import OOS, load_oos, load_run, results_dir

LABEL = {'prior': 'class prior (fit slice)', 'lightgbm': 'LightGBM', 'mlp': 'MLP 5-seed ensemble'}
PIP = 0.0001


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def stats(y: np.ndarray, p: np.ndarray) -> dict:
    hit = (p > 0.5) == (y == 1)
    lo, hi = wilson(int(hit.sum()), len(y))
    return {'n': len(y), 'log_loss': float(row_logloss(y, p).mean()), 'brier': float(np.mean((p - y) ** 2)),
            'acc': float(hit.mean()) if len(y) else float('nan'), 'lo': lo, 'hi': hi}


def parity(d: pd.DataFrame, tol: float) -> dict:
    mid, bid = float(d['y'].mean()), float(d['y_bid'].dropna().mean())
    gap = 100 * abs(mid - bid)
    return {'mid': mid, 'bid': bid, 'gap': gap, 'flag': gap > tol}


def sign_rule(d: pd.DataFrame, p: np.ndarray) -> dict:
    side = np.where(p > 0.5, 1.0, -1.0)
    move = (d['exit_mid'] - d['entry_mid']).to_numpy()
    pips = side * move / PIP
    pct = side * move / d['entry_mid'].to_numpy() * 100
    gains, losses = pips[pips > 0].sum(), -pips[pips < 0].sum()
    return {'days': len(pips), 'long_share': float((side > 0).mean()), 'win': float((pips > 0).mean()),
            'mean_pips': float(pips.mean()), 'total_pips': float(pips.sum()), 'total_pct': float(pct.sum()),
            'pf': float(gains / losses) if losses > 0 else float('inf')}


def reliability(y, p, bins=(0, .4, .45, .48, .5, .52, .55, .6, 1.0)) -> list:
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a)
        out.append((f'{a:.2f}-{b:.2f}', int(m.sum()), float(p[m].mean()) if m.any() else float('nan'),
                    float(y[m].mean()) if m.any() else float('nan')))
    return out


def _f(x, nd=4):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    sc, tol = cfg['scoring'], cfg['target']['parity_tolerance_pp']
    oos = load_oos(cfg, 'lgbm')
    run = load_run(cfg, 'lgbm')
    models = ['prior', 'lightgbm']
    if os.path.exists(os.path.join(results_dir(cfg), OOS['mlp'])):
        m = load_oos(cfg, 'mlp')
        if len(m):
            oos = oos.merge(m[['pair', 'date', 'p_mlp']], on=['pair', 'date'], how='left')
            models.append('mlp')
    reg = read_registry(cfg)
    L = ['# fx-session walk-forward', '', f'> **{GROSS_HEADER}**', '',
         f"Stage-1 run {run['run_at']}, config digest `{run['config_digest'][:12]}`, feature list "
         f"`{run['feature_list_digest'][:12]}`. One forecast per day: features at the close of the bar labelled "
         f"{cfg['session']['entry_hour']}:00 Berlin (16:00 Sofia), target = sign of the mid close at "
         f"{cfg['session']['exit_hour']}:00 Berlin (23:00 Sofia). Out-of-sample days "
         f"{oos['date'].min().date()}..{oos['date'].max().date()} in {oos['fold'].nunique()} folds of "
         f"{cfg['folds']['fold_months']} months.", '', '## Verdicts', '']
    if reg:
        L += [f"alpha = {sc['alpha']} (0.05/4), one-sided circular block bootstrap, block {sc['block_days']} days, "
              f"{sc['resamples']} resamples. Difference = model − reference log-loss; KEEP needs the whole interval "
              'below zero. An untriggered cell keeps its alpha unspent.', '',
              '| Cell | Days | Model log-loss | Reference log-loss | Accuracy | Mean diff | Upper bound | Verdict |',
              '|---|---|---|---|---|---|---|---|']
        for r in reg:
            if r['verdict'] == 'NOT TRIGGERED':
                L.append(f"| {r['hypothesis']} | — | — | — | — | — | — | **NOT TRIGGERED** |")
            else:
                L.append(f"| {r['hypothesis']} | {int(r['n_days']):,} | {float(r['logloss_model']):.5f} | "
                         f"{float(r['logloss_reference']):.5f} | {float(r['accuracy_model']):.2%} | "
                         f"{float(r['mean_diff']):+.5f} | {float(r['ci_upper']):+.5f} | **{r['verdict']}** |")
    else:
        L.append('Not scored yet: run `python -m src.fx_session.score --stage lgbm` once.')
    L += ['', '## Power (from day counts only)', '', '| Pair | Out-of-sample days | Minimum detectable accuracy |',
          '|---|---|---|']
    for pair in cfg['pairs']:
        r = power_row(int((oos['pair'] == pair).sum()), sc['alpha'], sc['power'])
        L.append(f"| {pair} | {r['n']:,} | {r['mde_accuracy']:.2%} |")

    L += ['', '## Pooled out-of-sample metrics (descriptive)', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date')
        y = d['y'].to_numpy(float)
        L += [f"### {pair} ({len(d):,} days, up-rate {y.mean():.2%})", '',
              '| Model | Log-loss | Brier | Accuracy | Wilson 95 % | Accuracy on DST-mismatch days |',
              '|---|---|---|---|---|---|']
        dst = d['dst_mismatch'].to_numpy() == 1
        for m in models:
            p = d[f'p_{m}'].to_numpy(float)
            s, sd = stats(y, p), stats(y[dst], p[dst])
            L.append(f"| {LABEL[m]} | {s['log_loss']:.5f} | {s['brier']:.5f} | {s['acc']:.2%} | "
                     f"{s['lo']:.2%}–{s['hi']:.2%} | {sd['acc']:.2%} (n = {sd['n']}) |")
        L += ['', '| Year | Days | ' + ' | '.join(f'{LABEL[m]} accuracy' for m in models) + ' |',
              '|---|---|' + '---|' * len(models)]
        for yr, g in d.groupby(d['date'].dt.year):
            yy = g['y'].to_numpy(float)
            L.append(f'| {yr} | {len(g)} | ' + ' | '.join(
                f"{stats(yy, g[f'p_{m}'].to_numpy(float))['acc']:.1%}" for m in models) + ' |')
        L.append('')

    L += ['## Log-loss per fold', '', '| Fold | Start | ' + ' | '.join(
        f'{p} {m}' for p in cfg['pairs'] for m in models) + ' |', '|---|---|' + '---|' * (len(models) * len(cfg['pairs']))]
    for k in sorted(oos['fold'].unique()):
        cells = []
        for pair in cfg['pairs']:
            d = oos[(oos['pair'] == pair) & (oos['fold'] == k)]
            y = d['y'].to_numpy(float)
            cells += [_f(float(row_logloss(y, d[f'p_{m}'].to_numpy(float)).mean()) if len(d) else None, 5)
                      for m in models]
        L.append(f"| {k} | {oos.loc[oos['fold'] == k, 'date'].min().date()} | " + ' | '.join(cells) + ' |')

    L += ['', '## Reliability (predicted P(up) vs observed up-rate)', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair]
        for m in [x for x in models if x != 'prior']:
            L += [f'**{pair} — {LABEL[m]}**', '', '| P(up) bin | Days | Mean P(up) | Observed up-rate |',
                  '|---|---|---|---|']
            L += [f'| {b} | {n:,} | {_f(mp)} | {_f(o)} |' for b, n, mp, o in
                  reliability(d['y'].to_numpy(float), d[f'p_{m}'].to_numpy(float))]
            L.append('')

    L += ['## Bid-versus-mid parity', '', f'A gap above {tol} pp is flagged ⚠ as a possible spread artifact. '
          'Verdicts use the mid target only.', '', '| Pair | Up-rate (mid) | Up-rate (bid) | Gap (pp) | Flag |',
          '|---|---|---|---|---|']
    for pair in cfg['pairs']:
        pr = parity(oos[oos['pair'] == pair], tol)
        L.append(f"| {pair} | {pr['mid']:.2%} | {pr['bid']:.2%} | {pr['gap']:.2f} | {'⚠' if pr['flag'] else '—'} |")

    L += ['', '## Sign rule — gross of spread and slippage, descriptive, not a trading result', '',
          'Long at the 16:00 Sofia mid close when P(up) > 0.5, else short; exit at the 23:00 Sofia mid close. '
          'One position per pair per day, no sizing, no cost.', '',
          '| Pair | Model | Days | Long share | Win rate | Mean pips | Total pips | Total % | Profit factor |',
          '|---|---|---|---|---|---|---|---|---|']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date')
        rules = [('always long', np.ones(len(d)))] + [(LABEL[m], d[f'p_{m}'].to_numpy(float))
                                                     for m in models if m != 'prior']
        for name, p in rules:
            s = sign_rule(d, p)
            L.append(f"| {pair} | {name} | {s['days']:,} | {s['long_share']:.1%} | {s['win']:.1%} | "
                     f"{s['mean_pips']:+.2f} | {s['total_pips']:+.0f} | {s['total_pct']:+.2f} | {_f(s['pf'], 3)} |")
    L += ['', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'walk_forward_report.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-session walk-forward report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
