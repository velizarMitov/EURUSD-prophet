"""results/fx_fib/walk_forward_report.md (spec fx-fibonacci/evaluation).

    python -m src.fx_fib.report

Built only from the saved out-of-sample predictions, the run log and the registry, so two
builds differ only in the generation line.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import gross as G
from . import load_config, repo_path
from . import metrics as MT
from .score import read_registry
from .walk_forward_backtest import MODELS, RUN_FILE, load_oos

HEADER = 'Gross of spread and slippage — not a trading result.'
LABEL = {'prior': 'class prior (train fold)', 'mlp': 'MLP 5-seed ensemble (registered)',
         'lightgbm': 'LightGBM (registered baseline)', 'logistic': 'L2 logistic (descriptive)',
         'mlp_small': 'MLP 64-32 (descriptive)'}


def _p(d: pd.DataFrame, m: str) -> np.ndarray:
    return d[[f'p_{m}_{c}' for c in range(3)]].to_numpy()


def _fmt(x, nd=4):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '∞' if x == float('inf') else '—'
    return f'{x:.{nd}f}'


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    oos = load_oos(cfg)
    with open(os.path.join(repo_path(cfg['paths']['results']), RUN_FILE), encoding='utf-8') as fh:
        run = json.load(fh)
    reg = read_registry(cfg)
    L = ['# fx-fibonacci walk-forward', '', f'> **{HEADER}**', '',
         f"Run {run['run_at']} on {run.get('device', '?')} (torch {run.get('torch', '?')}, CUDA {run.get('cuda', '?')}). "
         f"Config digest `{run['config_digest'][:12]}`, feature list `{run['feature_list_digest'][:12]}`. "
         f"Out-of-sample rows {oos['time'].min().date()}..{oos['time'].max().date()} in "
         f"{oos['fold'].nunique()} folds of {cfg['folds']['fold_months']} months, purge + embargo "
         f"{cfg['labels']['horizon_bars']} bars each side.", '',
         '## Verdicts', '']
    if reg:
        L += [f"alpha = {reg[0]['alpha']} (0.05/4), one-sided circular block bootstrap, block "
              f"{reg[0]['block_bars']} bars, {reg[0]['resamples']} resamples. Difference = model − reference "
              'log-loss; KEEP needs the whole interval below zero.', '',
              '| Cell | Rows | Model log-loss | Reference log-loss | Mean diff | Upper bound | MDE (80 %) | Verdict |',
              '|---|---|---|---|---|---|---|---|']
        for r in reg:
            L.append(f"| {r['hypothesis']} | {int(r['n_rows']):,} | {float(r['logloss_model']):.5f} | "
                     f"{float(r['logloss_reference']):.5f} | {float(r['mean_diff']):+.5f} | "
                     f"{float(r['ci_upper']):+.5f} | {float(r['mde_80']):.5f} | **{r['verdict']}** |")
        notes = {r['pair']: r['notes'] for r in reg}
        L.append('')
        L += [f'- **{p}:** {n}' for p, n in notes.items()]
    else:
        L.append('Not scored yet: run `python -m src.fx_fib.score` once.')

    L += ['', '## Pooled out-of-sample metrics', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair]
        y = d['label'].to_numpy().astype(int)
        L += [f'### {pair} ({len(d):,} rows; neutral / up / down = '
              f"{np.mean(y == 0):.1%} / {np.mean(y == 1):.1%} / {np.mean(y == 2):.1%})", '',
              '| Model | Log-loss | Macro F1 | Brier | ECE |', '|---|---|---|---|---|']
        for m in ('prior',) + MODELS:
            s = MT.summary(y, _p(d, m))
            L.append(f"| {LABEL[m]} | {s['log_loss']:.5f} | {s['macro_f1']:.4f} | {s['brier']:.5f} | {s['ece']:.4f} |")
        L.append('')

    L += ['## Log-loss per fold', '', '| Fold | Start | ' + ' | '.join(
        f'{p} {m}' for p in cfg['pairs'] for m in ('prior', 'mlp', 'lightgbm')) + ' |',
          '|---|---|' + '---|' * (3 * len(cfg['pairs']))]
    for k in sorted(oos['fold'].unique()):
        cells = []
        start = oos.loc[oos['fold'] == k, 'time'].min().date()
        for pair in cfg['pairs']:
            d = oos[(oos['pair'] == pair) & (oos['fold'] == k)]
            y = d['label'].to_numpy().astype(int)
            cells += [_fmt(MT.log_loss(y, _p(d, m)) if len(d) else np.nan, 5) for m in ('prior', 'mlp', 'lightgbm')]
        L.append(f'| {k} | {start} | ' + ' | '.join(cells) + ' |')

    L += ['', '## MLP reliability (top-class confidence vs accuracy)', '']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair]
        r = MT.reliability(d['label'].to_numpy().astype(int), _p(d, 'mlp'))
        L += [f'**{pair}**', '', '| Confidence | Rows | Mean confidence | Accuracy |', '|---|---|---|---|']
        L += [f"| {b.bin} | {b.n:,} | {_fmt(b.mean_confidence)} | {_fmt(b.accuracy)} |" for b in r.itertuples()]
        L.append('')

    L += ['## Fixed rule — gross of spread and slippage, descriptive, not a trading result', '',
          'Acts when the predicted class is up or down, one position per pair, exit at the first '
          'barrier or the time limit. Percent of the entry close.', '',
          '| Pair | Model | Trades | Skipped | Win rate | Expectancy % | Profit factor | Sharpe (√252) | Max DD % | Total % |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('time').reset_index(drop=True)
        for m in ('mlp', 'lightgbm'):
            tr, skipped = G.trades(d, _p(d, m).argmax(axis=1))
            s = G.stats(tr, skipped, d['exit_day'])
            L.append(f"| {pair} | {LABEL[m]} | {s['trades']:,} | {s['skipped']:,} | {_fmt(s['win_rate'], 3)} | "
                     f"{_fmt(s['expectancy_pct'])} | {_fmt(s['profit_factor'], 3)} | {_fmt(s['sharpe'], 2)} | "
                     f"{_fmt(s['max_drawdown_pct'], 2)} | {_fmt(s['total_pct'], 2)} |")
    L += ['', f'_Generated {now_iso}._', '']
    path = os.path.join(repo_path(cfg['paths']['results']), 'walk_forward_report.md')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-fibonacci walk-forward report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
