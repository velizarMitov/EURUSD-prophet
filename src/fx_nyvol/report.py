"""results/fx_nyvol/walk_forward_report.md (spec fx-ny-volatility/evaluation; design D8).

    python -m src.fx_nyvol.report

Built only from the saved predictions, run files, label record and registry; two builds differ only
in the generation line. Everything but the verdict table is descriptive and GROSS of cost.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import GROSS_HEADER, load_config
from .registry import LABELS_JSON, RUNS, read_json, read_registry, results_dir
from .walk_forward import OOS, load_oos

LABEL = {'har': 'HAR (reference)', 'harx': 'HAR-X (OLS, all 14 features; descriptive)', 'lightgbm': 'LightGBM',
         'mlp': 'MLP 5-seed ensemble'}


def metrics(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray) -> dict:
    mse = float(np.mean((y - p) ** 2))
    return {'mse': mse, 'mae': float(np.mean(np.abs(y - p))), 'r2': 1 - mse / float(np.var(y)),
            'ratio': mse / float(np.mean((y - p_ref) ** 2)), 'rho': float(spearmanr(y, p).statistic)}


def hot(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> dict:
    act, pred = y > thr, p > thr
    tp = int((act & pred).sum())
    return {'base': float(act.mean()), 'precision': tp / pred.sum() if pred.sum() else float('nan'),
            'recall': tp / act.sum() if act.sum() else float('nan'), 'accuracy': float((act == pred).mean()),
            'called': int(pred.sum())}


def _f(x, nd=4):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    sc = cfg['scoring']
    reg = read_registry(cfg)
    lab = read_json(cfg, LABELS_JSON)
    run = read_json(cfg, RUNS['lgbm'])
    L = ['# fx-ny-volatility walk-forward', '', f'> **{GROSS_HEADER}**', '',
         'Target: y = log(New York session mid range / mid close of the 14:00 bar), bars labelled 15:00–21:00 Berlin '
         '(16:00–23:00 Sofia); features at the close of the bar labelled 14:00 Berlin. Reference: HAR OLS on the '
         'range\'s own 1/5/20-day history, fitted per fold on the training days only.', '', '## Verdicts', '']
    if reg:
        L += [f"alpha = {sc['alpha']} (0.05/{sc['family_size']}), one-sided circular block bootstrap, block "
              f"{sc['block_days']} days, {sc['resamples']} resamples, on the per-day squared-error difference (model − "
              'reference). KEEP needs the whole interval below zero. An untriggered cell keeps its alpha unspent.', '',
              '| Cell | Days | Model MSE | Reference MSE | Relative MSE reduction | Mean diff | Upper bound | mde_80 | Verdict |',
              '|---|---|---|---|---|---|---|---|---|']
        for r in reg:
            if r['verdict'] == 'NOT TRIGGERED':
                L.append(f"| {r['hypothesis']} | — | — | — | — | — | — | — | **NOT TRIGGERED** |")
            else:
                L.append(f"| {r['hypothesis']} | {int(r['n_days']):,} | {float(r['mse_model']):.5f} | "
                         f"{float(r['mse_reference']):.5f} | {float(r['rel_mse_reduction']):+.2%} | "
                         f"{float(r['mean_diff']):+.5f} | {float(r['ci_upper']):+.5f} | {float(r['mde_80']):.5f} | "
                         f"**{r['verdict']}** |")
    else:
        L.append('Not scored yet: run `python -m src.fx_nyvol.score --stage lgbm` once.')
    L += ['', '## Power (from labels.md, before any model)', '']
    if lab:
        L += ['| Pair | Out-of-sample days | Detectable difference (σ units) |', '|---|---|---|']
        L += [f"| {p} | {r['oos']:,} | {r['mde_sd']:.4f} |" for p, r in lab['pairs'].items()]
    else:
        L.append('No label report yet.')
    if not reg or not run:
        L += ['', 'Descriptive results are shown after the LightGBM cells are scored.', '', f'_Generated {now_iso}._', '']
        return _save(cfg, L)

    oos = load_oos(cfg, 'lgbm')
    models = ['har', 'harx', 'lightgbm']
    if os.path.exists(os.path.join(results_dir(cfg), OOS['mlp'])):
        m = load_oos(cfg, 'mlp')
        if len(m):
            oos = oos.merge(m[['pair', 'date', 'p_mlp']], on=['pair', 'date'], how='left')
            models.append('mlp')
    L += ['', f"## Out-of-sample accuracy (descriptive; run {run['run_at']}, config `{run['config_digest'][:12]}`)", '',
          'R² is 1 − MSE / var(y) on the out-of-sample days. HAR-X separates "pre-session information helps" (a linear '
          'model gets it too) from "a non-linear model helps" (only LightGBM or the MLP gets it).']
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('date')
        y, ref = d['y'].to_numpy(float), d['p_har'].to_numpy(float)
        L += ['', f"### {pair} ({len(d):,} days, {d['fold'].nunique()} folds; mean y {y.mean():.4f}, SD {y.std():.4f})", '',
              '| Model | MSE | MAE | R² | MSE / HAR | Spearman ρ |', '|---|---|---|---|---|---|']
        for mname in models:
            if f'p_{mname}' not in d or d[f'p_{mname}'].isna().all():
                continue
            g = d[d[f'p_{mname}'].notna()]
            mt = metrics(g['y'].to_numpy(float), g[f'p_{mname}'].to_numpy(float), g['p_har'].to_numpy(float))
            L.append(f"| {LABEL[mname]} | {mt['mse']:.5f} | {mt['mae']:.5f} | {mt['r2']:.4f} | {mt['ratio']:.4f} | "
                     f"{mt['rho']:.4f} |")
        thr = d['hot_threshold'].to_numpy(float)
        L += ['', f"**Hot sessions** — y above its causal Q{int(sc['hot_quantile'] * 100)} (the previous "
              f"{sc['hot_window_days']} eligible days only). Share of hot days: {hot(y, y, thr)['base']:.1%}.", '',
              '| Model | Days called hot | Precision | Recall | Accuracy |', '|---|---|---|---|---|']
        for mname in models:
            if f'p_{mname}' in d and d[f'p_{mname}'].notna().all():
                h = hot(y, d[f'p_{mname}'].to_numpy(float), thr)
                L.append(f"| {LABEL[mname]} | {h['called']} | {_f(h['precision'], 3)} | {_f(h['recall'], 3)} | "
                         f"{h['accuracy']:.3f} |")
        L += ['', '| Year | Days | ' + ' | '.join(f'{LABEL[m]} MSE' for m in models if m != 'mlp') + ' |',
              '|---|---|' + '---|' * len([m for m in models if m != 'mlp'])]
        for yr, g in d.groupby(d['date'].dt.year):
            yy = g['y'].to_numpy(float)
            L.append(f'| {yr} | {len(g)} | ' + ' | '.join(
                f"{np.mean((yy - g[f'p_{m}'].to_numpy(float)) ** 2):.5f}" for m in models if m != 'mlp') + ' |')
        L += ['', '| Day type | Days | HAR MSE | LightGBM MSE | MSE / HAR |', '|---|---|---|---|---|']
        for name, mask in (('NFP', d['is_nfp_day'] == 1), ('FOMC', d['is_fomc_day'] == 1),
                           ('DST mismatch', d['dst_mismatch'] == 1),
                           ('other', (d['is_nfp_day'] == 0) & (d['is_fomc_day'] == 0) & (d['dst_mismatch'] == 0))):
            g = d[mask]
            if len(g):
                yy = g['y'].to_numpy(float)
                a = np.mean((yy - g['p_har'].to_numpy(float)) ** 2)
                b = np.mean((yy - g['p_lightgbm'].to_numpy(float)) ** 2)
                L.append(f'| {name} | {len(g)} | {a:.5f} | {b:.5f} | {b / a:.4f} |')
        yb = d['y_bid'].to_numpy(float)
        mb = np.mean((yb - d['p_lightgbm'].to_numpy(float)) ** 2) / np.mean((yb - ref) ** 2)
        L += ['', f"**Bid parity:** corr(y, y_bid) = {np.corrcoef(y, yb)[0, 1]:.4f}; LightGBM MSE / HAR MSE against the "
              f"bid target = {mb:.4f} (against the mid target {np.mean((y - d['p_lightgbm']) ** 2) / np.mean((y - ref) ** 2):.4f})."]
    L += ['', '## MSE per fold', '', '| Fold | Start | ' + ' | '.join(f'{p} HAR | {p} LightGBM' for p in cfg['pairs']) + ' |',
          '|---|---|' + '---|---|' * len(cfg['pairs'])]
    for k in sorted(oos['fold'].unique()):
        cells = []
        for pair in cfg['pairs']:
            g = oos[(oos['pair'] == pair) & (oos['fold'] == k)]
            yy = g['y'].to_numpy(float)
            cells += [_f(float(np.mean((yy - g['p_har'].to_numpy(float)) ** 2)), 5) if len(g) else '—',
                      _f(float(np.mean((yy - g['p_lightgbm'].to_numpy(float)) ** 2)), 5) if len(g) else '—']
        L.append(f"| {k} | {oos.loc[oos['fold'] == k, 'date'].min().date()} | " + ' | '.join(cells) + ' |')
    L += ['', 'Only the registry rows are verdicts; every other number is descriptive and none is a trading result.',
          '', f'_Generated {now_iso}._', '']
    return _save(cfg, L)


def _save(cfg, L) -> str:
    path = os.path.join(results_dir(cfg), 'walk_forward_report.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-ny-volatility report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
