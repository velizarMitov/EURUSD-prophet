"""One-shot scoring of the fx-fibonacci hypothesis family (spec fx-fibonacci/evaluation).

Four cells, decided on the pooled out-of-sample walk-forward rows (owner decision 2026-11-21):
    <PAIR>_mlp_vs_prior      MLP ensemble log-loss below the train-fold class-prior's
    <PAIR>_mlp_vs_lightgbm   MLP ensemble log-loss below LightGBM's
Statistic: mean per-row log-loss difference (model - reference). Test: one-sided circular
block bootstrap (block H bars, fixed resamples and seed) at alpha = 0.05/4. KEEP only when the
upper bound of the interval is below zero; DROP otherwise. Each cell is scored once.
GROSS predictive results, never a trading edge.

    python -m src.fx_fib.score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config_digest, feature_list_digest, load_config, repo_path
from .artifacts import manifest_path, sha256
from .metrics import row_logloss

REGISTRY = 'fx_fib_hypothesis_log.csv'
FIELDS = ['n', 'date', 'hypothesis', 'pair', 'model', 'reference', 'metric', 'n_rows', 'n_folds',
          'eval_start', 'eval_end', 'logloss_model', 'logloss_reference', 'mean_diff', 'ci_upper',
          'alpha', 'block_bars', 'resamples', 'seed', 'boot_se', 'mde_80', 'verdict',
          'config_digest', 'feature_list_digest', 'artifact_manifest_sha256', 'device', 'notes']
CELLS = (('mlp', 'prior'), ('mlp', 'lightgbm'))

PRIOR_FACTS = {
    'EURUSD': ('Its 2024-11-19..2026-07-28 rows overlap the H1 direction family block read once on '
               '2026-07-30 (results/h1_direction_hypothesis_log.csv), and daily EURUSD Fibonacci '
               'features were rejected as hypothesis #7 (results/feature_hypothesis_log.csv).'),
    'GBPUSD': ('GBPUSD H1 history was used by the H1 direction family replication H_dir.3 (DROP, '
               'results/h1_direction_hypothesis_log.csv).'),
}


class AlreadyScored(RuntimeError):
    pass


class ConfigChanged(RuntimeError):
    pass


def registry_path(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['results']), REGISTRY)


def read_registry(cfg: dict) -> list:
    path = registry_path(cfg)
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def assert_config_matches_verdicts(cfg: dict) -> None:
    """Refuse any label, training or scoring stage once verdicts exist under another config."""
    rows = read_registry(cfg)
    if not rows:
        return
    if {r['config_digest'] for r in rows} != {config_digest(cfg)} or \
            {r['feature_list_digest'] for r in rows} != {feature_list_digest(cfg)}:
        raise ConfigChanged('the family config differs from the one recorded with the verdicts in '
                            f'{REGISTRY}; a changed config is a new hypothesis that enlarges the family')


def circular_block_means(x: np.ndarray, block: int, resamples: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the mean under the circular block bootstrap."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = max(1, min(block, n))
    ext = np.r_[x, x[:block]]
    c = np.r_[0.0, np.cumsum(ext)]
    full = c[np.arange(n) + block] - c[np.arange(n)]            # every circular block sum
    k, r = divmod(n, block)
    part = c[np.arange(n) + r] - c[np.arange(n)]                 # partial block of length r
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for i0 in range(0, resamples, 500):
        m = min(500, resamples - i0)
        s = full[rng.integers(0, n, size=(m, k))].sum(axis=1)
        if r:
            s += part[rng.integers(0, n, size=m)]
        out[i0:i0 + m] = s / n
    return out


def test_cell(diff: np.ndarray, alpha: float, block: int, resamples: int, seed: int) -> dict:
    boots = circular_block_means(diff, block, resamples, seed)
    upper = float(np.quantile(boots, 1 - alpha))
    se = float(boots.std(ddof=1))
    z = NormalDist().inv_cdf
    return {'mean_diff': float(np.mean(diff)), 'ci_upper': upper, 'boot_se': se,
            'mde_80': float((z(1 - alpha) + z(0.80)) * se),
            'verdict': 'KEEP' if upper < 0 else 'DROP'}


def score_all(cfg: dict | None = None, now_iso: str | None = None, oos: pd.DataFrame | None = None,
              run: dict | None = None) -> list:
    from .walk_forward_backtest import RUN_FILE, load_oos
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    if read_registry(cfg):
        raise AlreadyScored(f'{REGISTRY} already holds verdicts; each cell is scored once')
    if run is None:
        with open(os.path.join(repo_path(cfg['paths']['results']), RUN_FILE), encoding='utf-8') as fh:
            run = json.load(fh)
    if run['config_digest'] != config_digest(cfg) or run['feature_list_digest'] != feature_list_digest(cfg):
        raise ConfigChanged('the out-of-sample predictions were produced under a different config; rerun the walk-forward')
    oos = load_oos(cfg) if oos is None else oos
    sc = cfg['scoring']
    man = manifest_path(cfg)
    man_sha = sha256(man) if os.path.exists(man) else ''
    rows, n = [], 0
    for pair in cfg['pairs']:
        d = oos[oos['pair'] == pair].sort_values('time')
        y = d['label'].to_numpy().astype(int)
        ll = {m: row_logloss(y, d[[f'p_{m}_{c}' for c in range(3)]].to_numpy()) for m in ('mlp', 'prior', 'lightgbm')}
        for model, ref in CELLS:
            n += 1
            res = test_cell(ll[model] - ll[ref], sc['alpha'], sc['block_bars'], sc['resamples'], sc['seed'])
            notes = (f"Decided once on this family's pooled out-of-sample walk-forward rows "
                     f"({d['time'].min().date()}..{d['time'].max().date()}), owner decision 2026-11-21. "
                     f"Gross predictive result, not a trading edge: spread and slippage are excluded by owner "
                     f"decision. {PRIOR_FACTS.get(pair, '')} Recorded as a fact, not a blocker.")
            rows.append({'n': n, 'date': now_iso[:10], 'hypothesis': f'{pair}_{model}_vs_{ref}', 'pair': pair,
                         'model': 'mlp_5seed_ensemble', 'reference': ref, 'metric': 'log_loss',
                         'n_rows': int(len(d)), 'n_folds': int(d['fold'].nunique()),
                         'eval_start': str(d['time'].min()), 'eval_end': str(d['time'].max()),
                         'logloss_model': float(ll[model].mean()), 'logloss_reference': float(ll[ref].mean()),
                         **{k: res[k] for k in ('mean_diff', 'ci_upper', 'boot_se', 'mde_80', 'verdict')},
                         'alpha': sc['alpha'], 'block_bars': sc['block_bars'], 'resamples': sc['resamples'],
                         'seed': sc['seed'], 'config_digest': config_digest(cfg),
                         'feature_list_digest': feature_list_digest(cfg), 'artifact_manifest_sha256': man_sha,
                         'device': run.get('device', ''), 'notes': notes})
    path = registry_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'x', newline='', encoding='utf-8') as fh:     # 'x': never overwrite a registry
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None):
    argparse.ArgumentParser(description='score the fx-fibonacci hypothesis family once').parse_args(argv)
    for r in score_all():
        print(f"{r['hypothesis']}: mean diff {r['mean_diff']:+.5f}, upper {r['ci_upper']:+.5f} -> {r['verdict']}")


if __name__ == '__main__':
    main()
