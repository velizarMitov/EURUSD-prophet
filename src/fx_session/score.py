"""One-shot scoring of the four fx-session cells (spec fx-session/evaluation).

    python -m src.fx_session.score --stage lgbm    EURUSD/GBPUSD LightGBM vs the class prior
    python -m src.fx_session.score --stage mlp     EURUSD/GBPUSD MLP vs LightGBM, or NOT TRIGGERED

Statistic: mean per-day binary log-loss difference, model minus reference, over the pair's pooled
out-of-sample days in date order. Test: one-sided circular block bootstrap (block 5 days, fixed
resamples and seed, the fx_fib implementation read-only) at alpha = 0.05/4. KEEP only when the
upper bound is below zero. A cell is written once; alpha an untriggered cell does not spend is
never reassigned. GROSS predictive results, never a trading edge.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_fib.score import test_cell
from ..fx_fib.artifacts import sha256
from . import config_digest, feature_list_digest, load_config
from .artifacts import manifest_path
from .registry import (AlreadyScored, ConfigChanged, append_rows, assert_config_matches_verdicts,  # noqa: F401
                       read_registry)
from .walk_forward import StageOrderError, load_oos, load_run

PRIOR_FACTS = (
    'Earlier results, recorded as facts and not blockers: the horizon study session cells '
    '(results/horizon_study/README.md 1b) scored EURUSD direction from every in-session bar at 4 h and 6.5 h '
    'at 48.6 % and 48.8 % accuracy (coin level); the H1 direction family kept EURUSD next-hour GBM '
    '(H_dir.1) and dropped its GBPUSD replication (H_dir.3); the fx-fibonacci family dropped all four '
    'cells on 2026-11-21.')
GROSS = 'Gross predictive result, not a trading edge: spread and slippage are excluded by owner decision.'


def row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _check_run(cfg, run) -> None:
    if run['config_digest'] != config_digest(cfg) or run['feature_list_digest'] != feature_list_digest(cfg):
        raise ConfigChanged('the out-of-sample predictions were produced under a different config; '
                            'rerun the walk-forward')


def _row(cfg, n, now_iso, pair, model, ref, d, p_model, p_ref, run) -> dict:
    sc = cfg['scoring']
    y = d['y'].to_numpy(float)
    res = test_cell(row_logloss(y, p_model) - row_logloss(y, p_ref), sc['alpha'], sc['block_days'],
                    sc['resamples'], sc['seed'])
    man = manifest_path(cfg)
    return {'n': n, 'date': now_iso[:10], 'hypothesis': f'{pair}_{model}_vs_{ref}', 'pair': pair,
            'model': model, 'reference': ref, 'metric': 'binary_log_loss', 'n_days': int(len(d)),
            'n_folds': int(d['fold'].nunique()), 'eval_start': str(d['date'].min().date()),
            'eval_end': str(d['date'].max().date()), 'logloss_model': float(row_logloss(y, p_model).mean()),
            'logloss_reference': float(row_logloss(y, p_ref).mean()),
            'accuracy_model': float(np.mean((p_model > 0.5) == (y == 1))),
            **{k: res[k] for k in ('mean_diff', 'ci_upper', 'boot_se', 'mde_80', 'verdict')},
            'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': sc['resamples'], 'seed': sc['seed'],
            'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
            'artifact_manifest_sha256': sha256(man) if os.path.exists(man) else '',
            'device': run.get('device', ''),
            'notes': (f"Decided once on this family's pooled out-of-sample walk-forward days "
                      f"({d['date'].min().date()}..{d['date'].max().date()}), alpha 0.05/4. {GROSS} {PRIOR_FACTS}")}


def score_lgbm(cfg: dict | None = None, now_iso: str | None = None) -> list:
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    have = {r['hypothesis'] for r in read_registry(cfg)}
    if any(f'{p}_lightgbm_vs_prior' in have for p in cfg['pairs']):
        raise AlreadyScored('the LightGBM cells are already scored; each cell is scored once')
    run = load_run(cfg, 'lgbm')
    _check_run(cfg, run)
    oos = load_oos(cfg, 'lgbm')
    rows = []
    for i, pair in enumerate(cfg['pairs']):
        d = oos[oos['pair'] == pair].sort_values('date')
        rows.append(_row(cfg, 1 + i, now_iso, pair, 'lightgbm', 'prior', d,
                         d['p_lightgbm'].to_numpy(float), d['p_prior'].to_numpy(float), run))
    append_rows(cfg, rows)
    return rows


def score_mlp(cfg: dict | None = None, now_iso: str | None = None) -> list:
    cfg = cfg or load_config()
    assert_config_matches_verdicts(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    if any(f'{p}_lightgbm_vs_prior' not in reg for p in cfg['pairs']):
        raise StageOrderError('score the LightGBM cells first')
    if any(f'{p}_mlp_vs_lightgbm' in reg for p in cfg['pairs']):
        raise AlreadyScored('the MLP cells are already recorded; each cell is scored once')
    run = load_run(cfg, 'mlp')
    _check_run(cfg, run)
    lg = load_oos(cfg, 'lgbm')
    mlp = load_oos(cfg, 'mlp')
    rows = []
    for i, pair in enumerate(cfg['pairs']):
        n = 3 + i
        if reg[f'{pair}_lightgbm_vs_prior']['verdict'] != 'KEEP':
            rows.append({'n': n, 'date': now_iso[:10], 'hypothesis': f'{pair}_mlp_vs_lightgbm', 'pair': pair,
                         'model': 'mlp', 'reference': 'lightgbm', 'metric': 'binary_log_loss',
                         'alpha': cfg['scoring']['alpha'], 'verdict': 'NOT TRIGGERED',
                         'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
                         'notes': (f'Not trained or tested: {pair} LightGBM vs prior was DROP. Its alpha '
                                   f"{cfg['scoring']['alpha']} is not reassigned to any other cell. {GROSS}")})
            continue
        if pair not in run.get('trained_pairs', []):
            raise StageOrderError(f'{pair} LightGBM is KEEP but the MLP stage did not train it')
        a = lg[lg['pair'] == pair][['date', 'fold', 'y', 'p_lightgbm']]
        b = mlp[mlp['pair'] == pair][['date', 'p_mlp']]
        d = a.merge(b, on='date', how='inner').sort_values('date')
        if len(d) != len(a):
            raise StageOrderError(f'{pair}: MLP predictions do not cover every LightGBM day')
        rows.append(_row(cfg, n, now_iso, pair, 'mlp', 'lightgbm', d, d['p_mlp'].to_numpy(float),
                         d['p_lightgbm'].to_numpy(float), run))
    append_rows(cfg, rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='score fx-session cells once')
    ap.add_argument('--stage', choices=['lgbm', 'mlp'], required=True)
    a = ap.parse_args(argv)
    for r in (score_lgbm if a.stage == 'lgbm' else score_mlp)():
        if r['verdict'] == 'NOT TRIGGERED':
            print(f"{r['hypothesis']}: NOT TRIGGERED")
        else:
            print(f"{r['hypothesis']}: log-loss {r['logloss_model']:.5f} vs {r['logloss_reference']:.5f}, "
                  f"mean diff {r['mean_diff']:+.5f}, upper {r['ci_upper']:+.5f}, accuracy {r['accuracy_model']:.2%} "
                  f"-> {r['verdict']}")


if __name__ == '__main__':
    main()
