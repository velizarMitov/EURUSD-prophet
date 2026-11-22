"""One-shot scoring of the four fx-ny-volatility cells (spec fx-ny-volatility/evaluation; design D7).

    python -m src.fx_nyvol.score --stage lgbm    EURUSD/GBPUSD LightGBM vs HAR
    python -m src.fx_nyvol.score --stage mlp     EURUSD/GBPUSD MLP vs LightGBM, or NOT TRIGGERED

Statistic: the per-day squared-error difference (model minus reference) over the pair's out-of-sample
days in date order; one-sided circular block bootstrap, 5-day blocks, 10 000 resamples, fixed seed,
alpha 0.05/4 (fx_fib's test_cell, read-only). KEEP only when the upper bound is below zero. An
untriggered MLP cell keeps its alpha unspent. GROSS predictive results, never a trading result.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np

from ..fx_fib.artifacts import sha256
from ..fx_fib.score import test_cell
from ..fx_session.artifacts import manifest_path
from . import config_digest, feature_list_digest, load_config
from .registry import (RUNS, AlreadyScored, ConfigChanged, StageOrderError, append_rows, guard,  # noqa: F401
                       read_json, read_registry)

GROSS = 'Gross of spread and slippage, not a trading result.'
PRIOR_FACTS = ('Earlier results, recorded as facts and not blockers: fx-session-ny forecast the DIRECTION of this same '
               '16:00-23:00 Sofia session and dropped both LightGBM cells (MLP not triggered) on 2026-11-21; '
               'fx-fibonacci, fx-rates, fx-bb-reversion and fx-bb-sma-exit were all DROP. The production volatility '
               'family (daily, results/volatility_hypothesis_log.csv) is separate and untouched.')


def _check_run(cfg, stage) -> dict:
    run = read_json(cfg, RUNS[stage])
    if run is None:
        raise StageOrderError(f'run the {stage} walk-forward stage first')
    if run['config_digest'] != config_digest(cfg) or run['feature_list_digest'] != feature_list_digest(cfg):
        raise ConfigChanged('the predictions were produced under a different config')
    return run


def _row(cfg, n, now_iso, pair, model, ref, d, p_model, p_ref, run) -> dict:
    sc = cfg['scoring']
    y = d['y'].to_numpy(float)
    se_m, se_r = (y - p_model) ** 2, (y - p_ref) ** 2
    res = test_cell(se_m - se_r, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed'])
    man = manifest_path(cfg)
    return {'n': n, 'date': now_iso[:10], 'hypothesis': f'{pair}_{model}_vs_{ref}', 'pair': pair, 'model': model,
            'reference': ref, 'metric': 'squared_error_log_range', 'n_days': int(len(d)),
            'n_folds': int(d['fold'].nunique()), 'eval_start': str(d['date'].min().date()),
            'eval_end': str(d['date'].max().date()), 'mse_model': float(se_m.mean()), 'mse_reference': float(se_r.mean()),
            'rel_mse_reduction': float(1 - se_m.mean() / se_r.mean()),
            **{k: res[k] for k in ('mean_diff', 'ci_upper', 'boot_se', 'mde_80', 'verdict')},
            'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': sc['resamples'], 'seed': sc['seed'],
            'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
            'artifact_manifest_sha256': sha256(man) if os.path.exists(man) else '', 'device': run.get('device', ''),
            'notes': (f"Decided once on this pair's out-of-sample walk-forward days "
                      f"({d['date'].min().date()}..{d['date'].max().date()}), target log(NY mid range / close14), "
                      f"squared error, alpha 0.05/4, 5-day blocks. {GROSS} {PRIOR_FACTS}")}


def score_lgbm(cfg: dict | None = None, now_iso: str | None = None) -> list:
    from .walk_forward import load_oos
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    have = {r['hypothesis'] for r in read_registry(cfg)}
    if any(f'{p}_lightgbm_vs_har' in have for p in cfg['pairs']):
        raise AlreadyScored('the LightGBM cells are already scored; each cell is scored once')
    run = _check_run(cfg, 'lgbm')
    oos = load_oos(cfg, 'lgbm')
    rows = []
    for i, pair in enumerate(cfg['pairs']):
        d = oos[oos['pair'] == pair].sort_values('date')
        rows.append(_row(cfg, 1 + i, now_iso, pair, 'lightgbm', 'har', d, d['p_lightgbm'].to_numpy(float),
                         d['p_har'].to_numpy(float), run))
    append_rows(cfg, rows)
    return rows


def score_mlp(cfg: dict | None = None, now_iso: str | None = None) -> list:
    from .walk_forward import load_oos
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    if any(f'{p}_lightgbm_vs_har' not in reg for p in cfg['pairs']):
        raise StageOrderError('score the LightGBM cells first')
    if any(f'{p}_mlp_vs_lightgbm' in reg for p in cfg['pairs']):
        raise AlreadyScored('the MLP cells are already recorded; each cell is scored once')
    run = _check_run(cfg, 'mlp')
    lg = load_oos(cfg, 'lgbm')
    mlp = load_oos(cfg, 'mlp')
    rows = []
    for i, pair in enumerate(cfg['pairs']):
        n = 3 + i
        if reg[f'{pair}_lightgbm_vs_har']['verdict'] != 'KEEP':
            rows.append({'n': n, 'date': now_iso[:10], 'hypothesis': f'{pair}_mlp_vs_lightgbm', 'pair': pair,
                         'model': 'mlp', 'reference': 'lightgbm', 'metric': 'squared_error_log_range',
                         'alpha': cfg['scoring']['alpha'], 'verdict': 'NOT TRIGGERED',
                         'config_digest': config_digest(cfg), 'feature_list_digest': feature_list_digest(cfg),
                         'notes': (f'Not trained or tested: {pair} LightGBM vs HAR was DROP. Its alpha '
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
    ap = argparse.ArgumentParser(description='score fx-ny-volatility cells once')
    ap.add_argument('--stage', choices=['lgbm', 'mlp'], required=True)
    a = ap.parse_args(argv)
    for r in (score_lgbm if a.stage == 'lgbm' else score_mlp)():
        if r['verdict'] == 'NOT TRIGGERED':
            print(f"{r['hypothesis']}: NOT TRIGGERED")
        else:
            print(f"{r['hypothesis']}: MSE {r['mse_model']:.5f} vs {r['mse_reference']:.5f} "
                  f"({r['rel_mse_reduction']:+.2%}), mean diff {r['mean_diff']:+.5f}, upper {r['ci_upper']:+.5f}, "
                  f"mde_80 {r['mde_80']:.5f} -> {r['verdict']}")


if __name__ == '__main__':
    main()
