"""Tuning rule (task 7.9; design D9; spec horizon-study "One pre-declared
tuning rule per cell").

  inner_tune      nested purged walk-forward INSIDE one training block over the
                  grid declared in the study record; picks the combination with
                  the best mean inner ROC-AUC. Refuses any value not in the grid.
  select_variant  the declared rule: the tuned configuration replaces the frozen
                  one only if its median net return per trade on development data
                  improves by at least `min_net_improvement_pct` AND the PBO of the
                  {frozen, tuned} pair is below `max_pbo`. Exactly one leaves.
  variant_trials  both variants as trial-log rows -- tuning is never free.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np

from . import splits as S
from .challengers import make


class UndeclaredGridValue(ValueError):
    pass


def grid_points(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def with_params(config: dict, params: dict, grid: dict) -> dict:
    """The frozen config with GBM params overridden -- only by declared values."""
    for k, v in params.items():
        if k not in grid or v not in grid[k]:
            raise UndeclaredGridValue(f'{k}={v!r} is not in the declared grid {grid.get(k)}')
    cfg = copy.deepcopy(config)
    cfg['gbm'] = {**cfg['gbm'], **params}
    return cfg


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y) & ~np.isnan(p)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float('nan')
    return float(roc_auc_score(y[m], p[m]))


def inner_tune(name: str, frozen_cfg: dict, grid: dict, inputs, train_pos, targets: dict,
               h: int, seed: int, inner_folds: int = 4) -> dict:
    train_pos = np.sort(np.asarray(train_pos))
    if not np.array_equal(train_pos, np.arange(train_pos[0], train_pos[-1] + 1)):
        raise ValueError('inner tuning needs a contiguous training block (walk-forward)')
    base = int(train_pos[0])
    n = len(train_pos)
    inner = S.walk_forward(n, h=h, n_splits=inner_folds, min_train=max(h + 1, n // 2))
    scores = []
    for params in grid_points(grid):
        cfg = with_params(frozen_cfg, params, grid)
        fold_auc = []
        for sp in inner:
            m = make(name, cfg).fit(inputs, base + sp.train, targets, seed, h)
            p = m.predict(inputs)['p_up']
            fold_auc.append(_auc(targets['dir'][base + sp.test], p[base + sp.test]))
        scores.append((float(np.nanmean(fold_auc)) if np.isfinite(fold_auc).any() else -np.inf, params))
    best_score, best = max(scores, key=lambda t: t[0])
    return {'config': with_params(frozen_cfg, best, grid), 'params': best,
            'inner_auc': best_score, 'n_candidates': len(scores)}


def select_variant(frozen_net: float, tuned_net: float, pbo: float, rule: dict) -> tuple[str, str]:
    need = rule['min_net_improvement_pct'] / 100.0 * max(abs(frozen_net), 1e-12)
    if not (np.isfinite(tuned_net) and np.isfinite(frozen_net)):
        return 'frozen', 'tuned or frozen result not finite'
    if tuned_net - frozen_net < need:
        return 'frozen', (f'improvement {tuned_net - frozen_net:.3g} below the declared '
                          f'{rule["min_net_improvement_pct"]}%')
    if not (np.isfinite(pbo) and pbo < rule['max_pbo']):
        return 'frozen', f'PBO {pbo:.3f} not below {rule["max_pbo"]}'
    return 'tuned', 'improvement and PBO both pass the declared rule'


def variant_trials(study_id: str, name: str, cadence: str, h: int, frozen_cfg: dict,
                   tuned_cfg: dict, frozen_sharpe: float, tuned_sharpe: float,
                   n_obs: int, selected: str) -> list[dict]:
    from .challengers.base import config_hash
    rows = []
    for variant, cfg, sr in (('frozen', frozen_cfg, frozen_sharpe), ('tuned', tuned_cfg, tuned_sharpe)):
        rows.append({'study_id': study_id, 'model': name, 'cadence': cadence, 'horizon': h,
                     'config_hash': config_hash(cfg), 'variant': variant, 'sharpe': sr,
                     'n_obs': n_obs, 'notes': 'SELECTED' if variant == selected else 'not selected'})
    return rows
