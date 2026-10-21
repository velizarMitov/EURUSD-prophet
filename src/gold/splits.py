"""Chronological stage splits (spec "Unified chronological split"; design D8).

    validation   fit [0:70 %]   score [70:80 %]   the only arbiter
    test         fit [0:80 %]   score [80:100 %]  one shot
    final        fit [0:100 %]  no score          the frozen forward artifacts

The fractions are EUR/USD's (config.json split, mirrored in gold_config.json --
config.json itself is pinned and never read here). At h = 1 the last fit row's
target is the first scored bar, so it is purged from the fit block
(forecast_eval.splits.purged_train).
"""

from __future__ import annotations

import numpy as np

from src.forecast_eval.splits import purged_train

STAGES = ('validation', 'test', 'final')


def stage_bounds(n: int, stage: str, cfg: dict) -> tuple[int, int]:
    """(fit_end, score_end) as row counts; score is [fit_end, score_end)."""
    tr, ve = cfg['split']['train_fraction'], cfg['split']['val_end_fraction']
    if stage == 'validation':
        return int(n * tr), int(n * ve)
    if stage == 'test':
        return int(n * ve), n
    if stage == 'final':
        return n, n
    raise ValueError(f'unknown stage {stage!r}; expected one of {STAGES}')


def stage_positions(n: int, stage: str, cfg: dict, h: int = 1) -> dict:
    fit_end, score_end = stage_bounds(n, stage, cfg)
    score = np.arange(fit_end, score_end)
    if score.size:
        fit = purged_train(n, score, h, candidates=np.arange(fit_end))
    else:
        fit = np.arange(max(0, n - h))           # the last h rows have no target
    return {'stage': stage, 'n': n, 'fit': fit, 'score': score,
            'fit_end': fit_end, 'score_end': score_end}
