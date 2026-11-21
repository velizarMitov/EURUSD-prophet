"""Chronological walk-forward folds with purge and embargo (spec fx-fibonacci/evaluation, models).

Fold k evaluates the rows labelled in [start_k, start_k + fold_months). Its training data are
the eligible rows before start_k; the chronological tail (val_fraction of those rows) is the
early-stopping slice. Between consecutive slices, measured in H1 bar positions:

    purge     a row whose horizon t+1 .. t+H reaches the next slice's first bar is dropped
    embargo   a further H bars: the last kept row's horizon ends >= H bars before that bar

so a kept row t satisfies  pos(t) + 2H <= pos(first row of the next slice).
Fold boundaries are calendar dates shared by every pair. Nothing here shuffles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    start: pd.Timestamp
    end: pd.Timestamp
    fit: np.ndarray          # positional indices into the H1 frame
    val: np.ndarray
    eval: np.ndarray


def fold_bounds(cfg: dict, last_label_time: pd.Timestamp) -> list:
    """[(start, end)] 6-month evaluation windows from first_eval_start up to the last label."""
    f = cfg['folds']
    start = pd.Timestamp(f['first_eval_start'], tz='UTC')
    out = []
    while start <= last_label_time:
        end = start + pd.DateOffset(months=f['fold_months'])
        out.append((start, end))
        start = end
    return out


def _gap_keep(rows: np.ndarray, next_first: int, horizon: int) -> np.ndarray:
    return rows[rows + 2 * horizon <= next_first]


def make_folds(index: pd.DatetimeIndex, eligible: np.ndarray, cfg: dict) -> list:
    """Folds over one pair's H1 index. `eligible` is a boolean mask of rows with a complete
    label and finite features."""
    H = cfg['labels']['horizon_bars']
    val_frac = cfg['folds']['val_fraction']
    pos = np.flatnonzero(eligible)
    times = index[pos]
    folds = []
    for k, (start, end) in enumerate(fold_bounds(cfg, times.max())):
        ev = pos[(times >= start) & (times < end)]
        if len(ev) == 0:
            continue
        before = pos[pos < ev[0]]
        before = _gap_keep(before, ev[0], H)
        n_val = int(round(len(before) * val_frac))
        if n_val == 0 or n_val >= len(before):
            continue
        val = before[-n_val:]
        fit = _gap_keep(before[:-n_val], val[0], H)
        folds.append(Fold(k, start, end, fit, val, ev))
    return folds


def fit_scaler(X: np.ndarray, fold: Fold):
    """RobustScaler fitted on the fold's fit rows only."""
    from sklearn.preprocessing import RobustScaler
    return RobustScaler().fit(X[fold.fit])


def fold_arrays(X: np.ndarray, y: np.ndarray, fold: Fold, scaler=None) -> dict:
    """The one set of scaled arrays every model of the fold receives."""
    scaler = scaler or fit_scaler(X, fold)
    out = {'scaler': scaler}
    for name in ('fit', 'val', 'eval'):
        rows = getattr(fold, name)
        out[f'X_{name}'] = scaler.transform(X[rows]).astype(np.float32)
        out[f'y_{name}'] = y[rows].astype(np.int64)
        out[f'rows_{name}'] = rows
    return out


def check_fold(f: Fold, horizon: int) -> None:
    """Assert the ordering and gap contract; raises AssertionError on any breach."""
    assert len(f.fit) and len(f.val) and len(f.eval)
    assert f.fit.max() < f.val.min() < f.val.max() < f.eval.min()
    assert f.fit.max() + 2 * horizon <= f.val.min()
    assert f.val.max() + 2 * horizon <= f.eval.min()
