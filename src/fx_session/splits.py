"""Expanding walk-forward folds over usable days (spec fx-session/evaluation).

Fold k evaluates the usable days in [start_k, start_k + fold_months), starting first_eval_start;
the boundaries are calendar dates shared by both pairs. Training data are the usable days before
start_k. Between slices, `embargo_days` usable days are dropped:

    ... fit ... | embargo | ... val (last val_fraction) ... | embargo | eval fold

Targets never overlap across days, so the embargo is a safety margin, not a purge.
Nothing is shuffled; the scaler is fit on the fit slice only.
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
    fit: np.ndarray          # positions into the pair's usable-day array
    val: np.ndarray
    eval: np.ndarray


def fold_bounds(cfg: dict, last_day: pd.Timestamp) -> list:
    f = cfg['folds']
    start = pd.Timestamp(f['first_eval_start'], tz='UTC')
    out = []
    while start <= last_day:
        end = start + pd.DateOffset(months=int(f['fold_months']))
        out.append((start, end))
        start = end
    return out


def make_folds(days: pd.DatetimeIndex, cfg: dict, last_day: pd.Timestamp | None = None) -> list:
    """Folds over one pair's sorted usable days. Pass the same `last_day` for both pairs."""
    days = pd.DatetimeIndex(days)
    if not days.is_monotonic_increasing:
        raise ValueError('days must be sorted')
    emb = int(cfg['folds']['embargo_days'])
    frac = float(cfg['folds']['val_fraction'])
    pos = np.arange(len(days))
    out = []
    for k, (start, end) in enumerate(fold_bounds(cfg, last_day if last_day is not None else days.max())):
        ev = pos[(days >= start) & (days < end)]
        if not len(ev):
            continue
        before = pos[days < start][:-emb] if emb else pos[days < start]
        n_val = int(round(len(before) * frac))
        if n_val == 0 or n_val + emb >= len(before):
            continue
        val = before[-n_val:]
        fit = before[:-n_val - emb] if emb else before[:-n_val]
        out.append(Fold(k, start, end, fit, val, ev))
    return out


def fit_scaler(X: np.ndarray, fold: Fold):
    from sklearn.preprocessing import RobustScaler
    return RobustScaler().fit(X[fold.fit])


def fold_arrays(X: np.ndarray, y: np.ndarray, fold: Fold, scaler=None) -> dict:
    """Raw arrays for LightGBM plus the fit-only-scaled float32 arrays for the MLP."""
    scaler = scaler or fit_scaler(X, fold)
    out = {'scaler': scaler}
    for name in ('fit', 'val', 'eval'):
        rows = getattr(fold, name)
        out[f'X_{name}'] = X[rows]
        out[f'Xs_{name}'] = scaler.transform(X[rows]).astype(np.float32)
        out[f'y_{name}'] = y[rows]
        out[f'rows_{name}'] = rows
    return out
