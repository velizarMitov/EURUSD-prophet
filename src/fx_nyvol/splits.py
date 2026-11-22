"""Expanding 6-month walk-forward folds with a 1-day embargo (spec fx-ny-volatility/evaluation; design D5).

The folds are fx_session's (read-only): fold k evaluates the eligible days in [start_k, start_k + 6
months) from 2013-01-01, on calendar boundaries shared by both pairs; training is the eligible days
before start_k minus a 1-day embargo, and the last 15 % of those (minus a 1-day embargo before it)
is the early-stopping slice. Nothing is shuffled.

`fold_arrays` adds the MLP's scaling: features and target standardised by the FIT slice's mean and
standard deviation only.
"""

from __future__ import annotations

import numpy as np

from ..fx_session.splits import Fold, fold_bounds, make_folds  # noqa: F401  (read-only reuse)


class Standardiser:
    def __init__(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean) / self.sd

    def inverse(self, z):
        return np.asarray(z, dtype=float) * self.sd + self.mean


def fold_arrays(X: np.ndarray, y: np.ndarray, fold: Fold) -> dict:
    """Raw slices plus fit-only standardised float32 slices for the MLP."""
    xs, ys = Standardiser(X[fold.fit]), Standardiser(y[fold.fit])
    out = {'x_scaler': xs, 'y_scaler': ys}
    for name in ('fit', 'val', 'eval'):
        rows = getattr(fold, name)
        out[f'X_{name}'], out[f'y_{name}'] = X[rows], y[rows]
        out[f'Xs_{name}'] = xs.transform(X[rows]).astype(np.float32)
        out[f'ys_{name}'] = ys.transform(y[rows]).astype(np.float32)
    return out
