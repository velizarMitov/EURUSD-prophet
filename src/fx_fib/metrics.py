"""Out-of-sample classification metrics (spec fx-fibonacci/evaluation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EPS = 1e-15
N_CLASSES = 3


def row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y.astype(int)], EPS, 1.0))


def log_loss(y, p) -> float:
    return float(row_logloss(np.asarray(y), np.asarray(p)).mean())


def macro_f1(y, p) -> float:
    return float(f1_score(np.asarray(y).astype(int), np.asarray(p).argmax(axis=1),
                          labels=list(range(N_CLASSES)), average='macro', zero_division=0))


def brier(y, p) -> float:
    y = np.asarray(y).astype(int)
    onehot = np.eye(N_CLASSES)[y]
    return float(((np.asarray(p) - onehot) ** 2).sum(axis=1).mean())


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Top-class confidence vs accuracy in equal-width confidence bins."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        rows.append({'bin': f'{edges[i]:.1f}-{edges[i + 1]:.1f}', 'n': int(m.sum()),
                     'mean_confidence': float(conf[m].mean()) if m.any() else np.nan,
                     'accuracy': float(correct[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(y, p, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    r = r[r['n'] > 0]
    return float((r['n'] * (r['accuracy'] - r['mean_confidence']).abs()).sum() / r['n'].sum())


def prior_proba(y_fit, n_rows: int) -> np.ndarray:
    """The class-prior baseline: train-fold class frequencies for every row."""
    freq = np.bincount(np.asarray(y_fit).astype(int), minlength=N_CLASSES).astype(float)
    freq = np.clip(freq / freq.sum(), EPS, 1.0)
    return np.tile(freq / freq.sum(), (n_rows, 1))


def summary(y, p) -> dict:
    return {'n': int(len(y)), 'log_loss': log_loss(y, p), 'macro_f1': macro_f1(y, p),
            'brier': brier(y, p), 'ece': ece(y, p)}
