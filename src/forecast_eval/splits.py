"""Leakage-free splitting for overlapping horizon labels (design D4).

Positions are integer row numbers 0..n-1 on one contiguous bar grid. A label at
row i with horizon h is determined by close[i+h]; for leakage it is treated as
occupying the CLOSED window [i, i+h]. Two labels can leak into each other when
those windows intersect, i.e. when |i - j| <= h. Purge therefore removes h rows
on each side of every test run -- one row more than the information span
strictly needs, which only ever errs on the safe side.

Embargo extends the exclusion after each test run to max(h, embargo) rows, so
that serially correlated features right after a test block cannot carry it back.

Label uniqueness uses the information span [i, i+h-1] (h bars), the convention
of src/h1_horizon_feasibility.uniqueness_from_spans, so h = 1 gives exactly 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    h: int
    embargo: int
    test_groups: tuple = field(default_factory=tuple)


def _runs(sorted_idx: np.ndarray):
    """Contiguous runs [(start, end), ...] of a sorted integer array."""
    if sorted_idx.size == 0:
        return []
    breaks = np.where(np.diff(sorted_idx) != 1)[0]
    starts = np.concatenate([[sorted_idx[0]], sorted_idx[breaks + 1]])
    ends = np.concatenate([sorted_idx[breaks], [sorted_idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def purged_train(n: int, test_idx, h: int, embargo: int = 0,
                 candidates=None) -> np.ndarray:
    """Training positions left after removing test rows, h rows before each
    test run (purge) and max(h, embargo) rows after it (purge + embargo)."""
    if h < 1:
        raise ValueError('h must be >= 1')
    test_idx = np.unique(np.asarray(test_idx, dtype=np.int64))
    keep = np.ones(n, dtype=bool) if candidates is None else np.isin(np.arange(n), candidates)
    keep[test_idx] = False
    after = max(h, int(embargo))
    for a, b in _runs(test_idx):
        keep[max(0, a - h):a] = False
        keep[b + 1:min(n, b + 1 + after)] = False
    return np.flatnonzero(keep)


def walk_forward(n: int, h: int, n_splits: int, min_train: int,
                 embargo: int = 0) -> list[Split]:
    """Expanding-window purged walk-forward: training is everything strictly
    before the test block minus the purge; the remainder after min_train is cut
    into n_splits contiguous test blocks. Training never uses rows after its
    own test block, so the embargo is satisfied trivially and is recorded."""
    if min_train <= h or min_train >= n:
        raise ValueError('min_train must exceed h and be smaller than n')
    blocks = np.array_split(np.arange(min_train, n), n_splits)
    out = []
    for blk in blocks:
        if blk.size == 0:
            continue
        a = int(blk[0])
        train = np.arange(0, max(0, a - h))
        out.append(Split(train=train, test=blk, h=h, embargo=max(h, int(embargo))))
    return out


@dataclass
class CPCV:
    n: int
    n_groups: int
    k: int
    h: int
    splits: list
    paths: list          # paths[p][g] -> index into splits of the split testing group g
    groups: list

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def assemble(self, split_predictions) -> np.ndarray:
        """split_predictions[s] = predictions for splits[s].test (same order).
        Returns an (n_paths, n) array: one complete out-of-sample series per path."""
        out = np.full((self.n_paths, self.n), np.nan)
        for p, mapping in enumerate(self.paths):
            for g, s in mapping.items():
                sp = self.splits[s]
                pred = np.asarray(split_predictions[s], dtype=float)
                pos = {int(t): i for i, t in enumerate(sp.test)}
                rows = self.groups[g]
                out[p, rows] = pred[[pos[int(r)] for r in rows]]
        return out


def cpcv(n: int, h: int, n_groups: int = 6, k: int = 2, embargo: int = 0) -> CPCV:
    """Combinatorial purged cross-validation (Lopez de Prado). C(N, k) splits;
    C(N-1, k-1) paths, each covering every observation exactly once out of
    sample."""
    if not 1 <= k < n_groups:
        raise ValueError('need 1 <= k < n_groups')
    groups = [g for g in np.array_split(np.arange(n), n_groups)]
    splits = []
    for combo in combinations(range(n_groups), k):
        test = np.concatenate([groups[g] for g in combo])
        splits.append(Split(train=purged_train(n, test, h, embargo), test=test, h=h,
                            embargo=max(h, int(embargo)), test_groups=combo))
    testing = {g: [s for s, sp in enumerate(splits) if g in sp.test_groups]
               for g in range(n_groups)}
    n_paths = comb(n_groups - 1, k - 1)
    paths = [{g: testing[g][p] for g in range(n_groups)} for p in range(n_paths)]
    return CPCV(n=n, n_groups=n_groups, k=k, h=h, splits=splits, paths=paths, groups=groups)


def uniqueness_weights(label_positions, h: int, grid_len: int) -> np.ndarray:
    """Per-label average uniqueness over its information span [i, i+h-1]."""
    starts = np.asarray(label_positions, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0)
    ends = np.minimum(starts + h - 1, grid_len - 1)
    conc = np.zeros(grid_len + 1)
    np.add.at(conc, starts, 1.0)
    np.add.at(conc, ends + 1, -1.0)
    conc = np.cumsum(conc)[:grid_len]
    conc[conc == 0] = 1.0
    csum = np.concatenate([[0.0], np.cumsum(1.0 / conc)])
    return (csum[ends + 1] - csum[starts]) / (ends - starts + 1)


def effective_n(label_positions, h: int, grid_len: int) -> float:
    """Sum of uniqueness weights: the number of independent labels the rows are
    worth. Always reported next to the raw row count."""
    return float(uniqueness_weights(label_positions, h, grid_len).sum())
