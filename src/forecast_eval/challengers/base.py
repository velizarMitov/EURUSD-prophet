"""Common interface for every challenger (design D1).

A challenger is (inputs builder, fit, predict, save/load). The study and the
forward logger only ever talk to this interface, so adding a model type never
touches the scoring or the logging.

  Inputs    rows on one time index: `X` (2-D flat or 3-D sequences), the aligned
            `close`, and the row timestamps. `take(pos)` slices by position.
  fit       receives the FULL inputs plus the training positions, so sequence
            models can build look-back windows that end at a training row;
            targets is a dict of full-length arrays ('dir', 'ret', 'vol').
  predict   returns a dict of FULL-length arrays: always `p_up` (NaN where the
            model is volatility-only); `ret_pct` where it forecasts returns;
            `vol_pct` where it forecasts volatility. Rows without enough
            history are NaN.

Every challenger runs on CPU; GPU is never required (design Goals).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class ChallengerUnavailable(RuntimeError):
    """The model cannot run here (missing optional dependency or weights).
    Callers record a failure for that cell and carry on."""


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Inputs:
    index: pd.DatetimeIndex
    close: np.ndarray
    X: np.ndarray
    extra: dict = field(default_factory=dict)     # e.g. day-of-week, second tensor

    def __len__(self):
        return len(self.index)

    def take(self, pos) -> 'Inputs':
        pos = np.asarray(pos)
        return Inputs(index=self.index[pos], close=self.close[pos], X=self.X[pos],
                      extra={k: v[pos] for k, v in self.extra.items()})


class Challenger:
    """Subclasses set `name`, `cadence` ('H1' | 'D1'), `kind` ('direction' |
    'volatility') and implement build_inputs / fit / predict / _save / _load."""

    name = 'base'
    cadence = 'D1'
    kind = 'direction'
    produces_returns = False

    def __init__(self, config: dict):
        self.config = dict(config)
        self.fitted = False

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    # -- to implement --------------------------------------------------------
    def build_inputs(self, source) -> Inputs:
        raise NotImplementedError

    def fit(self, inputs: Inputs, train_pos: np.ndarray, targets: dict, seed: int,
            h: int) -> 'Challenger':
        """Learn only from rows in train_pos (scalers/decompositions included).
        targets['dir'] is 0/1 with NaN on zero moves and the tail; rows whose
        needed target is NaN are skipped."""
        raise NotImplementedError

    def predict(self, inputs: Inputs) -> dict:
        raise NotImplementedError

    def _save(self, path: str) -> None:
        raise NotImplementedError

    def _load(self, path: str) -> None:
        raise NotImplementedError

    # -- shared --------------------------------------------------------------
    def save(self, path: str) -> list[str]:
        """Write the fitted model under `path`; return the files written."""
        os.makedirs(path, exist_ok=True)
        before = set(_walk(path))
        self._save(path)
        with open(os.path.join(path, 'challenger.json'), 'w', encoding='utf-8') as fh:
            json.dump({'name': self.name, 'config': self.config,
                       'config_hash': self.config_hash}, fh, indent=1, sort_keys=True)
        return sorted(set(_walk(path)) - before | {os.path.join(path, 'challenger.json')})

    def load(self, path: str) -> 'Challenger':
        with open(os.path.join(path, 'challenger.json'), encoding='utf-8') as fh:
            meta = json.load(fh)
        if meta['config_hash'] != self.config_hash:
            raise ValueError(f'{self.name}: saved config {meta["config_hash"]} != declared {self.config_hash}')
        self._load(path)
        self.fitted = True
        return self


def _walk(path):
    for root, _dirs, files in os.walk(path):
        for f in files:
            yield os.path.join(root, f)


def labelled(y: np.ndarray) -> np.ndarray:
    """Positions with a defined label (zero moves and the tail are NaN)."""
    return np.flatnonzero(~np.isnan(np.asarray(y, dtype=float)))


def early_stop_split(n_train: int, h: int, frac: float = 0.125):
    """Fit / early-stopping positions inside a training block: the last `frac`
    is the early-stopping slice, with h rows purged between them so no fit
    label overlaps an early-stopping label."""
    es_start = int(n_train * (1 - frac))
    fit_end = max(1, es_start - h)
    return np.arange(fit_end), np.arange(es_start, n_train)


def windows(X: np.ndarray, time_steps: int):
    """(sequences ending at each row, row positions that have a full window)."""
    n = len(X)
    rows = np.arange(time_steps - 1, n)
    seq = np.stack([X[t - time_steps + 1:t + 1] for t in rows]) if len(rows) else \
        np.zeros((0, time_steps, X.shape[1]))
    return seq, rows
