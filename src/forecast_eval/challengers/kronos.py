"""Kronos, run as-is (task 7.8).

Kronos is an external, pinned foundation model: nothing is trained. It is run
at its pinned generation settings (temperature, top_p, context length) through
src/external/kronos, imported read-only. Only the forecast length changes.

One sampling per as-of bar at pred_len = the longest requested horizon serves
every horizon: generation is autoregressive, so the first h steps of a longer
path are drawn exactly as a length-h path would be. Reading step h of each path
is therefore "forecast length set to the horizon" without resampling.

  direction   p_up(h)    share of paths whose close at step h is above the last
                         observed close
  volatility  vol_pct(h) mean over paths of |log return to step h|, in percent
                         (scaled through targets._to_pct, the package's single
                         scaling site)

Kronos was trained on data up to 2024-06. It is only ever scored on as-of bars
from the clean window (2024-07-01 onward); earlier bars are in its training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import targets as T
from .base import Challenger, ChallengerUnavailable, Inputs

CLEAN_WINDOW_START = pd.Timestamp('2024-07-01', tz='UTC')


class Kronos(Challenger):
    cadence = 'H1'
    train_free = True

    def __init__(self, config: dict, kind: str = 'direction'):
        super().__init__(config)
        self.kind = kind
        self.name = f'kronos_{kind}'
        self._predictor = None

    def _ensure(self):
        if self._predictor is not None:
            return self._predictor
        from src.external.kronos import loader
        ok, why = loader.probe()
        if not ok:
            raise ChallengerUnavailable(f'Kronos unavailable: {why}')
        try:
            self._predictor, self.meta = loader.load(which=self.config.get('model', 'mini'))
        except loader.KronosUnavailable as e:
            raise ChallengerUnavailable(f'Kronos unavailable: {e}') from e
        return self._predictor

    def build_inputs(self, h1: pd.DataFrame) -> Inputs:
        idx = pd.DatetimeIndex(h1.index)
        return Inputs(index=idx, close=h1['close'].to_numpy(float),
                      X=h1[['open', 'high', 'low', 'close']].to_numpy(float),
                      extra={'dow': idx.dayofweek.to_numpy()})

    def fit(self, inputs, train_pos, targets, seed, h):
        self.fitted = True          # nothing to learn
        return self

    def sample(self, inputs: Inputs, as_of_pos, max_h: int, seed: int = 42) -> dict:
        """{as_of position: (n_paths, max_h) sampled closes}. Positions before
        the clean window or without enough context are skipped."""
        from src.external.kronos.loader import CONTEXT_BARS
        from src.external.kronos.predict import sample_paths
        predictor = self._ensure()
        import torch
        frame = pd.DataFrame(inputs.X, index=inputs.index, columns=['open', 'high', 'low', 'close'])
        out = {}
        for p in np.asarray(as_of_pos, dtype=int):
            if p + 1 < CONTEXT_BARS or inputs.index[p] < CLEAN_WINDOW_START:
                continue
            torch.manual_seed(seed + int(p))
            ctx = frame.iloc[p + 1 - CONTEXT_BARS:p + 1]
            start = inputs.index[p] + pd.Timedelta(hours=1)
            out[int(p)] = sample_paths(predictor, ctx, start, n_paths=self.config['n_paths'],
                                       pred_len=max_h)
        return out

    @staticmethod
    def from_paths(paths_by_pos: dict, last_close: np.ndarray, h: int, n: int, kind: str) -> np.ndarray:
        out = np.full(n, np.nan)
        for p, paths in paths_by_pos.items():
            step = np.asarray(paths)[:, h - 1]
            if kind == 'direction':
                out[p] = float((step > last_close[p]).mean())
            else:
                out[p] = float(np.mean(np.abs(T._to_pct(np.log(step / last_close[p])))))
        return out

    def predict(self, inputs, as_of_pos=None, h: int = 1, paths=None):
        n = len(inputs)
        if paths is None:
            pos = np.arange(n) if as_of_pos is None else as_of_pos
            paths = self.sample(inputs, pos, h)
        key = 'p_up' if self.kind == 'direction' else 'vol_pct'
        return {key: self.from_paths(paths, inputs.close, h, n, self.kind)}

    def _save(self, path):
        pass                          # pinned external weights: nothing of ours to store

    def _load(self, path):
        pass
