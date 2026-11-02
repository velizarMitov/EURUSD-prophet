"""The frozen EUR/USD baseline: GARCH(1,1) x hour-of-day (design D5).

EUR/USD has no hourly in-house size model and no hourly GARCH row, so its
cells are judged against the estimator gold's `h1_garch_hod` uses, imported
read-only from src/gold/challengers: z_t = r_t / s(hour_t), GARCH(1,1) on z,
E|r_{t+1}| = sqrt(2/pi) * sigma_z(t+1|t) * s(hour_{t+1}). It is fitted ONCE on
the frozen EUR/USD history from 2020-11-01 to 2025-10-31 and never refitted.

    python -m src.external.tsfm.baseline --fit
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import ARTIFACTS, CLEAN_START, REPO, repo_path
from . import data as D
from . import vendor as V

FIT_START = '2020-11-01'
ARTIFACT = 'baseline_eurusd'
LOOKBACK = 3000              # bars the forward pass filters through (GARCH memory is ~70 bars)


def artifact_dir(repo: str = REPO) -> str:
    return repo_path(f'{ARTIFACTS}/{ARTIFACT}', repo)


def make():
    from src.gold.challengers import GoldGarchHOD
    return GoldGarchHOD({'cell': ARTIFACT, 'cadence': 'H1', 'columns': []}, name=ARTIFACT)


def inputs(frame: pd.DataFrame):
    from src.forecast_eval.challengers.base import Inputs
    return Inputs(index=pd.DatetimeIndex(frame.index), close=frame['close'].to_numpy(float),
                  X=np.zeros((len(frame), 0)))


def fit_frame(history: pd.DataFrame) -> pd.DataFrame:
    """The estimation bars: FIT_START <= bar < CLEAN_START, refusing any later one."""
    if len(history) and history.index[-1] >= D.clean_start():
        raise D.BoundaryViolation(f'the baseline fit was given a bar on or after {CLEAN_START}')
    return history[history.index >= pd.Timestamp(FIT_START, tz='UTC')]


def fit(repo: str = REPO, history: pd.DataFrame | None = None, manifest: str | None = None) -> dict:
    if history is None:
        V.verify('history_eurusd', repo=repo)
        history = D.read_frozen(D.history_path('eurusd', repo))
    frame = fit_frame(history)
    model = make().fit(inputs(frame), np.arange(len(frame)))
    out = artifact_dir(repo)
    files = model.save(out)
    V.record(ARTIFACT, files, f'GARCH(1,1) x hour-of-day on EUR/USD H1 {frame.index[0]} .. {frame.index[-1]}',
             manifest, repo)
    return {'rows': int(len(frame)), 'first': str(frame.index[0]), 'last': str(frame.index[-1]),
            'params': model.params, 'hod_all': model.hod_all}


def load(repo: str = REPO, manifest: dict | None = None):
    V.verify(ARTIFACT, manifest, repo)
    return make().load(artifact_dir(repo))


def size_pct(model, frame: pd.DataFrame) -> np.ndarray:
    """E|r_{t+1}| in % at every bar of `frame` (NaN at the first)."""
    return model.predict(inputs(frame))['vol_pct']


def main(argv=None):
    ap = argparse.ArgumentParser(description='The frozen EUR/USD GARCH x hour-of-day baseline.')
    ap.add_argument('--fit', action='store_true')
    a = ap.parse_args(argv)
    if a.fit:
        print(json.dumps(fit(), indent=1, default=str))


if __name__ == '__main__':
    main()
