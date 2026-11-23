"""Load every input once: gold H1 (clock-checked), NY-close D1, daily and hourly DXY (lag-checked), and the
digests of the three caches (spec gold-bb-nyrange/gold-bars, dollar-index)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import repo_path, sha256_file
from . import bars as B
from . import dxy as D


@dataclass
class Inputs:
    h1: pd.DataFrame
    d1: pd.DataFrame
    dxy_d1: pd.DataFrame
    dxy_h1: pd.DataFrame


def cache_digests(cfg: dict) -> dict:
    return {'gold_h1': sha256_file(repo_path(cfg['paths_read']['gold_h1'])),
            'dxy_h1': sha256_file(D.data_path(cfg, 'h1_file')),
            'dxy_d1': sha256_file(D.data_path(cfg, 'd1_file'))}


def load(cfg: dict) -> Inputs:
    h1 = B.load_h1(cfg)
    return Inputs(h1, B.d1_ny(h1, cfg), D.load_d1(cfg), D.load_h1(cfg))


def dxy_gaps(dxy_h1: pd.DataFrame, limit: float) -> list:
    """Days whose first DXY bar opens more than `limit` (relative) away from the previous bar's close."""
    day = dxy_h1.index.normalize()
    first = ~pd.Series(day).duplicated().to_numpy()
    prev_close = dxy_h1['close'].shift(1)
    g = (dxy_h1['open'] / prev_close - 1).abs()
    hit = first & (g > limit).to_numpy()
    return [(str(t.date()), round(float(v), 5)) for t, v in zip(dxy_h1.index[hit], g[hit])]
