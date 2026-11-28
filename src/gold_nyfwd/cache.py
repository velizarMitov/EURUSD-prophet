"""The family's own append-only H1 caches (spec gold-nyrange-forward/live-data; design D3).

One CSV per symbol under research_models/gold_nyfwd/cache/<symbol>_h1.csv: `time` (the Berlin label,
stored UTC-localised as every cache in the repo) and bid OHLC. A top-up appends only bars labelled after
the cache's last label; an earlier bar is never rewritten, even when the broker now reports it
differently (such revisions are counted, not applied).
"""

from __future__ import annotations

import os

import pandas as pd

from . import art_dir

OHLC = ['open', 'high', 'low', 'close']


def cache_dir(cfg: dict) -> str:
    return os.path.join(art_dir(cfg), 'cache')


def cache_file(cfg: dict, symbol: str) -> str:
    return os.path.join(cache_dir(cfg), f'{symbol}_h1.csv')


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], tz='UTC', name='time'), dtype=float)


def load(cfg: dict, symbol: str) -> pd.DataFrame:
    p = cache_file(cfg, symbol)
    if not os.path.exists(p):
        return empty()
    df = pd.read_csv(p)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.set_index('time').sort_index()[OHLC].astype(float)


def update(cfg: dict, symbol: str, fresh: pd.DataFrame) -> dict:
    """Append the bars of `fresh` labelled after the cache's last label. Never rewrites."""
    old = load(cfg, symbol)
    fresh = fresh[~fresh.index.duplicated(keep='last')].sort_index()
    revised = 0
    if len(old):
        shared = old.index.intersection(fresh.index)
        if len(shared):
            d = (old.loc[shared, OHLC] - fresh.loc[shared, OHLC]).abs().to_numpy().max(axis=1)
            revised = int((d > 1e-9).sum())
        new = fresh[fresh.index > old.index.max()]
    else:
        new = fresh
    if len(new):
        os.makedirs(cache_dir(cfg), exist_ok=True)
        out = new[OHLC].copy()
        out.index.name = 'time'
        p = cache_file(cfg, symbol)
        out.to_csv(p, mode='a', header=not os.path.exists(p), lineterminator='\n')
    return {'appended': int(len(new)), 'revised_not_applied': revised}
