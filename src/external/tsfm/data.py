"""Frozen bars for fine-tuning and the replay (design D4, D10).

Two frozen files per instrument, each recorded in the artifact manifest:

    results/external_tsfm/finetune/<instrument>_h1_history.csv
        every H1 bar BEFORE the clean window -- the only data the fine-tuning
        ever reads. Writing a bar on or after CLEAN_START is refused.
    results/external_tsfm/replay/<instrument>_h1_window.csv
        the clean window's bars (as-of 2025-11-01 .. 2026-10-31, plus the bar
        after, which is the last target). A request that starts earlier is
        refused, naming the clean-window start.

Gold comes from its append-only H1 cache; EUR/USD from the deepest MT5 read
the terminal serves, through the forward arbiter's read-only reader. Both
store OHLC only, on the broker's Europe/Berlin label clock localised to UTC as
is (CLAUDE.md invariant 6) -- never UTC.

Series for the models: `returns` is 100 * ln(c_t / c_{t-1}) per bar (weekend
gaps are consecutive steps, as for Kronos); `closes` is the closes. An as-of
bar at close position i is forecast from the series up to i; its target is
bar i + 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import CLEAN_START, FINETUNE, REPLAY, REPO, repo_path
from . import sources as S
from . import vendor as V

OHLC = ['open', 'high', 'low', 'close']
WINDOW_END = '2026-10-31'          # last as-of DAY of the replay window
HOLDOUT_START = '2024-11-01'       # stage 1 of the fine-tuning trains before this
EURUSD_SYMBOL = 'EURUSD'
MT5_MAX_PROBE = 400_000


class BoundaryViolation(ValueError):
    """A bar on the wrong side of the clean-window boundary."""


def clean_start() -> pd.Timestamp:
    return pd.Timestamp(CLEAN_START, tz='UTC')


# ── sources ─────────────────────────────────────────────────────────────────

def gold_h1(repo: str = REPO) -> pd.DataFrame:
    from src.gold import data as GD
    df = GD.read_cache(S.gold_h1_cache_path(repo))
    if df is None or df.empty:
        raise FileNotFoundError('the gold H1 cache is missing')
    return df[OHLC]


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    df = df.copy()
    df.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    return df


def eurusd_h1_deepest(reader=None, hi: int = MT5_MAX_PROBE) -> tuple[pd.DataFrame, int]:
    """(closed OHLC bars, bars requested): the largest count the terminal serves,
    by bisection -- MT5 returns nothing at all, not a partial frame, past its
    limit."""
    from src.forecast_eval.mt5_reader import MT5Reader, MT5Unavailable
    reader = reader or MT5Reader()
    with reader:
        def get(n):
            try:
                return reader.bars(EURUSD_SYMBOL, 'H1', n, sync=False)
            except MT5Unavailable:
                return None
        lo, best = 1000, get(1000)
        if best is None:
            raise MT5Unavailable('MT5 serves no EURUSD H1 bars')
        top = get(hi)
        if top is not None:
            return _utc(top[OHLC].iloc[:-1]), hi
        while hi - lo > 1000:
            mid = (lo + hi) // 2
            got = get(mid)
            if got is None:
                hi = mid
            else:
                lo, best = mid, got
    return _utc(best[OHLC].iloc[:-1]), lo           # the newest bar may still be forming


# ── frozen files ────────────────────────────────────────────────────────────

def history_path(instrument: str, repo: str = REPO) -> str:
    return repo_path(f'{FINETUNE}/{instrument}_h1_history.csv', repo)


def window_path(instrument: str, repo: str = REPO) -> str:
    return repo_path(f'{REPLAY}/{instrument}_h1_window.csv', repo)


def _write(df: pd.DataFrame, path: str) -> None:
    out = df.reset_index()
    out.columns = ['time'] + list(df.columns)
    out['time'] = pd.DatetimeIndex(out['time']).strftime('%Y-%m-%dT%H:%M:%S+00:00')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_frozen(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.set_index('time').sort_index()[OHLC].astype(float)


def summary(df: pd.DataFrame) -> dict:
    return {'rows': int(len(df)), 'first': str(df.index[0]), 'last': str(df.index[-1]),
            'years': round((df.index[-1] - df.index[0]).days / 365.25, 1)}


def write_history(instrument: str, df: pd.DataFrame, source: str, repo: str = REPO,
                  manifest: str | None = None) -> dict:
    """Freeze the pre-window history. Refuses ANY bar on or after CLEAN_START:
    the caller slices, this only checks, so the boundary lives in one place."""
    late = df.index[df.index >= clean_start()]
    if len(late):
        raise BoundaryViolation(f'{instrument}: {len(late)} bar(s) on or after the clean-window start '
                                f'{CLEAN_START} (first {late[0]}); the fine-tuning history must end before it')
    path = history_path(instrument, repo)
    _write(df[OHLC], path)
    V.record(f'history_{instrument}', [path], source, manifest, repo)
    return {**summary(df), 'path': os.path.relpath(path, repo).replace('\\', '/'), 'source': source}


def check_window_request(start) -> None:
    ts = pd.Timestamp(start)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
    if ts < clean_start():
        raise BoundaryViolation(f'the replay starts at the clean-window start {CLEAN_START}; {ts.date()} is '
                                'inside the models\' pretraining and fine-tuning period and is never scored')


def write_window(instrument: str, df: pd.DataFrame, source: str, start=CLEAN_START, repo: str = REPO,
                 manifest: str | None = None) -> dict:
    """Freeze the clean window: as-of bars from `start` to WINDOW_END, plus the
    first bar after it (the last as-of bar's target)."""
    check_window_request(start)
    end = pd.Timestamp(WINDOW_END, tz='UTC') + pd.Timedelta(days=1)
    inside = df[(df.index >= clean_start()) & (df.index < end)]
    after = df[df.index >= end].iloc[:1]
    if after.empty:
        raise BoundaryViolation(f'{instrument}: no bar after {WINDOW_END} yet, so the last as-of bar has no target')
    win = pd.concat([inside, after])
    path = window_path(instrument, repo)
    _write(win[OHLC], path)
    V.record(f'window_{instrument}', [path], source, manifest, repo)
    return {**summary(win), 'path': os.path.relpath(path, repo).replace('\\', '/'), 'source': source}


# ── series and positions ────────────────────────────────────────────────────

def returns_pct(close) -> np.ndarray:
    """100 * ln(c_t / c_{t-1}) at position t; NaN at 0. The package's only
    percent scaling of a return."""
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    out[1:] = 100.0 * np.diff(np.log(c))
    return out


def model_series(close, mode: str) -> np.ndarray:
    """The series a model reads, aligned to bar positions. In returns mode
    position 0 is NaN and is never inside a window (windows start at 1)."""
    if mode == 'returns':
        return returns_pct(close)
    if mode == 'closes':
        return np.asarray(close, dtype=float)
    raise ValueError(f'unknown input mode {mode!r}')


def eligible(index) -> np.ndarray:
    """The H1 market-break rule both instruments settle by: the as-of bar
    follows its predecessor by one hour and its target follows it by one hour."""
    from src.gold.targets import h1_eligible
    return h1_eligible(index)


def as_of_positions(index, start, end, need_target: bool = True, only_eligible: bool = True) -> np.ndarray:
    """Positions whose as-of bar is in [start, end) and, by default, that have a
    next bar and pass the market-break rule."""
    idx = pd.DatetimeIndex(index)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    s = s.tz_localize('UTC') if s.tzinfo is None else s
    e = e.tz_localize('UTC') if e.tzinfo is None else e
    ok = (idx >= s) & (idx < e)
    if need_target:
        ok[-1:] = False
    if only_eligible:
        ok &= eligible(idx)
    return np.flatnonzero(ok)


def spaced(positions: np.ndarray, cap: int) -> np.ndarray:
    """At most `cap` positions, evenly spread over the whole range."""
    if len(positions) <= cap:
        return positions
    return positions[np.unique(np.linspace(0, len(positions) - 1, cap).round().astype(int))]
