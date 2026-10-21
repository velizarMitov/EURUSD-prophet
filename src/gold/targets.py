"""Next-bar targets and the H1 eligibility rule (design D5).

Targets come from src/forecast_eval/targets at h = 1, so the percent scaling
stays at that package's single site:
    ret   100 * log(close[t+1] / close[t])
    dir   1 if ret > 0, 0 if ret < 0, NaN on a zero move
    vol   |ret|
A price forecast is derived (close * exp(ret / 100)) and never trained on.

H1 ELIGIBILITY. Spread is used nowhere (owner decision), so the bid-versus-mid
parity check that guards EUR/USD's intraday cells cannot run. Bid-only prices
are most misleading across a market break, where the spread widens and the bid
jumps. An H1 row is therefore scored only if its as-of bar and target bar are
consecutive hours, and the as-of bar itself follows its predecessor by one
hour: the last bar before every break and the first bar after it are out,
whichever hour the break falls on (it moves in DST-mismatch weeks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast_eval import targets as T

HOUR_NS = 3_600_000_000_000


def build_targets(close: pd.Series) -> dict:
    """Full-length float arrays aligned to `close`; the last row is NaN."""
    d, _ = T.direction_target(close, 1)
    return {'dir': d.to_numpy(float),
            'ret': T.return_pct_target(close, 1).to_numpy(float),
            'vol': T.volatility_pct_target(close, 1).to_numpy(float)}


def h1_eligible(index) -> np.ndarray:
    t = pd.DatetimeIndex(index).asi8
    ok = np.zeros(len(t), dtype=bool)
    if len(t) < 3:
        return ok
    step = np.diff(t)
    ok[1:-1] = (step[1:] == HOUR_NS) & (step[:-1] == HOUR_NS)
    return ok


def eligible(index, cadence: str) -> np.ndarray:
    """Rows that may be fitted and scored: H1 by the gap rule, D1 wherever a
    next bar exists (every daily close sits at the same break, so its bid-side
    bias largely cancels in the close-to-close difference)."""
    if cadence == 'H1':
        return h1_eligible(index)
    ok = np.ones(len(index), dtype=bool)
    ok[-1:] = False
    return ok


def implied_price(close: float, ret_pct: float) -> float:
    return float(close) * float(np.exp(float(ret_pct) / 100.0))
