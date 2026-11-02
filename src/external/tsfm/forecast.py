"""From deciles to a size and a probability (design D3; spec "A forecast is a
size and a probability read from quantiles").

Both models are asked for the deciles 0.1 ... 0.9 of the next bar. The
predictive CDF F is piecewise-linear through those nine points, extended
linearly beyond q10 and q90 with the end segments' slopes, and clipped to
[0, 1].

    size   the m >= 0 with F(m) - F(-m) = 0.5: the median of |R|, the point
           forecast that minimises mean absolute error -- the registered score
    p_up   1 - F(0), clipped to [0.1, 0.9], because outside the decile grid it
           is extrapolation. Descriptive only: direction is never registered.

`closes` mode turns each price decile q into 100 * ln(q / last close), so both
input modes are scored on the same log-return-in-percent scale as every
target in this repo.
"""

from __future__ import annotations

import numpy as np

LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
P_CLIP = (0.1, 0.9)
_EPS = 1e-12


def _strict(values) -> np.ndarray:
    """Non-decreasing deciles made strictly increasing (ties only nudge by EPS)."""
    v = np.maximum.accumulate(np.asarray(values, dtype=float))
    return v + _EPS * np.arange(len(v))


def cdf(values, x):
    """F(x) through the deciles, linear beyond the ends, clipped to [0, 1]."""
    v = _strict(values)
    x = np.asarray(x, dtype=float)
    inner = np.interp(x, v, LEVELS)
    lo_slope = (LEVELS[1] - LEVELS[0]) / (v[1] - v[0])
    hi_slope = (LEVELS[-1] - LEVELS[-2]) / (v[-1] - v[-2])
    out = np.where(x < v[0], LEVELS[0] + (x - v[0]) * lo_slope,
                   np.where(x > v[-1], LEVELS[-1] + (x - v[-1]) * hi_slope, inner))
    return np.clip(out, 0.0, 1.0)


def size(values, iters: int = 80) -> float:
    """Median of |R| by bisection on G(m) = F(m) - F(-m), which rises from 0."""
    v = _strict(values)
    hi = max(abs(v[0]), abs(v[-1]), 1e-9)
    while float(cdf(v, hi) - cdf(v, -hi)) < 0.5:
        hi *= 2.0
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(cdf(v, mid) - cdf(v, -mid)) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def p_up(values) -> float:
    return float(np.clip(1.0 - cdf(values, 0.0), *P_CLIP))


def cdf_many(values, x) -> np.ndarray:
    """`cdf` row by row for an (n, 9) array of deciles and n points."""
    v = np.maximum.accumulate(np.asarray(values, dtype=float), axis=1) + _EPS * np.arange(len(LEVELS))
    x = np.asarray(x, dtype=float)
    k = np.clip((v < x[:, None]).sum(axis=1), 1, len(LEVELS) - 1)    # segment [k-1, k]; ends extend
    rows = np.arange(len(v))
    lo, hi = v[rows, k - 1], v[rows, k]
    out = LEVELS[k - 1] + (x - lo) * (LEVELS[k] - LEVELS[k - 1]) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def size_many(values, iters: int = 80) -> np.ndarray:
    """`size` for every row of an (n, 9) array, by one vectorised bisection."""
    v = np.asarray(values, dtype=float)
    hi = np.maximum(np.maximum(np.abs(v[:, 0]), np.abs(v[:, -1])), 1e-9)
    for _ in range(64):
        short = (cdf_many(v, hi) - cdf_many(v, -hi)) < 0.5
        if not short.any():
            break
        hi = np.where(short, hi * 2.0, hi)
    lo = np.zeros(len(v))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = (cdf_many(v, mid) - cdf_many(v, -mid)) < 0.5
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def p_up_many(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.clip(1.0 - cdf_many(v, np.zeros(len(v))), *P_CLIP)


def closes_to_returns(price_deciles, last_close) -> np.ndarray:
    """100 * ln(q / last close); `last_close` may be a column (n, 1) for n rows."""
    q = np.maximum(np.asarray(price_deciles, dtype=float), 1e-12)
    return 100.0 * np.log(q / np.asarray(last_close, dtype=float))


def summarise(return_deciles) -> dict:
    """{'size_pct', 'p_up', 'q10' ... 'q90'} from deciles of the next return in %."""
    v = np.asarray(return_deciles, dtype=float)
    out = {'size_pct': size(v), 'p_up': p_up(v)}
    out.update({f'q{int(round(l * 100))}': float(x) for l, x in zip(LEVELS, v)})
    return out


def to_return_deciles(deciles, mode: str, last_close: float | None = None) -> np.ndarray:
    if mode == 'returns':
        return np.asarray(deciles, dtype=float)
    if mode == 'closes':
        if last_close is None:
            raise ValueError('closes mode needs the as-of close')
        return closes_to_returns(deciles, last_close)
    raise ValueError(f'unknown input mode {mode!r}')
