"""Random-walk benchmarks (spec backtesting "Random-walk benchmarks").

Every result is reported next to three nulls computed on the SAME rows:

  1. direction  -- a coin flip (50 %), alongside the train-majority class, which
                   is the honest "no information" rule when classes are uneven;
  2. return     -- the zero forecast: a driftless random walk (Meese-Rogoff), the
                   benchmark FX return forecasts historically fail to beat;
  3. strategy   -- random-sign position series with EXACTLY the model's turnover,
                   so the model is compared with luck that pays the same costs.
"""

from __future__ import annotations

import numpy as np

from . import costs as C

COIN_FLIP_ACCURACY = 0.5


def train_majority_class(y_train) -> int:
    y = np.asarray(y_train, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        raise ValueError('empty training labels')
    return int(y.mean() >= 0.5)


def accuracy(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m = ~np.isnan(yt) & ~np.isnan(yp)
    return float((yt[m] == yp[m]).mean()) if m.any() else float('nan')


def majority_accuracy(y_train, y_test) -> float:
    cls = train_majority_class(y_train)
    yt = np.asarray(y_test, dtype=float)
    return accuracy(yt, np.full(yt.shape, float(cls)))


def zero_return_forecast(n: int) -> np.ndarray:
    return np.zeros(int(n))


def turnover(positions) -> float:
    """Fraction of bars after the first on which the position changes."""
    pos = np.asarray(positions, dtype=float)
    if pos.size < 2:
        return 0.0
    return float((pos[1:] != pos[:-1]).mean())


def n_changes(positions) -> int:
    pos = np.asarray(positions, dtype=float)
    return int((pos[1:] != pos[:-1]).sum()) if pos.size > 1 else 0


def random_sign_strategies(n: int, k_changes: int, n_sims: int, seed: int = 42) -> np.ndarray:
    """(n_sims, n) +/-1 positions, each with EXACTLY k_changes flips among the
    n-1 bar transitions, placed uniformly at random; the starting side is a
    fair coin."""
    if not 0 <= k_changes <= max(0, n - 1):
        raise ValueError('k_changes out of range')
    rng = np.random.default_rng(seed)
    out = np.empty((n_sims, n))
    for s in range(n_sims):
        flips = np.zeros(n)
        if k_changes:
            flips[1 + rng.choice(n - 1, size=k_changes, replace=False)] = 1
        start = 1.0 if rng.random() < 0.5 else -1.0
        out[s] = start * np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    return out


def random_sign_benchmark(model_positions, price_moves, cost_price: float,
                          n_sims: int = 1000, seed: int = 42) -> dict:
    """Net P&L of the model versus random-sign strategies with matched turnover.
    `percentile` is the share of random strategies the model beats."""
    pos = np.asarray(model_positions, dtype=float)
    model_net = C.net_pnl(pos, price_moves, cost_price)['net_total']
    k = n_changes(pos)
    sims = random_sign_strategies(len(pos), k, n_sims, seed)
    nets = np.array([C.net_pnl(s, price_moves, cost_price)['net_total'] for s in sims])
    return {
        'model_net': model_net, 'model_changes': k,
        'random_net_mean': float(nets.mean()), 'random_net_p95': float(np.percentile(nets, 95)),
        'percentile': float((nets < model_net).mean()), 'n_sims': n_sims,
    }
