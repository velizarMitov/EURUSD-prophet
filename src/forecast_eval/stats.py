"""Statistical tests (spec backtesting; design D7).

  pesaran_timmermann    directional accuracy vs independence (PT 1992)
  block bootstrap       moving-block (circular) CIs, block >= h; refuses n <= block
  clark_west            nested return forecasts vs the zero random walk (CW 2007)
  diebold_mariano_hln   NON-nested comparisons only (DM 1995, HLN 1997 correction)
  bonferroni, romano_wolf   family-wise error control across cells

The block resampler draws block starts exactly as
src/h1_direction_model._block_bootstrap_delta does (same RNG calls in the same
order), so for identical inputs and seed the intervals are identical; a parity
test holds that. It is written here rather than imported because that helper
is private and does not refuse a sample shorter than one block.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st

N_BOOT = 2000
SEED = 42


def block_length(h: int, cadence: str) -> int:
    """max(h, 24) for H1 cadence (|return| autocorrelation at lag 24 measured on
    H1); max(h, 5) for daily (one trading week)."""
    floor = {'H1': 24, 'D1': 5}[cadence]
    return max(int(h), floor)


def _block_index(rng, n: int, block_len: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    return np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]


def _refused(n: int, block_len: int) -> bool:
    return n == 0 or n <= block_len


def block_bootstrap_ci(x, block_len: int, alpha: float, n_boot: int = N_BOOT,
                       seed: int = SEED):
    """(lo, hi, point) for mean(x); (nan, nan, point) when refused."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    point = float(x.mean()) if n else float('nan')
    if _refused(n, block_len):
        return float('nan'), float('nan'), point
    rng = np.random.default_rng(seed)
    means = np.array([x[_block_index(rng, n, block_len)].mean() for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)), point)


def block_bootstrap_delta(cc, cr, block_len: int, alpha: float, n_boot: int = N_BOOT,
                          seed: int = SEED):
    """Paired CI for mean(cc) - mean(cr); (nan, nan, point) when refused."""
    cc, cr = np.asarray(cc, dtype=float), np.asarray(cr, dtype=float)
    n = len(cc)
    point = float(cc.mean() - cr.mean()) if n else float('nan')
    if _refused(n, block_len):
        return float('nan'), float('nan'), point
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_index(rng, n, block_len)
        deltas[b] = cc[idx].mean() - cr[idx].mean()
    return (float(np.quantile(deltas, alpha / 2)),
            float(np.quantile(deltas, 1 - alpha / 2)), point)


def one_sided_lower_bound(x, block_len: int, alpha: float, n_boot: int = N_BOOT,
                          seed: int = SEED) -> float:
    """Lower 1-alpha one-sided bound for mean(x) (the verdict rule's bound)."""
    lo, _, _ = block_bootstrap_ci(x, block_len, 2 * alpha, n_boot, seed)
    return lo


def one_sided_upper_bound(x, block_len: int, alpha: float, n_boot: int = N_BOOT,
                          seed: int = SEED) -> float:
    _, hi, _ = block_bootstrap_ci(x, block_len, 2 * alpha, n_boot, seed)
    return hi


def pesaran_timmermann(actual_up, pred_up) -> dict:
    """PT test of directional accuracy. Inputs are 0/1 arrays. One-sided
    p-value for 'better than independent'. Refused (nan) when either series is
    constant: the statistic's variance is then zero."""
    y = np.asarray(actual_up, dtype=float)
    x = np.asarray(pred_up, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(x)
    y, x = y[m], x[m]
    n = len(y)
    if n == 0:
        return {'n': 0, 'accuracy': float('nan'), 'stat': float('nan'), 'p_value': float('nan')}
    p_hat = float((y == x).mean())
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_p = p_star * (1 - p_star) / n
    v_ps = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
            + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = v_p - v_ps
    if denom <= 0 or px in (0.0, 1.0) or py in (0.0, 1.0):
        return {'n': n, 'accuracy': p_hat, 'stat': float('nan'), 'p_value': float('nan')}
    stat = (p_hat - p_star) / np.sqrt(denom)
    return {'n': n, 'accuracy': p_hat, 'stat': float(stat), 'p_value': float(st.norm.sf(stat))}


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return float(v)


def clark_west(actual, pred_model, pred_null=None, h: int = 1) -> dict:
    """Clark-West MSPE-adjusted test: does the model beat the nested null
    (default: the zero forecast, a driftless random walk)? One-sided; HAC
    variance with h-1 lags for overlapping forecasts."""
    y = np.asarray(actual, dtype=float)
    f1 = np.asarray(pred_model, dtype=float)
    f0 = np.zeros_like(y) if pred_null is None else np.asarray(pred_null, dtype=float)
    adj = (y - f0) ** 2 - ((y - f1) ** 2 - (f0 - f1) ** 2)
    n = len(adj)
    var = _newey_west_var(adj, max(0, h - 1))
    if n < 2 or var <= 0:
        return {'n': n, 'stat': float('nan'), 'p_value': float('nan')}
    stat = adj.mean() / np.sqrt(var / n)
    return {'n': n, 'stat': float(stat), 'p_value': float(st.norm.sf(stat))}


def diebold_mariano_hln(e1, e2, h: int = 1, nested: bool = False) -> dict:
    """DM with the Harvey-Leybourne-Newbold small-sample correction, squared
    loss, two-sided, t(n-1). For NON-nested models only: DM is undersized
    against a nested null, which is what clark_west is for."""
    if nested:
        raise ValueError('nested comparison: use clark_west, not Diebold-Mariano')
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    n = len(d)
    var = _newey_west_var(d, max(0, h - 1))
    if n < 3 or var <= 0:
        return {'n': n, 'stat': float('nan'), 'p_value': float('nan')}
    dm = d.mean() / np.sqrt(var / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * k
    return {'n': n, 'stat': float(stat), 'p_value': float(2 * st.t.sf(abs(stat), df=n - 1))}


def compare_return_forecasts(actual, pred_model, pred_other=None, h: int = 1,
                             nested: bool = True) -> dict:
    """Route to the right test: Clark-West when the other forecast is nested in
    the model (the random walk always is), Diebold-Mariano otherwise."""
    if nested:
        return {'test': 'clark_west', **clark_west(actual, pred_model, pred_other, h)}
    y = np.asarray(actual, dtype=float)
    return {'test': 'diebold_mariano_hln',
            **diebold_mariano_hln(y - np.asarray(pred_model), y - np.asarray(pred_other), h)}


def bonferroni(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    return np.minimum(1.0, p * len(p))


def romano_wolf(t_obs, t_boot_centered) -> np.ndarray:
    """Step-down max-t adjusted p-values (Romano & Wolf 2005).

    t_obs: (m,) observed statistics, larger = stronger evidence.
    t_boot_centered: (B, m) bootstrap statistics re-centred on the observed
    ones, i.e. draws from the null distribution, resampled JOINTLY so the
    correlation between cells is kept."""
    t = np.asarray(t_obs, dtype=float)
    tb = np.asarray(t_boot_centered, dtype=float)
    order = np.argsort(-t)
    p_adj = np.empty(len(t))
    running = 0.0
    for step, j in enumerate(order):
        remaining = order[step:]
        p = float((tb[:, remaining].max(axis=1) >= t[j]).mean())
        running = max(running, p)
        p_adj[j] = running
    return p_adj


def romano_wolf_mean_test(X, block_len: int, n_boot: int = N_BOOT, seed: int = SEED):
    """H0: mean(X[:, j]) <= 0 for every column j. Returns (raw p, Romano-Wolf p)
    from a joint moving-block bootstrap of studentised means."""
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    if _refused(n, block_len):
        nan = np.full(m, np.nan)
        return nan, nan
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd[sd == 0] = np.inf
    t_obs = np.sqrt(n) * mu / sd
    rng = np.random.default_rng(seed)
    tb = np.empty((n_boot, m))
    for b in range(n_boot):
        Xb = X[_block_index(rng, n, block_len)]
        sb = Xb.std(axis=0, ddof=1)
        sb[sb == 0] = np.inf
        tb[b] = np.sqrt(n) * (Xb.mean(axis=0) - mu) / sb
    raw = (tb >= t_obs).mean(axis=0)
    return raw, romano_wolf(t_obs, tb)
