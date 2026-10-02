"""Overfitting diagnostics (spec backtesting; design D8).

  trial log   results/horizon_study/trial_log.csv, APPEND-ONLY. One row per
              model x horizon x configuration ever fitted, frozen and tuned
              variants alike. Its row count is the N every Deflated Sharpe is
              charged for, so trying more things visibly costs something.
  DSR         Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): probability
              the true Sharpe exceeds the maximum expected under N trials of
              pure noise, corrected for skew, kurtosis and sample length.
  PBO         Probability of Backtest Overfitting by combinatorially symmetric
              cross-validation (Bailey, Borwein, Lopez de Prado & Zhu 2014):
              how often the in-sample winner lands below the out-of-sample
              median.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIAL_LOG = os.path.join(REPO, 'results', 'horizon_study', 'trial_log.csv')
TRIAL_FIELDS = ['trial_id', 'logged_at', 'study_id', 'model', 'cadence', 'horizon',
                'config_hash', 'variant', 'sharpe', 'n_obs', 'notes']
EULER_GAMMA = 0.5772156649015329


# ── trial log ──────────────────────────────────────────────────────────────

def read_trials(path: str = TRIAL_LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def n_trials(path: str = TRIAL_LOG) -> int:
    return len(read_trials(path))


def append_trials(rows: list[dict], path: str = TRIAL_LOG) -> int:
    """Append rows and return the new cumulative count. Opens in 'a' mode only:
    an earlier row can never be rewritten by this module."""
    allowed = set(TRIAL_FIELDS) - {'trial_id', 'logged_at'}
    for r in rows:
        unknown = set(r) - allowed
        if unknown:
            raise ValueError(f'unknown trial-log field(s): {sorted(unknown)}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = n_trials(path)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS, extrasaction='raise')
        if new_file:
            w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({'trial_id': start + i + 1, 'logged_at': now,
                        **{k: r.get(k, '') for k in TRIAL_FIELDS if k not in ('trial_id', 'logged_at')}})
    return start + len(rows)


def trial_sharpe_variance(path: str = TRIAL_LOG) -> float:
    vals = [float(r['sharpe']) for r in read_trials(path) if r.get('sharpe') not in ('', None)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.var(vals, ddof=1)) if len(vals) > 1 else float('nan')


# ── Deflated Sharpe ────────────────────────────────────────────────────────

def sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float('nan')


def expected_max_sharpe(n_trials_: int, sharpe_variance: float) -> float:
    """E[max SR] over N independent trials whose true Sharpe is zero."""
    if n_trials_ <= 1 or not np.isfinite(sharpe_variance):
        return 0.0
    z1 = st.norm.ppf(1 - 1 / n_trials_)
    z2 = st.norm.ppf(1 - 1 / (n_trials_ * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns, n_trials_: int, sharpe_variance: float) -> dict:
    """Per-period Sharpe, the noise benchmark SR0 for N trials, and the DSR
    probability. With N <= 1 this is the Probabilistic Sharpe Ratio vs 0."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials_, sharpe_variance)
    if T < 3 or not np.isfinite(sr):
        return {'sharpe': sr, 'sr0': sr0, 'dsr': float('nan'), 'n_trials': n_trials_, 'T': T}
    skew = float(st.skew(r))
    kurt = float(st.kurtosis(r, fisher=False))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    dsr = float(st.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(denom))) if denom > 0 else float('nan')
    return {'sharpe': sr, 'sr0': sr0, 'dsr': dsr, 'n_trials': n_trials_, 'T': T}


# ── PBO by CSCV ────────────────────────────────────────────────────────────

def pbo_cscv(perf_matrix, candidates: list[str] | None = None, n_submatrices: int = 16) -> dict:
    """perf_matrix: (T, N) per-period returns of N candidate configurations on
    the same rows. Rows are cut into S contiguous blocks; every half of them is
    in-sample once. PBO = share of splits whose in-sample-best candidate ranks
    at or below the out-of-sample median (logit <= 0)."""
    M = np.asarray(perf_matrix, dtype=float)
    T, N = M.shape
    S = int(n_submatrices)
    if S % 2 or S < 2:
        raise ValueError('n_submatrices must be even and >= 2')
    if N < 2:
        raise ValueError('PBO needs at least two candidates')
    blocks = np.array_split(np.arange(T), S)
    bsum = np.array([M[b].sum(axis=0) for b in blocks])          # (S, N)
    bsq = np.array([(M[b] ** 2).sum(axis=0) for b in blocks])
    bn = np.array([len(b) for b in blocks], dtype=float)

    def sr(sel):
        n = bn[sel].sum()
        mu = bsum[sel].sum(axis=0) / n
        var = bsq[sel].sum(axis=0) / n - mu ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(var > 0, mu / np.sqrt(var), -np.inf)

    logits = []
    all_idx = np.arange(S)
    for is_sel in combinations(range(S), S // 2):
        is_sel = np.array(is_sel)
        oos_sel = np.setdiff1d(all_idx, is_sel)
        best = int(np.argmax(sr(is_sel)))
        oos = sr(oos_sel)
        rank = (oos < oos[best]).sum() + 1                 # 1..N, N = best OOS
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {'pbo': float((logits <= 0).mean()), 'n_splits': len(logits),
            'n_submatrices': S, 'candidates': list(candidates) if candidates else
            [f'c{i}' for i in range(N)], 'logit_median': float(np.median(logits))}
