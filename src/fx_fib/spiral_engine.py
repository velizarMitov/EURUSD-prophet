"""Fischer's logarithmic spiral, corrected (spec fx-fibonacci/structure-features; ch. 9, app. B).

R(phi) = R0 * exp(k * phi), with a POSITIVE constant. Two sets are built (owner, 2026-11-21):
    spiral306  k = (2/pi) ln(phi) ~ 0.306349 (x phi per quarter turn), rings 0..2
    spiral077  k = ln(phi)/(2 pi) ~ 0.076587 (x phi per full turn),    rings 2..4
With k = 0.306349 rings 2..4 lie 25-175 ATR away and were capped on ~98 % of rows, hence
rings 0..2 for that constant. The book's printed log(1/phi) is negative and is refused.

Coordinates are made scale-free around pivot B of the newest confirmed D1 A-B-C swing:
    x = (t - t_B) / H        t in trading days, H = h_days (default 5)
    y = (P - P_B) / V        V = ATR_D1 at B
The spiral starts at A (or C) and turns from there; ring n is phi in [2 pi (n-1), 2 pi n),
so ring 0 is the turn just inside the start. For each direction (DIR = +1 counter-clockwise,
-1 clockwise) the feature is the vertical distance from the H1 close to the nearest branch of
the set's rings crossing the current x.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PHI = (1 + math.sqrt(5)) / 2
K_DEFAULT = 2 / math.pi * math.log(PHI)
K_ALT = math.log(PHI) / (2 * math.pi)


def check_k(k: float) -> float:
    if not (isinstance(k, (int, float)) and math.isfinite(k) and k > 0):
        raise ValueError(f'spiral k must be positive: use (2/pi)*ln(phi) = {K_DEFAULT:.6f} '
                         f'or ln(phi)/(2*pi) = {K_ALT:.6f}; got {k!r} (log(1/phi) is a misprint)')
    return float(k)


def sample_rings(r0: float, theta0: float, k: float, direction: int, rings, samples_per_turn: int):
    """Sampled branch points (x, y, ring) for the given rings."""
    xs, ys, rs = [], [], []
    for n in rings:
        phi = np.linspace(2 * np.pi * (n - 1), 2 * np.pi * n, samples_per_turn + 1)
        r = r0 * np.exp(k * phi)
        th = theta0 + direction * phi
        xs.append(r * np.cos(th)); ys.append(r * np.sin(th)); rs.append(np.full(len(phi), n))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)


def nearest_branch(x_now: np.ndarray, y_now: np.ndarray, xs, ys, rs):
    """For each (x_now, y_now): signed distance y_now - y_branch to the nearest branch that
    crosses the vertical line x = x_now, and that branch's ring. NaN / 0 when none crosses."""
    x_now, y_now = np.atleast_1d(x_now), np.atleast_1d(y_now)
    same_ring = rs[:-1] == rs[1:]
    f = xs[None, :] - x_now[:, None]
    cross = (np.sign(f[:, :-1]) != np.sign(f[:, 1:])) & same_ring[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        w = f[:, :-1] / (f[:, :-1] - f[:, 1:])
        yc = ys[None, :-1] + w * (ys[None, 1:] - ys[None, :-1])
    d = np.where(cross, y_now[:, None] - yc, np.nan)
    absd = np.where(cross, np.abs(d), np.inf)
    j = np.argmin(absd, axis=1)
    rows = np.arange(len(x_now))
    has = np.isfinite(absd[rows, j])
    signed = np.where(has, d[rows, j], np.nan)
    ring = np.where(has, rs[:-1][j], 0)
    return signed, ring


def spiral_features(d1_kind: np.ndarray, d1_idx: np.ndarray, d1_price: np.ndarray,
                    atr_d1_full: np.ndarray, t_days: np.ndarray, close: np.ndarray, cfg: dict,
                    variant: dict) -> dict:
    """Per-H1-row features of one spiral set from the newest three visible D1 pivots (A, B, C)."""
    s = cfg['spiral']
    k = check_k(variant['k'])
    prefix, rings = variant['name'], variant['rings']
    cap = float(s['distance_cap_atr'])
    n = len(close)
    have = (d1_kind[:, 1] != 0) & np.isfinite(t_days) & np.isfinite(close)
    b_idx = np.where(have, d1_idx[:, 2], 0)
    s_col = 1 if s['start'] == 'A' else 3
    if s['start'] not in ('A', 'C'):
        raise ValueError("spiral start must be 'A' or 'C'")
    v = np.where(have, atr_d1_full[np.clip(b_idx, 0, len(atr_d1_full) - 1)], np.nan)
    have &= np.isfinite(v) & (v > 0)

    x_now = (t_days - d1_idx[:, 2]) / s['h_days']
    y_now = (close - d1_price[:, 2]) / v
    x_s = (d1_idx[:, s_col] - d1_idx[:, 2]) / s['h_days']
    y_s = (d1_price[:, s_col] - d1_price[:, 2]) / v
    r0 = np.hypot(x_s, y_s)
    th0 = np.arctan2(y_s, x_s)
    have &= np.isfinite(r0) & (r0 > 0)

    out = {}
    key = pd.DataFrame({'b': d1_idx[:, 2], 's': d1_idx[:, s_col], 'r0': r0, 'th0': th0})
    groups = key[have].groupby(['b', 's', 'r0', 'th0'], sort=False).indices
    for name, direction in (('ccw', 1), ('cw', -1)):
        signed = np.full(n, np.nan)
        ring = np.zeros(n)
        rows_all = np.flatnonzero(have)
        for (_, _, g_r0, g_th0), pos in groups.items():
            rows = rows_all[pos]
            xs, ys, rs = sample_rings(g_r0, g_th0, k, direction, rings, s['samples_per_turn'])
            sd, rg = nearest_branch(x_now[rows], y_now[rows], xs, ys, rs)
            signed[rows], ring[rows] = sd, rg
        dist = np.where(np.isfinite(signed), np.minimum(np.abs(signed), cap), cap)
        prev_signed = np.r_[np.nan, signed[:-1]]
        prev_b = np.r_[-1, d1_idx[:-1, 2]]
        crossed = (np.isfinite(signed) & np.isfinite(prev_signed) & (prev_b == d1_idx[:, 2])
                   & (np.sign(signed) != np.sign(prev_signed)))
        out[f'{prefix}_dist_{name}_atr'] = np.where(have, dist, np.nan)
        out[f'{prefix}_ring_id_{name}'] = np.where(have, ring, np.nan)
        out[f'{prefix}_crossed_{name}'] = np.where(have, crossed.astype(float), np.nan)
    return out
