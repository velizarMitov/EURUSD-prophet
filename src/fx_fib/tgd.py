"""D1 Time Goal Days (spec fx-fibonacci/structure-features; Fischer ch. 7-8).

From the last two confirmed D1 highs A, B (and separately the last two lows) a TGD is
projected at C = B + 1.618 * (B - A), in D1 bar indices, rounded to a whole day. When
Fischer's 15-bar rule is active, the 15-bar extremes replace the pivots as the swing.

`cur_day` is the D1 index the H1 bar is trading in: the newest closed D1 bar + 1.
"""

from __future__ import annotations

import numpy as np


def projections(d1_kind: np.ndarray, d1_idx: np.ndarray, quiet_active: np.ndarray,
                quiet_first: np.ndarray, quiet_second: np.ndarray, ratio: float) -> np.ndarray:
    """(n, 2) projected TGD day indices per row; NaN where a projection does not exist."""
    n = len(d1_kind)
    out = np.full((n, 2), np.nan)
    for col, k in enumerate((1, -1)):                    # highs pair, lows pair
        mask = d1_kind == k
        cnt = mask.sum(axis=1)
        # the two newest pivots of this kind among the last four (alternating -> columns)
        newer = np.where(mask[:, 3], d1_idx[:, 3], d1_idx[:, 2]).astype(float)
        older = np.where(mask[:, 1], d1_idx[:, 1], d1_idx[:, 0]).astype(float)
        c = newer + ratio * (newer - older)
        out[:, col] = np.where(cnt >= 2, c, np.nan)
    q = quiet_second + ratio * (quiet_second - quiet_first)
    out[quiet_active] = np.column_stack([q, np.full(n, np.nan)])[quiet_active]
    return np.rint(out)


def tgd_features(proj: np.ndarray, cur_day: np.ndarray, d1_dist_618: np.ndarray,
                 d1_dist_ext: np.ndarray, cfg: dict) -> dict:
    t = cfg['tgd']
    band, cap = t['band_days'], t['no_tgd_days']
    delta = proj - cur_day[:, None]                       # signed days to each projection
    active = np.isfinite(delta) & (delta >= -band)
    absd = np.where(active, np.abs(delta), np.inf)
    nearest = np.argmin(absd, axis=1)
    has = np.isfinite(absd.min(axis=1))
    bars_to = np.where(has, delta[np.arange(len(delta)), nearest], cap)
    cluster = (active & (np.abs(delta) <= band)).sum(axis=1).astype(float)
    d_price = np.fmin(np.abs(d1_dist_618), np.abs(d1_dist_ext))
    conf = np.exp(-d_price / t['confluence_price_scale_atr']) * np.exp(-np.abs(bars_to) / t['confluence_time_scale_days'])
    conf = np.where(has, conf, 0.0)
    ok = np.isfinite(cur_day) & np.isfinite(d_price)
    return {'d1_bars_to_nearest_tgd': np.where(ok, bars_to, np.nan),
            'd1_tgd_cluster_count': np.where(ok, cluster, np.nan),
            'price_time_confluence': np.where(ok, conf, np.nan)}
