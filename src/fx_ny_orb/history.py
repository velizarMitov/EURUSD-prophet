"""The historical days (spec fx-ny-orb-hot/evaluation; design D4).

For each pair: fx_nyvol's eligible out-of-sample days (its saved walk-forward `p_harx`, causal
`hot_threshold`, `y` and calendar flags, read-only and never refitted) inside the M15 window,
joined to that day's M15 mid bars from the M1 parquet.

    close14 = mid close of the M15 bar labelled 14:45 (the close of the H1 bar labelled 14:00)
    F       = exp(p_harx) * close14
    hot     = p_harx > hot_threshold

`day_reason` is '' when the day can be simulated, else 'no M15 data' or 'corrupt spread' (a bar of
14:00-20:45 above 50 pips). Nothing here computes an entry or an exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import m15 as M

DAY_COLS = ['pair', 'date', 'p_harx', 'hot_threshold', 'hot', 'y', 'dst_mismatch', 'is_nfp_day', 'is_fomc_day',
            'close14', 'F', 'day_reason']


def nyvol_oos() -> pd.DataFrame:
    from ..fx_nyvol import load_config as nyvol_config
    from ..fx_nyvol.walk_forward import load_oos
    return load_oos(nyvol_config(), 'lgbm')


def session_slice(bars: pd.DataFrame) -> pd.DataFrame:
    m = pd.DatetimeIndex(bars.index).hour * 60 + pd.DatetimeIndex(bars.index).minute
    return bars[(m >= 14 * 60) & (m <= 20 * 60 + 45)]


def assemble(cfg: dict, bars: dict | None = None, oos: pd.DataFrame | None = None) -> tuple:
    """(days frame, {pair: {date: that day's 14:00-20:45 M15 bars}})."""
    oos = oos if oos is not None else nyvol_oos()
    a = pd.Timestamp(cfg['data']['window_start'], tz='UTC')
    b = pd.Timestamp(cfg['data']['window_end'], tz='UTC')
    rows, day_bars = [], {}
    for pair in cfg['pairs']:
        src = bars[pair] if bars is not None else M.hist_bars(pair, cfg)
        groups = {d: session_slice(g) for d, g in M.by_day(src).items()}
        day_bars[pair] = {}
        o = oos[(oos['pair'] == pair) & (oos['date'] >= a) & (oos['date'] <= b)].sort_values('date')
        for _, r in o.iterrows():
            d = r['date']
            g = groups.get(d)
            reason, close14 = '', np.nan
            if g is None or not len(g):
                reason = 'no M15 data'
            else:
                if g['corrupt'].any():
                    reason = 'corrupt spread'
                hit = g.index == d + pd.Timedelta(hours=14, minutes=45)
                if hit.any():
                    close14 = float(g.loc[hit, 'mc'].iloc[0])
                day_bars[pair][d] = g
            F = float(np.exp(r['p_harx']) * close14) if np.isfinite(close14) else np.nan
            rows.append({'pair': pair, 'date': d, 'p_harx': float(r['p_harx']),
                         'hot_threshold': float(r['hot_threshold']), 'hot': bool(r['p_harx'] > r['hot_threshold']),
                         'y': float(r['y']), 'dst_mismatch': float(r['dst_mismatch']),
                         'is_nfp_day': float(r['is_nfp_day']), 'is_fomc_day': float(r['is_fomc_day']),
                         'close14': close14, 'F': F, 'day_reason': reason})
    return pd.DataFrame(rows, columns=DAY_COLS), day_bars
