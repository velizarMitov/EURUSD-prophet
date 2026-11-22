"""Eligibility and the New York range target (spec fx-ny-volatility/range-target; design D2, D3).

Data: the fx_session H1 caches of EURUSD, GBPUSD and AUDUSD, READ-ONLY, from 2012-01-01, on the
Berlin label clock. Dates are the Mon-Fri dates of fx_session's session calendar.

A day is TARGET-AVAILABLE (reason '') when, jointly for both scored pairs:
  1. fx_session's session_calendar marks it eligible (every bar labelled 14:00-21:00 for both
     scored pairs; the week's weekly open on the Berlin clock for every loaded pair) -- else
     `incomplete session` or `unverified clock week`;
  2. bars 00:00-14:00 exist for all three pairs -- else `missing pre-session bar`;
  3. no bar it uses has a spread above 50 pips (bars 00:00-21:00 of the scored pairs, 00:00-14:00 of
     AUDUSD) -- else `corrupt spread`.
Warm-up (HAR and 20-day windows) is applied after the features are built.

Target, per pair:  y = log((mid_high - mid_low) / mid_close14) over the bars labelled 15:00-21:00,
mid = bid + half the bar's own spread. The bid twin y_bid uses bid highs/lows and the bid close14
(parity only). The 22:00 rollover bar is outside the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_session.data_loader import load_h1, load_points
from ..fx_session.features import _window, hour_grid
from ..fx_session.session import session_calendar
from . import session_view

REASONS = ('incomplete session', 'unverified clock week', 'missing pre-session bar', 'corrupt spread', 'warm-up')


def load_frames(cfg: dict) -> tuple:
    view = session_view(cfg)
    start = pd.Timestamp(cfg['data']['history_start'], tz='UTC')
    frames = {}
    for p in cfg['factor_pairs']:
        f = load_h1(p, view)
        frames[p] = f[f.index >= start]
    pts = load_points(view)
    return frames, {p: float(pts[p]) for p in cfg['factor_pairs']}


def grids(h1: pd.DataFrame, dates) -> dict:
    """fx_session's hour grid plus a spread grid (n_dates x 24, NaN where a bar is missing)."""
    g = hour_grid(h1, dates)
    idx = pd.DatetimeIndex(h1.index)
    pos = pd.Index(dates).get_indexer(idx.normalize())
    keep = pos >= 0
    sp = np.full((len(dates), 24), np.nan)
    sp[pos[keep], idx.hour[keep]] = h1['spread'].to_numpy()[keep]
    g['spread'] = sp
    return g


def calendar(frames: dict, cfg: dict) -> pd.DataFrame:
    return session_calendar(frames, session_view(cfg))


def eligibility(frames: dict, points: dict, cfg: dict, cal: pd.DataFrame | None = None, gr: dict | None = None) -> pd.Series:
    """Reason per calendar date ('' = target-available), joint for both scored pairs."""
    cal = cal if cal is not None else calendar(frames, cfg)
    dates = cal.index
    gr = gr if gr is not None else {p: grids(frames[p], dates) for p in cfg['factor_pairs']}
    a, b = cfg['session']['pre_hours']
    pre_ok = np.ones(len(dates), dtype=bool)
    corrupt = np.zeros(len(dates), dtype=bool)
    limit = float(cfg['data']['max_spread_pips'])
    t_end = int(cfg['session']['target_hours'][1])
    for p in cfg['factor_pairs']:
        g = gr[p]
        pre_ok &= ~np.isnan(g['close'][:, a:b + 1]).any(axis=1)
        last = t_end if p in cfg['pairs'] else b
        pips = g['spread'][:, :last + 1] * points[p] / float(cfg['data']['pip'][p])
        corrupt |= np.nan_to_num(pips, nan=0.0).max(axis=1) > limit
    reason = cal['reason'].copy()
    reason[(reason == '') & ~pre_ok] = 'missing pre-session bar'
    reason[(reason == '') & corrupt] = 'corrupt spread'
    return reason


def targets(frames: dict, points: dict, cfg: dict, dates, gr: dict | None = None) -> dict:
    """{pair: DataFrame by date: y, y_bid, close14 (mid), close14_bid}; NaN where a window is incomplete."""
    a, b = cfg['session']['target_hours']
    h = int(cfg['session']['data_hour'])
    out = {}
    for p in cfg['pairs']:
        g = gr[p] if gr is not None else grids(frames[p], dates)
        half = g['spread'] * points[p] / 2.0
        mid = {k: g[k] + half for k in ('open', 'high', 'low', 'close')}
        w_mid, w_bid = _window(mid, a, b), _window(g, a, b)
        c14_mid, c14_bid = mid['close'][:, h], g['close'][:, h]
        with np.errstate(all='ignore'):
            y = np.log((w_mid['high'] - w_mid['low']) / c14_mid)
            y_bid = np.log((w_bid['high'] - w_bid['low']) / c14_bid)
        out[p] = pd.DataFrame({'y': y, 'y_bid': y_bid, 'close14': c14_mid, 'close14_bid': c14_bid},
                              index=pd.DatetimeIndex(dates, name='date'))
    return out


def dataset(frames: dict, points: dict, cfg: dict) -> dict:
    """{pair: features + y + y_bid + reason by calendar date}; reason '' = usable."""
    from .features import build_features
    cal = calendar(frames, cfg)
    dates = cal.index
    gr = {p: grids(frames[p], dates) for p in cfg['factor_pairs']}
    reason = eligibility(frames, points, cfg, cal, gr)
    tg = targets(frames, points, cfg, dates, gr)
    avail = reason == ''
    feats = build_features(frames, points, cfg, dates, {p: tg[p].loc[avail.to_numpy(), 'y'] for p in cfg['pairs']}, gr)
    warm = np.zeros(len(dates), dtype=bool)
    for p in cfg['pairs']:
        warm |= feats[p].isna().any(axis=1).to_numpy() | tg[p]['y'].isna().to_numpy()
    r = reason.copy()
    r[(r == '') & warm] = 'warm-up'
    out = {}
    for p in cfg['pairs']:
        d = feats[p].join(tg[p])
        d['reason'] = r.to_numpy()
        out[p] = d
    return out
