"""The fx-fibonacci feature matrix (spec fx-fibonacci/structure-features).

    build_features(h1, cfg) -> DataFrame indexed like h1, columns exactly
                               cfg['features']['columns'], in that order.

Every value at H1 row t depends only on H1 bars <= t and on D1 bars closed by t's close
(mtf_aligner), so the matrix is identical on a history truncated at t. Rows with any NaN
are warm-up; `valid_rows` marks the usable ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import fib_features as FF
from . import mtf_aligner as AL
from . import spiral_engine as SP
from . import swing_detector as SW
from . import tgd as TG


class FeatureColumnError(ValueError):
    pass


@dataclass
class Context:
    h1: pd.DataFrame
    d1: pd.DataFrame
    atr_h1: np.ndarray
    atr_d1: np.ndarray
    ph_h1: dict
    ph_d1_at: dict          # the D1 pivot history taken at each H1 row's newest closed D1 bar
    d1_quiet_at: pd.DataFrame
    j: np.ndarray           # newest closed D1 position per H1 row, -1 when none
    cur_day: np.ndarray     # j + 1, the D1 index being traded (NaN when j < 0)
    t_days: np.ndarray      # continuous time in trading days


def _take(arr: np.ndarray, j: np.ndarray, fill):
    out = arr[np.clip(j, 0, None)]
    if out.dtype.kind == 'f':
        out = out.copy()
        out[j < 0] = fill
    else:
        out = np.where((j < 0)[:, None] if out.ndim == 2 else j < 0, fill, out)
    return out


def build_context(h1: pd.DataFrame, cfg: dict) -> Context:
    period = cfg['atr_period']
    d1 = AL.build_d1(h1)
    atr_h1 = AL.atr(h1, period).to_numpy()
    atr_d1 = AL.atr(d1, period).to_numpy()
    ev_h1 = SW.swings_for(h1, 'H1', cfg)
    ev_d1 = SW.swings_for(d1, 'D1', cfg)
    ph_h1 = SW.pivot_history(ev_h1, len(h1), depth=4)
    ph_d1 = SW.pivot_history(ev_d1, len(d1), depth=4)
    quiet = SW.quiet_swing(d1['high'].to_numpy(), d1['low'].to_numpy(), ph_d1['idx'][:, 3],
                           quiet=cfg['swings']['quiet_bars_d1'])

    pos = d1[['close_time']].assign(pos=np.arange(len(d1)))
    j = AL.attach_d1(h1.index, pos)['pos'].fillna(-1).astype(int).to_numpy()

    td = AL.trading_date(h1.index)
    in_day = pd.Series(1, index=h1.index).groupby(td).cumsum().to_numpy()
    day_closed = np.zeros(len(h1), dtype=bool)
    ok = j >= 0
    day_closed[ok] = td[ok] == d1.index[j[ok]]
    cur_day = np.where(ok, j + 1, np.nan).astype(float)
    t_days = np.where(day_closed, cur_day, cur_day + np.minimum(in_day, 24) / 24.0)

    ph_d1_at = {'kind': _take(ph_d1['kind'], j, 0), 'idx': _take(ph_d1['idx'], j, -1),
                'price': _take(ph_d1['price'], j, np.nan)}
    q = quiet.iloc[np.clip(j, 0, None)].reset_index(drop=True)
    q.loc[j < 0, 'active'] = False
    return Context(h1, d1, atr_h1, atr_d1, ph_h1, ph_d1_at, q, j, cur_day, t_days)


def compute_blocks(ctx: Context, cfg: dict) -> dict:
    h1 = ctx.h1
    high, low, close = (h1[c].to_numpy() for c in ('high', 'low', 'close'))
    atr_d1_at = _take(ctx.atr_d1, ctx.j, np.nan)

    cols = {}
    d1_feat, d1_lv = FF.leg_features(ctx.ph_d1_at['price'], ctx.ph_d1_at['idx'], close, atr_d1_at, cfg, 'd1')
    h1_feat, h1_lv = FF.leg_features(ctx.ph_h1['price'], ctx.ph_h1['idx'], close, ctx.atr_h1, cfg, 'h1')
    cols.update(d1_feat)
    cols.update(h1_feat)
    br = FF.breakouts(high, low, close)
    cols.update(br)
    cols.update(FF.close_vs_extreme_bars(ctx.ph_h1['kind'], ctx.ph_h1['idx'], high, low, close, ctx.atr_h1))
    cols['d1_alignment_flag'] = FF.alignment_flag(br['h1_breakout_1'], close, d1_lv[0.618], d1_lv['ext'],
                                                  atr_d1_at, cfg['fib']['alignment_zone_atr_d1'])
    cols.update(FF.risk_features(high, low, close, ctx.atr_h1, h1_lv[0.5], cfg))

    q = ctx.d1_quiet_at
    proj = TG.projections(ctx.ph_d1_at['kind'], ctx.ph_d1_at['idx'], q['active'].to_numpy(),
                          q['first_idx'].to_numpy(dtype=float), q['second_idx'].to_numpy(dtype=float),
                          cfg['tgd']['ratio'])
    cols.update(TG.tgd_features(proj, ctx.cur_day, cols['d1_dist_to_fib_618'], cols['d1_dist_to_ext_1618_w3'], cfg))
    for variant in cfg['spiral']['variants']:
        cols.update(SP.spiral_features(ctx.ph_d1_at['kind'], ctx.ph_d1_at['idx'], ctx.ph_d1_at['price'],
                                       ctx.atr_d1, ctx.t_days, close, cfg, variant))
    return cols


def assemble(cols: dict, index, cfg: dict) -> pd.DataFrame:
    want = list(cfg['features']['columns'])
    extra = sorted(set(cols) - set(want))
    missing = [c for c in want if c not in cols]
    if extra:
        raise FeatureColumnError(f'feature column(s) not in the declared list: {extra}')
    if missing:
        raise FeatureColumnError(f'declared feature column(s) not built: {missing}')
    return pd.DataFrame({c: np.asarray(cols[c], dtype=float) for c in want}, index=index)


def build_features(h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return assemble(compute_blocks(build_context(h1, cfg), cfg), h1.index, cfg)


def valid_rows(features: pd.DataFrame) -> pd.Series:
    """True where every feature is finite (past the warm-up)."""
    return pd.Series(np.isfinite(features.to_numpy()).all(axis=1), index=features.index)
