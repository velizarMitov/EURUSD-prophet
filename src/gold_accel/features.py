"""Acceleration features and model inputs (design D2, spec "Acceleration features
are causal and scale-free").

The H1 move is r_t = 100 * ln(close_t / close_{t-1}), in %.

  accel_1h             |r_t| - |r_{t-1}|
  activity_rel         |r_t| / mean |r| at the same label hour over that hour's
                       previous 60 occurrences (about 60 trading days), today's
                       excluded
  accel_expected_next  the same trailing profile at hour(t + 1 h) minus at hour(t)
  accel_m15            inside bar t: |m4| - mean(|m1|, |m2|, |m3|), m_i the
                       M15 log moves (%); NaN unless all four bars exist

Every column at t reads bars labelled <= t only, and the profile is a trailing
window, so there is nothing to fit and the inputs can be built once over the
whole cache. All are log-move based, so a price rescale leaves them unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast_eval.challengers.base import Inputs
from src.gold import features as GF
from src.gold import targets as GT

M15_NS = 15 * 60 * 10 ** 9


def moves(close: pd.Series) -> pd.Series:
    return 100 * np.log(close / close.shift(1))


def trailing_profile(index: pd.DatetimeIndex, absr: pd.Series, at_hours: np.ndarray, days: int) -> np.ndarray:
    """For each bar t, the mean |r| of the latest `days` occurrences of label hour
    at_hours[t] strictly before t; NaN until that hour has `days` occurrences."""
    occ = pd.DataFrame({'time': index, 'hour': index.hour, 'absr': absr.to_numpy()})
    occ['prof'] = occ.groupby('hour')['absr'].transform(lambda s: s.rolling(days, min_periods=days).mean())
    occ = occ[['time', 'hour', 'prof']].sort_values('time')
    left = pd.DataFrame({'time': index, 'hour': np.asarray(at_hours, dtype=occ['hour'].dtype),
                         'row': np.arange(len(index))}).sort_values('time')
    m = pd.merge_asof(left, occ, on='time', by='hour', allow_exact_matches=False, direction='backward')
    out = np.full(len(index), np.nan)
    out[m['row'].to_numpy()] = m['prof'].to_numpy()
    return out


def h1_accel(h1: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    idx = pd.DatetimeIndex(h1.index)
    r = moves(h1['close'])
    absr = r.abs()
    hour = idx.hour.to_numpy()
    same = trailing_profile(idx, absr, hour, days)
    nxt = trailing_profile(idx, absr, (hour + 1) % 24, days)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(same > 0, absr.to_numpy() / same, np.nan)
    return pd.DataFrame({'accel_1h': (absr - absr.shift(1)).to_numpy(), 'activity_rel': rel,
                         'accel_expected_next': nxt - same}, index=h1.index)


def m15_accel(h1_index: pd.DatetimeIndex, m15: pd.DataFrame) -> pd.Series:
    """accel_m15 for each H1 label from the M15 bars labelled t, t+15, t+30, t+45."""
    m = m15.sort_index()
    t = pd.DatetimeIndex(m.index)
    mv = moves(m['close'])
    gap = np.diff(t.asi8, prepend=t.asi8[0] - M15_NS) != M15_NS   # a move across a gap is unknown
    mv[gap] = np.nan
    frame = pd.DataFrame({'hour': t.floor('h'), 'slot': t.minute // 15, 'abs': mv.abs().to_numpy()})
    wide = frame.pivot_table(index='hour', columns='slot', values='abs', aggfunc='first')
    wide = wide.reindex(columns=[0, 1, 2, 3])
    acc = wide[3] - wide[[0, 1, 2]].mean(axis=1, skipna=False)
    acc[wide.isna().any(axis=1)] = np.nan
    return acc.reindex(pd.DatetimeIndex(h1_index)).rename('accel_m15')


def feature_frame(h1: pd.DataFrame, m15: pd.DataFrame | None, days: int = 60) -> pd.DataFrame:
    """The 15 registered H1 columns, the H1 acceleration columns and accel_m15."""
    base = GF.h1_features(h1)
    out = pd.concat([base, h1_accel(h1, days)], axis=1)
    out['accel_m15'] = m15_accel(h1.index, m15) if m15 is not None and len(m15) else np.nan
    return out


def columns_for(model: str, cfg: dict) -> list[str]:
    spec = cfg['models'][model]['columns']
    f = cfg['features']
    parts = {'base': GF.h1_feature_columns(), 'h1': f['h1'], 'm15': f['m15'], 'calendar': f['calendar']}
    cols = [c for key in spec.split('+') for c in parts[key]]
    return GF.assert_scale_free(cols)


def build_inputs(h1: pd.DataFrame, frame: pd.DataFrame, columns: list[str]) -> Inputs:
    """Gold Inputs over exactly `columns`, with gold's H1 eligibility rule."""
    X = frame[columns].to_numpy(float)
    return Inputs(index=pd.DatetimeIndex(h1.index), close=h1['close'].to_numpy(float), X=X,
                  extra={'eligible': GT.eligible(h1.index, 'H1'), 'complete': ~np.isnan(X).any(axis=1)})
