"""Part 1a: Bollinger first breaches on gold D1 bars and their 3-day outcome (spec gold-bb-nyrange/bb-reversion;
design D5).

fx_bb's band, ATR and cooldown code is reused read-only on gold's valid NY-close D1 days (bid prices):

  bands    SMA20 +/- 2 * SD20 (population SD) of the closes of t-19 .. t
  ATR20    simple mean of the true range of t-19 .. t
  event    close[t] outside its bands while close[t-1] was inside the bands of t-1
           (dir -1 above the upper band, +1 below the lower band)
  cooldown accepted only when >= 3 valid days since the last accepted event
  outcome  r = dir * (close[t+3] - close[t]) / ATR20[t], t+3 the third valid day after t

`detect` stores only the DATE of t+3 and reads no close after t. `outcomes` is separate and is never
called by the label report. An event without t+3 in the window is excluded as `no outcome`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fx_bb.events import apply_cooldown, indicators
from .bars import valid


def breaches(v: pd.DataFrame, cfg: dict) -> tuple:
    """(indicator frame, positions of every first breach) on valid days `v`. No close after t is used."""
    ind = indicators(v, cfg)
    return ind, np.flatnonzero(ind['first_breach'].to_numpy())


def detect(d1: pd.DataFrame, cfg: dict, cooldown: bool = True) -> tuple:
    """(events, counts). `cooldown=False` keeps every first breach (Part 1b's entries)."""
    v = valid(d1)
    ind, first = breaches(v, cfg)
    acc, rem = apply_cooldown(first, int(cfg['cooldown_days'])) if cooldown else (list(first), [])
    h = int(cfg['horizon_days'])
    dates = v.index
    rows = []
    for k in acc:
        r = ind.iloc[k]
        rows.append({'date': dates[k], 'k': int(k), 'dir': int(r['dir']), 'side': -int(r['dir']), 'close': r['close'],
                     'sma': r['sma'], 'sd': r['sd'], 'atr': r['atr'],
                     'date_t3': dates[k + h] if k + h < len(dates) else pd.NaT})
    ev = pd.DataFrame(rows, columns=['date', 'k', 'dir', 'side', 'close', 'sma', 'sd', 'atr', 'date_t3'])
    ev['reason'] = np.where(ev['date_t3'].isna(), 'no outcome', '')
    counts = {'valid_days': len(v), 'outside_days': int(ind['outside'].sum()), 'first_breaches': len(first),
              'cooldown_removed': len(rem), 'up': int((ind['dir'].to_numpy()[first] == -1).sum()),
              'down': int((ind['dir'].to_numpy()[first] == 1).sum())}
    return ev, counts


# -- the outcome (never called by the label report) ------------------------------------------------

def outcome_r(direction, close_t, close_t3, atr):
    return direction * (close_t3 - close_t) / atr


def outcomes(ev: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    v = valid(d1)
    out = ev.copy()
    c3 = v['close'].reindex(pd.DatetimeIndex(out['date_t3'])).to_numpy()
    out['close_t3'] = c3
    out['r'] = outcome_r(out['dir'].to_numpy(float), out['close'].to_numpy(float), c3, out['atr'].to_numpy(float))
    out['y'] = np.where(np.isnan(out['r']), np.nan, (out['r'] > 0).astype(float))
    return out
