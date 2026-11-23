"""Part 1b: the Bollinger trade with an SMA20 exit on gold (spec gold-bb-nyrange/bb-sma-trade; design D5).

  entry      the D1 close (bid) of every first breach (no cooldown), dir -1 above the upper band, +1 below
  levels     take profit = SMA20[t]; stop = entry - dir * |SMA20[t] - entry|
  exit scan  H1 bars after the entry close, in order, through the last H1 bar of t+10 (tenth valid day):
               a bar opening beyond the stop fills at its open; a bar opening beyond the take profit fills at
               the take-profit level; otherwise a bar touching both is a stop; otherwise the level touched
  time stop  the close of the last H1 bar of t+10
  result     R = dir * (exit - entry) / |entry - stop|, gross of cost; a win is R > 0
  one trade  a breach opens nothing while the previous trade is open; a trade exiting at or before the
             new entry close does not block; an excluded trade blocks through the close of its t+10
  excluded   `no exit data` when t+10 is not in the window; `warm-up` (model cell only) as in Part 1a

`entries` reads no price after t. `build_stream` runs the exit scan and is never called by the label report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bb_events as E
from .bars import h1_day_positions, valid

REASONS = ('no exit data', 'warm-up', 'thin fold')


@dataclass
class Ctx:
    dates: pd.DatetimeIndex          # valid D1 days
    last_bar: np.ndarray             # position of each valid day's last H1 bar
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    times: pd.DatetimeIndex


def context(h1: pd.DataFrame, d1: pd.DataFrame, cfg: dict) -> Ctx:
    v = valid(d1)
    td = pd.DatetimeIndex(h1_day_positions(h1, cfg))
    pos = pd.Series(np.arange(len(h1)), index=td)
    last = pos.groupby(level=0).max().reindex(v.index).to_numpy()
    if np.isnan(last.astype(float)).any():
        raise ValueError('a valid D1 day has no H1 bar')
    return Ctx(v.index, last.astype(int), h1['open'].to_numpy(float), h1['high'].to_numpy(float),
               h1['low'].to_numpy(float), h1['close'].to_numpy(float), pd.DatetimeIndex(h1.index))


def entries(d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every first breach with its levels; dates only after t (date_t10). No price after t is read."""
    ev, _ = E.detect(d1, cfg, cooldown=False)
    n = int(cfg['trade']['time_stop_days'])
    dates = valid(d1).index
    ev['entry'] = ev['close']
    dist = (ev['sma'] - ev['entry']).abs() * float(cfg['trade']['stop_multiple'])
    ev['tp'] = ev['sma']
    ev['sl'] = ev['entry'] - ev['dir'] * dist
    ev['date_t10'] = [dates[k + n] if k + n < len(dates) else pd.NaT for k in ev['k']]
    ev['reason'] = np.where(ev['date_t10'].isna(), 'no exit data', '')
    return ev.drop(columns=['date_t3'])


def scan(x: Ctx, a: int, b: int, d: int, tp: float, sl: float) -> tuple:
    """(exit position, exit price, reason) scanning bars a..b; time stop at the close of b."""
    for i in range(a, b + 1):
        o, hi, lo = x.o[i], x.h[i], x.l[i]
        if (d > 0 and o <= sl) or (d < 0 and o >= sl):
            return i, o, 'stop'
        if (d > 0 and o >= tp) or (d < 0 and o <= tp):
            return i, tp, 'take profit'
        hit_sl = lo <= sl if d > 0 else hi >= sl
        hit_tp = hi >= tp if d > 0 else lo <= tp
        if hit_sl:
            return i, sl, 'stop'
        if hit_tp:
            return i, tp, 'take profit'
    return b, x.c[b], 'time stop'


def build_stream(ev: pd.DataFrame, x: Ctx, cfg: dict) -> pd.DataFrame:
    """Apply the one-open-trade rule and scan every opened trade. Returns the opened and excluded trades
    with exit time, price, reason and R; blocked breaches are dropped and counted in attrs."""
    n = int(cfg['trade']['time_stop_days'])
    rows, busy_until, blocked = [], -1, 0
    for _, r in ev.sort_values('date').iterrows():
        k = int(r['k'])
        p_entry = int(x.last_bar[k])
        if p_entry < busy_until:
            blocked += 1
            continue
        rec = r.to_dict()
        rec['entry_time'] = x.times[p_entry]
        if r['reason'] != '':
            busy_until = len(x.o) + 1                     # no t+10: blocks to the end of the window
            rec.update(exit_time=pd.NaT, exit=np.nan, exit_reason='', R=np.nan, date_exit=pd.NaT)
            rows.append(rec)
            continue
        b = int(x.last_bar[k + n])
        i, px, why = scan(x, p_entry + 1, b, int(r['dir']), float(r['tp']), float(r['sl']))
        busy_until = i
        risk = abs(float(r['entry']) - float(r['sl']))
        rec.update(exit_time=x.times[i], exit=px, exit_reason=why, R=int(r['dir']) * (px - float(r['entry'])) / risk,
                   date_exit=x.dates[np.searchsorted(x.last_bar, i)])
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out['y'] = np.where(out['R'].isna(), np.nan, (out['R'] > 0).astype(float))
    out.attrs['blocked'] = blocked
    return out
