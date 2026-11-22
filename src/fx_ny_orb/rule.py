"""The New York opening-range breakout (spec fx-ny-orb-hot/orb-rule; design D3).

One day's M15 mid bars (labels on the Berlin clock) and the forecast F (price units):

  range      highest mid high / lowest mid low of the bars labelled 14:00, 14:15, 14:30, 14:45
             (all four needed)                                   -> else 'no range'
  width      range width <= 0.50 F                               -> else 'range too wide'
  breakout   the FIRST bar labelled >= 15:00 (and <= 20:45) whose mid close is above the range
             high (long) or below the range low (short)          -> else 'no breakout'
  distance   that close is <= 0.25 F beyond the broken level     -> else 'breakout too far'
  entry      the next bar's mid open, the entry bar labelled before 19:00 -> else 'too late'

Only the first breakout counts. `entries` stops here and never reads a price after the entry bar's
open; `scan_exit` does the rest:

  stop 0.25 F against, target 0.60 F with the direction, scanned from the entry bar through the
  bar labelled 20:45; a bar reaching both is a STOP; the target fills at its level; the stop at its
  level or at the bar's mid open when the bar opens beyond it; no touch -> the 20:45 bar's mid close.
  net = gross - half the entry bar's spread - half the exit bar's spread - slippage_bps of the
  entry price. R = net / (0.25 F). One trade per day; a stopped trade never re-enters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import minutes

TRADE_COLS = ['direction', 'or_high', 'or_low', 'breakout_time', 'entry_time', 'entry', 'stop', 'target',
              'exit_time', 'exit', 'exit_reason', 'conflict', 'gross_pips', 'net_pips', 'r_gross', 'r_net']


def _mins(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return idx.hour.to_numpy() * 60 + idx.minute.to_numpy()


def entries(day: pd.DataFrame, F: float, cfg: dict) -> dict:
    """The entry decision for one day: {'reason': ''|why not, ...}. Reads nothing after the entry bar's open."""
    r = cfg['rule']
    day = day.sort_index()
    m = _mins(day.index)
    or_m = [minutes(x) for x in r['or_bars']]
    if not np.isfinite(F) or F <= 0:
        return {'reason': 'no forecast'}
    rng = day[np.isin(m, or_m)]
    if len(rng) != len(or_m) or rng[['mh', 'ml']].isna().any().any():
        return {'reason': 'no range'}
    hi, lo = float(rng['mh'].max()), float(rng['ml'].min())
    out = {'or_high': hi, 'or_low': lo}
    if hi - lo > float(r['or_width_max_f']) * F:
        return {**out, 'reason': 'range too wide'}
    after = np.flatnonzero((m >= minutes(r['breakout_from'])) & (m <= minutes(r['last_bar'])))
    mc = day['mc'].to_numpy(float)
    brk = next((i for i in after if mc[i] > hi or mc[i] < lo), None)
    if brk is None:
        return {**out, 'reason': 'no breakout'}
    d = 1 if mc[brk] > hi else -1
    beyond = mc[brk] - hi if d > 0 else lo - mc[brk]
    out.update(direction=d, breakout_time=day.index[brk])
    if beyond > float(r['breakout_max_f']) * F:
        return {**out, 'reason': 'breakout too far'}
    e = brk + 1
    if e >= len(day) or m[e] >= minutes(r['entry_before']):
        return {**out, 'reason': 'too late'}
    entry = float(day['mo'].iloc[e])
    return {**out, 'reason': '', 'entry_pos': int(e), 'entry_time': day.index[e], 'entry': entry,
            'stop': entry - d * float(r['stop_f']) * F, 'target': entry + d * float(r['target_f']) * F}


def scan_exit(day: pd.DataFrame, ent: dict, F: float, pip: float, cfg: dict) -> dict:
    """The exit of an entered trade, from the entry bar through the bar labelled 20:45."""
    r = cfg['rule']
    day = day.sort_index()
    m = _mins(day.index)
    d, e = int(ent['direction']), int(ent['entry_pos'])
    stop, target = ent['stop'], ent['target']
    last = np.flatnonzero(m <= minutes(r['last_bar']))
    end = int(last[-1]) if len(last) else e
    mh, ml, mo = (day[c].to_numpy(float) for c in ('mh', 'ml', 'mo'))
    reason, fill, conflict, x = 'forced exit', float(day['mc'].iloc[end]), False, end
    for i in range(e, end + 1):
        sl = ml[i] <= stop if d > 0 else mh[i] >= stop
        tp = mh[i] >= target if d > 0 else ml[i] <= target
        if sl or tp:
            x, conflict = i, bool(sl and tp)
            if sl:
                reason, fill = 'stop loss', (min(stop, mo[i]) if d > 0 else max(stop, mo[i]))
            else:
                reason, fill = 'take profit', target
            break
    sp = day['spread_price'].to_numpy(float)
    gross = d * (fill - ent['entry'])
    net = gross - sp[e] / 2 - sp[x] / 2 - float(r['slippage_bps']) * 1e-4 * ent['entry']
    risk = float(r['stop_f']) * F
    return {'exit_time': day.index[x], 'exit': fill, 'exit_reason': reason, 'conflict': conflict,
            'gross_pips': gross / pip, 'net_pips': net / pip, 'r_gross': gross / risk, 'r_net': net / risk}


def simulate_day(day: pd.DataFrame, F: float, hot: bool, pip: float, cfg: dict) -> dict:
    ent = entries(day, F, cfg)
    row = {'hot': bool(hot), 'F': F, 'reason': ent['reason'], **{k: ent.get(k) for k in
                                                                 ('direction', 'or_high', 'or_low', 'breakout_time')}}
    if ent['reason']:
        return row
    row.update({k: ent[k] for k in ('entry_time', 'entry', 'stop', 'target')})
    row.update(scan_exit(day, ent, F, pip, cfg))
    return row
