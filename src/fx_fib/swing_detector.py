"""Robert Fischer's swing detector (spec fx-fibonacci/structure-features).

One pass over the bars emits pivot EVENTS:

    kind           'H' or 'L'
    extreme_idx    the bar that made the high / low
    extreme_price  that bar's high / low
    confirm_idx    the bar at whose close the event becomes known
    replaces       True when the event moves the previous same-kind pivot to a
                   more extreme bar (the swing simply continued)

A swing high is the highest bar since the last low pivot. A LATER bar confirms it
when all hold at that bar:
  1. leg size:   high[c] - low_pivot >= k_atr * ATR[t]
  2. reversal:   some close since c is below low[c], OR the counter-move since c
                 is >= k_atr * ATR[t], OR it retraces >= 38.2 % of the leg
  3. 3-bar rule: the two pivots span >= 3 bars counting both pivot bars, i.e.
                 c - low_pivot_idx >= 2 (owner's choice B, 2026-11-21)
Swing lows mirror this. The bar that sets an extreme never confirms it.

If price makes a new extreme beyond the last pivot before the opposite pivot
confirms (a lower low while seeking a high), the last pivot moves there. That is a
`replaces` event at that bar; earlier bars keep seeing the old pivot.

Before the first pivot, the newer of the running highest-high and lowest-low is the
candidate and the older one anchors its leg (the anchor itself is not emitted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mtf_aligner import atr as wilder_atr

EVENT_COLUMNS = ['kind', 'extreme_idx', 'extreme_price', 'confirm_idx', 'replaces']


def _check_high(high, low, close, atr, a, A, t, k, retrace, min_bars):
    """Candidate high after anchor (a, A) at bar t: (cand, confirmed)."""
    c = a + 1 + int(np.argmax(high[a + 1:t + 1]))
    if c == t:
        return c, False
    ms = k * atr[t]
    move = high[c] - A
    counter = high[c] - low[c + 1:t + 1].min()
    reversal = (close[c + 1:t + 1].min() < low[c]) or (counter >= ms) or (counter >= retrace * move)
    return c, bool(move >= ms and reversal and c - a >= min_bars)


def _check_low(high, low, close, atr, a, A, t, k, retrace, min_bars):
    c = a + 1 + int(np.argmin(low[a + 1:t + 1]))
    if c == t:
        return c, False
    ms = k * atr[t]
    move = A - low[c]
    counter = high[c + 1:t + 1].max() - low[c]
    reversal = (close[c + 1:t + 1].max() > high[c]) or (counter >= ms) or (counter >= retrace * move)
    return c, bool(move >= ms and reversal and c - a >= min_bars)


def detect_swings(high, low, close, atr, k_atr: float, retrace: float = 0.382,
                  min_bars: int = 2) -> pd.DataFrame:
    """Pivot events, in confirmation order. Pure: depends on bars <= each confirm_idx only."""
    high, low, close, atr = (np.asarray(x, dtype=float) for x in (high, low, close, atr))
    n = len(high)
    events = []
    seeking = None                          # None (no pivot yet), 'H' or 'L'
    a = A = None                            # last pivot (index, price)
    for t in range(n):
        if not np.isfinite(atr[t]):
            if seeking == 'H' and low[t] < A:
                a, A = t, low[t]
                events.append(('L', t, low[t], t, True))
            elif seeking == 'L' and high[t] > A:
                a, A = t, high[t]
                events.append(('H', t, high[t], t, True))
            continue
        if seeking is None:
            hc = int(np.argmax(high[:t + 1]))
            lc = int(np.argmin(low[:t + 1]))
            if hc > lc:
                c, ok = _check_high(high, low, close, atr, lc, low[lc], t, k_atr, retrace, min_bars)
                if ok and c == hc:
                    events.append(('H', c, high[c], t, False))
                    seeking, a, A = 'L', c, high[c]
            elif lc > hc:
                c, ok = _check_low(high, low, close, atr, hc, high[hc], t, k_atr, retrace, min_bars)
                if ok and c == lc:
                    events.append(('L', c, low[c], t, False))
                    seeking, a, A = 'H', c, low[c]
            continue
        if seeking == 'H':
            if low[t] < A:                  # the down-swing continued: move the low
                a, A = t, low[t]
                events.append(('L', t, low[t], t, True))
                continue
            c, ok = _check_high(high, low, close, atr, a, A, t, k_atr, retrace, min_bars)
            if ok:
                events.append(('H', c, high[c], t, False))
                seeking, a, A = 'L', c, high[c]
        else:
            if high[t] > A:                 # the up-swing continued: move the high
                a, A = t, high[t]
                events.append(('H', t, high[t], t, True))
                continue
            c, ok = _check_low(high, low, close, atr, a, A, t, k_atr, retrace, min_bars)
            if ok:
                events.append(('L', c, low[c], t, False))
                seeking, a, A = 'H', c, low[c]
    out = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return out.astype({'extreme_idx': int, 'confirm_idx': int, 'extreme_price': float, 'replaces': bool})


def swings_for(df: pd.DataFrame, timeframe: str, cfg: dict) -> pd.DataFrame:
    """Events for one timeframe's OHLC frame, with that timeframe's own k_atr and ATR."""
    sw = cfg['swings']
    k = {'D1': sw['k_atr_d1'], 'H1': sw['k_atr_h1']}[timeframe]
    a = wilder_atr(df, cfg['atr_period']).to_numpy()
    return detect_swings(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), a,
                         k_atr=k, retrace=sw['retrace_confirm'], min_bars=sw['min_bars_between_pivots'])


def visible_pivots(events: pd.DataFrame, t: int) -> list:
    """The alternating pivot list known at the close of bar t: [(kind, idx, price), ...]."""
    out = []
    for kind, idx, price, conf, rep in events.itertuples(index=False):
        if conf > t:
            break
        if rep and out and out[-1][0] == kind:
            out[-1] = (kind, idx, price)
        else:
            out.append((kind, idx, price))
    return out


def pivot_history(events: pd.DataFrame, n: int, depth: int = 4) -> dict:
    """For every bar t, the last `depth` visible pivots, oldest first.

    Returns arrays of shape (n, depth): `kind` (+1 high, -1 low, 0 none), `idx`
    (-1 none) and `price` (NaN none). Column depth-1 is the newest pivot."""
    kind = np.zeros((n, depth), dtype=np.int8)
    idx = np.full((n, depth), -1, dtype=np.int64)
    price = np.full((n, depth), np.nan)
    pivots = []
    ev = events.itertuples(index=False)
    nxt = next(ev, None)
    for t in range(n):
        while nxt is not None and nxt.confirm_idx <= t:
            item = (1 if nxt.kind == 'H' else -1, nxt.extreme_idx, nxt.extreme_price)
            if nxt.replaces and pivots and pivots[-1][0] == item[0]:
                pivots[-1] = item
            else:
                pivots.append(item)
                if len(pivots) > depth:
                    pivots.pop(0)
            nxt = next(ev, None)
        for j, (k_, i_, p_) in enumerate(pivots):
            col = depth - len(pivots) + j
            kind[t, col], idx[t, col], price[t, col] = k_, i_, p_
    return {'kind': kind, 'idx': idx, 'price': price}


def quiet_swing(high, low, last_pivot_idx, quiet: int = 15) -> pd.DataFrame:
    """Fischer's 15-bar rule (D1, for the TGD features only).

    At bar t, when at least `quiet` bars have closed since the newest visible pivot's
    extreme (or no pivot exists yet), the swing is the highest high and the lowest low
    of the last `quiet` bars, in time order. Inactive rows are NaN / -1."""
    high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    last_pivot_idx = np.asarray(last_pivot_idx)
    n = len(high)
    cols = {c: np.full(n, -1, dtype=np.int64) for c in ('first_idx', 'second_idx')}
    cols.update({c: np.zeros(n, dtype=np.int8) for c in ('first_kind', 'second_kind')})
    cols.update({c: np.full(n, np.nan) for c in ('first_price', 'second_price')})
    active = np.zeros(n, dtype=bool)
    for t in range(quiet - 1, n):
        p = last_pivot_idx[t]
        if p >= 0 and t - p < quiet:
            continue
        lo_w = t - quiet + 1
        hi_i = lo_w + int(np.argmax(high[lo_w:t + 1]))
        lo_i = lo_w + int(np.argmin(low[lo_w:t + 1]))
        pts = sorted([(hi_i, 1, high[hi_i]), (lo_i, -1, low[lo_i])])
        active[t] = True
        (cols['first_idx'][t], cols['first_kind'][t], cols['first_price'][t]) = pts[0]
        (cols['second_idx'][t], cols['second_kind'][t], cols['second_price'][t]) = pts[1]
    return pd.DataFrame({'active': active, **cols})
