"""The simulated trade (spec fx-bb-sma-exit/trade-rule; design D2-D6).

Entry   at the mid close of day t on a FIRST breach of the Bollinger Bands (fx_bb's indicators):
        direction -1 above the upper band, +1 below the lower band.
Levels  TP = SMA20[t] (fixed); dist = |Close[t] - SMA20[t]|; SL = Close[t] - dir * dist.
Scan    the pair's H1 bars AFTER the entry bar (the closing bar of t), in time order, through the
        closing bar of valid day t+10. Mid high/low/open = bid + half the bar's own spread.
        Long: stop when mid_low <= SL, take profit when mid_high >= TP; short mirrored.
        A bar touching both is a STOP. TP fills at TP; the stop fills at SL, or at the mid open
        when the bar opens beyond SL.
Time    no touch by the closing bar of t+10 -> exit at the D1 mid close of t+10.
Costs   net = gross - (spread at entry bar + spread at exit bar) * point / 2. R = net / dist.
One     a first breach opens nothing while the pair's previous trade is open (exit after the
open    new entry bar). Excluded trades hold their slot until the closing bar of t+10.
Exclude `open at data end` (no t+10 yet, no touch), `pair gap` (fx_bb reference calendar over
        t-19 .. t+10), `corrupt spread` (entry bar .. closing bar of t+10, spread > 50 pips;
        the whole window, so the exclusion never depends on when the trade exits).

`build_stream` returns the trades with their EXIT TIMES only; `results` adds the exit reason,
price and result. The label report calls only `build_stream`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..fx_bb import events as BE
from ..fx_bb.daily import corrupt_spread, trading_dates, valid
from . import bb_view

STREAM_COLS = ['pair', 'date', 'k', 'dir', 'side', 'entry_pos', 'entry_time', 'close', 'sma', 'sd', 'atr', 'dist',
               'tp', 'sl', 'date_t19', 'date_t10', 'end_pos', 'exit_pos', 'touched', 'exit_date', 'reason']


@dataclass
class Ctx:
    pair: str
    point: float
    pip: float
    index: pd.DatetimeIndex     # H1 labels
    td: np.ndarray              # trading date of every H1 bar
    o: np.ndarray               # bid
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    sp: np.ndarray              # spread, points
    close_pos: pd.Series        # trading date -> position of its closing H1 bar
    v: pd.DataFrame             # valid D1 days
    ind: pd.DataFrame           # fx_bb indicators on v

    def mid(self, arr, i):
        return arr[i] + self.sp[i] * self.point / 2.0


def context(pair: str, h1: pd.DataFrame, point: float, d1: pd.DataFrame, cfg: dict) -> Ctx:
    bb = bb_view(cfg)
    td = trading_dates(h1.index, bb)
    pos = pd.Series(np.arange(len(h1)), index=td)
    v = valid(d1)
    return Ctx(pair, float(point), float(cfg['pip'][pair]), pd.DatetimeIndex(h1.index), td.to_numpy(),
               h1['open'].to_numpy(float), h1['high'].to_numpy(float), h1['low'].to_numpy(float),
               h1['close'].to_numpy(float), h1['spread'].to_numpy(float), pos.groupby(level=0).max(), v,
               BE.indicators(v, bb))


def contexts(frames: dict, points: dict, d1_all: dict, cfg: dict) -> dict:
    return {p: context(p, frames[p], points[p], d1_all[p], cfg) for p in cfg['pairs']}


def first_touch(hi: np.ndarray, lo: np.ndarray, a: int, b: int, direction: int, tp: float, sl: float) -> tuple:
    """(position of the first bar in [a, b] touching TP or SL, True) or (b, False)."""
    if a > b:
        return b, False
    h, l_ = hi[a:b + 1], lo[a:b + 1]
    hit = (l_ <= sl) | (h >= tp) if direction > 0 else (h >= sl) | (l_ <= tp)
    if hit.any():
        return a + int(np.argmax(hit)), True
    return b, False


def build_stream(ctxs: dict, d1_all: dict, cfg: dict) -> tuple:
    """(trades frame with exit times and exclusion `reason`, per-pair counts). No result is computed."""
    bb = bb_view(cfg)
    ref = BE.reference_calendar(d1_all, bb)
    n_days = int(cfg['trade']['time_stop_days'])
    lo_w = -int(cfg['trade']['exclusion_window'][0])
    mult = float(cfg['trade']['stop_multiple'])
    rows, counts = [], {}
    for pair in cfg['pairs']:
        x = ctxs[pair]
        mh = x.h + x.sp * x.point / 2.0
        ml = x.lo + x.sp * x.point / 2.0
        bad = corrupt_spread(x.sp, x.point, x.pip, cfg)
        first = np.flatnonzero(x.ind['first_breach'].to_numpy())
        block = -1                      # H1 position until which the pair is busy
        skipped = 0
        dates = x.v.index
        for k in first:
            date = dates[k]
            e = int(x.close_pos[date])
            if block > e:
                skipped += 1
                continue
            r = x.ind.iloc[k]
            d = int(r['dir'])
            close, sma = float(r['close']), float(r['sma'])
            dist = abs(close - sma) * mult
            tp, sl = sma, close - d * dist
            has10 = k + n_days < len(dates)
            date10 = dates[k + n_days] if has10 else pd.NaT
            end = int(x.close_pos[date10]) if has10 else len(x.index) - 1
            pos, touched = first_touch(mh, ml, e + 1, end, d, tp, sl)
            exit_date = pd.Timestamp(x.td[pos]) if touched else date10
            win_end = date10 if has10 else (exit_date if touched else dates[-1])
            gap = BE.pair_gap(pd.DataFrame({'date_t19': [dates[k - lo_w]], 'date_t3': [win_end], 'date': [date]}),
                              dates, ref)[0]
            if not touched and not has10:
                reason = 'open at data end'
            elif gap:
                reason = 'pair gap'
            elif bad[e:end + 1].any():
                reason = 'corrupt spread'
            else:
                reason = ''
            block = (end if has10 else len(x.index)) if reason else pos
            rows.append({'pair': pair, 'date': date, 'k': int(k), 'dir': d, 'side': -d, 'entry_pos': e,
                         'entry_time': x.index[e], 'close': close, 'sma': sma, 'sd': float(r['sd']),
                         'atr': float(r['atr']), 'dist': dist, 'tp': tp, 'sl': sl, 'date_t19': dates[k - lo_w],
                         'date_t10': date10, 'end_pos': end, 'exit_pos': pos, 'touched': bool(touched),
                         'exit_date': exit_date, 'reason': reason})
        counts[pair] = {'first_breaches': len(first), 'trades': sum(1 for r in rows if r['pair'] == pair),
                        'skipped_open': skipped, 'valid_days': len(dates)}
    tr = pd.DataFrame(rows, columns=STREAM_COLS)
    tr = tr.sort_values(['date', 'pair'], kind='mergesort').reset_index(drop=True)
    return tr, counts


def _fill(x: Ctx, i: int, d: int, tp: float, sl: float, o, h, lo) -> tuple:
    """(reason, fill, conflict) on bar i, which touched a level. Stop wins a conflict."""
    sl_hit = lo[i] <= sl if d > 0 else h[i] >= sl
    tp_hit = h[i] >= tp if d > 0 else lo[i] <= tp
    if sl_hit:
        fill = min(sl, o[i]) if d > 0 else max(sl, o[i])
        return 'stop loss', fill, bool(tp_hit)
    return 'take profit', tp, False


def results(tr: pd.DataFrame, ctxs: dict, cfg: dict) -> pd.DataFrame:
    """Add the exit reason, prices, spreads and results (mid, net of half spreads) and the bid/ask control."""
    out = tr.copy()
    cols = {c: [] for c in ('exit_reason', 'exit_time', 'exit_price', 'conflict', 'spread_entry_pips', 'spread_exit_pips',
                            'gross_pips', 'net_pips', 'r_gross', 'r_net', 'r_ctrl', 'hold_hours')}
    for _, t in tr.iterrows():
        x = ctxs[t['pair']]
        e, i, d, tp, sl, dist = int(t['entry_pos']), int(t['exit_pos']), int(t['dir']), t['tp'], t['sl'], t['dist']
        if t['reason'] == 'open at data end':
            for c in cols:
                cols[c].append(np.nan if c not in ('exit_reason', 'exit_time') else ('' if c == 'exit_reason' else pd.NaT))
            continue
        mo, mh, ml = (x.o + x.sp * x.point / 2.0, x.h + x.sp * x.point / 2.0, x.lo + x.sp * x.point / 2.0)
        if t['touched']:
            reason, fill, conflict = _fill(x, i, d, tp, sl, mo, mh, ml)
        else:
            reason, fill, conflict = 'time stop', float(x.v['close'].iloc[t['k'] + int(cfg['trade']['time_stop_days'])]), False
        gross = d * (fill - t['close'])
        cost = (x.sp[e] + x.sp[i]) * x.point / 2.0
        net = gross - cost
        cols['exit_reason'].append(reason)
        cols['exit_time'].append(x.index[i])
        cols['exit_price'].append(fill)
        cols['conflict'].append(conflict)
        cols['spread_entry_pips'].append(x.sp[e] * x.point / x.pip)
        cols['spread_exit_pips'].append(x.sp[i] * x.point / x.pip)
        cols['gross_pips'].append(gross / x.pip)
        cols['net_pips'].append(net / x.pip)
        cols['r_gross'].append(gross / dist)
        cols['r_net'].append(net / dist)
        cols['r_ctrl'].append(control_r(x, t, cfg))
        cols['hold_hours'].append((x.index[i] - x.index[e]) / pd.Timedelta(hours=1))
    for c, vals in cols.items():
        out[c] = vals
    return out


def control_r(x: Ctx, t, cfg: dict) -> float:
    """Bid/ask fills (spec: Bid/ask control fill): a long enters at the ask and exits on the bid,
    a short enters at the bid and exits on the ask; same levels, same scan window, stop wins."""
    e, end, d, tp, sl = int(t['entry_pos']), int(t['end_pos']), int(t['dir']), t['tp'], t['sl']
    add = x.sp * x.point
    if d > 0:
        entry = x.c[e] + add[e]
        o, h, lo, c = x.o, x.h, x.lo, x.c                     # exits on the bid
    else:
        entry = x.c[e]
        o, h, lo, c = x.o + add, x.h + add, x.lo + add, x.c + add  # exits on the ask
    pos, touched = first_touch(h, lo, e + 1, end, d, tp, sl)
    if touched:
        _, fill, _ = _fill(x, pos, d, tp, sl, o, h, lo)
    elif pd.notna(t['date_t10']):
        fill = c[end]                                          # time stop at the closing bar of t+10
    else:
        return float('nan')
    return float(d * (fill - entry) / t['dist'])


def load_inputs(cfg: dict) -> tuple:
    """(contexts, D1 frames) from the fx_bb caches, after fx_bb's clock and lag checks."""
    from ..fx_bb.daily import build_all
    from ..fx_bb.data_loader import load_all
    bb = bb_view(cfg)
    frames, points = load_all(bb)
    d1_all = build_all(frames, points, bb)
    return contexts(frames, points, d1_all, cfg), d1_all


def cache_digests(cfg: dict) -> dict:
    from ..fx_bb.data_loader import cache_digests as cd
    return cd(bb_view(cfg))


def summary(cfg: dict | None = None) -> pd.DataFrame:
    """Trade counts per pair and exclusions by reason on the real caches. Prints no exit reason or result."""
    from . import load_config
    cfg = cfg or load_config()
    ctxs, d1_all = load_inputs(cfg)
    tr, counts = build_stream(ctxs, d1_all, cfg)
    tab = pd.DataFrame(counts).T
    tab['eligible'] = tr[tr['reason'] == ''].groupby('pair').size().reindex(tab.index).fillna(0).astype(int)
    print(tab.to_string())
    print('exclusions:', tr.loc[tr['reason'] != '', 'reason'].value_counts().to_dict())
    print(f"trades {len(tr)}, eligible {int((tr['reason'] == '').sum())}, distinct eligible entry dates "
          f"{tr.loc[tr['reason'] == '', 'date'].nunique()}")
    return tab


if __name__ == '__main__':
    summary()
