"""The base trade: one RTS spot recommendation replayed on the owner's MT5 M1 bars (spec
rts-signals/trade-simulation; rule-book; design D5).

    python -m src.rts_signals.simulate      rebuild results/rts_signals/base_trades.csv

Session: from 18:00 New York on the report date to 16:00 New York on the next weekday (12:00 on a
Friday), on the Berlin label clock (rules.session).

Entry
  * Rule 0 (entry = the 4 pm price): at the open of the first bar of the session. If that open is
    `proportionality_points` (15) or more from the 4 pm price, the stop and target keep their point
    distances from the actual open.
  * Rule 1, guerilla (entry != the 4 pm price): a limit at the entry, filled when a bar touches it after the
    open. If the session opens already through it in the limit's favour, it fills at that open. If the
    open is already beyond the stop, there is no trade (`open beyond stop`). Never touched: `entry not
    reached`.

Exit, scanned minute by minute on bid bars:
  * stop when the low (buy) / high (sell) reaches it, at the stop or at a worse bar open (gap);
  * target when the high (buy) / low (sell) reaches it, at the target;
  * a bar that touches both counts as the stop; in a guerilla's fill bar only the stop is checked;
  * otherwise at the close of the last bar before the session end (time stop).

Gross is on bid prices. Net subtracts the recorded spread once: a buy pays it at entry (it buys at
bid + spread), a sell at exit. Nothing else (slippage, swap, commission) is modelled.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import COST_HEADER, load_config, results_dir
from .rules import load_rules, session

BASE = 'base_trades.csv'
FIELDS = ['report_date', 'trade_date', 'pair', 'symbol', 'direction', 'entry_type', 'status', 'exit_reason',
          'close_4pm', 'mt5_close_4pm', 'close_diff_points', 'open_time', 'open_price', 'open_gap_points',
          'proportional', 'entry_time', 'entry', 'stop', 'target', 'exit_time', 'exit', 'spread_cost', 'r_dist',
          'gross_pips', 'net_pips', 'gross_r', 'net_r', 'session_end']


def _bar_window(bars: pd.DataFrame, start, end) -> pd.DataFrame:
    return bars[(bars.index >= start) & (bars.index < end)]


def data_problem(w: pd.DataFrame, s: dict, cfg: dict) -> str:
    """'' when the window is usable, else 'no price data'."""
    sim = cfg['simulation']
    if not len(w):
        return 'no price data'
    first_gap = (w.index[0] - s['open']) / pd.Timedelta(minutes=1)
    weekend_open = s['open'].dayofweek >= 5
    if first_gap > sim['first_bar_within_minutes'] and not weekend_open:
        return 'no price data'
    if (s['end'] - w.index[-1]) / pd.Timedelta(minutes=1) > sim['max_gap_minutes'] + 1:
        return 'no price data'
    t = w.index
    gaps = (t[1:] - t[:-1]) / pd.Timedelta(minutes=1)
    for i in np.flatnonzero(gaps > sim['max_gap_minutes']):
        days = pd.date_range(t[i].normalize(), t[i + 1].normalize(), freq='D')
        if not (days.dayofweek == 5).any():          # a gap that holds no Saturday is missing data
            return 'no price data'
    return ''


def base_trade(cfg: dict, rec: dict, bars: pd.DataFrame, now: pd.Timestamp, rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    p = cfg['pairs'][rec['pair']]
    pip, point = p['pip'], p['point']
    s = session(rec['report_date'], rules)
    out = {k: '' for k in FIELDS}
    out.update(report_date=str(s['report_date'].date()), trade_date=str(s['trade_date'].date()), pair=rec['pair'],
               symbol=p['symbol'], direction=rec['direction'], entry_type=rec.get('entry_type', ''),
               close_4pm=rec.get('close'), session_end=str(s['end']))
    if str(rec.get('npr')).lower() in ('true', '1') or rec['direction'] == 'NPR':
        out['status'] = 'npr'
        return out
    if now < s['end']:
        out['status'] = 'pending'
        return out
    e0, st0, tg0, c4 = (float(rec[k]) for k in ('entry', 'stop', 'target', 'close'))
    buy = rec['direction'] == 'BUY'
    if (buy and not st0 < e0 < tg0) or (not buy and not tg0 < e0 < st0):
        out['status'] = 'levels out of order'
        return out
    before = bars[(bars.index < s['close_4pm']) & (bars.index >= s['close_4pm'] - pd.Timedelta(minutes=30))]
    if len(before):
        mc = float(before['close'].iloc[-1])
        out.update(mt5_close_4pm=mc, close_diff_points=round((mc - c4) / pip, 1))
        if abs(mc - c4) / pip > cfg['parse']['price_mismatch_points']:
            out['status'] = 'price mismatch'
            return out
    w = _bar_window(bars, s['open'], s['end'])
    prob = data_problem(w, s, cfg)
    if prob:
        out['status'] = prob
        return out
    o = float(w['open'].iloc[0])
    out.update(open_time=str(w.index[0]), open_price=o, open_gap_points=round((o - c4) / pip, 1))
    sgn = 1 if buy else -1
    d_stop, d_tgt = abs(e0 - st0), abs(tg0 - e0)
    if rec.get('entry_type') == 'rule0':
        i0, entry = 0, o
        if abs(o - c4) / pip >= rules['rule0']['proportionality_points']:
            stop, target, prop = o - sgn * d_stop, o + sgn * d_tgt, True
        else:
            stop, target, prop = st0, tg0, False
        fill_bar_target = True
    else:
        stop, target, prop = st0, tg0, False
        if (buy and o <= st0) or (not buy and o >= st0):
            out['status'] = 'open beyond stop'
            return out
        if (buy and o <= e0) or (not buy and o >= e0):
            i0, entry = 0, o
        else:
            hit = (w['low'] <= e0) if buy else (w['high'] >= e0)
            if not hit.any():
                out['status'] = 'entry not reached'
                return out
            i0, entry = int(np.argmax(hit.to_numpy())), e0
        fill_bar_target = i0 == 0 and entry == o
    out['proportional'] = prop
    hi, lo, op = w['high'].to_numpy(), w['low'].to_numpy(), w['open'].to_numpy()
    reason, j, px = 'time', len(w) - 1, float(w['close'].iloc[-1])
    for i in range(i0, len(w)):
        hs = lo[i] <= stop if buy else hi[i] >= stop
        ht = hi[i] >= target if buy else lo[i] <= target
        if i == i0 and not fill_bar_target:
            ht = False
        if hs:
            reason, j = 'stop', i
            px = min(stop, op[i]) if buy and i > i0 else (max(stop, op[i]) if not buy and i > i0 else stop)
            break
        if ht:
            reason, j, px = 'target', i, target
            break
    sp = w['spread'].to_numpy()
    cost = float(sp[i0] if buy else sp[j]) * point
    gross = sgn * (px - entry)
    r = abs(entry - stop)
    out.update(status='filled', exit_reason=reason, entry_time=str(w.index[i0]), entry=entry, stop=stop, target=target,
               exit_time=str(w.index[j]), exit=px, spread_cost=cost, r_dist=r, gross_pips=gross / pip,
               net_pips=(gross - cost) / pip, gross_r=gross / r, net_r=(gross - cost) / r)
    return out


def load_recs(cfg: dict) -> pd.DataFrame:
    from .ingest import RECS, read_csv, rpath
    d = pd.DataFrame(read_csv(rpath(cfg, RECS)))
    return d[d['kind'] == 'spot'] if len(d) else d


def simulate_all(cfg: dict | None = None, now_label=None, bars: dict | None = None,
                 recs: pd.DataFrame | None = None) -> pd.DataFrame:
    """`now_label` is "now" on the Berlin label clock (default: bars.label_now())."""
    from .bars import label_now, load
    cfg = cfg or load_config()
    now = label_now() if now_label is None else pd.Timestamp(now_label)
    recs = load_recs(cfg) if recs is None else recs
    rules = load_rules()
    cache = bars or {}
    rows = []
    for rec in recs.to_dict('records'):
        sym = cfg['pairs'][rec['pair']]['symbol']
        if sym not in cache:
            cache[sym] = load(cfg, sym)
        rows.append(base_trade(cfg, rec, cache[sym], now, rules))
    return pd.DataFrame(rows, columns=FIELDS)


def write(cfg: dict | None = None, now_label=None) -> str:
    cfg = cfg or load_config()
    df = simulate_all(cfg, now_label)
    p = os.path.join(results_dir(cfg), BASE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False, lineterminator='\n', float_format='%.6g')
    return p


def main(argv=None):
    argparse.ArgumentParser(description=f'rebuild the base trades. {COST_HEADER}').parse_args(argv)
    print(write())


if __name__ == '__main__':
    main()
