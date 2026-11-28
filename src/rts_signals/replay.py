"""The full-rules replay: RTS's contingency rules applied minute by minute to each spot session
(spec rts-signals/trade-simulation "Full-rules replay"; rule-book "Contingency rules"; design D7).

DESCRIPTIVE ONLY. It is never scored; the base trade (simulate.py) is the scored object.

One position at a time per pair and session, gross on bid prices, every number from rules.json:
  * start: Rule 5 when the open gaps `gap_points` against the direction (the advice is ignored; a further
    `extra_points` reverses, with Rule 4's stop and target); a same-direction gap of
    `gap_same_direction_points` enters at the open as Rule 2; otherwise Rule 0 (with proportionality) or a
    pending Rule 1 guerilla. NPR in the entry column allows only Rules 2 and 4 off the close; NPR as the
    direction allows nothing.
  * Rule 2: flat, `trigger_points` beyond the last profitable exit (or beyond the close while a guerilla is
    unfilled) in the trade direction -> same direction, stop/target 40/60. Never on a Friday in spot.
  * Rule 3: after a losing stop, the price returns to that trade's entry -> re-enter with the same stop and
    target (38/44 when a distance is negative or <= 15 points). At most `max_losses_spot` losses on the
    trade; never under NPR.
  * Rule 4: `trigger_points` against the direction from the last exit (or off the close while nothing was
    entered) -> reverse, stop/target 44/48.
  * The first rule-based entry kills the close and the recommended entry for the rest of the session.
  * A bar touching both stop and target is the stop. In an entry bar only the stop is checked. The session
    ends at 16:00 New York (Friday 12:00): an open position exits at the last close. RTS itself may carry a
    position into the next session; such trades reconcile as `different exit`.
Simultaneous triggers in one minute are taken in the order guerilla, Rule 3, Rule 2, Rule 4.
"""

from __future__ import annotations

import pandas as pd

from .rules import load_rules, session
from .simulate import data_problem


def replay_session(cfg: dict, rec: dict, bars: pd.DataFrame, rules: dict | None = None) -> list:
    rules = rules or load_rules()
    if rec['direction'] == 'NPR':
        return []
    s = session(rec['report_date'], rules)
    w = bars[(bars.index >= s['open']) & (bars.index < s['end'])]
    if data_problem(w, s, cfg):
        return []
    p = cfg['pairs'][rec['pair']]
    pip = p['pip']
    r2, r3, r4, r5 = rules['rule2'], rules['rule3'], rules['rule4'], rules['rule5']
    friday = s['friday_close']
    d = 1 if rec['direction'] == 'BUY' else -1
    close = float(rec['close'])
    npr_entry = str(rec.get('npr')).lower() in ('true', '1')
    hi, lo, op, cl = (w[c].to_numpy() for c in ('high', 'low', 'open', 'close'))
    t = w.index
    trades, pos = [], None
    st = {'pending': None, 'anchor2': None, 'anchor4': None, 'last_loss': None, 'losses': 0, 'rule_based': False,
          'r5_level': None}

    def opened(side, entry, stop, target, rule, i, orig=None):
        return {'side': side, 'entry': entry, 'stop': stop, 'target': target, 'rule': rule, 'i': i,
                'orig': orig or (entry, stop, target)}

    o = op[0]
    gap = (o - close) / pip * d
    if gap <= -r5['gap_points']:
        st['r5_level'] = o - d * r5['extra_points'] * pip
    elif gap >= r2['gap_same_direction_points'] and not (friday and not r2['friday_spot']) and not npr_entry:
        pos = opened(d, o, o - d * r2['stop_points'] * pip, o + d * r2['target_points'] * pip, 2, 0)
        st['rule_based'] = True
    elif npr_entry:
        st['anchor2'] = st['anchor4'] = close
    elif rec.get('entry_type') == 'rule0':
        e0, s0, g0 = (float(rec[k]) for k in ('entry', 'stop', 'target'))
        if abs(o - close) / pip >= rules['rule0']['proportionality_points']:
            pos = opened(d, o, o - (e0 - s0), o + (g0 - e0), 0, 0)
        else:
            pos = opened(d, o, s0, g0, 0, 0)
        pos['first_bar_full'] = True
    else:
        st['pending'] = float(rec['entry'])
        st['anchor2'] = st['anchor4'] = close

    def touched_up(i, lvl):
        return hi[i] >= lvl

    def touched_dn(i, lvl):
        return lo[i] <= lvl

    for i in range(len(w)):
        if pos is not None:
            sd = pos['side']
            hs = touched_dn(i, pos['stop']) if sd > 0 else touched_up(i, pos['stop'])
            ht = touched_up(i, pos['target']) if sd > 0 else touched_dn(i, pos['target'])
            if i == pos['i'] and not pos.get('first_bar_full'):
                ht = False
            if hs or ht:
                px = pos['stop'] if hs else pos['target']
                pts = sd * (px - pos['entry']) / pip
                trades.append({'side': 'buy' if sd > 0 else 'sell', 'rule': pos['rule'], 'entry': pos['entry'],
                               'entry_time': str(t[pos['i']]), 'exit': px, 'exit_time': str(t[i]),
                               'exit_reason': 'stop' if hs else 'target', 'points': round(pts, 1)})
                d = sd
                st['anchor4'] = px
                if pts > 0:
                    st['anchor2'], st['last_loss'] = px, None
                else:
                    st['anchor2'] = None
                    st['losses'] += 1
                    if pos['rule'] in (0, 1, 3) and not npr_entry and st['losses'] < r3['max_losses_spot']:
                        st['last_loss'] = pos
                    else:
                        st['last_loss'] = None
                pos = None
            continue
        cand = None
        if st['r5_level'] is not None and (touched_dn(i, st['r5_level']) if d > 0 else touched_up(i, st['r5_level'])):
            lv = st['r5_level']
            cand = (-d, lv, lv + d * r4['stop_points'] * pip, lv - d * r4['target_points'] * pip, 5)
            st['r5_level'] = None
        elif st['pending'] is not None and not st['rule_based'] and \
                (touched_dn(i, st['pending']) if d > 0 else touched_up(i, st['pending'])):
            e0, s0, g0 = (float(rec[k]) for k in ('entry', 'stop', 'target'))
            cand = (d, e0, s0, g0, 1)
            st['pending'] = None
        elif st['last_loss'] is not None:
            L = st['last_loss']
            e, sp, tg = L['orig']
            if touched_up(i, e) if L['side'] > 0 else touched_dn(i, e):
                ds, dt = L['side'] * (e - sp) / pip, L['side'] * (tg - e) / pip
                if ds <= r3['min_distance_points'] or dt <= r3['min_distance_points']:
                    sp = e - L['side'] * r3['fallback_stop_points'] * pip
                    tg = e + L['side'] * r3['fallback_target_points'] * pip
                cand = (L['side'], e, sp, tg, 3)
                st['last_loss'] = None
        if cand is None and st['anchor2'] is not None and not (friday and not r2['friday_spot']):
            lv = st['anchor2'] + d * r2['trigger_points'] * pip
            if touched_up(i, lv) if d > 0 else touched_dn(i, lv):
                cand = (d, lv, lv - d * r2['stop_points'] * pip, lv + d * r2['target_points'] * pip, 2)
        if cand is None and st['anchor4'] is not None:
            lv = st['anchor4'] - d * r4['trigger_points'] * pip
            if touched_dn(i, lv) if d > 0 else touched_up(i, lv):
                cand = (-d, lv, lv + d * r4['stop_points'] * pip, lv - d * r4['target_points'] * pip, 4)
        if cand is not None:
            side, e, sp, tg, rule = cand
            pos = opened(side, e, sp, tg, rule, i)
            if rule != 1:
                st['rule_based'] = True
                st['pending'] = None
            st['anchor2'] = st['anchor4'] = None
            d = side
    if pos is not None:
        px = cl[-1]
        trades.append({'side': 'buy' if pos['side'] > 0 else 'sell', 'rule': pos['rule'], 'entry': pos['entry'],
                       'entry_time': str(t[pos['i']]), 'exit': px, 'exit_time': str(t[-1]), 'exit_reason': 'time',
                       'points': round(pos['side'] * (px - pos['entry']) / pip, 1)})
    for tr in trades:
        tr.update(report_date=str(s['report_date'].date()), trade_date=str(s['trade_date'].date()), pair=rec['pair'],
                  symbol=p['symbol'])
    return trades


def replay_all(cfg: dict, recs: pd.DataFrame, bars: dict) -> pd.DataFrame:
    rules = load_rules()
    rows = []
    for rec in recs.sort_values('report_date').to_dict('records'):
        sym = cfg['pairs'][rec['pair']]['symbol']
        rows += replay_session(cfg, rec, bars[sym], rules)
    cols = ['report_date', 'trade_date', 'pair', 'symbol', 'side', 'rule', 'entry', 'entry_time', 'exit', 'exit_time',
            'exit_reason', 'points']
    return pd.DataFrame(rows, columns=cols)
