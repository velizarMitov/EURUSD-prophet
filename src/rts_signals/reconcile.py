"""Reconciliation of RTS's own spot track record with the owner's MT5 prices (spec
rts-signals/trade-simulation "Reconciliation"; design D7). DESCRIPTIVE ONLY.

    python -m src.rts_signals.reconcile     rebuild results/rts_signals/replay_trades.csv and reconciliation.csv

Each closed RTS spot trade (latest version of each line, see track_record.csv) is checked directly on the
M1 bars, with a `tolerance_points` (3) allowance because RTS's eSignal composite differs from the broker's
prices by a point or two:
  * window: from 18:00 New York on the weekday before its entry date to 16:00 New York (Friday 12:00) on
    its exit date;
  * `not reproducible`: our bars never trade at the entry (+-3 points) on the entry date's session;
  * `confirmed`: after the entry touch our bars reach RTS's exit price (+-3) by the end of the window, and
    before the stop when the rule fixes it (Rule 2: 40, Rule 4: 44 points);
  * `different exit`: otherwise.
`replay_match` says whether the full-rules replay opened the same trade (pair, side, date, entry +-3);
it exists only for sessions whose report was received.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import load_config, results_dir
from .rules import load_rules, ny_to_label

REPLAY = 'replay_trades.csv'
RECON = 'reconciliation.csv'
LABELS = ('confirmed', 'different exit', 'not reproducible')


def prev_weekday(day) -> pd.Timestamp:
    d = pd.Timestamp(day) - pd.Timedelta(days=1)
    while d.dayofweek >= 5:
        d -= pd.Timedelta(days=1)
    return d


def window(entry_date, exit_date, rules) -> tuple:
    ck = rules['clock']
    start = ny_to_label(prev_weekday(entry_date), ck['open_ny'])
    ex = pd.Timestamp(exit_date)
    end = ny_to_label(ex, ck['friday_spot_close_ny'] if ex.dayofweek == 4 else ck['close_ny'])
    first_end = pd.Timestamp(entry_date)
    first_end = ny_to_label(first_end, ck['friday_spot_close_ny'] if first_end.dayofweek == 4 else ck['close_ny'])
    return start, first_end, end


def latest_track(cfg: dict) -> pd.DataFrame:
    from .ingest import TRACK, read_csv, rpath
    t = pd.DataFrame(read_csv(rpath(cfg, TRACK)))
    if not len(t):
        return t
    t = t[(t['kind'] == 'spot') & (t['status'] == 'closed')].copy()
    t = t.sort_values('first_report_date')
    return t.drop_duplicates(['pair', 'entry_date', 'side', 'entry', 'rule'], keep='last')


def check_trade(cfg: dict, tr: dict, bars: pd.DataFrame, rules: dict) -> dict:
    p = cfg['pairs'][tr['pair']]
    pip, tol = p['pip'], cfg['reconcile']['tolerance_points'] * p['pip']
    start, first_end, end = window(tr['entry_date'], tr['exit_date'], rules)
    w = bars[(bars.index >= start) & (bars.index < end)]
    entry, exit_ = float(tr['entry']), float(tr['exit'])
    buy = tr['side'] == 'buy'
    first = w[w.index < first_end]
    if not len(first):
        return {'label': 'not reproducible', 'note': 'no bars'}
    touch = (first['low'] <= entry + tol) & (first['high'] >= entry - tol)
    if not touch.any():
        return {'label': 'not reproducible', 'note': 'entry never traded'}
    i0 = int(np.argmax(touch.to_numpy()))
    after = w.iloc[i0:]
    sp = float(first['spread'].iloc[i0]) * p['point'] / pip if 'spread' in first else float('nan')
    rule = int(tr['rule']) if str(tr.get('rule', '')).strip() not in ('', 'nan') else None
    stop_pts = {2: rules['rule2']['stop_points'], 4: rules['rule4']['stop_points']}.get(rule)
    hit_exit = (after['low'] <= exit_ + tol) & (after['high'] >= exit_ - tol)
    if not hit_exit.any():
        return {'label': 'different exit', 'note': 'exit price never traded after the entry', 'spread_pips': sp}
    j = int(np.argmax(hit_exit.to_numpy()))
    if stop_pts is not None and float(tr['points']) > 0:
        stop = entry - stop_pts * pip if buy else entry + stop_pts * pip
        hs = (after['low'] <= stop) if buy else (after['high'] >= stop)
        if hs.any() and int(np.argmax(hs.to_numpy())) < j:
            return {'label': 'different exit', 'note': 'our prices reach the stop first', 'spread_pips': sp}
    return {'label': 'confirmed', 'note': '', 'spread_pips': sp}


def reconcile(cfg: dict | None = None, bars: dict | None = None, replay: pd.DataFrame | None = None) -> pd.DataFrame:
    from .bars import load
    cfg = cfg or load_config()
    rules = load_rules()
    track = latest_track(cfg)
    bars = bars or {}
    rows = []
    for tr in track.to_dict('records'):
        sym = cfg['pairs'][tr['pair']]['symbol']
        if sym not in bars:
            bars[sym] = load(cfg, sym)
        res = check_trade(cfg, tr, bars[sym], rules)
        match = ''
        if replay is not None and len(replay):
            tol = cfg['reconcile']['tolerance_points'] * cfg['pairs'][tr['pair']]['pip']
            cand = replay[(replay['pair'] == tr['pair']) & (replay['trade_date'] == tr['entry_date'])]
            if len(cand):
                match = 'yes' if ((cand['side'] == tr['side']) &
                                  ((cand['entry'].astype(float) - float(tr['entry'])).abs() <= tol)).any() else 'no'
        rows.append({'entry_date': tr['entry_date'], 'pair': tr['pair'], 'symbol': sym, 'side': tr['side'],
                     'entry': tr['entry'], 'rule': tr['rule'], 'exit_date': tr['exit_date'], 'exit': tr['exit'],
                     'rts_points': tr['points'], 'spread_pips': res.get('spread_pips', ''), 'label': res['label'], 'note': res['note'], 'replay_match': match,
                     'track_flags': tr.get('flags', '')})
    return pd.DataFrame(rows)


def write(cfg: dict | None = None) -> tuple:
    from .bars import load
    from .replay import replay_all
    from .simulate import load_recs
    cfg = cfg or load_config()
    bars = {p['symbol']: load(cfg, p['symbol']) for p in cfg['pairs'].values()}
    recs = load_recs(cfg)
    rp = replay_all(cfg, recs, bars) if len(recs) else pd.DataFrame()
    rc = reconcile(cfg, bars, rp)
    out = results_dir(cfg)
    os.makedirs(out, exist_ok=True)
    rp.to_csv(os.path.join(out, REPLAY), index=False, lineterminator='\n', float_format='%.6g')
    rc.to_csv(os.path.join(out, RECON), index=False, lineterminator='\n')
    return os.path.join(out, REPLAY), os.path.join(out, RECON)


def main(argv=None):
    argparse.ArgumentParser(description='replay the full rules and reconcile RTS\'s track record (descriptive)').parse_args(argv)
    a, b = write()
    rc = pd.read_csv(b)
    print(a, b)
    print(rc['label'].value_counts().to_string() if len(rc) else 'no track record')


if __name__ == '__main__':
    main()
