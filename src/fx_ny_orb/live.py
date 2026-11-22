"""The live passes (spec fx-ny-orb-hot/range-forward; design D7).

    python -m src.fx_ny_orb.live --pass forecast   after the bar labelled 14:00 Berlin has closed (16:05 Sofia)
    python -m src.fx_ny_orb.live --pass signal     every 15 minutes, 16:01-20:01 Sofia
    python -m src.fx_ny_orb.live --pass settle     after the bar labelled 21:00 Berlin has closed (23:05 Sofia)

The clock is the broker's server time from MT5's latest tick, i.e. the bar-label clock; each pass
refuses outside its window, so a wrong local time zone gives a refusal, never a bad row.

forecast  tops up the family's own H1 (EURUSD, GBPUSD, AUDUSD) and M15 (EURUSD, GBPUSD) caches from
          MT5 (read-only), builds fx_nyvol's 14 features for today from bars labelled 14:00 or earlier,
          applies the frozen HAR-X / HAR, the causal Q75 of the previous 60 eligible days' realised range,
          and logs p_harx, p_har, F, the hot flag and the opening range. Once per pair per day.
signal    applies the orb-rule to today's closed M15 bars (plus only the OPEN of the bar in progress,
          final at its first tick, as the entry price) and logs the first entry once per pair per day,
          whatever the hot flag or the chart gate.
settle    records the realised range, the squared errors and the day's simulated trade, and settles every
          logged signal into the ledger (or marks the day 'missed' / 'no signal'). Never rewritten.

Logs (append-only): results/fx_ny_orb/forward/{forecasts,signals,settlements,signal_ledger}.csv.
After each pass the chart export is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load_config, minutes
from . import frozen as FZ
from . import m15 as M
from . import rule as R
from .registry import guard, results_dir

LOGS = {
    'forecasts': ['date', 'pair', 'status', 'p_harx', 'p_har', 'F', 'close14', 'hot_threshold', 'hot', 'or_high',
                  'or_low', 'logged_at'],
    'signals': ['date', 'pair', 'direction', 'breakout_time', 'entry_time', 'entry', 'stop', 'target', 'hot', 'F',
                'late', 'logged_at'],
    'settlements': ['date', 'pair', 'status', 'y', 'p_harx', 'p_har', 'se_harx', 'se_har', 'F', 'hot', 'real_high',
                    'real_low', 'trade_reason', 'direction', 'entry_time', 'entry', 'stop', 'target', 'exit_time',
                    'exit', 'exit_reason', 'r_net', 'net_pips', 'settled_at'],
    'signal_ledger': ['date', 'pair', 'status', 'direction', 'entry_time', 'entry', 'stop', 'target', 'exit_time',
                      'exit', 'exit_reason', 'r_net', 'net_pips', 'hot', 'late', 'settled_at'],
}
TIME_COLS = ('date', 'breakout_time', 'entry_time', 'exit_time')


# -- logs --------------------------------------------------------------------------------------

def log_path(cfg: dict, name: str) -> str:
    return os.path.join(results_dir(cfg), 'forward', f'{name}.csv')


def read_log(cfg: dict, name: str) -> pd.DataFrame:
    p = log_path(cfg, name)
    if not os.path.exists(p):
        return pd.DataFrame(columns=LOGS[name])
    d = pd.read_csv(p, keep_default_na=False, na_values=[''])
    for c in TIME_COLS:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], utc=True)
    return d


def append_log(cfg: dict, name: str, rows: list) -> None:
    if not rows:
        return
    p = log_path(cfg, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=LOGS[name], lineterminator='\n')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k)))
                            else r.get(k)) for k in LOGS[name]})


def has_row(log: pd.DataFrame, date, pair) -> bool:
    return bool(len(log)) and bool(((log['date'] == date) & (log['pair'] == pair)).any())


# -- data ----------------------------------------------------------------------------------------

def nyvol() -> dict:
    from ..fx_nyvol import load_config as nyvol_config
    return nyvol_config()


def top_up(cfg: dict, feed, syms: dict) -> dict:
    """Append newly closed H1 (three pairs) and M15 (scored pairs) bars; return {pair: point}."""
    pts = {}
    for p in cfg['factor_pairs']:
        pts[p] = feed.point(syms[p])
        M.update_cache(p, 'H1', feed.closed_bars(syms[p], 'H1', int(cfg['forward']['h1_bars'])), cfg)
    for p in cfg['pairs']:
        M.update_cache(p, 'M15', feed.closed_bars(syms[p], 'M15', int(cfg['forward']['m15_bars'])), cfg)
    return pts


def h1_frames(cfg: dict) -> dict:
    return {p: M.load_cache(p, 'H1', cfg) for p in cfg['factor_pairs']}


def m15_day(cfg: dict, pair: str, point: float, date) -> pd.DataFrame:
    bars = M.live_m15(pair, point, cfg)
    if not len(bars):
        return bars
    idx = pd.DatetimeIndex(bars.index)
    sel = (idx.normalize() == date) & (idx >= date + pd.Timedelta(hours=14)) & \
        (idx <= date + pd.Timedelta(minutes=minutes(cfg['rule']['last_bar'])))
    return bars[sel]


def realised(frames: dict, points: dict, cfg: dict, ny: dict) -> tuple:
    """(calendar, eligibility reason, targets) on the live H1 caches via fx_nyvol (read-only)."""
    from ..fx_nyvol import target as TG
    cal = TG.calendar(frames, ny)
    reason = TG.eligibility(frames, points, ny, cal)
    tg = TG.targets(frames, points, ny, cal.index)
    return cal, reason, tg


def forecast_rows(cfg: dict, frames: dict, points: dict, today: pd.Timestamp, models: dict) -> list:
    """Today's forecast per scored pair, from bars labelled 14:00 or earlier only."""
    from ..fx_nyvol.features import build_features
    from ..fx_nyvol.target import grids
    from ..fx_session.session import verified_weeks
    ny = nyvol()
    cal, reason, tg = realised(frames, points, cfg, ny)
    if today not in cal.index:
        return [{'date': today, 'pair': p, 'status': 'no bars today'} for p in cfg['pairs']]
    monday = today - pd.Timedelta(days=today.dayofweek)
    clock_ok = all(bool(verified_weeks(frames[p].index).get(monday, False)) for p in cfg['factor_pairs'])
    one = pd.DatetimeIndex([today])
    pre_ok, corrupt = True, False
    for p in cfg['factor_pairs']:
        g = grids(frames[p], one)
        pre_ok &= not np.isnan(g['close'][0, 0:15]).any()
        pips = g['spread'][0, 0:15] * points[p] / float(cfg['pip'][p])
        corrupt |= bool(np.nan_to_num(pips, nan=0.0).max() > float(cfg['data']['max_spread_pips']))
    past = ((reason == '') & (cal.index < today)).to_numpy()
    feats = build_features(frames, points, ny, cal.index, {p: tg[p].loc[past, 'y'] for p in cfg['pairs']})
    fw = cfg['forward']
    rows = []
    for p in cfg['pairs']:
        base = {'date': today, 'pair': p}
        if not clock_ok:
            rows.append({**base, 'status': 'unverified clock week'})
            continue
        if not pre_ok:
            rows.append({**base, 'status': 'missing pre-session bar'})
            continue
        if corrupt:
            rows.append({**base, 'status': 'corrupt spread'})
            continue
        x = feats[p].loc[today, ny['features']['columns']].to_numpy(float)
        ys = tg[p].loc[past, 'y'].dropna()
        if np.isnan(x).any() or len(ys) < int(fw['hot_window_days']):
            rows.append({**base, 'status': 'warm-up'})
            continue
        mdl = models[p]
        p_harx = FZ.predict(mdl['coef_harx'], x)
        p_har = FZ.predict(mdl['coef_har'], feats[p].loc[today, ny['features']['har_columns']].to_numpy(float))
        thr = float(ys.iloc[-int(fw['hot_window_days']):].quantile(float(fw['hot_quantile'])))
        day = m15_day(cfg, p, points[p], today)
        ent = R.entries(day, 1.0, cfg) if len(day) else {'reason': 'no range'}
        c14 = day.loc[day.index == today + pd.Timedelta(hours=14, minutes=45), 'mc']
        if 'or_high' not in ent or not len(c14):
            rows.append({**base, 'status': 'no M15 range', 'p_harx': p_harx, 'p_har': p_har, 'hot_threshold': thr,
                         'hot': p_harx > thr})
            continue
        close14 = float(c14.iloc[0])
        rows.append({**base, 'status': 'ok', 'p_harx': p_harx, 'p_har': p_har, 'F': float(np.exp(p_harx) * close14),
                     'close14': close14, 'hot_threshold': thr, 'hot': bool(p_harx > thr), 'or_high': ent['or_high'],
                     'or_low': ent['or_low']})
    return rows


# -- passes --------------------------------------------------------------------------------------

def _clock(cfg: dict, feed) -> tuple:
    """Real time on the label clock. Never the last tick: a stale weekend tick would make Friday 'today'
    and log a forecast for a day that is already over (a backfill)."""
    syms = {p: feed.symbol(p) for p in cfg['factor_pairs']}
    now = feed.label_now()
    return syms, now, now.normalize()


def _rebuild_export(cfg: dict, now) -> str:
    try:
        from .export import rebuild
        rebuild(cfg, now)
        return 'export ok'
    except Exception as e:                       # the logs stand even when the chart file cannot be written
        return f'export failed: {e}'


def run_forecast(cfg: dict | None = None, feed=None, models: dict | None = None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    feed = feed or M.Mt5Live()
    syms, now, today = _clock(cfg, feed)
    if today.dayofweek >= 5:
        return {'status': 'weekend'}
    if now < today + pd.Timedelta(minutes=minutes(cfg['forward']['forecast_after'])):
        return {'status': 'too early: the bar labelled 14:00 has not closed'}
    log = read_log(cfg, 'forecasts')
    todo = [p for p in cfg['pairs'] if not has_row(log, today, p)]
    if not todo:
        return {'status': 'already logged'}
    points = top_up(cfg, feed, syms)
    models = models or {p: FZ.load(cfg, p) for p in cfg['pairs']}
    rows = [r for r in forecast_rows(cfg, h1_frames(cfg), points, today, models) if r['pair'] in todo]
    for r in rows:
        r['logged_at'] = now
    append_log(cfg, 'forecasts', rows)
    return {'status': 'logged', 'rows': rows, 'export': _rebuild_export(cfg, now)}


def run_signal(cfg: dict | None = None, feed=None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    feed = feed or M.Mt5Live()
    syms, now, today = _clock(cfg, feed)
    start = today + pd.Timedelta(minutes=minutes(cfg['rule']['breakout_from']) + 15)
    if today.dayofweek >= 5 or now < start:
        return {'status': 'too early'}
    if now > today + pd.Timedelta(minutes=minutes(cfg['forward']['signal_until'])):
        return {'status': 'after the entry cut-off'}
    fc = read_log(cfg, 'forecasts')
    fc = fc[(fc['date'] == today) & (fc['status'] == 'ok')] if len(fc) else fc
    if not len(fc):
        return {'status': 'no forecast today'}
    sig = read_log(cfg, 'signals')
    logged = []
    for _, f in fc.iterrows():
        p = f['pair']
        if has_row(sig, today, p):
            continue
        point = feed.point(syms[p])
        closed = feed.closed_bars(syms[p], 'M15', 60)
        closed = closed[(pd.DatetimeIndex(closed.index).normalize() == today) &
                        (closed.index + pd.Timedelta(minutes=15) <= now)]
        bars = closed[['open', 'high', 'low', 'close']].copy()
        bars['spread_price'] = closed['spread'] * point
        prog = feed.bar_in_progress_open(syms[p])
        if prog is not None and prog[0].normalize() == today and prog[0] not in bars.index and prog[0] <= now:
            bars.loc[prog[0]] = [prog[1], np.nan, np.nan, np.nan, prog[2] * point]
        day = M.with_mid(bars.sort_index(), float(cfg['pip'][p]), cfg)
        day = day[day.index >= today + pd.Timedelta(hours=14)]
        ent = R.entries(day, float(f['F']), cfg)
        if ent['reason']:
            continue
        row = {'date': today, 'pair': p, 'direction': ent['direction'], 'breakout_time': ent['breakout_time'],
               'entry_time': ent['entry_time'], 'entry': ent['entry'], 'stop': ent['stop'], 'target': ent['target'],
               'hot': bool(f['hot']), 'F': float(f['F']), 'logged_at': now,
               'late': bool(now > ent['entry_time'] + pd.Timedelta(minutes=2))}
        logged.append(row)
    append_log(cfg, 'signals', logged)
    return {'status': 'ok', 'signals': logged, 'export': _rebuild_export(cfg, now)}


def run_settle(cfg: dict | None = None, feed=None) -> dict:
    cfg = cfg or load_config()
    guard(cfg)
    feed = feed or M.Mt5Live()
    syms, now, today = _clock(cfg, feed)
    fc = read_log(cfg, 'forecasts')
    st = read_log(cfg, 'settlements')
    due = fc[fc['date'] + pd.Timedelta(minutes=minutes(cfg['forward']['settle_after'])) <= now] if len(fc) else fc
    due = due[[not has_row(st, d, p) for d, p in zip(due['date'], due['pair'])]] if len(due) else due
    if not len(due):
        return {'status': 'nothing to settle'}
    points = top_up(cfg, feed, syms)
    frames = h1_frames(cfg)
    ny = nyvol()
    cal, reason, tg = realised(frames, points, cfg, ny)
    sig = read_log(cfg, 'signals')
    rows, ledger = [], []
    for _, f in due.iterrows():
        d, p = f['date'], f['pair']
        base = {'date': d, 'pair': p, 'settled_at': now}
        if f['status'] != 'ok':
            rows.append({**base, 'status': f'forecast {f["status"]}'})
            continue
        r = reason.get(d, 'no bars')
        y = float(tg[p]['y'].get(d, np.nan))
        h1 = frames[p]
        win = h1[(h1.index >= d + pd.Timedelta(hours=15)) & (h1.index <= d + pd.Timedelta(hours=21))]
        half = win['spread'] * points[p] / 2
        row = {**base, 'status': 'ok' if r == '' and np.isfinite(y) else (r or 'incomplete session'),
               'y': y, 'p_harx': f['p_harx'], 'p_har': f['p_har'],
               'se_harx': (y - f['p_harx']) ** 2 if np.isfinite(y) else np.nan,
               'se_har': (y - f['p_har']) ** 2 if np.isfinite(y) else np.nan, 'F': f['F'], 'hot': bool(f['hot']),
               'real_high': float((win['high'] + half).max()) if len(win) else np.nan,
               'real_low': float((win['low'] + half).min()) if len(win) else np.nan}
        day = m15_day(cfg, p, points[p], d)
        pip = float(cfg['pip'][p])
        t = R.simulate_day(day, float(f['F']), bool(f['hot']), pip, cfg) if len(day) else {'reason': 'no M15 data'}
        row.update(trade_reason=t['reason'], **{k: t.get(k) for k in ('direction', 'entry_time', 'entry', 'stop',
                                                                        'target', 'exit_time', 'exit', 'exit_reason',
                                                                        'r_net', 'net_pips')})
        rows.append(row)
        s = sig[(sig['date'] == d) & (sig['pair'] == p)] if len(sig) else sig
        lg = {'date': d, 'pair': p, 'hot': bool(f['hot']), 'settled_at': now}
        if len(s):
            s = s.iloc[0]
            pos = np.flatnonzero(day.index == s['entry_time'])
            if len(pos):
                ent = {'direction': int(s['direction']), 'entry_pos': int(pos[0]), 'entry': float(s['entry']),
                       'stop': float(s['stop']), 'target': float(s['target'])}
                x = R.scan_exit(day, ent, float(f['F']), pip, cfg)
                lg.update(status='settled', **{k: s[k] for k in ('direction', 'entry_time', 'entry', 'stop', 'target',
                                                                 'late')},
                          **{k: x[k] for k in ('exit_time', 'exit', 'exit_reason', 'r_net', 'net_pips')})
            else:
                lg.update(status='entry bar missing', direction=s['direction'], entry_time=s['entry_time'])
        elif t['reason'] == '':
            lg.update(status='missed', direction=t['direction'], entry_time=t['entry_time'])
        else:
            lg.update(status=f"no signal: {t['reason']}")
        ledger.append(lg)
    append_log(cfg, 'settlements', rows)
    append_log(cfg, 'signal_ledger', ledger)
    return {'status': 'settled', 'rows': len(rows), 'export': _rebuild_export(cfg, now)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='one live fx-ny-orb-hot pass')
    ap.add_argument('--pass', dest='which', choices=['forecast', 'signal', 'settle'], required=True)
    a = ap.parse_args(argv)
    run = {'forecast': run_forecast, 'signal': run_signal, 'settle': run_settle}[a.which]
    out = run()
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), a.which,
          {k: v for k, v in out.items() if k != 'rows'})


if __name__ == '__main__':
    main()
