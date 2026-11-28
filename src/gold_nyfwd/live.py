"""The two daily passes (spec gold-nyrange-forward/daily-passes; design D4, D5, D8).

    python -m src.gold_nyfwd.live --pass forecast   after the bar labelled 14:00 Berlin has closed (16:08 Sofia)
    python -m src.gold_nyfwd.live --pass settle     after the bar labelled 21:00 Berlin has closed (23:08 Sofia)

Order in every pass: the config guard, the pre-registration gate, the label-clock window, then
once-per-day. The clock is real time on the Berlin label clock (feed.label_now), never the last tick.

forecast  tops up the family's caches (GOLD, EURUSD and every listed USDInd contract) read-only, records new
          contracts and their lag check, truncates gold and the dollar frame to bars labelled <= 14:00 on
          the day, runs gold_bbnr's d1_ny and ny_features.dataset unchanged and applies the pinned
          coefficients. It logs p_harx, p_har, F = exp(p_harx) x close14, close14 and the contract, or one
          reason: `unverified clock week`, `missing pre-session bar`, `no dollar data` or `warm-up`.
          Today's own eligibility is ignored (its session has not happened); its features use only earlier
          days' targets.
settle    settles every logged ok forecast whose bar labelled 21:00 is in the cache, on the cache truncated
          to that bar, with gold_bbnr's eligibility and target: y, both squared errors and the realised
          high and low of the bars labelled 15:00-21:00, or `invalid day` / `incomplete session`.
          A day with no forecast row is never settled.

Logs (append-only, never rewritten or backfilled): results/gold_nyfwd/forward/{forecasts,settlements}.csv.
After each pass that logs, the chart file is rebuilt; a failed export never changes a log.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load_config, minutes, results_dir
from . import cache as C
from . import contracts as K
from .registry import guard

LOGS = {
    'forecasts': ['date', 'status', 'contract', 'close14', 'p_harx', 'p_har', 'F', 'logged_at'],
    'settlements': ['date', 'status', 'y', 'p_harx', 'p_har', 'se_harx', 'se_har', 'realised_high', 'realised_low',
                    'settled_at'],
}
REASONS = ('unverified clock week', 'missing pre-session bar', 'no dollar data', 'warm-up')


# -- logs --------------------------------------------------------------------------------------

def log_path(cfg: dict, name: str) -> str:
    return os.path.join(results_dir(cfg), 'forward', f'{name}.csv')


def read_log(cfg: dict, name: str) -> pd.DataFrame:
    p = log_path(cfg, name)
    if not os.path.exists(p):
        return pd.DataFrame(columns=LOGS[name])
    d = pd.read_csv(p, keep_default_na=False, na_values=[''])
    d['date'] = pd.to_datetime(d['date'], utc=True)
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
            out = {}
            for k in LOGS[name]:
                v = r.get(k)
                if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                    out[k] = ''
                elif isinstance(v, pd.Timestamp):
                    out[k] = v.isoformat()
                elif isinstance(v, (float, np.floating)):
                    out[k] = repr(float(v))
                else:
                    out[k] = v
            w.writerow(out)


def logged_dates(log: pd.DataFrame) -> set:
    return set(log['date']) if len(log) else set()


# -- inputs ------------------------------------------------------------------------------------

def gb_config() -> dict:
    from ..gold_bbnr import load_config as gb
    return gb()


def top_up(cfg: dict, feed, now, contracts_too: bool = True) -> dict:
    """Append newly closed bars to the caches; record new contracts and run pending lag checks."""
    m = cfg['mt5']
    listed = feed.symbols()
    missing = [s for s in (m['gold_symbol'], m['lag_pair_symbol']) if s not in listed]
    if missing:
        raise LookupError(f'symbol(s) {missing} not listed in the terminal: the pass refuses')
    n = int(m['h1_bars'])
    syms = [m['gold_symbol']]
    if contracts_too:
        syms += [m['lag_pair_symbol']] + sorted(s for s in listed if s.startswith(m['contract_prefix']))
    out = {s: C.update(cfg, s, feed.closed_bars(s, n)) for s in syms}
    if contracts_too:
        rec = K.remember(cfg, listed, now)
        K.check_pending(cfg, rec, lambda s: C.load(cfg, s), C.load(cfg, m['lag_pair_symbol']), now)
    return out


def clock_ok(gold: pd.DataFrame, today, cfg: dict) -> bool:
    """The gold family's weekly-open check on the whole cache, and this week's own open on its hour."""
    from ..forecast_eval.m15_data import weekly_opens
    from ..gold.clock import GoldClockMismatch, expected_open_hour, verify
    try:
        verify(gold.index, float(cfg['clock']['gold_threshold']))
    except GoldClockMismatch:
        return False
    sunday = today - pd.Timedelta(days=int(today.dayofweek) + 1)
    opens = [t for t in weekly_opens(gold.index) if sunday <= t <= today + pd.Timedelta(hours=14)]
    return bool(opens) and all(int(t.hour) == expected_open_hour(t) for t in opens)


def frames_until(cfg: dict, gold: pd.DataFrame, cut) -> pd.DataFrame:
    return gold[gold.index <= cut]


def forecast_row(cfg: dict, today: pd.Timestamp, model: dict, gb: dict | None = None,
                 load_bars=None) -> dict:
    """Today's row from the caches, using bars labelled <= 14:00 on `today` only."""
    from ..gold_bbnr import ny_features as NF
    from ..gold_bbnr.bars import d1_ny
    from ..gold_bbnr.ny_target import calendar_dates, grid
    gb = gb or gb_config()
    load_bars = load_bars or (lambda s: C.load(cfg, s))
    s = gb['session']
    cut = today + pd.Timedelta(hours=int(cfg['forward']['pre_session_last_hour']))
    gold = frames_until(cfg, load_bars(cfg['mt5']['gold_symbol']), cut)
    base = {'date': today}
    if not clock_ok(gold, today, cfg):
        return {**base, 'status': 'unverified clock week'}
    dates = calendar_dates(gold)
    if today not in dates:
        return {**base, 'status': 'missing pre-session bar'}
    g = grid(gold, pd.DatetimeIndex([today]))
    if np.isnan(g['close'][0, 0:s['data_hour'] + 1]).any():
        return {**base, 'status': 'missing pre-session bar'}
    rec = K.read(cfg)
    mapping = K.day_map(dates, rec, int(cfg['mt5']['roll_margin_trading_days']))
    contract = mapping.get(today)
    dxy = K.dollar_frame(dates, mapping, load_bars)
    dxy = dxy[dxy.index <= cut]
    pre = dxy[(dxy.index >= today) & (dxy.index <= today + pd.Timedelta(hours=s['pre_hours'][1]))]
    if contract is None or len(pre) < int(gb['dxy']['min_pre_bars']):
        return {**base, 'status': 'no dollar data', 'contract': contract}
    ds = NF.dataset(gold, d1_ny(gold, gb), dxy, gb, read_target=True)
    x = ds.loc[today, model['columns']].to_numpy(float)
    if np.isnan(x).any():
        return {**base, 'status': 'warm-up', 'contract': contract}
    xh = ds.loc[today, model['har_columns']].to_numpy(float)
    p_harx = float(model['coef_harx'][0] + x @ np.asarray(model['coef_harx'][1:], dtype=float))
    p_har = float(model['coef_har'][0] + xh @ np.asarray(model['coef_har'][1:], dtype=float))
    close14 = float(g['close'][0, s['data_hour']])
    return {**base, 'status': 'ok', 'contract': contract, 'close14': close14, 'p_harx': p_harx, 'p_har': p_har,
            'F': float(np.exp(p_harx) * close14)}


def settle_row(cfg: dict, f: pd.Series, gold: pd.DataFrame, gb: dict | None = None) -> dict | None:
    """The settlement of one ok forecast, or None while its bar labelled 21:00 is not yet cached."""
    from ..gold_bbnr.bars import d1_ny
    from ..gold_bbnr.ny_target import calendar_dates, eligibility, grid, target
    gb = gb or gb_config()
    s = gb['session']
    d = f['date']
    cut = d + pd.Timedelta(hours=int(cfg['forward']['settle_last_hour']))
    if not len(gold) or gold.index.max() < cut:
        return None
    g1 = frames_until(cfg, gold, cut)
    dates = calendar_dates(g1)
    one = pd.DatetimeIndex([d])
    g = grid(g1, one)
    a, b = s['target_hours']
    win_h, win_l = g['high'][0, a:b + 1], g['low'][0, a:b + 1]
    hi = float(np.nanmax(win_h)) if np.isfinite(win_h).any() else np.nan
    lo = float(np.nanmin(win_l)) if np.isfinite(win_l).any() else np.nan
    base = {'date': d, 'p_harx': f['p_harx'], 'p_har': f['p_har'], 'realised_high': hi, 'realised_low': lo}
    if d not in dates:
        return {**base, 'status': 'incomplete session'}
    reason = eligibility(grid(g1, dates), dates, d1_ny(g1, gb), gb).loc[d]
    y = float(target(g, one, gb).iloc[0])
    if reason != '' or not np.isfinite(y):
        return {**base, 'status': reason or 'incomplete session'}
    return {**base, 'status': 'ok', 'y': y, 'se_harx': (y - float(f['p_harx'])) ** 2,
            'se_har': (y - float(f['p_har'])) ** 2}


# -- passes --------------------------------------------------------------------------------------

def _rebuild_export(cfg: dict, now) -> str:
    try:
        from .export import rebuild
        rebuild(cfg, now)
        return 'export ok'
    except Exception as e:                       # the logs stand even when the chart file cannot be written
        return f'export failed: {e}'


def _start(cfg, feed, gate):
    from .feed import Mt5Feed
    from .prereg import gate as prereg_gate
    cfg = cfg or load_config()
    guard(cfg)
    (gate or prereg_gate)(cfg)
    feed = feed or Mt5Feed(cfg)
    now = feed.label_now()
    return cfg, feed, now, now.normalize()


def run_forecast(cfg: dict | None = None, feed=None, gate=None, ols_path: str | None = None) -> dict:
    from .prereg import coefficients
    cfg, feed, now, today = _start(cfg, feed, gate)
    if today.dayofweek >= 5:
        return {'status': 'weekend'}
    if now < today + pd.Timedelta(minutes=minutes(cfg['forward']['forecast_after'])):
        return {'status': 'too early: the bar labelled 14:00 has not closed'}
    if today in logged_dates(read_log(cfg, 'forecasts')):
        return {'status': 'already logged'}
    model = coefficients(cfg, ols_path)
    top_up(cfg, feed, now)
    gold = C.load(cfg, cfg['mt5']['gold_symbol'])
    if not len(gold) or gold.index.max() < today + pd.Timedelta(hours=int(cfg['forward']['pre_session_last_hour'])):
        return {'status': 'waiting: the bar labelled 14:00 is not in the cache yet'}
    row = forecast_row(cfg, today, model)
    row['logged_at'] = now
    append_log(cfg, 'forecasts', [row])
    return {'status': 'logged', 'row': row, 'export': _rebuild_export(cfg, now)}


def run_settle(cfg: dict | None = None, feed=None, gate=None) -> dict:
    cfg, feed, now, today = _start(cfg, feed, gate)
    if today.dayofweek >= 5:
        return {'status': 'weekend'}
    if now < today + pd.Timedelta(minutes=minutes(cfg['forward']['settle_after'])):
        return {'status': 'too early: the bar labelled 21:00 has not closed'}
    fc = read_log(cfg, 'forecasts')
    done = logged_dates(read_log(cfg, 'settlements'))
    due = fc[(fc['status'] == 'ok') & (~fc['date'].isin(done)) & (fc['date'] <= today)] if len(fc) else fc
    if not len(due):
        return {'status': 'nothing to settle'}
    top_up(cfg, feed, now, contracts_too=False)
    gold = C.load(cfg, cfg['mt5']['gold_symbol'])
    gb = gb_config()
    rows = []
    for _, f in due.sort_values('date').iterrows():
        r = settle_row(cfg, f, gold, gb)
        if r is not None:
            r['settled_at'] = now
            rows.append(r)
    if not rows:
        return {'status': 'waiting: the bar labelled 21:00 is not in the cache yet'}
    append_log(cfg, 'settlements', rows)
    return {'status': 'settled', 'rows': rows, 'export': _rebuild_export(cfg, now)}


def main(argv=None):
    ap = argparse.ArgumentParser(description='one live gold-nyrange-forward pass')
    ap.add_argument('--pass', dest='which', choices=['forecast', 'settle'], required=True)
    a = ap.parse_args(argv)
    out = (run_forecast if a.which == 'forecast' else run_settle)()
    print(datetime.now(timezone.utc).isoformat(timespec='seconds'), a.which,
          {k: v for k, v in out.items() if k not in ('rows',)})


if __name__ == '__main__':
    main()
