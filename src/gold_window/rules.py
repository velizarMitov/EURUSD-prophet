"""The four window rules, window eligibility and the gates (spec window-strategy;
design D3).

Every rule returns its CANDIDATE trades: one row per trade the rule would take
with its gate open, carrying the gate value, its threshold and `gate_open`.
The backtest filters on `gate_open`; the gate-always-open comparator and the
placebo reuse the same rows. Rows are pure data: no fills, no costs.

Clock: labels are Berlin wall time; Sofia = Berlin + 1 h. An H1 bar labelled L
closes at L + 1 h, an M15 bar at label + 15 min. Entry and exit times are bar
CLOSE times on that clock, so every trade lies inside 14:30-22:00 Berlin =
15:30-23:00 Sofia, and nothing is held past 22:00 Berlin.

Column contract:
  h1    time, close, spread_usd
  m15   time, high, low, close, spread_usd
  h1 forecasts   time (as-of H1 label), p_up, vol_pct
  d1 forecasts   time (D1 label), p_up (may be NaN), vol_pct
"""

from __future__ import annotations

import numpy as np
import pandas as pd

H1 = pd.Timedelta(hours=1)
M15 = pd.Timedelta(minutes=15)
TRADE_COLUMNS = ['rule', 'day', 'entry_time', 'entry_price', 'entry_spread', 'side',
                 'exit_time', 'exit_price', 'exit_spread', 'gate_value', 'threshold', 'gate_open']


def _hm(s: str) -> pd.Timedelta:
    h, m = s.split(':')
    return pd.Timedelta(hours=int(h), minutes=int(m))


# ── eligibility ─────────────────────────────────────────────────────────────

def eligible_days_h1(h1: pd.DataFrame, labels=range(14, 22), weekdays=range(5)) -> pd.DatetimeIndex:
    """Weekdays with every H1 bar labelled 14:00-21:00 present."""
    t = h1['time']
    t = t[t.dt.hour.isin(list(labels)) & (t.dt.minute == 0)]
    day = t.dt.normalize()
    counts = t.groupby(day).nunique()
    ok = counts[counts == len(list(labels))].index
    return pd.DatetimeIndex([d for d in ok if d.dayofweek in weekdays])


def eligible_days_m15(m15: pd.DataFrame, first='14:30', last='21:45', weekdays=range(5)) -> pd.DatetimeIndex:
    """Weekdays with every M15 bar labelled first..last present."""
    lo, hi = _hm(first), _hm(last)
    need = int((hi - lo) / M15) + 1
    t = m15['time']
    day = t.dt.normalize()
    tod = t - day
    inwin = (tod >= lo) & (tod <= hi)
    counts = t[inwin].groupby(day[inwin]).nunique()
    ok = counts[counts == need].index
    return pd.DatetimeIndex([d for d in ok if d.dayofweek in weekdays])


# ── gates ───────────────────────────────────────────────────────────────────

def trailing_threshold(values: pd.Series, q: float, lookback: int) -> pd.Series:
    """For each entry, the q-quantile of the `lookback` finite values strictly
    before it (index order); NaN until `lookback` prior values exist."""
    out = pd.Series(np.nan, index=values.index)
    v = values.dropna()
    arr = v.to_numpy(float)
    for j in range(lookback, len(arr)):
        out.loc[v.index[j]] = np.quantile(arr[j - lookback:j], q, method='linear')
    return out


def pooled_threshold(rows: pd.DataFrame, q: float, lookback: int) -> pd.Series:
    """Per row, the q-quantile of every finite value on the `lookback` distinct
    days with values strictly before the row's day; NaN until that many exist."""
    r = rows[['day', 'value']].dropna().sort_values('day', kind='stable')
    codes, days = pd.factorize(r['day'], sort=True)
    starts = np.searchsorted(codes, np.arange(len(days)), side='left')
    arr = r['value'].to_numpy(float)
    thr_by_day = {days[i]: np.quantile(arr[starts[i - lookback]:starts[i]], q, method='linear')
                  for i in range(lookback, len(days))}
    # a day with no finite value of its own still gets the threshold from the days before it
    for d in pd.DatetimeIndex(rows['day'].dropna().unique()):
        if d not in thr_by_day:
            i = int(np.searchsorted(days, d))
            if i >= lookback:
                thr_by_day[d] = np.quantile(arr[starts[i - lookback]:starts[i] if i < len(days) else len(arr)],
                                            q, method='linear')
    return rows['day'].map(thr_by_day).astype(float)


def daily_forecast(d1_pred: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    """For each trading day, `col` of the last D1 bar labelled strictly before it."""
    left = pd.DataFrame({'day': pd.DatetimeIndex(days)}).sort_values('day')
    right = d1_pred[['time', col]].dropna().sort_values('time')
    m = pd.merge_asof(left, right, left_on='day', right_on='time', direction='backward',
                      allow_exact_matches=False)
    return pd.Series(m[col].to_numpy(float), index=pd.DatetimeIndex(m['day']))


def daily_gate(d1_pred: pd.DataFrame, calendar: pd.DatetimeIndex, q: float, lookback: int) -> pd.DataFrame:
    """Gate value and threshold per trading day of `calendar` (vol_pct of the
    last D1 bar before the day, against its trailing quantile)."""
    f = daily_forecast(d1_pred, calendar, 'vol_pct')
    thr = trailing_threshold(f, q, lookback)
    return pd.DataFrame({'gate_value': f, 'threshold': thr})


def _open(value, thr) -> bool:
    return bool(np.isfinite(value) and np.isfinite(thr) and value >= thr)


def _frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _bar(h1i: pd.DataFrame, day, label: int):
    """(close, spread) of the H1 bar labelled `label` on `day`, or None."""
    key = day + pd.Timedelta(hours=label)
    if key not in h1i.index:
        return None
    r = h1i.loc[key]
    return float(r['close']), float(r['spread_usd'])


# ── the rules ───────────────────────────────────────────────────────────────

def h1_gated_direction(h1: pd.DataFrame, h1_pred: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    p = cfg['rules']['h1_gated_direction']
    lookback = cfg['rules']['gate_lookback_days']
    dayset = set(pd.DatetimeIndex(days))
    f = h1_pred[['time', 'p_up', 'vol_pct']].copy()
    f['day'] = f['time'].dt.normalize()
    f = f[f['time'].dt.hour.isin(p['as_of_labels']) & (f['time'].dt.minute == 0) & f['day'].isin(dayset)]
    f = f.sort_values('time').reset_index(drop=True)
    f['value'] = f['vol_pct']
    f['threshold'] = pooled_threshold(f, p['gate_quantile'], lookback)
    h1i = h1.set_index('time')
    rows = []
    for r in f.itertuples(index=False):
        if not (np.isfinite(r.p_up) and np.isfinite(r.vol_pct)):
            continue
        label = r.time.hour
        a, b = _bar(h1i, r.day, label), _bar(h1i, r.day, label + 1)
        if a is None or b is None:
            continue
        rows.append(['h1_gated_direction', r.day, r.time + H1, a[0], a[1], 1 if r.p_up > 0.5 else -1,
                     r.time + 2 * H1, b[0], b[1], r.vol_pct, r.threshold, _open(r.vol_pct, r.threshold)])
    return _frame(rows)


def continuation(h1: pd.DataFrame, gate: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    p = cfg['rules']['continuation']
    h1i = h1.set_index('time')
    frm = h1[(h1['time'].dt.hour == p['from_label']) & (h1['time'].dt.minute == 0)][['time', 'close']]
    frm = frm.sort_values('time')
    rows = []
    for day in pd.DatetimeIndex(days):
        prev = frm[frm['time'] < day]
        if prev.empty:
            continue
        a, b = _bar(h1i, day, p['to_label']), _bar(h1i, day, p['exit_label'])
        if a is None or b is None:
            continue
        move = a[0] - float(prev['close'].iloc[-1])
        if move == 0:
            continue
        g = gate.loc[day] if day in gate.index else None
        gv, th = (np.nan, np.nan) if g is None else (float(g['gate_value']), float(g['threshold']))
        rows.append(['continuation', day, day + pd.Timedelta(hours=p['to_label']) + H1, a[0], a[1],
                     1 if move > 0 else -1, day + pd.Timedelta(hours=p['exit_label']) + H1, b[0], b[1],
                     gv, th, _open(gv, th)])
    return _frame(rows)


def us_open_breakout(m15: pd.DataFrame, gate: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    p = cfg['rules']['us_open_breakout']
    r_lo, r_hi = (_hm(x) - M15 for x in p['range_closes'])      # labels of the range bars
    e_lo, e_hi = (_hm(x) - M15 for x in p['entry_closes'])      # labels of the entry candidates
    exit_label = pd.Timedelta(hours=p['exit_label'], minutes=45)
    m = m15.copy()
    m['day'] = m['time'].dt.normalize()
    m['tod'] = m['time'] - m['day']
    by_day = {d: g for d, g in m.groupby('day')}
    rows = []
    for day in pd.DatetimeIndex(days):
        g = by_day.get(day)
        if g is None:
            continue
        rng = g[(g.tod >= r_lo) & (g.tod <= r_hi)]
        cand = g[(g.tod >= e_lo) & (g.tod <= e_hi)].sort_values('time')
        ex = g[g.tod == exit_label]
        if rng.empty or ex.empty:
            continue
        hi, lo = float(rng['high'].max()), float(rng['low'].min())
        side, entry = 0, None
        for c in cand.itertuples(index=False):
            if c.close > hi:
                side, entry = 1, c
                break
            if c.close < lo:
                side, entry = -1, c
                break
        if not side:
            continue
        x = ex.iloc[0]
        gr = gate.loc[day] if day in gate.index else None
        gv, th = (np.nan, np.nan) if gr is None else (float(gr['gate_value']), float(gr['threshold']))
        rows.append(['us_open_breakout', day, entry.time + M15, float(entry.close), float(entry.spread_usd), side,
                     x['time'] + M15, float(x['close']), float(x['spread_usd']), gv, th, _open(gv, th)])
    return _frame(rows)


def d1_gated_direction(h1: pd.DataFrame, d1_pred: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    p = cfg['rules']['d1_gated_direction']
    lookback = cfg['rules']['gate_lookback_days']
    days = pd.DatetimeIndex(days)
    pu = daily_forecast(d1_pred, days, 'p_up')
    vol = daily_forecast(d1_pred, days, 'vol_pct')
    thr = trailing_threshold(vol, p['gate_quantile'], lookback)
    h1i = h1.set_index('time')
    rows = []
    for day in days:
        if not (np.isfinite(pu.get(day, np.nan)) and np.isfinite(vol.get(day, np.nan))):
            continue
        a, b = _bar(h1i, day, p['entry_label']), _bar(h1i, day, p['exit_label'])
        if a is None or b is None:
            continue
        rows.append(['d1_gated_direction', day, day + pd.Timedelta(hours=p['entry_label']) + H1, a[0], a[1],
                     1 if pu[day] > 0.5 else -1, day + pd.Timedelta(hours=p['exit_label']) + H1, b[0], b[1],
                     float(vol[day]), float(thr[day]), _open(vol[day], thr[day])])
    return _frame(rows)
