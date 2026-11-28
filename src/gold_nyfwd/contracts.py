"""The dollar-index contract roll and its lag check (spec gold-nyrange-forward/live-data; design D6).

The rule, a pure function of the day and the known expiries: use the `USDInd` contract with the nearest
expiry at least `roll_margin_trading_days` (2) trading days after the day, i.e. with at least two
weekdays in (day, expiry]. USDIndDec26 (expiry 2026-12-11) therefore covers 2026-12-09 and the next
contract covers 2026-12-10. Every contract ever listed is remembered in
research_models/gold_nyfwd/contracts.json with its expiry, so an earlier day keeps its contract after the
broker delists it.

Lag check, before a contract is first used: on the label hours both caches share, the correlation of
the contract's hourly log returns with EURUSD's shifted by each lag in ±3 h must be most negative at
lag 0, over at least `min_returns` returns. A pass is recorded and kept; a fail is recorded and the
contract is never used (its days are `no dollar data`); too few returns is not recorded and is checked
again on the next pass.

The dollar frame for gold_bbnr's `dollar_pre_range_ratio` takes, for every gold date, the bars of that
date's contract labelled on that date. Ranges of different contracts are mixed in the 20-day mean
without adjustment, exactly as the feature is defined.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import art_dir

FILE = 'contracts.json'
OHLC = ['open', 'high', 'low', 'close']


def path(cfg: dict) -> str:
    return os.path.join(art_dir(cfg), FILE)


def read(cfg: dict) -> dict:
    p = path(cfg)
    if not os.path.exists(p):
        return {'contracts': {}}
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def write(cfg: dict, rec: dict) -> None:
    os.makedirs(art_dir(cfg), exist_ok=True)
    tmp = path(cfg) + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    os.replace(tmp, path(cfg))


def remember(cfg: dict, listed: dict, now) -> dict:
    """Add every listed contract (name -> expiry date) not yet known. Known entries never change."""
    rec = read(cfg)
    pre = cfg['mt5']['contract_prefix']
    changed = False
    for name, exp in sorted(listed.items()):
        if name.startswith(pre) and exp is not None and name not in rec['contracts']:
            rec['contracts'][name] = {'expiry': str(pd.Timestamp(exp).date()), 'first_seen': str(now)}
            changed = True
    if changed:
        write(cfg, rec)
    return rec


def expiries(rec: dict) -> dict:
    return {n: pd.Timestamp(c['expiry']) for n, c in rec['contracts'].items()}


def trading_days_until(day, expiry) -> int:
    """Weekdays in (day, expiry]."""
    a = (pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)).normalize()
    b = pd.Timestamp(expiry).normalize()
    return int(np.busday_count((a + pd.Timedelta(days=1)).date(), (b + pd.Timedelta(days=1)).date())) if b > a else 0


def choose(day, exp: dict, margin: int) -> str | None:
    ok = [(e, n) for n, e in exp.items() if trading_days_until(day, e) >= margin]
    return min(ok)[1] if ok else None


def lag_correlations(dxy: pd.DataFrame, eur: pd.DataFrame, lags) -> tuple:
    j = pd.concat([np.log(dxy['close']).diff().rename('dx'), np.log(eur['close']).diff().rename('eu')], axis=1)
    j = j[(j.index >= dxy.index.min()) & (j.index <= dxy.index.max())]
    n = int(j.dropna().shape[0])
    return {int(lag): float(j['dx'].corr(j['eu'].shift(int(lag)))) for lag in lags}, n


def lag_check(dxy: pd.DataFrame, eur: pd.DataFrame, cfg: dict) -> dict:
    lc = cfg['lag_check']
    if len(dxy) < 2 or len(eur) < 2:
        return {'status': 'insufficient', 'n_returns': 0}
    corr, n = lag_correlations(dxy, eur, lc['lags'])
    if n < int(lc['min_returns']) or any(np.isnan(v) for v in corr.values()):
        return {'status': 'insufficient', 'n_returns': n}
    best = min(corr, key=lambda k: corr[k])
    return {'status': 'pass' if best == 0 else 'fail', 'best_lag': int(best), 'n_returns': n,
            'corr': {str(k): round(v, 4) for k, v in corr.items()}}


def check_pending(cfg: dict, rec: dict, load_bars, eur: pd.DataFrame, now) -> dict:
    """Run the lag check for every known contract without a recorded result; record pass or fail."""
    changed = False
    for name, c in sorted(rec['contracts'].items()):
        if c.get('lag_check', {}).get('status') in ('pass', 'fail'):
            continue
        res = lag_check(load_bars(name), eur, cfg)
        if res['status'] in ('pass', 'fail'):
            c['lag_check'] = {**res, 'checked_at': str(now)}
            changed = True
    if changed:
        write(cfg, rec)
    return rec


def usable(rec: dict, name: str | None) -> bool:
    return name is not None and rec['contracts'].get(name, {}).get('lag_check', {}).get('status') == 'pass'


def day_map(dates, rec: dict, margin: int) -> dict:
    """{gold date: contract or None}; None also when the chosen contract has not passed the lag check."""
    exp = expiries(rec)
    out = {}
    for d in pd.DatetimeIndex(dates):
        n = choose(d, exp, margin)
        out[d] = n if usable(rec, n) else None
    return out


def dollar_frame(dates, mapping: dict, load_bars) -> pd.DataFrame:
    """H1 bars of each date's own contract labelled on that date, stacked into one frame."""
    parts, cache = [], {}
    dates = pd.DatetimeIndex(dates)
    for name in sorted({n for n in mapping.values() if n is not None}):
        cache[name] = load_bars(name)
    for d in dates:
        n = mapping.get(d)
        if n is None or not len(cache[n]):
            continue
        b = cache[n]
        parts.append(b[b.index.normalize() == d])
    if not parts:
        return pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], tz='UTC', name='time'), dtype=float)
    return pd.concat(parts).sort_index()[OHLC]
