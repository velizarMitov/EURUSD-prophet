"""Bollinger events, the cooldown, the pair-gap exclusion and the outcomes
(spec fx-bb-reversion/band-events; design D4).

All windows run over the pair's own VALID D1 days (a bar with a usable mid close).

  bands    SMA20 +/- 2 * SD20 (population SD) of the closes of t-19 .. t
  ATR20    simple mean of the bid true range of t-19 .. t (needs the bid close of t-20)
  event    close[t] outside its bands while close[t-1] was inside the bands of t-1
           (direction -1 above the upper band, +1 below the lower band)
  cooldown an event is accepted only when >= 3 valid days have passed since the pair's last
           accepted event; applied to the signal, before any exclusion
  outcome  r = dir * (close[t+3] - close[t]) / ATR20[t], t+3 the third valid day after t

The reference calendar is every date on which >= `reference_quorum` started pairs have a valid
D1 bar. An event is excluded as `pair gap` when its pair lacks a reference date between the dates
of t-19 and t+3, and as `no outcome` when t+3 does not exist yet.

Event detection never reads a close after t: `detect` stores only the DATE of t+3. Outcomes are
computed separately by `outcomes`; the label report never calls it.
GROSS of cost: no spread, stop or target enters any outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .daily import valid

PRICE_COL = {'mid': 'close', 'bid': 'close_bid'}


def indicators(v: pd.DataFrame, cfg: dict, price: str = 'mid') -> pd.DataFrame:
    """Bands, ATR and breach flags on a pair's valid days `v` (sorted)."""
    n, w, ddof = int(cfg['bands']['length']), float(cfg['bands']['width']), int(cfg['bands']['ddof'])
    c = v[PRICE_COL[price]]
    sma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=ddof)
    prev = v['close_bid'].shift(1)
    tr = pd.concat([v['high'] - v['low'], (v['high'] - prev).abs(), (v['low'] - prev).abs()], axis=1).max(axis=1)
    tr[prev.isna()] = np.nan
    atr = tr.rolling(int(cfg['atr_period'])).mean()
    out = pd.DataFrame({'close': c, 'sma': sma, 'sd': sd, 'upper': sma + w * sd, 'lower': sma - w * sd, 'atr': atr},
                       index=v.index)
    above, below = c > out['upper'], c < out['lower']
    out['outside'] = above | below
    out['dir'] = np.where(above, -1, np.where(below, 1, 0))
    ready = sma.notna() & sma.shift(1).notna() & atr.notna()
    out['first_breach'] = out['outside'] & ~out['outside'].shift(1, fill_value=False) & ready
    return out


def apply_cooldown(positions, cooldown: int) -> tuple:
    """(accepted positions, removed positions): accept when >= cooldown days since the last accepted."""
    acc, rem, last = [], [], None
    for k in positions:
        if last is None or k - last >= cooldown:
            acc.append(int(k))
            last = k
        else:
            rem.append(int(k))
    return acc, rem


def detect(d1: pd.DataFrame, pair: str, cfg: dict, price: str = 'mid') -> tuple:
    """(events frame, counts) for one pair. No close after t is read."""
    v = valid(d1)
    ind = indicators(v, cfg, price)
    h, n = int(cfg['horizon_days']), int(cfg['bands']['length'])
    first = np.flatnonzero(ind['first_breach'].to_numpy())
    acc, rem = apply_cooldown(first, int(cfg['cooldown_days']))
    dates = v.index
    rows = []
    for k in acc:
        r = ind.iloc[k]
        rows.append({'pair': pair, 'date': dates[k], 'dir': int(r['dir']), 'side': -int(r['dir']), 'k': k,
                     'close': r['close'], 'sma': r['sma'], 'sd': r['sd'], 'upper': r['upper'], 'lower': r['lower'],
                     'atr': r['atr'], 'date_t19': dates[k - (n - 1)],
                     'date_t3': dates[k + h] if k + h < len(dates) else pd.NaT})
    ev = pd.DataFrame(rows, columns=['pair', 'date', 'dir', 'side', 'k', 'close', 'sma', 'sd', 'upper', 'lower', 'atr',
                                     'date_t19', 'date_t3'])
    counts = {'outside_days': int(ind['outside'].sum()), 'first_breaches': len(first), 'cooldown_removed': len(rem),
              'valid_days': len(v)}
    return ev, counts


def reference_calendar(d1_all: dict, cfg: dict) -> pd.DatetimeIndex:
    have = pd.concat({p: pd.Series(1, index=valid(d).index) for p, d in d1_all.items()}, axis=1).fillna(0)
    return pd.DatetimeIndex(have.index[have.sum(axis=1) >= int(cfg['reference_quorum'])], name='date').sort_values()


def pair_gap(ev: pd.DataFrame, pair_dates: pd.DatetimeIndex, ref: pd.DatetimeIndex) -> np.ndarray:
    """True where the pair lacks a reference date within [date_t19, date_t3] (date_t3 or date if none)."""
    have = set(pair_dates)
    out = np.zeros(len(ev), dtype=bool)
    refv = ref.to_numpy()
    for i, (a, b, t) in enumerate(zip(ev['date_t19'], ev['date_t3'], ev['date'])):
        end = b if pd.notna(b) else t
        lo, hi = np.searchsorted(refv, np.datetime64(a), 'left'), np.searchsorted(refv, np.datetime64(end), 'right')
        out[i] = any(pd.Timestamp(d) not in have for d in refv[lo:hi])
    return out


def build_events(d1_all: dict, cfg: dict, price: str = 'mid') -> tuple:
    """(all accepted events over the six pairs with `reason`, per-pair counts)."""
    ref = reference_calendar(d1_all, cfg)
    frames, counts = [], {}
    for pair in cfg['pairs']:
        ev, cnt = detect(d1_all[pair], pair, cfg, price)
        gap = pair_gap(ev, valid(d1_all[pair]).index, ref)
        ev['reason'] = np.where(ev['date_t3'].isna(), 'no outcome', np.where(gap, 'pair gap', ''))
        frames.append(ev)
        counts[pair] = cnt
    ev = pd.concat(frames, ignore_index=True).sort_values(['date', 'pair'], kind='mergesort').reset_index(drop=True)
    return ev, counts


# -- outcomes (never called by the label report) -------------------------------------------------

def outcome_r(direction, close_t, close_t3, atr):
    return direction * (close_t3 - close_t) / atr


def outcomes(ev: pd.DataFrame, d1_all: dict, price: str = 'mid') -> pd.DataFrame:
    """Add close_t3, r, the binary target y (NaN on a tie) and `tie`."""
    out = ev.copy()
    c3 = np.full(len(out), np.nan)
    for pair, idx in out.groupby('pair').groups.items():
        col = valid(d1_all[pair])[PRICE_COL[price]]
        sub = out.loc[idx]
        have = sub['date_t3'].notna()
        c3[np.asarray(idx)[have.to_numpy()]] = col.reindex(pd.DatetimeIndex(sub.loc[have, 'date_t3'])).to_numpy()
    out['close_t3'] = c3
    out['r'] = outcome_r(out['dir'].to_numpy(float), out['close'].to_numpy(float), c3, out['atr'].to_numpy(float))
    out['tie'] = out['r'] == 0
    out['y'] = np.where(out['r'] > 0, 1.0, np.where(out['r'] < 0, 0.0, np.nan))
    return out


def ties(ev: pd.DataFrame, d1_all: dict, price: str = 'mid') -> np.ndarray:
    """Equality only (close[t+3] == close[t]); never a sign or size."""
    out = np.zeros(len(ev), dtype=bool)
    for pair, idx in ev.groupby('pair').groups.items():
        col = valid(d1_all[pair])[PRICE_COL[price]]
        sub = ev.loc[idx]
        have = sub['date_t3'].notna().to_numpy()
        c3 = col.reindex(pd.DatetimeIndex(sub.loc[have, 'date_t3'])).to_numpy()
        out[np.asarray(idx)[have]] = c3 == sub.loc[have, 'close'].to_numpy()
    return out


def dispersion(d1_all: dict, cfg: dict, start=None) -> dict:
    """Per pair: SD of (close[t+3] - close[t]) / ATR20[t] over ALL valid days (never events),
    and the pooled value. Unconditional input to the power calculation."""
    h = int(cfg['horizon_days'])
    out, pooled = {}, []
    for pair in cfg['pairs']:
        v = valid(d1_all[pair])
        ind = indicators(v, cfg)
        x = ((ind['close'].shift(-h) - ind['close']) / ind['atr']).dropna()
        if start is not None:
            x = x[x.index >= pd.Timestamp(start)]
        out[pair] = float(x.std(ddof=1))
        pooled.append(x)
    out['pooled'] = float(pd.concat(pooled).std(ddof=1))
    return out
