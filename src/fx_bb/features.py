"""The ten oriented Cell 2 features at the close of the event day t (spec fx-bb-reversion/event-features;
design D5).

Each feature uses only the pair's own valid D1 days up to t, plus the other pairs' closes on t
(all pairs close together at 17:00 New York). The breach side s = -dir (+1 upper, -1 lower);
directional features are multiplied by s, so pooled long and short events share one meaning.

  bb_zscore_abs          |close - SMA20| / SD20
  bb_width_rank_60d      share of t-59 .. t whose band width 4*SD20/SMA20 is <= today's
  rsi_14_dist            s * (RSI14 - 50), Wilder RSI on mid closes
  bar_range_vs_atr20     (bid high - bid low) / ATR20
  close_pos_in_bar       bid close's position in the bid range toward the breach side (1 = at the extreme;
                         0.5 when the range is zero)
  ret_3d_atr             s * (close[t] - close[t-3]) / ATR20
  trend_ma50_alignment   s * (close - SMA50) / ATR20
  atr_ratio_20_60        ATR20 / ATR60
  simultaneous_frac      share of the OTHER started pairs whose mid close on t is outside their own bands
  breach_side            s

All are scale-free. An event without every feature is excluded from Cell 2 only ("warm-up").
No pair identity is a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import events as E
from .daily import valid


class FeatureColumnsError(ValueError):
    pass


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    gain, loss = d.clip(lower=0), (-d).clip(lower=0)
    a = 1.0 / period
    ag = gain.ewm(alpha=a, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=a, adjust=False, min_periods=period).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rsi = pd.Series(np.where(al == 0, 100.0, 100 - 100 / (1 + ag / al)), index=close.index)
    return rsi.where(ag.notna() & al.notna())


def pair_frame(d1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every day's unoriented building blocks on the pair's valid days."""
    f = cfg['features']
    v = valid(d1)
    ind = E.indicators(v, cfg)
    c = ind['close']
    out = pd.DataFrame(index=v.index)
    out['zabs'] = (c - ind['sma']).abs() / ind['sd']
    width = 4 * ind['sd'] / ind['sma']
    n = int(f['width_rank_days'])
    out['width_rank'] = width.rolling(n).apply(lambda w: float(np.mean(w <= w[-1])), raw=True)
    out['rsi'] = wilder_rsi(c, int(f['rsi_period']))
    rng = v['high'] - v['low']
    out['range_atr'] = rng / ind['atr']
    out['pos_up'] = np.where(rng > 0, (v['close_bid'] - v['low']) / rng.where(rng > 0), 0.5)
    m = int(f['momentum_days'])
    out['ret_atr'] = (c - c.shift(m)) / ind['atr']
    out['ma50_atr'] = (c - c.rolling(int(f['sma_slow'])).mean()) / ind['atr']
    prev = v['close_bid'].shift(1)
    tr = pd.concat([rng, (v['high'] - prev).abs(), (v['low'] - prev).abs()], axis=1).max(axis=1)
    tr[prev.isna()] = np.nan
    out['atr_ratio'] = ind['atr'] / tr.rolling(int(f['atr_slow'])).mean()
    out['outside'] = ind['outside'] & ind['sma'].notna()
    return out


def build_features(ev: pd.DataFrame, d1_all: dict, cfg: dict) -> pd.DataFrame:
    """One row per event (same index as `ev`) with the configured columns, oriented by side."""
    blocks = {p: pair_frame(d1_all[p], cfg) for p in cfg['pairs']}
    starts = {p: pd.Timestamp(cfg['data']['start'][p]) for p in cfg['pairs']}
    outside = pd.concat({p: b['outside'].astype(float) for p, b in blocks.items()}, axis=1).fillna(0.0) > 0
    out = pd.DataFrame(index=ev.index, columns=cfg['features']['columns'], dtype=float)
    for i, r in ev.iterrows():
        b = blocks[r['pair']].loc[r['date']]
        s = float(r['side'])
        others = [p for p in cfg['pairs'] if p != r['pair'] and starts[p] <= r['date']]
        row = outside.loc[r['date'], others] if r['date'] in outside.index else pd.Series(False, index=others)
        out.loc[i] = {
            'bb_zscore_abs': b['zabs'],
            'bb_width_rank_60d': b['width_rank'],
            'rsi_14_dist': s * (b['rsi'] - 50.0),
            'bar_range_vs_atr20': b['range_atr'],
            'close_pos_in_bar': b['pos_up'] if s > 0 else 1.0 - b['pos_up'],
            'ret_3d_atr': s * b['ret_atr'],
            'trend_ma50_alignment': s * b['ma50_atr'],
            'atr_ratio_20_60': b['atr_ratio'],
            'simultaneous_frac': float(row.mean()) if len(others) else 0.0,
            'breach_side': s,
        }
    return out.astype(float)


def assemble(feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The configured columns in order; raise on any extra or missing column."""
    want = list(cfg['features']['columns'])
    extra = [c for c in feats.columns if c not in want]
    missing = [c for c in want if c not in feats.columns]
    if extra or missing:
        raise FeatureColumnsError(f'feature frame does not match the config: extra {extra}, missing {missing}')
    return feats[want]


def with_features(ev: pd.DataFrame, d1_all: dict, cfg: dict) -> pd.DataFrame:
    """Events plus the features, with reason 'warm-up' (Cell 2 only) where any feature is missing.
    The Cell 1 `reason` column is kept unchanged; Cell 2 uses `reason2`."""
    feats = assemble(build_features(ev, d1_all, cfg), cfg)
    out = ev.join(feats)
    warm = feats.isna().any(axis=1).to_numpy()
    out['reason2'] = np.where(out['reason'] != '', out['reason'], np.where(warm, 'warm-up', ''))
    return out


def summary(cfg: dict | None = None) -> pd.DataFrame:
    """Print the real-data event counts with features, exclusions and a feature summary.
    Raises if any feature column is constant over the Cell 2 events. Reads no outcome."""
    from . import load_config
    from .daily import build_all
    from .data_loader import load_all
    cfg = cfg or load_config()
    d1_all = build_all(*load_all(cfg), cfg)
    ev, _ = E.build_events(d1_all, cfg)
    out = with_features(ev, d1_all, cfg)
    use = out[out['reason2'] == '']
    print(f'events {len(out)}; Cell 2 events with every feature {len(use)} '
          f"(from {cfg['folds']['first_eval_start']}: {int((use['date'] >= cfg['folds']['first_eval_start']).sum())})")
    print('exclusions (Cell 2):', out.loc[out['reason2'] != '', 'reason2'].value_counts().to_dict())
    print('events with features per pair:', use['pair'].value_counts().reindex(cfg['pairs']).to_dict())
    desc = use[cfg['features']['columns']].describe().T[['mean', 'std', 'min', '50%', 'max']]
    print(desc.round(4).to_string())
    const = [c for c in cfg['features']['columns'] if use[c].nunique() < 2]
    if const:
        raise FeatureColumnsError(f'constant feature columns: {const}')
    return desc


if __name__ == '__main__':
    summary()
