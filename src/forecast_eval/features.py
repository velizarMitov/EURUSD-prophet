"""Read-only adapters over the project's feature builders (spec horizon-study
"Standalone from production code"; design D1).

Features are COMPUTED by the production modules, imported and called unchanged,
so the challengers see the market exactly as the production models do. This
module only selects columns and assembles rows. It never builds a target --
where a production builder also returns one, it is discarded, and the study
attaches its own horizon target from targets.py.

  daily_features      src.features.compute_features on OHLCV + merged macro,
                      the euro-era row set, columns by feature set
  h1_bar_features     src.h1_features.compute_h1_direction_features (served
                      definition; byte-identical to the pooled research one)
  h1_daily_features   src.h1_features.aggregate_daily_features +
                      build_lstm_tensor (the H1->daily ensemble's inputs)
  ti_daily_sequences  src.ti_lstm_h1_experimental's (24, 8) daily tensors

Every builder here keeps the newest complete bar or day, which a production
TRAINING builder drops because it has no target yet: the forward logger has to
predict from exactly that row.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modules whose code determines a challenger's input. Their digest is stored
# with every model version, and the forward logger refuses to predict when the
# code has changed since the version was trained (design D11).
FEATURE_MODULES = (
    'src/features.py',
    'src/macro_data.py',
    'src/h1_features.py',
    'src/triple_barrier.py',
    'src/ti_lstm_h1_experimental.py',
)

FEATURE_SETS = ('price', 'macro')


# ── daily ───────────────────────────────────────────────────────────────────

def daily_columns(feature_set: str) -> list:
    from src.features import variant_feature_columns
    return variant_feature_columns({'price': 'baseline', 'macro': 'with_macro'}[feature_set])


def daily_features(ohlcv: pd.DataFrame, macro_df: pd.DataFrame | None, feature_set: str) -> pd.DataFrame:
    """Production daily features, one row per daily bar, plus `close`.

    Rows: those with every one of the 27 FEATURE_COLUMNS defined -- the same
    euro-era row set for BOTH feature sets, as in production, so price-only vs
    with-macro differ in columns only."""
    from src.features import FEATURE_COLUMNS, compute_features, merge_macro_features
    if feature_set not in FEATURE_SETS:
        raise KeyError(f'unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}')
    raw = ohlcv[['open', 'high', 'low', 'close', 'tick_volume']]
    merged = merge_macro_features(raw, macro_df if macro_df is not None else pd.DataFrame(index=raw.index))
    feats = compute_features(merged).dropna(subset=FEATURE_COLUMNS)
    cols = daily_columns(feature_set)
    out = feats[cols].copy()
    if 'close' not in cols:
        out['close'] = feats['close']
    return out


def load_daily_ohlcv(path: str | None = None) -> pd.DataFrame:
    from src.features import DEFAULT_HISTORY_PATH
    df = pd.read_csv(path or DEFAULT_HISTORY_PATH, index_col='time', parse_dates=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]


# ── H1 ──────────────────────────────────────────────────────────────────────

def normalize_h1(df: pd.DataFrame) -> pd.DataFrame:
    from src.h1_features import _normalize_h1
    if 'tick_volume' not in df.columns:
        df = df.assign(tick_volume=0.0)
    return _normalize_h1(df)


def load_h1(path: str | None = None) -> pd.DataFrame:
    from src.h1_features import DEFAULT_H1_CACHE
    return normalize_h1(pd.read_csv(path or DEFAULT_H1_CACHE, index_col=0, parse_dates=True))


def h1_bar_features(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 next-bar features (served definition) with `close`; warm-up rows dropped."""
    from src.h1_features import compute_h1_direction_features
    f = compute_h1_direction_features(h1).dropna()
    return f.assign(close=h1['close'].reindex(f.index))


def h1_daily_features(h1: pd.DataFrame):
    """(flat_df, seq_tensor, daily_close) for the H1->daily ensemble, one row per
    complete session, INCLUDING the newest session."""
    from src.h1_features import aggregate_daily_features, build_lstm_tensor
    flat, daily_close = aggregate_daily_features(h1)
    return flat, build_lstm_tensor(h1, flat.index), daily_close


def ti_daily_sequences(h1: pd.DataFrame):
    """(X, index, daily_close) of TI-LSTM's (24, n) right-aligned tensors for
    every complete session, INCLUDING the newest one.

    build_ti_datasets drops the newest session because its target is not yet
    known. Rather than re-implement its tensor assembly, one synthetic session
    is appended after the real data, so that the real newest session gets a
    (meaningless, discarded) target and stays in the output. Every indicator is
    trailing, so bars appended AFTER a session cannot change its features; a
    parity test holds that against build_ti_datasets."""
    from src.h1_features import MIN_HOURS
    from src.ti_lstm_h1_experimental import build_ti_datasets
    days = h1.index.normalize()
    counts = pd.Series(1, index=days).groupby(level=0).size()
    complete = counts[counts >= MIN_HOURS].index
    if complete.empty:
        raise ValueError('no complete H1 session in the frame')
    # The pad must itself be a COMPLETE session, dated after every real bar,
    # or the newest real session still has no next session and is dropped.
    pad = h1[days == complete[-1]].copy()
    pad.index = pad.index + (days[-1] - complete[-1]) + pd.Timedelta(days=1)
    X, _y_ret, _y_dir, index = build_ti_datasets(h1=pd.concat([h1, pad]))
    daily_close = h1['close'].groupby(h1.index.normalize()).last().reindex(index)
    return X, index, daily_close


# ── code digest ─────────────────────────────────────────────────────────────

def file_digest(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def feature_code_digest(paths=None, base: str = REPO) -> dict:
    """{relative path: sha256} for every feature module, plus a combined
    '_combined' digest over them in sorted order."""
    rels = sorted(paths or FEATURE_MODULES)
    out = {rel: file_digest(os.path.join(base, rel)) for rel in rels}
    h = hashlib.sha256()
    for rel in rels:
        h.update(rel.encode())
        h.update(out[rel].encode())
    out['_combined'] = h.hexdigest()
    return out
