"""
H1 -> Daily feature engineering for the auxiliary "H1-to-Daily" predictor.

Two *aligned* representations are produced from the same UTC-indexed H1 stream:

  * a FLATTENED daily feature table for the tree/SVM models -- one row per day
    of hand-engineered intraday statistics (volatility, momentum, range, ...).
    Collapsing 24 hourly bars into a few summary numbers keeps dimensionality
    low and strips microstructure noise, which is what keeps the tree/SVM
    models on the low-variance side of the bias-variance tradeoff.

  * a 3D TENSOR ``(samples, HOURS_PER_DAY, n_seq_features)`` for the LSTM, so the
    recurrent layers can process the intraday hourly sequence directly and learn
    ordering effects the flattened statistics throw away.

Both share ONE daily index and ONE ``shift(-1)`` next-day target, so every model
trains and is scored on identical samples. There is no look-ahead: the target for
day *t* is the return realised on *t+1*, and every feature for day *t* is built
strictly from that day's own hourly bars.
"""
import os

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24
MIN_HOURS = 12  # drop half-empty sessions (holidays / partial Fridays)

# Flattened daily features consumed by XGBoost / RandomForest / SVM.
FLAT_FEATURE_COLUMNS = [
    "Intraday_Volatility",   # std of the day's H1 log returns
    "Intraday_Momentum",     # last H1 close - first H1 open
    "Daily_Range",           # max H1 high - min H1 low
    "H1_Moving_Average",     # mean of the day's H1 closes
    "H1_Volume_Mean",        # mean tick volume across the day
    "H1_Return_Skew",        # skew of the H1 log returns (intraday asymmetry)
    "H1_Max_Abs_Return",     # largest single-hour move (intraday tail risk)
    "First_Half_Return",     # summed log return over the first half of the day
    "Second_Half_Return",    # summed log return over the second half of the day
]

# Per-hour features fed to the LSTM at each of the 24 timesteps.
SEQ_FEATURE_COLUMNS = ["log_return", "hl_range", "co_change", "volume"]

DEFAULT_H1_CACHE = "results/eurusd_h1.csv"


def load_h1_frame(cache_path: str = DEFAULT_H1_CACHE, allow_fetch: bool = True) -> pd.DataFrame:
    """
    Load H1 OHLCV as a UTC-indexed frame. Reads the cache CSV if present;
    otherwise (``allow_fetch``) pulls a fresh copy via ``src.live_data``.
    """
    df = None
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    elif allow_fetch:
        from src.live_data import fetch_h1_market_data
        df, _ = fetch_h1_market_data(cache_path=cache_path)

    if df is None or len(df) == 0:
        raise RuntimeError("No H1 data available (cache missing and live fetch failed).")

    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    df = df.sort_index()
    return df[['open', 'high', 'low', 'close', 'tick_volume']].astype(float)


def _enrich_hourly(h1: pd.DataFrame) -> pd.DataFrame:
    """Add the per-hour derived columns used by both representations and a
    ``date`` bucket (UTC midnight) for chronological grouping."""
    h1 = h1.copy()
    h1['log_return'] = np.log(h1['close'] / h1['close'].shift(1)).fillna(0.0)
    h1['hl_range'] = h1['high'] - h1['low']
    h1['co_change'] = h1['close'] - h1['open']
    h1['volume'] = h1['tick_volume']
    h1['date'] = h1.index.normalize()
    return h1


def aggregate_daily_features(h1: pd.DataFrame):
    """
    Collapse each UTC day's H1 bars into one flattened feature row.

    Returns ``(features_df, daily_close)`` both indexed by day (UTC midnight),
    with short/incomplete sessions (< ``MIN_HOURS`` bars) dropped.
    """
    h1 = _enrich_hourly(h1)
    g = h1.groupby('date')

    feats = pd.DataFrame({
        "Intraday_Volatility": g['log_return'].std(),
        "Intraday_Momentum":   g['close'].last() - g['open'].first(),
        "Daily_Range":         g['high'].max() - g['low'].min(),
        "H1_Moving_Average":   g['close'].mean(),
        "H1_Volume_Mean":      g['volume'].mean(),
        "H1_Return_Skew":      g['log_return'].skew(),
        "H1_Max_Abs_Return":   g['log_return'].apply(lambda s: s.abs().max()),
        "First_Half_Return":   g['log_return'].apply(lambda s: s.iloc[:len(s) // 2].sum()),
        "Second_Half_Return":  g['log_return'].apply(lambda s: s.iloc[len(s) // 2:].sum()),
        "hours_in_day":        g.size(),
    })
    feats = feats[feats['hours_in_day'] >= MIN_HOURS].drop(columns='hours_in_day')
    feats = feats[FLAT_FEATURE_COLUMNS].fillna(0.0)  # skew is NaN for tiny days

    daily_close = g['close'].last().reindex(feats.index)
    return feats, daily_close


def build_daily_target(daily_close: pd.Series) -> pd.Series:
    """
    Next-day (t+1) continuous return in PERCENT via strict ``shift(-1)``.

    ``r_t = 100 * ln(close_{t+1} / close_t)``. The shift guarantees the label of
    row *t* is only known *after* day *t* closes -- no look-ahead bias.
    """
    log_ret_pct = np.log(daily_close / daily_close.shift(1)) * 100.0
    return log_ret_pct.shift(-1)


def build_lstm_tensor(h1: pd.DataFrame, index: pd.DatetimeIndex,
                      hours: int = HOURS_PER_DAY, feature_cols=SEQ_FEATURE_COLUMNS) -> np.ndarray:
    """
    Shape each day's hourly bars into ``(hours, n_features)`` and stack them into
    a 3D tensor ``(len(index), hours, n_features)`` aligned to ``index``.

    Days are **right-aligned**: their last ``hours`` bars are kept, and short
    sessions are front-padded with zeros so every sample is a fixed-length
    sequence the recurrent layer can consume.
    """
    h1 = _enrich_hourly(h1)
    by_day = {d: grp for d, grp in h1.groupby('date')}
    n_feat = len(feature_cols)

    out = np.zeros((len(index), hours, n_feat), dtype=np.float32)
    for i, day in enumerate(index):
        grp = by_day.get(pd.Timestamp(day))
        if grp is None:
            continue
        arr = grp[feature_cols].to_numpy(dtype=np.float32)[-hours:]
        out[i, hours - len(arr):, :] = arr
    return out


def build_h1_datasets(cache_path: str = DEFAULT_H1_CACHE, h1: pd.DataFrame = None):
    """
    One call -> the fully aligned H1->Daily training set.

    Returns ``(X_flat_df, X_seq, y_return, y_direction, index)`` where:
      * ``X_flat_df`` is the flattened daily feature table (tree/SVM input),
      * ``X_seq`` is the ``(samples, 24, n_seq_features)`` tensor (LSTM input),
      * ``y_return`` is the next-day % return (regression target),
      * ``y_direction`` is its sign as 0/1 (for directional ROC-AUC),
      * ``index`` is the shared daily DatetimeIndex.

    The final day (whose ``shift(-1)`` target is NaN) and any other NaN-target
    rows are dropped from every representation so they stay row-aligned. ``h1``
    may be passed directly (tests); otherwise it is loaded from ``cache_path``.
    """
    h1 = load_h1_frame(cache_path) if h1 is None else h1
    feats, daily_close = aggregate_daily_features(h1)
    target = build_daily_target(daily_close)

    valid = target.notna()
    feats = feats[valid]
    target = target[valid]
    index = feats.index

    X_seq = build_lstm_tensor(h1, index)
    y_return = target.values.astype(np.float32)
    y_direction = (target.values > 0).astype(int)
    return feats, X_seq, y_return, y_direction, index


def build_h1_inference_sample(cache_path: str = DEFAULT_H1_CACHE, now=None, h1: pd.DataFrame = None):
    """
    Latest COMPLETE trading day's flattened features + ``(1, HOURS_PER_DAY, f)``
    tensor for live inference.

    The current (still-forming) UTC day is dropped so a live forecast never
    bases itself on a half-formed session -- the intraday analogue of
    ``live_data.drop_incomplete_bars``. ``h1`` may be passed directly (tests);
    otherwise it is loaded from ``cache_path``.

    Returns ``(flat_row_df, seq_tensor, as_of_date)`` where ``flat_row_df`` is a
    single-row DataFrame in ``FLAT_FEATURE_COLUMNS`` order and ``seq_tensor`` has
    shape ``(1, HOURS_PER_DAY, len(SEQ_FEATURE_COLUMNS))``.
    """
    h1 = load_h1_frame(cache_path) if h1 is None else h1
    feats, _ = aggregate_daily_features(h1)

    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    today = now.normalize()
    feats = feats[feats.index < today]  # drop the still-forming current session
    if len(feats) == 0:
        raise RuntimeError("No completed H1 trading day available for inference.")

    as_of = feats.index[-1]
    flat_row = feats.iloc[[-1]]
    seq = build_lstm_tensor(h1, feats.index[[-1]])
    return flat_row, seq, as_of
