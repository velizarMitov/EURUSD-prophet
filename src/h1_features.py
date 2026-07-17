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

SMA_TREND_PERIOD = 504  # 504 H1 bars ~= 21 trading days: a slow trend baseline
RSI_PERIOD = 24         # one trading day of hourly momentum

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
    "Trend_vs_SMA504",       # day-end close / 504h SMA - 1 (position vs slow trend)
    "RSI_24",                # day-end 24-period RSI (intraday momentum, 0-100)
]

# Per-hour features fed to the LSTM at each of the 24 timesteps.
SEQ_FEATURE_COLUMNS = ["log_return", "hl_range", "co_change", "volume", "rsi_24"]

DEFAULT_H1_CACHE = "results/eurusd_h1.csv"


def _normalize_h1(df: pd.DataFrame) -> pd.DataFrame:
    """UTC-localise, sort, and reduce to the canonical OHLCV float columns."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    df = df.sort_index()
    return df[['open', 'high', 'low', 'close', 'tick_volume']].astype(float)


def _to_utc(now=None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    return now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')


def load_h1_frame(cache_path: str = DEFAULT_H1_CACHE, allow_fetch: bool = True) -> pd.DataFrame:
    """
    Load H1 OHLCV as a UTC-indexed frame. Reads the cache CSV if present;
    otherwise (``allow_fetch``) pulls a fresh copy via ``src.live_data``.

    Cache-first: suitable for TRAINING, where ``_train_pipeline.py`` explicitly
    refreshes the cache beforehand. Live inference must go through
    ``refresh_h1_frame`` instead — reading the cache blindly here is exactly
    what froze the served H1 'as of' date at the last retrain.
    """
    df = None
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    elif allow_fetch:
        from src.live_data import fetch_h1_market_data
        df, _ = fetch_h1_market_data(cache_path=cache_path)

    if df is None or len(df) == 0:
        raise RuntimeError("No H1 data available (cache missing and live fetch failed).")

    return _normalize_h1(df)


def last_complete_session(h1: pd.DataFrame, now: pd.Timestamp):
    """Latest UTC day strictly before ``now``'s date holding >= ``MIN_HOURS``
    bars — the same completeness rule ``aggregate_daily_features`` applies.
    ``None`` when no such day exists."""
    counts = h1.groupby(h1.index.normalize()).size()
    counts = counts[(counts.index < now.normalize()) & (counts >= MIN_HOURS)]
    return counts.index[-1] if len(counts) else None


def expected_latest_session(now: pd.Timestamp) -> pd.Timestamp:
    """Most recent weekday (Mon–Fri) strictly before ``now``, at UTC midnight.

    FX prints no Saturday H1 bars and only a thin (< ``MIN_HOURS``) late-Sunday
    open, so this is the newest session an up-to-date H1 cache can contain
    COMPLETE. A market holiday can make this date unattainable — the gate then
    errs toward a redundant live fetch, never toward serving stale data.
    """
    day = now.normalize() - pd.Timedelta(days=1)
    while day.dayofweek >= 5:  # Sat=5 / Sun=6
        day -= pd.Timedelta(days=1)
    return day


def refresh_h1_frame(cache_path: str = DEFAULT_H1_CACHE, now=None):
    """
    Staleness-gated, live-first H1 load for INFERENCE.

    Gate first: if the cache's last COMPLETE session already is the expected
    latest FX session, the cache is served untouched — a dashboard load must
    not hit MT5/yfinance when there is nothing newer to gain. Only a genuinely
    behind (or missing/unreadable) cache triggers ``fetch_h1_market_data``
    (MT5 -> yfinance chain, which self-writes the cache).

    A live pull rich in new bars but thin in history would silently truncate
    the SMA504/RSI trailing warm-ups (the H1 analogue of the old daily SMA_200
    bug), so cached rows absent from the live frame are concatenated back on
    (dedup by index, live wins) and the merged frame is rewritten to the cache.
    A fully failed live chain falls back to the stale cache — degraded freshness
    beats no H1 block at all.

    Returns ``(frame, source)`` with source in ``{"cache", "live",
    "live+history_backfill"}``.
    """
    now = _to_utc(now)

    cached = None
    if cache_path and os.path.exists(cache_path):
        try:
            cached = _normalize_h1(pd.read_csv(cache_path, index_col=0, parse_dates=True))
        except Exception:
            cached = None  # unreadable cache -> treat as absent, go live
    if cached is not None and len(cached) > 0:
        last = last_complete_session(cached, now)
        if last is not None and last >= expected_latest_session(now):
            return cached, "cache"

    from src.live_data import fetch_h1_market_data
    live, live_source = fetch_h1_market_data(cache_path=cache_path)

    if live is None or len(live) == 0 or live_source == "cache":
        # Live chain exhausted (fetch may itself have fallen back to the same
        # cache file). Serve the best copy we have rather than hard-failing.
        fallback = live if live is not None and len(live) > 0 else cached
        if fallback is None:
            raise RuntimeError("No H1 data available (live fetch failed and no cache).")
        return _normalize_h1(fallback), "cache"

    live = _normalize_h1(live)
    if cached is not None and len(cached) > 0:
        merged = pd.concat([cached, live])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        if len(merged) > len(live):
            # fetch_h1_market_data already wrote the live-only frame to the
            # cache; persist the merged one so the warm-up history survives
            # for the next reader.
            try:
                merged.to_csv(cache_path)
            except OSError:
                pass
            return merged, "live+history_backfill"
    return live, "live"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Classic RSI over a trailing ``period`` window (past bars only -> no
    look-ahead). Pure uptrend (no losses) -> 100; a flat/empty window -> a
    neutral 50 rather than NaN."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0.0).rolling(period).mean()
    avg_loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)   # avg_loss==0 -> +inf -> 100
    return rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def _enrich_hourly(h1: pd.DataFrame) -> pd.DataFrame:
    """Add the per-hour derived columns used by both representations and a
    ``date`` bucket (UTC midnight) for chronological grouping. Trend/RSI are
    computed on the *continuous* hourly stream with trailing windows, so they
    carry cross-day context without leaking the future."""
    h1 = h1.copy()
    h1['log_return'] = np.log(h1['close'] / h1['close'].shift(1)).fillna(0.0)
    h1['hl_range'] = h1['high'] - h1['low']
    h1['co_change'] = h1['close'] - h1['open']
    h1['volume'] = h1['tick_volume']

    sma504 = h1['close'].rolling(SMA_TREND_PERIOD).mean()
    h1['trend_vs_sma504'] = (h1['close'] / sma504 - 1.0).fillna(0.0)  # 0 = at/undefined trend
    h1['rsi_24'] = _rsi(h1['close'], RSI_PERIOD).fillna(50.0)

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
        "Trend_vs_SMA504":     g['trend_vs_sma504'].last(),   # day-end trend position
        "RSI_24":              g['rsi_24'].last(),            # day-end momentum
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
    otherwise it is loaded live-first through ``refresh_h1_frame`` (staleness
    gate + MT5/yfinance chain + history backfill), so the served day tracks the
    market instead of freezing at the last retrain's cache write.

    Returns ``(flat_row_df, seq_tensor, as_of_date, data_source)`` where
    ``flat_row_df`` is a single-row DataFrame in ``FLAT_FEATURE_COLUMNS`` order,
    ``seq_tensor`` has shape ``(1, HOURS_PER_DAY, len(SEQ_FEATURE_COLUMNS))``,
    and ``data_source`` is ``refresh_h1_frame``'s label (``"preloaded"`` when
    ``h1`` was passed in directly).
    """
    now = _to_utc(now)
    if h1 is None:
        h1, data_source = refresh_h1_frame(cache_path, now=now)
    else:
        data_source = "preloaded"
    feats, _ = aggregate_daily_features(h1)

    today = now.normalize()
    feats = feats[feats.index < today]  # drop the still-forming current session
    if len(feats) == 0:
        raise RuntimeError("No completed H1 trading day available for inference.")

    as_of = feats.index[-1]
    flat_row = feats.iloc[[-1]]
    seq = build_lstm_tensor(h1, feats.index[[-1]])
    return flat_row, seq, as_of, data_source
