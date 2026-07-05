import os

import pandas as pd
import yfinance as yf


def drop_incomplete_bars(ohlcv_df, now=None):
    """
    Keep only fully-closed weekday sessions. Shared by PredictionService
    (so a live forecast never bases itself on a half-formed bar) and
    tracking.build_history_html (so the realised "actual market" close used
    to score a forecast is never read off today's still-moving intraday
    price). Two kinds of bars are stripped from a live feed:

      * The current day's bar while that session is still forming. A fetch
        at 11:00 on 23.04 exposes a 23.04 bar whose high/low/close are not
        final yet -- using it (whether as the prediction base or as the
        "actual" result) is wrong until the session has actually closed.
      * Any Saturday/Sunday bar. MT5 brokers emit a short partial Sunday
        (weekend) bar that the yfinance-derived training data never
        contained, so it is out-of-distribution.

    `now` is injectable for tests; it defaults to the local wall clock.
    Comparison is on the calendar date (the broker bar timestamps and the
    local clock are both treated as tz-naive, which is correct for a
    single-machine Windows/MT5 deployment).
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return ohlcv_df
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    today = now.normalize()
    idx = ohlcv_df.index
    keep = (idx.normalize() < today) & (idx.weekday < 5)
    return ohlcv_df[keep]


def _fetch_from_mt5(symbol: str, bars: int):
    """Try a live MT5 terminal session first. Returns an OHLCV DataFrame
    (tz-naive, ascending date index) or None if no terminal is reachable."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    try:
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, bars)
        mt5.shutdown()
    except Exception:
        return None

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]


def _fetch_from_yfinance(symbol: str, bars: int):
    """Fall back to Yahoo Finance. Returns an OHLCV DataFrame (tz-naive,
    ascending date index) or None if the fetch fails for any reason."""
    try:
        # Daily FX trades ~5 days/week; pad the requested period so `bars`
        # trading days are comfortably covered after weekends/holidays.
        period_days = int(bars * 1.6) + 30
        history = yf.Ticker(symbol).history(period=f"{period_days}d", interval="1d")
    except Exception:
        return None

    if history is None or history.empty:
        return None

    history = history.tail(bars)
    index = history.index.tz_localize(None) if history.index.tz is not None else history.index
    volume = history['Volume'].astype(float) if 'Volume' in history.columns else 0.0

    return pd.DataFrame({
        'open': history['Open'].astype(float).values,
        'high': history['High'].astype(float).values,
        'low': history['Low'].astype(float).values,
        'close': history['Close'].astype(float).values,
        'tick_volume': volume.values if hasattr(volume, 'values') else volume,
    }, index=index)


def _fetch_h1_from_mt5(symbol: str, bars: int):
    """H1 (1-hour) OHLCV from a live MT5 terminal. Returns a UTC-indexed
    DataFrame (ascending) or None if no terminal is reachable. Mirrors
    _fetch_from_mt5 but requests mt5.TIMEFRAME_H1 instead of D1."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    try:
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
        mt5.shutdown()
    except Exception:
        return None

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    # MT5 broker timestamps are seconds since epoch; localise straight to UTC so
    # the H1->Daily predictor always groups on a single, unambiguous timezone.
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]


def _fetch_h1_from_yfinance(symbol: str, days: int = 730):
    """H1 (1-hour) OHLCV from Yahoo Finance. Yahoo caps hourly history at
    ~730 calendar days. Returns a UTC-indexed DataFrame or None on failure."""
    try:
        history = yf.Ticker(symbol).history(period=f"{min(days, 730)}d", interval="1h")
    except Exception:
        return None

    if history is None or history.empty:
        return None

    idx = history.index
    idx = idx.tz_convert('UTC') if idx.tz is not None else idx.tz_localize('UTC')
    volume = history['Volume'].astype(float) if 'Volume' in history.columns else 0.0

    return pd.DataFrame({
        'open': history['Open'].astype(float).values,
        'high': history['High'].astype(float).values,
        'low': history['Low'].astype(float).values,
        'close': history['Close'].astype(float).values,
        'tick_volume': volume.values if hasattr(volume, 'values') else volume,
    }, index=idx)


def fetch_h1_market_data(mt5_symbol: str = "EURUSD", yf_symbol: str = "EURUSD=X",
                         bars: int = 60000, cache_path: str = "results/eurusd_h1.csv"):
    """
    Fetch historical H1 (1-hour) EURUSD OHLCV for the auxiliary H1->Daily
    predictor. Fallback chain mirrors fetch_live_market_data(): a live MT5
    terminal (TIMEFRAME_H1, deep history) -> Yahoo Finance interval='1h'
    (~730 days) -> the on-disk cache CSV.

    The returned DataFrame always carries a **UTC-localised** DateTimeIndex, so
    downstream daily bucketing is timezone-unambiguous. Returns
    (dataframe, source_label) where source is "MT5", "yfinance", "cache", or
    (None, None) if nothing is reachable and no cache exists.
    """
    df = _fetch_h1_from_mt5(mt5_symbol, bars)
    source = "MT5"
    if df is None or len(df) == 0:
        df = _fetch_h1_from_yfinance(yf_symbol, days=730)
        source = "yfinance"

    if df is not None and len(df) > 0:
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path)
            except OSError:
                pass
        return df, source

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = (
            cached.index.tz_localize('UTC') if cached.index.tz is None
            else cached.index.tz_convert('UTC')
        )
        return cached, "cache"

    return None, None


def fetch_live_market_data(mt5_symbol: str = "EURUSD", yf_symbol: str = "EURUSD=X", bars: int = 250):
    """
    Fetch the latest `bars` daily OHLCV bars needed to satisfy the largest
    rolling window (SMA_200) and lag requirements, knowing "today" implicitly
    as whatever the live source's most recent closed/forming bar is.

    Tries a live MT5 terminal session first (if one is installed and logged
    in), then Yahoo Finance. Returns (dataframe, source_label); source_label
    is one of "MT5", "yfinance", or (dataframe=None) if neither is reachable,
    in which case the caller is expected to fall back to its own bundled
    historical data.
    """
    df = _fetch_from_mt5(mt5_symbol, bars)
    if df is not None and len(df) > 0:
        return df, "MT5"

    df = _fetch_from_yfinance(yf_symbol, bars)
    if df is not None and len(df) > 0:
        return df, "yfinance"

    return None, None


def fetch_latest_bar(symbol: str = "EURUSD=X") -> dict | None:
    """
    Fetch just the single most recent daily OHLCV bar from Yahoo Finance.
    Returns a dict with date/open/high/low/close/tick_volume, or None on
    failure. Retained as a lightweight helper for callers that only need the
    newest bar rather than a full rolling-window history (use
    fetch_live_market_data() for that).
    """
    try:
        history = yf.Ticker(symbol).history(period="5d", interval="1d")
        if history is None or history.empty:
            return None

        last = history.iloc[-1]
        volume = float(last["Volume"]) if "Volume" in history.columns and pd.notna(last["Volume"]) else 0.0

        return {
            "date": history.index[-1].date().isoformat(),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "tick_volume": volume,
        }
    except Exception:
        return None
