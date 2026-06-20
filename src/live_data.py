import pandas as pd
import yfinance as yf


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
