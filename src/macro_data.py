import os

import pandas as pd

DEFAULT_FRED_SERIES = {"us10y": "DGS10", "de10y": "IRLTLT01DEM156N"}
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'yield_differential.csv'
)


def _combine(us: pd.Series, de: pd.Series) -> pd.DataFrame:
    """
    Align the two FRED series to a daily UTC index and take their spread.
    The German series (IRLTLT01DEM156N) is monthly, so it is forward-filled
    onto the US series' daily index -- this is the same "carry the last known
    value forward, never backward" rule used for bond-market holidays, and it
    never leaks a future German print into a past US date.
    """
    combined = pd.concat([us.rename('us10y'), de.rename('de10y')], axis=1)
    combined.index = pd.to_datetime(combined.index)
    combined.index = (
        combined.index.tz_localize('UTC') if combined.index.tz is None
        else combined.index.tz_convert('UTC')
    )
    combined = combined.sort_index().ffill()
    combined.index.name = 'DATE'
    combined['yield_differential'] = combined['us10y'] - combined['de10y']
    # Raw us10y/de10y are kept alongside yield_differential for display/
    # diagnostic purposes (e.g. the notebook's Section 2C charts) --
    # merge_macro_features() below only ever selects yield_differential, so
    # production training/inference is unaffected by these extra columns.
    return combined[['us10y', 'de10y', 'yield_differential']].dropna()


def _fetch_via_fredapi(series_ids: dict, start, end):
    """Official FRED API client. Requires FRED_API_KEY; returns None if unset
    (including the literal .env.example placeholder) or on any fetch error."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        us = fred.get_series(series_ids['us10y'], start, end)
        de = fred.get_series(series_ids['de10y'], start, end)
    except Exception:
        return None
    return _combine(us, de)


def _fetch_via_pandas_datareader(series_ids: dict, start, end):
    """FRED's public CSV download endpoint -- no API key required. Used as
    the fallback tier when no key is configured or the official API fails."""
    try:
        from pandas_datareader import data as pdr
        us = pdr.DataReader(series_ids['us10y'], 'fred', start, end)[series_ids['us10y']]
        de = pdr.DataReader(series_ids['de10y'], 'fred', start, end)[series_ids['de10y']]
    except Exception:
        return None
    return _combine(us, de)


def fetch_yield_differential(start, end, series_ids: dict = None, cache_path: str = DEFAULT_CACHE_PATH):
    """
    US 10Y - DE 10Y government bond yield differential, daily, UTC-indexed.

    Fallback chain (mirrors src/live_data.py's MT5 -> yfinance pattern):
    official FRED API (if FRED_API_KEY set) -> FRED public CSV endpoint (no
    key needed) -> last cached snapshot on disk. Returns (dataframe, source)
    where source is one of "FRED_api", "FRED_public", "cache", or
    (None, None) if nothing is reachable and no cache exists yet.
    """
    series_ids = series_ids or DEFAULT_FRED_SERIES

    df = _fetch_via_fredapi(series_ids, start, end)
    source = "FRED_api"
    if df is None or df.empty:
        df = _fetch_via_pandas_datareader(series_ids, start, end)
        source = "FRED_public"

    if df is not None and not df.empty:
        # A live fetch only covers [start, end] of the *current* price history,
        # which can be much narrower than what's already cached (e.g. a short
        # MT5 backfill window). Merge onto the existing cache instead of
        # overwriting it outright, or older history gets silently destroyed.
        if cache_path and os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                cached.index.name = 'DATE'
                cached.index = (
                    cached.index.tz_localize('UTC') if cached.index.tz is None
                    else cached.index.tz_convert('UTC')
                )
                df = pd.concat([cached, df]).sort_index()
                df = df[~df.index.duplicated(keep='last')]
            except (OSError, ValueError):
                pass
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
        except OSError:
            pass
        return df, source

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = (
            cached.index.tz_localize('UTC') if cached.index.tz is None else cached.index.tz_convert('UTC')
        )
        return cached, "cache"

    return None, None
