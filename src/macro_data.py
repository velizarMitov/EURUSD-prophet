"""
FRED macro-feature ingestion for the EURUSD model.

Every macro feature shares ONE fallback chain -- official FRED API (needs
FRED_API_KEY) -> FRED public CSV endpoint (no key) -> last cached snapshot on
disk -- written once in `fetch_fred_feature` and reused by all four features
(yield differential + the three added in the macro-expansion backlog). Each
feature has its OWN cache file so a thin live fetch of one never overwrites
another's longer cached history (the merge-not-overwrite cache fix, §4.4 of
ARCHITECTURE_DOCS.md, generalized).

The fetchers only ever pull RAW LEVEL series; the per-feature `combine` callable
turns those levels into the derived column(s). Non-pointwise transforms that need
history beyond the requested window (CPI year-over-year needs 12 prior months)
pass `extra_lookback_days` so the fetch start is pushed back far enough that the
transform is defined across the whole requested [start, end] -- otherwise the
feature would be NaN at the live edge (a real bug now that the model trades).
"""
import os

import numpy as np
import pandas as pd

# Yield differential keeps its historical defaults + cache path for backward
# compatibility with existing callers/imports of fetch_yield_differential.
DEFAULT_FRED_SERIES = {"us10y": "DGS10", "de10y": "IRLTLT01DEM156N"}
DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'yield_differential.csv'
)


def _utc(df):
    """Coerce a frame/series index to a sorted UTC DatetimeIndex named DATE."""
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    df = df.sort_index()
    df.index.name = 'DATE'
    return df


def _assemble(cols: dict):
    """Stack the fetched raw level series (keyed by the caller's names) into one
    UTC-indexed, forward-filled frame. Returns None if any series is missing or
    empty so the caller falls through to the next tier of the chain."""
    if any(v is None or len(v) == 0 for v in cols.values()):
        return None
    frame = pd.concat({k: pd.Series(v) for k, v in cols.items()}, axis=1)
    # ffill carries a past monthly/holiday value forward onto later (e.g. daily)
    # dates -- never a future value backward, so no look-ahead is introduced.
    frame = _utc(frame).ffill()
    return frame


def _fetch_raw_via_api(series_ids: dict, start, end):
    """Official FRED API client -> raw level frame keyed by `series_ids` KEYS,
    or None if no usable key is set or any fetch fails."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        cols = {name: fred.get_series(sid, start, end) for name, sid in series_ids.items()}
    except Exception:
        return None
    return _assemble(cols)


def _fetch_raw_via_public(series_ids: dict, start, end):
    """FRED's public CSV endpoint (no API key) -> raw level frame, or None."""
    try:
        from pandas_datareader import data as pdr
        cols = {name: pdr.DataReader(sid, 'fred', start, end)[sid] for name, sid in series_ids.items()}
    except Exception:
        return None
    return _assemble(cols)


def _read_cache(cache_path):
    """Load a cached (already-combined) feature frame as UTC, or None."""
    if not (cache_path and os.path.exists(cache_path)):
        return None
    try:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        return None
    return _utc(cached)


def _merge_write_cache(df, cache_path):
    """Merge the freshly combined frame onto the existing cache (keep='last')
    and write it back, so a narrow live fetch never truncates longer history."""
    cached = _read_cache(cache_path)
    if cached is not None:
        df = pd.concat([cached, df]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
    try:
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
    except OSError:
        pass
    return df


def fetch_fred_feature(series_ids: dict, start, end, combine, cache_path,
                       extra_lookback_days: int = 0):
    """
    Generic FRED feature fetch with the shared API -> public -> cache fallback
    chain. Pulls the raw level series, applies `combine(raw_levels) -> frame`,
    and merge-writes the combined frame to `cache_path`. `extra_lookback_days`
    pushes the fetch start back (for shift-based transforms like YoY) so the
    derived column is defined across the whole requested window. Returns
    (combined_frame, source) where source is FRED_api / FRED_public / cache, or
    (None, None) if nothing is reachable and no cache exists.
    """
    fetch_start = start
    if extra_lookback_days:
        fetch_start = pd.Timestamp(start) - pd.Timedelta(days=extra_lookback_days)

    raw = _fetch_raw_via_api(series_ids, fetch_start, end)
    source = "FRED_api"
    if raw is None or raw.empty:
        raw = _fetch_raw_via_public(series_ids, fetch_start, end)
        source = "FRED_public"

    if raw is not None and not raw.empty:
        combined = combine(raw)
        combined = _merge_write_cache(combined, cache_path)
        return combined, source

    cached = _read_cache(cache_path)
    if cached is not None:
        return cached, "cache"
    return None, None


# ── Per-feature combine functions (levels -> derived columns) ───────────────

def _combine_yield(raw):
    """US 10Y - DE 10Y bond-yield differential (raw levels kept for display)."""
    out = raw.copy()
    out['yield_differential'] = out['us10y'] - out['de10y']
    return out[['us10y', 'de10y', 'yield_differential']].dropna()


def _combine_usd_index(raw):
    """Trade-weighted USD index LEVEL (DTWEXBGS). The stationary transform
    (log-return) is derived downstream in src/features.py::compute_features from
    this merged level -- computing it there, not here, keeps the return correct
    across cache boundaries (it is never spliced from two fetch windows)."""
    return raw[['usd_index']].dropna()


def _combine_policy(raw):
    """US policy rate (DFF, effective fed funds) - ECB deposit facility rate
    (ECBDFR). ECBDFR can be negative; the subtraction handles it. Raw legs kept
    for display/diagnostics."""
    out = raw.copy()
    out['policy_rate_differential'] = out['fed'] - out['ecb']
    return out[['fed', 'ecb', 'policy_rate_differential']].dropna()


def _combine_vix(raw):
    """CBOE Volatility Index close LEVEL (VIXCLS). Only the raw level is kept
    here -- exactly like _combine_usd_index -- because the model-facing stationary
    transforms (a trailing rolling z-score to remove VIX's multi-year regime
    drift, plus a day-over-day % change) are derived DOWNSTREAM in
    src/vix_features.py on the native business-day cadence. Deriving them there,
    not here, keeps the z-score window from being corrupted by the ffilled
    weekend/holiday duplicate bars in the FX daily history, and keeps the change
    correct across cache-fetch boundaries. VIXCLS begins 1990, so it covers the
    whole euro era with no leading-NaN truncation."""
    return raw[['vix']].dropna()


def _combine_inflation(raw):
    """US CPI YoY% (CPIAUCSL) - Germany HICP YoY% (CP0000DEM086NEST). Each CPI
    is a monthly index; YoY = (level_t / level_{t-12} - 1) * 100. Series are
    resampled to month-start first so the 12-period shift is exactly 12 months
    regardless of how the two calendars align on the outer join. The first 12
    months (no YoY yet) are dropped -- fetch_fred_feature's extra_lookback_days
    guarantees enough prior history that this drop never eats into the caller's
    requested window."""
    us = raw['us_cpi'].resample('MS').last()
    de = raw['de_cpi'].resample('MS').last()
    out = pd.DataFrame(index=us.index.union(de.index))
    out = _utc(out)
    out['us_cpi_yoy'] = ((us / us.shift(12) - 1.0) * 100.0).reindex(out.index)
    out['de_cpi_yoy'] = ((de / de.shift(12) - 1.0) * 100.0).reindex(out.index)
    out['inflation_differential'] = out['us_cpi_yoy'] - out['de_cpi_yoy']
    return out.dropna(subset=['inflation_differential'])


def fetch_yield_differential(start, end, series_ids: dict | None = None, cache_path: str = DEFAULT_CACHE_PATH):
    """
    US 10Y - DE 10Y government bond yield differential, daily, UTC-indexed.
    Thin backward-compatible wrapper over fetch_fred_feature (unchanged public
    contract: returns (dataframe_with_yield_differential, source)).
    """
    return fetch_fred_feature(
        series_ids or DEFAULT_FRED_SERIES, start, end,
        combine=_combine_yield, cache_path=cache_path,
    )


# Registry mapping each config `macro.features` key to its (combine fn, the
# model-merge column it contributes, default YoY lookback). The transform LOGIC
# lives here in code (you cannot express a YoY in JSON); config supplies only
# the series IDs and cache paths.
_FEATURE_COMBINERS = {
    "usd_index":   (_combine_usd_index, "usd_index", 0),
    "policy_rate": (_combine_policy, "policy_rate_differential", 0),
    "inflation":   (_combine_inflation, "inflation_differential", 420),
    # VIX is fetched through this shared framework so results/vix.csv stays fresh
    # via the same merge-not-overwrite cache discipline, but its merge column is
    # DELIBERATELY absent from features.MACRO_MERGE_COLUMNS: the candidate VIX
    # features are ablation-only (src/vix_features.py, computed on native
    # cadence) and must not enter the served model frame until they clear the
    # Bonferroni bar. So fetch_macro_features maintains the cache; nothing merges
    # the level into the model input.
    "vix":         (_combine_vix, "vix", 0),
}


def fetch_macro_features(start, end, macro_cfg: dict, base_dir: str = ""):
    """
    Fetch ALL macro features and return (macro_df, sources).

    macro_df is one UTC-indexed frame carrying the model-merge columns
    (yield_differential, usd_index, policy_rate_differential,
    inflation_differential) outer-joined on date; downstream
    merge_macro_features/compute_features turn these into the final model
    features. `sources` maps each feature name to the tier that answered
    (FRED_api / FRED_public / cache / unavailable). Any feature that is entirely
    unreachable is simply omitted from macro_df -- merge_macro_features defaults
    a missing macro column to a neutral value, so the pipeline never hard-fails.
    """
    def _abs(p):
        return os.path.join(base_dir, p) if base_dir else p

    frames, sources = [], {}

    # 1. Yield differential (existing feature / cache).
    yld, s = fetch_yield_differential(
        start, end,
        series_ids=macro_cfg.get('fred_series'),
        cache_path=_abs(macro_cfg.get('cache_path', 'results/yield_differential.csv')),
    )
    sources['yield_differential'] = s if yld is not None else "unavailable"
    if yld is not None:
        frames.append(yld[['yield_differential']])

    # 2-4. The added features, driven by config `macro.features`.
    for key, spec in (macro_cfg.get('features') or {}).items():
        combine, merge_col, default_lookback = _FEATURE_COMBINERS[key]
        lookback = spec.get('yoy_lookback_days', default_lookback)
        df, src = fetch_fred_feature(
            spec['series'], start, end,
            combine=combine, cache_path=_abs(spec['cache_path']),
            extra_lookback_days=lookback,
        )
        sources[merge_col] = src if df is not None else "unavailable"
        if df is not None and merge_col in df.columns:
            frames.append(df[[merge_col]])

    if not frames:
        return None, sources
    macro_df = pd.concat(frames, axis=1).sort_index()
    return macro_df, sources
