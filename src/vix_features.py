"""
CBOE Volatility Index (VIX) regime features for the EURUSD direction/return model.

Genuinely different information from price, rates, or positioning: broad EQUITY
risk sentiment (the "fear gauge"). Two candidate features, bundled as ONE
hypothesis (same convention as the fibonacci / COT / FOMC blocks — one Bonferroni
slot for the whole bundle, not one each):

  * vix_zscore     -- trailing rolling z-score of the VIX level. VIX has genuine
                      multi-year REGIME drift (the 2008 and 2020 spikes dwarf
                      calm-era levels), so a raw level is non-stationary; the
                      z-score removes that drift — the same treatment COT
                      positioning gets, NOT the pass-through the already-stationary
                      yield/policy/inflation DIFFERENTIALS get.
  * vix_change_pct -- day-over-day % change in the VIX level (already stationary),
                      capturing the volatility SHOCK component distinctly from
                      the regime level.

The raw VIX LEVEL is ingested through the shared FRED framework
(`src/macro_data.py::fetch_fred_feature`, config `macro.features.vix`, cache
`results/vix.csv`) — no parallel fetch mechanism. The two transforms live HERE,
downstream, exactly as `usd_index_return` is derived downstream from the merged
`usd_index` level.

LOOK-AHEAD DISCIPLINE (verified in STEP 0, not assumed)
-------------------------------------------------------
The EURUSD D1 bar (src/live_data.py: MT5 TIMEFRAME_D1 / yfinance daily, tz-naive)
closes at the retail-FX broker rollover ~17:00 US/Eastern on day D. FRED's VIXCLS
is the CBOE equity-session close (~16:15 ET) — only a ~45-minute, DST-fragile
margin before that rollover. Worse, a live probe of FRED's public feed on Sunday
2026-07-26 returned its most recent VIXCLS print as 2026-07-23 (Thu): Friday
2026-07-24's print was still ABSENT two days later, confirming FRED publishes
VIXCLS with a business-day lag.

    DECISION (conservative, mirrors the COT "+buffer" and macro-ffill rule of
    erring toward LATER availability, never earlier): a VIX print dated D is
    treated as available only on **D + 1 business day**. So day D's EURUSD bar
    uses day D-1's VIX — a print can never leak into the very bar it might not
    have been public for. This is the convention verified and used.

Both transforms are computed on the NATIVE business-day VIX series (never on the
ffilled FX-daily series, whose weekend/holiday duplicate bars would corrupt the
756-day z-score window — the same native-cadence rule COT uses), then stamped at
the availability date and as-of ffilled onto the daily FX bars. A missing/None
VIX feed is neutralized to 0 downstream, exactly like an unreachable FRED/COT
feature: the pipeline degrades, never hard-fails, and the baseline variant is
immune by construction.

Ablation-first: nothing here is wired into FEATURE_COLUMNS. Tested once via
`src/ablation.py::run_addition_test` on the validation slice [70:80].
"""
import os

import numpy as np
import pandas as pd

from src.macro_data import fetch_fred_feature, _combine_vix

VIX_SERIES_IDS = {"vix": "VIXCLS"}
VIX_FEATURE_COLUMNS = ["vix_zscore", "vix_change_pct"]

# Trailing z-score window on the NATIVE (business-day) VIX series: ~3 years of
# trading days with at least ~1 year of history before a z-score is emitted
# (earlier rows -> neutral 0 downstream). Comparable to COT's 156-week / 3-year
# weekly window, expressed in daily bars.
VIX_ZSCORE_WINDOW = 756
VIX_ZSCORE_MIN = 252

# Conservative publish lag (STEP 0): a print dated D is available on D + this many
# BUSINESS days. 1 == "use yesterday's VIX for today's bar".
VIX_AVAILABILITY_LAG_BDAYS = 1

# VIXCLS begins 1990; pull the whole history so the trailing window is always
# consistent regardless of the caller's requested span.
VIX_HISTORY_START = "1990-01-01"

DEFAULT_VIX_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "vix.csv"
)


def _abs(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def _vix_spec(config, base_dir):
    """(series_ids, cache_path) from config.macro.features.vix, with defaults."""
    spec = (((config or {}).get("macro", {}) or {}).get("features", {}) or {}).get("vix", {})
    series = spec.get("series", VIX_SERIES_IDS)
    cache = _abs(base_dir, spec["cache_path"]) if spec.get("cache_path") else DEFAULT_VIX_CACHE
    return series, cache


def _utc_normalize(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.normalize()


# ── raw level fetch (reuses the shared FRED framework) ───────────────────────

def fetch_vix_level(config=None, base_dir="", start=VIX_HISTORY_START, end=None):
    """Fetch the raw VIX LEVEL via the shared `fetch_fred_feature` chain
    (FRED API -> FRED public CSV -> on-disk cache -> None). Returns
    (level_frame_with_'vix'_column, source) or (None, None) if nothing is
    reachable and no cache exists."""
    series, cache = _vix_spec(config, base_dir)
    end = pd.Timestamp.utcnow().normalize() if end is None else end
    frame, source = fetch_fred_feature(series, start, end, combine=_combine_vix,
                                       cache_path=cache)
    return frame, source


# ── native-cadence transforms + availability stamping ────────────────────────

def _compute_vix_frame(level_frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a native business-day VIX LEVEL frame into an AVAILABILITY-date-indexed
    frame of the VIX_FEATURE_COLUMNS.

    The z-score and %-change are computed on the NATIVE series (a trailing window
    ending AT each business day -> uses only data public by that day, no
    look-ahead), then the index is pushed forward by VIX_AVAILABILITY_LAG_BDAYS
    business days so an as-of ffill onto daily bars is look-ahead-safe by
    construction (a print dated D only ever reaches bars on/after D+1 business
    day)."""
    lvl = level_frame["vix"].copy()
    lvl.index = _utc_normalize(lvl.index)
    lvl = lvl[~lvl.index.duplicated(keep="last")].sort_index()

    roll = lvl.rolling(VIX_ZSCORE_WINDOW, min_periods=VIX_ZSCORE_MIN)
    z = (lvl - roll.mean()) / roll.std()
    z = z.replace([np.inf, -np.inf], np.nan)      # degenerate flat window -> neutral (0 downstream)
    change = lvl.pct_change() * 100.0

    avail = lvl.index + pd.tseries.offsets.BusinessDay(VIX_AVAILABILITY_LAG_BDAYS)
    out = pd.DataFrame({"vix_zscore": z.to_numpy(), "vix_change_pct": change.to_numpy()},
                       index=_utc_normalize(avail))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_vix_features(config=None, base_dir=""):
    """Fetch VIX and return (vix_frame, source). `vix_frame` is
    availability-date-indexed (UTC) with the VIX_FEATURE_COLUMNS, or None if the
    feed is entirely unreachable (no API/public/cache)."""
    level, source = fetch_vix_level(config=config, base_dir=base_dir)
    if level is None or level.empty or "vix" not in level.columns:
        return None, "unavailable"
    return _compute_vix_frame(level), source


# ── daily join (mirrors merge_macro_features / add_cot_features as-of ffill) ──

def add_vix_features(df: pd.DataFrame, vix_frame=None, base_dir="", config=None) -> pd.DataFrame:
    """Join vix_zscore / vix_change_pct onto a daily-indexed frame by AVAILABILITY
    date (as-of ffill), neutral 0 where unavailable (z-score warm-up, before the
    series exists, or an entirely missing feed).

    Look-ahead-safe: `vix_frame` is indexed by the date each reading became usable
    (print date + 1 business day, STEP 0), and ffill only ever carries a PAST
    reading forward onto later bars — identical discipline to add_cot_features /
    merge_macro_features. When `vix_frame` is None it is fetched (offline-safe via
    cache); tests inject a synthetic frame."""
    if vix_frame is None:
        vix_frame, _src = fetch_vix_features(config=config, base_dir=base_dir)

    out = df.copy()
    idx = pd.DatetimeIndex(df.index)
    naive = (idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx).normalize()

    for col in VIX_FEATURE_COLUMNS:
        if vix_frame is not None and col in vix_frame.columns:
            s = vix_frame[col].dropna()
            if len(s):
                si = pd.DatetimeIndex(s.index)
                si = (si.tz_convert("UTC").tz_localize(None) if si.tz is not None else si).normalize()
                s = pd.Series(s.to_numpy(), index=si).sort_index()
                s = s[~s.index.duplicated(keep="last")]
                aligned = s.reindex(s.index.union(naive)).ffill().reindex(naive)
                out[col] = np.where(np.isnan(aligned.to_numpy()), 0.0, aligned.to_numpy())
                continue
        out[col] = 0.0
    return out


if __name__ == "__main__":
    frame, src = fetch_vix_features()
    if frame is None:
        print("VIX unavailable (no API, no public, no cache).")
    else:
        print(f"source={src}  rows={len(frame)}  "
              f"availability {frame.index.min().date()} -> {frame.index.max().date()}")
        print(frame.tail(6).to_string())
