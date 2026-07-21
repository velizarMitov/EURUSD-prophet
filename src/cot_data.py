"""
CFTC Commitment-of-Traders (COT) positioning features for the EURUSD model.

Genuinely different information from price or FRED macro data: weekly NET
SPECULATIVE POSITIONING (sentiment / flow of leveraged money), from the CFTC
"Traders in Financial Futures" (TFF) Futures-Only report. Two instruments ->
two candidate features:

  * cot_eur_zscore       -- leveraged-funds net position (long - short) in
                            EURO FX futures, z-scored over a trailing 3-year
                            window.
  * cot_usdindex_zscore  -- same for ICE's USD INDEX (DX) futures.

The two markets are ONE weekly report from ONE endpoint (unlike the FRED
features, which are independent series with their own histories), so they share
a single cache file and are always fetched together.

Fallback chain (mirrors src/macro_data.py's API -> ... -> cache -> None):
    CFTC Socrata JSON API  ->  on-disk cache  ->  None
There is no separate "public CSV" tier because the Socrata JSON endpoint IS the
free, key-less public source (it already plays the role FRED's public-CSV tier
plays for macro_data). A missing/None COT feed is neutralized to a z-score of 0
downstream, exactly like an unreachable FRED feature -- the pipeline degrades,
never hard-fails, and the baseline variant is immune by construction.

LOOK-AHEAD DISCIPLINE (a positioning feature is easy to leak):
    CFTC reports a Tuesday "as of" business date but PUBLISHES the following
    Friday -- a structural ~3-day lag -- and holidays / government shutdowns
    delay that release irregularly, by up to weeks. So the daily join must use
    the date the data was actually AVAILABLE, never the "as of" date.

    Socrata exposes a TRUE system publish timestamp, `:created_at`, but only for
    rows inserted after the dataset's 2022-09-13 bulk reload; every earlier row
    carries that single reload timestamp instead of its original publish date
    (verified against the live API: EURO FX rows 2006..2022 all share
    created_at = 2022-09-13T14:16:09Z). Availability is therefore HYBRID:

      * trust `:created_at` when its lag over the as_of date is small/plausible
        (recent rows) -- this handles real holiday/shutdown delays exactly
        (e.g. as_of 2026-06-30 was not published until 2026-07-06, a 6.8-day
        lag, and created_at captures that);
      * otherwise fall back to a CONSERVATIVE `as_of + PUBLISH_BUFFER_DAYS`
        (deep history) -- deliberately erring toward LATER availability, never
        earlier. A fixed +3 offset is NOT used: it would leak during exactly
        the shutdown/holiday periods this buffer exists to cover.

    Residual limitation (documented honestly, not hidden): for the pre-2022
    bulk-loaded rows the true original publish date is unknown, so a
    multi-week government-shutdown delay in that deep history (e.g. Oct 2013)
    could exceed the conservative buffer and be treated as available a few days
    early. This affects only a handful of deep-train rows via ffill; all live /
    forward data -- the actual trading risk -- uses the true `:created_at`.

The z-score is computed on the NATIVE WEEKLY cadence (a trailing-window mean/std
over real weeks, never over ffilled daily duplicates, which would corrupt the
window and the stationarity), then stamped at each reading's availability date.
`add_cot_features` ffills those availability-dated z-scores onto the daily bars,
so a bar can only ever see a COT reading that was already public by that date --
the exact as-of ffill discipline merge_macro_features uses for the monthly FRED
series. Raw contract counts are deliberately NOT used as features: open interest
and participation have grown structurally over two decades, so raw levels are
non-stationary; the trailing z-score removes that drift.
"""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Official CFTC Socrata dataset: "Traders in Financial Futures (TFF) - Futures
# Only". Verified to carry both EURO FX and USD INDEX (ICE DX) leveraged-funds
# positioning, 2006-06 -> present, ~1049 weekly reports each.
COT_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"

# Feature key -> exact CFTC `contract_market_name`. The key also names the
# derived column: `<key>_zscore`.
COT_MARKETS = {"cot_eur": "EURO FX", "cot_usdindex": "USD INDEX"}
COT_FEATURE_COLUMNS = ["cot_eur_zscore", "cot_usdindex_zscore"]

# Availability-date (look-ahead) knobs.
PUBLISH_BUFFER_DAYS = 10           # conservative buffer when :created_at is untrustworthy
CREATED_AT_TRUST_MAX_LAG_DAYS = 30  # a created_at lag beyond this is the 2022 bulk-reload artifact
# Trailing z-score window (weekly cadence): 3-year lookback, at least 1 year of
# history before a z-score is emitted (earlier rows -> neutral 0 downstream).
ZSCORE_WINDOW_WEEKS = 156
ZSCORE_MIN_WEEKS = 52

DEFAULT_COT_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cot_positioning.csv"
)


def _abs(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def _utc_normalize(idx) -> pd.DatetimeIndex:
    """Coerce to a tz-aware (UTC), date-normalized DatetimeIndex."""
    idx = pd.DatetimeIndex(idx)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.normalize()


# ── availability-date logic (the look-ahead heart of this module) ────────────

def availability_date(as_of, created_at) -> pd.Timestamp:
    """Return the UTC date a weekly report became publicly usable.

    Trusts the true Socrata publish timestamp `created_at` when its lag over the
    `as_of` (Tuesday) business date is a plausible real publish lag
    (0..CREATED_AT_TRUST_MAX_LAG_DAYS) -- pushing one extra day past the publish
    instant so an evening release never informs an earlier-close bar the same
    day. Otherwise (missing timestamp, or the 2022 bulk-reload artifact whose
    lag is years) falls back to the conservative `as_of + PUBLISH_BUFFER_DAYS`.
    Never returns a date earlier than the real availability.
    """
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    if created_at is not None and pd.notna(created_at):
        created_at = pd.Timestamp(created_at)
        created_at = created_at.tz_localize("UTC") if created_at.tzinfo is None \
            else created_at.tz_convert("UTC")
        lag_days = (created_at.normalize() - as_of.normalize()).days
        if 0 <= lag_days <= CREATED_AT_TRUST_MAX_LAG_DAYS:
            return created_at.normalize() + pd.Timedelta(days=1)
    return as_of.normalize() + pd.Timedelta(days=PUBLISH_BUFFER_DAYS)


# ── raw fetch (CFTC Socrata) ─────────────────────────────────────────────────

def _fetch_market_rows(market_name: str, url: str = COT_SOCRATA_URL):
    """Fetch all weekly rows for one contract_market_name, or None on failure.
    Selects only the leveraged-funds long/short legs, the as_of business date,
    and the system publish timestamp :created_at. Retries transient errors."""
    params = {
        "$select": (
            "report_date_as_yyyy_mm_dd,"
            "lev_money_positions_long,"
            "lev_money_positions_short,"
            ":created_at"
        ),
        "$where": f"contract_market_name = '{market_name}'",
        "$order": "report_date_as_yyyy_mm_dd asc",
        "$limit": "50000",
    }
    full = f"{url}?{urlencode(params)}"
    for attempt in range(3):
        try:
            with urlopen(full, timeout=40) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def _fetch_all_markets_via_api(url: str = COT_SOCRATA_URL, markets: dict = None):
    """Fetch every configured market and return ONE raw frame indexed by as_of
    date with columns `<key>_net` per market plus a shared `created_at`, or None
    if any market is unreachable (so the caller falls through to the cache)."""
    markets = markets or COT_MARKETS
    per = {}
    for key, market in markets.items():
        rows = _fetch_market_rows(market, url)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        as_of = _utc_normalize(pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce"))
        net = (pd.to_numeric(df.get("lev_money_positions_long"), errors="coerce").fillna(0)
               - pd.to_numeric(df.get("lev_money_positions_short"), errors="coerce").fillna(0))
        created = pd.to_datetime(df.get(":created_at"), errors="coerce", utc=True)
        one = pd.DataFrame({f"{key}_net": net.to_numpy(), "created_at": created.to_numpy()},
                           index=as_of)
        one = one[~one.index.duplicated(keep="last")].sort_index()
        per[key] = one

    combined = pd.concat([per[k][[f"{k}_net"]] for k in markets], axis=1)
    # created_at is the same publish event for every market on a given as_of;
    # coalesce across markets so a gap in one is filled from another.
    created = None
    for k in markets:
        cs = per[k]["created_at"]
        created = cs if created is None else created.combine_first(cs)
    combined["created_at"] = created.reindex(combined.index)
    return combined.sort_index()


# ── cache (offline tier) ─────────────────────────────────────────────────────

def _read_cot_cache(path):
    """Load the cached raw frame (as_of index, `<key>_net` cols, created_at), or
    None. Stored raw -- not the derived z-scores -- so the trailing window is
    always recomputed consistently over the full history on every load."""
    if not (path and os.path.exists(path)):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except (OSError, ValueError):
        return None
    if df.empty:
        return None
    df.index = _utc_normalize(df.index)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _merge_write_cot_cache(df, path):
    """Merge the freshly fetched raw frame onto any existing cache (keep last)
    and write it back, so a fetch can never truncate longer cached history."""
    cached = _read_cot_cache(path)
    if cached is not None:
        df = pd.concat([cached, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    try:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path)
    except OSError:
        pass
    return df


# ── z-score derivation (weekly cadence) + availability stamping ──────────────

def _compute_cot_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw weekly frame (as_of index, `<key>_net`, created_at) into an
    AVAILABILITY-date-indexed frame of the COT_FEATURE_COLUMNS z-scores.

    The z-score for each market is a trailing-window standardization of the
    weekly net series (window ends AT the current week -> uses only data public
    by that week's report, no look-ahead), then re-indexed by the availability
    date so ffill onto daily bars is look-ahead-safe by construction.
    """
    raw = raw.sort_index()
    created = raw["created_at"] if "created_at" in raw.columns else pd.Series(pd.NaT, index=raw.index)
    avail = pd.DatetimeIndex([availability_date(ts, ct) for ts, ct in zip(raw.index, created)])

    out = pd.DataFrame(index=avail)
    for key in COT_MARKETS:
        net_col = f"{key}_net"
        if net_col not in raw.columns:
            out[f"{key}_zscore"] = np.nan
            continue
        net = raw[net_col].astype(float)
        roll = net.rolling(ZSCORE_WINDOW_WEEKS, min_periods=ZSCORE_MIN_WEEKS)
        z = (net - roll.mean()) / roll.std()
        z = z.replace([np.inf, -np.inf], np.nan)   # degenerate flat window -> neutral (0 downstream)
        out[f"{key}_zscore"] = z.to_numpy()

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_cot_features(start=None, end=None, config=None, base_dir="", cache_path=None):
    """Fetch COT positioning and return (cot_frame, source).

    `cot_frame` is availability-date-indexed (UTC) with the COT_FEATURE_COLUMNS
    z-scores; `source` is 'CFTC_api' / 'cache' / 'unavailable'. `start`/`end` are
    accepted for signature parity with fetch_macro_features but the whole (small)
    weekly history is always pulled -- the trailing z-score needs it. Returns
    (None, 'unavailable') only if the API is down AND no cache exists.
    """
    cot_cfg = (config or {}).get("cot", {}) if config else {}
    if cache_path is None:
        cache_path = _abs(base_dir, cot_cfg.get("cache_path")) if cot_cfg.get("cache_path") \
            else DEFAULT_COT_CACHE
    url = cot_cfg.get("socrata_url", COT_SOCRATA_URL)
    markets = cot_cfg.get("markets", COT_MARKETS)

    raw = _fetch_all_markets_via_api(url=url, markets=markets)
    if raw is not None and not raw.empty:
        raw = _merge_write_cot_cache(raw, cache_path)
        return _compute_cot_frame(raw), "CFTC_api"

    cached = _read_cot_cache(cache_path)
    if cached is not None:
        return _compute_cot_frame(cached), "cache"
    return None, "unavailable"


# ── daily join (mirrors merge_macro_features' as-of ffill) ───────────────────

def add_cot_features(df: pd.DataFrame, cot_frame=None, base_dir="", config=None) -> pd.DataFrame:
    """Join cot_eur_zscore / cot_usdindex_zscore onto a daily-indexed frame by
    AVAILABILITY date (as-of ffill), neutral 0 where unavailable (pre-2006 or
    z-score warm-up, or an entirely missing feed).

    Look-ahead-safe: `cot_frame` is indexed by the date each weekly reading
    became public, and ffill only ever carries a PAST reading forward onto later
    bars -- identical discipline to merge_macro_features. When `cot_frame` is
    None it is fetched (offline-safe via cache); tests inject a synthetic frame.
    """
    if cot_frame is None:
        cot_frame, _src = fetch_cot_features(base_dir=base_dir, config=config)

    out = df.copy()
    idx = pd.DatetimeIndex(df.index)
    naive = (idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx).normalize()

    for col in COT_FEATURE_COLUMNS:
        if cot_frame is not None and col in cot_frame.columns:
            s = cot_frame[col].dropna()
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


def cot_staleness_days(cot_frame, as_of_date):
    """Days between `as_of_date` and the availability date of the COT reading in
    effect on that date (latest availability date <= as_of_date). None if no
    reading is available yet -- surfaced as a diagnostic so a stale feed during
    a future shutdown is visible, never silently degraded."""
    if cot_frame is None or len(cot_frame) == 0:
        return None
    ad = pd.Timestamp(as_of_date)
    ad = ad.tz_localize("UTC") if ad.tzinfo is None else ad.tz_convert("UTC")
    ad = ad.normalize()
    idx = _utc_normalize(cot_frame.index)
    prior = idx[idx <= ad]
    if len(prior) == 0:
        return None
    return int((ad - prior.max()).days)


if __name__ == "__main__":
    frame, src = fetch_cot_features()
    if frame is None:
        print("COT unavailable (no API, no cache).")
    else:
        print(f"source={src}  rows={len(frame)}  "
              f"availability {frame.index.min().date()} -> {frame.index.max().date()}")
        print(frame.tail(6).to_string())
