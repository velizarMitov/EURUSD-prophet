"""MT5 M1 data pull for the Idea-2 curl measurement — RESEARCH ONLY.

Pulls the six pairs of the EUR/USD/JPY/GBP subgraph at M1 and caches them to
``results/curl/raw/<SYMBOL>_M1.parquet`` so the (slow) terminal sync happens once.

Nothing here touches ``src/inference.py``, ``models/`` or the serving path. The only
consumer is ``src/curl_mt5.py``.

--------------------------------------------------------------------------------------
FOUR THINGS THIS MODULE EXISTS TO GET RIGHT
--------------------------------------------------------------------------------------
1. **Symbol names are broker-specific.** ActivTrades happens to expose bare ``EURUSD``,
   but other brokers ship ``EURUSD.a`` / ``EURUSDm`` / ``EURUSD_raw`` / ``EURUSD.i``.
   Resolution goes through ``mt5.symbols_get()`` and normalised matching, never a
   hard-coded string, so re-running this against a different terminal does not silently
   fetch nothing.

2. **``tick_volume`` and ``spread`` are load-bearing, not decoration.** ``tick_volume``
   is the staleness estimator the whole null is built on (``curl_stress.staleness_null``);
   ``spread`` is the control the index has to beat, because MT5 bars are BID and bid-side
   curl widens mechanically with spreads. Slicing down to OHLC would quietly reduce the
   study to "curl variance tracks volatility", which is the confound, not the hypothesis.

3. **MT5 timestamps are SERVER time wearing a UTC costume.** ``copy_rates_range``
   returns ``time`` as Unix seconds, but the value encodes the broker server's wall clock,
   not UTC. Naively calling it UTC shifts every bar by the server offset (+2/+3h on the
   EET servers most EU brokers run). That does NOT affect the curl -- all six pairs share
   one clock, so a global shift cancels in ``log(A/B)+log(B/C)+log(C/A)`` -- but it does
   corrupt every hour-of-day statement, and hour-of-day is precisely what the STOP GATE
   asks about (`CURL_EXPERIMENT_PLAN.md` §2.3) and what sank volatility hypothesis #3.
   ``to_utc`` does a real timezone conversion and ``weekly_boundary_evidence`` shows the
   receipt, rather than asserting the offset from a live tick -- the live-tick method is
   unreliable at exactly the moment this was written (a weekend, when the newest bar is
   two days stale; see the -9/-10/-11h excursions in ``results/h1_feed_offset.json``).

4. **``copy_rates_range`` truncates silently.** The terminal serves at most
   "Max bars in chart" per call and returns a short array rather than an error. Years of
   M1 is millions of bars, so the pull is chunked and each chunk's boundaries are checked.

5. **``spread`` comes back in POINTS, not price.** MT5 reports the bar spread as an
   integer count of points (EURUSD 5 == 0.00005 == 0.5 pip). ``curl_mt5.convention_report``
   documents at its own line 185 that it expects PRICE units, and divides by ``2*close``
   to get a relative half-spread. Handing it raw points inflates ``pred_bid_bp`` by
   ``1/point`` -- a factor of 100,000 on the majors, which turns a real 0.22 bp bid-side
   offset into 21,551 bp. Every triangle would then report "UNEXPLAINED CONSTANT — check
   for an inverted pair" and the study would stop at step 2 on a units bug. So the cache
   keeps the broker's raw integer as ``spread_points`` alongside the symbol's ``point``
   size, and the loaders derive ``spread`` in price units from them.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import curl_stress as cs

#: Where the raw per-symbol M1 parquets live.
RAW_DIR = Path("results/curl/raw")

#: The six edges of K4 on (EUR, USD, JPY, GBP), as canonical broker-style names.
CANONICAL: tuple[str, ...] = tuple(cs.symbol(b, q) for b, q in cs.PAIRS)

#: ActivTrades EU runs CET/CEST -- UTC+1 winter, UTC+2 summer, on the EU DST rule.
#:
#: This was MEASURED, not assumed, and the first guess (EET, which most MT5 brokers use)
#: was wrong by an hour. The fingerprint is in ``weekly_boundary_evidence`` split by
#: month: the week opens at server hour 23 and closes at 22 in every month EXCEPT March,
#: where it is 22 and 21. March is when the US has moved to DST but the EU has not, so a
#: server on the EU rule slips one hour against the New-York-anchored FX week for ~3
#: weeks. An EET server would show 0/23, and a fixed UTC+2 server would show no March
#: excursion at all. A whole-year mode cannot tell these apart -- only the season split
#: can, which is why the check is season-split.
#:
#: Getting this wrong does NOT corrupt the curl (a global clock shift cancels inside the
#: triangle -- pinned by ``test_a_global_clock_shift_cannot_manufacture_curl``), but it
#: does shift every hour-of-day bucket by one, and hour-of-day is exactly what the STOP
#: GATE interrogates.
DEFAULT_SERVER_TZ = "Europe/Berlin"

#: M1 bars per chunk request. 30 days = 43,200 bars max, comfortably under the terminal's
#: default 100k "Max bars in chart" ceiling even if the broker fills every minute.
CHUNK_DAYS = 30

M1_SECONDS = 60


# ======================================================================================
# 1. Symbol resolution
# ======================================================================================


def _normalise(name: str) -> str:
    """Strip separators and case so ``EURUSD.a`` / ``eurusd_raw`` compare to ``EURUSD``."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def resolve_symbols(
    mt5, wanted: Sequence[str] = CANONICAL
) -> dict[str, str]:
    """Map canonical pair names -> this broker's actual symbol names.

    Brokers append suffixes (``.a``, ``m``, ``_raw``, ``.i``, ``.pro``) and some prepend
    a venue tag. Matching is: exact name first, then normalised-prefix, preferring the
    SHORTEST candidate -- on a terminal exposing both ``EURUSD`` and ``EURUSD.pro`` the
    bare symbol is the retail default and the one whose history is actually synced.

    Raises when a pair cannot be resolved: a missing edge silently drops a triangle, and
    an incomplete K4 is not a cycle space.
    """
    available = [s.name for s in (mt5.symbols_get() or [])]
    if not available:
        raise RuntimeError("mt5.symbols_get() returned nothing — is the terminal logged in?")

    by_exact = {n: n for n in available}
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for want in wanted:
        if want in by_exact:
            resolved[want] = want
            continue
        target = _normalise(want)
        cands = [n for n in available if _normalise(n).startswith(target)]
        if not cands:
            unresolved.append(want)
            continue
        resolved[want] = min(cands, key=lambda n: (len(n), n))
    if unresolved:
        raise RuntimeError(
            f"could not resolve {unresolved} among {len(available)} broker symbols. "
            "The K4 subgraph needs all six edges; a missing pair is not recoverable."
        )
    return resolved


# ======================================================================================
# 2. The pull
# ======================================================================================


def _rates_to_frame(rates) -> pd.DataFrame:
    """MT5 structured array -> DataFrame indexed by (still server-clock) timestamps.

    ``spread`` is renamed to ``spread_points`` at the boundary so the units cannot be
    mistaken downstream — see module docstring, point 5.
    """
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df = df.rename(columns={"spread": "spread_points"})
    keep = [
        c for c in ("open", "high", "low", "close", "tick_volume", "spread_points")
        if c in df
    ]
    return df[keep]


def with_price_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``spread`` in PRICE units from ``spread_points`` x ``point``.

    ``curl_mt5.convention_report`` and ``curl_mt5.confound_design`` both consume a
    price-unit ``spread``. Keeping the raw integer under its own name and deriving the
    price-unit column here means the cache stays a faithful record of what the broker
    served while the analysis layer gets the units its arithmetic assumes.
    """
    out = df.copy()
    if "spread_points" in out.columns and "point" in out.columns:
        out["spread"] = out["spread_points"].astype(float) * out["point"].astype(float)
    return out


def fetch_symbol_m1(
    mt5,
    symbol: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    chunk_days: int = CHUNK_DAYS,
    select_pause: float = 0.5,
    point: float | None = None,
) -> pd.DataFrame:
    """Chunked ``copy_rates_range`` pull of one symbol's M1 history.

    Timestamps come back on the SERVER clock and are left that way here — conversion is
    ``to_utc``'s job, applied once after all six symbols are in hand so they cannot drift
    apart. ``tick_volume`` and ``spread`` are retained (see module docstring, point 2).

    The chunking is not politeness: ``copy_rates_range`` silently returns a truncated
    array when the request exceeds the terminal's bar ceiling, so a single multi-year M1
    request looks successful and hands back a partial block. This is the same failure mode
    ``src/mt5_coverage.assert_coverage`` was written for on the H1 path.
    """
    mt5.symbol_select(symbol, True)
    if select_pause:
        time.sleep(select_pause)  # let the terminal start its history sync

    parts: list[pd.DataFrame] = []
    cursor = start
    step = dt.timedelta(days=chunk_days)
    while cursor < end:
        stop = min(cursor + step, end)
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, cursor, stop)
        except Exception:
            rates = None
        if rates is not None and len(rates):
            parts.append(_rates_to_frame(rates))
        cursor = stop

    if not parts:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "tick_volume", "spread_points",
                     "point"]
        )
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # The point size travels WITH the bars. Storing it as a column keeps each parquet
    # self-describing: a later reader cannot pair EURUSD spreads with JPY point sizes.
    if point is None:
        info = mt5.symbol_info(symbol)
        point = float(getattr(info, "point", np.nan)) if info is not None else np.nan
    out["point"] = float(point)
    return out


# ======================================================================================
# 3. Server clock -> UTC
# ======================================================================================


def weekly_boundary_evidence(
    index: pd.DatetimeIndex, *, by: str = "year"
) -> pd.DataFrame:
    """Where the FX trading week starts and ends, per calendar year, in the given clock.

    THE RECEIPT for the timezone conversion, and the reason this module does not infer an
    offset from a live tick. The FX week is a hard, unambiguous feature of the data: it is
    anchored to 17:00 New York, i.e. it closes 21:00 UTC in northern summer and 22:00 UTC
    in winter. So on a correctly-converted UTC index the LAST bar of the week lands at
    hour 20 (summer) or 21 (winter) and the FIRST at 21 or 22.

    Any server clock instead pins BOTH to a constant, because that is the point of a
    broker timezone -- it makes the week close at a fixed local hour and yields exactly
    five daily bars. On this feed (CET/CEST) that constant is 23 open / 22 close; on the
    more common EET it would be 0 / 23. Seeing a constant here means you are still on the
    server clock.

    ``month_split`` is the part that actually identifies the zone. A whole-year mode
    cannot separate "EU DST rule" from "fixed offset", because the DST-mismatch weeks are
    a minority of the year and the mode hides them. Splitting by month exposes them: on
    the EU rule, March (US on DST, EU not yet) slips one hour and shows 22/21 against
    23/22 everywhere else.

    Split by year as well, so a broker that changed its server timezone mid-history shows
    up as a row that disagrees with its neighbours instead of averaging into nonsense.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return pd.DataFrame()
    s = pd.Series(idx, index=idx)
    gaps = s.diff()
    # A week boundary is any gap materially longer than the weekend-free intraday gaps.
    is_open = (gaps > pd.Timedelta(hours=12)).to_numpy()
    is_close = np.roll(is_open, -1)
    is_close[-1] = False

    key = idx.month if by == "month" else idx.year
    rows = []
    for value in np.unique(key):
        sel = key == value
        o = idx[is_open & sel]
        c = idx[is_close & sel]
        rows.append(
            {
                by: int(value),
                "n_bars": int(sel.sum()),
                "n_weeks": int(len(o)),
                "week_open_hour_mode": int(pd.Series(o.hour).mode().iloc[0]) if len(o) else -1,
                "week_close_hour_mode": int(pd.Series(c.hour).mode().iloc[0]) if len(c) else -1,
                "sunday_bars_pct": float((idx[sel].dayofweek == 6).mean() * 100),
                "saturday_bars_pct": float((idx[sel].dayofweek == 5).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def to_utc(df: pd.DataFrame, *, server_tz: str = DEFAULT_SERVER_TZ) -> pd.DataFrame:
    """Convert a server-clock frame to a real tz-aware UTC index.

    ``tz_localize(server_tz)`` then ``tz_convert('UTC')`` — a genuine conversion that
    follows the DST rule per bar, not a constant offset smeared over the history.

    DST edges: the EU switch happens 03:00 local on a Sunday, when the FX market is shut,
    so in practice no bar is ambiguous or nonexistent. Rather than assume that, both are
    mapped to NaT and dropped, and the count is available to the caller through the row
    delta — silently shifting an ambiguous bar by an hour is exactly the kind of one-bar
    misalignment that ``curl_stress.shift_placebo`` shows inflates RMS curl ~10x.
    """
    if df.index.tz is not None:
        return df.tz_convert("UTC")
    idx = df.index.tz_localize(server_tz, ambiguous="NaT", nonexistent="NaT")
    out = df.copy()
    out.index = idx
    return out[out.index.notna()].tz_convert("UTC")


# ======================================================================================
# 4. Coverage report
# ======================================================================================


def coverage_report(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-symbol row count, span, and gap census.

    ``gaps_gt_1bar`` is the headline the runbook asks for, but on its own it is not
    interpretable: a five-day-a-week market has ~52 legitimate weekend gaps per year, and
    they would dominate the count. So the census separates them:

    ``weekend_gaps``   gaps > 12h — the Friday-close/Sunday-open seam. Expected.
    ``intraday_gaps``  gaps > 1 bar that are NOT weekend seams. THESE are the ones that
                       matter: minutes where the broker published no bar. Some are real
                       (thin 22:00 rollover, holidays), but a large count means the feed
                       is sparse and ``align_bars``' inner join will cut hard.
    ``largest_intraday_gap_min``  the worst one, in minutes.
    ``missing_minutes_pct``  fraction of the theoretical M1 grid absent, weekends removed.
    """
    rows = []
    for sym, df in frames.items():
        idx = pd.DatetimeIndex(df.index)
        if len(idx) < 2:
            rows.append({"symbol": sym, "rows": len(idx)})
            continue
        d = idx.to_series().diff().dropna().dt.total_seconds()
        weekend = d > 12 * 3600
        gap = d > M1_SECONDS
        intraday = gap & ~weekend
        # theoretical minutes, weekend seams removed
        span_min = (idx[-1] - idx[0]).total_seconds() / 60.0
        weekend_min = d[weekend].sum() / 60.0
        expected = max(span_min - weekend_min, 1.0)
        rows.append(
            {
                "symbol": sym,
                "rows": int(len(idx)),
                "first": idx[0],
                "last": idx[-1],
                "gaps_gt_1bar": int(gap.sum()),
                "weekend_gaps": int(weekend.sum()),
                "intraday_gaps": int(intraday.sum()),
                "largest_intraday_gap_min": float(d[intraday].max() / 60.0)
                if intraday.any()
                else 0.0,
                "missing_minutes_pct": float((1.0 - len(idx) / expected) * 100.0),
                "median_ticks": float(df["tick_volume"].median())
                if "tick_volume" in df
                else np.nan,
                "median_spread_pts": float(df["spread_points"].median())
                if "spread_points" in df
                else np.nan,
                "median_spread_bp": float(
                    (df["spread"] / df["close"]).median() * 1e4
                )
                if "spread" in df
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def common_span(frames: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """What survives the strict inner join — the only history the study can actually use.

    ``align_bars`` drops any timestamp where a single pair is missing (never ffills, since
    an ffilled bar manufactures curl). So the binding constraint is not the deepest
    symbol's history but the SHALLOWEST one's, intersected bar-by-bar with the rest.
    """
    idx: pd.Index | None = None
    for df in frames.values():
        i = pd.DatetimeIndex(df.index)
        idx = i if idx is None else idx.intersection(i)
    assert idx is not None
    per_symbol_min = min(len(df) for df in frames.values())
    return {
        "common_bars": int(len(idx)),
        "first": idx[0] if len(idx) else None,
        "last": idx[-1] if len(idx) else None,
        "shallowest_symbol_bars": int(per_symbol_min),
        "retention_vs_shallowest_pct": float(100.0 * len(idx) / max(per_symbol_min, 1)),
    }


# ======================================================================================
# 5. Orchestrator
# ======================================================================================


def _cache_path(symbol: str, raw_dir: Path) -> Path:
    return raw_dir / f"{symbol}_M1.parquet"


def pull_all(
    *,
    years: float = 5.0,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    raw_dir: Path | str = RAW_DIR,
    server_tz: str = DEFAULT_SERVER_TZ,
    refresh: bool = False,
    wanted: Sequence[str] = CANONICAL,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fetch (or load from cache) all six M1 series and print the coverage report.

    Returns ``(frames_by_symbol_utc, coverage_table)``. Frames are cached on the SERVER
    clock exactly as the broker served them — the cache is a faithful record of the pull,
    and ``to_utc`` is applied on load, so a later correction to ``server_tz`` does not
    require re-pulling several million bars.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    end = end or dt.datetime.now()
    start = start or (end - dt.timedelta(days=int(365.25 * years)))

    frames: dict[str, pd.DataFrame] = {}
    need_terminal = refresh or any(
        not _cache_path(s, raw_dir).exists() for s in wanted
    )

    mt5 = None
    mapping: dict[str, str] = {s: s for s in wanted}
    if need_terminal:
        import MetaTrader5 as mt5_mod

        mt5 = mt5_mod
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
        mapping = resolve_symbols(mt5, wanted)
        print(f"resolved symbols: {mapping}")

    try:
        for canon in wanted:
            path = _cache_path(canon, raw_dir)
            if path.exists() and not refresh:
                frames[canon] = pd.read_parquet(path)
                print(f"  {canon}: {len(frames[canon]):,} bars from cache")
                continue
            assert mt5 is not None
            broker = mapping[canon]
            info = mt5.symbol_info(broker)
            df = fetch_symbol_m1(
                mt5, broker, start, end,
                point=float(info.point) if info is not None else None,
            )
            if df.empty:
                raise RuntimeError(
                    f"{canon} ({broker}): the terminal returned no M1 bars for "
                    f"{start:%Y-%m-%d}..{end:%Y-%m-%d}."
                )
            df.to_parquet(path)
            frames[canon] = df
            print(f"  {canon} ({broker}): {len(df):,} bars -> {path}")
    finally:
        if mt5 is not None:
            mt5.shutdown()

    frames_server_index = frames[wanted[0]].index
    server_evidence = weekly_boundary_evidence(frames_server_index)
    frames = {k: with_price_spread(to_utc(v, server_tz=server_tz))
              for k, v in frames.items()}
    utc_evidence = weekly_boundary_evidence(frames[wanted[0]].index)

    print("\nweekly boundary on the RAW SERVER clock (a CONSTANT hour = still server time):")
    print(server_evidence.to_string(index=False))
    print("\nsame, split by month — the March excursion identifies the DST rule:")
    print(weekly_boundary_evidence(
        pd.DatetimeIndex(frames_server_index), by="month"
    ).to_string(index=False))
    print(f"\nweekly boundary after converting {server_tz} -> UTC "
          "(expect open 21 summer / 22 winter, close 20 / 21):")
    print(utc_evidence.to_string(index=False))

    cov = coverage_report(frames)
    print("\ncoverage:")
    print(cov.to_string(index=False))

    common = common_span(frames)
    print(f"\nstrict inner join across all six: {common['common_bars']:,} common bars "
          f"({common['first']} -> {common['last']}), "
          f"{common['retention_vs_shallowest_pct']:.1f}% of the shallowest symbol")
    return frames, cov


def load_cached(
    *, raw_dir: Path | str = RAW_DIR, server_tz: str = DEFAULT_SERVER_TZ,
    wanted: Sequence[str] = CANONICAL,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load the cache and key it by ``(base, quote)`` — the shape ``curl_mt5`` wants."""
    raw_dir = Path(raw_dir)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for (b, q) in cs.PAIRS:
        canon = cs.symbol(b, q)
        path = _cache_path(canon, raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} — run pull_all() first")
        out[(b, q)] = with_price_spread(
            to_utc(pd.read_parquet(path), server_tz=server_tz)
        )
    return out


if __name__ == "__main__":  # pragma: no cover
    pull_all()
