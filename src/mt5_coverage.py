"""
MT5 coverage guard -- detect SILENTLY holed history before it reaches disk.

WHY THIS EXISTS
---------------
``results/pooled_h1/USDCHF_h1.csv`` was found missing 2026-06-15 06:00 through
2026-07-28 00:00 -- 42 days -- while reporting 70,000 bars and a last timestamp
of 2026-07-28. Every count-based check passed.

CAUSE: an MT5 symbol that is not selected in Market Watch has only a partially
synced history block on disk. ``copy_rates_from_pos`` does not error; it reaches
further back to satisfy the requested bar count, so the total still looks right
and the hole sits in the middle. **A bar count and a last timestamp are not
evidence of completeness.** The only checks that catch this are interior gaps
and per-month density.

TWO LAYERS
----------
1. PREVENTION -- :func:`sync_symbol` selects the symbol into Market Watch and
   issues a ``copy_rates_range`` probe to force the terminal to pull the recent
   range, BEFORE ``copy_rates_from_pos`` is called. This is the fix proven
   during the 19-instrument pull that first hit the artifact.
2. DETECTION -- :func:`assert_coverage` raises :class:`CoverageError` naming the
   missing range. It never repairs by filling: a fabricated bar is worse than a
   loud failure, and a holed frame must never overwrite a good cache.

WHY 72h IS NOT THE RIGHT SINGLE THRESHOLD
-----------------------------------------
Measured on this broker's own history, the year-end closures run **73.0 to 77.0
hours** -- above 72h. A flat 72h rule would fire every Christmas. So the tight
72h bound applies for ~11.5 months of the year and a wider
``YEAR_END_ALLOWANCE_HOURS`` applies to gaps overlapping the turn of the year.
The 42-day hole that motivated this module is 1026h and is caught by either.
"""

import calendar
import datetime as dt

import pandas as pd

# Normal FX weekend is ~49h (Fri close -> Sun open); 72h clears it with room.
MAX_GAP_HOURS = 72.0

# Measured year-end closures on this feed: 73.0, 73.0, 74.0, 74.0, 77.0 h.
# 96h clears every observed closure while still catching a multi-day hole.
YEAR_END_ALLOWANCE_HOURS = 96.0
YEAR_END_START = (12, 20)      # inclusive
YEAR_END_END = (1, 5)          # inclusive

# A month below this fraction of its expected bar count is a hole, not a holiday.
MONTHLY_FLOOR = 0.60

# FX trades ~5 days in 7. Bars per CALENDAR day, by timeframe.
BARS_PER_DAY = {'D1': 5.0 / 7.0, 'H1': 24.0 * 5.0 / 7.0, 'M15': 96.0 * 5.0 / 7.0}

# Below this many whole months, only the (exempt) partial end months exist.
MIN_MONTHS_FOR_DENSITY = 3


class CoverageError(RuntimeError):
    """Raised when an MT5-sourced frame has a hole. Names the missing range."""


def _naive_utc(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    return idx


def _in_year_end(ts) -> bool:
    """True if `ts` falls in the Dec 20 -- Jan 5 turn-of-year window."""
    m, d = ts.month, ts.day
    return (m, d) >= YEAR_END_START or (m, d) <= YEAR_END_END


def find_gaps(index, max_gap_hours: float = MAX_GAP_HOURS,
              year_end_allowance: float = YEAR_END_ALLOWANCE_HOURS):
    """
    Interior gaps that exceed the allowance for where they sit.

    Returns a list of ``(start, end, hours)`` -- ``start`` is the last bar
    BEFORE the hole and ``end`` the first bar after it, so the missing range is
    the open interval between them. A gap is judged against
    ``year_end_allowance`` if EITHER endpoint lies in the turn-of-year window,
    otherwise against ``max_gap_hours``.
    """
    idx = _naive_utc(index).sort_values()
    if len(idx) < 2:
        return []
    s = pd.Series(idx)
    hours = s.diff().dt.total_seconds() / 3600.0
    out = []
    for i in range(1, len(s)):
        h = float(hours.iloc[i])
        a, b = s.iloc[i - 1], s.iloc[i]
        allowance = year_end_allowance if (_in_year_end(a) or _in_year_end(b)) else max_gap_hours
        if h > allowance:
            out.append((a, b, h))
    return out


def find_thin_months(index, timeframe: str, floor: float = MONTHLY_FLOOR):
    """
    Months whose bar count falls below ``floor`` of the expected count for the
    timeframe. The FIRST and LAST months are exempt -- they are partial by
    construction, not holed.

    Returns ``[(period_start, observed, expected), ...]``.
    """
    idx = _naive_utc(index).sort_values()
    per_day = BARS_PER_DAY.get(timeframe.upper())
    if per_day is None:
        raise ValueError('unknown timeframe %r; expected one of %s'
                         % (timeframe, sorted(BARS_PER_DAY)))
    counts = pd.Series(1, index=idx).resample('MS').sum()
    if len(counts) < MIN_MONTHS_FOR_DENSITY:
        return []
    out = []
    for period, got in counts.iloc[1:-1].items():        # drop partial ends
        days = calendar.monthrange(period.year, period.month)[1]
        expected = per_day * days
        if got < floor * expected:
            out.append((period, int(got), int(round(expected))))
    return out


def assert_coverage(df, timeframe: str, label: str = 'frame',
                    max_gap_hours: float = MAX_GAP_HOURS,
                    monthly_floor: float = MONTHLY_FLOOR):
    """
    Raise :class:`CoverageError` if ``df`` has an interior hole. Returns ``df``
    unchanged when clean, so it can wrap a fetch inline.

    NEVER fills. A holed frame is refused, not repaired -- the caller's job is
    to retry or to leave the existing cache alone, not to serve fabricated bars.
    """
    if df is None or len(df) == 0:
        return df
    gaps = find_gaps(df.index, max_gap_hours=max_gap_hours)
    thin = find_thin_months(df.index, timeframe, floor=monthly_floor)
    if not gaps and not thin:
        return df

    parts = []
    for a, b, h in gaps:
        parts.append('missing %s -> %s (%.1f h = %.1f days)'
                     % (a, b, h, h / 24.0))
    for period, got, expected in thin:
        parts.append('thin month %s: %d bars vs ~%d expected (%.0f%%)'
                     % (period.strftime('%Y-%m'), got, expected,
                        100.0 * got / max(expected, 1)))
    raise CoverageError(
        '%s (%s) has a coverage hole and was REFUSED, not repaired: %s. '
        'Most likely the MT5 symbol was not selected in Market Watch, so only a '
        'partial history block was synced -- call sync_symbol() before fetching '
        'and retry. The frame was not written to disk.'
        % (label, timeframe.upper(), '; '.join(parts)))


def sync_symbol(mt5, symbol: str, probe_days: int = 20, attempts: int = 3,
                sleep_seconds: float = 5.0) -> bool:
    """
    PREVENTION. Select ``symbol`` into Market Watch and force the terminal to
    pull its recent history before any ``copy_rates_from_pos`` call, which is
    what stops the partial-block artifact at source.

    Returns True when the probe comes back with data. Never raises -- a False
    return simply means the caller should expect ``assert_coverage`` to do the
    catching. ``mt5`` is injected so this is unit-testable without a terminal.
    """
    import time

    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        return False
    end = dt.datetime.now()
    start = end - dt.timedelta(days=probe_days)
    for attempt in range(max(1, attempts)):
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
        except Exception:
            rates = None
        if rates is not None and len(rates) > 0:
            return True
        if attempt < attempts - 1:
            time.sleep(sleep_seconds)
    return False
