import json
import math
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
    Comparison is on the calendar date. D1 bars (`_fetch_from_mt5`),
    `history_df` and the yfinance fallback are all tz-naive by convention,
    and MT5's raw epoch field itself already bakes in the broker SERVER's
    own wall-clock date (verified live against a real ActivTrades session
    and across four years of EU DST transitions -- the epoch never carries a
    genuine UTC correction, so parsing it with or without `utc=True`
    produces byte-identical wall-clock/weekday fields). That makes tz-naive
    the correct, already-verified-safe convention here -- it is NOT the same
    thing as "local machine time", it is "server-labelled calendar date"
    read literally off the broker feed.

    H1/M15 (`_fetch_h1_from_mt5`/`_fetch_m15_from_mt5`) tag the identical raw
    values as UTC for their own intraday-grouping needs, but that tag is a
    label over the same server wall-clock numbers, not a real conversion. If
    a tz-aware index or `now` ever reaches this function (e.g. a future
    caller reusing an H1 frame, or a test constructing one), the tz tag is
    stripped -- not converted -- before comparing, so the wall-clock date
    used for the weekday/forming-bar checks stays identical to the tz-naive
    convention above instead of silently raising a naive-vs-aware
    TypeError or, worse, shifting the calendar date under a real conversion.
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return ohlcv_df
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_localize(None)
    today = now.normalize()
    idx = ohlcv_df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    keep = (idx.normalize() < today) & (idx.weekday < 5)
    return ohlcv_df[keep]


# The FX week's first H1 bar in this feed's server-labelled clock. Chosen from
# the DATA, not from a convention: across the 70,000-bar EURUSD H1 cache there
# are ZERO Saturday bars and Sunday bars at exactly two hours -- 22 (34 bars) and
# 23 (579 bars). Those 613 Sunday bars ARE in H_dir.1's training set, so they are
# in-distribution and must NOT be dropped. Anything earlier on a Sunday is a
# short partial weekend-open bar the training data never contained.
H1_WEEKLY_OPEN_HOUR = 22

# Plausible broker server offsets from UTC. Outside this band the inferred
# offset is not an offset at all -- it is a stale feed -- and we say so.
H1_MIN_BROKER_OFFSET_H, H1_MAX_BROKER_OFFSET_H = -12, 14

# Where the accepted offset (and its change history) is persisted.
H1_OFFSET_STATE_PATH = 'results/h1_feed_offset.json'
# A raw inference landing within this many minutes BELOW its own ceiling is
# indistinguishable from the top-of-hour emit-lag artifact and is not trusted on
# its own. Comfortably clear of the boundary => trusted immediately.
H1_OFFSET_BOUNDARY_MARGIN_MIN = 2.0
# A boundary-adjacent change must repeat at least this long after first sight.
H1_OFFSET_CONFIRM_MINUTES = 5.0
H1_OFFSET_HISTORY_LIMIT = 50


def _raw_feed_offset(index, now_utc):
    """
    The UNSTICKY inference: (offset, diff_hours, margin_minutes), or None when
    the feed is empty or too stale for the result to mean anything.

    `margin_minutes` is how far `diff_hours` sits BELOW its own ceiling. It is
    the artifact detector -- see `infer_h1_feed_now` for why.

    Exposed separately so the test suite can drive the unfixed path directly and
    demonstrate the wrong value the stickiness prevents.
    """
    if index is None or len(index) == 0:
        return None
    idx = index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    newest = pd.Timestamp(max(idx))

    diff_hours = (newest - now_utc).total_seconds() / 3600.0
    offset = int(math.ceil(diff_hours))
    if not (H1_MIN_BROKER_OFFSET_H <= offset <= H1_MAX_BROKER_OFFSET_H):
        return None
    margin_minutes = (offset - diff_hours) * 60.0
    return offset, diff_hours, margin_minutes


def _load_offset_state(path):
    try:
        with open(path) as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save_offset_state(path, state):
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(state, fh, indent=1)
    except OSError:
        pass                       # persistence is best-effort, never fatal


def _accept_offset(state, offset, now_utc, reason):
    """Record an accepted offset and append to the change log."""
    previous = state.get('offset_hours')
    state['offset_hours'] = int(offset)
    state['accepted_at_utc'] = now_utc.isoformat()
    state['pending'] = None
    if previous != offset:
        history = list(state.get('history') or [])
        history.append({'offset_hours': int(offset),
                        'previous_offset_hours': previous,
                        'accepted_at_utc': now_utc.isoformat(),
                        'reason': reason})
        state['history'] = history[-H1_OFFSET_HISTORY_LIMIT:]
    return state


def resolve_h1_feed_offset(index, now_utc, state_path=H1_OFFSET_STATE_PATH,
                           persist=True):
    """
    The STICKY offset. Returns (offset_or_None, state, confirmed).

    `confirmed` is False only in the COLD-START-INSIDE-THE-EMIT-LAG-WINDOW case
    described below. Callers must treat an unconfirmed clock as a degraded state
    and must NOT present the resulting prediction as actionable or settle it into
    the ledger.

    The raw inference is correct almost always, but it has one recurring failure
    mode. Immediately after an hour boundary, before MT5 has emitted the new bar,
    the newest label is still the PREVIOUS hour, so `diff` falls just below the
    integer boundary and `ceil` yields OFFSET - 1:

        utcnow 18:00:05, true server 20:00:05, offset +2, the 20:00 bar not yet
        emitted so newest = 19:00  ->  diff 0.9986h  ->  inferred offset 1.

    The service would then treat an ALREADY CLOSED hour as the open forecast bar
    and report ~59 minutes remaining -- the completed-bar bug, back again. The
    plausibility band cannot catch it: the confusable pair is +1 and +2, this
    broker's own CET and CEST, and both are legitimate.

    So the offset is treated as what it actually is -- STICKY. It changes twice a
    year, not twice an hour:

      * Inference EQUALS the persisted value -> accept.
      * Inference DIFFERS -> provisional, and it must earn acceptance either by
        sitting comfortably clear of the integer boundary (more than
        H1_OFFSET_BOUNDARY_MARGIN_MIN below its ceiling, which the emit-lag
        artifact never is -- it is seconds wide), or by REPEATING at least
        H1_OFFSET_CONFIRM_MINUTES after it was first seen.
      * A real DST transition confirms within minutes. The emit-lag artifact
        appears for seconds at the top of an hour, is superseded the moment the
        bar is emitted, and never confirms.

    Every accepted CHANGE is appended to the state file's history with its
    timestamp, so a genuine DST transition is visible in the record and an
    unexplained flip is too.

    COLD START IS HELD TO THE SAME BAR. With no persisted value there is nothing
    to defend against the artifact, so a first observation that lands inside the
    emit-lag window would become the BASELINE and stickiness would then protect
    the wrong offset. It self-heals on the next clean fetch, but the predictions
    made in between reach the ledger -- and ledger rows are the only evidence
    this model has. So a boundary-adjacent first reading is used for that single
    call, persisted NOWHERE, and reported as unconfirmed. A cold start with no
    clean reading yet is a known-uncertain state and says so rather than guessing
    and committing.
    """
    state = _load_offset_state(state_path) if state_path else {}
    raw = _raw_feed_offset(index, now_utc)
    persisted = state.get('offset_hours')

    if raw is None:
        # Stale/empty feed: never disturb the sticky value, never invent one.
        return (persisted if persisted is not None else None), state, True

    offset, _diff_hours, margin_minutes = raw
    dirty = False

    if persisted is None:
        if margin_minutes <= H1_OFFSET_BOUNDARY_MARGIN_MIN:
            # Boundary-adjacent FIRST reading: usable for this call, never a
            # baseline. Nothing is written, so the next clean observation still
            # cold-starts cleanly.
            return offset, state, False
        state = _accept_offset(state, offset, now_utc, 'cold_start')
        dirty = True
    elif offset == persisted:
        if state.get('pending'):
            state['pending'] = None        # provisional change did not survive
            dirty = True
    elif margin_minutes > H1_OFFSET_BOUNDARY_MARGIN_MIN:
        # Comfortably clear of the boundary: this cannot be the emit-lag artifact.
        state = _accept_offset(state, offset, now_utc, 'clear_of_boundary')
        dirty = True
    else:
        pending = state.get('pending') or {}
        first_seen = pending.get('first_seen_utc')
        same = pending.get('offset_hours') == offset
        elapsed_min = None
        if same and first_seen:
            try:
                elapsed_min = (now_utc - pd.Timestamp(first_seen)).total_seconds() / 60.0
            except Exception:
                elapsed_min = None
        if same and elapsed_min is not None and elapsed_min >= H1_OFFSET_CONFIRM_MINUTES:
            state = _accept_offset(state, offset, now_utc, 'confirmed_after_delay')
            dirty = True
        else:
            if not same:
                state['pending'] = {'offset_hours': int(offset),
                                    'first_seen_utc': now_utc.isoformat()}
                dirty = True
            # Keep serving the persisted offset until the change earns acceptance.
            if persist and dirty and state_path:
                _save_offset_state(state_path, state)
            return persisted, state, True

    if persist and dirty and state_path:
        _save_offset_state(state_path, state)
    return state.get('offset_hours'), state, True


def h1_feed_now_with_status(index, now_utc=None, state_path=H1_OFFSET_STATE_PATH,
                            persist=True):
    """(feed_now, clock_confirmed). See `infer_h1_feed_now` and
    `resolve_h1_feed_offset`; this is the variant serving uses, because it needs
    to know when the clock is only provisional."""
    now_utc = pd.Timestamp.utcnow() if now_utc is None else pd.Timestamp(now_utc)
    if now_utc.tzinfo is not None:
        now_utc = now_utc.tz_localize(None)
    if index is None or len(index) == 0:
        return now_utc, True
    if _raw_feed_offset(index, now_utc) is None:
        return now_utc, True               # stale feed -> plain UTC, unchanged
    offset, _state, confirmed = resolve_h1_feed_offset(
        index, now_utc, state_path=state_path, persist=persist)
    if offset is None:
        return now_utc, True
    return now_utc + pd.Timedelta(hours=int(offset)), confirmed


def infer_h1_feed_now(index, now_utc=None, state_path=H1_OFFSET_STATE_PATH,
                      persist=True):
    """
    "Now", expressed in the FEED'S OWN CLOCK, inferred from the feed itself.

    WHY THIS IS NECESSARY, and it is not optional. H1 bar labels are NOT UTC and
    NOT local time: `_fetch_h1_from_mt5` tags the raw MT5 epoch as UTC, but that
    epoch bakes in the BROKER SERVER's wall clock (the same fact the daily
    `drop_incomplete_bars` docstring records). Measured live against this feed:

        local wall clock   21:16      (machine, UTC+3)
        utcnow             18:16
        newest bar label   20:00      -> server clock is 20:16, i.e. UTC+2

    Comparing bar labels against `utcnow` makes the serving path TWO HOURS too
    conservative -- it bases the forecast on a bar that closed two hours ago and
    reports the already-closed bar as still open. Comparing against LOCAL time
    makes it one hour too permissive -- it treats the still-forming bar as
    closed, which is exactly the completed-bar bug at hourly resolution. Both
    were observed before this function existed. Neither wall clock is correct;
    only the feed's own clock is.

    THE INFERENCE, and its one failure mode. On a live feed the newest bar is the
    one currently forming, so feed_now lies in [newest, newest + 1h) and
    diff = newest - utcnow lies in (offset - 1, offset], whose ceiling is exactly
    `offset`. That premise silently assumes ZERO EMIT LAG. In the seconds after
    an hour boundary, before the broker has emitted the new bar, it is false and
    the ceiling yields offset - 1. `resolve_h1_feed_offset` handles that by
    making the offset sticky and requiring a change to be confirmed.

    STALE FEEDS degrade correctly: a weekend or dead feed puts the newest bar
    hours or days back, the inferred offset falls outside the plausible band, and
    this returns plain UTC (never a stale sticky value). The forecast bar is then
    far in the past and the caller correctly reports it as already closed rather
    than actionable.
    """
    feed_now, _confirmed = h1_feed_now_with_status(
        index, now_utc=now_utc, state_path=state_path, persist=persist)
    return feed_now


def drop_incomplete_h1_bars(ohlcv_df, now=None):
    """
    Keep only fully-closed, in-distribution HOURLY bars. The H1 analogue of
    ``drop_incomplete_bars`` above, and it exists for the same reason at a finer
    resolution: the H1 direction model is trained on *features from a COMPLETED
    bar -> direction of the NEXT bar*, so at serve time the base bar must be the
    last FULLY CLOSED hourly bar. Basing a forecast on the currently-forming hour
    would feed the model a bar whose high/low/close are still moving -- the
    hourly cousin of the Sunday-bar class of bug this project has already been
    bitten by once.

    Three kinds of bar are stripped:

      * The CURRENTLY-FORMING hourly bar. A fetch at 14:37 exposes a 14:00 bar
        that will not be final until 15:00. Dropped by comparing against
        ``now`` floored to the hour, so the 14:00 bar only survives once the
        clock has actually reached 15:00.
      * Any SATURDAY bar. The real feed contains none; a broker emitting one is
        out-of-distribution by construction.
      * Any SUNDAY bar BEFORE the weekly open (hour < ``H1_WEEKLY_OPEN_HOUR``).
        MT5 brokers sometimes emit a thin partial Sunday bar ahead of the true
        weekly open. The genuine weekly-open bars at 22:00/23:00 are KEPT --
        they are in the training distribution and dropping them would make
        serving inconsistent with training, which is the error this whole
        contract exists to prevent.

    ``now`` is injectable for tests and IS INTERPRETED IN THE FEED'S CLOCK. It
    defaults to ``infer_h1_feed_now(ohlcv_df.index)`` -- NOT to the local wall
    clock the daily function uses, because H1 bar labels carry the broker server
    clock and comparing them against either local time or UTC is wrong by hours
    in opposite directions (see ``infer_h1_feed_now``). Following the daily
    function's verified convention, a tz-aware index or ``now`` has its tz tag
    STRIPPED rather than converted, so the hour used for the comparisons stays
    identical to the server-labelled numbers the feed carries instead of silently
    shifting under a real conversion.

    NOTE on the weekly-open threshold: it is calibrated on the MT5-labelled cache.
    The yfinance fallback labels in true UTC and can sit 2-3h off, which would
    make the guard slightly conservative rather than unsafe -- it can only ever
    drop bars, never admit a forming one.
    """
    if ohlcv_df is None or len(ohlcv_df) == 0:
        return ohlcv_df
    now = infer_h1_feed_now(ohlcv_df.index) if now is None else pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_localize(None)
    current_hour = now.floor('h')

    idx = ohlcv_df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    closed = idx < current_hour
    not_saturday = idx.weekday != 5
    not_early_sunday = ~((idx.weekday == 6) & (idx.hour < H1_WEEKLY_OPEN_HOUR))
    return ohlcv_df[closed & not_saturday & not_early_sunday]


def _fetch_from_mt5(symbol: str, bars: int):
    """Try a live MT5 terminal session first. Returns an OHLCV DataFrame
    (tz-naive, ascending date index) or None if no terminal is reachable."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    from .mt5_coverage import assert_coverage, sync_symbol

    try:
        if not mt5.initialize():
            return None
        sync_symbol(mt5, symbol)          # see _fetch_h1_from_mt5 for why
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
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    # The daily path shares the same failure mode, so it gets the same guard.
    return assert_coverage(df, 'D1', label='MT5 %s D1' % symbol)


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

    from .mt5_coverage import assert_coverage, sync_symbol

    try:
        if not mt5.initialize():
            return None
        # PREVENTION: an unselected symbol yields a partially synced history
        # block, and copy_rates_from_pos then reaches further back to satisfy
        # `bars` -- so the count looks right and months are missing from the
        # middle. Select and force the pull BEFORE requesting bars.
        sync_symbol(mt5, symbol)
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
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    # DETECTION: raises rather than returning a holed frame. Deliberately NOT
    # caught by fetch_h1_market_data -- a holed frame must not overwrite the
    # good cache, and must not be silently replaced by the yfinance leg, which
    # would mix providers inside the production H1 history.
    return assert_coverage(df, 'H1', label='MT5 %s H1' % symbol)


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


def _fetch_m15_from_mt5(symbol: str, bars: int):
    """M15 (15-minute) OHLCV from a live MT5 terminal. Returns a UTC-indexed
    DataFrame (ascending) or None if no terminal is reachable. Mirrors
    _fetch_h1_from_mt5 but requests mt5.TIMEFRAME_M15 instead of H1, and via
    the SAME bar-count `copy_rates_from_pos` API (never tick-level data --
    M15 OHLC bars are sufficient for the harmonic-pattern M15 hypotheses,
    tick granularity is unneeded overhead)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    from .mt5_coverage import assert_coverage, sync_symbol

    try:
        if not mt5.initialize():
            return None
        sync_symbol(mt5, symbol)          # see _fetch_h1_from_mt5 for why
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars)
        mt5.shutdown()
    except Exception:
        return None

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    return assert_coverage(df, 'M15', label='MT5 %s M15' % symbol)


def fetch_m15_market_data(mt5_symbol: str = "EURUSD", bars: int = 350000,
                          cache_path: str = "results/eurusd_m15.csv"):
    """
    Fetch historical M15 (15-minute) EURUSD OHLCV for the harmonic-pattern
    M15 hypotheses (H1.5/H1.6). UNLIKE fetch_live_market_data/
    fetch_h1_market_data, MT5 is the SOLE live source here -- there is
    deliberately NO Yahoo Finance fallback for this timeframe (Yahoo's
    intraday history is short and inconsistent enough that mixing it into a
    would-be MT5-native swing/pattern hypothesis is not worth the
    provenance risk). An on-disk cache of previously-fetched MT5 bars remains
    an acceptable OFFLINE fallback -- same "never hard-fail" convention as
    every other fetch chain in this project -- but no other LIVE source is
    ever tried.

    Returns (dataframe, source_label) where source is "MT5", "cache", or
    (None, None) if MT5 is unreachable and no cache exists. The returned
    DataFrame always carries a UTC-localised DateTimeIndex.
    """
    df = _fetch_m15_from_mt5(mt5_symbol, bars)
    if df is not None and len(df) > 0:
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_csv(cache_path)
            except OSError:
                pass
        return df, "MT5"

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached.index = (
            cached.index.tz_localize('UTC') if cached.index.tz is None
            else cached.index.tz_convert('UTC')
        )
        return cached, "cache"

    return None, None


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
