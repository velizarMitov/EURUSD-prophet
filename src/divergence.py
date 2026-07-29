"""
OSCILLATOR DIVERGENCE ON M15 — detection library (research-only).

Builds the event set for the divergence hypothesis family: converts the cached
M15 series to New York time, computes the three classical oscillators on the
CONTINUOUS series, pairs consecutive same-type ZigZag swings, classifies
divergence, and attaches features and forward returns.

The hypothesis program itself lives in src/divergence_check.py. This module
makes no decisions, writes no hypothesis log and computes no P&L.

THE TIMING RULE — the single most important thing in this module
----------------------------------------------------------------
An event's signal time is **reveal_bar(s2)**, never s2 itself. The swing at s2
is not knowable when it happens; the causal ZigZag only confirms it later.
Entry price, every feature value and the start of the forward-return window are
ALL taken at reveal_bar(s2). Measuring the forward move from P(s2) would be
look-ahead, and that single error is why divergence backtests well and trades
badly. `price_move_since_s2_pips` records exactly how much of the move is
already gone by the time the signal can honestly be acted on.

WHAT IS DECISION-BEARING
------------------------
REGULAR divergence only:
    REGULAR BULLISH  (on lows) : P(s2) < P(s1) AND O(s2) > O(s1) -> UP
    REGULAR BEARISH  (on highs): P(s2) > P(s1) AND O(s2) < O(s1) -> DOWN
HIDDEN divergence is detected and reported DESCRIPTIVELY only. Testing it would
be a separate pre-registered hypothesis with its own alpha, so no hidden result
may influence any verdict in this family.

P is the CLOSE at the swing bar, exactly as specified -- not the ZigZag's
high/low extreme (`level`), which is also carried on each event for reference.

REUSE, NOT REINVENTION
----------------------
  * swings          -> src/zigzag_swings.py, UNCHANGED (causal, ATR(14)x1.5,
                       each pivot carrying idx and reveal_bar, already covered
                       by the repainting-guard test). ATR is computed on M15
                       bars, so the threshold adapts to M15 scale by itself.
  * ATR             -> the same _atr14 the ZigZag thresholds on
  * RSI             -> src/pooled_h1_model._wilder_rsi, the project's existing
                       Wilder convention
  * broker -> NY    -> src/h1_newyork_time, including the empirically derived
                       mid-history rule change. Never re-derived, never assumed
                       to be a fixed offset.
"""

import os

import numpy as np
import pandas as pd

from src.h1_newyork_time import (
    BROKER_TZ, NY_TZ, FX_ROLL_HOUR_NY, PIP,
    determine_dst_rule, to_new_york, verify_conversion,
)
from src.pooled_h1_model import _wilder_rsi
from src.zigzag_swings import _atr14, zigzag_swings, ZIGZAG_ATR_MULT

M15_SOURCE_CSV = 'results/eurusd_m15.csv'
M15_NY_CSV = 'results/pooled_h1/EURUSD_m15_newyork.csv'
BAR_MINUTES = 15

# ── STEP 4: session restriction (on reveal_bar, never on s2) ──
SESSION_START_NY = 7        # inclusive
SESSION_END_NY = 10         # EXCLUSIVE -> entry hours are 07, 08, 09 NY
# The forward window is deliberately allowed to run past 10:00 into the
# 10:00-11:00 NY hour, so a signal fired at 09:50 still has an hour to resolve.

# ── STEP 5: horizons ──
PRIMARY_HORIZON = 4         # 4 M15 bars = 1 hour; DECISION-BEARING
CORROBORATING_HORIZONS = (8, 12)    # context only, never a second path to KEEP

# ── AMENDMENT: the SETUP is not the ENTRY (H_div.3 / H_div.4) ──
# A confirmed divergence is a SETUP. Entry requires a STRUCTURAL BREAK, and it
# is taken on CLOSED BARS ONLY:
#
#   BEARISH (swing HIGH at s2): trigger = first bar at or after reveal_bar(s2)
#       whose CLOSE is strictly BELOW low[s2]; the setup dies if any bar's HIGH
#       exceeds high[s2].
#   BULLISH (swing LOW at s2) : trigger = first bar at or after reveal_bar(s2)
#       whose CLOSE is strictly ABOVE high[s2]; the setup dies if any bar's LOW
#       falls below low[s2].
#
# Entry is the CLOSE of the triggering bar, never the break level itself. We
# hold bar data, not ticks, so assuming a fill at the exact level would be an
# intrabar assumption the data cannot support.
EXPIRY_BARS = 20            # fixed NOW; must not be tuned

OUTCOME_TRIGGERED = 'TRIGGERED'
OUTCOME_INVALIDATED = 'INVALIDATED'
OUTCOME_EXPIRED = 'EXPIRED'
SETUP_OUTCOMES = (OUTCOME_TRIGGERED, OUTCOME_INVALIDATED, OUTCOME_EXPIRED)

# ── STEP 6: chronological split on the M15 series ──
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85             # test = [0.85:1.0], RESERVED, never indexed

# ── STEP 2: PRIMARY, decision-bearing oscillator parameters ──
# The platform defaults: Wilder (1978), Appel (late 1970s), Lane. These are the
# out-of-the-box settings in MT5 and TradingView. If divergence carries an edge
# at all, the most plausible mechanism is coordination -- many participants
# watching the same lines -- and that only operates on the parameters people
# actually have on screen. A bespoke period would test a signal nobody sees.
#
# THE HONEST WEAKNESS, stated rather than hidden: these periods were designed
# for DAILY bars. RSI(14) meant 14 days; on M15 it is 3.5 hours. Applying them
# unchanged is a real assumption. There is no principled rescaling (x4 for
# hourly? x96 for daily?) and each candidate is a fresh researcher degree of
# freedom, so the assumption is probed by the declared band below rather than
# by a parameter search.
DEFAULT_RSI = 14
DEFAULT_MACD = (12, 26, 9)
DEFAULT_STOCH = (14, 3, 3)

PRIMARY_OSCILLATORS = {
    'rsi14': ('rsi', DEFAULT_RSI),
    'macd_12_26_9': ('macd', DEFAULT_MACD),
    'stoch_14_3_3': ('stoch', DEFAULT_STOCH),
}

# ── DECLARED SENSITIVITY BAND — fixed BEFORE execution, descriptive only ──
# Binding rules (STEP 2): the band consumes NO alpha and yields NO verdict; the
# H_div.1 verdict comes from the DEFAULT parameters alone; a band member
# clearing while the default DROPs is PARAMETER FRAGILITY, never a KEEP; no
# member may be added after seeing results.
SENSITIVITY_BAND = {
    'rsi': (7, 14, 21, 28),
    'macd': ((6, 13, 5), (12, 26, 9), (24, 52, 18)),
    'stoch': ((7, 3, 3), (14, 3, 3), (28, 3, 3)),
}
BAND_DECLARED_NOTE = (
    "SENSITIVITY BAND DECLARED BEFORE EXECUTION (RSI 7/14/21/28; MACD "
    "(6,13,5)/(12,26,9)/(24,52,18); Stochastic (7,3,3)/(14,3,3)/(28,3,3)). It is "
    "DESCRIPTIVE ONLY: it consumes no alpha and produces no verdict. The H_div.1 "
    "verdict comes from the DEFAULT platform parameters alone. A band member "
    "clearing while the default DROPs is PARAMETER FRAGILITY -- an effect selected "
    "by hindsight, not discovered -- and is never reported as a KEEP."
)

DIVERGENCE_TYPES = ('regular_bullish', 'regular_bearish',
                    'hidden_bullish', 'hidden_bearish')
REGULAR_TYPES = ('regular_bullish', 'regular_bearish')


class MissingCachedDataError(FileNotFoundError):
    """Raised when results/eurusd_m15.csv is absent. This program must never
    refetch from MT5 -- it STOPS and reports."""


# ───────────────────── STEP 1: M15 in New York time ───────────────────────────

def load_m15_server_frame(path: str = M15_SOURCE_CSV) -> pd.DataFrame:
    """
    Read the cached M15 OHLC and strip the FAKE UTC tag, leaving the raw broker
    wall-clock values those numbers actually are. The source file is read-only.
    """
    if not os.path.exists(path):
        raise MissingCachedDataError(
            f"Cached M15 data not found at {path!r}. This program must NOT "
            "refetch from MT5 -- STOP and report.")
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    out = df[['open', 'high', 'low', 'close']].copy()
    out.index = idx
    out.index.name = 'server_time'
    return out


def build_m15_newyork(path: str = M15_SOURCE_CSV, out_path: str = M15_NY_CSV,
                      write: bool = True, verbose: bool = True):
    """
    Convert the M15 series to New York time by REUSING the broker DST rule from
    src/h1_newyork_time -- including the mid-history change (US-DST tracking
    through 2016-10-30, EU-DST tracking from 2017-03-12). The rule is not
    assumed: `determine_dst_rule` re-establishes it from the M15 series' own
    weekly anchors, which also settles whether it holds over 2012-2015, a period
    the H1-derived rule never saw.

    Returns (ny_df, rule, verification_report).
    """
    server = load_m15_server_frame(path)
    rule = determine_dst_rule(server, verbose=verbose)
    ny = to_new_york(server, era_start=rule['eu_rule_era_start'])
    report = verify_conversion(server, ny, verbose=verbose)

    # Row count and OHLC byte-identity, asserted here too rather than trusted.
    assert len(ny) == len(server)
    for col in ('open', 'high', 'low', 'close'):
        assert np.array_equal(ny[col].to_numpy(), server[col].to_numpy())

    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        ny.to_csv(out_path)
    return ny, rule, report


def load_m15_newyork(path: str = M15_NY_CSV) -> pd.DataFrame:
    """Read the NY-time M15 frame. The index alternates -05:00/-04:00 across
    DST, so it must be parsed through UTC and converted back."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_convert(NY_TZ)
    df.index.name = 'ny_timestamp'
    return df.sort_index()


def contiguous_mask(index) -> np.ndarray:
    """True where the NEXT bar is exactly one M15 step later -- used to exclude
    forward windows that would jump a weekend or holiday gap."""
    idx = pd.DatetimeIndex(index)
    out = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        out[:-1] = np.diff(idx.to_numpy()) == np.timedelta64(BAR_MINUTES, 'm')
    return out


def window_is_contiguous(contig: np.ndarray, start: int, n: int) -> bool:
    """True iff bars start..start+n are an unbroken run of M15 bars."""
    if start < 0 or start + n >= len(contig) + 1:
        return False
    return bool(contig[start:start + n].all())


# ───────────────────── STEP 2: oscillators (continuous series) ────────────────

def compute_rsi(close: pd.Series, period: int = DEFAULT_RSI) -> pd.Series:
    """Wilder RSI, bounded 0-100, reusing the project's existing convention."""
    return _wilder_rsi(close, period)


def compute_macd(close: pd.Series, atr: pd.Series, params=DEFAULT_MACD):
    """
    MACD line (EMA_fast - EMA_slow), its signal, and the histogram.

    MACD is UNBOUNDED and scales with volatility, so the LINE is normalised by
    ATR14 before any cross-period comparison: `macd_norm = macd_line / ATR14`.
    That normalised series is what divergence is measured on -- comparing a raw
    2012 MACD level with a 2024 one would compare two different price scales.
    The histogram is returned as a secondary DESCRIPTIVE series only.
    """
    fast, slow, signal_period = params
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    return {
        'macd_line': macd_line,
        'macd_signal': signal,
        'macd_hist': macd_line - signal,          # secondary, descriptive
        'macd_norm': macd_line / atr,             # THE comparable series
    }


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       params=DEFAULT_STOCH):
    """
    Slow stochastic. Raw %K = 100*(close - LL(k)) / (HH(k) - LL(k)); the SLOW %K
    used here is the `smooth_k`-period SMA of raw %K; %D is the further
    `smooth_d`-period SMA, returned for reference. Bounded 0-100. A window with
    no range maps to the neutral 50.
    """
    k_period, smooth_k, smooth_d = params
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    rng = hh - ll
    raw_k = 100.0 * (close - ll) / rng
    raw_k = raw_k.where(rng != 0.0, 50.0)
    slow_k = raw_k.rolling(smooth_k).mean()
    return {'stoch_raw_k': raw_k, 'stoch_slow_k': slow_k,
            'stoch_slow_d': slow_k.rolling(smooth_d).mean()}


def oscillator_series(df: pd.DataFrame, kind: str, params) -> pd.Series:
    """
    The single comparable series for one oscillator configuration, computed on
    the CONTINUOUS M15 frame (never on a session-filtered subset -- an
    oscillator needs its full preceding history to mean anything).
    """
    close, high, low = df['close'], df['high'], df['low']
    if kind == 'rsi':
        return compute_rsi(close, params)
    if kind == 'macd':
        atr = pd.Series(_atr14(high, low, close), index=df.index)
        return compute_macd(close, atr, params)['macd_norm']
    if kind == 'stoch':
        return compute_stochastic(high, low, close, params)['stoch_slow_k']
    raise ValueError(f'unknown oscillator kind {kind!r}')


def oscillator_name(kind: str, params) -> str:
    """Stable identifier used in the event table and the breakdowns."""
    if kind == 'rsi':
        return f'rsi{params}'
    return f"{kind}_{'_'.join(str(p) for p in params)}"


# ───────────────────── STEP 3: swings ─────────────────────────────────────────

def detect_swings(df: pd.DataFrame, atr_mult: float = ZIGZAG_ATR_MULT) -> list:
    """Causal ZigZag pivots on the CONTINUOUS M15 series, imported unchanged.
    ATR(14) is computed on M15 bars, so the 1.5xATR threshold adapts to M15
    scale automatically -- no fixed-percentage threshold, no whole-array scan."""
    return zigzag_swings(df['high'], df['low'], df['close'], atr_mult=atr_mult)


# ───────────────────── STEP 0: divergence classification ──────────────────────

def classify(kind: str, p1: float, p2: float, o1: float, o2: float):
    """
    The formal specification, implemented exactly and with no variations.

      REGULAR BULLISH  (lows) : P2 < P1 AND O2 > O1  -> UP   (+1)
      REGULAR BEARISH  (highs): P2 > P1 AND O2 < O1  -> DOWN (-1)
      HIDDEN  BULLISH  (lows) : P2 > P1 AND O2 < O1  -> UP   (+1)
      HIDDEN  BEARISH  (highs): P2 < P1 AND O2 > O1  -> DOWN (-1)

    Anything else (either series flat, or both moving the same way) is not a
    divergence and returns (None, 0).
    """
    if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(o1) and np.isfinite(o2)):
        return None, 0
    if kind == 'L':
        if p2 < p1 and o2 > o1:
            return 'regular_bullish', +1
        if p2 > p1 and o2 < o1:
            return 'hidden_bullish', +1
        return None, 0
    if kind == 'H':
        if p2 > p1 and o2 < o1:
            return 'regular_bearish', -1
        if p2 < p1 and o2 > o1:
            return 'hidden_bearish', -1
        return None, 0
    return None, 0


def build_events(df: pd.DataFrame, pivots: list, osc: pd.Series,
                 osc_label: str, atr: np.ndarray) -> pd.DataFrame:
    """
    Pair every two CONSECUTIVE same-type confirmed swings and emit one row per
    divergence found. The causal ZigZag alternates H/L strictly, so consecutive
    same-type pivots are `pivots[i]` and `pivots[i+2]`.

    Every feature is evaluated at s1/s2 (both already in the past at signal
    time) or at reveal_bar(s2) itself -- never at a bar after the signal.
    """
    close = df['close'].to_numpy(dtype=float)
    osc_v = np.asarray(osc, dtype=float)
    n = len(close)
    rows = []

    for i in range(len(pivots) - 2):
        s1, s2 = pivots[i], pivots[i + 2]
        if s1['kind'] != s2['kind']:
            continue                       # defensive; alternation makes this dead
        i1, i2 = s1['idx'], s2['idx']
        reveal = s2['reveal_bar']
        if reveal >= n:
            continue
        p1, p2 = close[i1], close[i2]
        o1, o2 = osc_v[i1], osc_v[i2]
        dtype, direction = classify(s1['kind'], p1, p2, o1, o2)
        if dtype is None:
            continue

        gap = int(i2 - i1)
        atr2 = float(atr[i2])
        rows.append({
            'oscillator': osc_label, 'div_type': dtype, 'direction': direction,
            'is_regular': dtype in REGULAR_TYPES, 'swing_kind': s1['kind'],
            's1_idx': int(i1), 's2_idx': int(i2), 'reveal_idx': int(reveal),
            'p1': p1, 'p2': p2, 'o1': o1, 'o2': o2,
            's1_level': float(s1['level']), 's2_level': float(s2['level']),
            # ── STEP 0 continuous strength measures ──
            'price_slope_norm': ((p2 - p1) / (atr2 * gap)
                                 if atr2 > 0 and gap > 0 else np.nan),
            'osc_slope': ((o2 - o1) / gap) if gap > 0 else np.nan,
            'swing_gap_bars': gap,
            'confirm_lag_bars': int(reveal - i2),
            'osc_level_at_s1': o1, 'osc_level_at_s2': o2,
            # how much of the move is ALREADY GONE by the time it can be acted on
            'price_move_since_s2_pips': (close[reveal] - p2) / PIP,
            'entry_close': close[reveal],
        })

    events = pd.DataFrame(rows)
    if len(events):
        events['div_magnitude'] = (events['price_slope_norm'].abs()
                                   * events['osc_slope'].abs())
        # STEP 0 fixes `price_move_since_s2_pips` as the RAW signed move, and
        # that is the feature. But its median across a mix of bullish and
        # bearish events is ~0 by symmetry (price rises off a low, falls off a
        # high), which says nothing. This DIRECTIONAL version -- positive means
        # the move has already started going the signalled way -- is the one
        # that answers "how much is gone before it can honestly be acted on".
        # Reporting only; never a model feature (it is a linear function of one).
        events['price_given_up_pips'] = (events['direction']
                                         * events['price_move_since_s2_pips'])
    return events


def build_all_events(df: pd.DataFrame, pivots: list, configs: dict) -> pd.DataFrame:
    """Run `build_events` for each oscillator configuration and stack the
    results. Configs maps label -> (kind, params)."""
    atr = _atr14(df['high'], df['low'], df['close'])
    frames = []
    for label, (kind, params) in configs.items():
        osc = oscillator_series(df, kind, params)
        ev = build_events(df, pivots, osc, label, atr)
        if len(ev):
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ['reveal_idx', 'oscillator']).reset_index(drop=True)


# ───────────────────── STEP 4/5/6: session, target, split ─────────────────────

def attach_session(events: pd.DataFrame, ny_index) -> pd.DataFrame:
    """
    Tag each event with the NY clock AT ITS REVEAL BAR -- the signal time. The
    session filter selects on reveal_bar, never on s2: an event whose swing sat
    at 06:00 NY but was only confirmed at 07:30 NY is a 07:00-session signal,
    and one whose swing sat at 09:00 but was confirmed at 11:00 is not.
    """
    idx = pd.DatetimeIndex(ny_index)
    out = events.copy()
    rev = out['reveal_idx'].to_numpy()
    out['reveal_ts'] = idx[rev]
    out['reveal_ny_hour'] = idx[rev].hour
    out['s2_ny_hour'] = idx[out['s2_idx'].to_numpy()].hour
    out['in_session'] = ((out['reveal_ny_hour'] >= SESSION_START_NY)
                         & (out['reveal_ny_hour'] < SESSION_END_NY))
    return out


def attach_forward_returns(events: pd.DataFrame, df: pd.DataFrame,
                           horizons=(PRIMARY_HORIZON,) + CORROBORATING_HORIZONS,
                           test_start_idx: int = None) -> pd.DataFrame:
    """
    Signed forward return in the event's PREDICTED direction, measured from
    close[reveal_bar] -- never from P(s2).

        signed_return_pips = direction * (close[reveal+N] - close[reveal]) / PIP

    Windows crossing a weekend/holiday gap are EXCLUDED (NaN), never padded.
    Windows reaching into the reserved test block are excluded too.
    """
    close = df['close'].to_numpy(dtype=float)
    contig = contiguous_mask(df.index)
    n = len(close)
    out = events.copy()
    rev = out['reveal_idx'].to_numpy()
    direction = out['direction'].to_numpy()

    for h in horizons:
        vals = np.full(len(out), np.nan)
        ok = np.zeros(len(out), dtype=bool)
        for j, (r, d) in enumerate(zip(rev, direction)):
            end = r + h
            if end >= n:
                continue
            if not window_is_contiguous(contig, r, h):
                continue
            if test_start_idx is not None and end >= test_start_idx:
                continue
            vals[j] = d * (close[end] - close[r]) / PIP
            ok[j] = True
        out[f'signed_return_pips_n{h}'] = vals
        out[f'window_ok_n{h}'] = ok
    return out


def split_bounds(n_bars: int, train_frac: float = TRAIN_FRAC,
                 val_frac: float = VAL_FRAC):
    """Chronological bar-position boundaries: train [0:70%], validation
    [70%:85%], test [85%:100%] RESERVED."""
    return int(n_bars * train_frac), int(n_bars * val_frac)


def assign_slice(events: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Label each event by the slice its REVEAL BAR falls in."""
    train_end, val_end = split_bounds(n_bars)
    rev = events['reveal_idx'].to_numpy()
    out = events.copy()
    out['slice'] = np.where(rev < train_end, 'train',
                            np.where(rev < val_end, 'val', 'test'))
    return out


def build_event_table(df: pd.DataFrame, configs: dict = None,
                      horizons=(PRIMARY_HORIZON,) + CORROBORATING_HORIZONS):
    """
    End-to-end event construction on one continuous NY-time M15 frame:
    swings -> oscillators -> divergence -> session tag -> forward returns ->
    slice label. Returns (events, pivots, counts).
    """
    configs = PRIMARY_OSCILLATORS if configs is None else configs
    pivots = detect_swings(df)
    events = build_all_events(df, pivots, configs)
    counts = {'n_bars': int(len(df)), 'n_pivots': int(len(pivots)),
              'n_events_all_types': int(len(events))}
    if not len(events):
        return events, pivots, counts

    events = attach_session(events, df.index)
    train_end, val_end = split_bounds(len(df))
    events = attach_forward_returns(events, df, horizons=horizons,
                                    test_start_idx=val_end)
    events = assign_slice(events, len(df))

    regular = events['is_regular']
    counts.update({
        'n_events_raw': int(regular.sum()),                 # regular, pre-session
        'n_events_hidden': int((~regular).sum()),
        'n_events_after_session_filter': int((regular & events['in_session']).sum()),
        'n_events_dropped_by_session': int((regular & ~events['in_session']).sum()),
        'train_end_idx': int(train_end), 'val_end_idx': int(val_end),
    })
    return events, pivots, counts


# ────────── AMENDMENT: setup -> structural-break entry (H_div.3/.4) ──────────

def resolve_setup(high, low, close, contig, kind, s2, reveal, max_idx,
                  expiry_bars: int = EXPIRY_BARS):
    """
    Resolve ONE setup causally, bar by bar, on closed bars only. Every setup
    ends in exactly one of TRIGGERED / INVALIDATED / EXPIRED.

    ORDER OF CHECKS WITHIN A BAR -- invalidation FIRST, then trigger. On a bar
    that both takes out the swing level and closes through the break level, we
    cannot know from bar data which came first intrabar, so the setup is treated
    as INVALIDATED. That is the conservative choice: it removes the trade rather
    than awarding it. `both_on_same_bar` counts how often it mattered.

    A gap in the bar series before resolution ends the scan as EXPIRED (flagged
    `expiry_by_gap`). Hitting `max_idx` -- the reserved test block -- before the
    expiry window is exhausted leaves the setup UNRESOLVED, flagged
    `truncated_by_bound`, and such setups are excluded rather than being counted
    as expiries they may not be.
    """
    n = len(close)
    if kind == 'H':                     # bearish setup off a swing HIGH
        invalidation_level, trigger_level = high[s2], low[s2]
    else:                               # bullish setup off a swing LOW
        invalidation_level, trigger_level = low[s2], high[s2]

    hard_stop = n - 1 if max_idx is None else min(n - 1, int(max_idx) - 1)
    window_end = reveal + expiry_bars
    limit = min(window_end, hard_stop)

    t = reveal
    while t <= limit:
        if kind == 'H':
            invalid = high[t] > invalidation_level
            triggered = close[t] < trigger_level
        else:
            invalid = low[t] < invalidation_level
            triggered = close[t] > trigger_level

        if invalid:
            return {'outcome': OUTCOME_INVALIDATED, 'trigger_idx': -1,
                    'bars_reveal_to_trigger': -1, 'resolved_at': int(t),
                    'both_on_same_bar': bool(triggered),
                    'expiry_by_gap': False, 'truncated_by_bound': False}
        if triggered:
            return {'outcome': OUTCOME_TRIGGERED, 'trigger_idx': int(t),
                    'bars_reveal_to_trigger': int(t - reveal),
                    'resolved_at': int(t), 'both_on_same_bar': False,
                    'expiry_by_gap': False, 'truncated_by_bound': False}
        if t < n - 1 and not contig[t]:
            return {'outcome': OUTCOME_EXPIRED, 'trigger_idx': -1,
                    'bars_reveal_to_trigger': -1, 'resolved_at': int(t),
                    'both_on_same_bar': False, 'expiry_by_gap': True,
                    'truncated_by_bound': False}
        t += 1

    return {'outcome': OUTCOME_EXPIRED, 'trigger_idx': -1,
            'bars_reveal_to_trigger': -1, 'resolved_at': int(limit),
            'both_on_same_bar': False, 'expiry_by_gap': False,
            'truncated_by_bound': bool(limit < window_end)}


def resolve_setups(events: pd.DataFrame, df: pd.DataFrame, max_idx=None,
                   expiry_bars: int = EXPIRY_BARS) -> pd.DataFrame:
    """Attach the setup outcome, the trigger bar and the entry close to every
    confirmed divergence."""
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    contig = contiguous_mask(df.index)

    recs = [resolve_setup(high, low, close, contig, k, int(s2), int(rv),
                          max_idx, expiry_bars)
            for k, s2, rv in zip(events['swing_kind'], events['s2_idx'],
                                 events['reveal_idx'])]
    out = pd.concat([events.reset_index(drop=True),
                     pd.DataFrame(recs)], axis=1)
    trig = out['trigger_idx'].to_numpy()
    ok = trig >= 0
    out['trigger_close'] = np.where(ok, close[np.clip(trig, 0, len(close) - 1)], np.nan)
    # How much of the move is gone before an HONEST entry is possible: the
    # distance from P(s2) to the entry close, measured IN the signalled
    # direction, so bullish and bearish do not cancel.
    out['pips_given_up'] = np.where(
        ok, out['direction'] * (out['trigger_close'] - out['p2']) / PIP, np.nan)
    return out


def attach_trigger_session(events: pd.DataFrame, ny_index) -> pd.DataFrame:
    """Session membership for the TRIGGERED arm is decided by the TRIGGER bar's
    NY hour -- the bar the entry is actually taken on."""
    idx = pd.DatetimeIndex(ny_index)
    out = events.copy()
    trig = out['trigger_idx'].to_numpy()
    ok = trig >= 0
    hours = np.full(len(out), -1)
    hours[ok] = idx[trig[ok]].hour
    out['trigger_ny_hour'] = hours
    out['trigger_in_session'] = (ok & (hours >= SESSION_START_NY)
                                 & (hours < SESSION_END_NY))
    return out


def attach_trigger_returns(events: pd.DataFrame, df: pd.DataFrame,
                           horizons=(PRIMARY_HORIZON,) + CORROBORATING_HORIZONS,
                           test_start_idx: int = None) -> pd.DataFrame:
    """
    Signed forward return measured from the TRIGGER bar's close. Same gap and
    reserved-block exclusions as the pattern-only arm.

    INVALIDATED and EXPIRED setups carry no return: there was no trade. That is
    CAUSAL exclusion, not survivorship bias -- a live trader following the same
    rule would have been excluded identically, on the same bar, without knowing
    the future.
    """
    close = df['close'].to_numpy(dtype=float)
    contig = contiguous_mask(df.index)
    n = len(close)
    out = events.copy()
    trig = out['trigger_idx'].to_numpy()
    direction = out['direction'].to_numpy()

    for h in horizons:
        vals = np.full(len(out), np.nan)
        ok = np.zeros(len(out), dtype=bool)
        for j, (r, d) in enumerate(zip(trig, direction)):
            if r < 0:
                continue
            end = r + h
            if end >= n or not window_is_contiguous(contig, r, h):
                continue
            if test_start_idx is not None and end >= test_start_idx:
                continue
            vals[j] = d * (close[end] - close[r]) / PIP
            ok[j] = True
        out[f'trig_return_pips_n{h}'] = vals
        out[f'trig_window_ok_n{h}'] = ok
    return out


def build_triggered_table(events: pd.DataFrame, df: pd.DataFrame,
                          horizons=(PRIMARY_HORIZON,) + CORROBORATING_HORIZONS):
    """
    The H_div.3/.4 event set: every confirmed REGULAR divergence resolved into
    TRIGGERED / INVALIDATED / EXPIRED, with trigger-bar session membership,
    trigger-anchored forward returns and a slice label taken from the TRIGGER
    bar. The reserved test block bounds the resolution scan itself.
    """
    _train_end, val_end = split_bounds(len(df))
    setups = events[events['is_regular']].copy()
    setups = resolve_setups(setups, df, max_idx=val_end)
    setups = attach_trigger_session(setups, df.index)
    setups = attach_trigger_returns(setups, df, horizons=horizons,
                                    test_start_idx=val_end)

    train_end, val_end = split_bounds(len(df))
    trig = setups['trigger_idx'].to_numpy()
    setups['trigger_slice'] = np.where(
        trig < 0, 'none',
        np.where(trig < train_end, 'train',
                 np.where(trig < val_end, 'val', 'test')))
    return setups


def triggered_study_events(setups: pd.DataFrame, horizon: int = PRIMARY_HORIZON):
    """Setups that actually became trades: TRIGGERED, entered inside the
    session, with a usable forward window."""
    m = ((setups['outcome'] == OUTCOME_TRIGGERED)
         & ~setups['truncated_by_bound']
         & setups['trigger_in_session']
         & setups[f'trig_window_ok_n{horizon}'])
    return setups[m].copy()


def study_events(events: pd.DataFrame, horizon: int = PRIMARY_HORIZON,
                 regular_only: bool = True) -> pd.DataFrame:
    """
    The rows that actually enter a test: regular divergences, revealed inside
    the session, with a usable forward window at `horizon`.
    """
    m = events['in_session'] & events[f'window_ok_n{horizon}']
    if regular_only:
        m = m & events['is_regular']
    return events[m].copy()
