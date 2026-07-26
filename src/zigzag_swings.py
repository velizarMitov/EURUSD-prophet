"""
Causal, walk-forward ATR-scaled ZigZag swing-pivot detector.

STATUS: NEW code, an ALTERNATIVE swing basis for the harmonic-pattern
hypotheses (direction/return hypotheses H1.3/H1.4,
results/harmonic_pattern_hypothesis_log.csv), tried alongside — not instead
of — the fixed-lag Williams-fractal basis (`src.fibonacci_fractals`,
`src.harmonic_patterns.detect_harmonic_events`).

WHY TRY A SECOND SWING BASIS
----------------------------
A Williams fractal's 5-bar window is FIXED-LENGTH regardless of how volatile
or quiet the market is. On H1 bars specifically, a fixed 5-bar window is
short enough that it likely flags a lot of noisy MICRO-swings — local wiggles
that are statistically "extrema" but not genuinely meaningful market
structure — which then get fed into XABCD ratio scoring as if they were real
swing points. A ZigZag whose reversal threshold ADAPTS to current volatility
(ATR-scaled) targets cleaner, more meaningful swings: it only confirms a
pivot once price has genuinely reversed by a volatility-relative amount, not
merely "more extreme than its 2 immediate neighbors on each side".

ELEVATED LOOK-AHEAD RISK — AND HOW IT IS MITIGATED
-----------------------------------------------------
A Williams fractal's confirmation lag is FIXED (exactly 2 bars,
`fibonacci_fractals.CONFIRMATION_LAG`) — trivial to reason about and to get
right. A ZigZag pivot's confirmation lag (`reveal_bar - idx`) is VARIABLE and
unbounded: a pivot can sit "unconfirmed" for a handful of bars or for
hundreds, depending entirely on how long price takes to reverse by the
ATR-scaled threshold. This is a genuinely easier algorithm to get wrong in a
way that REPAINTS (a classic ZigZag bug: compute local extrema over the WHOLE
series first, then apply the threshold — this uses full hindsight, because
"was this the local max" implicitly depends on bars that had not happened yet
at the time). This module is written to make that mistake structurally
impossible:

  * `zigzag_swings()` processes bars STRICTLY IN ORDER, one at a time. There
    is no "scan the whole array for extrema" step anywhere. The only
    array-level (vectorized) computation is the ATR itself, and ATR is
    ITSELF a purely causal/trailing statistic (pandas `.ewm`, matching
    `src.features.py`'s `ATR_14` formula) — `ATR[t]` depends only on bars
    `<= t` regardless of whether it is computed one row at a time in a loop
    or via a single vectorized pandas call; vectorizing a causal recurrence
    does not introduce look-ahead, only vectorizing a NON-causal one
    (searching for extrema across a window that includes future bars) would.
  * Every returned pivot carries BOTH its `idx` (the bar the extreme
    price actually occurred) AND its `reveal_bar` (the bar the reversal was
    DETECTED and the pivot was locked in) — the variable-length analogue of
    `CONFIRMATION_LAG`. Downstream code (`src.harmonic_patterns
    .detect_harmonic_events_from_pivots`) uses `reveal_bar`, never `idx`, to
    decide when a pivot — and any harmonic event built from it — first
    becomes usable.
  * A confirmed pivot's `(idx, level)` never changes after confirmation —
    tested explicitly (`test_zigzag_confirmed_pivot_is_stable_under_future_extension`,
    the direct repainting guard: appending more bars after a pivot's
    `reveal_bar` must reproduce that pivot identically).
  * A pivot is provably invisible to any query truncated before its own
    `reveal_bar` (`test_zigzag_pivot_invisible_before_reveal_bar`).

PRE-REGISTERED ALGORITHM (fixed before running on real data)
--------------------------------------------------------------
    threshold[t] = 1.5 * ATR(14)[t]   (this project's existing 1.5x
    multiplier convention — see src.triple_barrier's TARGET_MULTIPLIER — ATR
    computed causally, only past/current bars).

Maintain a running "candidate extreme" (price + bar index) and a search
direction (seeking a HIGH or seeking a LOW). At each new bar t:

  * seeking a HIGH: if `high[t] > candidate.level`, update the candidate to
    `(high[t], t)` — still UNCONFIRMED. Then, if
    `candidate.level - close[t] > threshold[t]`, CONFIRM the candidate as a
    HIGH pivot: `reveal_bar = t` (the bar the reversal was detected),
    `idx` = the candidate's own bar, `level` = the candidate's price. Flip
    search direction to seeking a LOW and seed a NEW candidate at
    `(low[t], t)`.
  * seeking a LOW: the symmetric mirror (`low[t] < candidate.level` extends
    the candidate; `close[t] - candidate.level > threshold[t]` confirms).

Bootstrap (bar 0): starts "seeking a HIGH" with `candidate = (high[0], 0)` —
an arbitrary but immaterial bootstrap choice, stated plainly rather than
hidden. If the market's true first genuine swing is actually a low, this
bootstrap candidate simply never confirms (nothing ever reverses far enough
below a high that was never a real extreme); the real alternating structure
emerges on its own once price genuinely reverses by more than its own local
threshold.
"""
import numpy as np
import pandas as pd

ZIGZAG_ATR_PERIOD = 14
ZIGZAG_ATR_MULT = 1.5   # this project's existing multiplier convention


def _atr14(high, low, close, period: int = ZIGZAG_ATR_PERIOD) -> np.ndarray:
    """Causal ATR (Wilder-style EWM smoothing), IDENTICAL formula to
    `src.features.compute_features`'s `ATR_14` column — same true-range
    definition, same `ewm(com=period-1, adjust=False)` — so this reuses the
    project's existing ATR convention rather than inventing a second one.
    Purely trailing: row t depends only on bars `<= t`."""
    high = pd.Series(np.asarray(high, dtype=float))
    low = pd.Series(np.asarray(low, dtype=float))
    close = pd.Series(np.asarray(close, dtype=float))
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(com=period - 1, adjust=False).mean().to_numpy()


def zigzag_swings(high, low, close, atr_mult: float = ZIGZAG_ATR_MULT,
                  atr_period: int = ZIGZAG_ATR_PERIOD) -> list:
    """CAUSAL, walk-forward ATR-scaled ZigZag pivot detector — see module
    docstring for the full algorithm and its look-ahead-safety argument.

    Returns a chronological list of CONFIRMED pivot dicts, strictly
    alternating 'H'/'L' by construction (each confirmation immediately flips
    the search direction — no separate "collapse a same-kind run" step is
    needed, unlike the independent per-bar fractal detector):

        {'idx': pivot_bar, 'kind': 'H' or 'L', 'level': pivot_price,
         'reveal_bar': reveal_bar}

    `reveal_bar >= idx` always; a pivot must never be treated as knowable
    before its own `reveal_bar`.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    if n == 0:
        return []

    threshold = atr_mult * _atr14(high, low, close, period=atr_period)

    pivots = []
    seeking = 'H'          # arbitrary bootstrap direction, see module docstring
    cand_level, cand_idx = high[0], 0

    for t in range(1, n):
        thr = threshold[t]
        if seeking == 'H':
            if high[t] > cand_level:
                cand_level, cand_idx = high[t], t
            if not np.isnan(thr) and (cand_level - close[t]) > thr:
                pivots.append({'idx': cand_idx, 'kind': 'H', 'level': cand_level,
                              'reveal_bar': t})
                seeking = 'L'
                cand_level, cand_idx = low[t], t
        else:
            if low[t] < cand_level:
                cand_level, cand_idx = low[t], t
            if not np.isnan(thr) and (close[t] - cand_level) > thr:
                pivots.append({'idx': cand_idx, 'kind': 'L', 'level': cand_level,
                              'reveal_bar': t})
                seeking = 'H'
                cand_level, cand_idx = high[t], t

    return pivots
