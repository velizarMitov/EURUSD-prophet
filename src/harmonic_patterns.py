"""
XABCD harmonic-pattern detection + ratio scoring.

STATUS: NEW, UNVALIDATED code (added 2026-07-26 for the H1 harmonic-pattern
event-conditional hypothesis, results/harmonic_pattern_hypothesis_log.csv).
Despite the module docstring style matching this project's other reused
components, this module itself has NO prior track record here — it is being
introduced and tested for the first time by that hypothesis. It does not
implement fractal detection or swing construction itself: those are imported
UNCHANGED from `src.fibonacci_fractals` (`detect_fractals`, `_push_swing`,
`CONFIRMATION_LAG`) — the same look-ahead-safe primitives already validated
(via unit tests, not a KEEP hypothesis) for the daily fibonacci-retracement
feature. This module is timeframe-agnostic: it consumes whatever high/low/
close arrays it is given (daily OR H1); the H1 harmonic-pattern hypothesis
feeds it H1 bars.

WHAT AN XABCD PATTERN IS
-------------------------
A classical ("harmonic trading" / Scott Carney) reversal pattern drawn over
FIVE alternating swing points X, A, B, C, D (chronological, strictly
alternating high/low, exactly the same zigzag construction
`src.fibonacci_fractals` already uses for 2- and 3-point swings, extended to
five). Four ratios describe the pattern's geometry, each normalized against
the PRECEDING leg (the standard convention used by public harmonic-pattern
scanners, matching this module's r_AB/r_BC/r_CD/r_AD feature names):

    r_AB = |AB| / |XA|   (how far B retraces the XA leg)
    r_BC = |BC| / |AB|   (how far C retraces the AB leg)
    r_CD = |CD| / |BC|   (how far D extends the BC leg)
    r_AD = |AD| / |XA|   (the pattern's defining "completion" ratio — how far
                          the whole A->D move sits relative to XA; this is
                          the ratio most often quoted alone to name a pattern,
                          e.g. "0.786 Gartley")

Four classical patterns are scored (published Fibonacci ratio bands — NOT
tuned against this project's data): Gartley, Bat, Butterfly, Crab. Each
candidate 5-point swing is scored against all four templates; the BEST-fitting
template's score becomes `best_fit_score` (in [0, 1], 1.0 = ratios land
exactly inside every template band). `harmonic_pattern_score_signed =
best_fit_score * direction`, where `direction` is +1 (bullish — D is a LOW,
predicting an upward reversal) or -1 (bearish — D is a HIGH, predicting a
downward reversal); D's kind alone determines the sign, by construction of
the alternating swing (X, B, D share one kind; A, C the other).

LOOK-AHEAD DISCIPLINE (inherited, not reinvented)
--------------------------------------------------
Exactly `src.fibonacci_fractals`' CONFIRMATION_LAG=2 rule: a fractal at bar i
is only knowable at bar i+2. An XABCD pattern's D point is itself a fractal,
so the WHOLE PATTERN — and every ratio/score/direction derived from it — is
only confirmed at `D_idx + CONFIRMATION_LAG`. `detect_harmonic_events` reports
this explicitly as `confirmed_at_idx`; nothing about an event may be consumed
by any bar before that index. Because the swing-reveal walk below only ever
pushes a fractal at index j into the swing list at bar j+CONFIRMATION_LAG (the
exact moment it becomes knowable), and because `_push_swing`'s "collapse a
same-kind run to the more extreme point" rule only ever touches the LATEST
swing (never rewrites X/A/B/C once a later D has been appended), an event's
X/A/B/C/D levels are permanently fixed at the moment it is emitted — later
bars can supersede D with a more extreme point of the same kind (a fresh,
separately-confirmed event), but never retroactively alter an already-emitted
event's geometry.
"""
import numpy as np
import pandas as pd

from src.fibonacci_fractals import detect_fractals, _push_swing, CONFIRMATION_LAG

# Published Fibonacci ratio bands per pattern (Carney "harmonic trading"
# convention). NOT fit/tuned against this project's data.
HARMONIC_PATTERNS = {
    'gartley':   {'r_AB': (0.58, 0.66),  'r_BC': (0.382, 0.886), 'r_CD': (1.13, 1.618), 'r_AD': (0.75, 0.82)},
    'bat':       {'r_AB': (0.382, 0.50), 'r_BC': (0.382, 0.886), 'r_CD': (1.618, 2.618), 'r_AD': (0.85, 0.92)},
    'butterfly': {'r_AB': (0.75, 0.82),  'r_BC': (0.382, 0.886), 'r_CD': (1.618, 2.618), 'r_AD': (1.20, 1.618)},
    'crab':      {'r_AB': (0.382, 0.618), 'r_BC': (0.382, 0.886), 'r_CD': (2.24, 3.618), 'r_AD': (1.55, 1.68)},
}

HARMONIC_EVENT_COLUMNS = [
    'X_idx', 'A_idx', 'B_idx', 'C_idx', 'D_idx', 'confirmed_at_idx',
    'pattern', 'best_fit_score', 'r_AB', 'r_BC', 'r_CD', 'r_AD',
    'direction', 'harmonic_pattern_score_signed',
    'xa_amplitude', 'swing_duration_bars',
]


def _ratio_score(r, lo, hi):
    """1.0 inside the ideal [lo, hi] band; decays LINEARLY to 0.0 over an
    additional half-band-width beyond either edge (so a ratio just outside the
    published band still scores partial credit, but one a full band-width
    away scores 0). Symmetric, simple, and deterministic — easy to unit-test."""
    if lo <= r <= hi:
        return 1.0
    half_width = 0.5 * (hi - lo)
    if half_width <= 0:
        return 0.0
    dist = (lo - r) if r < lo else (r - hi)
    return max(0.0, 1.0 - dist / half_width)


def score_xabcd(X: dict, A: dict, B: dict, C: dict, D: dict):
    """Score one candidate 5-point swing (each a dict with 'idx'/'kind'/'level',
    strictly alternating kind, chronological idx) against every template in
    HARMONIC_PATTERNS. Returns a dict of the BEST-fitting template's name +
    score + the four raw ratios + direction, or None for a degenerate swing
    (a zero-length leg makes every ratio undefined)."""
    XA = abs(A['level'] - X['level'])
    AB = abs(B['level'] - A['level'])
    BC = abs(C['level'] - B['level'])
    CD = abs(D['level'] - C['level'])
    AD = abs(D['level'] - A['level'])
    if XA == 0 or AB == 0 or BC == 0:
        return None

    r_AB, r_BC, r_CD, r_AD = AB / XA, BC / AB, CD / BC, AD / XA
    direction = 1 if D['kind'] == 'L' else -1   # D is a LOW -> bullish reversal up

    best_name, best_score = None, -1.0
    for name, ratios in HARMONIC_PATTERNS.items():
        s = float(np.mean([
            _ratio_score(r_AB, *ratios['r_AB']),
            _ratio_score(r_BC, *ratios['r_BC']),
            _ratio_score(r_CD, *ratios['r_CD']),
            _ratio_score(r_AD, *ratios['r_AD']),
        ]))
        if s > best_score:
            best_name, best_score = name, s

    return {
        'pattern': best_name, 'best_fit_score': best_score,
        'r_AB': r_AB, 'r_BC': r_BC, 'r_CD': r_CD, 'r_AD': r_AD,
        'direction': direction, 'xa_amplitude': XA,
    }


def _harmonic_swing_walk(high, low):
    """Bar-by-bar reveal walk — mirrors `src.fibonacci_fractals._swing_walk`'s
    geometry exactly, extended from 2-3 points to a 5-point XABCD window.

    At bar t, the fractal confirmed at index t-CONFIRMATION_LAG (the one that
    just became knowable) is pushed into the alternating swing list via the
    UNCHANGED `_push_swing`. Whenever the list holds >= 5 swings, the most
    recent five (X, A, B, C, D chronological) are scored as one candidate
    event. Each event is emitted at most ONCE per distinct D `idx` (a run of
    same-kind fractals collapses to the most extreme one via `_push_swing`;
    only a genuinely NEW extreme, or a genuinely new opposite-kind fractal,
    produces a new D worth re-scoring).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    hf, lf = detect_fractals(high, low)

    swings = []
    seen_d_idx = set()
    events = []
    for t in range(n):
        j = t - CONFIRMATION_LAG
        if j >= 0:
            if hf[j]:
                _push_swing(swings, j, 'H', high[j])
            if lf[j]:
                _push_swing(swings, j, 'L', low[j])
        if len(swings) >= 5:
            X, A, B, C, D = swings[-5], swings[-4], swings[-3], swings[-2], swings[-1]
            if D['idx'] in seen_d_idx:
                continue
            seen_d_idx.add(D['idx'])
            res = score_xabcd(X, A, B, C, D)
            if res is not None:
                events.append({
                    **res,
                    'X_idx': X['idx'], 'A_idx': A['idx'], 'B_idx': B['idx'],
                    'C_idx': C['idx'], 'D_idx': D['idx'],
                    'confirmed_at_idx': D['idx'] + CONFIRMATION_LAG,
                    'swing_duration_bars': D['idx'] - X['idx'],
                })
    return events


def detect_harmonic_events(df: pd.DataFrame) -> pd.DataFrame:
    """Detect every completed XABCD swing in `df` (must carry 'high'/'low'
    columns; positional integer bar index — callers map back to `df.index` via
    `.iloc` on the *_idx columns). One row per completed pattern
    (`HARMONIC_EVENT_COLUMNS`); empty frame (still carrying the columns) if
    none are found. Look-ahead-safe by construction — see module docstring;
    `confirmed_at_idx` is the first bar an event may be used by."""
    events = _harmonic_swing_walk(df['high'].to_numpy(dtype=float),
                                  df['low'].to_numpy(dtype=float))
    if not events:
        return pd.DataFrame(columns=HARMONIC_EVENT_COLUMNS)
    out = pd.DataFrame(events)
    out['harmonic_pattern_score_signed'] = out['direction'] * out['best_fit_score']
    return out[HARMONIC_EVENT_COLUMNS].reset_index(drop=True)
