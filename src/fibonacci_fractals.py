"""
Williams-fractal + Fibonacci features for the EURUSD direction/return model.

Genuinely different information from price momentum or the FRED/COT feeds: the
GEOMETRY of recent swing structure — where price sits relative to the last
confirmed swing high/low and the Fibonacci retracement grid drawn on the most
recent swing. Everything here is derived from OHLC alone (no external feed, so
no publish-lag surface like COT/FRED), but it carries its OWN, sharper
look-ahead hazard: a Williams fractal is only *knowable two bars after it
forms*.

    CONFIRMATION LAG IS THE WHOLE RISK
    ----------------------------------
    A Williams fractal uses a symmetric 5-bar window: a HIGH fractal at bar i is
    a strict maximum of high[i-2 .. i+2]; a LOW fractal is a strict minimum of
    low[i-2 .. i+2]. Because the definition reaches TWO bars into the future,
    the fractal at bar i cannot be confirmed until bar i+2. When building any
    feature for bar t we may therefore only use fractals with i <= t-2 — a
    fractal is INVISIBLE on bars i and i+1 and first becomes usable on bar i+2.
    The reveal walk below enforces this by "revealing" the fractal at index t-2
    exactly at step t; `tests/test_unit.py` asserts the invisibility on bars i,
    i+1 directly (mirroring the FRED/COT no-look-ahead tests).

    Every feature is NEUTRAL 0 / NaN-safe until a genuine confirmed structure
    exists (no confirmed fractal yet, or no genuine two-point swing), so the
    columns are fully defined on every modeled row — never NaN, never
    back-filled from a future fractal.

Hypothesis families (Production Methodology — ablation-first, validation-arbiter)
--------------------------------------------------------------------------------
HYPOTHESIS #7 (TESTED THIS PASS) — bundled retracement + breakout block, in the
EXISTING direction/return family (`results/feature_hypothesis_log.csv`, tested
via `src/ablation.py::run_addition_test`, the same block-addition convention as
the macro and COT blocks). Three views of one underlying "swing geometry" fact,
so ONE Bonferroni slot for the whole bundle; the bar tightens to 0.05/7 ≈ 0.0071
(this is the 7th feature hypothesis ever spent in this family):
    * fractal_breakout_up    — 1 iff close[t] > most recent CONFIRMED high-fractal level
    * fractal_breakout_down  — 1 iff close[t] < most recent CONFIRMED low-fractal level
    * dist_to_nearest_fib_pct — signed distance from close[t] to the nearest
      Fibonacci retracement level of the most recent confirmed 2-point swing,
      normalized by the swing range (H-L) and expressed in percent.

HYPOTHESIS #8 (BUILT NOW, TESTED CONDITIONALLY — NOT run in this pass) —
Fibonacci extension/projection from a confirmed 3-point swing A -> B -> C (three
consecutive alternating confirmed fractals, chronological order preserved so the
projection direction is unambiguous):
    * dist_to_nearest_fib_extension_pct — same normalization style as the
      retracement feature, over the extension grid projected off C.

    PRE-REGISTERED CONTINGENCY: Hypothesis #8 (extension/projection) runs ONLY
    if hypothesis #7 (retracement + fractal breakout) clears its bar. If #7 is
    DROP, #8 is not tested at all — a more discretionary, 3-point-dependent
    feature is not worth spending hypothesis budget on when the simpler 2-point
    version already failed. This contingency is recorded in
    `feature_hypothesis_log.csv`'s notes for hypothesis #7 and in
    IMPROVEMENT_LOG.md, so the decision is traceable even if #8 is never run.

Nothing here is wired into FEATURE_COLUMNS directly — ablation-first discipline,
exactly like every other candidate feature in this project.
"""
import numpy as np
import pandas as pd

# Williams 5-bar fractal: strict extremum of the symmetric window [i-2, i+2].
FRACTAL_WINDOW = 5
# A fractal at bar i reaches 2 bars into the future, so it is only CONFIRMED
# (knowable) at bar i+2. This constant is the entire look-ahead guard.
CONFIRMATION_LAG = 2

# Fibonacci retracement ratios (hypothesis #7) and extension/projection ratios
# (hypothesis #8) — the standard grids.
FIB_RETRACEMENT_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_EXTENSION_LEVELS = (1.13, 1.272, 1.618, 2.0, 2.618)

# Hypothesis #7 — the bundle tested this pass (one Bonferroni slot).
FIBONACCI_FEATURE_COLUMNS = [
    'fractal_breakout_up', 'fractal_breakout_down', 'dist_to_nearest_fib_pct',
]
# Hypothesis #8 — built now, tested only if #7 clears its bar (see docstring).
FIBONACCI_EXTENSION_FEATURE_COLUMNS = ['dist_to_nearest_fib_extension_pct']


# ── fractal detection (Williams, 5-bar) ──────────────────────────────────────

def detect_fractals(high, low):
    """Return (high_fractal, low_fractal) boolean arrays over the bar index.

    `high_fractal[i]` is True iff high[i] is a STRICT maximum of high[i-2:i+3];
    `low_fractal[i]` is True iff low[i] is a STRICT minimum of low[i-2:i+3].
    Strict inequalities avoid flagging a flat run as a fractal. The first and
    last two bars can never be fractals (the window would run off the ends).

    NOTE: this function looks at i+1, i+2 by definition — it is the raw
    geometric detector. The look-ahead guard lives in the CALLERS, which only
    ever consume a fractal from bar i+2 onward (see CONFIRMATION_LAG).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    hf = np.zeros(n, dtype=bool)
    lf = np.zeros(n, dtype=bool)
    for i in range(CONFIRMATION_LAG, n - CONFIRMATION_LAG):
        h = high[i]
        if h > high[i - 2] and h > high[i - 1] and h > high[i + 1] and h > high[i + 2]:
            hf[i] = True
        lo = low[i]
        if lo < low[i - 2] and lo < low[i - 1] and lo < low[i + 1] and lo < low[i + 2]:
            lf[i] = True
    return hf, lf


def confirmed_high_low_levels(high, low):
    """Per-bar arrays of the most recent CONFIRMED high- and low-fractal level.

    `conf_high[t]` is the price level of the latest high fractal knowable by bar
    t (i.e. a fractal at index i with i <= t-2), NaN before any is confirmed;
    `conf_low[t]` symmetric on lows. This is the reveal walk that makes the
    confirmation lag explicit: the fractal at index t-2 is revealed exactly at
    step t, so bars i and i+1 never see the fractal forming at bar i.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    hf, lf = detect_fractals(high, low)
    conf_high = np.full(n, np.nan)
    conf_low = np.full(n, np.nan)
    last_high = np.nan
    last_low = np.nan
    for t in range(n):
        j = t - CONFIRMATION_LAG           # the fractal that becomes confirmed AT bar t
        if j >= 0:
            if hf[j]:
                last_high = high[j]
            if lf[j]:
                last_low = low[j]
        conf_high[t] = last_high
        conf_low[t] = last_low
    return conf_high, conf_low


# ── alternating swing (zigzag) construction ──────────────────────────────────

def _push_swing(swings, idx, kind, level):
    """Append a confirmed fractal to the alternating swing sequence.

    A run of same-type fractals (two highs with no intervening low) is collapsed
    to the MORE EXTREME one — the standard zigzag construction — so the sequence
    strictly alternates H, L, H, L. Indices stay non-decreasing, so chronology
    (needed for the unambiguous A->B->C projection direction of hypothesis #8)
    is preserved.
    """
    if swings and swings[-1]['kind'] == kind:
        prev = swings[-1]['level']
        if (kind == 'H' and level > prev) or (kind == 'L' and level < prev):
            swings[-1] = {'idx': idx, 'kind': kind, 'level': level}
        return
    swings.append({'idx': idx, 'kind': kind, 'level': level})


def _retracement_dist(a, b, close):
    """Signed, range-normalized distance (percent) from `close` to the nearest
    Fibonacci retracement level of the most recent 2-point swing (a older, b
    newer, strictly alternating). Neutral 0 until a genuine swing exists."""
    if b['kind'] == 'H' and a['kind'] == 'L':          # uptrend swing: L -> H
        high_lvl, low_lvl = b['level'], a['level']
        rng = high_lvl - low_lvl
        if rng <= 0:
            return 0.0
        levels = [high_lvl - r * rng for r in FIB_RETRACEMENT_LEVELS]
    elif b['kind'] == 'L' and a['kind'] == 'H':        # downtrend swing: H -> L
        high_lvl, low_lvl = a['level'], b['level']
        rng = high_lvl - low_lvl
        if rng <= 0:
            return 0.0
        levels = [low_lvl + r * rng for r in FIB_RETRACEMENT_LEVELS]
    else:
        return 0.0                                      # not alternating (defensive)
    nearest = min(levels, key=lambda x: abs(close - x))
    return (close - nearest) / rng * 100.0


def _extension_dist(a, b, c, close):
    """Signed, range-normalized distance (percent) from `close` to the nearest
    Fibonacci EXTENSION level projected off a confirmed 3-point swing
    A -> B -> C (chronological). level(r) = C + r*(B - A); normalized by the
    projected swing span |B - A|. Neutral 0 for a degenerate (zero-span) swing.
    """
    span = b['level'] - a['level']
    if span == 0:
        return 0.0
    levels = [c['level'] + r * span for r in FIB_EXTENSION_LEVELS]
    nearest = min(levels, key=lambda x: abs(close - x))
    return (close - nearest) / abs(span) * 100.0


def _swing_walk(high, low, close):
    """Single reveal-and-swing pass. Returns (dist_fib, dist_ext) per-bar arrays.

    At each bar t the fractal confirmed at index t-2 is pushed into the
    alternating swing sequence, so every feature reads only structure that was
    knowable by t. Retracement needs the last 2 alternating swings; extension
    the last 3 (A->B->C in chronological order).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    hf, lf = detect_fractals(high, low)
    dist_fib = np.zeros(n)
    dist_ext = np.zeros(n)
    swings = []
    for t in range(n):
        j = t - CONFIRMATION_LAG
        if j >= 0:
            if hf[j]:
                _push_swing(swings, j, 'H', high[j])
            if lf[j]:
                _push_swing(swings, j, 'L', low[j])
        c = close[t]
        if len(swings) >= 2:
            dist_fib[t] = _retracement_dist(swings[-2], swings[-1], c)
        if len(swings) >= 3:
            dist_ext[t] = _extension_dist(swings[-3], swings[-2], swings[-1], c)
    return dist_fib, dist_ext


# ── public feature joins (add columns onto a daily OHLC frame) ────────────────

def add_fibonacci_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the hypothesis-#7 bundle (fractal_breakout_up / fractal_breakout_down
    / dist_to_nearest_fib_pct) onto a daily OHLC-indexed frame.

    Look-ahead-safe by construction: breakouts compare close to the most recent
    CONFIRMED fractal level (revealed at i+2), and the retracement distance uses
    only confirmed swings. Every column is fully defined (neutral 0 where no
    confirmed structure exists yet) — never NaN — so the ablation harness's
    "extra candidate columns must be fully defined" assert always holds.
    """
    out = df.copy()
    conf_high, conf_low = confirmed_high_low_levels(df['high'].to_numpy(float),
                                                    df['low'].to_numpy(float))
    close = df['close'].to_numpy(float)
    out['fractal_breakout_up'] = ((~np.isnan(conf_high)) & (close > conf_high)).astype(float)
    out['fractal_breakout_down'] = ((~np.isnan(conf_low)) & (close < conf_low)).astype(float)
    dist_fib, _dist_ext = _swing_walk(df['high'].to_numpy(float),
                                      df['low'].to_numpy(float), close)
    out['dist_to_nearest_fib_pct'] = dist_fib
    return out


def add_fibonacci_extension_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the hypothesis-#8 candidate (dist_to_nearest_fib_extension_pct) onto a
    daily OHLC frame. BUILT and unit-tested now, but NOT wired into ablation nor
    spent as a hypothesis this pass — see the module docstring's pre-registered
    contingency (runs only if #7 clears its bar). Same confirmation-lag guard as
    #7, now over a 3-point (more discretionary) swing. Neutral 0 / NaN-safe."""
    out = df.copy()
    _dist_fib, dist_ext = _swing_walk(df['high'].to_numpy(float),
                                      df['low'].to_numpy(float),
                                      df['close'].to_numpy(float))
    out['dist_to_nearest_fib_extension_pct'] = dist_ext
    return out


if __name__ == '__main__':
    from src.ablation import run_addition_test
    # Hypothesis #7: the retracement + breakout bundle as ONE hypothesis.
    run_addition_test('fibonacci_retracement_block', FIBONACCI_FEATURE_COLUMNS)
