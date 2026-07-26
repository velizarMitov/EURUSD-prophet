"""
Generic triple-barrier event labeling (Lopez de Prado, "Advances in Financial
Machine Learning" ch.3) + the horizon-matched volatility scale it needs.

STATUS: NEW code, introduced for the H1 harmonic-pattern hypothesis
(results/harmonic_pattern_hypothesis_log.csv) but written event-source-agnostic
— nothing here is specific to harmonic patterns. Given an entry bar, a signed
direction, and a volatility-scaled target/stop distance, `triple_barrier_label`
walks FORWARD bar-by-bar and reports which barrier is touched FIRST:

  * the barrier ALIGNED with `direction` (the "target")           -> label 1
  * the barrier OPPOSITE `direction` (the "stop")                 -> label 0
  * neither, by the fixed TIME/vertical barrier                   -> label 1
    ONLY IF the signed return at the time barrier, in the predicted
    direction, clears a transaction-cost threshold (a move smaller than the
    round-trip spread is not a realizable win — mirrors how
    `src/paper_trading.py` already nets cost, never scoring a bare
    sign(>0) as a win) -- otherwise label 0.

SAME-BAR TIE-BREAK (both barriers touched within one bar's high/low range):
OHLC data cannot resolve true intrabar sequencing, so this ties toward the
STOP — the conservative, standard Lopez de Prado convention (never overstates
the edge by assuming the more favorable order happened).

NO LOOK-AHEAD: labeling walks strictly FORWARD from the entry bar (exclusive),
using only bars entry_idx+1 .. entry_idx+horizon_bars; entry/direction/
target/stop are computed from data available AT or BEFORE entry_idx and never
change during the walk. An event whose horizon would run past the end of the
available series is EXCLUDED (returns `(None, 'insufficient_history')`) —
never padded or estimated.

HORIZON-MATCHED VOLATILITY (why sqrt-time-scaled EWMA, not a plain ATR)
-------------------------------------------------------------------------
`ewma_log_return_std` is the EWMA standard deviation of per-bar log returns
(pandas' `.ewm(span=...).std()` — trailing/causal by construction, so it never
uses a future bar). A plain ATR (or this per-bar std alone) measures PER-BAR
range/dispersion, not the dispersion an entry should actually expect over its
full HOLDING HORIZON. `horizon_vol_from_ewma_std` applies the standard
square-root-of-time scaling (`sigma_horizon = sigma_per_bar * sqrt(horizon_bars)`,
the same scaling stochastic-process theory uses to project single-step
variance to a multi-step horizon) so the barrier distances are genuinely
matched to the 120-bar holding period the triple barrier actually spans.
"""
import numpy as np
import pandas as pd

TARGET_MULTIPLIER_DEFAULT = 1.5
STOP_MULTIPLIER_DEFAULT = 1.0


def ewma_log_return_std(close, span: int) -> np.ndarray:
    """Per-bar EWMA standard deviation of log returns (pandas `.ewm` — causal/
    trailing, so entry t only ever reflects bars <= t). Returned array is the
    same length as `close`; index 0 is NaN (no return is defined there), index
    1 is NaN (std of a single point is undefined), defined from index 2 on."""
    close = np.asarray(close, dtype=float)
    log_ret = np.log(close[1:] / close[:-1])
    ewm_std = pd.Series(log_ret).ewm(span=span, adjust=True).std().to_numpy()
    return np.concatenate([[np.nan], ewm_std])


def horizon_vol_from_ewma_std(r_ewma_std, horizon_bars: int) -> np.ndarray:
    """Square-root-of-time scale a per-bar EWMA return std to a holding
    horizon of `horizon_bars` bars: sigma_per_bar * sqrt(horizon_bars). This
    is the volatility measure genuinely matched to a fixed-horizon time
    barrier -- NOT a plain single-bar ATR, which measures per-bar range, not
    horizon-scaled dispersion."""
    return np.asarray(r_ewma_std, dtype=float) * np.sqrt(horizon_bars)


def triple_barrier_label(high, low, close, entry_idx: int, direction: int,
                         horizon_vol: float, horizon_bars: int, cost_price: float,
                         target_mult: float = TARGET_MULTIPLIER_DEFAULT,
                         stop_mult: float = STOP_MULTIPLIER_DEFAULT):
    """Label ONE event.

    `high`/`low`/`close` are full-length arrays of one OHLC series (positional
    integer indexing). `entry_idx` is the bar the position is opened AT — its
    own `close` is the entry price; the barriers are walked over bars
    `entry_idx+1 .. entry_idx+horizon_bars` (inclusive), i.e. exclusive of the
    entry bar itself. `direction` is +1 (long) or -1 (short). `horizon_vol` is
    the ALREADY sqrt-time-scaled volatility at the entry bar (see
    `horizon_vol_from_ewma_std`) -- this function does no scaling itself.
    `cost_price` is the raw-price transaction-cost threshold (e.g.
    `spread_pips * PIP_SIZE`) the time-barrier resolution must clear to be
    labeled a win.

    Returns `(label, outcome)` where `outcome` is one of
    `{'target', 'stop', 'time_win', 'time_loss'}`, or
    `(None, 'insufficient_history')` if `entry_idx + horizon_bars` exceeds the
    available series -- the caller MUST exclude such events, never pad/estimate.
    """
    n = len(close)
    if entry_idx + horizon_bars >= n:
        return None, 'insufficient_history'

    entry = float(close[entry_idx])
    target = entry * np.exp(direction * target_mult * horizon_vol)
    stop = entry * np.exp(-direction * stop_mult * horizon_vol)

    for k in range(entry_idx + 1, entry_idx + horizon_bars + 1):
        if direction > 0:
            hit_target = high[k] >= target
            hit_stop = low[k] <= stop
        else:
            hit_target = low[k] <= target
            hit_stop = high[k] >= stop
        if hit_stop:
            # Same-bar ambiguity resolves to the stop (conservative tie-break,
            # see module docstring) -- checked before target so a same-bar
            # double-touch never credits the more favorable ordering.
            return 0, 'stop'
        if hit_target:
            return 1, 'target'

    exit_price = float(close[entry_idx + horizon_bars])
    signed_move = direction * (exit_price - entry)
    if signed_move > cost_price:
        return 1, 'time_win'
    return 0, 'time_loss'
