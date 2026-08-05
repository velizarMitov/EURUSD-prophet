"""Turn Kronos's sampled paths into p_up, the native forecast object.

WHY p_up AND NOT A DIRECTION. Kronos emits a DISTRIBUTION over future price
paths. `direction` is a lossy reduction of it and is provided only because the
ledger needs a binary to settle; `p_up` is the headline field.

WHY THE PATHS ARE DRAWN THIS WAY. KronosPredictor.predict averages its
sample_count paths internally (auto_regressive_inference ends with
`np.mean(preds, axis=1)`), so the individual paths -- the only thing from which
p_up can be computed -- are not reachable through it. Rather than vendor a
modified copy of that function, the same window is submitted to the PUBLIC
predict_batch N times at sample_count=1, which draws N independent paths.
"""

import numpy as np
import pandas as pd

from .loader import (CONTEXT_BARS, PRED_LEN, SAMPLE_COUNT, T, TOP_P, TRAILING_BARS)

# Sunday bars before this hour are not real trading bars on this feed; matches
# live_data.H1_WEEKLY_OPEN_HOUR, which drop_incomplete_h1_bars already enforces.
WEEKLY_OPEN_HOUR = 22

PRICE_COLS = ['open', 'high', 'low', 'close']
WEEKEND_GAP_HOURS = 2.0
RECENT_CONTEXT_BARS = 24
# 8 GB card: ~1.26 GB at 30 series; 90 leaves headroom beside the daily models.
MAX_BATCH_SERIES = 90


def context_has_weekend_gap(index, recent_bars: int = RECENT_CONTEXT_BARS) -> bool:
    """True when a >2h discontinuity sits in the most recent `recent_bars` of the
    context, i.e. the model's recent history crosses a weekend or holiday break.

    NOTE this is deliberately NOT the definition used in the clean-window
    evaluation. There, the split is on the HORIZON step (last context bar ->
    forecast bar), which is the sharp test of the uniform-spacing assumption. In
    live serving that step is contiguous by construction -- a prediction is only
    served while the forecast hour is open -- so the informative live quantity is
    whether the recent context is discontinuous. Both are reported; they are not
    interchangeable.
    """
    idx = pd.DatetimeIndex(index)[-(recent_bars + 1):]
    if len(idx) < 2:
        return False
    gaps = idx.to_series().diff().dropna().dt.total_seconds() / 3600.0
    return bool((gaps > WEEKEND_GAP_HOURS).any())


def forward_bar_index(start, n_bars: int) -> pd.DatetimeIndex:
    """The next `n_bars` REAL H1 trading bars from `start` inclusive.

    A plain ``date_range(freq='h')`` would hand the model 48 weekend hours that
    do not exist on this feed. The clean-window evaluation passed the ACTUAL
    future bar timestamps, so serving has to project the same calendar or the
    time features differ from what was measured. Skips Saturdays and Sundays
    before the weekly open, matching drop_incomplete_h1_bars.
    """
    out, t = [], pd.Timestamp(start)
    while len(out) < n_bars:
        if not (t.dayofweek == 5 or (t.dayofweek == 6 and t.hour < WEEKLY_OPEN_HOUR)):
            out.append(t)
        t = t + pd.Timedelta(hours=1)
    return pd.DatetimeIndex(out)


def realised_vol(closes_with_anchor) -> np.ndarray:
    """Paper eq.12: sqrt of summed squared log returns, over an array whose FIRST
    element is the anchor close. Anchoring means predicted, realised and trailing
    all count exactly `pred_len` returns -- the step from the last actual close
    into the first forecast bar is included on every side."""
    a = np.asarray(closes_with_anchor, dtype=float)
    lr = np.diff(np.log(np.maximum(a, 1e-12)), axis=-1)
    return np.sqrt((lr ** 2).sum(axis=-1))


def apply_isotonic(p, calibration: dict) -> float:
    """The FROZEN monotone map, evaluated by interpolation over persisted knots.
    Never refits, and cannot execute code from the calibration file.

    The result is clipped to the persisted support bound. Without it the top
    knot -- which maps to exactly 1.0 on the strength of a SINGLE fit-window row
    -- would let the endpoint serve a probability of 1.000, a claim of certainty
    from one observation. The clip is monotone, so it cannot change the ranking
    or the AUC; it only stops the levels claiming more resolution than n
    supports. See calibration['output_clip'].
    """
    iso = calibration['isotonic']
    q = float(np.interp(float(p), iso['x'], iso['y']))
    clip = calibration.get('output_clip')
    if clip:
        q = min(max(q, float(clip['lo'])), float(clip['hi']))
    return q


def volatility_from_paths(paths: np.ndarray, last_close: float,
                          trailing_closes, calibration: dict) -> dict:
    """The two numbers the authors' demo publishes, plus our labelled corrections.

    `paths` is (n_paths, pred_len) predicted closes; `trailing_closes` is the last
    TRAILING_BARS+1 actual closes ending at `last_close`, so the trailing
    reference is measured over exactly TRAILING_BARS returns like everything else.

    Every OUR-correction field is named as such: the scale constant and the
    isotonic map are ours, fitted once on the clean window, not model output.
    """
    p = np.asarray(paths, dtype=float)
    anchored = np.concatenate([np.full((len(p), 1), float(last_close)), p], axis=1)
    path_rv = realised_vol(anchored)                       # (n_paths,)
    trail = np.asarray(trailing_closes, dtype=float)
    if len(trail) < TRAILING_BARS + 1:
        raise RuntimeError('need %d trailing closes for the reference, got %d'
                           % (TRAILING_BARS + 1, len(trail)))
    trail_rv = float(realised_vol(trail[-(TRAILING_BARS + 1):]))
    scale = float(calibration['scale']['constant'])
    pred_rv = float(path_rv.mean())
    p_raw = float((path_rv > trail_rv).mean())
    finals = p[:, -1]
    return {
        # The quantity the clean window measured at corr 0.4585 vs realised.
        'pred_vol_pct_24h': pred_rv * 100.0,
        'pred_vol_pct_24h_scaled': (pred_rv / scale) * 100.0,
        'pred_vol_scale_note': (
            'Generated dispersion runs at %.3fx realised for kronos-mini, measured '
            'over %d clean-window rows. pred_vol_pct_24h_scaled is that raw '
            'forecast divided by %.6f -- OUR correction, not the model output.'
            % (scale, int(calibration['fit_window']['n']), scale)),
        'pred_vol_scale_constant': scale,
        # Reported separately because it is a DIFFERENT, weaker quantity
        # (corr 0.366 vs 0.459); quoting the headline number against it would
        # describe something the endpoint does not serve.
        'path_endpoint_sd_pct': float(finals.std() / float(last_close)) * 100.0,
        'p_vol_amp_raw': p_raw,
        'p_vol_amp_calibrated': apply_isotonic(p_raw, calibration),
        'trailing_rv_24h_pct': trail_rv * 100.0,
        'n_paths': int(len(p)),
        'gen_distinct_endpoints': int(len(np.unique(finals))),
        'calibration_fitted_utc': calibration.get('fitted_utc'),
        'calibration_fit_window': calibration.get('fit_window'),
    }


def sample_paths(predictor, context: pd.DataFrame, forecast_start,
                 n_paths: int = SAMPLE_COUNT, pred_len: int = PRED_LEN) -> np.ndarray:
    """(n_paths, pred_len) predicted closes for the bar(s) after `context`."""
    if len(context) < CONTEXT_BARS:
        raise RuntimeError('Kronos needs %d context bars, got %d'
                           % (CONTEXT_BARS, len(context)))
    ctx = context.iloc[-CONTEXT_BARS:]
    missing = [c for c in PRICE_COLS if c not in ctx.columns]
    if missing:
        raise RuntimeError('context missing price columns: %s' % missing)
    x = ctx[PRICE_COLS].reset_index(drop=True)
    if x.isnull().values.any():
        raise RuntimeError('context contains NaN prices; refusing to impute.')

    x_ts = pd.Series(pd.DatetimeIndex(ctx.index))
    y_ts = pd.Series(forward_bar_index(forecast_start, pred_len))

    out, drawn = [], 0
    while drawn < n_paths:
        k = min(MAX_BATCH_SERIES, n_paths - drawn)
        res = predictor.predict_batch(
            df_list=[x] * k, x_timestamp_list=[x_ts] * k, y_timestamp_list=[y_ts] * k,
            pred_len=pred_len, T=T, top_p=TOP_P, sample_count=1, verbose=False)
        out.extend(r['close'].to_numpy(float) for r in res)
        drawn += k
    return np.asarray(out)


def p_up_from_paths(paths: np.ndarray, last_close: float) -> dict:
    """The forecast object. `gen_distinct` matters: tokenizer quantisation makes
    the sampled closes land on roughly 6 distinct values out of 30, so p_up is
    coarser than a binomial at n=30 would suggest."""
    finals = np.asarray(paths)[:, -1]
    return {
        'p_up': float((finals > last_close).mean()),
        'n_paths': int(len(finals)),
        'gen_mean': float(finals.mean()),
        'gen_sd': float(finals.std()),
        'gen_distinct': int(len(np.unique(finals))),
    }
