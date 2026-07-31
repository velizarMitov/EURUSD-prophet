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

from .loader import (CONTEXT_BARS, PRED_LEN, SAMPLE_COUNT, T, TOP_P)

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
    y_ts = pd.Series(pd.date_range(pd.Timestamp(forecast_start), periods=pred_len, freq='h'))

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
