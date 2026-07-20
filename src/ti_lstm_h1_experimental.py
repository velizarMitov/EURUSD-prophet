"""
STANDALONE H1-native technical-indicator LSTM — RESEARCH ONLY, fully isolated.

Isolation contract (Step 0 of the experiment spec):
  * This module is NEVER imported by api.py, src/inference.py or
    _train_pipeline.py. It writes NO artifacts under models/ — production
    (baseline / with_macro / volatility / the existing H1 ensemble) is
    untouched. Promotion to serving is a separate, gated follow-up pass.
  * Entry point:  python -m src.ti_lstm_h1_experimental
    (add `quick` to skip the slow full run and only self-check CUDA).

Backend (Step 1): standalone Keras 3 on the PYTORCH backend, CUDA required.
KERAS_BACKEND is set process-locally below, BEFORE any keras import, so
production's tf.keras processes are unaffected. The run FAILS LOUDLY if
torch.cuda is unavailable or the device is not the expected RTX 4070 — no
silent CPU fallback.

Data (Step 2): the existing H1 chain only — src.live_data.fetch_h1_market_data
(MT5 -> yfinance -> cache) to refresh, then src.h1_features.load_h1_frame
(cache-first, the training-side convention). No second fetch path.

Indicators (Step 3, all trailing-window on the continuous H1 stream):
  pct_b        Bollinger %B, period=20, 2.0 std, on close
  macd, macd_hist   MACD with the SPECIFIED Fibonacci params 13/34, signal =
                    8-period SMA of the MACD line (NOT the conventional 12/26/9,
                    and the signal is an SMA by spec, not an EMA)
  trend_sma504, trend_sma168   close/SMA - 1 (the exact `trend_vs_sma504`
                    pattern from h1_features; 504 H1 bars ~= 21 trading days,
                    168 ~= 7)
  rsi_24       literally h1_features._rsi(close, 24) — REUSED, period 24, per
               the module's existing convention (flagged in the report: change
               to 14 would be a different experiment)
  cci_20       Lambert 1980: (TP - SMA20(TP)) / (0.015 * meandev20(TP))
  adx_14       Wilder 14-period ADX (+DI/-DI -> DX -> Wilder-smoothed ADX),
               dedicated hand-checkable unit test in tests/test_unit.py

Target (Step 4): ONE row per day (h1_features conventions: MIN_HOURS session
completeness, right-aligned (24, n) hourly tensor, build_daily_target's strict
shift(-1)) — the target is the NEXT-DAY percent log return / its sign,
matching the existing H1->Daily architecture. NOT next-hour prediction.

Methodology (Steps 5–7): chronological 70/10/20; architecture chosen on the
validation slice only; PyTorch run-to-run determinism CHECKED before any
single run is trusted (5-seed ensemble 42–46 otherwise); evaluation reports
FULL-coverage metrics (every row — no low-confidence-exclusion denominator
tricks); own hypothesis family results/ti_lstm_h1_hypothesis_log.csv.
"""
import os

# Process-local backend selection — MUST precede any keras import. Harmless to
# the rest of the codebase: production imports `tensorflow.keras`, and no
# production process imports this module.
os.environ.setdefault("KERAS_BACKEND", "torch")

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from src.h1_features import (
    _rsi, load_h1_frame, build_daily_target, MIN_HOURS, HOURS_PER_DAY,
    DEFAULT_H1_CACHE,
)

TI_FEATURE_COLUMNS = ['pct_b', 'macd', 'macd_hist', 'trend_sma504',
                      'trend_sma168', 'rsi_24', 'cci_20', 'adx_14']

HYPOTHESIS_LOG = 'results/ti_lstm_h1_hypothesis_log.csv'
VALIDATION_CSV = 'results/ti_lstm_h1_validation.csv'
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# Step 3 — indicators (pure pandas, trailing windows only; importable without
# keras/torch so the unit tests stay light).
# ---------------------------------------------------------------------------

def bollinger_percent_b(close: pd.Series, period: int = 20, ndev: float = 2.0) -> pd.Series:
    """%B = (close - lower) / (upper - lower), bands = SMA(period) ± ndev·std.
    Degenerate zero-width band -> neutral mid-band 0.5 (project convention)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    pct = (close - (mid - ndev * std)) / (2 * ndev * std)
    return pct.mask(std == 0, 0.5)


def macd_features(close: pd.Series, fast: int = 13, slow: int = 34, signal: int = 8):
    """MACD with the SPECIFIED 13/34 EMAs and an 8-period **SMA** signal line
    (spec is explicit: SMA, not the conventional EMA-9). Returns
    (macd_line, histogram = macd - signal)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.rolling(signal).mean()
    return macd_line, macd_line - signal_line


def trend_vs_sma(close: pd.Series, period: int) -> pd.Series:
    """close / SMA(period) - 1 — the exact `trend_vs_sma504` pattern from
    h1_features (0 = at/undefined trend, same fillna convention)."""
    sma = close.rolling(period).mean()
    return (close / sma - 1.0).fillna(0.0)


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20, c: float = 0.015) -> pd.Series:
    """Lambert (1980): CCI = (TP - SMA(TP, n)) / (c * meandev(TP, n)),
    TP = (H+L+C)/3. Zero mean deviation (flat window) -> 0, not inf/NaN."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    out = (tp - sma_tp) / (c * mean_dev)
    return out.mask(mean_dev == 0, 0.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Wilder's ADX(period): directional movement -> Wilder-smoothed +DI/-DI
    -> DX -> Wilder-smoothed ADX. Wilder's recursive smoothing is implemented
    as ewm(alpha=1/period, adjust=False) — algebraically the same recursion
    s_t = s_{t-1} + (x_t - s_{t-1})/n. A zero-TR / zero-DI window contributes
    0, never a division blow-up. Returns (adx, plus_di, minus_di)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).fillna(0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0).fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    safe_atr = atr.replace(0, np.nan)
    plus_di = (100.0 * plus_s / safe_atr).fillna(0.0)
    minus_di = (100.0 * minus_s / safe_atr).fillna(0.0)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).fillna(0.0)
    return dx.ewm(alpha=alpha, adjust=False).mean(), plus_di, minus_di


def enrich_h1_with_indicators(h1: pd.DataFrame) -> pd.DataFrame:
    """All TI_FEATURE_COLUMNS on the CONTINUOUS H1 stream (trailing windows
    carry cross-day context, exactly like h1_features._enrich_hourly). Leading
    warm-up NaNs get each indicator's neutral value — they only touch the
    first ~20 days of 2016, deep inside the train block."""
    h1 = h1.copy()
    close, high, low = h1['close'], h1['high'], h1['low']
    h1['pct_b'] = bollinger_percent_b(close).fillna(0.5)
    macd_line, macd_hist = macd_features(close)
    h1['macd'] = macd_line.fillna(0.0)
    h1['macd_hist'] = macd_hist.fillna(0.0)
    h1['trend_sma504'] = trend_vs_sma(close, 504)
    h1['trend_sma168'] = trend_vs_sma(close, 168)
    h1['rsi_24'] = _rsi(close, 24).fillna(50.0)
    h1['cci_20'] = cci(high, low, close).fillna(0.0)
    adx_line, _pdi, _mdi = adx(high, low, close)
    h1['adx_14'] = adx_line.fillna(0.0)
    h1['date'] = h1.index.normalize()
    return h1


# ---------------------------------------------------------------------------
# Step 4 — one row per day: (24, 8) right-aligned hourly tensor, next-day
# percent-return target via the exact h1_features shift(-1) convention.
# ---------------------------------------------------------------------------

def build_ti_datasets(cache_path: str = DEFAULT_H1_CACHE, h1: pd.DataFrame = None):
    """(X_seq, y_return_pct, y_direction, index) aligned per COMPLETE session
    (>= MIN_HOURS bars, same rule as aggregate_daily_features); the final
    (target-undefined) day is dropped."""
    h1 = load_h1_frame(cache_path) if h1 is None else h1
    h1 = enrich_h1_with_indicators(h1)

    g = h1.groupby('date')
    counts = g.size()
    daily_close = g['close'].last()
    keep = counts[counts >= MIN_HOURS].index
    daily_close = daily_close.reindex(keep)

    target = build_daily_target(daily_close)
    valid = target.notna()
    index = daily_close.index[valid]
    y_ret = target[valid].values.astype(np.float32)
    y_dir = (y_ret > 0).astype(int)

    by_day = {d: grp for d, grp in g}
    n_feat = len(TI_FEATURE_COLUMNS)
    X = np.zeros((len(index), HOURS_PER_DAY, n_feat), dtype=np.float32)
    for i, day in enumerate(index):
        arr = by_day[day][TI_FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-HOURS_PER_DAY:]
        X[i, HOURS_PER_DAY - len(arr):, :] = arr   # right-aligned, front-padded
    return X, y_ret, y_dir, index


# ---------------------------------------------------------------------------
# Step 1 — CUDA gate (loud failure, no CPU fallback).
# ---------------------------------------------------------------------------

def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False — REFUSING to fall back to CPU "
            "(experiment spec Step 1). Install the Windows CUDA wheel: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128")
    name = torch.cuda.get_device_name(0)
    print(f"CUDA OK: {name} (torch {torch.__version__}, backend={os.environ['KERAS_BACKEND']})")
    if '4070' not in name:
        raise RuntimeError(f"Expected the RTX 4070, got '{name}' — refusing to "
                           f"train on an unexpected device.")
    return name


# ---------------------------------------------------------------------------
# Step 5 — model + training (Keras 3 / torch backend, imported lazily).
# ---------------------------------------------------------------------------

def build_model(n_layers: int, units: int, n_features: int,
                dropout: float = 0.25, lr: float = 1e-3):
    import keras
    from keras import layers

    inputs = keras.Input(shape=(HOURS_PER_DAY, n_features), name='ti_hourly_window')
    x = inputs
    for i in range(n_layers):
        x = layers.LSTM(units, return_sequences=(i < n_layers - 1),
                        name=f'ti_lstm_{i + 1}')(x)
    x = layers.Dropout(dropout, name='ti_dropout')(x)
    ret_out = layers.Dense(1, activation='linear', name='return_output')(x)
    dir_out = layers.Dense(1, activation='sigmoid', name='direction_output')(x)
    model = keras.Model(inputs, [ret_out, dir_out],
                        name=f'ti_lstm_h1_{n_layers}x{units}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
                  loss_weights={'return_output': 1.0, 'direction_output': 1.0})
    return model


def train_once(X_tr, y_tr, X_es, y_es, X_all, n_layers, units, seed,
               epochs=100, batch_size=32, patience=10, verbose=0):
    """One fit -> full-length (return_pred, direction_prob). y_* are
    (ret, dir) tuples; X_es/y_es drive early stopping."""
    import keras
    keras.utils.set_random_seed(seed)   # seeds python/numpy/torch together

    model = build_model(n_layers, units, X_tr.shape[2])
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                       restore_best_weights=True, verbose=0)
    model.fit(X_tr, {'return_output': y_tr[0], 'direction_output': y_tr[1]},
              validation_data=(X_es, {'return_output': y_es[0], 'direction_output': y_es[1]}),
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=verbose)
    pred_ret, prob_dir = model.predict(X_all, verbose=0)
    return pred_ret.ravel(), prob_dir.ravel(), len(model.history.history['loss'])


def bootstrap_auc_delta(y, score_a, score_b, alpha, n_boot=BOOTSTRAP_RESAMPLES,
                        random_state=42):
    """Paired bootstrap of AUC(score_a) - AUC(score_b) on identical rows.
    Returns (point_delta, ci_low, ci_high, frac_positive)."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    deltas = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas[i] = roc_auc_score(yt, score_a[idx]) - roc_auc_score(yt, score_b[idx])
    point = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)
    lo, hi = np.nanpercentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, float(np.nanmean(deltas > 0))


# ---------------------------------------------------------------------------
# Step 6 comparators — existing H1 ensemble + daily baseline, READ-ONLY loads.
# ---------------------------------------------------------------------------

def existing_h1_ensemble_score(h1, index):
    """Mean next-day-return prediction of the shipped 4-model H1 ensemble on
    the given days (directional score, same convention as its own evaluation).
    Read-only artifact loads; returns None if anything is missing."""
    import joblib
    import keras
    from src.h1_features import build_h1_datasets

    X_flat_df, X_seq, _yr, _yd, idx = build_h1_datasets(h1=h1)
    pos = idx.get_indexer(index)
    assert (pos >= 0).all(), "existing-ensemble day set must cover the TI day set"

    xgb_m = joblib.load('models/h1_xgb_regressor.pkl')
    rf_m = joblib.load('models/h1_rf_regressor.pkl')
    svm_m = joblib.load('models/h1_svm_regressor.pkl')
    flat_scaler = joblib.load('models/h1_feature_scaler.pkl')
    seq_scaler = joblib.load('models/h1_lstm_scaler.pkl')
    lstm_m = keras.saving.load_model('models/h1_lstm.keras')   # backend-portable .keras

    X_flat = X_flat_df.values[pos]
    X_flat_s = flat_scaler.transform(X_flat)
    n_feat = X_seq.shape[2]
    X_seq_s = seq_scaler.transform(
        X_seq[pos].reshape(-1, n_feat)).reshape(len(pos), X_seq.shape[1], n_feat).astype('float32')

    preds = [xgb_m.predict(X_flat), rf_m.predict(X_flat),
             svm_m.predict(X_flat_s), lstm_m.predict(X_seq_s, verbose=0).ravel()]
    return np.mean(preds, axis=0)


def daily_baseline_score(index):
    """The production daily `baseline` GBM classifier's P(up) for the given
    dates (read-only; cache-only macro so the run is offline-deterministic).
    Benchmark ONLY — no daily-family decision is made here."""
    import joblib
    from src.features import (
        load_history, merge_macro_features, add_advanced_features, LAG_COLUMNS,
        apply_lag_pca, model_input_columns, variant_feature_columns,
    )
    from src.macro_data import _read_cache, _FEATURE_COMBINERS

    with open('config.json') as f:
        config = json.load(f)
    raw = load_history(config['data']['history_csv_path'])
    frames = [_read_cache(config['macro']['cache_path'])[['yield_differential']]]
    for key, spec in config['macro']['features'].items():
        _c, col, _l = _FEATURE_COMBINERS[key]
        frames.append(_read_cache(spec['cache_path'])[[col]])
    feat = add_advanced_features(merge_macro_features(raw, pd.concat(frames, axis=1).sort_index()))

    lag_scaler = joblib.load('models/baseline/lag_scaler.pkl')
    lag_pca = joblib.load('models/baseline/lag_pca.pkl')
    global_scaler = joblib.load('models/baseline/global_scaler.pkl')
    gbm = joblib.load('models/baseline/best_gbm_eurusd.pkl')
    red = apply_lag_pca(feat, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    cols = model_input_columns(lag_pca, base_columns=variant_feature_columns('baseline'),
                               lag_columns=LAG_COLUMNS)
    X = global_scaler.transform(red[cols])

    daily_days = pd.DatetimeIndex(feat.index).normalize()
    if daily_days.tz is None:
        daily_days = daily_days.tz_localize('UTC')
    pos = daily_days.get_indexer(pd.DatetimeIndex(index))
    covered = pos >= 0
    prob = np.full(len(index), np.nan)
    if covered.any():
        prob[covered] = gbm.predict_proba(X[pos[covered]])[:, 1]
    return prob, covered


# ---------------------------------------------------------------------------
# The experiment driver.
# ---------------------------------------------------------------------------

def run(cache_path: str = DEFAULT_H1_CACHE):
    require_cuda()

    # Step 2: refresh via the existing chain (guarded), then cache-first load —
    # the exact _train_pipeline.py H1 convention. No second fetch path.
    try:
        from src.live_data import fetch_h1_market_data
        _df, _src = fetch_h1_market_data(cache_path=cache_path)
        print(f"H1 cache refreshed from {_src}: {0 if _df is None else len(_df)} bars.")
    except Exception as e:
        print(f"H1 cache refresh skipped ({e}); using existing cache.")
    h1 = load_h1_frame(cache_path)

    X, y_ret, y_dir, index = build_ti_datasets(h1=h1)
    n = len(index)
    train_end, val_end = int(n * 0.70), int(n * 0.80)
    print(f"TI datasets: X{X.shape}, days {index.min().date()} -> {index.max().date()}")
    print(f"split: train [0:{train_end}]  val [{train_end}:{val_end}]  "
          f"test [{val_end}:{n}] (touched ONCE, at the end)")

    # Per-hour scaler fit on TRAIN timesteps only (mirrors the production H1
    # LSTM's seq scaler discipline).
    n_feat = X.shape[2]
    scaler = StandardScaler().fit(X[:train_end].reshape(-1, n_feat))
    X_s = scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype('float32')

    # Inner early-stopping tail = last 1/7 of train (~[60%:70%]) so the
    # validation slice stays a pure decision block (volatility-experiment
    # discipline: the arbiter is never the early-stopping monitor).
    inner_end = int(train_end * 6 / 7)
    X_tr, X_es = X_s[:inner_end], X_s[inner_end:train_end]
    y_tr = (y_ret[:inner_end], y_dir[:inner_end])
    y_es = (y_ret[inner_end:train_end], y_dir[inner_end:train_end])
    val = slice(train_end, val_end)
    test = slice(val_end, n)
    y_dir_val = y_dir[val]

    # ---- Step 5a: PyTorch nondeterminism check (same seed, two runs) -------
    print("\n--- Determinism check (seed 42, 1x32, two identical runs) ---")
    _r1, p1, e1 = train_once(X_tr, y_tr, X_es, y_es, X_s, 1, 32, seed=42)
    _r2, p2, e2 = train_once(X_tr, y_tr, X_es, y_es, X_s, 1, 32, seed=42)
    max_diff = float(np.max(np.abs(p1 - p2)))
    auc_diff = abs(roc_auc_score(y_dir_val, p1[val]) - roc_auc_score(y_dir_val, p2[val]))
    deterministic = max_diff < 1e-6
    print(f"epochs {e1} vs {e2}; max|prob diff|={max_diff:.2e}; "
          f"val AUC diff={auc_diff:.4f} -> "
          f"{'DETERMINISTIC' if deterministic else 'NON-DETERMINISTIC -> 5-seed ensembles for every decision'}")

    seeds = [42] if deterministic else ENSEMBLE_SEEDS

    # ---- Step 5b: architecture comparison on the VALIDATION slice ----------
    print(f"\n--- Architecture comparison on validation (seeds {seeds}) ---")
    arch_results = {}
    for n_layers in (1, 2):
        for units in (32, 64):
            probs, epochs_used = [], []
            for seed in seeds:
                _r, p, e = train_once(X_tr, y_tr, X_es, y_es, X_s, n_layers, units, seed)
                probs.append(p)
                epochs_used.append(e)
            prob_mean = np.mean(probs, axis=0)
            auc_v = roc_auc_score(y_dir_val, prob_mean[val])
            acc_v = accuracy_score(y_dir_val, (prob_mean[val] >= 0.5).astype(int))
            arch_results[(n_layers, units)] = (prob_mean, auc_v, acc_v)
            print(f"  {n_layers}x{units}: val AUC={auc_v:.4f}  acc={acc_v:.4f}  "
                  f"epochs={epochs_used}")

    (best_layers, best_units), (prob_ti, auc_val, acc_val) = max(
        arch_results.items(), key=lambda kv: kv[1][1])
    print(f"  -> selected {best_layers}x{best_units} (val AUC {auc_val:.4f}) — "
          f"validation is the architecture arbiter, test stays untouched")

    # ---- Step 6: comparisons on validation (paired bootstrap) --------------
    log = _load_log()
    family_size = max(1, len(set(log['hypothesis']) | {'ti_lstm_h1_vs_existing_h1_ensemble'}))
    alpha = FAMILY_ALPHA / family_size
    print(f"\n--- Validation comparisons (own family, size {family_size}, "
          f"alpha={alpha:.4g}) ---")

    ens_score = existing_h1_ensemble_score(h1, index)
    auc_ens_val = roc_auc_score(y_dir_val, ens_score[val])
    d_ens, lo_ens, hi_ens, frac_ens = bootstrap_auc_delta(
        y_dir_val, prob_ti[val], ens_score[val], alpha=alpha)
    print(f"  existing H1 ensemble val AUC={auc_ens_val:.4f}  "
          f"[CAVEAT: its models TRAINED on [0:80%] — the val slice is inside "
          f"their training block, biasing this comparison AGAINST the TI model]")
    print(f"  dAUC(TI - ensemble) = {d_ens:+.4f}  CI[{lo_ens:+.4f}, {hi_ens:+.4f}]  "
          f"frac>0={frac_ens:.3f}")

    daily_prob, covered = daily_baseline_score(index[val])
    if covered.all():
        auc_daily_val = roc_auc_score(y_dir_val, daily_prob)
        d_dly, lo_dly, hi_dly, _f = bootstrap_auc_delta(
            y_dir_val, prob_ti[val], daily_prob, alpha=alpha)
        print(f"  daily baseline GBM val AUC={auc_daily_val:.4f}  "
              f"[benchmark only; scored against the H1-derived target on shared days]")
        print(f"  dAUC(TI - daily) = {d_dly:+.4f}  CI[{lo_dly:+.4f}, {hi_dly:+.4f}]")
    else:
        auc_daily_val, d_dly, lo_dly, hi_dly = (float('nan'),) * 4
        print(f"  daily baseline: only {int(covered.sum())}/{len(covered)} val days "
              f"covered by the daily frame — comparison skipped")

    cleared = bool(lo_ens > 0)
    verdict = ('CLEARS validation bar vs existing H1 ensemble — proceed to the '
               'gated production path (Step 7 KEEP branch)'
               if cleared else
               'DROP — no CI-confirmed edge over the existing H1 ensemble on '
               'validation; stays research-only')
    print(f"  VERDICT (validation): {verdict}")

    # ---- one-shot TEST block report (every row, full coverage) -------------
    y_dir_test = y_dir[test]
    auc_test = roc_auc_score(y_dir_test, prob_ti[test])
    acc_test = accuracy_score(y_dir_test, (prob_ti[test] >= 0.5).astype(int))
    auc_ens_test = roc_auc_score(y_dir_test, ens_score[test])
    d_te, lo_te, hi_te, _ = bootstrap_auc_delta(y_dir_test, prob_ti[test],
                                                ens_score[test], alpha=alpha)
    print(f"\n--- ONE-SHOT test report [80:100] (full coverage, n={len(y_dir_test)}; "
          f"a report, never a search knob) ---")
    print(f"  TI-LSTM:      AUC={auc_test:.4f}  acc={acc_test:.4f}")
    print(f"  H1 ensemble:  AUC={auc_ens_test:.4f}  (fair here: both out-of-sample)")
    print(f"  dAUC(TI-ens) = {d_te:+.4f}  CI[{lo_te:+.4f}, {hi_te:+.4f}]")

    result = {
        'hypothesis': 'ti_lstm_h1_vs_existing_h1_ensemble',
        'date': pd.Timestamp.utcnow().date().isoformat(),
        'arbiter': 'validation[70:80] of H1 day set',
        'n_days': n, 'n_val': val_end - train_end, 'n_test': n - val_end,
        'backend': f"keras3-torch-cuda",
        'deterministic': deterministic,
        'seeds': ' '.join(str(s) for s in seeds),
        'architecture': f'{best_layers}x{best_units}',
        'ti_val_auc': round(auc_val, 4), 'ti_val_acc': round(acc_val, 4),
        'h1_ensemble_val_auc': round(auc_ens_val, 4),
        'daily_baseline_val_auc': round(auc_daily_val, 4) if auc_daily_val == auc_daily_val else '',
        'point_dauc_vs_ensemble': round(d_ens, 4),
        'ci_dauc_low': round(lo_ens, 4), 'ci_dauc_high': round(hi_ens, 4),
        'point_dauc_vs_daily': round(d_dly, 4) if d_dly == d_dly else '',
        'alpha_bar': alpha,
        'cleared_bar': cleared,
        'test_ti_auc': round(auc_test, 4), 'test_ti_acc': round(acc_test, 4),
        'test_h1_ensemble_auc': round(auc_ens_test, 4),
        'test_dauc': round(d_te, 4),
        'test_ci_dauc_low': round(lo_te, 4), 'test_ci_dauc_high': round(hi_te, 4),
        'verdict': verdict,
    }
    _register(result)
    pd.DataFrame([result]).to_csv(VALIDATION_CSV, index=False)
    print(f"\nSaved: {VALIDATION_CSV}\nHypothesis logged: {HYPOTHESIS_LOG}")
    return result


def _load_log():
    if not os.path.exists(HYPOTHESIS_LOG):
        return pd.DataFrame(columns=['n', 'hypothesis'])
    return pd.read_csv(HYPOTHESIS_LOG)


def _register(result):
    """Own hypothesis family (idempotent by hypothesis name) — this model is a
    qualitatively new object, so it never dilutes the direction/return or
    volatility families' Bonferroni counts."""
    log = _load_log()
    if result['hypothesis'] in set(log.get('hypothesis', [])):
        return log
    row = {'n': len(log) + 1, **result}
    out = pd.DataFrame([row]) if log.empty else pd.concat(
        [log, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(HYPOTHESIS_LOG, index=False)
    return out


# ---------------------------------------------------------------------------
# PRODUCTION promotion (owner override 2026-07-18) — this model SHIPPED with a
# DROP verdict, by explicit owner decision, for transparent forward observation
# only. It has NO demonstrated edge (test AUC 0.5128 vs the existing H1
# ensemble's 0.5283, ΔAUC CI [−0.072, +0.042] — includes 0, point NEGATIVE).
# Every consumer of these artifacts must surface that status verbatim
# (`validated: false` + the actual numbers), never the volatility model's
# validated framing. See IMPROVEMENT_LOG.md "owner override".
# ---------------------------------------------------------------------------
TI_MODEL_DIR = os.path.join('models', 'ti_lstm_h1')
TI_ARTIFACTS = ['ti_lstm_h1.keras', 'ti_scaler.pkl', 'ti_config.pkl', 'ti_metrics.json']
PRODUCTION_ARCHITECTURE = (2, 64)   # selected on the validation slice (see run())


def train_production_ti_model(base_dir: str = '', cache_path: str = DEFAULT_H1_CACHE):
    """Train + persist the promoted TI-LSTM under models/ti_lstm_h1/.

    Production split convention (same as the daily/volatility families):
    scaler fit on [0:80%], the LSTM trains [0:70%] with [70%:80%] as the
    early-stopping tail, test block untouched. Architecture = the
    validation-selected 2x64; seed 42 (the torch backend was verified
    bit-deterministic, so a single seeded run IS the reproducible object).

    MUST run in its own process (python -m src.ti_lstm_h1_experimental
    train-production): KERAS_BACKEND is frozen at the first keras import, so a
    process that already imported tf.keras cannot host this torch-backend fit.
    The saved .keras file itself is backend-portable (standard layers only) —
    serving loads it under tf.keras with no torch dependency.
    """
    require_cuda()
    out_dir = os.path.join(base_dir, TI_MODEL_DIR)
    os.makedirs(out_dir, exist_ok=True)

    try:
        from src.live_data import fetch_h1_market_data
        _df, _src = fetch_h1_market_data(cache_path=os.path.join(base_dir, cache_path)
                                         if base_dir else cache_path)
        print(f"H1 cache refreshed from {_src}.")
    except Exception as e:
        print(f"H1 cache refresh skipped ({e}); using existing cache.")

    X, y_ret, y_dir, index = build_ti_datasets(
        cache_path=os.path.join(base_dir, cache_path) if base_dir else cache_path)
    n = len(index)
    train_end, val_end = int(n * 0.70), int(n * 0.80)

    n_feat = X.shape[2]
    scaler = StandardScaler().fit(X[:val_end].reshape(-1, n_feat))
    X_s = scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype('float32')

    n_layers, units = PRODUCTION_ARCHITECTURE
    import keras
    keras.utils.set_random_seed(42)
    model = build_model(n_layers, units, n_feat)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                       restore_best_weights=True, verbose=0)
    model.fit(X_s[:train_end], {'return_output': y_ret[:train_end],
                                'direction_output': y_dir[:train_end]},
              validation_data=(X_s[train_end:val_end],
                               {'return_output': y_ret[train_end:val_end],
                                'direction_output': y_dir[train_end:val_end]}),
              epochs=100, batch_size=32, callbacks=[es], verbose=0)
    epochs_trained = len(model.history.history['loss'])
    print(f"Trained {n_layers}x{units} for {epochs_trained} epochs (seed 42, CUDA).")

    # Honest status metadata, carried verbatim from the experiment's run report
    # (results/ti_lstm_h1_validation.csv) into serving.
    evidence = {}
    exp_csv = os.path.join(base_dir, VALIDATION_CSV) if base_dir else VALIDATION_CSV
    if os.path.exists(exp_csv):
        evidence = pd.read_csv(exp_csv).iloc[0].to_dict()
    metrics = {
        'validated': False,
        'verdict': 'DROP on its own hypothesis bar — SHIPPED ANYWAY by explicit '
                   'owner override (2026-07-18) for transparent forward '
                   'observation. No demonstrated edge.',
        'val_auc': evidence.get('ti_val_auc'),
        'val_acc': evidence.get('ti_val_acc'),
        'test_auc': evidence.get('test_ti_auc'),
        'test_acc': evidence.get('test_ti_acc'),
        'test_h1_ensemble_auc': evidence.get('test_h1_ensemble_auc'),
        'test_dauc_vs_ensemble': evidence.get('test_dauc'),
        'test_dauc_ci': [evidence.get('test_ci_dauc_low'), evidence.get('test_ci_dauc_high')],
        'hypothesis_log': 'results/ti_lstm_h1_hypothesis_log.csv',
        'architecture': f'{n_layers}x{units}',
        'epochs_trained': epochs_trained,
        'trained_backend': 'keras3-torch-cuda',
        'n_days': n, 'trained_through': str(index[train_end - 1].date()),
    }

    model.save(os.path.join(out_dir, 'ti_lstm_h1.keras'))
    import joblib
    joblib.dump(scaler, os.path.join(out_dir, 'ti_scaler.pkl'))
    joblib.dump({'hours': HOURS_PER_DAY, 'features': list(TI_FEATURE_COLUMNS)},
                os.path.join(out_dir, 'ti_config.pkl'))
    with open(os.path.join(out_dir, 'ti_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved under {out_dir}/: {', '.join(TI_ARTIFACTS)}")
    return metrics


def build_ti_inference_sample(cache_path: str = DEFAULT_H1_CACHE, now=None, h1=None):
    """Latest COMPLETE session's (1, 24, 8) TI tensor for live serving —
    mirrors h1_features.build_ti/build_h1_inference_sample: live-first
    staleness-gated load (refresh_h1_frame), the still-forming current UTC day
    dropped, right-aligned last 24 bars. Returns (tensor, as_of, source).
    Pure pandas/numpy — no keras/torch needed at serving time."""
    from src.h1_features import refresh_h1_frame, _to_utc

    now = _to_utc(now)
    if h1 is None:
        h1, source = refresh_h1_frame(cache_path, now=now)
    else:
        source = "preloaded"
    h1 = enrich_h1_with_indicators(h1)

    today = now.normalize()
    g = h1[h1['date'] < today].groupby('date')
    counts = g.size()
    complete = counts[counts >= MIN_HOURS]
    if not len(complete):
        raise RuntimeError("No completed H1 trading day available for TI inference.")
    as_of = complete.index[-1]

    day = h1[h1['date'] == as_of]
    arr = day[TI_FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-HOURS_PER_DAY:]
    X = np.zeros((1, HOURS_PER_DAY, len(TI_FEATURE_COLUMNS)), dtype=np.float32)
    X[0, HOURS_PER_DAY - len(arr):, :] = arr
    return X, as_of, source


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['quick']:
        require_cuda()
    elif sys.argv[1:2] == ['train-production']:
        train_production_ti_model()
    else:
        run()
