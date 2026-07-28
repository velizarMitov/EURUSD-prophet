"""
Walk-forward validation report — RESEARCH-ONLY robustness check on the
ALREADY-FIXED production model configurations (GBM+LSTM direction/return
ensemble, both variants, and the 5-seed volatility ensemble).

============================== HARD BOUNDARY ===============================
This is entirely SEPARATE from production. It does NOT modify
`_train_pipeline.py`, `src/inference.py`, `config.json`, or any file under
`models/` — no serving artifact is touched, no live prediction changes. This
produces a DESCRIPTIVE research report only, logged to
`results/walk_forward_validation.csv` (per-window detail) and
`results/walk_forward_validation_summary.csv` (the cross-window aggregate).
It does NOT touch `feature_hypothesis_log.csv`, `volatility_hypothesis_log.csv`,
or any harmonic-pattern log — those are out of scope by construction (this is
not a feature/model hypothesis; nothing is being added or re-tuned).

If the results here look good, that is the TRIGGER for a SEPARATE, subsequent
conversation about whether/how to wire up scheduled periodic retraining in
production — this module does NOT build that infrastructure. It answers
"would our EXISTING approach have worked robustly over time," never "should
we change anything."
==============================================================================

Why this exists — VALIDATION, explicitly NOT OPTIMIZATION
------------------------------------------------------------------------------
Every other research module in this project asks "should we ADD/KEEP a
feature or model." This one asks something categorically different: "is the
model configuration we have ALREADY decided on (fixed hyperparameters, fixed
architecture, fixed feature set per variant) robust across many historical
re-training cycles, or did the single static 70/15/15 split get lucky/unlucky
on one particular slice of history?" No hyperparameter is searched, widened,
or narrowed here, and no feature is added or removed — the GBM `param_grid`,
the LSTM architecture, and `config.json`'s hyperparameters are read AS-IS on
every window. This is walk-forward VALIDATION, not walk-forward OPTIMIZATION
(which would re-select hyperparameters per window and reintroduce exactly the
data-snooping risk the Production Methodology exists to prevent). If this
module ever re-selected a `param_grid` entry per window, it would silently
become an optimization exercise wearing a validation label — it does not.

Why reusing the historically-reserved test-block date range is sound HERE
------------------------------------------------------------------------------
Every prior hypothesis test in this project scrupulously avoided the static
test block `[80%:100%]` because reusing it as a repeated KEEP/DROP arbiter is
data-snooping. This module DOES walk across those same calendar years — that
needs justification, not silent assumption:
  (a) No NEW keep/drop decision about any feature is made from this exercise
      — the feature set and model configuration are already fixed and
      unchanged BEFORE this report is even run. There is nothing left to
      snoop into a decision.
  (b) Each window's model is trained ONLY on data strictly BEFORE that
      window's own test period — chronological causality is preserved at
      EVERY single window, no exceptions (see `tests/test_unit.py`'s explicit
      causality checks). Nothing future ever leaks into any window's own
      training, regardless of which historical years that window covers.
The question here — "is this already-fixed configuration robust over time" —
is fundamentally different from "should we add this feature," which is
precisely why reusing this date range is fine in THIS module and nowhere
else.

Scheme — PRE-REGISTERED, fixed before looking at any result
------------------------------------------------------------------------------
    train_window = 3 years trailing
    step / test_window = 1 year (annual re-validation cadence)

Annual, not quarterly, cadence: a defensible institutional retraining cadence
that also keeps this report computationally practical. `_train_pipeline.py`'s
own GridSearchCV+LSTM fit was assumed to cost real time per run; measured
directly before committing to this design (see `run()`'s printed wall-clock
estimate, computed from a REAL timed first window, never guessed) — a
quarterly cadence across ~27 years of history would mean ~100+ reruns per
variant, plausibly impractical; annual cuts that to ~24, which measured
timing shows comfortably fits in one sitting (see below). Slides forward one
step at a time across the FULL available euro-era history (1999+, back to
the start of the engineered daily series — see ARCHITECTURE_DOCS.md
"Production Methodology"), INCLUDING the years inside the currently-reserved
static test block `[80%:100%]` (justified above).

A real engineering finding worth stating plainly: `_train_pipeline.py`
enables GPU (CUDA) XGBoost because it pays off on the FULL ~6,850-row
production training block. On this module's much SMALLER per-window training
slices (~900-1,000 rows for a 3-year window), the fixed per-call GPU
overhead dominates — measured directly: CPU GridSearchCV finished in ~10s vs
~326s on CUDA for the SAME window (over 30x slower on GPU). This module
therefore forces `device='cpu'` for its GBM heads. This is a compute-BACKEND
choice, not a change to the pre-registered `param_grid` or search procedure
itself — the search space, folds, and scoring are byte-identical either way.

Per-window procedure (reusing the EXISTING, ALREADY-FIXED recipes UNCHANGED)
------------------------------------------------------------------------------
Direction/return, BOTH variants — the same recipe as
`_train_pipeline.py::train_variant()` (same GBM `param_grid` + GridSearchCV
over `TimeSeriesSplit`, same LSTM architecture/hyperparameters from
`config.json`, same PCA/scaler fit-train-only convention). `train_variant()`
itself cannot be imported here: `_train_pipeline.py` is a top-level SCRIPT
whose module-level code trains and persists real production artifacts to
`models/` as a side effect of merely importing it — doing that would violate
the hard boundary outright. This module therefore duplicates the identical,
UNCHANGED recipe (same `config.json` values, same architecture, read at run
time) as pure, in-memory functions instead of sharing code via import.

    Mapping the window onto the production 80/10/10 convention: production
    fits PCA/scaler/GBM on `[0:80%]` of the WHOLE series, carves the LSTM's
    early-stopping tail as the `[70%:80%]` slice (a 12.5% tail relative to
    the 80% fit block: `val_fraction/train_fraction = 0.10/0.80`), and
    reports on the held-out `[80%:100%]` test block. Here, the window's own
    3-year `train_window` plays the role of that 80% fit block in full (PCA,
    global scaler, and GBM's GridSearchCV all fit on the ENTIRE
    `train_window`), the LSTM's early-stopping tail is the SAME 12.5%
    proportion carved from the END of that `train_window` (not the whole
    27-year series), and the window's 1-year `test_window` — entirely
    separate, never touched by any fit — plays the role of the held-out test
    block. This preserves production's exact internal ratios while
    respecting the walk-forward's own 3-year/1-year window definition.

Volatility — the exact 5-seed ensemble procedure (seeds 42-46, identical
3-head architecture) AND the exact train-only-fit GARCH(1,1) baseline,
refit FRESH per window (this mirrors how a real periodic-retrain deployment
of the GARCH baseline would also behave — not a special exception).
`src.volatility.train_production_volatility_model` cannot be called directly
either (it hardcodes writing `.keras` files under `models/volatility/`), so
this module reuses its truly side-effect-free building blocks
(`make_sequences`, `garch_forecasts`, `persistence_forecasts`, `fit_lag_pca`,
`apply_lag_pca`) and reimplements only the 5-seed training loop itself
in-memory, with no disk writes.

Aggregation across all windows
------------------------------------------------------------------------------
  * Per-window metric distribution (mean/median/min/max/std) — reveals
    whether an edge is stable or concentrated in a few lucky windows.
  * Time-trend check — Spearman correlation of window-end-date (ordinal) vs.
    that window's own metric, a direct trend statistic (not eyeballing) for
    whether more recent windows are degrading relative to older ones.
  * Pooled OOS series — every window's own out-of-sample test predictions,
    concatenated in chronological order (now MORE rows than the static test
    block's fixed ~20% slice), scored with the SAME paired-bootstrap + exact
    McNemar convention used throughout this project vs. the pooled
    majority-class baseline — but via a MOVING-BLOCK (circular) bootstrap
    (block length = one window's own test-row count), not i.i.d., since
    adjacent windows' test days share serial correlation (the same guard
    already used for the vol-scaled-sizing backtest).

Run:  python -m src.walk_forward_validation
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb

from src.macro_data import fetch_macro_features
from src.features import (
    load_history, merge_macro_features, add_advanced_features,
    LAG_COLUMNS, PRICE_FEATURE_COLUMNS, TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN,
    TARGET_VOLATILITY_COLUMN, fit_lag_pca, apply_lag_pca, model_input_columns,
    variant_feature_columns,
)
from src.volatility import (
    make_sequences, garch_forecasts, persistence_forecasts, VOL_ENSEMBLE_SEEDS,
)

WALK_FORWARD_LOG = 'results/walk_forward_validation.csv'
WALK_FORWARD_SUMMARY_LOG = 'results/walk_forward_validation_summary.csv'

TRAIN_WINDOW_YEARS = 3
TEST_WINDOW_YEARS = 1
RANDOM_STATE = 42
XGB_DEVICE = 'cpu'   # see module docstring: measured >30x faster than CUDA at this row count
BOOTSTRAP_RESAMPLES = 2000
# A single-sitting research report should not become a multi-hour unattended
# commitment; if the REAL measured first-window time extrapolates past this,
# run() stops and reports rather than silently coarsening the schedule.
MAX_ESTIMATED_TOTAL_SECONDS = 3 * 3600

VARIANTS = ('baseline', 'with_macro')

PER_WINDOW_COLUMNS = [
    'window_id', 'train_start', 'train_end', 'test_end', 'n_train', 'n_test',
    'baseline_gbm_acc', 'baseline_gbm_auc', 'baseline_gbm_majority_acc', 'baseline_gbm_ret_mae',
    'baseline_lstm_acc', 'baseline_lstm_auc', 'baseline_lstm_majority_acc', 'baseline_lstm_ret_mae',
    'with_macro_gbm_acc', 'with_macro_gbm_auc', 'with_macro_gbm_majority_acc', 'with_macro_gbm_ret_mae',
    'with_macro_lstm_acc', 'with_macro_lstm_auc', 'with_macro_lstm_majority_acc', 'with_macro_lstm_ret_mae',
    'vol_ensemble_mae', 'vol_ensemble_r2', 'vol_garch_mae', 'vol_garch_r2',
    'vol_persistence_mae', 'vol_persistence_r2',
]


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def load_engineered_history(config, base_dir=''):
    """The SAME shared euro-era engineered frame every production model
    trains on (`_train_pipeline.py`'s own step 1/1B/2, reused unchanged via
    the shared `src.features`/`src.macro_data` functions — nothing
    `_train_pipeline.py`-specific is imported)."""
    ohlcv = load_history(_p(base_dir, config['data']['history_csv_path']))
    macro, _ = fetch_macro_features(ohlcv.index.min(), ohlcv.index.max(), config['macro'], base_dir=base_dir)
    return add_advanced_features(merge_macro_features(ohlcv.copy(), macro))


def compute_windows(index: pd.DatetimeIndex, train_years=TRAIN_WINDOW_YEARS, test_years=TEST_WINDOW_YEARS):
    """PRE-REGISTERED rolling-window schedule: `train_years` trailing train
    window, `test_years` step/test window, sliding forward across the FULL
    available history — fixed before looking at any result, never tuned
    after. Pure calendar-date arithmetic (independent of which rows actually
    exist on a given date); returns a list of dicts with `train_start`
    (inclusive), `train_end` (exclusive; test starts here), `test_end`
    (exclusive) — a caller slices the engineered frame by these dates."""
    start, end = index.min(), index.max()
    windows = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        windows.append({'train_start': train_start, 'train_end': train_end, 'test_end': test_end})
        train_start = train_start + pd.DateOffset(years=test_years)
    return windows


def _slice_window(feat, window):
    """The window's own combined train+test rows (by calendar date, half-open
    `[train_start, test_end)`), plus `train_end_local` — the row POSITION
    within this slice where the test period begins. Strict causality: every
    row before `train_end_local` has a date < window['train_end'] (the test
    period's own start), and no row at/after `train_end_local` is ever used
    for fitting anything below."""
    mask = (feat.index >= window['train_start']) & (feat.index < window['test_end'])
    window_df = feat.loc[mask]
    train_end_local = int((window_df.index < window['train_end']).sum())
    return window_df, train_end_local


def _lstm_es_tail_end(train_end_local, config):
    """The LSTM's inner early-stopping tail boundary WITHIN the window's own
    train block — the SAME proportion (`val_fraction/train_fraction`, 12.5%
    by default) production carves out of its own `[0:80%]` fit block,
    applied here to this window's `train_window` instead of the whole
    27-year series (see module docstring's mapping rationale)."""
    es_frac = config['split']['val_fraction'] / config['split']['train_fraction']
    return int(train_end_local * (1 - es_frac))


# ── direction/return: one variant, one window (mirrors train_variant()) ────

def _create_block_sequences(X, y_ret, y_dir, time_steps):
    """IDENTICAL geometry to `_train_pipeline.py::train_variant`'s own
    `create_mt_sequences`: sequences built INDEPENDENTLY within this block
    (never reaching into a neighboring block for context) — a sequence's
    target is its own window's LAST row."""
    Xs, ys_ret, ys_dir = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys_ret.append(y_ret[i + time_steps - 1])
        ys_dir.append(y_dir[i + time_steps - 1])
    return np.array(Xs), np.array(ys_ret), np.array(ys_dir)


def train_direction_variant_window(window_df, variant, train_end_local, config, random_state=RANDOM_STATE):
    """ONE variant's GBM+LSTM direction/return procedure on ONE walk-forward
    window — the identical recipe `_train_pipeline.py::train_variant()` uses
    (same param_grid/architecture/PCA-scaler-fit-train-only convention), but
    entirely in-memory: no file under `models/` is ever written, no mlflow
    run is started. Returns per-row test predictions/labels for both models,
    for later per-window scoring AND cross-window pooling."""
    feature_columns = variant_feature_columns(variant)
    lag_scaler, lag_pca = fit_lag_pca(window_df.iloc[:train_end_local], lag_columns=LAG_COLUMNS,
                                      variance_threshold=config['pca']['variance_threshold'])
    reduced = apply_lag_pca(window_df, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    model_cols = model_input_columns(lag_pca, base_columns=feature_columns, lag_columns=LAG_COLUMNS)
    X_all = reduced[model_cols]
    scaler = StandardScaler().fit(X_all.iloc[:train_end_local])
    X_scaled = scaler.transform(X_all)

    y_dir = reduced[TARGET_DIRECTION_COLUMN].to_numpy()
    y_ret = reduced[TARGET_RETURN_COLUMN].to_numpy()
    X_train_s, X_test_s = X_scaled[:train_end_local], X_scaled[train_end_local:]
    y_dir_train, y_dir_test = y_dir[:train_end_local], y_dir[train_end_local:]
    y_ret_train, y_ret_test = y_ret[:train_end_local], y_ret[train_end_local:]

    # ---- GBM: identical param_grid + GridSearchCV/TimeSeriesSplit ----------
    tscv = TimeSeriesSplit(n_splits=config['gbm']['cv_splits'])
    param_grid = config['gbm']['param_grid']
    gs_clf = GridSearchCV(
        xgb.XGBClassifier(device=XGB_DEVICE, eval_metric='auc', random_state=random_state, verbosity=0),
        param_grid=param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1,
    )
    gs_clf.fit(X_train_s, y_dir_train)
    gbm_prob_test = gs_clf.best_estimator_.predict_proba(X_test_s)[:, 1]
    gbm_pred_dir_test = (gbm_prob_test >= 0.5).astype(int)

    gs_reg = GridSearchCV(
        xgb.XGBRegressor(device=XGB_DEVICE, objective='reg:pseudohubererror', random_state=random_state, verbosity=0),
        param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    gs_reg.fit(X_train_s, y_ret_train)
    gbm_pred_ret_test = gs_reg.best_estimator_.predict(X_test_s)

    # ---- LSTM: identical architecture/hyperparameters from config.json ----
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    lstm_train_end_local = _lstm_es_tail_end(train_end_local, config)
    time_steps = config['lstm']['time_steps']
    X_lstm_train_s = X_scaled[:lstm_train_end_local]
    X_lstm_es_s = X_scaled[lstm_train_end_local:train_end_local]
    Xtr_seq, ytr_ret_seq, ytr_dir_seq = _create_block_sequences(
        X_lstm_train_s, y_ret[:lstm_train_end_local], y_dir[:lstm_train_end_local], time_steps)
    Xes_seq, yes_ret_seq, yes_dir_seq = _create_block_sequences(
        X_lstm_es_s, y_ret[lstm_train_end_local:train_end_local], y_dir[lstm_train_end_local:train_end_local], time_steps)
    Xte_seq, yte_ret_seq, yte_dir_seq = _create_block_sequences(X_test_s, y_ret_test, y_dir_test, time_steps)

    tf.random.set_seed(random_state)
    inputs = Input(shape=(Xtr_seq.shape[1], Xtr_seq.shape[2]))
    shared = LSTM(config['lstm']['units'])(inputs)
    shared = Dropout(config['lstm']['dropout'])(shared)
    ret_out = Dense(1, activation='linear', name='return_output')(shared)
    dir_out = Dense(1, activation='sigmoid', name='direction_output')(shared)
    model = Model(inputs, [ret_out, dir_out])
    model.compile(optimizer=Adam(learning_rate=config['lstm']['learning_rate']),
                  loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
                  loss_weights=config['lstm']['loss_weights'])
    es = EarlyStopping(monitor='val_loss', patience=config['lstm']['patience'],
                       restore_best_weights=True, verbose=0)
    model.fit(Xtr_seq, {'return_output': ytr_ret_seq, 'direction_output': ytr_dir_seq},
             validation_data=(Xes_seq, {'return_output': yes_ret_seq, 'direction_output': yes_dir_seq}),
             epochs=config['lstm']['epochs'], batch_size=config['lstm']['batch_size'],
             callbacks=[es], verbose=0)
    lstm_pred_ret_test, lstm_prob_test = (p.ravel() for p in model.predict(Xte_seq, verbose=0))
    lstm_pred_dir_test = (lstm_prob_test >= 0.5).astype(int)

    return {
        'majority_class': int(round(y_dir_train.mean())),
        'y_dir_test': y_dir_test, 'y_ret_test': y_ret_test,
        'gbm_pred_dir_test': gbm_pred_dir_test, 'gbm_pred_ret_test': gbm_pred_ret_test,
        'gbm_prob_test': gbm_prob_test,
        'y_dir_test_lstm': yte_dir_seq, 'y_ret_test_lstm': yte_ret_seq,
        'lstm_pred_dir_test': lstm_pred_dir_test, 'lstm_pred_ret_test': lstm_pred_ret_test,
        'lstm_prob_test': lstm_prob_test,
    }


# ── volatility: 5-seed ensemble + GARCH, one window ─────────────────────────

def train_volatility_window(window_df, train_end_local, config, seeds=VOL_ENSEMBLE_SEEDS,
                            random_state=RANDOM_STATE):
    """The exact 5-seed 3-head multi-task LSTM ensemble procedure
    (`src.volatility.train_production_volatility_model`'s architecture,
    losses, and `make_sequences` GLOBAL windowing convention — reused
    verbatim) on ONE walk-forward window, entirely in-memory (no `.keras`
    file is ever written, unlike the production function this mirrors,
    which hardcodes persisting under `models/volatility/`). GARCH(1,1) and
    the persistence baseline are refit fresh on this window too, via
    `src.volatility.garch_forecasts`/`persistence_forecasts` UNCHANGED."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    lag_scaler, lag_pca = fit_lag_pca(window_df.iloc[:train_end_local], lag_columns=LAG_COLUMNS,
                                      variance_threshold=config['pca']['variance_threshold'])
    reduced = apply_lag_pca(window_df, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    model_cols = model_input_columns(lag_pca, base_columns=PRICE_FEATURE_COLUMNS, lag_columns=LAG_COLUMNS)
    scaler = StandardScaler().fit(reduced[model_cols].iloc[:train_end_local])
    X_scaled = scaler.transform(reduced[model_cols])

    y_vol = reduced[TARGET_VOLATILITY_COLUMN].to_numpy()
    y_ret = reduced[TARGET_RETURN_COLUMN].to_numpy()
    y_dir = reduced[TARGET_DIRECTION_COLUMN].to_numpy()

    lstm_cfg = config['lstm']
    time_steps = lstm_cfg['time_steps']
    lstm_train_end_local = _lstm_es_tail_end(train_end_local, config)

    windows_v, targets_vol, target_rows = make_sequences(X_scaled, y_vol, time_steps)
    targets_ret = y_ret[target_rows]
    targets_dir = y_dir[target_rows]
    is_tr = target_rows < lstm_train_end_local
    is_es = (target_rows >= lstm_train_end_local) & (target_rows < train_end_local)
    is_te = target_rows >= train_end_local

    seed_test_preds = []
    for seed in seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        inputs = Input(shape=(time_steps, X_scaled.shape[1]))
        shared = LSTM(lstm_cfg['units'])(inputs)
        shared = Dropout(lstm_cfg['dropout'])(shared)
        ret_out = Dense(1, activation='linear', name='return_output')(shared)
        dir_out = Dense(1, activation='sigmoid', name='direction_output')(shared)
        vol_out = Dense(1, activation='softplus', name='volatility_output')(shared)
        model = Model(inputs, [ret_out, dir_out, vol_out])
        model.compile(
            optimizer=Adam(learning_rate=lstm_cfg['learning_rate']),
            loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy', 'volatility_output': 'mse'},
            loss_weights={'return_output': 1.0, 'direction_output': 1.0, 'volatility_output': 1.0},
        )
        es = EarlyStopping(monitor='val_loss', patience=lstm_cfg['patience'],
                           restore_best_weights=True, verbose=0)
        model.fit(
            windows_v[is_tr],
            {'return_output': targets_ret[is_tr], 'direction_output': targets_dir[is_tr],
             'volatility_output': targets_vol[is_tr]},
            validation_data=(windows_v[is_es],
                            {'return_output': targets_ret[is_es], 'direction_output': targets_dir[is_es],
                             'volatility_output': targets_vol[is_es]}),
            epochs=lstm_cfg['epochs'], batch_size=lstm_cfg['batch_size'], callbacks=[es], verbose=0)
        _r, _d, pv = model.predict(windows_v[is_te], verbose=0)
        seed_test_preds.append(pv.ravel())

    pred_ens_test = np.mean(seed_test_preds, axis=0)
    y_vol_test = targets_vol[is_te]

    split = {'train_end': train_end_local, 'n': len(window_df)}
    pred_garch_full, _garch_res = garch_forecasts(window_df, split, config)
    pred_pers_full = persistence_forecasts(window_df)
    test_rows_abs = target_rows[is_te]
    pred_garch_test = pred_garch_full[test_rows_abs]
    pred_pers_test = pred_pers_full[test_rows_abs]

    return {
        'y_vol_test': y_vol_test, 'pred_ens_test': pred_ens_test,
        'pred_garch_test': pred_garch_test, 'pred_pers_test': pred_pers_test,
    }


# ── moving-block (circular) bootstrap over pooled OOS rows ─────────────────

def _circular_block_bootstrap_indices(n, block_len, rng):
    block_len = max(1, min(block_len, n))
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend((start + k) % n for k in range(block_len))
    return np.array(idx[:n])


def _mcnemar_exact(correct_a, correct_b):
    from math import comb
    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * p)


def pooled_direction_bootstrap(y_true, pred_challenger, pred_baseline, block_len,
                               n_boot=BOOTSTRAP_RESAMPLES, alpha=0.05, random_state=RANDOM_STATE):
    """Moving-BLOCK (circular) paired bootstrap of (acc_challenger -
    acc_baseline) over the POOLED cross-window OOS rows, block length = one
    window's own test-row count — adjacent windows' test days are NOT
    independent (they may even overlap in market regime), so an i.i.d.
    bootstrap would understate the true uncertainty, same reasoning already
    applied to the vol-scaled-sizing backtest. McNemar (exact on the observed
    table) is reported alongside as corroboration."""
    y_true = np.asarray(y_true)
    pred_challenger = np.asarray(pred_challenger)
    pred_baseline = np.asarray(pred_baseline)
    correct_chg = (pred_challenger == y_true)
    correct_base = (pred_baseline == y_true)
    acc_chg, acc_base = float(correct_chg.mean()), float(correct_base.mean())

    n = len(y_true)
    rng = np.random.default_rng(random_state)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_len, rng)
        deltas[i] = correct_chg[idx].mean() - correct_base[idx].mean()
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    b, c, mcp = _mcnemar_exact(correct_base, correct_chg)
    return {
        'n_pooled': n, 'acc_challenger': round(acc_chg, 4), 'acc_baseline': round(acc_base, 4),
        'delta_acc': round(acc_chg - acc_base, 4),
        'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': round(mcp, 4),
        'cleared': bool(lo > 0 and mcp < alpha),
    }


def pooled_regression_bootstrap(y_true, pred_challenger, pred_baseline, block_len,
                                n_boot=BOOTSTRAP_RESAMPLES, alpha=0.05, random_state=RANDOM_STATE):
    """Moving-BLOCK analogue of `src.volatility.bootstrap_delta` for the
    pooled volatility-ensemble-vs-GARCH comparison: paired bootstrap of
    (MAE improvement, R2 improvement) using block resampling instead of
    i.i.d., same clustering rationale as `pooled_direction_bootstrap`."""
    y_true = np.asarray(y_true)
    pred_challenger = np.asarray(pred_challenger)
    pred_baseline = np.asarray(pred_baseline)
    mae_chg = mean_absolute_error(y_true, pred_challenger)
    mae_base = mean_absolute_error(y_true, pred_baseline)
    r2_chg = r2_score(y_true, pred_challenger)
    r2_base = r2_score(y_true, pred_baseline)

    n = len(y_true)
    rng = np.random.default_rng(random_state)
    d_mae = np.empty(n_boot)
    d_r2 = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_len, rng)
        yt = y_true[idx]
        d_mae[i] = mean_absolute_error(yt, pred_baseline[idx]) - mean_absolute_error(yt, pred_challenger[idx])
        d_r2[i] = r2_score(yt, pred_challenger[idx]) - r2_score(yt, pred_baseline[idx])
    lo_mae, hi_mae = np.percentile(d_mae, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lo_r2, hi_r2 = np.percentile(d_r2, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        'n_pooled': n, 'mae_ensemble': round(mae_chg, 6), 'mae_garch': round(mae_base, 6),
        'r2_ensemble': round(r2_chg, 4), 'r2_garch': round(r2_base, 4),
        'delta_mae_ci_low': round(float(lo_mae), 6), 'delta_mae_ci_high': round(float(hi_mae), 6),
        'delta_r2_ci_low': round(float(lo_r2), 4), 'delta_r2_ci_high': round(float(hi_r2), 4),
        'cleared': bool(lo_mae > 0 and lo_r2 > 0),
    }


# ── one full window, both variants + volatility ─────────────────────────────

def run_one_window(feat, window, config, random_state=RANDOM_STATE):
    window_df, train_end_local = _slice_window(feat, window)
    n_train, n_test = train_end_local, len(window_df) - train_end_local

    direction_results = {
        variant: train_direction_variant_window(window_df, variant, train_end_local, config, random_state)
        for variant in VARIANTS
    }
    vol_result = train_volatility_window(window_df, train_end_local, config, random_state=random_state)

    row = {
        'train_start': window['train_start'].date().isoformat(),
        'train_end': window['train_end'].date().isoformat(),
        'test_end': window['test_end'].date().isoformat(),
        'n_train': n_train, 'n_test': n_test,
    }
    for variant in VARIANTS:
        r = direction_results[variant]
        maj_pred = np.full(len(r['y_dir_test']), r['majority_class'])
        row[f'{variant}_gbm_acc'] = round(accuracy_score(r['y_dir_test'], r['gbm_pred_dir_test']), 4)
        row[f'{variant}_gbm_auc'] = round(roc_auc_score(r['y_dir_test'], r['gbm_prob_test']), 4) \
            if len(np.unique(r['y_dir_test'])) == 2 else float('nan')
        row[f'{variant}_gbm_majority_acc'] = round(accuracy_score(r['y_dir_test'], maj_pred), 4)
        row[f'{variant}_gbm_ret_mae'] = round(mean_absolute_error(r['y_ret_test'], r['gbm_pred_ret_test']), 6)

        maj_pred_lstm = np.full(len(r['y_dir_test_lstm']), r['majority_class'])
        row[f'{variant}_lstm_acc'] = round(accuracy_score(r['y_dir_test_lstm'], r['lstm_pred_dir_test']), 4)
        row[f'{variant}_lstm_auc'] = round(roc_auc_score(r['y_dir_test_lstm'], r['lstm_prob_test']), 4) \
            if len(np.unique(r['y_dir_test_lstm'])) == 2 else float('nan')
        row[f'{variant}_lstm_majority_acc'] = round(accuracy_score(r['y_dir_test_lstm'], maj_pred_lstm), 4)
        row[f'{variant}_lstm_ret_mae'] = round(mean_absolute_error(r['y_ret_test_lstm'], r['lstm_pred_ret_test']), 6)

    row['vol_ensemble_mae'] = round(mean_absolute_error(vol_result['y_vol_test'], vol_result['pred_ens_test']), 6)
    row['vol_ensemble_r2'] = round(r2_score(vol_result['y_vol_test'], vol_result['pred_ens_test']), 4)
    row['vol_garch_mae'] = round(mean_absolute_error(vol_result['y_vol_test'], vol_result['pred_garch_test']), 6)
    row['vol_garch_r2'] = round(r2_score(vol_result['y_vol_test'], vol_result['pred_garch_test']), 4)
    row['vol_persistence_mae'] = round(mean_absolute_error(vol_result['y_vol_test'], vol_result['pred_pers_test']), 6)
    row['vol_persistence_r2'] = round(r2_score(vol_result['y_vol_test'], vol_result['pred_pers_test']), 4)

    return row, direction_results, vol_result


# ── orchestration ─────────────────────────────────────────────────────────

def _dist(series):
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {'mean': float('nan'), 'median': float('nan'), 'min': float('nan'),
               'max': float('nan'), 'std': float('nan')}
    return {'mean': float(np.mean(x)), 'median': float(np.median(x)), 'min': float(np.min(x)),
           'max': float(np.max(x)), 'std': float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}


def _time_trend(window_end_dates, metric_series):
    """Spearman rho of window-end-date (ordinal) vs. the metric — a direct
    trend statistic for degradation/drift in more recent windows, not
    eyeballing a chart."""
    x = np.asarray([pd.Timestamp(d).toordinal() for d in window_end_dates], dtype=float)
    y = np.asarray(metric_series, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return float('nan'), float('nan')
    rho, p = spearmanr(x[valid], y[valid])
    return float(rho), float(p)


def run(base_dir='', config_path='config.json', out_log=WALK_FORWARD_LOG,
       out_summary=WALK_FORWARD_SUMMARY_LOG, random_state=RANDOM_STATE, register=True,
       max_windows=None):
    with open(_p(base_dir, config_path)) as f:
        config = json.load(f)

    print('=' * 78)
    print('WALK-FORWARD VALIDATION — research-only robustness report (NOT production)')
    print('  HARD BOUNDARY: no models/, _train_pipeline.py, inference.py, or config.json')
    print('  file is touched. Descriptive report only; scheduled production retraining,')
    print('  if ever built, is a SEPARATE subsequent conversation.')
    print('=' * 78)

    feat = load_engineered_history(config, base_dir=base_dir)
    windows = compute_windows(feat.index)
    if max_windows is not None:
        windows = windows[:max_windows]
    print(f'\n  euro-era engineered history: {len(feat):,} rows, {feat.index.min().date()} -> '
          f'{feat.index.max().date()}')
    print(f'  scheme: {TRAIN_WINDOW_YEARS}yr trailing train / {TEST_WINDOW_YEARS}yr step+test, '
          f'annual cadence -> {len(windows)} windows')

    if not windows:
        print('\n  NO WINDOWS FIT the available history -- aborting (nothing logged).')
        return None

    # ---- REAL timed first window, then an honest extrapolated estimate ----
    t0 = time.time()
    first_row, first_dir, first_vol = run_one_window(feat, windows[0], config, random_state)
    first_window_seconds = time.time() - t0
    estimated_total = first_window_seconds * len(windows)
    print(f'\n  window 1 REAL measured time: {first_window_seconds:.1f}s')
    print(f'  -> extrapolated total for all {len(windows)} windows: {estimated_total:.1f}s '
          f'({estimated_total / 60:.1f} min)')

    if estimated_total > MAX_ESTIMATED_TOTAL_SECONDS:
        print(f'\n  STOPPING: extrapolated total ({estimated_total / 3600:.2f}h) exceeds the '
              f'{MAX_ESTIMATED_TOTAL_SECONDS / 3600:.1f}h practicality ceiling for a single-sitting '
              f'research report. Nothing logged. Re-run with a coarser schedule or accept a '
              f'longer background run explicitly -- this is NOT silently coarsened.')
        return None

    rows = [first_row]
    per_window_direction = {v: [first_dir[v]] for v in VARIANTS}
    per_window_vol = [first_vol]
    for i, window in enumerate(windows[1:], start=2):
        t_w = time.time()
        row, dir_res, vol_res = run_one_window(feat, window, config, random_state)
        rows.append(row)
        for v in VARIANTS:
            per_window_direction[v].append(dir_res[v])
        per_window_vol.append(vol_res)
        print(f'  window {i}/{len(windows)} ({row["train_start"]} -> {row["test_end"]}): '
              f'{time.time() - t_w:.1f}s')

    detail = pd.DataFrame(rows)
    detail.insert(0, 'window_id', range(1, len(detail) + 1))
    detail = detail[PER_WINDOW_COLUMNS]
    out_path = _p(base_dir, out_log)
    if register:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        detail.to_csv(out_path, index=False)
        print(f'\nLogged per-window detail: {out_path}')

    # ---- Aggregate: per-window distributions + time-trend -----------------
    print('\n' + '=' * 78)
    print('AGGREGATE ACROSS ALL WINDOWS')
    print('=' * 78)
    summary_rows = []
    metric_cols = [c for c in PER_WINDOW_COLUMNS if c not in
                  ('window_id', 'train_start', 'train_end', 'test_end', 'n_train', 'n_test')]
    for col in metric_cols:
        dist = _dist(detail[col])
        rho, p = _time_trend(detail['test_end'], detail[col])
        print(f'  {col:28s} mean={dist["mean"]:+.4f} median={dist["median"]:+.4f} '
              f'min={dist["min"]:+.4f} max={dist["max"]:+.4f} std={dist["std"]:.4f}  '
              f'| time-trend rho={rho:+.3f} p={p:.3f}')
        summary_rows.append({
            'metric': col, 'mean': dist['mean'], 'median': dist['median'], 'min': dist['min'],
            'max': dist['max'], 'std': dist['std'], 'time_trend_rho': rho, 'time_trend_p': p,
        })

    # ---- Pooled OOS: moving-block bootstrap (block = 1 window's test rows) --
    print('\n--- Pooled out-of-sample direction (moving-block bootstrap, McNemar corroborating) ---')
    pooled = {}
    for v in VARIANTS:
        y_all = np.concatenate([r['y_dir_test'] for r in per_window_direction[v]])
        maj_all = np.concatenate([np.full(len(r['y_dir_test']), r['majority_class']) for r in per_window_direction[v]])
        gbm_all = np.concatenate([r['gbm_pred_dir_test'] for r in per_window_direction[v]])
        block_len_gbm = int(np.median([len(r['y_dir_test']) for r in per_window_direction[v]]))
        res_gbm = pooled_direction_bootstrap(y_all, gbm_all, maj_all, block_len=block_len_gbm,
                                             random_state=random_state)
        pooled[f'{v}_gbm'] = res_gbm
        print(f'  [{v}][GBM] n_pooled={res_gbm["n_pooled"]}  acc={res_gbm["acc_challenger"]:.4f} '
              f'vs majority={res_gbm["acc_baseline"]:.4f}  delta={res_gbm["delta_acc"]:+.4f}  '
              f'CI[{res_gbm["ci_low"]:+.4f}, {res_gbm["ci_high"]:+.4f}]  McNemar p={res_gbm["mcnemar_p"]:.4f}  '
              f'{"CLEARED" if res_gbm["cleared"] else "not cleared"}')

        y_all_l = np.concatenate([r['y_dir_test_lstm'] for r in per_window_direction[v]])
        maj_all_l = np.concatenate([np.full(len(r['y_dir_test_lstm']), r['majority_class']) for r in per_window_direction[v]])
        lstm_all = np.concatenate([r['lstm_pred_dir_test'] for r in per_window_direction[v]])
        block_len_lstm = int(np.median([len(r['y_dir_test_lstm']) for r in per_window_direction[v]]))
        res_lstm = pooled_direction_bootstrap(y_all_l, lstm_all, maj_all_l, block_len=block_len_lstm,
                                              random_state=random_state)
        pooled[f'{v}_lstm'] = res_lstm
        print(f'  [{v}][LSTM] n_pooled={res_lstm["n_pooled"]}  acc={res_lstm["acc_challenger"]:.4f} '
              f'vs majority={res_lstm["acc_baseline"]:.4f}  delta={res_lstm["delta_acc"]:+.4f}  '
              f'CI[{res_lstm["ci_low"]:+.4f}, {res_lstm["ci_high"]:+.4f}]  McNemar p={res_lstm["mcnemar_p"]:.4f}  '
              f'{"CLEARED" if res_lstm["cleared"] else "not cleared"}')

    print('\n--- Pooled out-of-sample volatility (moving-block bootstrap, ensemble vs GARCH) ---')
    y_vol_all = np.concatenate([r['y_vol_test'] for r in per_window_vol])
    ens_all = np.concatenate([r['pred_ens_test'] for r in per_window_vol])
    garch_all = np.concatenate([r['pred_garch_test'] for r in per_window_vol])
    block_len_vol = int(np.median([len(r['y_vol_test']) for r in per_window_vol]))
    res_vol = pooled_regression_bootstrap(y_vol_all, ens_all, garch_all, block_len=block_len_vol,
                                          random_state=random_state)
    pooled['volatility'] = res_vol
    print(f'  n_pooled={res_vol["n_pooled"]}  MAE ensemble={res_vol["mae_ensemble"]:.6f} vs '
          f'GARCH={res_vol["mae_garch"]:.6f}  R2 ensemble={res_vol["r2_ensemble"]:+.4f} vs '
          f'GARCH={res_vol["r2_garch"]:+.4f}  '
          f'dMAE CI[{res_vol["delta_mae_ci_low"]:+.6f}, {res_vol["delta_mae_ci_high"]:+.6f}]  '
          f'dR2 CI[{res_vol["delta_r2_ci_low"]:+.4f}, {res_vol["delta_r2_ci_high"]:+.4f}]  '
          f'{"CLEARED" if res_vol["cleared"] else "not cleared"}')

    print('\n  REMINDER: this report answers "would our EXISTING approach have worked robustly '
          'over time," not "should we change anything." Any decision to build scheduled '
          'production retraining is a separate, subsequent, explicitly-approved conversation.')

    summary_path = _p(base_dir, out_summary)
    if register:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f'\nLogged aggregate summary: {summary_path}')

    return {
        'detail': detail, 'summary_rows': summary_rows, 'pooled': pooled,
        'n_windows': len(windows), 'first_window_seconds': first_window_seconds,
        'estimated_total_seconds': estimated_total,
    }


if __name__ == '__main__':
    run()
