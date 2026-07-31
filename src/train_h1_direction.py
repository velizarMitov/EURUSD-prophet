"""
H_dir.1 PRODUCTION TRAINER — full-history, observational serving.

WHAT SHIPS AND WHAT IT IS WORTH
-------------------------------
H_dir.1 cleared its reserved test block on the [0:70%] model: 52.96% vs 49.85%
next-H1-bar direction, +3.10pp, block CI [+0.0114, +0.0497], McNemar p = 1.7e-05,
on 10,403 rows it had never seen. That confirmation belongs to THAT model.

This trainer ships a DIFFERENT model: the same architecture and the same
hyperparameters refit on the FULL available history, including the rows that
served as that test block. It therefore has NO out-of-sample confirmation of its
own and `validated_out_of_sample` is recorded as FALSE. Its only honest evidence
is its own forward ledger.

BUILD-TIME PIPELINE CHECK (mandatory, and the reason this file is safe)
----------------------------------------------------------------------
Because the shipped model has no held-out data, it has no reproduction gate of
its own -- a broken feature extraction would produce a confidently wrong model
with nothing to catch it. So before training it, this trainer FIRST fits the
[0:70%] model through the IDENTICAL pipeline and asserts its validation accuracy
reproduces 0.527462 within +/- 0.003. That proves the feature extraction, split,
standardiser and fit path are intact end to end. Only then is the full-history
model trained, with the same pipeline, and only ITS artifacts are written. The
[0:70%] model is a CHECK: never served, never persisted.

If the check fails, NO artifacts are written.

RETRAINING GUIDANCE (recorded here so it is not lost)
-----------------------------------------------------
One week adds ~120 bars to ~69,000 (+0.17%) and will not move the model. Retrain
when the increment is MATERIAL -- roughly annually (+12%) -- or when there is a
specific reason to suspect a regime change. Every retrain starts that version's
forward evidence from ZERO, so retraining frequently means never accumulating
enough forward observations to learn anything. At this effect size a version
needs on the order of 1,000 settled observations before its ledger says anything.

NO TRADING FRAME. This trainer produces a probability. It contains no order
placement, no execution, no position sizing, no stop-loss, no take-profit, no
leverage and no risk limits, and nothing here converts a prediction into an
action. The ledger downstream is simulated.

BOUNDARIES: writes ONLY models/h1_direction/. Never modifies models/baseline/,
models/with_macro/, models/volatility/, _train_pipeline.py, src/features.py,
src/paper_trading.py, src/pooled_h1_model.py, src/h1_direction_model.py, or any
results/*hypothesis_log.csv.
"""

import json
import os
import random

import joblib
import numpy as np
import pandas as pd

# THE canonical feature contract -- the same module the serving path imports.
from src.h1_features import (
    DIRECTION_FEATURE_COLUMNS, apply_direction_standardizer,
    compute_h1_direction_features, fit_direction_standardizer,
)

H1_CACHE = 'results/pooled_h1/EURUSD_h1.csv'      # the cache H_dir.1 was validated on
MODEL_DIR = 'models/h1_direction'
GBM_PATH = os.path.join(MODEL_DIR, 'h1_direction_gbm.json')
SCALER_PATH = os.path.join(MODEL_DIR, 'h1_direction_scaler.pkl')
META_PATH = os.path.join(MODEL_DIR, 'h1_direction_meta.json')

# ── Frozen pre-registered constants. Identical to the validated model. ──
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85
EMBARGO_BARS = 24
RANDOM_SEED = 42

# The build-time pipeline check.
REPRO_TARGET_ACC = 0.527462
REPRO_TOLERANCE = 0.003

PROVENANCE_NOTE = (
    "Trained on the full available history. The out-of-sample confirmation "
    "(52.96% vs 49.85%, +3.10pp, block CI [+0.0114,+0.0497], McNemar p=1.7e-05) "
    "belongs to the [0:70%] model and does NOT transfer to this one, which has "
    "seen that test block during training. This model's only honest evidence is "
    "its own forward ledger."
)


class PipelineCheckError(RuntimeError):
    """The build-time [0:70%] reproduction check failed. NO artifacts are
    written: a full-history model built on a broken pipeline would be silently
    wrong with nothing to catch it."""


def load_h1_cache(path: str = H1_CACHE) -> pd.DataFrame:
    """The cached EURUSD H1 OHLC frame, UTC-indexed. Never refetches."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Cached EURUSD H1 data not found at {path!r}. This trainer does not '
            'fetch; run the acquisition step first.')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = (df.index.tz_localize('UTC') if df.index.tz is None
                else df.index.tz_convert('UTC'))
    return df[['open', 'high', 'low', 'close']].sort_index()


def build_direction_dataset(df: pd.DataFrame):
    """
    Features from a COMPLETED bar -> direction of the NEXT bar.

        y[t] = 1 if log(close[t+1]/close[t]) > 0
        y[t] = 0 if log(close[t+1]/close[t]) < 0
        rows with EXACTLY zero next-bar return are DROPPED (a zero move has no
        direction to predict), as are the feature warm-up rows and the final bar.

    Returns (frame, target_index). A 1-bar-ahead label has no overlap with its
    neighbours, so label uniqueness is 1.0 by construction.
    """
    feats = compute_h1_direction_features(df)
    log_close = np.log(df['close'])
    fwd = log_close.shift(-1) - log_close            # reads close[t+1] ONLY

    out = feats.copy()
    out['fwd_logret'] = fwd
    out = out.dropna(subset=list(DIRECTION_FEATURE_COLUMNS) + ['fwd_logret'])

    zero = out['fwd_logret'].to_numpy() == 0.0
    out['label'] = np.where(zero, np.nan, (out['fwd_logret'].to_numpy() > 0.0).astype(float))
    return out, out.index[~zero]


def split_purge_embargo(target_index):
    """The validated chronological split: train [0:70%], validation [70:85%],
    test [85:100%]. Purge the final training row (its label is the return into
    the first validation bar); embargo the first 24 validation rows."""
    n = len(target_index)
    train_end, val_end = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    train, val = target_index[:train_end], target_index[train_end:val_end]
    return {'train': train[:-1] if len(train) else train,
            'val': val[min(EMBARGO_BARS, len(val)):],
            'test': target_index[val_end:]}


def _balanced_scale_pos_weight(y) -> float:
    pos = float((np.asarray(y) == 1).sum())
    neg = float((np.asarray(y) == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def seed_everything(seed: int = RANDOM_SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def train_gbm(X, y, seed: int = RANDOM_SEED):
    """
    The validated booster. Hyperparameters are FIXED and are not to be tuned to
    suit the larger sample: tuning on data with no held-out slice left has no
    honest stopping rule.
    """
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', device='cuda',
        scale_pos_weight=_balanced_scale_pos_weight(y),
        random_state=seed, n_jobs=0,
    )
    clf.fit(X, y)                                   # NO early_stopping, NO eval_set
    return clf


def _matrix(frame: pd.DataFrame, index, scaler):
    d = frame.loc[index]
    X = apply_direction_standardizer(d[DIRECTION_FEATURE_COLUMNS].to_numpy(float), scaler)
    return X, d['label'].to_numpy(float).astype(int)


def build_time_pipeline_check(frame: pd.DataFrame, target_index,
                              seed: int = RANDOM_SEED) -> dict:
    """
    Fit the [0:70%] model through the IDENTICAL pipeline and assert it reproduces
    the registered validation accuracy. This is the ONLY thing standing between a
    silently broken feature extraction and a shipped full-history model, because
    the shipped model has no held-out data of its own.

    Raises PipelineCheckError on drift. The model fitted here is discarded.
    """
    splits = split_purge_embargo(target_index)
    scaler = fit_direction_standardizer(
        frame.loc[splits['train'], DIRECTION_FEATURE_COLUMNS].to_numpy(float))
    X_tr, y_tr = _matrix(frame, splits['train'], scaler)
    X_va, y_va = _matrix(frame, splits['val'], scaler)

    clf = train_gbm(X_tr, y_tr, seed=seed)
    acc = float(((clf.predict_proba(X_va)[:, 1] >= 0.5).astype(int) == y_va).mean())
    delta = abs(acc - REPRO_TARGET_ACC)
    if delta > REPRO_TOLERANCE:
        raise PipelineCheckError(
            f'build-time [0:70%] check FAILED: validation accuracy {acc:.6f} vs '
            f'registered {REPRO_TARGET_ACC:.6f}, |diff| {delta:.6f} > '
            f'{REPRO_TOLERANCE}. NO artifacts written -- a full-history model '
            'built on a broken pipeline would be silently wrong with nothing to '
            'catch it.')
    return {'check_val_accuracy': acc, 'check_abs_delta': delta,
            'check_target': REPRO_TARGET_ACC, 'check_tolerance': REPRO_TOLERANCE,
            'check_n_train': int(len(splits['train'])),
            'check_n_val': int(len(splits['val'])), 'passed': True}


def model_version(train_end: pd.Timestamp, n_rows: int) -> str:
    """A stable stamp. Every retrain produces a DIFFERENT model and the forward
    ledger must never mix them silently, so the version carries both the last bar
    trained on and the row count."""
    return f'h1dir-full-{pd.Timestamp(train_end):%Y%m%d}-{int(n_rows)}'


def train(cache_path: str = H1_CACHE, model_dir: str = MODEL_DIR,
          seed: int = RANDOM_SEED, write: bool = True, verbose: bool = True) -> dict:
    """Build-time check first, then the full-history fit. Artifacts only on pass."""
    seed_everything(seed)
    raw = load_h1_cache(cache_path)
    frame, target_index = build_direction_dataset(raw)

    if verbose:
        print(f'  labelled rows: {len(target_index)}  '
              f'{target_index[0]:%Y-%m-%d %H:%M} .. {target_index[-1]:%Y-%m-%d %H:%M}')
        print('  running build-time [0:70%] pipeline check ...')
    check = build_time_pipeline_check(frame, target_index, seed=seed)
    if verbose:
        print(f"  CHECK PASSED: val acc {check['check_val_accuracy']:.6f} vs "
              f"{REPRO_TARGET_ACC:.6f} (|diff| {check['check_abs_delta']:.6f})")

    # ── the shipped model: ALL labelled rows, same pipeline, same hyperparameters ──
    scaler = fit_direction_standardizer(
        frame.loc[target_index, DIRECTION_FEATURE_COLUMNS].to_numpy(float))
    X, y = _matrix(frame, target_index, scaler)
    clf = train_gbm(X, y, seed=seed)

    train_start, train_end = target_index[0], target_index[-1]
    version = model_version(train_end, len(target_index))
    meta = {
        'model_version': version,
        'train_start': pd.Timestamp(train_start).isoformat(),
        'train_end': pd.Timestamp(train_end).isoformat(),
        'n_train_rows': int(len(target_index)),
        'trained_at_utc': pd.Timestamp.utcnow().isoformat(),
        'seed': int(seed),
        'feature_columns': list(DIRECTION_FEATURE_COLUMNS),
        'validated_out_of_sample': False,
        'provenance_note': PROVENANCE_NOTE,
        'hyperparameters': {
            'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0,
            'tree_method': 'hist', 'device': 'cuda',
            'scale_pos_weight': 'balanced', 'early_stopping': False,
            'eval_set': False,
        },
        'build_time_pipeline_check': check,
        'source_cache': cache_path,
        'train_class_balance_pct_up': round(100.0 * float((y == 1).mean()), 6),
    }

    if write:
        os.makedirs(model_dir, exist_ok=True)
        clf.save_model(os.path.join(model_dir, os.path.basename(GBM_PATH)))
        joblib.dump(scaler, os.path.join(model_dir, os.path.basename(SCALER_PATH)))
        with open(os.path.join(model_dir, os.path.basename(META_PATH)), 'w',
                  encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
        if verbose:
            print(f'  wrote {model_dir}/ -> {version}  ({len(target_index)} rows)')
    return {'meta': meta, 'check': check, 'model': clf, 'scaler': scaler,
            'n_full': int(len(target_index)),
            'n_train_70': check['check_n_train']}


if __name__ == '__main__':
    print('=' * 74)
    print('H_dir.1 PRODUCTION TRAINER — full history, observational')
    print('=' * 74)
    out = train()
    m = out['meta']
    print()
    print(f"  model_version            : {m['model_version']}")
    print(f"  train_start / train_end  : {m['train_start']} .. {m['train_end']}")
    print(f"  n_train_rows             : {m['n_train_rows']} "
          f"(vs {out['n_train_70']} in the [0:70%] check model)")
    print(f"  validated_out_of_sample  : {m['validated_out_of_sample']}")
    print(f"  build-time check         : {m['build_time_pipeline_check']['check_val_accuracy']:.6f}")
