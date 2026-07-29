"""
H1 NEXT-BAR DIRECTION — a NEW hypothesis family (research-only).

WHAT THIS TESTS
---------------
Every prior H1 program in this project used a triple-barrier target with a
120-bar horizon. The label-geometry scan (results/h1_horizon_feasibility.csv)
showed why they all failed: overlapping 120-bar labels reduced 192,560 pooled
rows to ~2,164 INDEPENDENT observations -- three orders of magnitude below what
sequential deep learning needs. Those models never had a fair chance.

This program drops the triple-barrier target entirely and asks the plain
question: can a model predict the DIRECTION of the NEXT H1 bar?

The decisive property: a 1-bar-ahead label has NO overlap with its neighbours,
so mean label uniqueness = 1.0 BY CONSTRUCTION. Every training row is one
independent observation. For the first time in this project the sample size is
not the binding constraint. Expected outcome is still near-chance (50-51%) --
the point is to MEASURE it under conditions where a model could in principle
learn something, instead of drowning in ~2,000 observations.

NO P&L. NO position sizing. NO transaction costs. NO backtest. NO trading
simulation of any kind. The arbiter is prediction accuracy against a baseline
and nothing else. Profit, pips, Sharpe and returns are deliberately absent from
this module.

HARD BOUNDARY -- research only, production untouched
----------------------------------------------------
Writes ONLY results/h1_direction_hypothesis_log.csv. Never modifies models/,
_train_pipeline.py, src/inference.py, src/features.py, src/paper_trading.py,
config.json, results/eurusd_h1.csv, src/triple_barrier.py, src/pooled_h1_data.py,
src/pooled_h1_model.py, src/h1_horizon_feasibility.py, or any existing
results/*hypothesis_log.csv. A unit test hashes the whole protected set
before/after a full run().

Reads the ALREADY-CACHED results/pooled_h1/EURUSD_h1.csv. It never refetches
from MT5; if the cache is missing it STOPS (MissingCachedDataError).

PRE-REGISTERED DESIGN (fixed before any result was looked at)
-------------------------------------------------------------
* Instrument -- EURUSD ONLY. Pooled multi-instrument training is deliberately
  OUT OF SCOPE: it was already tested and DROPped for the triple-barrier target,
  and re-testing it against this NEW target would be a separate pre-registered
  hypothesis carrying its own Bonferroni cost.
* Target -- y[t] = 1 if log(close[t+1]/close[t]) > 0, 0 if < 0. Rows whose
  next-bar log return is EXACTLY 0 are DROPPED (never assigned to a class -- a
  zero move has no direction to predict).

FEED-INTEGRITY GATE — REVISED 2026-07-28, AFTER IT FIRED (recorded honestly)
----------------------------------------------------------------------------
The ORIGINAL gate was "STOP if the exact-zero next-bar return rate exceeds
0.5%". It fired on the real EURUSD H1 cache at 0.6461%. That threshold was a
round number chosen WITHOUT first measuring what rate H1 EURUSD naturally
produces -- it was set BELOW the normal rate of the very data it was meant to
protect.

The gate's stated rationale ("the feed carries stale/repeated closes") is a
testable claim, and the measured evidence REFUTES it:
    longest run of identical consecutive closes : 3 bars
    runs of >= 3 identical closes               : 5 (in 70,000 bars)
    concentration                               : thin-liquidity hours
                                                  (20-22h and 03-06h UTC)
    stability                                   : 0.40-0.90% EVERY year,
                                                  2015-2026, no bad patch
A genuinely stale feed produces MULTI-HOUR flat runs. There is not one. The
~0.64% rate is the discreteness of a 5-decimal quote at hourly resolution in
thin liquidity -- normal market microstructure, not a data fault.

The gate is therefore RE-SPECIFIED ON THE MECHANISM rather than on the rate.
STOP and report only if EITHER:
    (a) any run of identical consecutive closes is >= 6 bars, OR
    (b) the exact-zero next-bar return rate exceeds 3.0% (an order of magnitude
        above the 0.64% measured here).

WHAT THE REVISION DOES AND DOES NOT CHANGE. The 451 zero-return rows are dropped
IDENTICALLY under both the old and the new gate. The revision therefore changes
NO analysis input, NO model, NO split and NO decision rule -- it changes ONLY
whether the program is permitted to continue past STEP 1. It consumes NO alpha.
It was made AFTER the gate fired, and is justified by the refutation of the
gate's own stated mechanism, NOT by a wish to proceed.

KNOWN, DOCUMENTED SAMPLING SKEW. The dropped rows cluster in thin-liquidity
hours, so the hour-of-day feature is marginally UNDER-SAMPLED at 20-22h and
03-06h UTC. At 0.64% of rows this is negligible, but it is documented here
rather than left silent.

Every run records the observed zero-return rate, the dropped-row count and the
longest identical-close run in the hypothesis log, regardless of outcome.
* Features -- the IDENTICAL pre-registered set already built and unit-tested in
  src/pooled_h1_model.py, imported UNCHANGED (not one column added, removed or
  modified). This program changes the TARGET; changing the target and the
  feature set at once would confound the result. Feature changes are a separate
  future hypothesis.
* Split -- chronological: train [0:70%], validation [70:85%] (THE ARBITER),
  test [85:100%] (RESERVED -- never indexed by this program).
* Purge -- drop the final training row, whose label requires the first
  validation bar. Embargo -- drop the first 24 validation bars; with a 1-bar
  label that is more conservative than strictly necessary, and is set to 24 so
  the LSTM's 24-bar input sequences at the start of validation cannot straddle
  the train/validation boundary.
* Family -- NEW and independent: size 2, Bonferroni bar alpha = 0.05/2 = 0.025.
  It does not touch or tighten feature_hypothesis_log.csv, volatility_*,
  harmonic_pattern_*, pooled_h1_*, cot_*, fractal_* or ti_lstm_*.

CONDITIONAL REPLICATION ARM (registered up front, spent only if triggered)
--------------------------------------------------------------------------
H_dir.3 / H_dir.4 / H_dir.5 -- the WINNING model from H_dir.1/H_dir.2, refit and
scored independently on GBPUSD, AUDUSD and CHFUSD (inverted), each using that
instrument's OWN train/validation slices cut at the SAME global date boundaries.

* TRIGGER, fixed NOW: this arm runs ONLY if H_dir.1 or H_dir.2 clears its
  pre-registered bar (block-bootstrap CI entirely above zero AND McNemar
  p < 0.025). If BOTH DROP, the arm is never run and consumes NO alpha -- it is
  recorded in the log as REGISTERED-UNSPENT, exactly as the Fibonacci extension
  hypothesis was handled.
* If triggered, the family grows from 2 to 5 and the bar for the replication arm
  tightens to alpha = 0.05/5 = 0.01. This is stated AT REGISTRATION TIME, not
  after the fact.
* REPORTING RULE, fixed NOW: the question is NOT "does any pair clear the bar"
  -- that is cherry-picking across four correlated draws. The question is
  whether the effect's SIGN and rough MAGNITUDE persist across all three
  additional pairs. All three are reported regardless of outcome. ONE clearing
  pair out of three, with the other two at chance, is evidence AGAINST a real
  effect, not for it.
"""

import os
import random

import numpy as np
import pandas as pd

# Features are IMPORTED UNCHANGED from the pooled program (pre-registered and
# already unit-tested there). This module never redefines or edits them.
from src.pooled_h1_model import (
    FEATURE_COLUMNS, SEQ_LEN, compute_pooled_features, resolve_device,
)
from src.h1_horizon_feasibility import uniqueness_from_spans
from src.pooled_h1_data import POOLED_DIR, OHLC_COLUMNS

# ── Pre-registered constants (frozen; do not tune after seeing results) ──
INSTRUMENT = 'EURUSD'
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85                 # validation END (test = [0.85:1.0], RESERVED)
EMBARGO_BARS = 24               # >= SEQ_LEN so val sequences never straddle the split
BLOCK_LEN = 24                  # moving-block bootstrap block = one trading day
N_BOOT = 2000
FAMILY_SIZE = 2
FAMILY_ALPHA = 0.05 / FAMILY_SIZE               # 0.025
# ── Feed-integrity gate (REVISED after firing; see module docstring) ──
# Gate on the MECHANISM (a frozen feed = multi-hour flat runs), not on the rate.
MAX_IDENTICAL_CLOSE_RUN = 6     # STOP if any run of identical closes >= 6 bars
ZERO_RETURN_MAX_SHARE = 0.03    # STOP if zero-return rate > 3.0% (was 0.5%)
RANDOM_SEED = 42

# ── Conditional replication arm (registered NOW, spent only if triggered) ──
REPLICATION_PAIRS = ('GBPUSD', 'AUDUSD', 'CHFUSD')
REPLICATION_HYPOTHESES = {3: 'GBPUSD', 4: 'AUDUSD', 5: 'CHFUSD'}
FAMILY_SIZE_IF_TRIGGERED = 5
REPLICATION_ALPHA = 0.05 / FAMILY_SIZE_IF_TRIGGERED     # 0.01
UNSPENT_VERDICT = 'REGISTERED-UNSPENT'

# Recorded verbatim in EVERY log row: the honest account of the feed gate's
# post-hoc revision, plus the sampling skew it leaves behind.
FEED_GATE_NOTE = (
    "FEED GATE REVISED 2026-07-28, AFTER IT FIRED: the original gate (STOP if the "
    "exact-zero next-bar return rate exceeds 0.5%) tripped at 0.6461%. That 0.5% was "
    "a round number set WITHOUT first measuring the feed's natural rate -- i.e. below "
    "the normal rate of the data it was meant to protect. Its stated rationale "
    "('stale/repeated closes') was tested and REFUTED: longest identical-close run = 3 "
    "bars, only 5 runs of >=3 in 70,000 bars, rate stable at 0.40-0.90% every year "
    "2015-2026, concentrated in thin-liquidity hours -- a frozen feed instead produces "
    "multi-hour flat runs. Gate re-specified on the MECHANISM: STOP if any identical-"
    "close run >= 6 bars OR the zero-return rate exceeds 3.0%. The 451 rows are dropped "
    "IDENTICALLY under both the old and the new gate, so this revision changes NO "
    "analysis input, NO model, NO split and NO decision rule -- only whether the program "
    "may continue past STEP 1. It consumes NO alpha. Justified by the refutation of the "
    "gate's own mechanism, NOT by a wish to proceed. "
    "SAMPLING SKEW (documented, not silent): the dropped rows cluster in thin-liquidity "
    "hours, so the hour-of-day feature is marginally under-sampled at 20-22h and 03-06h "
    "UTC; at 0.64% of rows this is negligible."
)

HYPOTHESIS_LOG = 'results/h1_direction_hypothesis_log.csv'
ARBITER_LABEL = 'h1_direction_validation[70:85]_block_bootstrap'

LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter', 'instrument', 'n_rows_total',
    'n_zero_return_dropped', 'zero_return_rate_pct', 'longest_identical_close_run',
    'mean_label_uniqueness', 'n_purged', 'n_embargoed',
    'n_train', 'n_val', 'train_class_balance_pct1', 'val_class_balance_pct1',
    'acc_challenger', 'acc_reference', 'auc_challenger', 'delta_acc',
    'delta_acc_ci_low_iid', 'delta_acc_ci_high_iid', 'delta_acc_ci_low_block',
    'delta_acc_ci_high_block', 'mcnemar_b', 'mcnemar_c', 'mcnemar_p',
    'block_len', 'shuffled_label_control_acc', 'alpha', 'cleared_bar',
    'verdict', 'device_used', 'notes',
]


class MissingCachedDataError(FileNotFoundError):
    """Raised when results/pooled_h1/EURUSD_h1.csv is absent. This program is
    explicitly forbidden from refetching -- it STOPS and reports instead."""


class StaleFeedError(RuntimeError):
    """Raised when the feed shows the MECHANISM of staleness -- a multi-hour run
    of identical closes (>= MAX_IDENTICAL_CLOSE_RUN bars), or an exact-zero
    next-bar return rate an order of magnitude above normal
    (> ZERO_RETURN_MAX_SHARE). See the module docstring for why this gate was
    re-specified on the mechanism rather than on the rate."""


class LabelOverlapError(RuntimeError):
    """Raised when mean train label uniqueness is not exactly 1.0. That is the
    whole premise of this program, so a violation STOPS the run."""


# ───────────────────────── STEP 1: data + target ──────────────────────────────

def load_eurusd_h1(out_dir: str = POOLED_DIR) -> pd.DataFrame:
    """
    Read the ALREADY-CACHED EURUSD H1 OHLC frame. Deliberately does NOT refetch
    from MT5 -- if the cache is missing this raises MissingCachedDataError so the
    caller STOPS and reports, per the program's boundaries.
    """
    path = os.path.join(out_dir, f'{INSTRUMENT}_h1.csv')
    if not os.path.exists(path):
        raise MissingCachedDataError(
            f"Cached {INSTRUMENT} H1 data not found at {path!r}. This program must "
            "NOT refetch from MT5 -- STOP and report."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    return df[OHLC_COLUMNS].sort_index()


def build_direction_dataset(df: pd.DataFrame):
    """
    Build the next-bar-direction dataset from one OHLC frame.

    Target:  y[t] = 1 if log(close[t+1]/close[t]) > 0
             y[t] = 0 if log(close[t+1]/close[t]) < 0
             rows with EXACTLY 0 next-bar log return are DROPPED (not assigned).
    The FINAL bar has no next bar and is dropped by construction.

    Returns (context, target_index, counts) where:
      * `context`      -- every feature-valid bar, with a `label` column that is
                          NaN on zero-return bars. This is the SEQUENCE CONTEXT
                          frame: keeping the (rare) zero-return bars in it means
                          a 24-bar LSTM window is genuinely 24 CONTIGUOUS bars
                          rather than silently jumping a dropped row.
      * `target_index` -- the timestamps that carry a real 0/1 label (the only
                          rows ever trained on or scored).
      * `counts`       -- row accounting for the report and the log.

    The target reads close[t+1] and NOTHING else from the future: no future
    high/low/open ever reaches the feature matrix (features come from
    compute_pooled_features, which is trailing/causal by construction).
    """
    feats = compute_pooled_features(df)
    log_close = np.log(df['close'])
    fwd_logret = log_close.shift(-1) - log_close        # uses close[t+1] ONLY

    out = feats.copy()
    out['fwd_logret'] = fwd_logret
    # Drops the feature warm-up rows AND the final bar (no next close).
    out = out.dropna(subset=list(FEATURE_COLUMNS) + ['fwd_logret'])

    n_feature_valid = len(out)
    zero_mask = out['fwd_logret'].to_numpy() == 0.0
    n_zero = int(zero_mask.sum())
    zero_share = (n_zero / n_feature_valid) if n_feature_valid else 0.0

    out['label'] = np.where(zero_mask, np.nan,
                            (out['fwd_logret'].to_numpy() > 0.0).astype(float))

    counts = {
        'n_bars_raw': int(len(df)),
        'n_feature_valid': int(n_feature_valid),
        'n_zero_return_dropped': n_zero,
        'zero_return_share': float(zero_share),
        'n_labelled': int(n_feature_valid - n_zero),
        # Feed-integrity diagnostic, measured on the RAW frame and recorded on
        # every run regardless of outcome (see the revised gate).
        'longest_identical_close_run': longest_identical_close_run(df['close']),
    }
    target_index = out.index[~zero_mask]
    return out, target_index, counts


def longest_identical_close_run(close) -> int:
    """
    Length in bars of the LONGEST run of identical consecutive closes.

    This is the direct signature of a stale/frozen feed: a broker replaying the
    last quote produces a multi-hour flat run. Isolated single repeats are just
    5-decimal price discreteness in thin liquidity and are NOT staleness -- which
    is precisely why the feed gate keys on this statistic rather than on the raw
    zero-return rate.
    """
    c = np.asarray(close, dtype=float)
    if len(c) == 0:
        return 0
    changed = np.ones(len(c), dtype=bool)
    changed[1:] = c[1:] != c[:-1]
    return int(np.bincount(np.cumsum(changed)).max())


def assert_feed_not_stale(counts):
    """
    Hard gate on the MECHANISM of a stale feed, not on the zero-return rate.
    STOP if EITHER:
      (a) any run of identical consecutive closes is >= MAX_IDENTICAL_CLOSE_RUN
          (6) bars -- the actual signature of a frozen/replayed feed, OR
      (b) the exact-zero next-bar return rate exceeds ZERO_RETURN_MAX_SHARE
          (3.0%) -- an order of magnitude above the ~0.64% this feed naturally
          produces.

    REVISED 2026-07-28 after the ORIGINAL 0.5%-rate gate fired at 0.6461%; see
    the module docstring for the full, honest record of that revision.
    """
    run = counts['longest_identical_close_run']
    if run >= MAX_IDENTICAL_CLOSE_RUN:
        raise StaleFeedError(
            f"longest run of identical consecutive closes = {run} bars, at/above the "
            f"{MAX_IDENTICAL_CLOSE_RUN}-bar bar. That is the signature of a frozen/"
            "replayed feed -- STOP and report."
        )
    if counts['zero_return_share'] > ZERO_RETURN_MAX_SHARE:
        raise StaleFeedError(
            f"{counts['n_zero_return_dropped']} of {counts['n_feature_valid']} rows "
            f"({counts['zero_return_share']:.4%}) have EXACTLY zero next-bar return, "
            f"above the {ZERO_RETURN_MAX_SHARE:.2%} bar -- an order of magnitude "
            "above normal H1 discreteness. STOP and report."
        )


# ───────────────────────── split + purge + embargo ────────────────────────────

def split_purge_embargo(target_index):
    """
    Chronological split of the LABELLED rows into train [0:70%],
    validation [70:85%] and test [85:100%], then:

      * PURGE  -- drop the FINAL training row, whose label is the return into the
                  first validation bar.
      * EMBARGO -- drop the first EMBARGO_BARS (24) validation rows. With a
                  1-bar label this is more conservative than strictly necessary;
                  24 is chosen so the LSTM's 24-bar input sequences at the start
                  of validation cannot straddle the train/validation boundary.

    Returns (splits, counts). `splits['test']` is returned ONLY so its size can
    be reported -- no downstream step ever indexes it.
    """
    n = len(target_index)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * VAL_FRAC)

    train = target_index[:train_end]
    val = target_index[train_end:val_end]
    test = target_index[val_end:]

    n_purged = 1 if len(train) else 0
    train_purged = train[:-1] if n_purged else train

    n_embargoed = min(EMBARGO_BARS, len(val))
    val_embargoed = val[n_embargoed:]

    splits = {'train': train_purged, 'val': val_embargoed, 'test': test}
    counts = {
        'n_purged': int(n_purged), 'n_embargoed': int(n_embargoed),
        'n_train': int(len(train_purged)), 'n_val': int(len(val_embargoed)),
        'n_test_reserved': int(len(test)),
        'val_end_position': int(val_end),
        # GLOBAL date boundaries -- reused unchanged by the replication arm so
        # every instrument is cut at the SAME dates, never re-optimised per pair.
        'train_end_ts': train[-1] if len(train) else None,
        'val_end_ts': val[-1] if len(val) else None,
    }
    return splits, counts


def split_purge_embargo_by_dates(target_index, train_end_ts, val_end_ts):
    """
    Replication-arm split: cut ONE instrument's labelled index at the GLOBAL
    date boundaries taken from the EURUSD run (never re-derived per pair, which
    would silently re-optimise the split), then apply the identical purge (drop
    the final training row) and embargo (drop the first 24 validation rows).
    """
    train = target_index[target_index <= train_end_ts]
    val = target_index[(target_index > train_end_ts) & (target_index <= val_end_ts)]
    test = target_index[target_index > val_end_ts]

    n_purged = 1 if len(train) else 0
    train_purged = train[:-1] if n_purged else train
    n_embargoed = min(EMBARGO_BARS, len(val))
    val_embargoed = val[n_embargoed:]

    return ({'train': train_purged, 'val': val_embargoed, 'test': test},
            {'n_purged': int(n_purged), 'n_embargoed': int(n_embargoed),
             'n_train': int(len(train_purged)), 'n_val': int(len(val_embargoed)),
             'n_test_reserved': int(len(test)), 'val_end_position': int(len(train) + len(val))})


def mean_label_uniqueness(context: pd.DataFrame, target_index) -> float:
    """
    Lopez de Prado mean label uniqueness for this program's 1-bar-ahead labels,
    computed with the SAME estimator the feasibility scan used
    (h1_horizon_feasibility.uniqueness_from_spans, imported unchanged).

    A 1-bar-ahead label is determined by the return over the single bar interval
    (t, t+1], so its information span is exactly ONE bar of the grid: span
    [i, i]. Consecutive labels therefore never share a bar, concurrency is 1
    everywhere, and the mean uniqueness is exactly 1.0. That is the premise of
    this whole program, and run() gates on it.
    """
    pos = {ts: i for i, ts in enumerate(context.index)}
    starts = np.array([pos[ts] for ts in target_index], dtype=np.int64)
    return uniqueness_from_spans(starts, starts, grid_len=len(context))


def assert_labels_independent(uniqueness: float):
    """Hard gate: mean train label uniqueness MUST be exactly 1.0. Anything else
    means the label construction is wrong -- STOP."""
    if not (uniqueness == 1.0):
        raise LabelOverlapError(
            f"mean label uniqueness = {uniqueness!r}, expected exactly 1.0. The "
            "1-bar-ahead label construction is wrong -- STOP and report."
        )


# ───────────────────────── matrices + sequences ───────────────────────────────

def fit_standardizer(context: pd.DataFrame, train_index):
    """Per-feature mean/std fit on the TRAIN target rows ONLY (never validation,
    never test)."""
    X = context.loc[train_index, FEATURE_COLUMNS].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def flat_matrix(context: pd.DataFrame, index, mean, std):
    """Flat (row-per-sample) standardized feature matrix + labels for the GBM."""
    d = context.loc[index]
    X = (d[FEATURE_COLUMNS].to_numpy(dtype=float) - mean) / std
    y = d['label'].to_numpy(dtype=float).astype(int)
    return X, y


def build_sequences(context: pd.DataFrame, target_index, mean, std):
    """
    24-bar sequences ending at each target row, taken as CONTIGUOUS positional
    windows of the single-instrument context frame -- so every sequence is 24
    consecutive EURUSD bars with no dropped row inside it. Rows too early to
    carry a full window are skipped (never zero-padded).

    A validation sequence's preceding context may reach back before the scored
    row, which is exactly why the 24-bar embargo exists: the first scored
    validation row sits 24 bars into the validation slice, so its window lies
    entirely inside validation and never straddles the purge boundary.

    Returns (seqs, ys, kept_timestamps).
    """
    X = (context[FEATURE_COLUMNS].to_numpy(dtype=float) - mean) / std
    labels = context['label'].to_numpy(dtype=float)
    pos = {ts: i for i, ts in enumerate(context.index)}

    seqs, ys, kept = [], [], []
    for ts in target_index:
        j = pos[ts]
        if j - SEQ_LEN + 1 < 0:
            continue
        seqs.append(X[j - SEQ_LEN + 1: j + 1])
        ys.append(labels[j])
        kept.append(ts)

    if not seqs:
        return (np.empty((0, SEQ_LEN, len(FEATURE_COLUMNS)), dtype=np.float32),
                np.empty((0,), dtype=np.float32), [])
    return (np.asarray(seqs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32), kept)


# ───────────────────────── seeding (CUDA is a hard requirement) ───────────────

def seed_everything(seed: int = RANDOM_SEED):
    """Seed python/numpy/torch. Bitwise determinism is NOT guaranteed on GPU
    (accepted here); the seed is reported so a run is re-runnable in spirit."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ───────────────────────── baseline + models ──────────────────────────────────

def train_majority_class(y_train) -> int:
    """The train-majority baseline: whichever class was more frequent in the
    TRAIN slice, predicted for EVERY validation row. Ties -> class 1."""
    y = np.asarray(y_train).astype(int)
    return int((y == 1).sum() >= (y == 0).sum())


def _balanced_scale_pos_weight(y):
    pos = float((np.asarray(y) == 1).sum())
    neg = float((np.asarray(y) == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def train_gbm(Xtr, ytr, seed=RANDOM_SEED):
    """XGBoost on GPU (device='cuda'), balanced class weights."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', device='cuda',
        scale_pos_weight=_balanced_scale_pos_weight(ytr),
        random_state=seed, n_jobs=0,
    )
    clf.fit(Xtr, ytr)
    return clf


def predict_gbm_proba(clf, X):
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)


def _make_lstm(n_features):
    """Pre-registered, fixed architecture:
    LSTM(hidden=32, layers=1) -> Dropout(.3) -> Linear(16) -> ReLU
    -> Dropout(.3) -> Linear(1) -> Sigmoid."""
    import torch.nn as nn

    class DirectionLSTM(nn.Module):
        def __init__(self, n_feat):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_feat, hidden_size=32,
                                num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(32, 16), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(16, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return DirectionLSTM(n_features)


def train_lstm(Xtr, ytr, Xval, yval, device, seed=RANDOM_SEED,
               epochs=100, patience=10, batch_size=256):
    """
    PyTorch LSTM over 24-bar sequences. Adam lr=1e-3, weight_decay=1e-3,
    batch=256, epochs<=100, early stopping patience=10 on validation loss.

    EXPLICIT model.train() / model.eval()+torch.no_grad() toggling -- PyTorch
    does NOT auto-toggle Dropout. Balanced per-sample weights are applied via
    BCELoss(reduction='none') multiplied per batch: BCELoss has NO pos_weight
    argument (that is BCEWithLogitsLoss only).

    Returns (val_probabilities, best_val_loss).
    """
    import torch
    import torch.nn as nn

    seed_everything(seed)
    model = _make_lstm(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCELoss(reduction='none')

    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    w_pos = (len(ytr) / (2.0 * pos)) if pos > 0 else 1.0
    w_neg = (len(ytr) / (2.0 * neg)) if neg > 0 else 1.0
    sample_w = np.where(ytr == 1, w_pos, w_neg).astype(np.float32)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    w_t = torch.tensor(sample_w, dtype=torch.float32)
    Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device)
    yval_t = torch.tensor(yval, dtype=torch.float32).to(device)

    n = len(Xtr_t)
    best_val, best_state, since_improve = float('inf'), None, 0
    g = torch.Generator().manual_seed(seed)

    for _epoch in range(epochs):
        model.train()                              # Dropout ON
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb, wb = (Xtr_t[idx].to(device), ytr_t[idx].to(device),
                          w_t[idx].to(device))
            opt.zero_grad()
            loss = (loss_fn(model(xb), yb) * wb).mean()
            loss.backward()
            opt.step()

        model.eval()                               # Dropout OFF
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy(model(Xval_t), yval_t).item()
        if vloss < best_val - 1e-5:
            best_val, since_improve = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = model(Xval_t).detach().cpu().numpy()
    return np.asarray(val_prob, dtype=float), best_val


# ───────────────────────── STEP 3: leakage control ────────────────────────────

def shuffled_label_control(Xtr, ytr, Xval, yval, seed=RANDOM_SEED):
    """
    MANDATORY leakage control (NOT a hypothesis -- consumes no alpha).

    Refit the GBM with the TRAINING LABELS RANDOMLY PERMUTED (features
    untouched, same hyperparameters, same seed discipline) and score it on the
    REAL, unshuffled validation rows. Its accuracy MUST land near the majority
    baseline rate. If a model trained on destroyed labels still scores
    materially above that, the pipeline is leaking and EVERY other number in the
    program is void.

    Returns the shuffled-label model's validation accuracy.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = np.asarray(ytr).astype(int).copy()
    rng.shuffle(y_shuffled)
    clf = train_gbm(Xtr, y_shuffled, seed=seed)
    pred = (predict_gbm_proba(clf, Xval) >= 0.5).astype(int)
    return float((pred == np.asarray(yval).astype(int)).mean())


def leak_control_is_sane(acc, lo=0.48, hi=0.52):
    """The control must land near chance/majority. Returns True if it does."""
    return bool(lo <= acc <= hi)


# ───────────────────────── STEP 4: statistical arbiter ────────────────────────

def _iid_bootstrap_delta(cc, cr, n_boot, alpha, seed=RANDOM_SEED):
    """PAIRED i.i.d. bootstrap CI for delta accuracy = mean(cc) - mean(cr).
    Resamples rows independently -- reported for context, NOT decision-bearing."""
    rng = np.random.default_rng(seed)
    n = len(cc)
    point = float(cc.mean() - cr.mean()) if n else float('nan')
    if n == 0:
        return float('nan'), float('nan'), point
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = cc[idx].mean() - cr[idx].mean()
    return (float(np.percentile(deltas, 100 * (alpha / 2))),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))), point)


def _block_bootstrap_delta(cc, cr, block_len, n_boot, alpha, seed=RANDOM_SEED):
    """
    PAIRED MOVING-BLOCK (circular) bootstrap CI for delta accuracy. Resamples
    CONTIGUOUS blocks of `block_len` bars (one trading day) rather than
    independent rows.

    Rationale: although 1-bar labels do not overlap, hourly FX returns exhibit
    volatility clustering and intraday session structure, so consecutive
    predictions are not independent even when the LABELS are. This is the
    conservative interval and it GOVERNS the decision.
    """
    rng = np.random.default_rng(seed)
    n = len(cc)
    point = float(cc.mean() - cr.mean()) if n else float('nan')
    if n == 0:
        return float('nan'), float('nan'), point
    n_blocks = int(np.ceil(n / block_len))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        deltas[b] = cc[idx].mean() - cr[idx].mean()
    return (float(np.percentile(deltas, 100 * (alpha / 2))),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))), point)


def _mcnemar_exact(correct_c, correct_r):
    """Exact (binomial) two-sided McNemar on paired correctness. b = challenger
    right & reference wrong; c = challenger wrong & reference right."""
    from scipy.stats import binomtest
    a = np.asarray(correct_c).astype(bool)
    r = np.asarray(correct_r).astype(bool)
    b = int(np.sum(a & ~r))
    c = int(np.sum(~a & r))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue)


def _roc_auc(y_true, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def arbiter(pred_challenger, pred_reference, y_true, score_challenger=None,
            block_len=BLOCK_LEN, n_boot=N_BOOT, alpha=FAMILY_ALPHA,
            seed=RANDOM_SEED):
    """
    PRIMARY metric: accuracy on the validation slice. ROC-AUC is reported
    alongside as DESCRIPTIVE CONTEXT ONLY -- it is not decision-bearing.

    Decision rule: KEEP only if the delta-accuracy BLOCK-bootstrap CI lies
    ENTIRELY above zero AND McNemar p < alpha. Otherwise DROP. The i.i.d.
    bootstrap is reported for contrast but never governs.
    """
    y = np.asarray(y_true).astype(int)
    cc = (np.asarray(pred_challenger).astype(int) == y).astype(float)
    cr = (np.asarray(pred_reference).astype(int) == y).astype(float)

    lo_i, hi_i, _ = _iid_bootstrap_delta(cc, cr, n_boot, alpha, seed)
    lo_b, hi_b, delta = _block_bootstrap_delta(cc, cr, block_len, n_boot, alpha, seed)
    b, c, p = _mcnemar_exact(cc, cr)

    cleared = bool(lo_b > 0.0 and p < alpha)
    return {
        'acc_challenger': float(cc.mean()), 'acc_reference': float(cr.mean()),
        'auc_challenger': (_roc_auc(y, score_challenger)
                           if score_challenger is not None else float('nan')),
        'delta_acc': delta,
        'delta_acc_ci_low_iid': lo_i, 'delta_acc_ci_high_iid': hi_i,
        'delta_acc_ci_low_block': lo_b, 'delta_acc_ci_high_block': hi_b,
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p,
        'block_len': block_len, 'alpha': alpha,
        'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP',
    }


# ───────────────────────── STEP 5: hypothesis log ─────────────────────────────

def _upsert_log(row: dict, log_path: str = HYPOTHESIS_LOG):
    """Append/replace one hypothesis row (idempotent on the `hypothesis` name),
    keeping this family's log sorted by `n`. Touches NO other family's log."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    new = pd.DataFrame([row], columns=LOG_COLUMNS)
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        existing = existing[existing['hypothesis'] != row['hypothesis']]
        log = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    else:
        log = new
    log = log.sort_values('n').reset_index(drop=True)
    log.to_csv(log_path, index=False)
    return log


def _log_row(n, hypothesis, acc, counts, split_counts, uniqueness, balances,
             leak_acc, device_used, notes, instrument=INSTRUMENT):
    def _r(x, nd=6):
        return round(float(x), nd) if np.isfinite(float(x)) else float(x)

    return {
        'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
        'hypothesis': hypothesis, 'arbiter': ARBITER_LABEL,
        'instrument': instrument,
        'n_rows_total': counts['n_feature_valid'],
        'n_zero_return_dropped': counts['n_zero_return_dropped'],
        'zero_return_rate_pct': _r(100.0 * counts['zero_return_share'], 4),
        'longest_identical_close_run': counts['longest_identical_close_run'],
        'mean_label_uniqueness': _r(uniqueness),
        'n_purged': split_counts['n_purged'],
        'n_embargoed': split_counts['n_embargoed'],
        'n_train': split_counts['n_train'], 'n_val': split_counts['n_val'],
        'train_class_balance_pct1': _r(balances['train_pct1'], 4),
        'val_class_balance_pct1': _r(balances['val_pct1'], 4),
        'acc_challenger': _r(acc['acc_challenger']),
        'acc_reference': _r(acc['acc_reference']),
        'auc_challenger': _r(acc['auc_challenger']),
        'delta_acc': _r(acc['delta_acc']),
        'delta_acc_ci_low_iid': _r(acc['delta_acc_ci_low_iid']),
        'delta_acc_ci_high_iid': _r(acc['delta_acc_ci_high_iid']),
        'delta_acc_ci_low_block': _r(acc['delta_acc_ci_low_block']),
        'delta_acc_ci_high_block': _r(acc['delta_acc_ci_high_block']),
        'mcnemar_b': acc['mcnemar_b'], 'mcnemar_c': acc['mcnemar_c'],
        'mcnemar_p': _r(acc['mcnemar_p']),
        'block_len': acc['block_len'],
        'shuffled_label_control_acc': _r(leak_acc),
        'alpha': acc['alpha'], 'cleared_bar': acc['cleared_bar'],
        'verdict': acc['verdict'], 'device_used': device_used, 'notes': notes,
    }


# ────────────── conditional replication arm (H_dir.3 / .4 / .5) ───────────────

REPLICATION_RULE = (
    "CONDITIONAL replication of the WINNING model from H_dir.1/H_dir.2, refit and "
    "scored independently on this instrument's own train/val slices cut at the SAME "
    "global date boundaries. TRIGGER (fixed at registration): runs ONLY if H_dir.1 or "
    "H_dir.2 clears its bar (block CI entirely above 0 AND McNemar p<0.025). If both "
    "DROP the arm is never run and consumes NO alpha. IF TRIGGERED the family grows "
    "2 -> 5 and this arm's bar tightens to alpha=0.05/5=0.01. REPORTING RULE (fixed at "
    "registration): the question is NOT 'does any pair clear' -- that is cherry-picking "
    "across 4 correlated draws -- but whether the effect's SIGN and rough MAGNITUDE "
    "persist across ALL THREE pairs. All three are reported regardless. One clearing "
    "pair out of three with the other two at chance is evidence AGAINST a real effect."
)


def replication_triggered(h1, h2) -> bool:
    """The trigger, evaluated exactly as pre-registered: either primary
    hypothesis clearing its bar arms the replication arm."""
    return bool(h1['cleared_bar'] or h2['cleared_bar'])


def register_unspent_replication(counts, split_counts, uniqueness, balances,
                                 leak_acc, device_used, log_path=HYPOTHESIS_LOG):
    """
    Write H_dir.3/.4/.5 to the family log as REGISTERED-UNSPENT (the Fibonacci
    extension precedent): the arm exists on the record with its trigger and its
    alpha stated up front, but it was never run, so it consumed no alpha and the
    family stays at size 2 (bar 0.025).
    """
    rows = []
    for n, pair in sorted(REPLICATION_HYPOTHESES.items()):
        row = {c: '' for c in LOG_COLUMNS}
        row.update({
            'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
            'hypothesis': f'H_dir.{n}_replication_{pair}',
            'arbiter': ARBITER_LABEL, 'instrument': pair,
            'n_rows_total': counts['n_feature_valid'],
            'n_zero_return_dropped': counts['n_zero_return_dropped'],
            'zero_return_rate_pct': round(100.0 * counts['zero_return_share'], 4),
            'longest_identical_close_run': counts['longest_identical_close_run'],
            'mean_label_uniqueness': round(float(uniqueness), 6),
            'n_purged': split_counts['n_purged'],
            'n_embargoed': split_counts['n_embargoed'],
            'train_class_balance_pct1': round(balances['train_pct1'], 4),
            'val_class_balance_pct1': round(balances['val_pct1'], 4),
            'block_len': BLOCK_LEN,
            'shuffled_label_control_acc': round(float(leak_acc), 6),
            'alpha': REPLICATION_ALPHA, 'cleared_bar': '',
            'verdict': UNSPENT_VERDICT, 'device_used': device_used,
            'notes': (f"NOT RUN — trigger did not fire (H_dir.1 and H_dir.2 both "
                      f"DROPped), so this arm consumed NO alpha and the family "
                      f"stays at size 2 (bar 0.025). {REPLICATION_RULE} "
                      f"{FEED_GATE_NOTE}"),
        })
        rows.append(row)
        _upsert_log(row, log_path)
    return rows


def run_replication_arm(winner, context_eur, mean, std, gbm_model, device,
                        train_end_ts, val_end_ts, leak_acc, seed=RANDOM_SEED,
                        out_dir=POOLED_DIR, log_path=HYPOTHESIS_LOG,
                        register=True):
    """
    Run the replication arm -- ONLY reachable when the trigger fired. Refits the
    WINNING model on each of GBPUSD / AUDUSD / CHFUSD (inverted) using that
    instrument's own slices at the SAME global date boundaries, scores it against
    that instrument's own train-majority baseline, and judges each at the
    tightened alpha = 0.05/5 = 0.01.

    Returns a list of per-pair result dicts. All three are always returned, so
    the caller reports sign/magnitude persistence across the whole set rather
    than cherry-picking whichever pair happens to clear.
    """
    from src.pooled_h1_data import build_pooled_pairs
    # write_chfusd=False: never re-emit a cached CSV as a side effect of scoring.
    pairs = build_pooled_pairs(out_dir=out_dir, write_chfusd=False)

    results = []
    for n, pair in sorted(REPLICATION_HYPOTHESES.items()):
        ctx, tgt, counts = build_direction_dataset(pairs[pair])
        assert_feed_not_stale(counts)
        splits, sc = split_purge_embargo_by_dates(tgt, train_end_ts, val_end_ts)
        uniq = mean_label_uniqueness(ctx, splits['train'])
        assert_labels_independent(uniq)

        p_mean, p_std = fit_standardizer(ctx, splits['train'])
        Xtr, ytr = flat_matrix(ctx, splits['train'], p_mean, p_std)
        Xval, yval = flat_matrix(ctx, splits['val'], p_mean, p_std)
        majority = train_majority_class(ytr)
        pred_major = np.full(len(yval), majority, dtype=int)

        if winner == 'LSTM':
            Xs_tr, ys_tr, _ = build_sequences(ctx, splits['train'], p_mean, p_std)
            Xs_v, ys_v, kept = build_sequences(ctx, splits['val'], p_mean, p_std)
            score, _ = train_lstm(Xs_tr, ys_tr, Xs_v, ys_v, device, seed=seed)
            y_scored = ys_v.astype(int)
            ref = pred_major[[splits['val'].get_loc(ts) for ts in kept]]
        else:
            clf = train_gbm(Xtr, ytr, seed=seed)
            score = predict_gbm_proba(clf, Xval)
            y_scored, ref = yval, pred_major

        pred = (score >= 0.5).astype(int)
        acc = arbiter(pred, ref, y_scored, score_challenger=score,
                      alpha=REPLICATION_ALPHA, seed=seed)

        balances = {'train_pct1': 100.0 * float((ytr == 1).mean()),
                    'val_pct1': 100.0 * float((y_scored == 1).mean())}
        notes = (f"TRIGGERED replication of the winning model ({winner}) on {pair}; "
                 f"family grew 2 -> 5, alpha tightened to 0.05/5=0.01; own slices at "
                 f"the SAME global date boundaries; reference = {pair} train-majority "
                 f"class {majority}. {REPLICATION_RULE} {FEED_GATE_NOTE}")
        row = _log_row(n, f'H_dir.{n}_replication_{pair}', acc, counts, sc,
                       uniq, balances, leak_acc, str(device), notes,
                       instrument=pair)
        if register:
            _upsert_log(row, log_path)
        results.append({'pair': pair, 'n': n, 'arbiter': acc, 'row': row,
                        'majority_class': majority, 'balances': balances})
    return results


# ───────────────────────── orchestration ──────────────────────────────────────

def run(out_dir: str = POOLED_DIR, log_path: str = HYPOTHESIS_LOG,
        seed: int = RANDOM_SEED, register: bool = True, verbose: bool = True):
    """
    Full next-bar-direction program end to end. Reads the cached EURUSD H1 CSV,
    builds the 1-bar-ahead target, gates on feed sanity and label independence,
    runs the shuffled-label leakage control, then evaluates H_dir.1 and H_dir.2.
    Writes ONLY results/h1_direction_hypothesis_log.csv (when register=True).
    The test block [85:100%] is never indexed.
    """
    device, dev_info = resolve_device()
    used_seed = seed_everything(seed)
    if verbose:
        print('=' * 74)
        print('H1 NEXT-BAR DIRECTION — device & determinism')
        print(f"  torch.cuda.is_available() = {dev_info['cuda_available']}")
        if dev_info['cuda_available']:
            print(f"  torch.cuda.get_device_name(0) = {dev_info['cuda_device_name']}")
        else:
            print(f"  *** CUDA NOT AVAILABLE — running on fallback: {dev_info['device']} ***")
        print(f"  resolved device = {device}")
        print(f"  seed = {used_seed} (bitwise GPU determinism NOT guaranteed; accepted)")
        print('=' * 74)

    # ── STEP 1 ──
    raw = load_eurusd_h1(out_dir=out_dir)
    context, target_index, counts = build_direction_dataset(raw)
    assert_feed_not_stale(counts)

    splits, split_counts = split_purge_embargo(target_index)
    train_idx, val_idx = splits['train'], splits['val']

    uniqueness = mean_label_uniqueness(context, train_idx)
    assert_labels_independent(uniqueness)                      # the premise

    ytr_all = context.loc[train_idx, 'label'].to_numpy().astype(int)
    yval_all = context.loc[val_idx, 'label'].to_numpy().astype(int)
    balances = {
        'train_pct1': 100.0 * float((ytr_all == 1).mean()),
        'val_pct1': 100.0 * float((yval_all == 1).mean()),
    }

    # ── matrices ──
    mean, std = fit_standardizer(context, train_idx)
    Xtr, ytr = flat_matrix(context, train_idx, mean, std)
    Xval, yval = flat_matrix(context, val_idx, mean, std)

    # ── STEP 3: leakage control (BEFORE any hypothesis is read) ──
    leak_acc = shuffled_label_control(Xtr, ytr, Xval, yval, seed=seed)
    leak_ok = leak_control_is_sane(leak_acc)
    if verbose:
        print(f"\nSHUFFLED-LABEL CONTROL: acc = {leak_acc:.4f}  "
              f"({'SANE — near chance' if leak_ok else 'ANOMALOUS — PIPELINE LEAKING'})")

    majority = train_majority_class(ytr)
    pred_major = np.full(len(yval), majority, dtype=int)
    acc_major = float((pred_major == yval).mean())

    result = {
        'device_info': dev_info, 'seed': used_seed, 'device': str(device),
        'counts': counts, 'split_counts': split_counts,
        'uniqueness': uniqueness, 'balances': balances,
        'majority_class': majority, 'acc_majority': acc_major,
        'shuffled_label_control_acc': leak_acc, 'leak_control_sane': leak_ok,
        'span': (context.index.min().isoformat(), context.index.max().isoformat()),
    }
    if not leak_ok:
        result['stopped'] = 'LEAK: shuffled-label control outside [0.48, 0.52]'
        return result

    # ── H_dir.1 : GBM vs train-majority baseline ──
    gbm = train_gbm(Xtr, ytr, seed=seed)
    prob_gbm = predict_gbm_proba(gbm, Xval)
    pred_gbm = (prob_gbm >= 0.5).astype(int)
    h1 = arbiter(pred_gbm, pred_major, yval, score_challenger=prob_gbm, seed=seed)

    h1_notes = (
        f"XGBoost hist/device=cuda n_est300 depth4 lr.05 balanced scale_pos_weight; "
        f"reference=train-majority class {majority} (acc {acc_major:.4f}); "
        f"1-bar label => uniqueness {uniqueness:.4f}; block bootstrap governs "
        f"(block_len={BLOCK_LEN} bars = 1 trading day); AUC descriptive only; "
        f"shuffled-label control {leak_acc:.4f}. {FEED_GATE_NOTE}"
    )
    row1 = _log_row(1, 'H_dir.1_GBM_vs_train_majority', h1, counts, split_counts,
                    uniqueness, balances, leak_acc, dev_info['device'], h1_notes)

    # ── H_dir.2 : LSTM vs GBM (primary) ──
    Xs_tr, ys_tr, _ = build_sequences(context, train_idx, mean, std)
    Xs_val, ys_val, kept_val = build_sequences(context, val_idx, mean, std)

    prob_lstm, best_vloss = train_lstm(Xs_tr, ys_tr, Xs_val, ys_val, device, seed=seed)
    pred_lstm = (prob_lstm >= 0.5).astype(int)

    # Pair against the GBM on the IDENTICAL validation rows the LSTM scored.
    kept_pos = [val_idx.get_loc(ts) for ts in kept_val]
    pred_gbm_paired = pred_gbm[kept_pos]
    pred_major_paired = pred_major[kept_pos]
    y_paired = ys_val.astype(int)

    h2 = arbiter(pred_lstm, pred_gbm_paired, y_paired,
                 score_challenger=prob_lstm, seed=seed)
    # CORROBORATING ONLY — context, never a second path to KEEP.
    h2_corr = arbiter(pred_lstm, pred_major_paired, y_paired,
                      score_challenger=prob_lstm, seed=seed)

    h2_notes = (
        f"PyTorch LSTM(32,1)->Drop.3->Lin16->ReLU->Drop.3->Lin1->Sigmoid; Adam "
        f"lr1e-3 wd1e-3 batch256 <=100ep patience10 on val loss; per-sample "
        f"balanced BCELoss(weight=); explicit train()/eval() toggling; "
        f"device={dev_info['device']}; best_val_loss={best_vloss:.4f}; "
        f"PRIMARY reference=GBM on identical rows; CORROBORATING-ONLY vs "
        f"majority: lstm {h2_corr['acc_challenger']:.4f} vs {h2_corr['acc_reference']:.4f} "
        f"(delta {h2_corr['delta_acc']:+.4f}, block CI "
        f"[{h2_corr['delta_acc_ci_low_block']:+.4f},{h2_corr['delta_acc_ci_high_block']:+.4f}], "
        f"McNemar p={h2_corr['mcnemar_p']:.4g}) — NOT a path to KEEP. {FEED_GATE_NOTE}"
    )
    split_counts_lstm = dict(split_counts, n_train=int(len(ys_tr)), n_val=int(len(ys_val)))
    row2 = _log_row(2, 'H_dir.2_LSTM_vs_GBM', h2, counts, split_counts_lstm,
                    uniqueness, balances, leak_acc, dev_info['device'], h2_notes)

    if register:
        _upsert_log(row1, log_path)
        _upsert_log(row2, log_path)

    result.update({
        'h_dir_1': h1, 'h_dir_2': h2, 'h_dir_2_corroborating': h2_corr,
        'row1': row1, 'row2': row2, 'best_val_loss': best_vloss,
        'n_lstm_train': int(len(ys_tr)), 'n_lstm_val': int(len(ys_val)),
    })

    # ── Conditional replication arm (registered up front; spent only if fired) ──
    triggered = replication_triggered(h1, h2)
    result['replication_triggered'] = triggered
    if not triggered:
        result['replication'] = None
        result['replication_status'] = UNSPENT_VERDICT
        if register:
            register_unspent_replication(counts, split_counts, uniqueness,
                                         balances, leak_acc, dev_info['device'],
                                         log_path)
    else:
        winner = 'LSTM' if h2['cleared_bar'] else 'GBM'
        result['replication_winner'] = winner
        result['replication_status'] = f'TRIGGERED (alpha tightened to {REPLICATION_ALPHA})'
        result['replication'] = run_replication_arm(
            winner, context, mean, std, gbm, device,
            split_counts['train_end_ts'], split_counts['val_end_ts'],
            leak_acc, seed=seed, out_dir=out_dir, log_path=log_path,
            register=register)
    return result


def _print_report(r):
    """STEP 7 report, in the pre-registered order. RAW numbers only."""
    d, c, s, b = r['device_info'], r['counts'], r['split_counts'], r['balances']
    print('\n' + '=' * 74)
    print('H1 NEXT-BAR DIRECTION — RESULTS (raw)')
    print('=' * 74)

    print('\n1. DEVICE')
    print(f"   CUDA available      : {d['cuda_available']}")
    if d['cuda_available']:
        print(f"   CUDA device         : {d['cuda_device_name']}")
    print(f"   device used         : {r['device']}     seed = {r['seed']}")

    print('\n2. ROW COUNTS')
    print(f"   raw H1 bars         : {c['n_bars_raw']}")
    print(f"   feature-valid rows  : {c['n_feature_valid']}")
    print(f"   zero-return dropped : {c['n_zero_return_dropped']} "
          f"({c['zero_return_share']:.4%} of feature-valid; bar = "
          f"{ZERO_RETURN_MAX_SHARE:.1%})")
    print(f"   longest flat run    : {c['longest_identical_close_run']} bars "
          f"(bar = {MAX_IDENTICAL_CLOSE_RUN}; feed gate keys on THIS, not the rate)")
    print(f"   labelled rows       : {c['n_labelled']}")
    print(f"   purged              : {s['n_purged']}")
    print(f"   embargoed           : {s['n_embargoed']}")
    print(f"   train / val         : {s['n_train']} / {s['n_val']}")
    print(f"   test RESERVED       : {s['n_test_reserved']} (never indexed)")

    print('\n3. LABEL UNIQUENESS (the premise)')
    print(f"   mean label uniqueness = {r['uniqueness']:.6f}  "
          f"({'EXACTLY 1.0 — labels independent' if r['uniqueness'] == 1.0 else 'NOT 1.0'})")

    print('\n4. CLASS BALANCE')
    print(f"   train % class 1     : {b['train_pct1']:.4f}%")
    print(f"   val   % class 1     : {b['val_pct1']:.4f}%")
    print(f"   train-majority class: {r['majority_class']}   "
          f"its accuracy on val = {r['acc_majority']:.4f}")

    print('\n5. SHUFFLED-LABEL CONTROL (leakage gate, no alpha consumed)')
    print(f"   accuracy = {r['shuffled_label_control_acc']:.4f}  -> "
          f"{'SANE (near chance/majority)' if r['leak_control_sane'] else 'ANOMALOUS — RESULTS VOID'}")
    if not r['leak_control_sane']:
        print(f"   {r.get('stopped')}")
        return

    print('\n6. HYPOTHESES  (alpha = 0.05/2 = 0.025; BLOCK bootstrap governs)')
    for tag, res in (('H_dir.1  GBM vs train-majority', r['h_dir_1']),
                     ('H_dir.2  LSTM vs GBM (PRIMARY)', r['h_dir_2']),
                     ('         LSTM vs majority (CORROBORATING ONLY)',
                      r['h_dir_2_corroborating'])):
        print(f"\n   {tag}")
        print(f"     acc challenger = {res['acc_challenger']:.4f}   "
              f"acc reference = {res['acc_reference']:.4f}   "
              f"delta = {res['delta_acc']:+.4f}")
        print(f"     ROC-AUC (descriptive only) = {res['auc_challenger']:.4f}")
        print(f"     delta CI iid   [{res['delta_acc_ci_low_iid']:+.5f}, "
              f"{res['delta_acc_ci_high_iid']:+.5f}]")
        print(f"     delta CI BLOCK [{res['delta_acc_ci_low_block']:+.5f}, "
              f"{res['delta_acc_ci_high_block']:+.5f}]   (block_len={res['block_len']}) <- governs")
        print(f"     McNemar exact  b={res['mcnemar_b']} c={res['mcnemar_c']} "
              f"p={res['mcnemar_p']:.6g}")

    print('\n7. VERDICTS')
    print(f"   H_dir.1  GBM  vs train-majority : {r['h_dir_1']['verdict']}")
    print(f"   H_dir.2  LSTM vs GBM            : {r['h_dir_2']['verdict']}")
    print('   (corroborating comparison is context only — never a path to KEEP)')

    print('\n8. CONDITIONAL REPLICATION ARM (H_dir.3/.4/.5 — GBPUSD/AUDUSD/CHFUSD)')
    if not r['replication_triggered']:
        print(f"   trigger did NOT fire (H_dir.1 and H_dir.2 both DROPped)")
        print(f"   status = {UNSPENT_VERDICT}: never run, consumed NO alpha,")
        print(f"   family stays at size 2 (bar 0.025). Logged for the record.")
    else:
        print(f"   TRIGGERED — winning model = {r['replication_winner']}; family "
              f"2 -> 5, alpha tightened to {REPLICATION_ALPHA}")
        print("   Reporting rule: sign + rough magnitude must persist across ALL "
              "THREE pairs;\n   one clearing pair out of three is evidence AGAINST "
              "a real effect.")
        for rep in r['replication']:
            a = rep['arbiter']
            print(f"\n   H_dir.{rep['n']}  {rep['pair']}")
            print(f"     acc {a['acc_challenger']:.4f} vs majority {a['acc_reference']:.4f}   "
                  f"delta = {a['delta_acc']:+.4f}")
            print(f"     delta CI BLOCK [{a['delta_acc_ci_low_block']:+.5f}, "
                  f"{a['delta_acc_ci_high_block']:+.5f}]   McNemar p={a['mcnemar_p']:.6g}"
                  f"   -> {a['verdict']}")


if __name__ == '__main__':
    _print_report(run())
