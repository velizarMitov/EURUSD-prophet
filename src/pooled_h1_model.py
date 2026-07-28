"""
Pooled multi-instrument H1 model + pre-registered hypotheses (research-only).

WHAT THIS TESTS
---------------
Does training one shared architecture across FOUR correlated majors (EURUSD,
GBPUSD, AUDUSD, and the inverted CHFUSD) beat training the IDENTICAL architecture
on EURUSD ALONE, when BOTH arms are scored on the SAME EURUSD validation rows?
The question is strictly COMPARATIVE (pooled vs single) -- not "does the pooled
model predict well" (we expect near-chance, as always on H1 FX). Two
pre-registered hypotheses (family size 2, Bonferroni bar alpha = 0.05/2 = 0.025):

  * H_pool.1 -- XGBoost (GPU), pooled vs EURUSD-only
  * H_pool.2 -- PyTorch LSTM (24-bar seq), pooled vs EURUSD-only

HARD BOUNDARY -- research only, production untouched
----------------------------------------------------
Writes ONLY to results/pooled_h1/ and results/pooled_h1_hypothesis_log.csv.
Never modifies models/, _train_pipeline.py, src/inference.py, src/features.py,
src/paper_trading.py, config.json, or results/eurusd_h1.csv (the last feeds the
LIVE H1 ensemble). Reuses src/triple_barrier.py UNCHANGED and builds its OWN
pair-agnostic feature set here (does NOT touch the production D1 feature
contract in src/features.py). A unit test hashes the whole protected set
before/after a run.

PRE-REGISTERED DESIGN (fixed before any result was looked at)
-------------------------------------------------------------
* Features -- pair-agnostic, scale-free ONLY (no raw price level, no tick
  volume, no symbol id): log returns at lags {1,2,3,6,12,24}; ATR(14)/close;
  RSI(14); (close-SMA50)/ATR14; (close-SMA200)/ATR14; EWMA(24) std of log
  returns; hour-of-day sin/cos; day-of-week sin/cos.
* Target -- triple-barrier (src/triple_barrier.py, unchanged): per-pair
  horizon_vol = EWMA_std(span=24)*sqrt(120), target x1.5 / stop x1.0,
  time-barrier resolved with the cost-aware rule at 1.5 pips (0.00015 price
  units), long direction (+1) at every bar. Binary: 1 = target hit first.
* Global chronological split on the SHARED (intersected) window, IDENTICAL
  date boundaries for all four pairs: train [0:70%], validation [70:85%] (the
  arbiter), test [85:100%] (RESERVED -- untouched here). Per-pair splitting is
  forbidden (cross-sectional leakage -- the pairs move contemporaneously).
* Purge + embargo (mandatory here -- every bar is a sample, labels span 120
  bars): PURGE any train row whose 120-bar resolution window crosses the
  train/val boundary; EMBARGO the first 120 validation bars.
"""

import os
import random

import numpy as np
import pandas as pd

from src.triple_barrier import (
    ewma_log_return_std, horizon_vol_from_ewma_std, triple_barrier_label,
)
from src.pooled_h1_data import POOLED_PAIRS, PIP_SIZE, POOLED_DIR, build_pooled_pairs

# ── Pre-registered constants (frozen; do not tune after seeing results) ──
HORIZON_BARS = 120
EWMA_SPAN = 24
TARGET_MULT = 1.5
STOP_MULT = 1.0
COST_PIPS = 1.5
COST_PRICE = COST_PIPS * PIP_SIZE          # 1.5 pips = 0.00015 price units
LONG_DIRECTION = 1                          # every bar is a long entry
SEQ_LEN = 24                               # LSTM sequence length (bars)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85                            # validation END (test = [0.85:1.0])
BLOCK_LEN = HORIZON_BARS                    # moving-block bootstrap block (>= label horizon)
N_BOOT = 2000
FAMILY_SIZE = 2
FAMILY_ALPHA = 0.05 / FAMILY_SIZE          # 0.025
RANDOM_SEED = 42

FEATURE_COLUMNS = [
    'logret_1', 'logret_2', 'logret_3', 'logret_6', 'logret_12', 'logret_24',
    'atr14_norm', 'rsi14', 'dist_sma50_atr', 'dist_sma200_atr', 'ewma_std_24',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

HYPOTHESIS_LOG = 'results/pooled_h1_hypothesis_log.csv'
RUN_LOG = os.path.join(POOLED_DIR, 'pooled_h1_run_log.csv')

LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter', 'symbols_used', 'n_pooled_rows',
    'n_eurusd_rows', 'rho_bar', 'k_eff', 'mean_label_uniqueness', 'n_purged',
    'n_embargoed', 'n_train', 'n_val', 'acc_challenger', 'acc_reference',
    'delta_acc', 'delta_acc_ci_low', 'delta_acc_ci_high', 'mcnemar_b',
    'mcnemar_c', 'mcnemar_p', 'block_len', 'alpha', 'cleared_bar', 'verdict',
    'device_used', 'notes',
]


# ───────────────────────── features (fresh, pair-agnostic) ─────────────────────

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI on a close series (causal EWM smoothing, adjust=False). A
    flat window maps to the neutral 50 convention."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)          # all-gain window -> 100
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)  # flat -> 50
    return rsi


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR (causal): EWM(alpha=1/period, adjust=False) of the true range.
    Trailing by construction -- entry t never reflects a future bar."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_pooled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the pre-registered pair-agnostic, scale-free feature matrix from one
    OHLC frame. Every column is trailing/causal (no look-ahead): a feature at
    bar t is computable from bars <= t only. NO raw price level, NO tick volume,
    NO symbol identifier. Index is preserved; warm-up rows carry NaN and are
    dropped downstream.
    """
    close, high, low = df['close'], df['high'], df['low']
    out = pd.DataFrame(index=df.index)

    log_close = np.log(close)
    for k in (1, 2, 3, 6, 12, 24):
        out[f'logret_{k}'] = log_close - log_close.shift(k)

    atr14 = _wilder_atr(high, low, close, 14)
    out['atr14_norm'] = atr14 / close
    out['rsi14'] = _wilder_rsi(close, 14)

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out['dist_sma50_atr'] = (close - sma50) / atr14
    out['dist_sma200_atr'] = (close - sma200) / atr14

    # EWMA(24) std of log returns -- reuse the SAME causal estimator the
    # triple-barrier volatility scale uses, so feature and label share one
    # definition of "recent volatility".
    out['ewma_std_24'] = ewma_log_return_std(close.to_numpy(), EWMA_SPAN)

    hour = df.index.hour.to_numpy()
    dow = df.index.dayofweek.to_numpy()
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    return out[FEATURE_COLUMNS]


# ───────────────────────── labels (triple_barrier, unchanged) ─────────────────

def label_pair(df: pd.DataFrame):
    """
    Triple-barrier label EVERY bar of one pair as a long entry (direction +1),
    reusing src/triple_barrier.py unchanged. Returns (labels, resolve_end_ts)
    as float / datetime arrays aligned to df.index:
      * labels[i]         -- 1 target-first, 0 stop-or-time-first, NaN if the
                             120-bar horizon runs past the series end (excluded).
      * resolve_end_ts[i] -- timestamp of bar i+HORIZON_BARS (the far edge of
                             the resolution window), NaT where unlabelable. Used
                             by the purge to detect a window crossing the
                             train/val boundary -- conservatively the FULL
                             horizon, independent of where the barrier actually
                             hit.
    """
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    n = len(close)

    per_bar_std = ewma_log_return_std(close, EWMA_SPAN)
    horizon_vol = horizon_vol_from_ewma_std(per_bar_std, HORIZON_BARS)

    labels = np.full(n, np.nan)
    resolve_end = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
    index_values = df.index.values  # datetime64[ns] (UTC dropped to values ok for positional lookup)

    for i in range(n):
        hv = horizon_vol[i]
        if not np.isfinite(hv) or hv <= 0.0:
            continue
        label, _outcome = triple_barrier_label(
            high, low, close, entry_idx=i, direction=LONG_DIRECTION,
            horizon_vol=float(hv), horizon_bars=HORIZON_BARS, cost_price=COST_PRICE,
            target_mult=TARGET_MULT, stop_mult=STOP_MULT,
        )
        if label is None:
            continue  # insufficient_history -> excluded, never padded
        labels[i] = float(label)
        resolve_end[i] = index_values[i + HORIZON_BARS]

    return labels, resolve_end


def build_pair_dataset(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Feature matrix + triple-barrier label for one pair, indexed by timestamp,
    with NaN feature/label warm-up rows dropped. Carries a `pair` tag and the
    `resolve_end_ts` needed by the purge (neither is ever a model feature)."""
    feats = compute_pooled_features(df)
    labels, resolve_end = label_pair(df)
    feats = feats.copy()
    feats['label'] = labels
    feats['resolve_end_ts'] = pd.to_datetime(resolve_end, utc=True)
    feats['pair'] = pair
    feats = feats.dropna(subset=FEATURE_COLUMNS + ['label'])
    feats['label'] = feats['label'].astype(int)
    return feats


# ───────────────────────── shared window + global split ───────────────────────

def build_pooled_dataset(out_dir: str = POOLED_DIR):
    """
    Assemble the four per-pair datasets, intersect them to their common
    timestamp window, and return everything the split/accounting/model steps
    need. Returns a dict with:
      * 'per_pair'  -- {pair: DataFrame restricted to the shared window}
      * 'common'    -- the sorted shared DatetimeIndex
      * 'span'      -- (start, end) iso strings of the shared window
      * 'raw_returns' -- {pair: aligned log-return Series on the shared window}
    """
    pairs = build_pooled_pairs(out_dir=out_dir)
    datasets = {p: build_pair_dataset(pairs[p], p) for p in POOLED_PAIRS}

    common = None
    for p in POOLED_PAIRS:
        idx = datasets[p].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()

    per_pair = {p: datasets[p].loc[common].sort_index() for p in POOLED_PAIRS}

    # Contemporaneous log returns on the shared window (for STEP 2 accounting).
    raw_returns = {}
    for p in POOLED_PAIRS:
        c = np.log(pairs[p]['close']).reindex(common)
        raw_returns[p] = c.diff()

    return {
        'per_pair': per_pair,
        'common': common,
        'span': (common.min().isoformat(), common.max().isoformat()),
        'raw_returns': raw_returns,
    }


def global_split_boundaries(common: pd.DatetimeIndex):
    """IDENTICAL train/val/test date boundaries for every pair, from the shared
    window. Returns (train_end_ts, val_end_ts): train = [start, train_end_ts),
    val = [train_end_ts, val_end_ts), test = [val_end_ts, end]."""
    n = len(common)
    train_end_ts = common[int(n * TRAIN_FRAC)]
    val_end_ts = common[int(n * VAL_FRAC)]
    return train_end_ts, val_end_ts


def split_purge_embargo(per_pair, common, train_end_ts, val_end_ts):
    """
    Apply the GLOBAL split + purge + embargo to every pair identically.

    Returns (splits, counts) where splits[pair] = {'train','val','test'}
    DataFrames and counts[pair] = {'n_purged','n_embargoed'}.

      * PURGE (train): drop any train row whose 120-bar resolution window
        (resolve_end_ts) reaches at/after the train/val boundary -- its label
        is partly determined by validation-era bars.
      * EMBARGO (val): drop the first HORIZON_BARS rows of the validation slice
        (positional, per pair -- identical count across pairs since the shared
        window is aligned).
    """
    splits, counts = {}, {}
    for p in POOLED_PAIRS:
        d = per_pair[p]
        train = d[d.index < train_end_ts]
        val = d[(d.index >= train_end_ts) & (d.index < val_end_ts)]
        test = d[d.index >= val_end_ts]

        n_train_pre = len(train)
        train_purged = train[train['resolve_end_ts'] < train_end_ts]
        n_purged = n_train_pre - len(train_purged)

        n_val_pre = len(val)
        val_embargoed = val.iloc[HORIZON_BARS:] if len(val) > HORIZON_BARS else val.iloc[0:0]
        n_embargoed = n_val_pre - len(val_embargoed)

        splits[p] = {
            'train': train_purged, 'val': val_embargoed, 'test': test,
            # 'full' is the pair's whole shared-window cleaned frame -- the
            # source of a 24-bar sequence's PRECEDING context (which may reach
            # back across the split boundary; that is input context, never a
            # label, so it introduces no leakage -- same convention as the
            # production LSTM warm-up).
            'full': d,
        }
        counts[p] = {'n_purged': int(n_purged), 'n_embargoed': int(n_embargoed)}
    return splits, counts


# ───────────────────────── STEP 2: effective-sample accounting ────────────────

def effective_sample_accounting(splits, raw_returns, common, train_end_ts):
    """
    Honest denominator (run BEFORE any model). Returns a dict with:
      * 'corr_matrix'          -- 4x4 contemporaneous H1 log-return corr, TRAIN slice only
      * 'rho_bar'              -- average pairwise correlation
      * 'k_eff'                -- k / (1 + (k-1)*rho_bar), k=4
      * 'mean_label_uniqueness'-- Lopez de Prado label uniqueness, EURUSD train slice
      * 'n_pooled_rows'        -- total pooled train rows (all pairs)
      * 'n_eurusd_rows'        -- EURUSD train rows
    """
    train_mask = common < train_end_ts
    rets = pd.DataFrame({p: raw_returns[p][train_mask] for p in POOLED_PAIRS}).dropna()
    corr = rets.corr()

    k = len(POOLED_PAIRS)
    off = corr.to_numpy()[~np.eye(k, dtype=bool)]
    rho_bar = float(off.mean())
    k_eff = k / (1.0 + (k - 1) * rho_bar)

    uniqueness = mean_label_uniqueness(splits['EURUSD']['train'])

    n_pooled_rows = int(sum(len(splits[p]['train']) for p in POOLED_PAIRS))
    n_eurusd_rows = int(len(splits['EURUSD']['train']))

    return {
        'corr_matrix': corr,
        'rho_bar': rho_bar,
        'k_eff': float(k_eff),
        'mean_label_uniqueness': float(uniqueness),
        'n_pooled_rows': n_pooled_rows,
        'n_eurusd_rows': n_eurusd_rows,
    }


def mean_label_uniqueness(train_df: pd.DataFrame) -> float:
    """
    Mean Lopez de Prado label uniqueness on one pair's train rows: each label
    spans [entry_ts, resolve_end_ts]; a bar's concurrency c_t is how many labels
    are live at t; a label's uniqueness is the average of 1/c_t over its own
    span; the reported number is the mean over labels. Overlapping 120-bar
    labels drive this well below 1, so raw row counts overstate information.
    Computed on the label spans within `train_df` (positional over the shared
    train grid).
    """
    if len(train_df) == 0:
        return float('nan')
    idx = train_df.index
    pos = {ts: i for i, ts in enumerate(idx)}
    n = len(idx)
    concurrency = np.zeros(n)
    spans = []
    for ts, end_ts in zip(idx, train_df['resolve_end_ts']):
        start_i = pos[ts]
        # clamp the label span to the train grid (bars beyond it are not counted
        # here -- this is a within-train concurrency estimate).
        end_i = n - 1
        if pd.notna(end_ts):
            future = idx[idx <= end_ts]
            end_i = pos[future[-1]] if len(future) else start_i
        spans.append((start_i, end_i))
        concurrency[start_i:end_i + 1] += 1.0

    concurrency[concurrency == 0.0] = 1.0
    uniq = [np.mean(1.0 / concurrency[s:e + 1]) for s, e in spans]
    return float(np.mean(uniq))


# ───────────────────────── device + seeding (CUDA is a hard requirement) ──────

def resolve_device():
    """CUDA-first device resolution (NEVER hardcode 'cpu'). Returns
    (device, info) where info is a dict printed LOUDLY at the top of every run."""
    import torch
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    if cuda:
        device = torch.device('cuda')
    elif mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    info = {
        'cuda_available': cuda,
        'cuda_device_name': torch.cuda.get_device_name(0) if cuda else None,
        'mps_available': mps,
        'device': str(device),
    }
    return device, info


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


# ───────────────────────── arm matrix / sequence assembly ─────────────────────

def _arm_train_frames(splits, pooled: bool):
    """Ordered list of (pair, cleaned_windowed_frame, train_target_index) for an
    arm. Pooled arm = all four pairs; reference arm = EURUSD only. The frame is
    the pair's full shared-window cleaned frame (sequence context source); the
    target index is that pair's PURGED train rows."""
    pairs = POOLED_PAIRS if pooled else ('EURUSD',)
    return [(p, splits[p]['full'], splits[p]['train'].index) for p in pairs]


def fit_standardizer(splits, pooled: bool):
    """Fit a per-feature mean/std on an arm's TRAIN target rows only (pooled:
    all pairs; reference: EURUSD). Returns (mean, std) numpy arrays."""
    pairs = POOLED_PAIRS if pooled else ('EURUSD',)
    frames = [splits[p]['train'][FEATURE_COLUMNS] for p in pairs]
    allX = pd.concat(frames).to_numpy(dtype=float)
    mean = allX.mean(axis=0)
    std = allX.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def flat_matrix(splits, pooled: bool, which: str, mean, std):
    """Flat (row-per-sample) standardized feature matrix + labels for GBM.
    `which` in {'train','val'}. For 'val' the rows are ALWAYS EURUSD's scored
    val rows (both arms are judged on the identical EURUSD validation rows)."""
    if which == 'val':
        pairs = ('EURUSD',)
    else:
        pairs = POOLED_PAIRS if pooled else ('EURUSD',)
    frames = [splits[p][which] for p in pairs]
    d = pd.concat(frames)
    X = (d[FEATURE_COLUMNS].to_numpy(dtype=float) - mean) / std
    y = d['label'].to_numpy(dtype=int)
    return X, y


def build_sequences(pair_frame: pd.DataFrame, target_index, mean, std):
    """24-bar sequences ending at each target row, built STRICTLY within one
    pair's frame (a sequence can never span two instruments -- guaranteed by
    only ever indexing into a single pair_frame). Returns (seqs, ys, kept_ts)."""
    X = (pair_frame[FEATURE_COLUMNS].to_numpy(dtype=float) - mean) / std
    y = pair_frame['label'].to_numpy(dtype=int)
    pos = {ts: i for i, ts in enumerate(pair_frame.index)}
    seqs, ys, kept = [], [], []
    for ts in target_index:
        j = pos[ts]
        if j - SEQ_LEN + 1 < 0:
            continue
        seqs.append(X[j - SEQ_LEN + 1: j + 1])
        ys.append(y[j])
        kept.append(ts)
    if not seqs:
        return np.empty((0, SEQ_LEN, len(FEATURE_COLUMNS))), np.empty((0,)), []
    return np.asarray(seqs, dtype=np.float32), np.asarray(ys, dtype=np.float32), kept


def arm_sequences(splits, pooled: bool, which: str, mean, std):
    """Concatenate per-pair sequences for an arm. For 'val', sequences always
    end at EURUSD's scored val rows (identical for both arms)."""
    if which == 'val':
        pairs = ('EURUSD',)
    else:
        pairs = POOLED_PAIRS if pooled else ('EURUSD',)
    seqs, ys, pair_tags = [], [], []
    for p in pairs:
        s, yy, kept = build_sequences(splits[p]['full'], splits[p][which].index, mean, std)
        if len(s):
            seqs.append(s)
            ys.append(yy)
            pair_tags.extend([p] * len(s))
    if not seqs:
        return (np.empty((0, SEQ_LEN, len(FEATURE_COLUMNS)), dtype=np.float32),
                np.empty((0,), dtype=np.float32), [])
    return np.concatenate(seqs), np.concatenate(ys), pair_tags


# ───────────────────────── H_pool.1 : XGBoost (GPU) ───────────────────────────

def _balanced_scale_pos_weight(y):
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def train_gbm_arm(Xtr, ytr, seed=RANDOM_SEED):
    """One XGBoost arm (GPU, device='cuda'), IDENTICAL hyperparameters for both
    arms. Balanced via scale_pos_weight (the 1.5x/1.0x barriers geometrically
    favour label 0). Returns a fitted classifier."""
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


def predict_gbm(clf, Xval):
    prob = clf.predict_proba(Xval)[:, 1]
    return (prob >= 0.5).astype(int)


# ───────────────────────── H_pool.2 : PyTorch LSTM ────────────────────────────

def _make_lstm(n_features):
    import torch.nn as nn

    class PooledLSTM(nn.Module):
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
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)

    return PooledLSTM(n_features)


def train_lstm_arm(Xtr, ytr, Xval, yval, device, seed=RANDOM_SEED,
                   epochs=100, patience=10, batch_size=256):
    """One PyTorch LSTM arm. Explicit train()/eval()+no_grad() toggling (PyTorch
    does not auto-toggle Dropout). Per-sample balanced weights via BCELoss(
    weight=...) per batch (BCELoss has no pos_weight -- that is
    BCEWithLogitsLoss only). Early stopping patience=10 on validation loss.
    Returns (val_predictions, best_val_loss)."""
    import torch
    import torch.nn as nn

    seed_everything(seed)
    model = _make_lstm(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCELoss(reduction='none')

    # balanced per-sample weights (inverse class frequency), normalised to mean 1
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
    best_val = float('inf')
    best_state = None
    since_improve = 0
    g = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            wb = w_t[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = (loss_fn(pred, yb) * wb).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xval_t)
            vloss = nn.functional.binary_cross_entropy(vpred, yval_t).item()
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = model(Xval_t).detach().cpu().numpy()
    return (val_prob >= 0.5).astype(int), best_val


# ───────────────────────── STEP 4: statistical arbiter ────────────────────────

def _circular_block_bootstrap_delta(cc, cr, block_len, n_boot, alpha, seed=RANDOM_SEED):
    """Moving-block (circular) bootstrap CI for delta accuracy = mean(cc)-mean(cr)
    over the scored time axis. Resamples contiguous blocks (never i.i.d. rows)
    so overlapping labels + serial correlation are respected. Two-sided CI at
    `alpha` (percentiles alpha/2, 1-alpha/2). Returns (lo, hi, point)."""
    rng = np.random.default_rng(seed)
    n = len(cc)
    point = float(cc.mean() - cr.mean())
    if n == 0:
        return float('nan'), float('nan'), point
    n_blocks = int(np.ceil(n / block_len))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        deltas[b] = cc[idx].mean() - cr[idx].mean()
    lo = float(np.percentile(deltas, 100 * (alpha / 2)))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return lo, hi, point


def _mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) two-sided McNemar on paired correctness. b = challenger
    right & reference wrong; c = challenger wrong & reference right. Returns
    (b, c, p)."""
    from scipy.stats import binomtest
    a = np.asarray(correct_a).astype(bool)
    r = np.asarray(correct_b).astype(bool)
    b = int(np.sum(a & ~r))
    c = int(np.sum(~a & r))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative='two-sided').pvalue
    return b, c, float(p)


def arbiter(pred_challenger, pred_reference, y_true, block_len=BLOCK_LEN,
            n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=RANDOM_SEED):
    """
    Decision-bearing comparison: pooled (challenger) vs EURUSD-only (reference)
    on the IDENTICAL EURUSD validation rows. KEEP only if the delta-accuracy CI
    lies entirely above zero AND McNemar p < alpha; else DROP.
    """
    y = np.asarray(y_true).astype(int)
    cc = (np.asarray(pred_challenger).astype(int) == y).astype(float)
    cr = (np.asarray(pred_reference).astype(int) == y).astype(float)

    acc_c = float(cc.mean())
    acc_r = float(cr.mean())
    lo, hi, delta = _circular_block_bootstrap_delta(cc, cr, block_len, n_boot, alpha, seed)
    b, c, p = _mcnemar_exact(cc, cr)

    cleared = bool(lo > 0.0 and p < alpha)
    return {
        'acc_challenger': acc_c, 'acc_reference': acc_r, 'delta_acc': delta,
        'delta_acc_ci_low': lo, 'delta_acc_ci_high': hi,
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p,
        'block_len': block_len, 'alpha': alpha,
        'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP',
    }


# ───────────────────────── STEP 5: hypothesis log + orchestration ─────────────

ARBITER_LABEL = 'pooled_validation[70:85]_H1_block_bootstrap'
SYMBOLS_USED = 'EURUSD,GBPUSD,AUDUSD,CHFUSD(inverted from USDCHF)'


def _upsert_log(row: dict, log_path: str = HYPOTHESIS_LOG):
    """Append/replace one hypothesis row (idempotent on the `hypothesis`
    name), keeping the family log sorted by `n`."""
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


def _log_row(n, hypothesis, acc, accounting, counts_total, n_train, n_val,
             device_used, notes):
    return {
        'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
        'hypothesis': hypothesis, 'arbiter': ARBITER_LABEL,
        'symbols_used': SYMBOLS_USED,
        'n_pooled_rows': accounting['n_pooled_rows'],
        'n_eurusd_rows': accounting['n_eurusd_rows'],
        'rho_bar': round(accounting['rho_bar'], 6),
        'k_eff': round(accounting['k_eff'], 6),
        'mean_label_uniqueness': round(accounting['mean_label_uniqueness'], 6),
        'n_purged': counts_total['n_purged'],
        'n_embargoed': counts_total['n_embargoed'],
        'n_train': n_train, 'n_val': n_val,
        'acc_challenger': round(acc['acc_challenger'], 6),
        'acc_reference': round(acc['acc_reference'], 6),
        'delta_acc': round(acc['delta_acc'], 6),
        'delta_acc_ci_low': round(acc['delta_acc_ci_low'], 6),
        'delta_acc_ci_high': round(acc['delta_acc_ci_high'], 6),
        'mcnemar_b': acc['mcnemar_b'], 'mcnemar_c': acc['mcnemar_c'],
        'mcnemar_p': round(acc['mcnemar_p'], 6),
        'block_len': acc['block_len'], 'alpha': acc['alpha'],
        'cleared_bar': acc['cleared_bar'], 'verdict': acc['verdict'],
        'device_used': device_used, 'notes': notes,
    }


def run(out_dir: str = POOLED_DIR, log_path: str = HYPOTHESIS_LOG,
        seed: int = RANDOM_SEED, register: bool = True):
    """
    Full pooled-vs-single program end to end. Returns a dict with the device
    info, effective-sample accounting, per-pair purge/embargo counts, and both
    hypotheses' arbiter results. Writes results/pooled_h1_hypothesis_log.csv
    (when register=True) and a per-pair run log. Never touches any production
    artifact.
    """
    device, dev_info = resolve_device()
    used_seed = seed_everything(seed)
    print('=' * 70)
    print('POOLED MULTI-INSTRUMENT H1 — device & determinism')
    print(f"  torch.cuda.is_available() = {dev_info['cuda_available']}")
    if dev_info['cuda_available']:
        print(f"  torch.cuda.get_device_name(0) = {dev_info['cuda_device_name']}")
    else:
        print("  *** CUDA NOT AVAILABLE — running on fallback:",
              dev_info['device'], "***")
    print(f"  resolved device = {device}")
    print(f"  seed = {used_seed} (bitwise GPU determinism NOT guaranteed; accepted)")
    print('=' * 70)

    data = build_pooled_dataset(out_dir=out_dir)
    common = data['common']
    train_end_ts, val_end_ts = global_split_boundaries(common)
    splits, counts = split_purge_embargo(data['per_pair'], common, train_end_ts, val_end_ts)

    accounting = effective_sample_accounting(splits, data['raw_returns'], common, train_end_ts)
    counts_total = {
        'n_purged': int(sum(counts[p]['n_purged'] for p in POOLED_PAIRS)),
        'n_embargoed': int(sum(counts[p]['n_embargoed'] for p in POOLED_PAIRS)),
    }

    # ── H_pool.1 : XGBoost (GPU) ──
    mean_pool, std_pool = fit_standardizer(splits, pooled=True)
    mean_eur, std_eur = fit_standardizer(splits, pooled=False)

    Xtr_pool, ytr_pool = flat_matrix(splits, True, 'train', mean_pool, std_pool)
    Xtr_eur, ytr_eur = flat_matrix(splits, False, 'train', mean_eur, std_eur)
    Xval_pool, yval = flat_matrix(splits, True, 'val', mean_pool, std_pool)
    Xval_eur, _ = flat_matrix(splits, False, 'val', mean_eur, std_eur)

    gbm_pool = train_gbm_arm(Xtr_pool, ytr_pool, seed)
    gbm_eur = train_gbm_arm(Xtr_eur, ytr_eur, seed)
    pred_pool = predict_gbm(gbm_pool, Xval_pool)
    pred_eur = predict_gbm(gbm_eur, Xval_eur)
    gbm_acc = arbiter(pred_pool, pred_eur, yval, seed=seed)

    gbm_notes = (
        f"XGBoost hist/cuda identical HP both arms; scale_pos_weight balanced; "
        f"k=4 pooled pairs; k_eff={accounting['k_eff']:.3f} vs raw 4x; "
        f"label_uniqueness={accounting['mean_label_uniqueness']:.3f} -> raw rows overstate info; "
        f"purge/embargo per pair; corroborating-only vs majority baseline not decision-bearing."
    )
    gbm_row = _log_row(1, 'H_pool.1_GBM_pooled_vs_eurusd', gbm_acc, accounting,
                       counts_total, n_train=len(ytr_pool), n_val=len(yval),
                       device_used=dev_info['device'], notes=gbm_notes)

    # ── H_pool.2 : PyTorch LSTM ──
    Xs_pool, ys_pool, _ = arm_sequences(splits, True, 'train', mean_pool, std_pool)
    Xs_eur, ys_eur, _ = arm_sequences(splits, False, 'train', mean_eur, std_eur)
    Xsv_pool, ysv, _ = arm_sequences(splits, True, 'val', mean_pool, std_pool)
    Xsv_eur, ysv_eur, _ = arm_sequences(splits, False, 'val', mean_eur, std_eur)

    lstm_pred_pool, vloss_pool = train_lstm_arm(Xs_pool, ys_pool, Xsv_pool, ysv, device, seed)
    lstm_pred_eur, vloss_eur = train_lstm_arm(Xs_eur, ys_eur, Xsv_eur, ysv_eur, device, seed)
    lstm_acc = arbiter(lstm_pred_pool, lstm_pred_eur, ysv.astype(int), seed=seed)

    lstm_notes = (
        f"PyTorch LSTM(32)->Dropout.3->Linear16->ReLU->Dropout.3->Linear1->Sigmoid; "
        f"Adam lr1e-3 wd1e-3 batch256 <=100ep patience10 on val loss; per-sample "
        f"balanced BCELoss; seq WITHIN instrument only; device={dev_info['device']}; "
        f"best_val_loss pooled={vloss_pool:.4f} eurusd={vloss_eur:.4f}."
    )
    lstm_row = _log_row(2, 'H_pool.2_LSTM_pooled_vs_eurusd', lstm_acc, accounting,
                        counts_total, n_train=len(ys_pool), n_val=len(ysv),
                        device_used=dev_info['device'], notes=lstm_notes)

    if register:
        _upsert_log(gbm_row, log_path)
        _upsert_log(lstm_row, log_path)
        # per-pair run log (purge/embargo + counts) for reproducibility
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([
            {'pair': p, 'n_purged': counts[p]['n_purged'],
             'n_embargoed': counts[p]['n_embargoed'],
             'n_train': len(splits[p]['train']), 'n_val': len(splits[p]['val']),
             'n_test_reserved': len(splits[p]['test'])}
            for p in POOLED_PAIRS
        ]).to_csv(RUN_LOG, index=False)

    return {
        'device_info': dev_info, 'seed': used_seed,
        'span': data['span'], 'n_common': len(common),
        'train_end_ts': train_end_ts.isoformat(), 'val_end_ts': val_end_ts.isoformat(),
        'accounting': accounting, 'counts': counts, 'counts_total': counts_total,
        'gbm': gbm_acc, 'lstm': lstm_acc,
        'gbm_row': gbm_row, 'lstm_row': lstm_row,
    }


if __name__ == '__main__':
    result = run()
    acc = result['accounting']
    print('\n' + '=' * 70)
    print('POOLED H1 — RESULTS (raw)')
    print('=' * 70)
    print(f"Shared window: {result['span'][0]} -> {result['span'][1]}  ({result['n_common']} bars)")
    print(f"\nEffective-sample accounting (LEAD WITH THIS):")
    print(f"  pooled train rows = {acc['n_pooled_rows']}  vs  EURUSD-only = {acc['n_eurusd_rows']}")
    print(f"  rho_bar = {acc['rho_bar']:.4f}   k_eff = {acc['k_eff']:.3f} (k=4)")
    print(f"  mean label uniqueness = {acc['mean_label_uniqueness']:.4f}")
    print(f"  correlation matrix (train slice):\n{acc['corr_matrix'].round(3)}")
    print(f"\nPurge/embargo per pair: {result['counts']}")
    for tag, r in (('H_pool.1 GBM', result['gbm']), ('H_pool.2 LSTM', result['lstm'])):
        print(f"\n{tag}:")
        print(f"  acc pooled={r['acc_challenger']:.4f}  eurusd={r['acc_reference']:.4f}  "
              f"delta={r['delta_acc']:+.4f}  CI[{r['delta_acc_ci_low']:+.4f},{r['delta_acc_ci_high']:+.4f}]")
        print(f"  McNemar b={r['mcnemar_b']} c={r['mcnemar_c']} p={r['mcnemar_p']:.4f}  "
              f"alpha={r['alpha']}  -> {r['verdict']}")
