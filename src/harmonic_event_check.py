"""
H1 harmonic-pattern event-conditional model — OWN hypothesis family
(results/harmonic_pattern_hypothesis_log.csv). Separate from every other
family in this project (daily direction/return, daily volatility, weekly
COT) — a different event universe (H1 harmonic-pattern completions, not
every daily/H1 bar), a different target (Lopez de Prado triple-barrier
label, not next-bar direction/return/volatility) — so it does NOT dilute or
get diluted by any other family's Bonferroni count.

TWO SWING BASES have been tried in this family, each producing a LogReg +
MLP pair (`run(swing_source=...)`), with the family's Bonferroni bar computed
DYNAMICALLY at run time from however many distinct hypothesis names are
already registered (same convention as `src.ablation.run` /
`src.volatility.run_candidate_feature_tests`):

  'fractal' (first budget, alpha=0.05/2=0.025 each) — H1.1 (LogReg) / H1.2
    (MLP), swing points from Williams fractals
    (`src.harmonic_patterns.detect_harmonic_events`, which itself reuses
    `src.fibonacci_fractals.detect_fractals` / `_push_swing` UNCHANGED),
    fixed CONFIRMATION_LAG=2. Both DROPped.
  'zigzag' (family grown to 4, alpha=0.05/4=0.0125 each) — H1.3 (LogReg) /
    H1.4 (MLP), swing points from a CAUSAL, ATR(14)-scaled ZigZag
    (`src.zigzag_swings.zigzag_swings` ->
    `src.harmonic_patterns.detect_harmonic_events_from_pivots`, VARIABLE
    confirmation lag = each pivot's own reveal_bar). Tried because a
    Williams fractal's fixed 5-bar window on H1 bars likely flags a lot of
    noisy MICRO-swings unrepresentative of genuine harmonic structure; a
    volatility-adaptive ZigZag threshold targets cleaner, more meaningful
    swings instead. See `src.zigzag_swings`' module docstring for the
    elevated look-ahead risk this alternative carries (a variable, unbounded
    confirmation lag vs. the fractal path's trivial fixed one) and how it is
    mitigated (strictly causal walk-forward construction + an explicit
    repainting-guard unit test).

`src.harmonic_patterns.py` (both the Williams-fractal AND the ZigZag event
paths) is itself NEW, UNVALIDATED code introduced for this family (see its
module docstring) — only the fractal/swing PRIMITIVES it builds on
(`src.fibonacci_fractals`) have a prior track record.

Data: results/eurusd_h1.csv (via src.h1_features.load_h1_frame).

Everything below this point describes the pipeline SHARED identically by
both swing bases (event filter, triple-barrier labeling, features, split,
models) — only the swing-point SOURCE feeding `score_xabcd` differs between
a 'fractal' and a 'zigzag' run, so any outcome difference between them is
attributable to that alone.

PRE-REGISTERED PIPELINE (fixed before looking at any result; run ONCE per swing basis)
------------------------------------------------------------------------------------------
1. Event filter — H1 bars where an XABCD pattern completes (D confirmed, i.e.
   `confirmed_at_idx`, D_idx + CONFIRMATION_LAG) AND best_fit_score >= 0.5.
   No post-hoc tuning of this threshold.

2. Triple-barrier labeling (src.triple_barrier), final as written:
     r_ewma_std  = EWMA std of H1 log returns, span=24 (`EWMA_SPAN`).
     horizon_vol = r_ewma_std * sqrt(120)  (`HORIZON_BARS`) — sqrt-time-scaled
                   to the holding horizon, NOT a plain per-bar ATR (see
                   src.triple_barrier's module docstring for the full
                   horizon-matching rationale; this replaced an earlier plain-
                   ATR draft per owner review).
     entry       = close at the event's `confirmed_at_idx` bar.
     direction   = sign of harmonic_pattern_score_signed (+1 bullish D-low /
                   -1 bearish D-high).
     target (aligned)  = entry * exp( direction * 1.5 * horizon_vol)
     stop (opposite)   = entry * exp(-direction * 1.0 * horizon_vol)
     time barrier       = 120 H1 bars forward, fixed.
   The closer 1.0x stop is geometrically more likely to be touched first than
   the farther 1.5x target under a pure random walk — independent of any real
   edge — which is exactly why BOTH models below use `class_weight='balanced'`
   (see step 3). Time-barrier resolution requires the signed move to clear the
   round-trip transaction cost, EXPLICITLY: `config.json`
   `paper_trading.spread_pips` = 1.5 pips; for EURUSD 1 pip = 0.0001
   (`src.paper_trading.PIP_SIZE`), so the price-distance threshold is
   `1.5 * 0.0001 = 0.00015` — a move smaller than that is not a realizable
   win, mirroring how `src/paper_trading.py` already nets cost rather than
   scoring a bare `sign(>0)`. Events within 120 bars of the end of available
   H1 history are EXCLUDED (`insufficient_history`), never padded/estimated.

3. Two sequentially-scaled sub-hypotheses on the SAME event subset / SAME
   chronological 70/15/15 split / SAME 8 features
   (r_AB, r_BC, r_CD, r_AD, best_fit_score, direction, swing_duration_bars,
   norm_amplitude — the last is the XA-leg amplitude normalized by the SAME
   horizon_vol already computed at the event, not a second ad hoc
   normalizer) / SAME `class_weight='balanced'` / SAME `random_state=42`
   (config.json's project-wide convention):

     H1.1 (Linear Baseline) — LogisticRegression. Tests whether the harmonic
       geometry carries a raw linear edge. Judged against the TRAIN-majority-
       class baseline. alpha = 0.05/2 = 0.025.

     H1.2 (Non-Linear Interactions) — a feed-forward MLP (NOT an LSTM — these
       are already-extracted cross-sectional ratios per event, not a time
       series), implemented in RAW PYTORCH (deliberately not Keras, unlike
       every other neural model in this project — see train_h1_2_mlp's
       docstring for the two Keras-vs-PyTorch pitfalls this guards against):
       Linear(16, L2=1e-3 via weight_decay) -> ReLU -> Dropout(0.3) ->
       Linear(8, L2=1e-3) -> ReLU -> Dropout(0.3) -> Linear(1) -> Sigmoid;
       Adam lr=0.001; up to 100 epochs; early stopping patience=10 on
       validation loss (explicit model.eval()+no_grad() each check).
       Architecture FIXED, no tuning after seeing results. batch_size=32
       (this project's existing H1 LSTM convention, config.json
       `h1.lstm_batch`) — likewise fixed a
       priori, not tuned.
       PRIMARY decision test: MLP vs H1.1's OWN predictions on the IDENTICAL
       validation rows (paired bootstrap + exact McNemar) — this is the real
       "is the extra complexity worth it" question and ALONE governs whether
       1.2 clears its bar, alpha = 0.025.
       CORROBORATING ONLY (context, never a second path to KEEP): MLP vs the
       train-majority-class baseline. If the MLP beats the majority baseline
       but does NOT beat H1.1 at the pre-registered CI, the verdict is DROP
       for 1.2 — the anti-cherry-pick rule, same convention as the weekly-COT
       Spearman-primary / logistic-corroborating test
       (src.cot_weekly_check.run).

4. Validation-only arbiter: paired bootstrap 2000 resamples + exact McNemar,
   for BOTH sub-hypotheses. Following this project's volatility-family /
   weekly-COT-extremes convention (not the flat-95%-CI convention used
   elsewhere), the bootstrap CI's own width is ALPHA-SCALED to the
   Bonferroni-corrected bar (a 97.5% CI at alpha=0.025) — a stricter alpha
   only ever RAISES the bar a KEEP must clear, never loosens it. KEEP requires
   the CI entirely favoring the challenger (> 0) AND McNemar p < alpha. TEST
   SLICE (the final 15%) is reserved strictly untouched by both
   sub-hypotheses — sized/reported, never indexed for any metric. Run once —
   no tuning of the event threshold, EWMA span, sqrt-horizon scaling, barrier
   multipliers, spread-cost threshold, class weighting, or MLP architecture
   after seeing results.

Run:  python -m src.harmonic_event_check
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.h1_features import load_h1_frame, DEFAULT_H1_CACHE
from src.harmonic_patterns import detect_harmonic_events, detect_harmonic_events_from_pivots
from src.zigzag_swings import zigzag_swings
from src.triple_barrier import (
    ewma_log_return_std, horizon_vol_from_ewma_std, triple_barrier_label,
    TARGET_MULTIPLIER_DEFAULT, STOP_MULTIPLIER_DEFAULT,
)
from src.paper_trading import PIP_SIZE

HARMONIC_LOG = 'results/harmonic_pattern_hypothesis_log.csv'
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
RANDOM_STATE = 42   # config.json's project-wide convention

# Hypothesis names per swing source (SAME family/log; the family's Bonferroni
# bar is computed dynamically at run time from however many of these are
# already registered — see run()'s alpha computation, matching the
# ablation.py / volatility.py / cot_weekly_check.py dynamic-family-size
# convention elsewhere in this project).
HYPOTHESIS_NAMES = {
    'fractal': {
        'h1': 'harmonic_h1_1_logistic_vs_majority',
        'h2': 'harmonic_h1_2_mlp_vs_h1_1_primary',
        'label1': 'H1.1', 'label2': 'H1.2',
    },
    'zigzag': {
        'h1': 'harmonic_h1_3_logistic_vs_majority_zigzag',
        'h2': 'harmonic_h1_4_mlp_vs_h1_3_primary_zigzag',
        'label1': 'H1.3', 'label2': 'H1.4',
    },
}

MIN_BEST_FIT_SCORE = 0.5     # pre-registered event filter, no post-hoc tuning
EWMA_SPAN = 24
HORIZON_BARS = 120
TARGET_MULT = TARGET_MULTIPLIER_DEFAULT   # 1.5
STOP_MULT = STOP_MULTIPLIER_DEFAULT       # 1.0
SPREAD_PIPS_DEFAULT = 1.5                 # config.json paper_trading.spread_pips

MODEL_FEATURE_COLUMNS = [
    'r_AB', 'r_BC', 'r_CD', 'r_AD', 'best_fit_score', 'direction',
    'swing_duration_bars', 'norm_amplitude',
]

HARMONIC_LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter',
    'n_events_raw', 'n_events_filtered', 'n_events_labeled',
    'n_train', 'n_val', 'n_test',
    'acc_challenger', 'acc_reference', 'delta_acc',
    'delta_acc_ci_low', 'delta_acc_ci_high',
    'mcnemar_b', 'mcnemar_c', 'mcnemar_p',
    'alpha', 'cleared_bar', 'verdict', 'notes',
]


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def _upsert_log(row, out_path):
    """Upsert one hypothesis row by name (re-run refreshes that row; other
    hypotheses' rows are preserved) — same convention as
    src.cot_weekly_check._upsert_weekly_log."""
    new = pd.DataFrame([row], columns=HARMONIC_LOG_COLUMNS)
    if os.path.exists(out_path):
        log = pd.read_csv(out_path)
        log = log[log['hypothesis'] != row['hypothesis']]
        out = pd.concat([log, new], ignore_index=True)
    else:
        out = new
    out['n'] = range(1, len(out) + 1)
    out = out[HARMONIC_LOG_COLUMNS]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


# ── event dataset construction ────────────────────────────────────────────

def build_event_dataset(base_dir='', h1=None, cache_path=None,
                        min_best_fit_score=MIN_BEST_FIT_SCORE,
                        ewma_span=EWMA_SPAN, horizon_bars=HORIZON_BARS,
                        target_mult=TARGET_MULT, stop_mult=STOP_MULT,
                        spread_pips=SPREAD_PIPS_DEFAULT, random_state=RANDOM_STATE,
                        swing_source='fractal'):
    """Build the full labeled event dataset from H1 OHLC.

    `swing_source` selects the swing-point basis feeding the SAME
    `score_xabcd` ratio scoring, unchanged either way:
      'fractal' — Williams fractals (`src.harmonic_patterns.detect_harmonic_events`,
                  fixed CONFIRMATION_LAG=2 confirmation), H1.1/H1.2's basis.
      'zigzag'  — ATR-scaled causal ZigZag pivots
                  (`src.zigzag_swings.zigzag_swings` ->
                  `src.harmonic_patterns.detect_harmonic_events_from_pivots`,
                  VARIABLE confirmation lag = each pivot's own reveal_bar),
                  H1.3/H1.4's basis. Everything else in this function
                  (labeling, feature construction, baseline (b)) is IDENTICAL
                  regardless of `swing_source`, so any outcome difference is
                  attributable to the swing basis alone.

    Returns a dict:
      'h1'                the loaded H1 frame (UTC index)
      'events_raw'        every detected XABCD event (unfiltered)
      'events_filtered'   events_raw with best_fit_score >= min_best_fit_score
      'dataset'           ONE ROW PER successfully labeled filtered event
                          (label + outcome + MODEL_FEATURE_COLUMNS),
                          chronologically sorted by entry bar
      'excluded_insufficient_history'  filtered events dropped because their
                          120-bar horizon ran past the end of available
                          history (excluded, never padded/estimated)
      'baseline_b'        the non-event random-sample descriptive baseline
                          (item 3's baseline (b); context only, never part of
                          the KEEP/DROP decision)

    No look-ahead: horizon_vol[t] (via ewma_log_return_std, a causal/trailing
    pandas .ewm) depends only on bars <= t; each event's entry/direction is
    fixed at its own confirmed_at_idx (fixed-lag for 'fractal', the pivot's
    own reveal_bar for 'zigzag'); triple_barrier_label walks strictly forward
    from there. See tests for explicit truncation-equivalence checks.
    """
    cache_path = cache_path or DEFAULT_H1_CACHE
    h1 = load_h1_frame(_p(base_dir, cache_path)) if h1 is None else h1
    high = h1['high'].to_numpy(dtype=float)
    low = h1['low'].to_numpy(dtype=float)
    close = h1['close'].to_numpy(dtype=float)
    n = len(h1)

    r_ewma_std = ewma_log_return_std(close, ewma_span)
    horizon_vol = horizon_vol_from_ewma_std(r_ewma_std, horizon_bars)
    cost_price = spread_pips * PIP_SIZE

    if swing_source == 'fractal':
        events_raw = detect_harmonic_events(h1[['high', 'low', 'close']])
    elif swing_source == 'zigzag':
        pivots = zigzag_swings(high, low, close)
        events_raw = detect_harmonic_events_from_pivots(pivots)
    else:
        raise ValueError(f"unknown swing_source: {swing_source!r}")
    events_filtered = (events_raw[events_raw['best_fit_score'] >= min_best_fit_score]
                       .reset_index(drop=True))

    rows = []
    excluded = 0
    for _, ev in events_filtered.iterrows():
        entry_idx = int(ev['confirmed_at_idx'])
        if entry_idx >= n or np.isnan(horizon_vol[entry_idx]):
            excluded += 1
            continue
        direction = int(ev['direction'])
        hv = float(horizon_vol[entry_idx])
        label, outcome = triple_barrier_label(
            high, low, close, entry_idx, direction, hv, horizon_bars,
            cost_price, target_mult=target_mult, stop_mult=stop_mult)
        if label is None:
            excluded += 1
            continue
        rows.append({
            'entry_idx': entry_idx, 'entry_time': h1.index[entry_idx],
            'r_AB': float(ev['r_AB']), 'r_BC': float(ev['r_BC']),
            'r_CD': float(ev['r_CD']), 'r_AD': float(ev['r_AD']),
            'best_fit_score': float(ev['best_fit_score']), 'direction': direction,
            'swing_duration_bars': float(ev['swing_duration_bars']),
            'norm_amplitude': float(ev['xa_amplitude']) / hv if hv > 0 else 0.0,
            'label': int(label), 'outcome': outcome,
        })
    dataset = (pd.DataFrame(rows, columns=['entry_idx', 'entry_time'] + MODEL_FEATURE_COLUMNS
                            + ['label', 'outcome'])
              .sort_values('entry_idx').reset_index(drop=True))

    # ---- Baseline (b): SAME triple-barrier scheme on an equal-sized random
    # sample of NON-event H1 bars -- descriptive context only (item 3), NEVER
    # part of the H1.1/H1.2 KEEP/DROP decision (item 4 governs that). No
    # harmonic signal exists at a non-event bar, so direction is assigned at
    # random (explicit modeling choice, stated plainly).
    event_entry_idx = set(dataset['entry_idx'].tolist())
    valid_bars = np.array([
        i for i in range(2, n - horizon_bars)
        if i not in event_entry_idx and not np.isnan(horizon_vol[i])
    ])
    rng = np.random.default_rng(random_state)
    sample_n = min(len(dataset), len(valid_bars))
    sample_idx = (rng.choice(valid_bars, size=sample_n, replace=False)
                  if sample_n > 0 else np.array([], dtype=int))
    sample_dirs = rng.choice([-1, 1], size=sample_n) if sample_n > 0 else np.array([], dtype=int)
    baseline_labels = []
    for idx, d in zip(sample_idx, sample_dirs):
        hv = float(horizon_vol[int(idx)])
        lbl, _out = triple_barrier_label(high, low, close, int(idx), int(d), hv,
                                         horizon_bars, cost_price,
                                         target_mult=target_mult, stop_mult=stop_mult)
        if lbl is not None:
            baseline_labels.append(lbl)
    baseline_b = {
        'n_sample': len(baseline_labels),
        'label_1_rate': float(np.mean(baseline_labels)) if baseline_labels else float('nan'),
    }

    return {
        'h1': h1, 'events_raw': events_raw, 'events_filtered': events_filtered,
        'dataset': dataset, 'excluded_insufficient_history': excluded,
        'baseline_b': baseline_b,
    }


def _chronological_split(n, train_frac=0.70, val_frac=0.15):
    """70/15/15 chronological split on the EVENT subset (identical for both
    sub-hypotheses). The final 15% (test) is reserved and never read here."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return train_end, val_end


# ── statistics: generic paired-bootstrap + exact McNemar ─────────────────

def _mcnemar_exact(correct_a, correct_b):
    """Two-sided exact-binomial McNemar on the discordant pairs of A vs B over
    identical rows. b = A wrong & B correct, c = A correct & B wrong."""
    from math import comb
    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * p)


def bootstrap_delta_and_mcnemar(y_val, pred_reference, pred_challenger, alpha,
                                n_boot=BOOTSTRAP_RESAMPLES, random_state=RANDOM_STATE):
    """Paired bootstrap CI of (acc_challenger - acc_reference) on IDENTICAL
    validation rows + exact McNemar(reference, challenger). Positive delta
    favors the challenger. The CI's own width is ALPHA-SCALED (a
    `1 - alpha` CI, e.g. 97.5% at alpha=0.025) — matching this project's
    volatility-family / weekly-COT-extremes convention, not the flat-95%
    convention used elsewhere: a stricter alpha only ever RAISES the bar.

    Generic and reused for BOTH comparisons this module needs: H1.1 vs the
    train-majority baseline, and H1.2's PRIMARY test vs H1.1's OWN
    predictions — the SAME function, called on the SAME validation rows both
    times, so H1.2's primary test is never a separately (re)sampled
    "baseline" but a direct row-for-row comparison against H1.1's actual
    predictions.
    """
    y_val = np.asarray(y_val)
    pred_reference = np.asarray(pred_reference)
    pred_challenger = np.asarray(pred_challenger)
    correct_ref = (pred_reference == y_val)
    correct_chg = (pred_challenger == y_val)
    acc_ref, acc_chg = float(correct_ref.mean()), float(correct_chg.mean())

    rng = np.random.default_rng(random_state)
    n = len(y_val)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = correct_chg[idx].mean() - correct_ref[idx].mean()
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    b, c, mcp = _mcnemar_exact(correct_ref, correct_chg)

    cleared = bool(lo > 0 and mcp < alpha)
    return {
        'acc_reference': round(acc_ref, 4), 'acc_challenger': round(acc_chg, 4),
        'delta_acc': round(acc_chg - acc_ref, 4),
        'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': round(mcp, 4),
        'cleared': cleared,
    }


# ── models ─────────────────────────────────────────────────────────────

def _balanced_class_weight_dict(y_train):
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_h1_1_logistic(X_train, y_train, random_state=RANDOM_STATE):
    """H1.1: plain logistic regression, class_weight='balanced' -- the closer
    (1.0x) stop barrier is geometrically more likely to be touched before the
    farther (1.5x) target under a pure random walk, purely from distance,
    independent of any real edge; without balancing the model could trivially
    collapse to the majority class and become indistinguishable from the
    baseline it is judged against."""
    return LogisticRegression(class_weight='balanced', random_state=random_state,
                              max_iter=1000).fit(X_train, y_train)


class _HarmonicMLP:
    """H1.2's feed-forward MLP, raw PyTorch (deliberately NOT Keras — unlike
    every other neural model in this project — see class docstring below for
    why the two frameworks are not interchangeable here without care).
    Architecture FIXED, no tuning after results: Linear(n_features->16) ->
    ReLU -> Dropout(0.3) -> Linear(16->8) -> ReLU -> Dropout(0.3) ->
    Linear(8->1) -> Sigmoid. NOT an LSTM -- these are already-extracted
    cross-sectional per-event ratios, not a time series.

    L2=1e-3 is applied via the Adam optimizer's `weight_decay` (PyTorch's
    standard L2 idiom) -- NOT numerically identical to Keras's
    `kernel_regularizer=l2(1e-3)` (which adds the penalty to the loss before
    Adam's adaptive scaling; `weight_decay` on vanilla `torch.optim.Adam`
    perturbs the gradient directly), but the same L2=1e-3 STRENGTH, stated
    honestly as an implementation-detail difference between frameworks, not
    silently glossed over.
    """
    def __init__(self, n_features):
        import torch.nn as nn
        self.nn = nn
        self.module = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 8), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def train_mode(self):
        self.module.train()

    def eval_mode(self):
        self.module.eval()

    def __call__(self, x):
        return self.module(x)


def train_h1_2_mlp(X_train, y_train, X_val, y_val, random_state=RANDOM_STATE):
    """H1.2: the MLP above, trained with a MANUAL PyTorch loop (Adam lr=0.001,
    up to 100 epochs, early stopping patience=10 on VALIDATION loss,
    batch_size=32 -- this project's existing H1 LSTM convention,
    config.json h1.lstm_batch).

    TWO PyTorch pitfalls this function deliberately guards against (neither
    exists in Keras, whose `.fit()`/`.predict()` handle both automatically --
    exactly why this deserves its own docstring rather than silently reusing
    the Keras idiom the rest of this project uses for its LSTMs):

    1. PyTorch does NOT auto-toggle Dropout/BatchNorm between train and eval
       behavior -- `model.train()` is called before every training batch,
       and `model.eval()` (+ `torch.no_grad()`) before every validation-loss
       computation AND the final val-set prediction. Forgetting this leaves
       Dropout ACTIVE during validation, silently corrupting both the
       early-stopping signal and the reported predictions with no error.
    2. The architecture keeps an explicit Sigmoid output layer (matching the
       original Dense(1, sigmoid) spec), so the loss must be `BCELoss`, NOT
       `BCEWithLogitsLoss` (logits-only) -- and plain `BCELoss` has no
       `pos_weight` argument (that is `BCEWithLogitsLoss`-only). class
       balancing is therefore done via an explicit PER-SAMPLE weight tensor
       (`weight[i] = class_weight[y[i]]`, from the SAME balanced class
       weights H1.1 and the Keras draft used) passed to
       `BCELoss(weight=...)` for EACH batch (a per-batch weight vector, not a
       single scalar). Reason for balancing at all, same as H1.1: the closer
       (1.0x) stop barrier is geometrically more likely to be touched before
       the farther (1.5x) target under a pure random walk, independent of
       any real edge; unbalanced, the model could trivially collapse to the
       majority class and become indistinguishable from the baseline it is
       judged against.

    CPU-only, matching this project's existing determinism/simplicity
    convention for its other neural models (TensorFlow is explicitly
    documented elsewhere as GPU-unsupported on native Windows in this
    environment) -- run on CPU here too even though CUDA happens to be
    available, so results do not depend on which machine trains them.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(random_state)
    device = torch.device('cpu')

    n_features = X_train.shape[1]
    model = _HarmonicMLP(n_features)
    model.module.to(device)

    class_weight = _balanced_class_weight_dict(y_train)   # {0: w0, 1: w1}
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    optimizer = torch.optim.Adam(model.module.parameters(), lr=0.001, weight_decay=1e-3)

    best_val_loss = float('inf')
    best_state = None
    patience, patience_left = 10, 10

    for _epoch in range(100):
        # ---- train: Dropout ACTIVE ----------------------------------------
        model.train_mode()
        for xb, yb in loader:
            sample_weight = torch.tensor(
                [class_weight[int(v.item())] for v in yb], dtype=torch.float32, device=device
            ).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb, weight=sample_weight)
            loss.backward()
            optimizer.step()

        # ---- validate: Dropout OFF, no gradient tracking -------------------
        model.eval_mode()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = F.binary_cross_entropy(val_pred, y_val_t).item()   # UNWEIGHTED, matches
            # Keras's own default (class_weight applies to TRAINING only;
            # EarlyStopping's val_loss is plain BCE on the validation set).

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.module.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.module.load_state_dict(best_state)   # restore_best_weights equivalent
    model.eval_mode()
    return model


def predict_h1_2_mlp(model, X):
    """Final val/test prediction: model.eval() + no_grad(), exactly like the
    validation-loss computation inside train_h1_2_mlp -- Dropout must be OFF
    here too, or the reported predictions are corrupted the same way an
    un-toggled validation loss would be."""
    import torch
    model.eval_mode()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        prob = model(X_t).numpy().ravel()
    return (prob >= 0.5).astype(int)


# ── orchestration ─────────────────────────────────────────────────────

def run(base_dir='', out_log=HARMONIC_LOG, random_state=RANDOM_STATE, register=True,
        swing_source='fractal'):
    """Run the LogReg + MLP pair for one swing basis (`swing_source`:
    'fractal' -> H1.1/H1.2, 'zigzag' -> H1.3/H1.4). Every other step of the
    pipeline (event filter threshold, triple-barrier labeling, 8 features,
    chronological split, class_weight='balanced', random_state) is IDENTICAL
    between the two — only `build_event_dataset`'s swing-point source
    differs — so any difference in outcome between a fractal run and a
    zigzag run is attributable to the swing basis alone.

    The family's Bonferroni bar is computed DYNAMICALLY from the current
    hypothesis log (same convention as src.ablation.run /
    src.volatility.run_candidate_feature_tests): family_size = however many
    DISTINCT hypothesis names are already registered, unioned with this run's
    two names. A prior run's ALREADY-LOGGED alpha is never retroactively
    rewritten (matches every other family log in this project) — only a NEW
    run is judged at the newly-tightened bar.
    """
    names = HYPOTHESIS_NAMES[swing_source]
    h1_name, h2_name = names['h1'], names['h2']
    label1, label2 = names['label1'], names['label2']

    out_path = _p(base_dir, out_log)
    existing = set(pd.read_csv(out_path)['hypothesis']) if os.path.exists(out_path) else set()
    family_size = len(existing | {h1_name, h2_name})
    alpha = FAMILY_ALPHA / family_size

    built = build_event_dataset(base_dir=base_dir, random_state=random_state,
                                swing_source=swing_source)
    dataset = built['dataset']
    n_raw, n_filtered = len(built['events_raw']), len(built['events_filtered'])
    n_labeled = len(dataset)

    print('=' * 78)
    print(f'H1 HARMONIC-PATTERN EVENT-CONDITIONAL MODEL — swing_source={swing_source!r} '
          f'({label1}/{label2})')
    print(f'  raw XABCD events detected: {n_raw:,}')
    print(f'  filtered (best_fit_score >= {MIN_BEST_FIT_SCORE}): {n_filtered:,}')
    print(f'  excluded (insufficient forward history): {built["excluded_insufficient_history"]:,}')
    print(f'  FINAL labeled event dataset: {n_labeled:,}')
    print(f'  baseline (b) non-event random sample: n={built["baseline_b"]["n_sample"]:,}  '
          f'label==1 rate={built["baseline_b"]["label_1_rate"]:.4f}  (descriptive context only)')
    print(f'  hypotheses already registered: {len(existing)}  ->  family size {family_size}  '
          f'->  BONFERRONI BAR alpha = {FAMILY_ALPHA}/{family_size} = {alpha:.4g}')
    print('=' * 78)

    if n_labeled < 20:
        print('\n  TOO FEW LABELED EVENTS for a meaningful split/model — aborting run '
              '(nothing logged). Re-examine the event filter, not a tuning knob for THIS run.')
        return None

    train_end, val_end = _chronological_split(n_labeled)
    train, val, test = dataset.iloc[:train_end], dataset.iloc[train_end:val_end], dataset.iloc[val_end:]
    print(f'\n  chronological split: train[0:{train_end}]={len(train)}  '
          f'val[{train_end}:{val_end}]={len(val)}  test[{val_end}:{n_labeled}]={len(test)} RESERVED')

    scaler = StandardScaler().fit(train[MODEL_FEATURE_COLUMNS])
    X_train = scaler.transform(train[MODEL_FEATURE_COLUMNS]).astype('float32')
    X_val = scaler.transform(val[MODEL_FEATURE_COLUMNS]).astype('float32')
    y_train = train['label'].to_numpy()
    y_val = val['label'].to_numpy()

    maj_class = int(round(y_train.mean()))
    pred_majority_val = np.full(len(y_val), maj_class)

    # ---- H1.x: Logistic Regression vs train-majority baseline -------------
    print(f'\n--- {label1}: LOGISTIC REGRESSION (linear baseline, swing_source={swing_source!r}) ---')
    clf = train_h1_1_logistic(X_train, y_train, random_state=random_state)
    pred_h11_val = clf.predict(X_val)
    r11 = bootstrap_delta_and_mcnemar(y_val, pred_majority_val, pred_h11_val,
                                      alpha=alpha, random_state=random_state)
    print(f'  train-majority baseline acc = {r11["acc_reference"]:.4f}  |  '
          f'{label1} val acc = {r11["acc_challenger"]:.4f}  |  delta = {r11["delta_acc"]:+.4f}')
    print(f'  {100 * (1 - alpha):.1f}% CI[{r11["ci_low"]:+.4f}, {r11["ci_high"]:+.4f}]  '
          f'McNemar b={r11["mcnemar_b"]} c={r11["mcnemar_c"]} p={r11["mcnemar_p"]:.4f}  '
          f'alpha={alpha:.4g}')
    verdict_h11 = (
        f'KEEP — {label1} beats the train-majority baseline at the pre-registered bar'
        if r11['cleared'] else
        f'DROP — {label1} indistinguishable from the train-majority baseline at the pre-registered bar'
    )
    print(f'  VERDICT ({label1}): {verdict_h11}')

    # ---- H1.y: MLP — PRIMARY vs H1.x, CORROBORATING vs majority -----------
    print(f'\n--- {label2}: MLP (non-linear interactions) ---')
    mlp = train_h1_2_mlp(X_train, y_train, X_val, y_val, random_state=random_state)
    pred_h12_val = predict_h1_2_mlp(mlp, X_val)

    r12_primary = bootstrap_delta_and_mcnemar(y_val, pred_h11_val, pred_h12_val,
                                              alpha=alpha, random_state=random_state)
    r12_corroborating = bootstrap_delta_and_mcnemar(y_val, pred_majority_val, pred_h12_val,
                                                    alpha=alpha, random_state=random_state)
    print(f'  PRIMARY (MLP vs {label1}, same val rows): {label1} acc = {r12_primary["acc_reference"]:.4f}  |  '
          f'MLP acc = {r12_primary["acc_challenger"]:.4f}  |  delta = {r12_primary["delta_acc"]:+.4f}')
    print(f'    {100 * (1 - alpha):.1f}% CI[{r12_primary["ci_low"]:+.4f}, '
          f'{r12_primary["ci_high"]:+.4f}]  McNemar b={r12_primary["mcnemar_b"]} '
          f'c={r12_primary["mcnemar_c"]} p={r12_primary["mcnemar_p"]:.4f}')
    print(f'  CORROBORATING ONLY (MLP vs train-majority baseline, context — not decision-bearing): '
          f'delta = {r12_corroborating["delta_acc"]:+.4f}  '
          f'CI[{r12_corroborating["ci_low"]:+.4f}, {r12_corroborating["ci_high"]:+.4f}]')

    if r12_primary['cleared']:
        verdict_h12 = (f'KEEP — MLP beats {label1} at the pre-registered bar on the PRIMARY '
                       'comparison (the corroborating vs-majority result is consistent context)')
    elif r12_corroborating['cleared']:
        verdict_h12 = (f'DROP — MLP beats the majority baseline (corroborating context) but does '
                       f'NOT beat {label1} at the pre-registered bar on the PRIMARY comparison: the '
                       'extra non-linear capacity found nothing the linear model had not already '
                       'found (anti-cherry-pick rule)')
    else:
        verdict_h12 = (f'DROP — MLP beats neither {label1} (primary) nor the train-majority baseline '
                       '(corroborating) at the pre-registered bar')
    print(f'  VERDICT ({label2}): {verdict_h12}')

    print(f'\n  (power caveat: {len(val)} validation events — a small-n family; treat any KEEP as '
          f'preliminary and any DROP as correspondingly weak evidence of absence)')

    date = pd.Timestamp.utcnow().date().isoformat()
    swing_note = (f'swing_source={swing_source!r} '
                  f'({"Williams fractals, fixed CONFIRMATION_LAG=2" if swing_source == "fractal" else "ATR(14)*1.5 causal ZigZag pivots, variable reveal-bar lag"}). ')
    row_h11 = {
        'date': date, 'hypothesis': h1_name,
        'arbiter': 'event_validation[70:85]',
        'n_events_raw': n_raw, 'n_events_filtered': n_filtered, 'n_events_labeled': n_labeled,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'acc_challenger': r11['acc_challenger'], 'acc_reference': r11['acc_reference'],
        'delta_acc': r11['delta_acc'], 'delta_acc_ci_low': r11['ci_low'],
        'delta_acc_ci_high': r11['ci_high'], 'mcnemar_b': r11['mcnemar_b'],
        'mcnemar_c': r11['mcnemar_c'], 'mcnemar_p': r11['mcnemar_p'],
        'alpha': round(alpha, 4), 'cleared_bar': r11['cleared'], 'verdict': verdict_h11,
        'notes': (f'{swing_note}H1 XABCD triple-barrier events, best_fit_score>={MIN_BEST_FIT_SCORE}, '
                  f'EWMA(span={EWMA_SPAN})*sqrt({HORIZON_BARS}) horizon_vol, target={TARGET_MULT}x/'
                  f'stop={STOP_MULT}x, cost threshold={SPREAD_PIPS_DEFAULT}pips='
                  f'{SPREAD_PIPS_DEFAULT * PIP_SIZE}. LogisticRegression(class_weight=balanced). '
                  f'baseline(b) non-event random-sample label==1 rate='
                  f'{built["baseline_b"]["label_1_rate"]:.4f} (n={built["baseline_b"]["n_sample"]}, '
                  f'descriptive context only, not decision-bearing).'),
    }
    row_h12 = {
        'date': date, 'hypothesis': h2_name,
        'arbiter': 'event_validation[70:85]',
        'n_events_raw': n_raw, 'n_events_filtered': n_filtered, 'n_events_labeled': n_labeled,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'acc_challenger': r12_primary['acc_challenger'], 'acc_reference': r12_primary['acc_reference'],
        'delta_acc': r12_primary['delta_acc'], 'delta_acc_ci_low': r12_primary['ci_low'],
        'delta_acc_ci_high': r12_primary['ci_high'], 'mcnemar_b': r12_primary['mcnemar_b'],
        'mcnemar_c': r12_primary['mcnemar_c'], 'mcnemar_p': r12_primary['mcnemar_p'],
        'alpha': round(alpha, 4), 'cleared_bar': r12_primary['cleared'], 'verdict': verdict_h12,
        'notes': (f'{swing_note}PRIMARY reference = {label1} predictions on IDENTICAL val rows (not '
                  f'majority baseline). CORROBORATING (context only) MLP-vs-majority: '
                  f'delta_acc={r12_corroborating["delta_acc"]:+.4f} '
                  f'CI[{r12_corroborating["ci_low"]:+.4f}, {r12_corroborating["ci_high"]:+.4f}] '
                  f'McNemar p={r12_corroborating["mcnemar_p"]:.4f}, cleared='
                  f'{r12_corroborating["cleared"]}. MLP (raw PyTorch, not Keras): '
                  f'Linear(16,L2=1e-3 via weight_decay)-ReLU-Dropout(0.3)-Linear(8,L2=1e-3)-ReLU-'
                  f'Dropout(0.3)-Linear(1)-Sigmoid, Adam lr=0.001, epochs<=100, ES patience=10 on '
                  f'val_loss (explicit model.train()/eval()+no_grad() toggling), batch_size=32, '
                  f'per-sample balanced class weights via BCELoss(weight=...) each batch.'),
    }

    if register:
        _upsert_log(row_h11, out_path)
        _upsert_log(row_h12, out_path)
        print(f'\nLogged both hypotheses: {out_path}')

    return {label1.lower().replace('.', '_'): row_h11,
            label2.lower().replace('.', '_'): row_h12, 'built': built}


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['zigzag']:
        run(swing_source='zigzag')
    else:
        run(swing_source='fractal')
