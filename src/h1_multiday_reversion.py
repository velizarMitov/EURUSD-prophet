"""
MULTI-DAY MEAN REVERSION ON H1 — a NEW hypothesis family (research-only).

MOTIVATION AND ITS CONTAMINATION
--------------------------------
A variance-ratio profile computed on the EURUSD H1 series showed VR significantly
BELOW 1 at every horizon above two hours (VR 0.925 at 24 bars, z = -3.64; 0.886
at 120 bars, z = -2.39), deepening with horizon -- the signature of MEAN
REVERSION. It contrasts with the daily 1999-2026 series, where VR sits at
0.95-1.04 with |z| < 1.6 (nothing).

DISCLOSURE, stated before anything else: that profile was computed on the FULL
H1 series, INCLUDING the reserved test block. The horizon choice below is
therefore informed by data that should not have been looked at.

STEP 0 REMEDIES THIS AND IS A HARD GATE. The variance-ratio profile is
RE-DERIVED on the train slice [0:70%] ALONE. The program is permitted to
continue only if the train-only profile reproduces the qualitative pattern --
VR below 1 at q = 24 with the heteroskedasticity-robust z below -1.96. If it
does not, the motivating structure lives in the held-out periods, the entire
program is CANCELLED and NO hypothesis is registered (MotivationGateError).

PRIOR ATTEMPTS ON THIS FIELD (so a reader can judge the novelty claim)
----------------------------------------------------------------------
This is not a fresh field. H1 EURUSD direction has already been examined at:
  * 1 bar        -- h1_direction family, 5 hypotheses (H_dir.1 KEEP, rest DROP)
  * 120 bars     -- harmonic/triple-barrier family, 6 hypotheses (all DROP)
  * 4/8/12/96/192 bars -- divergence family, 6 hypotheses (all DROP)
  = 17 hypotheses. (A further 2, the pooled_h1 family, tested pooled-vs-single
  training on the SAME 120-bar triple-barrier target; both DROP. Counting those
  the field has absorbed 19 pre-registered H1 attempts.)

The justification for a NEW family rather than an extension of any of those is
that the horizon here is selected by an INDEPENDENT STATISTIC (the Lo-MacKinlay
variance ratio, re-derived on train data in STEP 0) and by COST GEOMETRY, rather
than by iterating horizons until something clears. That is the argument; the
prior-attempt count is stated above so the reader can weigh it themselves.

PRE-REGISTERED DESIGN (fixed before any result was looked at)
-------------------------------------------------------------
* HORIZON, primary and decision-bearing -- 24 H1 bars (1 trading day). The
  selection criterion, stated independently of any result: 24 bars is the
  SHORTEST horizon at which the dealing spread falls below 5% of the typical
  absolute move (1.5 pips against a ~35-pip mean move = 4.3%; at 8 bars 7.8%,
  at 4 bars 11.4%, at 1 bar 23.3%). Chosen on cost geometry, NOT on which
  horizon had the most negative z.
* DECLARED BAND, descriptive only, NO alpha, NO verdict -- 48, 72, 96 bars.
  BINDING RULE: if the primary DROPs but a band member would have cleared, that
  is NOT a KEEP -- it is HORIZON FRAGILITY and is reported as such.
* Observations use EVERY bar (overlapping windows) for FITTING, because more
  rows help the fit even when correlated. All STATISTICS use the moving-block
  bootstrap and every power figure is computed on n_independent = n_rows /
  horizon, never n_rows. Both numbers are stated everywhere.
* FEATURES -- the EXACT 15-column pair-agnostic set already built and
  no-look-ahead tested in src/pooled_h1_model.py, imported UNCHANGED, plus ONE
  pre-registered addition declared here with its justification: the existing lag
  set tops out at 24 bars (1 day) while the target now spans 24-96 bars, so a
  feature set that cannot see further back than the target reaches forward is
  structurally mismatched. Exactly three columns are added --
  logret_48, logret_72, logret_96 (cumulative log returns over 2, 3, 4 days) --
  for BOTH the primary and the band. Nothing else. Feature importance is
  reported so it is visible whether these three carry the model.
* SPLIT -- chronological on the H1 series: train [0:70%], validation [70:85%]
  (THE ARBITER), test [85:100%] RESERVED and never indexed. PURGE the last
  `horizon` training rows (their labels need validation bars; = 24 at the
  primary, exactly as specified) and EMBARGO the first 96 validation bars (the
  longest band horizon; also covers the 24-bar lookback in the feature set).
  Validation rows whose own label window would reach at or past the validation
  end are dropped too -- reading close[t+h] there would index the reserved test
  block. All three counts are reported.
* FAMILY -- NEW and independent: results/h1_multiday_hypothesis_log.csv,
  size 2, Bonferroni bar alpha = 0.05/2 = 0.025. It does not touch or tighten
  h1_direction_*, harmonic_pattern_*, divergence_*, pooled_h1_*,
  feature_hypothesis_log, volatility_*, cot_*, fractal_*, ti_lstm_* or
  macro_panel_*.

  H_md.1 -- THE TRIVIAL MEAN-REVERSION RULE vs the train-majority baseline.
      Rule, fixed now, no parameters to tune: predict DOWN if the trailing
      24-bar log return is positive, predict UP if it is negative (a zero
      trailing return predicts the train-majority class). One line, no fitting,
      no features. It asks the first-order question: does the mean reversion the
      variance ratio detects translate into directional predictability at all?
  H_md.2 -- GBM vs THE TRIVIAL RULE (primary reference), with GBM vs
      train-majority as CORROBORATING CONTEXT ONLY. A model that beats a coin
      flip but not a one-line rule has demonstrated nothing -- standing project
      convention.

  NO NEURAL NETWORK. At ~2,900 independent observations it is excluded by
  arithmetic, as this project has now shown four separate times.

MANDATORY CONTROLS (reported BEFORE the hypotheses, no alpha consumed)
----------------------------------------------------------------------
 1. Shuffled-label control on H_md.2's GBM: refit on permuted training labels,
    scored on the real validation rows. It must land near the majority rate.
 2. THE PERSISTENCE CHECK that kills the old story: the EXISTING committed
    H_dir.1 model, unchanged (next-bar target), scored against the 24-BAR
    forward direction instead of the 1-bar one. If the +2.30pp next-bar edge
    were a persistent drift it would grow as sqrt(24) to roughly +11pp, i.e.
    ~61% accuracy. Whatever it actually is, is reported. Descriptive: it settles
    whether the H1 microstructure edge COMPOUNDS.

NO P&L. NO ledger. NO equity curve. NO position sizing. NO cost subtracted from
any result. The BREAKEVEN ACCURACY reported in the power statement is arithmetic
on two already-computed numbers -- (1 + spread/mean_abs_move)/2 -- and exists
only so the reader can see whether a DETECTABLE edge would also be a MEANINGFUL
one. It is a descriptive anchor, never a P&L calculation and never a verdict.

HARD BOUNDARY -- research only, production untouched
----------------------------------------------------
Writes ONLY results/h1_multiday_hypothesis_log.csv. Never modifies models/,
_train_pipeline.py, src/inference.py, src/features.py, src/paper_trading.py,
config.json, results/eurusd_h1.csv, src/pooled_h1_model.py,
src/h1_direction_model.py, src/pooled_h1_data.py, or ANY existing
results/*hypothesis_log.csv. A unit test sha256-hashes the whole protected set
before/after a full run(). Reads the ALREADY-CACHED
results/pooled_h1/EURUSD_h1.csv and never refetches from MT5.
"""

import os

import numpy as np
import pandas as pd

# Features, device resolution and the GBM/arbiter machinery are IMPORTED
# UNCHANGED from the two committed H1 programs. This module restates none of
# them -- it changes the TARGET HORIZON and adds exactly three declared columns.
from src.pooled_h1_model import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS, compute_pooled_features, resolve_device,
)
from src.h1_direction_model import (
    load_eurusd_h1, longest_identical_close_run, train_majority_class,
    train_gbm, predict_gbm_proba, seed_everything, arbiter,
)
from src.h1_horizon_feasibility import uniqueness_from_spans
from src.pooled_h1_data import POOLED_DIR, PIP_SIZE

# ── Pre-registered constants (frozen; do not tune after seeing results) ──
INSTRUMENT = 'EURUSD'

PRIMARY_HORIZON = 24                    # 1 trading day -- decision-bearing
BAND_HORIZONS = (48, 72, 96)            # 2, 3, 4 days -- DESCRIPTIVE ONLY
ALL_HORIZONS = (PRIMARY_HORIZON,) + BAND_HORIZONS

TRAIN_FRAC = 0.70
VAL_FRAC = 0.85                         # validation END (test = [0.85:1.0], RESERVED)
EMBARGO_BARS = 96                       # longest band horizon; covers the 24-bar lookback
BLOCK_LEN = 96                          # moving-block bootstrap block >= longest horizon
N_BOOT = 2000
FAMILY_SIZE = 2
FAMILY_ALPHA = 0.05 / FAMILY_SIZE       # 0.025
RANDOM_SEED = 42

# STEP 0 -- the motivation gate.
VR_LAGS = (2, 4, 8, 24, 48, 72, 96, 120)
GATE_Q = 24                             # the decision-bearing horizon
GATE_MAX_Z_ROBUST = -1.96               # robust z must be BELOW this
GATE_MAX_VR = 1.0                       # and VR below 1

# STEP 4 -- power.
MIN_INDEPENDENT = 150                   # below this the primary is UNDERPOWERED -> STOP
SPREAD_PIPS = 1.5                       # dealing spread, for the breakeven ANCHOR only

# STEP 2 -- the one pre-registered feature addition.
ADDED_LAGS = (48, 72, 96)
ADDED_FEATURE_COLUMNS = [f'logret_{k}' for k in ADDED_LAGS]
FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS) + ADDED_FEATURE_COLUMNS

# The trailing window of the trivial rule. FIXED at 24 bars for the primary AND
# for every band member -- the rule has no parameters to tune, and matching it
# to each band horizon would be a second variant, i.e. extra researcher freedom.
TRIVIAL_RULE_LOOKBACK = 24
TRIVIAL_RULE_COLUMN = f'logret_{TRIVIAL_RULE_LOOKBACK}'

# H_dir.1's logged next-bar edge, for the persistence check's stated arithmetic.
H_DIR1_NEXT_BAR_EDGE_PP = 2.30

HYPOTHESIS_LOG = 'results/h1_multiday_hypothesis_log.csv'
ARBITER_LABEL = 'h1_multiday_validation[70:85]_block_bootstrap'
NY_HISTORY_CSV = os.path.join(POOLED_DIR, 'EURUSD_h1_newyork.csv')

LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter', 'horizon_bars', 'n_rows',
    'n_independent', 'mean_label_uniqueness', 'min_detectable_edge_pp',
    'breakeven_accuracy_pct', 'train_vr_24', 'train_vr_24_z_robust',
    'acc_challenger', 'acc_reference', 'auc_challenger', 'delta_acc',
    'delta_acc_ci_low', 'delta_acc_ci_high', 'mcnemar_b', 'mcnemar_c',
    'mcnemar_p', 'block_len', 'shuffled_label_control_acc',
    'h_dir1_scored_at_24bars_acc', 'alpha', 'cleared_bar', 'verdict',
    'device_used', 'notes',
]

# Recorded verbatim in EVERY log row (STEP 7 requires all three).
CONTAMINATION_NOTE = (
    "CONTAMINATION DISCLOSURE + STEP 0 REMEDY: the motivating variance-ratio profile "
    "(VR 0.925 at q=24, z=-3.64; 0.886 at q=120, z=-2.39) was computed by the advisor "
    "on the FULL H1 series INCLUDING the reserved test block, so the horizon choice was "
    "informed by data that should not have been looked at. REMEDY (hard gate, run "
    "before anything else): the whole Lo-MacKinlay profile was RE-DERIVED on the train "
    "slice [0:70%] ALONE, and the program was permitted to continue only because the "
    "train-only profile reproduced the pattern (VR<1 at q=24 with robust z<-1.96). The "
    "horizon choice is therefore justified from TRAIN DATA ALONE and the test block "
    "remains clean and unindexed for this family. The validation-slice profile is "
    "reported for comparison; the test block was NEVER profiled."
)
PRIOR_ATTEMPTS_NOTE = (
    "PRIOR-ATTEMPT COUNT (this is not a fresh field): H1 EURUSD direction has already "
    "been examined at 1 bar (h1_direction, 5 hypotheses), at 120 bars (harmonic/triple-"
    "barrier, 6), and at 4/8/12/96/192 bars (divergence, 6) = 17 pre-registered "
    "hypotheses; a further 2 (pooled_h1) tested pooled-vs-single training on the same "
    "120-bar triple-barrier target, making 19 H1 attempts in total. The justification "
    "for a NEW family rather than an extension is that the horizon is selected by an "
    "INDEPENDENT statistic (the variance ratio, re-derived on train data) and by cost "
    "geometry, rather than by iterating horizons until something clears -- the reader "
    "has the attempt count and can judge that claim for themselves."
)
ADDED_FEATURES_NOTE = (
    "THREE ADDED LAG FEATURES (logret_48, logret_72, logret_96 -- cumulative log returns "
    "over 2, 3 and 4 days), declared at registration with this justification: the "
    "imported 15-column set tops out at a 24-bar lag while the target now spans 24-96 "
    "bars, so a feature set that cannot see further back than the target reaches forward "
    "is structurally mismatched. Exactly three columns added, nothing removed or "
    "modified, applied to BOTH the primary and the band; feature importance is reported "
    "so it is visible whether these three carry the model."
)
STANDING_NOTES = f"{CONTAMINATION_NOTE} {PRIOR_ATTEMPTS_NOTE} {ADDED_FEATURES_NOTE}"


class MotivationGateError(RuntimeError):
    """Raised when the STEP 0 train-only variance ratio fails to reproduce the
    motivating pattern. The full-series profile would then have been driven by
    the held-out periods, so the entire program is CANCELLED and no hypothesis
    is registered."""


class UnderpoweredError(RuntimeError):
    """Raised when the primary horizon yields fewer than MIN_INDEPENDENT
    independent observations. A test that cannot resolve a meaningful edge is
    not run -- it is reported as underpowered."""


class TestBlockTouchedError(RuntimeError):
    """Raised if any row position at or beyond the validation end boundary is
    read -- including via a label's forward window. The test block [85:100%] is
    RESERVED for this family and must never be indexed."""


# ═════════════════ STEP 0 — Lo-MacKinlay variance ratio (the gate) ═════════════

def variance_ratio(log_prices, q: int) -> dict:
    """
    Lo-MacKinlay (1988) OVERLAPPING variance-ratio statistic at aggregation q,
    with BOTH standard errors.

        mu       = (p_n - p_0) / n
        sigma_a2 = sum_k (p_k - p_{k-1} - mu)^2 / (n - 1)
        sigma_c2 = sum_{k=q}^{n} (p_k - p_{k-q} - q*mu)^2 / m,
                   m = q (n - q + 1) (1 - q/n)
        VR(q)    = sigma_c2 / sigma_a2

    HOMOSKEDASTIC z:  sqrt(n) (VR-1) / sqrt(phi),   phi = 2(2q-1)(q-1) / (3q)
    ROBUST z (heteroskedasticity-consistent):
        delta_j = sum_{k=j+1}^{n} (r_k-mu)^2 (r_{k-j}-mu)^2 / [sum_k (r_k-mu)^2]^2
        phi*    = sum_{j=1}^{q-1} [2(q-j)/q]^2 delta_j
        z*      = (VR-1) / sqrt(phi*)

    NOTE the asymmetry in the two z's, which is easy to get wrong: delta_j is a
    ratio of a SINGLE sum to a SQUARED sum, so it is already O(1/n) and phi*
    carries the sample size internally -- the robust statistic therefore takes NO
    additional sqrt(n) factor, while the homoskedastic one does. (Sanity check:
    under i.i.d. returns delta_j -> 1/n and sum_j [2(q-j)/q]^2 = phi exactly, so
    phi* -> phi/n and z* -> sqrt(n)(VR-1)/sqrt(phi) = the homoskedastic z, as it
    must.) Inserting the sqrt(n) twice inflates |z| by a factor of ~220 at this
    sample size, which is how the error announces itself.

    VR < 1 => mean reversion; VR > 1 => trending; VR = 1 => random walk.
    Under the null both z statistics are asymptotically N(0,1).
    """
    p = np.asarray(log_prices, dtype=float)
    p = p[np.isfinite(p)]
    n = len(p) - 1                                    # number of 1-period returns
    if q < 2 or n < 2 * q:
        return {'q': int(q), 'n_returns': int(max(n, 0)), 'vr': float('nan'),
                'z_homoskedastic': float('nan'), 'z_robust': float('nan'),
                'phi': float('nan'), 'phi_robust': float('nan')}

    r = np.diff(p)
    mu = (p[-1] - p[0]) / n

    sigma_a2 = float(np.sum((r - mu) ** 2) / (n - 1))

    q_diff = p[q:] - p[:-q] - q * mu                  # k = q .. n
    m = q * (n - q + 1) * (1.0 - q / n)
    sigma_c2 = float(np.sum(q_diff ** 2) / m)

    vr = sigma_c2 / sigma_a2 if sigma_a2 > 0 else float('nan')

    # homoskedastic
    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q)
    z_homo = np.sqrt(n) * (vr - 1.0) / np.sqrt(phi)

    # heteroskedasticity-robust
    e2 = (r - mu) ** 2
    denom = float(np.sum(e2)) ** 2
    phi_star = 0.0
    for j in range(1, q):
        delta_j = float(np.sum(e2[j:] * e2[:-j])) / denom if denom > 0 else np.nan
        phi_star += (2.0 * (q - j) / q) ** 2 * delta_j
    # NO sqrt(n) here -- phi_star already carries it (see the docstring).
    z_robust = (vr - 1.0) / np.sqrt(phi_star) if phi_star > 0 else float('nan')

    return {'q': int(q), 'n_returns': int(n), 'vr': float(vr),
            'z_homoskedastic': float(z_homo), 'z_robust': float(z_robust),
            'phi': float(phi), 'phi_robust': float(phi_star)}


def variance_ratio_profile(close, lags=VR_LAGS) -> pd.DataFrame:
    """VR + both z statistics at every q in `lags`, from a CLOSE price series
    (logged here, so callers pass prices)."""
    log_p = np.log(np.asarray(close, dtype=float))
    return pd.DataFrame([variance_ratio(log_p, q) for q in lags])


def step0_gate(train_profile: pd.DataFrame, q: int = GATE_Q) -> dict:
    """
    THE HARD GATE. The TRAIN-ONLY profile must reproduce the qualitative pattern:
    VR below 1 at q = 24 with the ROBUST z below -1.96. Returns the gate record
    when it passes; raises MotivationGateError when it does not -- in which case
    the motivating structure does not exist in the train slice, the full-series
    profile was driven by the held-out periods, the program is CANCELLED and NO
    hypothesis is registered.
    """
    row = train_profile[train_profile['q'] == q]
    if row.empty:
        raise MotivationGateError(f'q = {q} missing from the train-only profile.')
    vr = float(row['vr'].iloc[0])
    z = float(row['z_robust'].iloc[0])
    passed = bool(np.isfinite(vr) and np.isfinite(z)
                  and vr < GATE_MAX_VR and z < GATE_MAX_Z_ROBUST)
    record = {'q': q, 'train_vr': vr, 'train_z_robust': z,
              'train_z_homoskedastic': float(row['z_homoskedastic'].iloc[0]),
              'passed': passed}
    if not passed:
        raise MotivationGateError(
            f"STEP 0 GATE FAILED: train-only VR({q}) = {vr:.4f} with robust "
            f"z = {z:.4f}; the bar is VR < {GATE_MAX_VR} AND robust z < "
            f"{GATE_MAX_Z_ROBUST}. The motivating mean-reversion structure does NOT "
            "exist in the train slice, so the advisor's full-series profile was "
            "driven by the HELD-OUT periods. The entire program is CANCELLED and no "
            "hypothesis is registered."
        )
    return record


# ═════════════════ STEP 2 — features (15 imported + 3 declared) ════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The EXACT 15-column pair-agnostic set from src/pooled_h1_model.py, imported
    and called UNCHANGED, plus the three pre-registered multi-day lags. Every
    added column is a difference of log closes at t and t-k, so like every
    imported column it is computable from bars <= t ONLY (no look-ahead).
    """
    feats = compute_pooled_features(df).copy()
    log_close = np.log(df['close'])
    for k in ADDED_LAGS:
        feats[f'logret_{k}'] = log_close - log_close.shift(k)
    return feats[FEATURE_COLUMNS]


# ═════════════════ STEP 1/3 — target, grid and split ═══════════════════════════

def build_multiday_context(df: pd.DataFrame, horizons=ALL_HORIZONS):
    """
    Build ONE feature-valid bar grid carrying a forward log return and a
    direction label for EVERY horizon, so the primary and the band share an
    IDENTICAL split boundary and an identical notion of "the test block".

    Target:  y_h[t] = 1 if log(close[t+h]/close[t]) > 0
                      0 if < 0
             EXACTLY zero -> NaN (dropped; a zero move has no direction to
             predict -- the same convention the h1_direction family uses).
    The final h bars have no close[t+h] and are NaN by construction.

    Returns (context, counts). `context` carries the feature columns plus
    fwd_logret_<h>, fwd_pips_<h> and label_<h> per horizon.
    """
    feats = build_features(df)
    log_close = np.log(df['close'])
    close = df['close']

    ctx = feats.dropna(subset=FEATURE_COLUMNS).copy()
    lc = log_close.reindex(ctx.index)
    c0 = close.reindex(ctx.index)

    for h in horizons:
        # shift(-h) on the FULL series first, then align: the forward bar is the
        # h-th RAW bar ahead, never the h-th surviving feature row ahead.
        fwd = (log_close.shift(-h) - log_close).reindex(ctx.index)
        pips = ((close.shift(-h) - close) / PIP_SIZE).reindex(ctx.index)
        ctx[f'fwd_logret_{h}'] = fwd
        ctx[f'fwd_pips_{h}'] = pips
        v = fwd.to_numpy()
        lab = np.where(np.isnan(v), np.nan, np.where(v > 0.0, 1.0,
                                                     np.where(v < 0.0, 0.0, np.nan)))
        ctx[f'label_{h}'] = lab

    counts = {
        'n_bars_raw': int(len(df)),
        'n_feature_valid': int(len(ctx)),
        'longest_identical_close_run': longest_identical_close_run(df['close']),
        'n_zero_return_dropped': {
            h: int(np.sum(ctx[f'fwd_logret_{h}'].to_numpy() == 0.0)) for h in horizons},
    }
    # silence the unused-alias lint while keeping the intent explicit
    del lc, c0
    return ctx, counts


def split_bounds(n_rows: int):
    """Positional [0:70%] / [70:85%] / [85:100%] boundaries on the feature-valid
    grid. Identical for the primary and every band member."""
    return int(n_rows * TRAIN_FRAC), int(n_rows * VAL_FRAC)


def split_purge_embargo(context: pd.DataFrame, horizon: int):
    """
    Chronological split with the pre-registered purge and embargo, at one horizon.

      * TRAIN  -- positions [0, train_end - horizon). PURGING the last `horizon`
        training rows, whose labels require validation bars (= 24 rows at the
        primary, exactly as specified).
      * VAL    -- positions [train_end + 96, val_end - horizon). EMBARGO of the
        first 96 validation bars (the longest band horizon, which also covers the
        24-bar lookback in the feature set), and the trailing `horizon` rows are
        dropped because reading their close[t+h] would index the RESERVED test
        block.
      * TEST   -- [val_end, n). Returned as a COUNT ONLY; never indexed.

    Rows whose label is NaN (exact-zero forward move) are excluded from the
    returned index sets. Returns (splits, counts).
    """
    n = len(context)
    train_end, val_end = split_bounds(n)
    lab = context[f'label_{horizon}'].to_numpy()
    labelled = np.isfinite(lab)

    pos = np.arange(n)
    train_mask = labelled & (pos < train_end - horizon)
    val_mask = (labelled & (pos >= train_end + EMBARGO_BARS)
                & (pos < val_end - horizon))

    n_train_pre = int((labelled & (pos < train_end)).sum())
    n_val_pre = int((labelled & (pos >= train_end) & (pos < val_end)).sum())
    n_embargoed = int((labelled & (pos >= train_end)
                       & (pos < train_end + EMBARGO_BARS)).sum())
    n_val_tail = int((labelled & (pos >= max(train_end + EMBARGO_BARS, val_end - horizon))
                      & (pos < val_end)).sum())

    splits = {'train': context.index[train_mask], 'val': context.index[val_mask]}
    counts = {
        'horizon': int(horizon),
        'n_grid': int(n),
        'train_end_pos': int(train_end), 'val_end_pos': int(val_end),
        'train_end_ts': context.index[train_end - 1],
        'val_end_ts': context.index[val_end - 1],
        'n_purged': int(n_train_pre - train_mask.sum()),
        'n_embargoed': n_embargoed,
        'n_val_tail_dropped': n_val_tail,
        'n_train': int(train_mask.sum()), 'n_val': int(val_mask.sum()),
        'n_val_pre': n_val_pre,
        'n_test_reserved': int(n - val_end),
        # the furthest position this horizon ever READS, label window included
        'max_position_read': int(pos[val_mask].max() + horizon) if val_mask.any() else -1,
    }
    return splits, counts


def assert_no_test_block(counts):
    """Hard guard: the furthest position read at this horizon -- INCLUDING each
    label's forward window -- must stay strictly inside the validation block."""
    if counts['max_position_read'] >= counts['val_end_pos']:
        raise TestBlockTouchedError(
            f"horizon {counts['horizon']}: position {counts['max_position_read']} "
            f"reaches at/beyond the validation end {counts['val_end_pos']} -- the "
            "test block [85:100%] is RESERVED and must never be indexed."
        )
    return True


def measure_label_uniqueness(context: pd.DataFrame, index, horizon: int) -> float:
    """
    Lopez de Prado mean label uniqueness, MEASURED (never assumed) with the same
    estimator the rest of the project uses. A label at bar t is determined by the
    return over the interval (t, t+h], i.e. an information span of h bars:
    positions [i, i+h-1] -- the convention under which the h1_direction family's
    1-bar labels score exactly 1.0. Overlapping h-bar windows therefore land
    near 1/h, which is what the power statement uses.
    """
    pos = {ts: i for i, ts in enumerate(context.index)}
    starts = np.array([pos[ts] for ts in index], dtype=np.int64)
    if len(starts) == 0:
        return float('nan')
    ends = np.minimum(starts + horizon - 1, len(context) - 1)
    return uniqueness_from_spans(starts, ends, grid_len=len(context))


# ═════════════════ STEP 4 — the power statement ════════════════════════════════

def power_statement(fwd_pips, n_rows: int, horizon: int, uniqueness: float,
                    alpha: float = FAMILY_ALPHA, spread_pips: float = SPREAD_PIPS):
    """
    Computed and printed BEFORE any accuracy figure.

    n_independent = n_rows / horizon (overlapping windows), NOT n_rows. The
    standard error on an accuracy near chance is sqrt(0.25 / n_independent), so
    the MINIMUM DETECTABLE ACCURACY EDGE at `alpha` (two-sided) is
    z_{1-alpha/2} * SE, reported in percentage points.

    BREAKEVEN ACCURACY, a DESCRIPTIVE ANCHOR and NOT a P&L calculation:
    (1 + spread/mean_abs_move) / 2, arithmetic on two already-computed numbers.
    No equity curve, no ledger, no cost subtracted from any result -- it exists
    only so the reader can see whether a detectable edge would also be a
    meaningful one.
    """
    from scipy.stats import norm
    v = np.asarray(fwd_pips, dtype=float)
    v = v[np.isfinite(v)]
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
    mean_abs = float(np.abs(v).mean()) if len(v) else float('nan')

    n_independent = n_rows / float(horizon)
    se_acc = float(np.sqrt(0.25 / n_independent)) if n_independent > 0 else float('nan')
    z = float(norm.ppf(1.0 - alpha / 2.0))
    mde_pp = 100.0 * z * se_acc
    breakeven = 100.0 * (1.0 + spread_pips / mean_abs) / 2.0 if mean_abs else float('nan')

    return {
        'horizon': int(horizon), 'n_rows': int(n_rows),
        'n_independent': float(n_independent),
        'mean_label_uniqueness': float(uniqueness),
        'std_signed_pips': sd, 'mean_abs_move_pips': mean_abs,
        'se_accuracy_from_independent': se_acc, 'z_alpha': z,
        'min_detectable_edge_pp': mde_pp,
        'breakeven_accuracy_pct': breakeven,
        'spread_share_of_move_pct': (100.0 * spread_pips / mean_abs) if mean_abs else float('nan'),
        'alpha': alpha,
        'underpowered': bool(n_independent < MIN_INDEPENDENT),
    }


# ═════════════════ STEP 5 — the trivial rule and the GBM ═══════════════════════

def trivial_rule(trailing_logret, majority_class: int) -> np.ndarray:
    """
    H_md.1's whole model, one line, no fitting and no parameters:

        trailing 24-bar log return  > 0  ->  predict DOWN (0)
        trailing 24-bar log return  < 0  ->  predict UP   (1)
        trailing 24-bar log return == 0  ->  predict the TRAIN-MAJORITY class

    The zero case is defined here rather than left implicit, and is unit-tested.
    """
    t = np.asarray(trailing_logret, dtype=float)
    out = np.where(t > 0.0, 0, np.where(t < 0.0, 1, int(majority_class)))
    return out.astype(int)


def flat_matrix(context: pd.DataFrame, index, horizon: int, mean, std):
    """Standardized feature matrix + labels for one row set at one horizon."""
    d = context.loc[index]
    X = (d[FEATURE_COLUMNS].to_numpy(dtype=float) - mean) / std
    y = d[f'label_{horizon}'].to_numpy(dtype=float).astype(int)
    return X, y


def fit_standardizer(context: pd.DataFrame, train_index):
    """Per-feature mean/std fit on the TRAIN rows ONLY (never validation, never
    test)."""
    X = context.loc[train_index, FEATURE_COLUMNS].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def shuffled_label_control(Xtr, ytr, Xval, yval, seed: int = RANDOM_SEED) -> float:
    """
    MANDATORY CONTROL 1 (no alpha). Refit the GBM with the TRAINING LABELS
    RANDOMLY PERMUTED -- features untouched, identical hyperparameters -- and
    score it on the REAL validation rows. It must land near the majority rate; a
    materially higher score means the pipeline leaks and every other number in
    this program is void.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = np.asarray(ytr).astype(int).copy()
    rng.shuffle(y_shuffled)
    clf = train_gbm(Xtr, y_shuffled, seed=seed)
    pred = (predict_gbm_proba(clf, Xval) >= 0.5).astype(int)
    return float((pred == np.asarray(yval).astype(int)).mean())


def h_dir1_persistence_check(context: pd.DataFrame, split_counts,
                             horizon: int = PRIMARY_HORIZON,
                             out_dir: str = POOLED_DIR, seed: int = RANDOM_SEED,
                             verbose: bool = True):
    """
    MANDATORY CONTROL 2 (no alpha) -- THE PERSISTENCE CHECK.

    Take the EXISTING committed H_dir.1 model UNCHANGED (rebuilt through
    src.h1_direction_diagnostics.rebuild_h_dir_1, which enforces its own
    reproduction gate) and score its next-bar predictions against the 24-BAR
    forward direction instead of the 1-bar one.

    If H_dir.1's +2.30pp next-bar edge were a persistent drift it would grow as
    sqrt(24) to roughly +11pp, i.e. ~61% accuracy. Whatever it actually is, is
    reported. This settles whether the H1 microstructure edge COMPOUNDS.

    Rows whose 24-bar window would reach at or past THIS program's validation end
    are excluded -- reading close[t+24] there would index the reserved test
    block. The excluded count is reported, never absorbed silently.
    """
    from src import h1_direction_diagnostics as diag

    reb = diag.rebuild_h_dir_1(out_dir=out_dir, seed=seed, verbose=verbose)
    val_end_pos = split_counts['val_end_pos']
    pos = {ts: i for i, ts in enumerate(context.index)}
    lab24 = context[f'label_{horizon}'].to_numpy()

    keep_k, keep_j = [], []
    for k, ts in enumerate(reb['val_idx']):
        j = pos.get(ts)
        if j is None or j + horizon >= val_end_pos or not np.isfinite(lab24[j]):
            continue
        keep_k.append(k)
        keep_j.append(j)

    keep_k = np.asarray(keep_k, dtype=int)
    keep_j = np.asarray(keep_j, dtype=int)
    pred = np.asarray(reb['pred_gbm'])[keep_k]
    y1 = np.asarray(reb['yval'])[keep_k]
    y24 = lab24[keep_j].astype(int)
    maj = np.asarray(reb['pred_major'])[keep_k]

    return {
        'n_h_dir1_val_rows': int(len(reb['val_idx'])),
        'n_scored': int(len(keep_k)),
        'n_dropped_for_test_block': int(len(reb['val_idx']) - len(keep_k)),
        'acc_at_1bar_full_slice': float(reb['acc_gbm']),
        'acc_at_1bar_on_scored_rows': float((pred == y1).mean()),
        'acc_at_1bar_majority': float((maj == y1).mean()),
        'acc_at_24bar': float((pred == y24).mean()),
        'acc_at_24bar_majority': float((maj == y24).mean()),
        'delta_at_24bar': float((pred == y24).mean() - (maj == y24).mean()),
        'delta_at_1bar_on_scored_rows': float((pred == y1).mean() - (maj == y1).mean()),
        'sqrt24_projection_acc': float(0.5 + (H_DIR1_NEXT_BAR_EDGE_PP / 100.0)
                                       * np.sqrt(horizon)),
        'reproduction_drift': float(reb['reproduction_drift']),
    }


# ═════════════════ STEP 6 — session breakdown (descriptive only) ═══════════════

def session_breakdown(index, y, preds: dict, ny_path: str = NY_HISTORY_CSV):
    """
    DESCRIPTIVE ONLY, never a path to KEEP. Accuracy by NY session bucket, using
    the VERIFIED results/pooled_h1/EURUSD_h1_newyork.csv mapping (built and
    gated by src/h1_newyork_time.py -- no clock arithmetic is redone here).

    At a 24-bar horizon the entry hour should matter far less than it did at 1
    bar; an edge appearing ONLY outside the London/NY overlap is the same warning
    sign as before.
    """
    from src.h1_newyork_time import ny_session, SESSION_ORDER

    if not os.path.exists(ny_path):
        return None
    ny = pd.read_csv(ny_path)
    key = pd.DatetimeIndex(pd.to_datetime(ny['server_timestamp'])).tz_localize('UTC')
    hour = pd.Series(ny['ny_hour'].to_numpy(), index=key)

    idx = pd.DatetimeIndex(index)
    h = hour.reindex(idx)
    bucket = np.array([ny_session(int(v)) if np.isfinite(v) else 'unmapped'
                       for v in h.to_numpy()])

    y = np.asarray(y).astype(int)
    rows = []
    for name in list(SESSION_ORDER) + (['unmapped'] if (bucket == 'unmapped').any() else []):
        m = bucket == name
        rec = {'session': name, 'n_val': int(m.sum()),
               'up_rate': float(y[m].mean()) if m.any() else float('nan')}
        for tag, p in preds.items():
            pa = np.asarray(p).astype(int)
            rec[f'acc_{tag}'] = float((pa[m] == y[m]).mean()) if m.any() else float('nan')
        rows.append(rec)
    return pd.DataFrame(rows)


# ═════════════════ feature importance ══════════════════════════════════════════

def feature_importance(gbm, Xval, yval, seed: int = RANDOM_SEED, n_repeats: int = 10):
    """
    Gain-based importance from the fitted booster and permutation importance on
    the validation rows, side by side, plus a group total for the three ADDED
    multi-day lags -- so "do the three added features carry the model?" is
    readable directly rather than inferred.

    Permutation importance is the imported, already-unit-tested implementation
    from src/h1_direction_diagnostics.py (it restores every column and never
    mutates the caller's matrix).
    """
    from src.h1_direction_diagnostics import permutation_importance, gain_importance

    baseline, perm = permutation_importance(gbm, Xval, yval, n_repeats=n_repeats,
                                            seed=seed, feature_names=tuple(FEATURE_COLUMNS))
    gain = gain_importance(gbm, feature_names=tuple(FEATURE_COLUMNS))
    df = gain.merge(perm, on='feature', how='outer')
    df['is_added_multiday_lag'] = df['feature'].isin(ADDED_FEATURE_COLUMNS)
    df['perm_baseline_acc'] = baseline
    total_perm = float(df['perm_importance_mean'].sum())
    df['perm_share'] = (df['perm_importance_mean'] / total_perm
                        if total_perm != 0 else float('nan'))
    return df.sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)


# ═════════════════ one horizon, end to end ═════════════════════════════════════

def evaluate_horizon(context: pd.DataFrame, horizon: int, seed: int = RANDOM_SEED,
                     with_importance: bool = False, with_sessions: bool = False,
                     verbose: bool = True):
    """
    Everything for ONE horizon: split/purge/embargo, power statement, the trivial
    rule, the GBM, the shuffled-label control and both arbiter comparisons.
    Identical code path for the primary and every band member -- the band differs
    only in `horizon`, so no band result can come from a different procedure.
    """
    splits, counts = split_purge_embargo(context, horizon)
    assert_no_test_block(counts)

    train_idx, val_idx = splits['train'], splits['val']
    uniqueness = measure_label_uniqueness(context, val_idx, horizon)
    power = power_statement(context.loc[val_idx, f'fwd_pips_{horizon}'].to_numpy(),
                            n_rows=len(val_idx), horizon=horizon,
                            uniqueness=uniqueness)

    mean, std = fit_standardizer(context, train_idx)
    Xtr, ytr = flat_matrix(context, train_idx, horizon, mean, std)
    Xval, yval = flat_matrix(context, val_idx, horizon, mean, std)

    majority = train_majority_class(ytr)
    pred_major = np.full(len(yval), majority, dtype=int)

    trail_val = context.loc[val_idx, TRIVIAL_RULE_COLUMN].to_numpy()
    pred_trivial = trivial_rule(trail_val, majority)
    # the rule's own decision function, for a descriptive AUC
    score_trivial = -trail_val

    leak_acc = shuffled_label_control(Xtr, ytr, Xval, yval, seed=seed)

    gbm = train_gbm(Xtr, ytr, seed=seed)
    prob_gbm = predict_gbm_proba(gbm, Xval)
    pred_gbm = (prob_gbm >= 0.5).astype(int)

    # H_md.1 -- trivial rule vs train-majority (PRIMARY reference)
    a1 = arbiter(pred_trivial, pred_major, yval, score_challenger=score_trivial,
                 block_len=BLOCK_LEN, n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=seed)
    # H_md.2 -- GBM vs the trivial rule (PRIMARY reference)
    a2 = arbiter(pred_gbm, pred_trivial, yval, score_challenger=prob_gbm,
                 block_len=BLOCK_LEN, n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=seed)
    # CORROBORATING CONTEXT ONLY -- never a second path to KEEP
    a2_corr = arbiter(pred_gbm, pred_major, yval, score_challenger=prob_gbm,
                      block_len=BLOCK_LEN, n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=seed)

    out = {
        'horizon': horizon, 'splits': splits, 'counts': counts,
        'power': power, 'uniqueness': uniqueness,
        'majority_class': majority,
        'acc_majority': float((pred_major == yval).mean()),
        'acc_trivial': float((pred_trivial == yval).mean()),
        'acc_gbm': float((pred_gbm == yval).mean()),
        'shuffled_label_control_acc': leak_acc,
        'train_class_balance_pct1': 100.0 * float((ytr == 1).mean()),
        'val_class_balance_pct1': 100.0 * float((yval == 1).mean()),
        'h_md_1': a1, 'h_md_2': a2, 'h_md_2_corroborating': a2_corr,
        'n_train': len(ytr), 'n_val': len(yval),
    }
    if with_importance:
        out['importance'] = feature_importance(gbm, Xval, yval, seed=seed)
    if with_sessions:
        out['sessions'] = session_breakdown(
            val_idx, yval,
            {'gbm': pred_gbm, 'trivial': pred_trivial, 'majority': pred_major})
    return out


# ═════════════════ STEP 7 — the family log ═════════════════════════════════════

def _upsert_log(row: dict, log_path: str = HYPOTHESIS_LOG):
    """Append/replace one hypothesis row (idempotent on `hypothesis`), keeping
    THIS family's log sorted by `n`. Touches no other family's log."""
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


def _r(x, nd=6):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return round(f, nd) if np.isfinite(f) else f


def _log_row(n, hypothesis, res, acc, gate, leak_acc, persistence, device_used, notes):
    p = res['power']
    return {
        'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
        'hypothesis': hypothesis, 'arbiter': ARBITER_LABEL,
        'horizon_bars': res['horizon'], 'n_rows': p['n_rows'],
        'n_independent': _r(p['n_independent'], 2),
        'mean_label_uniqueness': _r(p['mean_label_uniqueness']),
        'min_detectable_edge_pp': _r(p['min_detectable_edge_pp'], 4),
        'breakeven_accuracy_pct': _r(p['breakeven_accuracy_pct'], 4),
        'train_vr_24': _r(gate['train_vr']), 'train_vr_24_z_robust': _r(gate['train_z_robust']),
        'acc_challenger': _r(acc['acc_challenger']), 'acc_reference': _r(acc['acc_reference']),
        'auc_challenger': _r(acc['auc_challenger']), 'delta_acc': _r(acc['delta_acc']),
        'delta_acc_ci_low': _r(acc['delta_acc_ci_low_block']),
        'delta_acc_ci_high': _r(acc['delta_acc_ci_high_block']),
        'mcnemar_b': acc['mcnemar_b'], 'mcnemar_c': acc['mcnemar_c'],
        'mcnemar_p': _r(acc['mcnemar_p']), 'block_len': acc['block_len'],
        'shuffled_label_control_acc': _r(leak_acc),
        'h_dir1_scored_at_24bars_acc': (_r(persistence['acc_at_24bar'])
                                        if persistence else ''),
        'alpha': acc['alpha'], 'cleared_bar': acc['cleared_bar'],
        'verdict': acc['verdict'], 'device_used': device_used, 'notes': notes,
    }


# ═════════════════ orchestration ═══════════════════════════════════════════════

def run(out_dir: str = POOLED_DIR, log_path: str = HYPOTHESIS_LOG,
        seed: int = RANDOM_SEED, register: bool = True, verbose: bool = True,
        run_band: bool = True, run_persistence: bool = True):
    """
    The whole program in the pre-registered order: STEP 0 gate first (and if it
    fails, nothing else runs and no hypothesis is registered), then the power
    statement, then the two mandatory controls, then H_md.1 and H_md.2, then the
    descriptive band, sessions and feature importance.

    Writes ONLY results/h1_multiday_hypothesis_log.csv. The reserved test block
    [85:100%] is never indexed, label windows included.
    """
    device, dev_info = resolve_device()
    used_seed = seed_everything(seed)
    if verbose:
        print('=' * 78)
        print('MULTI-DAY MEAN REVERSION ON H1 — device & determinism')
        print(f"  torch.cuda.is_available() = {dev_info['cuda_available']}")
        if dev_info['cuda_available']:
            print(f"  torch.cuda.get_device_name(0) = {dev_info['cuda_device_name']}")
        else:
            print(f"  *** CUDA NOT AVAILABLE — fallback: {dev_info['device']} ***")
        print(f"  resolved device = {device}   seed = {used_seed}")
        print('=' * 78)

    raw = load_eurusd_h1(out_dir=out_dir)
    context, counts = build_multiday_context(raw)
    train_end, val_end = split_bounds(len(context))
    train_end_ts = context.index[train_end - 1]
    val_end_ts = context.index[val_end - 1]

    # ── STEP 0 — THE HARD GATE, on TRAIN DATA ALONE ──
    close = raw['close']
    train_close = close[close.index <= train_end_ts]
    val_close = close[(close.index > train_end_ts) & (close.index <= val_end_ts)]
    train_profile = variance_ratio_profile(train_close)
    val_profile = variance_ratio_profile(val_close)

    result = {
        'device_info': dev_info, 'device': str(device), 'seed': used_seed,
        'counts': counts, 'n_grid': len(context),
        'span': (context.index.min().isoformat(), context.index.max().isoformat()),
        'train_end_ts': train_end_ts, 'val_end_ts': val_end_ts,
        'train_end_pos': train_end, 'val_end_pos': val_end,
        'train_vr_profile': train_profile, 'val_vr_profile': val_profile,
    }
    try:
        gate = step0_gate(train_profile)
    except MotivationGateError as exc:
        result['gate'] = {'passed': False, 'message': str(exc)}
        result['cancelled'] = True
        return result
    result['gate'] = gate

    # ── STEP 3/4/5/6 — the primary ──
    primary = evaluate_horizon(context, PRIMARY_HORIZON, seed=seed,
                               with_importance=True, with_sessions=True,
                               verbose=verbose)
    result['primary'] = primary

    if primary['power']['underpowered']:
        raise UnderpoweredError(
            f"primary horizon {PRIMARY_HORIZON}: n_independent = "
            f"{primary['power']['n_independent']:.1f} < {MIN_INDEPENDENT}. "
            "UNDERPOWERED -- STOP."
        )

    # ── MANDATORY CONTROL 2 — H_dir.1 scored at 24 bars ──
    persistence = None
    if run_persistence:
        persistence = h_dir1_persistence_check(
            context, primary['counts'], horizon=PRIMARY_HORIZON,
            out_dir=out_dir, seed=seed, verbose=False)
    result['persistence'] = persistence

    # ── the DECLARED BAND — descriptive only, no alpha, no verdict ──
    band = {}
    if run_band:
        for h in BAND_HORIZONS:
            band[h] = evaluate_horizon(context, h, seed=seed, verbose=verbose)
    result['band'] = band

    # ── STEP 7 — register the two hypotheses ──
    dev = dev_info['device']
    leak = primary['shuffled_label_control_acc']
    common = (f"n_rows={primary['power']['n_rows']} overlapping windows but "
              f"n_independent={primary['power']['n_independent']:.1f} "
              f"(= n_rows/{PRIMARY_HORIZON}); measured mean label uniqueness "
              f"{primary['uniqueness']:.5f} (~1/{PRIMARY_HORIZON}); every statistic uses "
              f"the moving-block bootstrap at block_len={BLOCK_LEN} bars (>= the longest "
              f"band horizon) and every power figure uses n_independent. HORIZON "
              f"SELECTION, fixed before any result: 24 bars is the SHORTEST horizon at "
              f"which the 1.5-pip spread falls below 5% of the typical absolute move "
              f"({primary['power']['spread_share_of_move_pct']:.1f}% here) -- cost "
              f"geometry, not the most negative z. DECLARED BAND 48/72/96 is DESCRIPTIVE "
              f"ONLY: a band member clearing while the primary DROPs is HORIZON "
              f"FRAGILITY, never a KEEP. NO neural network (excluded by arithmetic at "
              f"~{primary['power']['n_independent']:.0f} independent observations). "
              f"NO P&L, no ledger, no equity curve; the breakeven accuracy "
              f"{primary['power']['breakeven_accuracy_pct']:.2f}% is a descriptive "
              f"anchor from mean|move| and the spread. {STANDING_NOTES}")

    row1 = _log_row(
        1, 'H_md.1_trivial_mean_reversion_vs_train_majority', primary,
        primary['h_md_1'], gate, leak, persistence, dev,
        f"THE TRIVIAL RULE, fixed at registration and never tuned: predict DOWN if the "
        f"trailing {TRIVIAL_RULE_LOOKBACK}-bar log return is positive, UP if negative, "
        f"train-majority class ({primary['majority_class']}) on an exact zero. No "
        f"fitting, no features. Reference = train-majority on IDENTICAL validation rows "
        f"(acc {primary['acc_majority']:.6f}). AUC is descriptive only and uses the "
        f"rule's own decision function (-logret_{TRIVIAL_RULE_LOOKBACK}). "
        f"Shuffled-label control (on H_md.2's GBM) = {leak:.6f}. {common}")

    row2 = _log_row(
        2, 'H_md.2_GBM_vs_trivial_rule', primary, primary['h_md_2'], gate, leak,
        persistence, dev,
        f"XGBoost device=cuda tree_method=hist n_estimators=300 max_depth=4 "
        f"learning_rate=0.05 subsample=0.8 colsample_bytree=0.8 balanced "
        f"scale_pos_weight, FIXED hyperparameters, NO early_stopping, NO eval_set. "
        f"PRIMARY reference = H_md.1's predictions on IDENTICAL validation rows "
        f"(acc {primary['acc_trivial']:.6f}); a model that beats a coin flip but not a "
        f"one-line rule has demonstrated nothing. CORROBORATING CONTEXT ONLY (never a "
        f"path to KEEP) vs train-majority: {primary['h_md_2_corroborating']['acc_challenger']:.6f} "
        f"vs {primary['h_md_2_corroborating']['acc_reference']:.6f}, delta "
        f"{primary['h_md_2_corroborating']['delta_acc']:+.6f}, block CI "
        f"[{primary['h_md_2_corroborating']['delta_acc_ci_low_block']:+.6f},"
        f"{primary['h_md_2_corroborating']['delta_acc_ci_high_block']:+.6f}], McNemar "
        f"p={primary['h_md_2_corroborating']['mcnemar_p']:.4g}. {common}")

    result['row1'], result['row2'] = row1, row2
    if register:
        _upsert_log(row1, log_path)
        _upsert_log(row2, log_path)

    # BINDING RULE evaluated explicitly rather than left to the reader.
    band_clears = {h: bool(band[h]['h_md_1']['cleared_bar']
                           or band[h]['h_md_2']['cleared_bar']) for h in band}
    result['horizon_fragility'] = bool(
        not (primary['h_md_1']['cleared_bar'] or primary['h_md_2']['cleared_bar'])
        and any(band_clears.values()))
    result['band_would_have_cleared'] = band_clears
    return result


# ═════════════════ STEP 9 — the report, in the pre-registered order ════════════

def _print_report(r):
    """RAW numbers. No P&L, no ledger, no equity curve. A DROP is not softened."""
    print('\n' + '=' * 78)
    print('MULTI-DAY MEAN REVERSION ON H1 — RESULTS (raw)')
    print('=' * 78)

    d = r['device_info']
    print('\n1. DEVICE / CUDA')
    print(f"   torch.cuda.is_available() : {d['cuda_available']}")
    if d['cuda_available']:
        print(f"   CUDA device              : {d['cuda_device_name']}")
    print(f"   device used              : {r['device']}     seed = {r['seed']}")

    print('\n2. STEP 0 — VARIANCE RATIO RE-DERIVED ON TRAIN DATA ALONE (hard gate)')
    print(f"   train slice : {r['span'][0][:10]} -> {str(r['train_end_ts'])[:10]}   "
          f"validation : -> {str(r['val_end_ts'])[:10]}   test block: NEVER profiled")
    print(f"\n   {'q':>5}{'VR(train)':>12}{'z_homo':>10}{'z_robust':>11}"
          f"{'  |':>4}{'VR(val)':>11}{'z_homo':>10}{'z_robust':>11}")
    tp, vp = r['train_vr_profile'], r['val_vr_profile']
    for _, row in tp.iterrows():
        v = vp[vp['q'] == row['q']].iloc[0]
        star = ' *' if row['q'] == GATE_Q else '  '
        print(f"   {int(row['q']):>3}{star}{row['vr']:>12.4f}{row['z_homoskedastic']:>10.2f}"
              f"{row['z_robust']:>11.2f}{'  |':>4}{v['vr']:>11.4f}"
              f"{v['z_homoskedastic']:>10.2f}{v['z_robust']:>11.2f}")
    g = r['gate']
    if not g.get('passed'):
        print(f"\n   GATE: FAILED\n   {g['message']}")
        print('\n   PROGRAM CANCELLED. No hypothesis registered.')
        return
    print(f"\n   GATE (q={GATE_Q}): VR = {g['train_vr']:.4f} < 1 and robust "
          f"z = {g['train_z_robust']:.4f} < {GATE_MAX_Z_ROBUST}  ->  PASSED")
    print("   The horizon choice is justified from TRAIN DATA ALONE; the reserved")
    print("   test block remains clean and unindexed for this family.")

    p, pw, c = r['primary'], r['primary']['power'], r['primary']['counts']
    print('\n3. ROW COUNTS, PURGE/EMBARGO AND THE POWER STATEMENT (lead with these)')
    print(f"   raw H1 bars              : {r['counts']['n_bars_raw']}")
    print(f"   feature-valid grid       : {r['counts']['n_feature_valid']}")
    print(f"   split positions          : train [0:{c['train_end_pos']})  "
          f"val [{c['train_end_pos']}:{c['val_end_pos']})  "
          f"test [{c['val_end_pos']}:{c['n_grid']}) RESERVED")
    print(f"   purged (last {PRIMARY_HORIZON} train rows) : {c['n_purged']}")
    print(f"   embargoed (first {EMBARGO_BARS} val bars): {c['n_embargoed']}")
    print(f"   val tail dropped (label window would reach the test block): "
          f"{c['n_val_tail_dropped']}")
    print(f"   train / val scored       : {c['n_train']} / {c['n_val']}")
    print(f"   test RESERVED            : {c['n_test_reserved']} rows, never indexed "
          f"(furthest position read = {c['max_position_read']} < {c['val_end_pos']})")

    def _power(tag, q):
        print(f"   {tag:<9}{q['n_rows']:>9}{q['n_independent']:>15.1f}"
              f"{q['mean_label_uniqueness']:>13.5f}{q['std_signed_pips']:>13.2f}"
              f"{q['mean_abs_move_pips']:>12.2f}{q['se_accuracy_from_independent'] * 100:>11.2f}pp"
              f"{q['min_detectable_edge_pp']:>11.2f}pp{q['breakeven_accuracy_pct']:>13.2f}%")

    print(f"\n   {'horizon':<9}{'n_rows':>9}{'n_independent':>15}{'uniqueness':>13}"
          f"{'sd(pips)':>13}{'mean|pips|':>12}{'SE(acc)':>13}{'min.det.edge':>13}"
          f"{'breakeven':>14}")
    _power(f'{PRIMARY_HORIZON} bars *', pw)
    for h in BAND_HORIZONS:
        if h in r['band']:
            _power(f'{h} bars', r['band'][h]['power'])
    print(f"\n   * = PRIMARY (decision-bearing). n_independent = n_rows / horizon; every")
    print(f"     power figure uses it, never the {pw['n_rows']} overlapping rows.")
    print(f"   min detectable edge at alpha={pw['alpha']} (z={pw['z_alpha']:.3f}) = "
          f"{pw['min_detectable_edge_pp']:.2f}pp")
    print(f"   breakeven accuracy = (1 + {SPREAD_PIPS}/{pw['mean_abs_move_pips']:.2f})/2 = "
          f"{pw['breakeven_accuracy_pct']:.2f}%  — DESCRIPTIVE ANCHOR, not a P&L")
    print(f"   the spread is {pw['spread_share_of_move_pct']:.1f}% of the typical move "
          f"at {PRIMARY_HORIZON} bars (the horizon-selection criterion)")
    if pw['min_detectable_edge_pp'] > (pw['breakeven_accuracy_pct'] - 50.0):
        print("   NOTE: the minimum DETECTABLE edge exceeds the breakeven excess over")
        print("   50%, so an edge this test could resolve need not be a meaningful one.")

    print('\n4. MANDATORY CONTROLS (no alpha consumed)')
    print(f"   (a) shuffled-label control on H_md.2's GBM : "
          f"{p['shuffled_label_control_acc']:.6f}")
    print(f"       train-majority rate on the same rows   : {p['acc_majority']:.6f}"
          f"   -> {'SANE (near majority/chance)' if abs(p['shuffled_label_control_acc'] - 0.5) < 0.02 else 'ANOMALOUS — INSPECT'}")
    pers = r.get('persistence')
    if pers:
        print(f"\n   (b) THE PERSISTENCE CHECK — the committed H_dir.1 model, UNCHANGED,")
        print(f"       scored against the {PRIMARY_HORIZON}-bar forward direction")
        print(f"       rows scored              : {pers['n_scored']} of "
              f"{pers['n_h_dir1_val_rows']} H_dir.1 validation rows "
              f"({pers['n_dropped_for_test_block']} dropped: their {PRIMARY_HORIZON}-bar "
              f"window would reach the reserved test block)")
        print(f"       at 1 bar (its own target): {pers['acc_at_1bar_on_scored_rows']:.6f}"
              f"   vs majority {pers['acc_at_1bar_majority']:.6f}   "
              f"delta {pers['delta_at_1bar_on_scored_rows']:+.6f}")
        print(f"       at {PRIMARY_HORIZON} bars              : {pers['acc_at_24bar']:.6f}"
              f"   vs majority {pers['acc_at_24bar_majority']:.6f}   "
              f"delta {pers['delta_at_24bar']:+.6f}")
        print(f"       sqrt({PRIMARY_HORIZON}) drift projection : "
              f"{pers['sqrt24_projection_acc']:.4f} "
              f"(what a persistent +{H_DIR1_NEXT_BAR_EDGE_PP}pp next-bar drift would imply)")
        print(f"       -> the H1 microstructure edge "
              f"{'COMPOUNDS' if pers['delta_at_24bar'] > 0.05 else 'does NOT compound'} "
              f"to a 1-day horizon.")

    print(f"\n5. H_md.1 — THE ONE-LINE RULE  (alpha = {FAMILY_ALPHA}; BLOCK bootstrap governs)")
    _print_arbiter(p['h_md_1'], 'trivial rule', 'train-majority')
    print(f"\n6. H_md.2 — DOES THE GBM BEAT IT?  (alpha = {FAMILY_ALPHA})")
    _print_arbiter(p['h_md_2'], 'GBM', 'trivial rule (PRIMARY)')
    print('\n   CORROBORATING CONTEXT ONLY — never a path to KEEP:')
    _print_arbiter(p['h_md_2_corroborating'], 'GBM', 'train-majority')

    if r['band']:
        print('\n7. DECLARED HORIZON BAND (48/72/96) — DESCRIPTIVE ONLY, no alpha, no verdict')
        print(f"   {'h':<7}{'n_rows':>8}{'n_ind':>8}{'majority':>10}{'trivial':>10}"
              f"{'GBM':>9}{'d(triv-maj)':>13}{'d(GBM-triv)':>13}{'blockCI(GBM-triv)':>26}")
        for h in BAND_HORIZONS:
            if h not in r['band']:
                continue
            b = r['band'][h]
            a1, a2 = b['h_md_1'], b['h_md_2']
            print(f"   {h:<7}{b['n_val']:>8}{b['power']['n_independent']:>8.0f}"
                  f"{b['acc_majority']:>10.4f}{b['acc_trivial']:>10.4f}{b['acc_gbm']:>9.4f}"
                  f"{a1['delta_acc']:>+13.4f}{a2['delta_acc']:>+13.4f}"
                  f"   [{a2['delta_acc_ci_low_block']:+.4f}, {a2['delta_acc_ci_high_block']:+.4f}]")
        print(f"\n   BINDING RULE: a band member clearing while the primary DROPs is NOT a")
        print(f"   KEEP — it is HORIZON FRAGILITY.  fragility observed = "
              f"{r['horizon_fragility']}")

    s = p.get('sessions')
    if s is not None:
        print('\n8. NY SESSION BREAKDOWN — DESCRIPTIVE ONLY, never a path to KEEP')
        print(f"   {'session':<12}{'n_val':>8}{'up_rate':>10}{'majority':>11}"
              f"{'trivial':>10}{'GBM':>9}{'d(GBM-maj)':>13}")
        for _, row in s.iterrows():
            print(f"   {row['session']:<12}{int(row['n_val']):>8}{row['up_rate']:>10.4f}"
                  f"{row['acc_majority']:>11.4f}{row['acc_trivial']:>10.4f}"
                  f"{row['acc_gbm']:>9.4f}"
                  f"{row['acc_gbm'] - row['acc_majority']:>+13.4f}")
        print("   At a 24-bar horizon the entry hour should matter far less than at 1 bar;")
        print("   an edge appearing ONLY outside the London/NY overlap is a warning sign.")

    imp = p.get('importance')
    if imp is not None:
        print('\n9. FEATURE IMPORTANCE — do the three ADDED multi-day lags carry the model?')
        print(f"   {'feature':<16}{'added?':>8}{'gain%':>10}{'perm_mean':>13}"
              f"{'perm_std':>11}{'perm%':>9}")
        for _, row in imp.iterrows():
            print(f"   {row['feature']:<16}{'YES' if row['is_added_multiday_lag'] else '-':>8}"
                  f"{row['gain_share'] * 100:>9.1f}%{row['perm_importance_mean']:>+13.5f}"
                  f"{row['perm_importance_std']:>11.5f}{row['perm_share'] * 100:>8.1f}%")
        added = imp[imp['is_added_multiday_lag']]
        print(f"\n   the three ADDED lags together: gain "
              f"{added['gain_share'].sum() * 100:.1f}%  of total, permutation "
              f"{added['perm_share'].sum() * 100:.1f}% of total")

    print('\n10. VERDICTS')
    print(f"   H_md.1  trivial rule vs train-majority : {p['h_md_1']['verdict']}")
    print(f"   H_md.2  GBM vs the trivial rule        : {p['h_md_2']['verdict']}")
    v1, v2 = p['h_md_1']['cleared_bar'], p['h_md_2']['cleared_bar']
    if v1 and not v2:
        print('\n   PLAIN READING: the finding is a ONE-LINE RULE. The machine learning')
        print('   added NOTHING — the GBM does not beat a rule with no parameters.')
    elif not v1 and not v2:
        print('\n   PLAIN READING: neither the one-line rule nor the GBM beats its')
        print('   pre-registered reference. The mean reversion the variance ratio')
        print('   detects does NOT translate into directional predictability here.')
    elif v2 and not v1:
        print('\n   PLAIN READING: the GBM beats the one-line rule, but the one-line rule')
        print('   itself does not beat the train-majority baseline.')
    else:
        print('\n   PLAIN READING: both cleared their pre-registered bars.')


def _print_arbiter(a, challenger, reference):
    print(f"   acc {challenger} = {a['acc_challenger']:.6f}   "
          f"acc {reference} = {a['acc_reference']:.6f}   "
          f"delta = {a['delta_acc']:+.6f}")
    print(f"   ROC-AUC (descriptive only)     = {a['auc_challenger']:.6f}")
    print(f"   delta CI iid   [{a['delta_acc_ci_low_iid']:+.5f}, "
          f"{a['delta_acc_ci_high_iid']:+.5f}]")
    print(f"   delta CI BLOCK [{a['delta_acc_ci_low_block']:+.5f}, "
          f"{a['delta_acc_ci_high_block']:+.5f}]  (block_len={a['block_len']}) <- GOVERNS")
    print(f"   McNemar exact  b={a['mcnemar_b']} c={a['mcnemar_c']} "
          f"p={a['mcnemar_p']:.6g}   alpha={a['alpha']}   -> {a['verdict']}")


if __name__ == '__main__':
    _print_report(run())
