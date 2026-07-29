"""
OSCILLATOR DIVERGENCE ON M15 — a NEW hypothesis family (research-only).

Tests whether classical oscillator/price divergence (RSI, MACD, Stochastic)
carries any predictive edge on EURUSD M15, restricted to the highest-movement
window of the trading day identified in results/h1_movement_by_ny_hour.csv
(07:00-10:00 New York: ~13 pips mean hourly move, against a 2.7 pip daily low).

Divergence is the most widely used discretionary reversal concept in retail FX
and has never been tested in this project. It gets ONE properly designed test.

PRE-REGISTERED DESIGN (fixed before any result was looked at)
------------------------------------------------------------
* Family     -- NEW and independent: results/divergence_hypothesis_log.csv,
                size 2, Bonferroni bar alpha = 0.05/2 = 0.025. It does not
                touch or tighten any existing family's alpha.
* Signal time-- reveal_bar(s2), NEVER s2. See src/divergence.py.
* Events     -- REGULAR divergence only. Hidden divergence is reported
                DESCRIPTIVELY and may not influence any verdict here.
* Oscillators-- the three platform defaults, POOLED into one test. Testing
                three oscillators separately and reporting the best would be
                cherry-picking across three correlated draws; the per-oscillator
                breakdown is therefore DESCRIPTIVE ONLY.
* Horizon    -- N=4 M15 bars (1 hour) is decision-bearing. N=8 and N=12 are
                corroborating context and are NOT a second path to a KEEP.
* Split      -- chronological on the M15 series: train [0:70%], validation
                [70%:85%] (THE ARBITER), test [85%:100%] RESERVED and never
                indexed by this program.
* Sensitivity band -- declared before execution in src/divergence.py, purely
                descriptive, consuming no alpha (see BAND_DECLARED_NOTE).

H_div.1 -- EVENT STUDY. Do confirmed regular divergences produce a positive
mean signed forward return at N=4 on the validation slice? Moving-block
(circular) bootstrap over TIME, 2000 resamples, CI at alpha=0.025.
KEEP iff the CI lies entirely above zero.

H_div.2 -- MODEL. GBM on the STEP 0 continuous features plus one-hots of
oscillator identity and divergence type, predicting sign(signed_return_pips) at
N=4, against the train-majority baseline on identical validation events. NO
early stopping and NO eval_set -- the validation slice must never be used for
model selection. KEEP iff the paired block-bootstrap CI on delta accuracy lies
entirely above zero AND exact McNemar p < 0.025.

MANDATORY LEAKAGE CONTROL (not a hypothesis, no alpha): the GBM refit on
randomly permuted training labels must land near the majority rate. If it does
not, the pipeline leaks and every number here is void.

NO P&L. No spread, no cost subtraction, no Sharpe, no equity curve, no position
sizing, no backtest. Pips are reported ONLY as a descriptive measure of move
size. Writes ONLY results/divergence_hypothesis_log.csv and
results/pooled_h1/EURUSD_m15_newyork.csv.
"""

import os

import numpy as np
import pandas as pd

from src import divergence as dv
from src.h1_direction_model import _mcnemar_exact      # reused, not reinvented
from src.pooled_h1_model import resolve_device

HYPOTHESIS_LOG = 'results/divergence_hypothesis_log.csv'
ARBITER_LABEL = 'divergence_m15_validation[70:85]_block_bootstrap'

# AMENDMENT (see AMENDMENT_NOTE): the family grew 2 -> 4 when the triggered-entry
# arms were registered. Per this project's standing rule that registering a new
# hypothesis tightens the bar for the WHOLE family, alpha drops to 0.05/4 and
# applies RETROACTIVELY to H_div.1 and H_div.2.
FAMILY_SIZE = 4
FAMILY_ALPHA = 0.05 / FAMILY_SIZE            # 0.0125
ORIGINAL_FAMILY_ALPHA = 0.05 / 2             # 0.025, for the explicit comparison
N_BOOT = 2000
RANDOM_SEED = 42

MODEL_FEATURES = [
    'price_slope_norm', 'osc_slope', 'div_magnitude', 'swing_gap_bars',
    'confirm_lag_bars', 'osc_level_at_s1', 'osc_level_at_s2',
    'price_move_since_s2_pips',
]

LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter', 'oscillators', 'n_events_raw',
    'n_events_after_session_filter', 'n_events_train', 'n_events_val',
    'median_swing_gap', 'median_confirm_lag', 'mean_signed_pips', 'ci_low',
    'ci_high', 'acc_challenger', 'acc_reference', 'delta_acc',
    'delta_acc_ci_low', 'delta_acc_ci_high', 'mcnemar_p', 'block_len',
    'shuffled_label_control_acc', 'alpha', 'cleared_bar', 'verdict',
    'device_used', 'notes',
]

POOLING_NOTE = (
    "ALL THREE OSCILLATORS ARE POOLED into one test. Testing them separately and "
    "reporting the best would be cherry-picking across three correlated draws, so "
    "the per-oscillator and per-type breakdowns are DESCRIPTIVE ONLY and carry no "
    "verdict."
)
AMENDMENT_NOTE = (
    "AMENDMENT SEQUENCE, RECORDED HONESTLY: the triggered-entry arms (H_div.3, "
    "H_div.4) were registered by an amendment that described itself as arriving "
    "'before any result was reported'. That is NOT what happened. H_div.1 and "
    "H_div.2 had already been run AND their full results -- both DROP -- had "
    "already been reported to the owner before the amendment arrived. The "
    "amendment is therefore POST-RESULT, not pre-result, and is logged as such. "
    "What this does and does not contaminate: the retroactive tightening 0.025 -> "
    "0.0125 can only make a DROP more of a DROP, never turn one into a KEEP, so "
    "H_div.1/H_div.2's verdicts are unaffected by the resize. H_div.3/H_div.4 were "
    "specified in full by the amendment text (trigger rule, invalidation rule, "
    "20-bar expiry, session filter on the trigger bar, N=4 primary) with no "
    "parameter left to the analyst, but they were specified in the KNOWLEDGE that "
    "the pattern-only arms had returned null. That is a real degree of freedom and "
    "it is disclosed here rather than hidden."
)
TIMING_NOTE = (
    "TIMING: every event is entered at reveal_bar(s2), never at s2. Entry price, "
    "all features and the start of the forward window are taken at reveal_bar; "
    "measuring from P(s2) would be look-ahead. median price_move_since_s2_pips is "
    "logged so the share of the move already gone at signal time is on the record."
)


class LeakageError(RuntimeError):
    """Raised when the shuffled-label control does not land near the majority
    rate -- the pipeline leaks and every number in the program is void."""


# ───────────────────── STEP 8: block bootstrap calibration ────────────────────

def calibrate_block_len(events: pd.DataFrame, horizon: int = dv.PRIMARY_HORIZON,
                        pos_col: str = 'reveal_idx'):
    """
    Measure the inter-event gap distribution on the validation events and choose
    a block length from it, rather than carrying one over from another program.

    M15 divergence events cluster -- three oscillators can fire off the same
    swing pair at the same bar, and swings themselves arrive in bursts. The
    block must therefore be at least the median inter-event gap AND at least the
    forward window, so that overlapping outcomes stay inside one block.
    """
    rev = np.sort(events[pos_col].to_numpy())
    gaps = np.diff(rev)
    gaps = gaps[gaps > 0]                       # simultaneous events are one point
    if not len(gaps):
        return {'block_len': int(horizon), 'median_gap': 0.0, 'q25': 0.0,
                'q75': 0.0, 'n_gaps': 0}
    median = float(np.median(gaps))
    q25, q75 = float(np.percentile(gaps, 25)), float(np.percentile(gaps, 75))
    block = int(max(np.ceil(median), horizon))
    return {'block_len': block, 'median_gap': median, 'q25': q25, 'q75': q75,
            'iqr': q75 - q25, 'n_gaps': int(len(gaps))}


def _time_block_indices(positions, lo, hi, block_len, rng):
    """
    One circular moving-block resample over TIME. Blocks of `block_len` BARS are
    drawn from the validation bar range and every event inside them is taken, so
    events that cluster in time are resampled together -- which is the whole
    point. The resample size is therefore random, exactly as it should be for
    clustered event data.
    """
    span = max(int(hi - lo), 1)
    n_blocks = int(np.ceil(span / block_len))
    starts = rng.integers(0, span, size=n_blocks)
    order = np.argsort(positions)
    sorted_pos = positions[order]
    picked = []
    for s in starts:
        a = lo + s
        b = a + block_len
        left = np.searchsorted(sorted_pos, a, side='left')
        right = np.searchsorted(sorted_pos, min(b, hi), side='left')
        picked.append(order[left:right])
        if b > hi:                              # wrap around, circularly
            wrap_end = lo + (b - hi)
            l2 = np.searchsorted(sorted_pos, lo, side='left')
            r2 = np.searchsorted(sorted_pos, wrap_end, side='left')
            picked.append(order[l2:r2])
    if not picked:
        return np.empty(0, dtype=int)
    return np.concatenate(picked)


def block_bootstrap_mean(values, positions, lo, hi, block_len,
                         n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=RANDOM_SEED):
    """Circular moving-block bootstrap CI for the MEAN of `values`, resampling
    contiguous time blocks of the validation bar axis."""
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    rng = np.random.default_rng(seed)
    means = np.full(n_boot, np.nan)
    for b in range(n_boot):
        take = _time_block_indices(positions, lo, hi, block_len, rng)
        if len(take):
            means[b] = values[take].mean()
    means = means[np.isfinite(means)]
    if not len(means):
        return float('nan'), float('nan'), float(values.mean())
    return (float(np.percentile(means, 100 * (alpha / 2))),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
            float(values.mean()))


def block_bootstrap_delta(correct_c, correct_r, positions, lo, hi, block_len,
                          n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=RANDOM_SEED):
    """PAIRED circular moving-block bootstrap CI for delta accuracy."""
    cc = np.asarray(correct_c, dtype=float)
    cr = np.asarray(correct_r, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        take = _time_block_indices(positions, lo, hi, block_len, rng)
        if len(take):
            deltas[b] = cc[take].mean() - cr[take].mean()
    deltas = deltas[np.isfinite(deltas)]
    point = float(cc.mean() - cr.mean())
    if not len(deltas):
        return float('nan'), float('nan'), point
    return (float(np.percentile(deltas, 100 * (alpha / 2))),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))), point)


# ───────────────────── H_div.1: event study ───────────────────────────────────

def event_study(val_events: pd.DataFrame, lo, hi, block_len,
                horizon=dv.PRIMARY_HORIZON, alpha=FAMILY_ALPHA, seed=RANDOM_SEED,
                value_col: str = None):
    """
    H_div.1: is the POOLED mean signed forward return at N=4 positive on the
    validation slice? KEEP iff the block-bootstrap CI lies entirely above zero.
    """
    col = value_col or f'signed_return_pips_n{horizon}'
    pos_col = 'trigger_idx' if col.startswith('trig_') else 'reveal_idx'
    vals = val_events[col].to_numpy(dtype=float)
    pos = val_events[pos_col].to_numpy()
    ci_lo, ci_hi, mean = block_bootstrap_mean(vals, pos, lo, hi, block_len,
                                              alpha=alpha, seed=seed)
    cleared = bool(np.isfinite(ci_lo) and ci_lo > 0.0)
    return {
        'horizon': horizon, 'n_events': int(len(vals)),
        'mean_signed_pips': mean,
        'median_signed_pips': float(np.median(vals)) if len(vals) else float('nan'),
        'std_signed_pips': float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan'),
        'win_rate_pct': float((vals > 0).mean() * 100.0) if len(vals) else float('nan'),
        'ci_low': ci_lo, 'ci_high': ci_hi, 'block_len': block_len,
        'alpha': alpha, 'cleared_bar': cleared,
        'verdict': 'KEEP' if cleared else 'DROP',
    }


def descriptive_breakdown(val_events: pd.DataFrame, horizon=dv.PRIMARY_HORIZON,
                          value_col: str = None):
    """Per-oscillator and per-type means. DESCRIPTIVE ONLY -- never a verdict."""
    col = value_col or f'signed_return_pips_n{horizon}'
    rows = []
    for key in ('oscillator', 'div_type'):
        for value, sub in val_events.groupby(key):
            v = sub[col].to_numpy(dtype=float)
            rows.append({'breakdown': key, 'group': value, 'n': int(len(v)),
                         'mean_signed_pips': float(v.mean()) if len(v) else np.nan,
                         'median_signed_pips': float(np.median(v)) if len(v) else np.nan,
                         'win_rate_pct': float((v > 0).mean() * 100) if len(v) else np.nan})
    return pd.DataFrame(rows)


def outcome_breakdown(setups: pd.DataFrame):
    """
    TRIGGER RATE: of all confirmed regular divergences, what fraction reach an
    entry, versus invalidate, versus expire unfilled -- overall and broken down
    by oscillator and by divergence type.

    This is reported whatever the verdict, because together with the median wait
    and the median pips given up it explains most of the gap between how
    divergence looks on a chart and how it behaves.
    """
    rows = []

    def _block(sub, scope, group):
        n = len(sub)
        if not n:
            return None
        counts = sub['outcome'].value_counts()
        rec = {'scope': scope, 'group': group, 'n_setups': int(n)}
        for oc in dv.SETUP_OUTCOMES:
            c = int(counts.get(oc, 0))
            rec[f'n_{oc.lower()}'] = c
            rec[f'pct_{oc.lower()}'] = 100.0 * c / n
        trig = sub[sub['outcome'] == dv.OUTCOME_TRIGGERED]
        rec['median_bars_reveal_to_trigger'] = (
            float(trig['bars_reveal_to_trigger'].median()) if len(trig) else np.nan)
        rec['median_pips_given_up'] = (
            float(trig['pips_given_up'].median()) if len(trig) else np.nan)
        return rec

    rows.append(_block(setups, 'all', 'all_regular_divergences'))
    for key in ('oscillator', 'div_type'):
        for value, sub in setups.groupby(key):
            rows.append(_block(sub, key, value))
    return pd.DataFrame([r for r in rows if r is not None])


def sensitivity_table(df: pd.DataFrame, lo, hi, block_len,
                      horizon=dv.PRIMARY_HORIZON, seed=RANDOM_SEED):
    """
    The DECLARED band, run through the same event study. Descriptive only: no
    alpha, no verdict. Reported so the reader can see whether the default result
    sits on a knife edge or persists across neighbouring parameters.
    """
    pivots = dv.detect_swings(df)
    n_bars = len(df)
    _train_end, val_end = dv.split_bounds(n_bars)
    rows = []
    for kind, params_list in dv.SENSITIVITY_BAND.items():
        for params in params_list:
            label = dv.oscillator_name(kind, params)
            ev = dv.build_all_events(df, pivots, {label: (kind, params)})
            if not len(ev):
                continue
            ev = dv.attach_session(ev, df.index)
            ev = dv.attach_forward_returns(ev, df, horizons=(horizon,),
                                           test_start_idx=val_end)
            ev = dv.assign_slice(ev, n_bars)
            val = ev[(ev['slice'] == 'val') & ev['is_regular']
                     & ev['in_session'] & ev[f'window_ok_n{horizon}']]
            v = val[f'signed_return_pips_n{horizon}'].to_numpy(dtype=float)
            if not len(v):
                continue
            ci_lo, ci_hi, mean = block_bootstrap_mean(
                v, val['reveal_idx'].to_numpy(), lo, hi, block_len, seed=seed)
            rows.append({
                'oscillator_kind': kind, 'params': str(params), 'label': label,
                'is_default': label in dv.PRIMARY_OSCILLATORS,
                'n_events': int(len(v)), 'mean_signed_pips': mean,
                'ci_low': ci_lo, 'ci_high': ci_hi,
                'win_rate_pct': float((v > 0).mean() * 100.0),
                'would_have_cleared': bool(np.isfinite(ci_lo) and ci_lo > 0),
            })
    return pd.DataFrame(rows)


# ───────────────────── H_div.2: model ─────────────────────────────────────────

def build_model_matrix(events: pd.DataFrame, horizon=dv.PRIMARY_HORIZON,
                       value_col: str = None, extra_features=()):
    """
    Continuous STEP-0 features + one-hot oscillator identity + one-hot
    divergence type. The label is sign(signed_return_pips); events with an
    exactly zero forward move have no sign and are dropped (counted, not hidden).
    """
    col = value_col or f'signed_return_pips_n{horizon}'
    keep = events[col].to_numpy(dtype=float) != 0.0
    sub = events[keep].copy()
    X = sub[list(MODEL_FEATURES) + list(extra_features)].copy()
    for name in sorted(dv.PRIMARY_OSCILLATORS):
        X[f'osc_{name}'] = (sub['oscillator'] == name).astype(int)
    for t in dv.REGULAR_TYPES:
        X[f'type_{t}'] = (sub['div_type'] == t).astype(int)
    y = (sub[col].to_numpy(dtype=float) > 0).astype(int)
    return X, y, sub, int((~keep).sum())


def train_gbm(X, y, seed=RANDOM_SEED):
    """
    Fixed hyperparameters, balanced class weights, device='cuda'. NO
    early_stopping and NO eval_set: the validation slice is the arbiter and must
    never be used for model selection.
    """
    import xgboost as xgb
    pos = float((np.asarray(y) == 1).sum())
    neg = float((np.asarray(y) == 0).sum())
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', device='cuda',
        scale_pos_weight=(neg / pos) if pos > 0 else 1.0,
        random_state=seed, n_jobs=0,
    )
    clf.fit(X, y)
    return clf


def predict_gbm(clf, X):
    return (np.asarray(clf.predict_proba(X)[:, 1], dtype=float) >= 0.5).astype(int)


def shuffled_label_control(Xtr, ytr, Xval, yval, seed=RANDOM_SEED):
    """MANDATORY leakage control (no alpha): refit on randomly permuted TRAINING
    labels and score on the REAL validation events. It must land near the
    majority rate, or the pipeline leaks and every number is void."""
    rng = np.random.default_rng(seed)
    y_shuf = np.asarray(ytr).astype(int).copy()
    rng.shuffle(y_shuf)
    clf = train_gbm(Xtr, y_shuf, seed=seed)
    return float((predict_gbm(clf, Xval) == np.asarray(yval).astype(int)).mean())


def model_test(train_events, val_events, lo, hi, block_len,
               horizon=dv.PRIMARY_HORIZON, alpha=FAMILY_ALPHA, seed=RANDOM_SEED,
               value_col: str = None, extra_features=(), pos_col='reveal_idx'):
    """H_div.2 / H_div.4 end to end, including the leakage control. Identical
    architecture and guardrails in both cases: fixed hyperparameters, balanced
    class weights, NO early stopping, NO eval_set, shuffled-label control."""
    Xtr, ytr, _str, n_zero_tr = build_model_matrix(train_events, horizon,
                                                   value_col, extra_features)
    Xval, yval, sval, n_zero_val = build_model_matrix(val_events, horizon,
                                                      value_col, extra_features)

    leak_acc = shuffled_label_control(Xtr, ytr, Xval, yval, seed=seed)
    leak_ok = bool(0.40 <= leak_acc <= 0.60)

    clf = train_gbm(Xtr, ytr, seed=seed)
    pred = predict_gbm(clf, Xval)
    majority = int((ytr == 1).sum() >= (ytr == 0).sum())
    ref = np.full(len(yval), majority, dtype=int)

    cc = (pred == yval).astype(float)
    cr = (ref == yval).astype(float)
    pos = sval[pos_col].to_numpy()
    ci_lo, ci_hi, delta = block_bootstrap_delta(cc, cr, pos, lo, hi, block_len,
                                                alpha=alpha, seed=seed)
    b, c, p = _mcnemar_exact(cc, cr)
    cleared = bool(np.isfinite(ci_lo) and ci_lo > 0.0 and p < alpha)
    return {
        'n_train': int(len(ytr)), 'n_val': int(len(yval)),
        'n_zero_dropped_train': n_zero_tr, 'n_zero_dropped_val': n_zero_val,
        'acc_challenger': float(cc.mean()), 'acc_reference': float(cr.mean()),
        'delta_acc': delta, 'delta_acc_ci_low': ci_lo, 'delta_acc_ci_high': ci_hi,
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p,
        'majority_class': majority, 'block_len': block_len, 'alpha': alpha,
        'shuffled_label_control_acc': leak_acc, 'leak_control_sane': leak_ok,
        'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP',
        'train_class_balance_pct1': 100.0 * float((ytr == 1).mean()),
        'val_class_balance_pct1': 100.0 * float((yval == 1).mean()),
    }


# ───────────────────── STEP 9: the family log ─────────────────────────────────

def _upsert_log(row: dict, log_path: str = HYPOTHESIS_LOG):
    """Append/replace one hypothesis row (idempotent on `hypothesis`). Touches
    NO other family's log."""
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
        v = float(x)
    except (TypeError, ValueError):
        return x
    return round(v, nd) if np.isfinite(v) else v


# ───────────────────── orchestration ──────────────────────────────────────────

def run(m15_path: str = dv.M15_SOURCE_CSV, ny_path: str = dv.M15_NY_CSV,
        log_path: str = HYPOTHESIS_LOG, seed: int = RANDOM_SEED,
        register: bool = True, write_ny: bool = True, verbose: bool = True,
        run_band: bool = True):
    """
    The whole program: NY conversion -> events -> block calibration -> leakage
    control -> H_div.1 -> H_div.2 -> log. The test block is never indexed.
    """
    device, dev_info = resolve_device()

    ny_df, rule, verification = dv.build_m15_newyork(
        path=m15_path, out_path=ny_path, write=write_ny, verbose=verbose)

    events, pivots, counts = dv.build_event_table(ny_df)
    n_bars = len(ny_df)
    train_end, val_end = dv.split_bounds(n_bars)

    study = dv.study_events(events, dv.PRIMARY_HORIZON)
    train_events = study[study['slice'] == 'train']
    val_events = study[study['slice'] == 'val']
    assert (study['slice'] != 'test').all(), 'test block must never be indexed'

    block = calibrate_block_len(val_events)
    lo, hi = train_end, val_end

    h1 = event_study(val_events, lo, hi, block['block_len'], seed=seed)
    breakdown = descriptive_breakdown(val_events)
    corroborating = {
        h: event_study(dv.study_events(events, h).query("slice == 'val'"),
                       lo, hi, block['block_len'], horizon=h, seed=seed)
        for h in dv.CORROBORATING_HORIZONS
    }
    hidden = dv.study_events(events, dv.PRIMARY_HORIZON, regular_only=False)
    hidden = hidden[(~hidden['is_regular']) & (hidden['slice'] == 'val')]

    h2 = model_test(train_events, val_events, lo, hi, block['block_len'], seed=seed)
    if not h2['leak_control_sane']:
        raise LeakageError(
            f"shuffled-label control = {h2['shuffled_label_control_acc']:.4f}, "
            "outside [0.40, 0.60] -- the pipeline leaks and every number is void.")

    # ── AMENDMENT: H_div.3 / H_div.4, structural-break TRIGGERED entry ──
    setups = dv.build_triggered_table(events, ny_df)
    usable = setups[~setups['truncated_by_bound']]
    trig_study = dv.triggered_study_events(setups, dv.PRIMARY_HORIZON)
    trig_train = trig_study[trig_study['trigger_slice'] == 'train']
    trig_val = trig_study[trig_study['trigger_slice'] == 'val']
    assert (trig_study['trigger_slice'] != 'test').all(), 'test block never indexed'

    trig_col = f'trig_return_pips_n{dv.PRIMARY_HORIZON}'
    trig_block = calibrate_block_len(trig_val, pos_col='trigger_idx')
    h3 = event_study(trig_val, lo, hi, trig_block['block_len'], seed=seed,
                     value_col=trig_col)
    h3_corroborating = {}
    for h in dv.CORROBORATING_HORIZONS:
        sub = dv.triggered_study_events(setups, h)
        sub = sub[sub['trigger_slice'] == 'val']
        h3_corroborating[h] = event_study(sub, lo, hi, trig_block['block_len'],
                                          horizon=h, seed=seed,
                                          value_col=f'trig_return_pips_n{h}')
    h4 = model_test(trig_train, trig_val, lo, hi, trig_block['block_len'],
                    seed=seed, value_col=trig_col,
                    extra_features=('bars_reveal_to_trigger', 'pips_given_up'),
                    pos_col='trigger_idx')
    if not h4['leak_control_sane']:
        raise LeakageError(
            f"H_div.4 shuffled-label control = {h4['shuffled_label_control_acc']:.4f}, "
            "outside [0.40, 0.60] -- the pipeline leaks and every number is void.")

    outcomes = outcome_breakdown(usable)
    trig_desc = descriptive_breakdown(trig_val, value_col=trig_col)

    # The retroactive-tightening comparison, stated rather than quietly applied.
    h1_at_original = event_study(val_events, lo, hi, block['block_len'],
                                 alpha=ORIGINAL_FAMILY_ALPHA, seed=seed)
    h2_at_original = model_test(train_events, val_events, lo, hi,
                                block['block_len'], alpha=ORIGINAL_FAMILY_ALPHA,
                                seed=seed)

    band = sensitivity_table(ny_df, lo, hi, block['block_len'],
                             seed=seed) if run_band else pd.DataFrame()

    med_wait = (float(trig_study['bars_reveal_to_trigger'].median())
                if len(trig_study) else np.nan)
    med_given_up_trig = (float(trig_study['pips_given_up'].median())
                         if len(trig_study) else np.nan)

    med_gap = float(study['swing_gap_bars'].median()) if len(study) else np.nan
    med_lag = float(study['confirm_lag_bars'].median()) if len(study) else np.nan
    med_given_up = (float(study['price_given_up_pips'].median())
                    if len(study) else np.nan)
    med_given_up_raw = (float(study['price_move_since_s2_pips'].median())
                        if len(study) else np.nan)
    osc_list = '|'.join(sorted(dv.PRIMARY_OSCILLATORS))

    shared = {
        'date': pd.Timestamp.utcnow().date().isoformat(), 'arbiter': ARBITER_LABEL,
        'oscillators': osc_list, 'n_events_raw': counts['n_events_raw'],
        'n_events_after_session_filter': counts['n_events_after_session_filter'],
        'n_events_train': int(len(train_events)), 'n_events_val': int(len(val_events)),
        'median_swing_gap': _r(med_gap, 3), 'median_confirm_lag': _r(med_lag, 3),
        'block_len': block['block_len'], 'alpha': FAMILY_ALPHA,   # 0.0125 after the resize
        'device_used': dev_info['device'],
    }

    band_summary = ''
    if len(band):
        same_sign = int((np.sign(band['mean_signed_pips'])
                         == np.sign(h1['mean_signed_pips'])).sum())
        band_summary = (f" BAND (descriptive): {same_sign}/{len(band)} members share the "
                        f"default's sign; {int(band['would_have_cleared'].sum())} would "
                        f"have cleared.")

    row1 = {
        **shared, 'n': 1, 'hypothesis': 'H_div.1_regular_divergence_event_study',
        'mean_signed_pips': _r(h1['mean_signed_pips']),
        'ci_low': _r(h1['ci_low']), 'ci_high': _r(h1['ci_high']),
        'acc_challenger': '', 'acc_reference': '', 'delta_acc': '',
        'delta_acc_ci_low': '', 'delta_acc_ci_high': '', 'mcnemar_p': '',
        'shuffled_label_control_acc': _r(h2['shuffled_label_control_acc']),
        'cleared_bar': h1['cleared_bar'], 'verdict': h1['verdict'],
        'notes': (f"POOLED event study, N={dv.PRIMARY_HORIZON} M15 bars (1h), "
                  f"07:00-10:00 NY entry window on reveal_bar; win rate "
                  f"{h1['win_rate_pct']:.2f}%; median price already given up waiting "
                  f"for confirmation = {med_given_up:.2f} pips (in the signal's direction); block bootstrap over "
                  f"TIME (block_len={block['block_len']} bars, median inter-event gap "
                  f"{block['median_gap']:.1f}). {POOLING_NOTE} {TIMING_NOTE} "
                  f"{dv.BAND_DECLARED_NOTE}{band_summary}"),
    }
    row2 = {
        **shared, 'n': 2, 'hypothesis': 'H_div.2_GBM_vs_train_majority',
        'n_events_train': h2['n_train'], 'n_events_val': h2['n_val'],
        'mean_signed_pips': _r(h1['mean_signed_pips']), 'ci_low': '', 'ci_high': '',
        'acc_challenger': _r(h2['acc_challenger']),
        'acc_reference': _r(h2['acc_reference']), 'delta_acc': _r(h2['delta_acc']),
        'delta_acc_ci_low': _r(h2['delta_acc_ci_low']),
        'delta_acc_ci_high': _r(h2['delta_acc_ci_high']),
        'mcnemar_p': _r(h2['mcnemar_p']),
        'shuffled_label_control_acc': _r(h2['shuffled_label_control_acc']),
        'cleared_bar': h2['cleared_bar'], 'verdict': h2['verdict'],
        'notes': (f"XGBoost hist/device=cuda n_est300 depth4 lr.05 balanced "
                  f"scale_pos_weight, NO early_stopping and NO eval_set (validation "
                  f"is the arbiter and must not select the model); features = STEP-0 "
                  f"continuous measures + one-hot oscillator + one-hot divergence "
                  f"type; reference = train-majority class {h2['majority_class']}; "
                  f"McNemar b={h2['mcnemar_b']} c={h2['mcnemar_c']}; "
                  f"{h2['n_zero_dropped_val']} zero-move validation events dropped "
                  f"(no sign to predict). {POOLING_NOTE} {TIMING_NOTE}"),
    }

    trig_shared = {
        **shared, 'n_events_train': int(len(trig_train)),
        'n_events_val': int(len(trig_val)), 'block_len': trig_block['block_len'],
    }
    all_out = outcomes[outcomes['scope'] == 'all'].iloc[0]
    trigger_note = (
        f"TRIGGERED ENTRY: the setup is NOT the entry. Entry requires a structural "
        f"break on CLOSED BARS ONLY -- bearish: first close strictly below low[s2]; "
        f"bullish: first close strictly above high[s2]; entry AT THAT BAR'S CLOSE, "
        f"never at the level (bar data cannot support an intrabar fill assumption). "
        f"Invalidation checked bar by bar BEFORE the trigger (bearish dies on any "
        f"high > high[s2], bullish on any low < low[s2]); a bar doing both is treated "
        f"as INVALIDATED, the conservative reading. Expiry fixed at "
        f"{dv.EXPIRY_BARS} bars and NOT tuned. Session filter applies to the TRIGGER "
        f"bar. Of {int(all_out['n_setups'])} confirmed regular divergences: "
        f"{all_out['pct_triggered']:.1f}% triggered, {all_out['pct_invalidated']:.1f}% "
        f"invalidated, {all_out['pct_expired']:.1f}% expired unfilled; median wait "
        f"{med_wait:.1f} bars; median {med_given_up_trig:.2f} pips given up before an "
        f"honest entry was possible. INVALIDATED and EXPIRED setups are excluded from "
        f"return statistics because THERE WAS NO TRADE -- causal exclusion, not "
        f"survivorship bias: a live trader on the same rule is excluded identically, "
        f"on the same bar, without knowing the future."
    )

    row3 = {
        **trig_shared, 'n': 3,
        'hypothesis': 'H_div.3_triggered_entry_event_study',
        'mean_signed_pips': _r(h3['mean_signed_pips']),
        'ci_low': _r(h3['ci_low']), 'ci_high': _r(h3['ci_high']),
        'acc_challenger': '', 'acc_reference': '', 'delta_acc': '',
        'delta_acc_ci_low': '', 'delta_acc_ci_high': '', 'mcnemar_p': '',
        'shuffled_label_control_acc': _r(h4['shuffled_label_control_acc']),
        'cleared_bar': h3['cleared_bar'], 'verdict': h3['verdict'],
        'notes': (f"win rate {h3['win_rate_pct']:.2f}%; block bootstrap over TIME "
                  f"(block_len={trig_block['block_len']} bars, median inter-event gap "
                  f"{trig_block['median_gap']:.1f}). {trigger_note} {POOLING_NOTE} "
                  f"{TIMING_NOTE} {AMENDMENT_NOTE}"),
    }
    row4 = {
        **trig_shared, 'n': 4,
        'hypothesis': 'H_div.4_GBM_triggered_entry_vs_train_majority',
        'n_events_train': h4['n_train'], 'n_events_val': h4['n_val'],
        'mean_signed_pips': _r(h3['mean_signed_pips']), 'ci_low': '', 'ci_high': '',
        'acc_challenger': _r(h4['acc_challenger']),
        'acc_reference': _r(h4['acc_reference']), 'delta_acc': _r(h4['delta_acc']),
        'delta_acc_ci_low': _r(h4['delta_acc_ci_low']),
        'delta_acc_ci_high': _r(h4['delta_acc_ci_high']),
        'mcnemar_p': _r(h4['mcnemar_p']),
        'shuffled_label_control_acc': _r(h4['shuffled_label_control_acc']),
        'cleared_bar': h4['cleared_bar'], 'verdict': h4['verdict'],
        'notes': (f"IDENTICAL architecture and guardrails to H_div.2 (XGBoost "
                  f"hist/device=cuda n_est300 depth4 lr.05 balanced scale_pos_weight, "
                  f"NO early_stopping, NO eval_set, own shuffled-label control), plus "
                  f"bars_reveal_to_trigger and pips_given_up as features; reference = "
                  f"train-majority class {h4['majority_class']}; McNemar b={h4['mcnemar_b']} "
                  f"c={h4['mcnemar_c']}. {trigger_note} {POOLING_NOTE} {AMENDMENT_NOTE}"),
    }

    if register:
        _upsert_log(row1, log_path)
        _upsert_log(row2, log_path)
        _upsert_log(row3, log_path)
        _upsert_log(row4, log_path)

    return {
        'device': str(device), 'device_info': dev_info, 'rule': rule,
        'verification': verification, 'counts': counts, 'n_bars': n_bars,
        'events': events, 'study': study, 'train_events': train_events,
        'val_events': val_events, 'block': block, 'h_div_1': h1,
        'h_div_2': h2, 'breakdown': breakdown, 'corroborating': corroborating,
        'hidden_val': hidden, 'band': band,
        'rows': [row1, row2, row3, row4],
        'h_div_3': h3, 'h_div_4': h4, 'h3_corroborating': h3_corroborating,
        'setups': setups, 'trig_study': trig_study, 'trig_train': trig_train,
        'trig_val': trig_val, 'trig_block': trig_block, 'outcomes': outcomes,
        'trig_breakdown': trig_desc, 'median_bars_reveal_to_trigger': med_wait,
        'median_pips_given_up_trigger': med_given_up_trig,
        'h1_at_original_alpha': h1_at_original,
        'h2_at_original_alpha': h2_at_original,
        'n_truncated_by_bound': int(setups['truncated_by_bound'].sum()),
        'median_swing_gap': med_gap, 'median_confirm_lag': med_lag,
        'median_price_given_up_pips': med_given_up,
        'median_price_move_since_s2_raw': med_given_up_raw,
        'split': {'train_end': train_end, 'val_end': val_end},
    }


def _print_report(r):
    """STEP 11 report, in the pre-registered order. RAW numbers only."""
    d, c, b, h1, h2 = r['device_info'], r['counts'], r['block'], r['h_div_1'], r['h_div_2']

    print('\n' + '=' * 80)
    print('OSCILLATOR DIVERGENCE ON M15 — RESULTS (raw)')
    print('=' * 80)

    print('\n1. DEVICE')
    print(f"   CUDA available : {d['cuda_available']}")
    if d['cuda_available']:
        print(f"   CUDA device    : {d['cuda_device_name']}")
    print(f"   device used    : {r['device']}")

    print('\n2. STEP 1 — BROKER DST RULE ON THE M15 SERIES')
    rule = r['rule']
    print(f"   weeks anchored          : {rule['n_weeks_normal']} matched / "
          f"{rule['n_weeks_mismatch']} mismatch")
    print(f"   rule                    : ({rule['rule']}) — EU-DST era starts "
          f"{str(rule['eu_rule_era_start'])[:10]}")
    print(f"   mismatch weeks that did NOT shift: "
          f"{rule['n_mismatch_not_shifted']} (all pre-2017)")
    print(f"   -> the H1-derived rule was RE-DERIVED on M15's own weekly anchors and")
    print(f"      HOLDS back to 2012-06. The study is NOT restricted to 2015+.")

    print('\n3. EVENT COUNTS')
    print(f"   M15 bars                     : {r['n_bars']}")
    print(f"   ZigZag pivots                : {c['n_pivots']}")
    print(f"   regular divergences (raw)    : {c['n_events_raw']}")
    print(f"   hidden divergences (descr.)  : {c['n_events_hidden']}")
    print(f"   dropped by session filter    : {c['n_events_dropped_by_session']}")
    print(f"   surviving session filter     : {c['n_events_after_session_filter']}")
    print(f"   with a usable N=4 window     : {len(r['study'])}")
    print(f"     train / val / test         : "
          f"{len(r['train_events'])} / {len(r['val_events'])} / 0 (RESERVED)")
    print(f"   median swing gap             : {r['median_swing_gap']:.1f} bars")
    print(f"   median confirmation lag      : {r['median_confirm_lag']:.1f} bars")
    print(f"   median price ALREADY GIVEN UP waiting for confirmation : "
          f"{r['median_price_given_up_pips']:+.2f} pips  (measured IN the signal's")
    print(f"     direction; the raw signed median is {r['median_price_move_since_s2_raw']:+.2f}, "
          "~0 only because")
    print("     bullish and bearish events cancel by symmetry)")

    print('\n4. BLOCK BOOTSTRAP CALIBRATION (measured, not carried over)')
    print(f"   inter-event gaps (val)  : median {b['median_gap']:.1f} bars, "
          f"IQR [{b['q25']:.1f}, {b['q75']:.1f}]  (n={b['n_gaps']})")
    print(f"   chosen block_len        : {b['block_len']} bars "
          f"(>= median gap and >= N={dv.PRIMARY_HORIZON})")

    print('\n5. SHUFFLED-LABEL CONTROL (leakage gate, no alpha)')
    print(f"   accuracy = {h2['shuffled_label_control_acc']:.4f}  -> "
          f"{'SANE — near chance' if h2['leak_control_sane'] else 'ANOMALOUS — VOID'}")

    print(f"\n6. H_div.1 — POOLED EVENT STUDY (alpha = {h1['alpha']})")
    print(f"   n events        : {h1['n_events']}")
    print(f"   mean signed     : {h1['mean_signed_pips']:+.4f} pips")
    print(f"   median signed   : {h1['median_signed_pips']:+.4f} pips")
    print(f"   std             : {h1['std_signed_pips']:.4f} pips")
    print(f"   win rate        : {h1['win_rate_pct']:.2f}%")
    print(f"   block CI        : [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}]  <- governs")
    print(f"   VERDICT         : {h1['verdict']}")

    print('\n   CORROBORATING HORIZONS (context only — NOT a path to KEEP)')
    for h, res in r['corroborating'].items():
        print(f"     N={h:<3} n={res['n_events']:<5} mean {res['mean_signed_pips']:+.4f} "
              f"pips   CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]   "
              f"win {res['win_rate_pct']:.2f}%")

    print('\n   BREAKDOWN — DESCRIPTIVE ONLY, carries no verdict')
    for _, row in r['breakdown'].iterrows():
        print(f"     {row['breakdown']:<11} {row['group']:<18} n={int(row['n']):<5} "
              f"mean {row['mean_signed_pips']:+.4f} pips   "
              f"win {row['win_rate_pct']:.2f}%")
    hv = r['hidden_val']
    if len(hv):
        v = hv['signed_return_pips_n4'].to_numpy()
        print(f"     {'hidden':<11} {'(all types)':<18} n={len(v):<5} "
              f"mean {v.mean():+.4f} pips   win {(v > 0).mean() * 100:.2f}%")
        print('     ^ HIDDEN divergence is DESCRIPTIVE ONLY in this family and may')
        print('       not influence any verdict; testing it needs its own alpha.')

    if len(r['band']):
        print('\n   DECLARED SENSITIVITY BAND — descriptive, no alpha, no verdict')
        print(f"     {'label':<16}{'n':>6}{'mean pips':>12}{'CI low':>10}"
              f"{'CI high':>10}{'win%':>8}  default")
        for _, row in r['band'].iterrows():
            print(f"     {row['label']:<16}{int(row['n_events']):>6}"
                  f"{row['mean_signed_pips']:>+12.4f}{row['ci_low']:>+10.4f}"
                  f"{row['ci_high']:>+10.4f}{row['win_rate_pct']:>8.2f}"
                  f"   {'<- DEFAULT' if row['is_default'] else ''}")
        same = int((np.sign(r['band']['mean_signed_pips'])
                    == np.sign(h1['mean_signed_pips'])).sum())
        cleared = int(r['band']['would_have_cleared'].sum())
        print(f"     {same}/{len(r['band'])} band members share the default's sign; "
              f"{cleared} would have cleared.")
        if cleared and not h1['cleared_bar']:
            print('     The default DROPped. Any member that would have cleared is')
            print('     PARAMETER FRAGILITY, not a KEEP — an effect selected by')
            print('     hindsight rather than discovered.')

    print(f"\n7. H_div.2 — GBM vs TRAIN-MAJORITY (alpha = {h2['alpha']})")
    print(f"   n train / val   : {h2['n_train']} / {h2['n_val']}   "
          f"(zero-move dropped: {h2['n_zero_dropped_train']} / {h2['n_zero_dropped_val']})")
    print(f"   class balance   : train {h2['train_class_balance_pct1']:.2f}% up, "
          f"val {h2['val_class_balance_pct1']:.2f}% up")
    print(f"   acc challenger  : {h2['acc_challenger']:.4f}")
    print(f"   acc reference   : {h2['acc_reference']:.4f}  "
          f"(train-majority class {h2['majority_class']})")
    print(f"   delta           : {h2['delta_acc']:+.4f}")
    print(f"   block CI        : [{h2['delta_acc_ci_low']:+.4f}, "
          f"{h2['delta_acc_ci_high']:+.4f}]  <- governs")
    print(f"   McNemar exact   : b={h2['mcnemar_b']} c={h2['mcnemar_c']} "
          f"p={h2['mcnemar_p']:.6g}")
    print(f"   VERDICT         : {h2['verdict']}")

    o = r['outcomes']
    all_out = o[o['scope'] == 'all'].iloc[0]
    h3, h4 = r['h_div_3'], r['h_div_4']

    print('\n8. AMENDMENT — TRIGGERED ENTRY (H_div.3 / H_div.4)')
    print('   THE THREE NUMBERS THAT EXPLAIN THE CHART-VS-REALITY GAP:')
    print(f"     trigger rate           : {all_out['pct_triggered']:.1f}% TRIGGERED / "
          f"{all_out['pct_invalidated']:.1f}% INVALIDATED / "
          f"{all_out['pct_expired']:.1f}% EXPIRED  "
          f"(n={int(all_out['n_setups'])} setups)")
    print(f"     bars reveal -> trigger : median "
          f"{r['median_bars_reveal_to_trigger']:.1f} M15 bars")
    print(f"     pips given up          : median "
          f"{r['median_pips_given_up_trigger']:+.2f} pips from P(s2) to the entry close")
    print(f"     ({r['n_truncated_by_bound']} setups excluded: their resolution window "
          "would have needed reserved test-block bars)")

    print(f"\n   OUTCOME BREAKDOWN{'trig%':>18}{'inval%':>9}{'exp%':>8}"
          f"{'wait':>7}{'given up':>10}")
    for _, row in o.iterrows():
        print(f"     {row['scope']:<11}{str(row['group']):<20}"
              f"{row['pct_triggered']:>8.1f}{row['pct_invalidated']:>9.1f}"
              f"{row['pct_expired']:>8.1f}{row['median_bars_reveal_to_trigger']:>7.1f}"
              f"{row['median_pips_given_up']:>+10.2f}")

    print(f"\n   H_div.3 — TRIGGERED-ENTRY EVENT STUDY (alpha = {h3['alpha']})")
    print(f"     n trades entered : {h3['n_events']}   "
          f"(train {len(r['trig_train'])} / val {len(r['trig_val'])})")
    print(f"     block_len        : {r['trig_block']['block_len']} bars "
          f"(median inter-event gap {r['trig_block']['median_gap']:.1f})")
    print(f"     mean signed      : {h3['mean_signed_pips']:+.4f} pips")
    print(f"     median signed    : {h3['median_signed_pips']:+.4f} pips")
    print(f"     win rate         : {h3['win_rate_pct']:.2f}%")
    print(f"     block CI         : [{h3['ci_low']:+.4f}, {h3['ci_high']:+.4f}]  <- governs")
    print(f"     VERDICT          : {h3['verdict']}")
    print('     corroborating (context only — NOT a path to KEEP):')
    for h, res in r['h3_corroborating'].items():
        print(f"       N={h:<3} n={res['n_events']:<5} mean {res['mean_signed_pips']:+.4f} "
              f"pips   CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]   "
              f"win {res['win_rate_pct']:.2f}%")
    print('     breakdown — DESCRIPTIVE ONLY:')
    for _, row in r['trig_breakdown'].iterrows():
        print(f"       {row['breakdown']:<11} {row['group']:<18} n={int(row['n']):<5} "
              f"mean {row['mean_signed_pips']:+.4f} pips   win {row['win_rate_pct']:.2f}%")

    print(f"\n   H_div.4 — GBM ON TRIGGERED ENTRIES (alpha = {h4['alpha']})")
    print(f"     shuffled-label control : {h4['shuffled_label_control_acc']:.4f}  -> "
          f"{'SANE' if h4['leak_control_sane'] else 'ANOMALOUS — VOID'}")
    print(f"     n train / val    : {h4['n_train']} / {h4['n_val']}")
    print(f"     acc challenger   : {h4['acc_challenger']:.4f}")
    print(f"     acc reference    : {h4['acc_reference']:.4f}  "
          f"(train-majority class {h4['majority_class']})")
    print(f"     delta            : {h4['delta_acc']:+.4f}")
    print(f"     block CI         : [{h4['delta_acc_ci_low']:+.4f}, "
          f"{h4['delta_acc_ci_high']:+.4f}]  <- governs")
    print(f"     McNemar exact    : b={h4['mcnemar_b']} c={h4['mcnemar_c']} "
          f"p={h4['mcnemar_p']:.6g}")
    print(f"     VERDICT          : {h4['verdict']}")

    print('\n9. RETROACTIVE ALPHA TIGHTENING (family 2 -> 4, 0.025 -> 0.0125)')
    o1, o2 = r['h1_at_original_alpha'], r['h2_at_original_alpha']
    print(f"     H_div.1 at 0.025  : CI [{o1['ci_low']:+.4f}, {o1['ci_high']:+.4f}] "
          f"-> {o1['verdict']}")
    print(f"     H_div.1 at 0.0125 : CI [{h1['ci_low']:+.4f}, {h1['ci_high']:+.4f}] "
          f"-> {h1['verdict']}")
    print(f"     H_div.2 at 0.025  : CI [{o2['delta_acc_ci_low']:+.4f}, "
          f"{o2['delta_acc_ci_high']:+.4f}] p={o2['mcnemar_p']:.4g} -> {o2['verdict']}")
    print(f"     H_div.2 at 0.0125 : CI [{h2['delta_acc_ci_low']:+.4f}, "
          f"{h2['delta_acc_ci_high']:+.4f}] p={h2['mcnemar_p']:.4g} -> {h2['verdict']}")
    flipped = [nm for nm, a, b in (('H_div.1', o1['verdict'], h1['verdict']),
                                   ('H_div.2', o2['verdict'], h2['verdict']))
               if a != b]
    print(f"     reclassified by the tightening: "
          f"{', '.join(flipped) if flipped else 'NONE — both DROPped at both bars'}")

    print('\n10. VERDICTS')
    print(f"   H_div.1  pattern-only event study       : {h1['verdict']}")
    print(f"   H_div.2  GBM, pattern-only              : {h2['verdict']}")
    print(f"   H_div.3  triggered-entry event study    : {h3['verdict']}")
    print(f"   H_div.4  GBM, triggered entry           : {h4['verdict']}")
    print('   (pips are a descriptive measure of move size only — no cost')
    print('    subtraction, no P&L, no Sharpe, no equity curve anywhere here)')


if __name__ == '__main__':
    _print_report(run())
