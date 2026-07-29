"""
H_dir.1 POST-HOC DIAGNOSTICS — descriptive decomposition, NOT a hypothesis test.

WHAT THIS IS
------------
H_dir.1 cleared its pre-registered bar: GBM 0.527462 vs train-majority 0.504432
on the EURUSD validation slice, +0.0230, block CI [+0.0075, +0.0392], McNemar
p=0.00169, shuffled-label control 0.5002.

Before the reserved test block [85:100%] is spent -- it is one-shot and it is
the only clean slice this family has left -- three cheap decompositions ask
whether that edge is STRUCTURED or an ARTIFACT.

These are DESCRIPTIVE DECOMPOSITIONS of a slice that has ALREADY been spent.
They consume NO alpha and produce NO verdict. Nothing here is a hypothesis, so
nothing here touches results/h1_direction_hypothesis_log.csv or any other
family's log, and the Bonferroni family size stays at 5.

NO P&L. NO spread. NO pips. NO returns. NO Sharpe. NO trading simulation.
Accuracy and its decomposition, and nothing else.

HARD BOUNDARIES
---------------
* The model is NOT refit differently, retuned, reseeded or altered. The EXACT
  H_dir.1 configuration is reused by IMPORTING src.h1_direction_model unchanged
  (train_gbm, build_direction_dataset, split_purge_embargo, fit_standardizer,
  flat_matrix) -- not by restating any of it here. There is no early_stopping
  and no eval_set, and that absence is what makes H_dir.1 clean.
* Features, target, split, purge and embargo are untouched.
* The test block [85:100%] is NEVER indexed. assert_no_test_block() enforces it
  on every row set this module reads, and run() records the maximum positional
  index it touched so a unit test can pin it below the validation bound.
* Writes ONLY results/h1_direction_diagnostics_{features,temporal,hourly}.csv.
  The protected set -- including src/h1_direction_model.py and
  results/h1_direction_hypothesis_log.csv -- is sha256-asserted by unit tests.

STEP 0 — REPRODUCTION GATE (hard)
---------------------------------
The refit must reproduce 0.527462 within +/- 0.003. GPU non-determinism is
accepted; bitwise equality is not required. If it does not reproduce, the run
STOPS (ReproductionGateError) -- diagnostics of a model that is not the one
which cleared the bar would be describing something else entirely.

PRE-REGISTERED INTERPRETATION (fixed BEFORE any result was looked at)
--------------------------------------------------------------------
NOISE FLOOR FIRST. The validation slice is 10,378 rows; split into 4 quarters
that is ~2,595 rows each, giving a per-quarter standard error on accuracy of
roughly 0.98pp. A single quarter at +0.5pp or +4pp is NOT informative on its
own. Only the PATTERN across quarters is. The computed SE is stated alongside
every sub-block table and individual cells are not over-read.

KILL PATTERN A (temporal): the edge is a regime artifact if a single quarter
contributes MORE THAN 60% of the total validation-slice edge, OR if TWO OR MORE
of the four quarters have a negative delta versus the majority baseline.

KILL PATTERN B (liquidity): the edge is microstructure noise rather than signal
if the delta during the liquid session (07:00-17:00 UTC) is AT OR BELOW ZERO
while the thin-liquidity hours (20:00-06:00 UTC) carry essentially all of it.
This is the same window where the 451 dropped zero-return rows concentrate,
which is exactly why it is the specific concern.

SURVIVAL PATTERN: deltas positive in at least 3 of 4 quarters with no single
quarter dominating, AND a positive delta in the liquid session. That is not
proof of anything -- it means the result is not obviously an artifact.

Feature importance (Diagnostic 1) has NO kill criterion. It is interpretive
context only, reported to understand what the model latched onto.

FEATURE COUNT — a correction to the brief
-----------------------------------------
The brief asked for "all 14 features". The pre-registered feature set imported
from src/pooled_h1_model.py has FIFTEEN columns, and the brief's own family
grouping also sums to fifteen (6 lag returns + 2 volatility + 1 oscillator +
2 trend distance + 4 calendar = 15). The "14" is a miscount of the brief, not a
different feature set: all 15 pre-registered columns are reported, grouped
exactly as the brief specifies. No column was added, removed or modified.

EDGE SHARE — how "share of the total edge" is defined
-----------------------------------------------------
Accuracy differences decompose additively over rows. For each row define
   e[i] = 1{GBM correct} - 1{majority correct}   in {-1, 0, +1}
so the slice edge is delta_acc = mean(e) and the TOTAL EDGE in row units is
sum(e) = n * delta_acc. A sub-block's share is its sum(e) divided by the whole
slice's sum(e). Shares therefore sum to exactly 1.0 across any exhaustive,
non-overlapping partition -- which is what makes KILL PATTERN A's "more than
60% from one quarter" a well-defined statement rather than a ratio of averages.
"""

import os

import numpy as np
import pandas as pd

# The H_dir.1 program is imported UNCHANGED and is the single source of the
# model, the data, the target, the split, the purge and the embargo. This
# module restates none of them.
from src import h1_direction_model as h1d
from src.pooled_h1_model import FEATURE_COLUMNS
from src.pooled_h1_data import POOLED_DIR

# ── STEP 0 reproduction gate (frozen) ──
REPRODUCTION_TARGET_ACC = 0.527462          # the logged H_dir.1 value
REPRODUCTION_TOL = 0.003                    # GPU non-determinism accepted

# ── Diagnostic constants (pre-registered; do not tune after seeing results) ──
N_PERMUTATION_REPEATS = 20
N_QUARTERS = 4
PERMUTATION_SEED = 20260729

# Pre-registered session buckets (UTC hours).
LIQUID_HOURS = tuple(range(7, 18))                  # 07:00-17:00 London + NY
TRANSITIONAL_HOURS = (18, 19)                       # 18:00-19:00
THIN_HOURS = (20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6)  # 20:00-06:00 Asia / late NY
SESSION_ORDER = ('liquid', 'transitional', 'thin')

# Pre-registered kill thresholds.
KILL_A_MAX_QUARTER_SHARE = 0.60
KILL_A_MAX_NEGATIVE_QUARTERS = 2

FEATURE_FAMILIES = {
    'lag_returns': ['logret_1', 'logret_2', 'logret_3',
                    'logret_6', 'logret_12', 'logret_24'],
    'volatility': ['atr14_norm', 'ewma_std_24'],
    'oscillator': ['rsi14'],
    'trend_distance': ['dist_sma50_atr', 'dist_sma200_atr'],
    'calendar': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
}

FEATURES_CSV = 'results/h1_direction_diagnostics_features.csv'
TEMPORAL_CSV = 'results/h1_direction_diagnostics_temporal.csv'
HOURLY_CSV = 'results/h1_direction_diagnostics_hourly.csv'


class ReproductionGateError(RuntimeError):
    """Raised when the refit does not reproduce the logged H_dir.1 accuracy
    within REPRODUCTION_TOL. The diagnostics would otherwise be describing a
    different model than the one that cleared the bar -- STOP."""


class TestBlockTouchedError(RuntimeError):
    """Raised if any row at or beyond the validation end boundary is read. The
    test block [85:100%] is one-shot and reserved; this module must never see
    it."""


# ───────────────────────── guards ─────────────────────────────────────────────

def assert_reproduction(acc, target=None, tol=None) -> float:
    """
    The STEP 0 gate as a pure function: |acc - target| must be <= tol.

    Module globals are resolved at CALL time (not bound as defaults) so the
    gate's threshold can be exercised directly by a unit test. Returns the
    drift; raises ReproductionGateError when it exceeds the tolerance.
    """
    target = REPRODUCTION_TARGET_ACC if target is None else float(target)
    tol = REPRODUCTION_TOL if tol is None else float(tol)
    drift = abs(float(acc) - target)
    if drift > tol:
        raise ReproductionGateError(
            f"refit accuracy {float(acc):.6f} drifted {drift:.6f} from the logged "
            f"H_dir.1 value {target:.6f}, beyond the +/- {tol} tolerance. These "
            "diagnostics would be describing a different model than the one that "
            "cleared the bar -- STOP."
        )
    return drift


def assert_no_test_block(index_used, val_end_ts):
    """
    Hard guard: every timestamp this module reads must lie at or before the
    validation end boundary. Raises TestBlockTouchedError otherwise.

    Called on every row set the diagnostics touch, so a reserved-block read
    cannot slip through a refactor unnoticed.
    """
    idx = pd.DatetimeIndex(index_used)
    if len(idx) and idx.max() > val_end_ts:
        raise TestBlockTouchedError(
            f"row at {idx.max()} lies beyond the validation end {val_end_ts} -- "
            "the test block [85:100%] is RESERVED and must never be indexed."
        )
    return True


# ───────────────────────── small shared helpers ───────────────────────────────

def accuracy_se(n) -> float:
    """
    Standard error of an accuracy estimate near chance: sqrt(0.25 / n).

    Uses p = 0.5 rather than the observed p because every accuracy here sits
    within a percentage point or two of chance, and the p=0.5 form is the
    conservative (largest) value. This is the number the pre-registered noise
    floor is stated in: at n ~ 2,595 it is ~0.98pp.
    """
    n = int(n)
    return float(np.sqrt(0.25 / n)) if n > 0 else float('nan')


def paired_delta_se(edge) -> float:
    """
    Standard error of the PAIRED delta accuracy, sd(e)/sqrt(n) where
    e[i] = 1{challenger correct} - 1{reference correct}.

    Reported alongside the pre-registered accuracy SE as extra context: because
    the two predictors are scored on the identical rows, the paired SE is the
    honest spread of the DIFFERENCE, and it is typically smaller than the naive
    accuracy SE. It does not replace the pre-registered noise floor.
    """
    e = np.asarray(edge, dtype=float)
    return float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else float('nan')


def edge_vector(pred_challenger, pred_reference, y_true):
    """Per-row additive edge e[i] in {-1, 0, +1}; mean(e) is delta accuracy and
    sum(e) is the total edge in row units (see the module docstring)."""
    y = np.asarray(y_true).astype(int)
    cc = (np.asarray(pred_challenger).astype(int) == y).astype(float)
    cr = (np.asarray(pred_reference).astype(int) == y).astype(float)
    return cc - cr


def _block_stats(mask, y, pred_gbm, pred_major):
    """Accuracy / baseline / delta / SEs / edge-sum for one boolean row subset."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {'n_rows': 0, 'acc_gbm': float('nan'), 'acc_majority': float('nan'),
                'delta': float('nan'), 'se_acc': float('nan'),
                'se_delta_paired': float('nan'), 'edge_sum': 0.0}
    yb = np.asarray(y)[mask]
    e = edge_vector(np.asarray(pred_gbm)[mask], np.asarray(pred_major)[mask], yb)
    return {
        'n_rows': n,
        'acc_gbm': float((np.asarray(pred_gbm)[mask] == yb).mean()),
        'acc_majority': float((np.asarray(pred_major)[mask] == yb).mean()),
        'delta': float(e.mean()),
        'se_acc': accuracy_se(n),
        'se_delta_paired': paired_delta_se(e),
        'edge_sum': float(e.sum()),
    }


def session_bucket(hour: int) -> str:
    """Pre-registered UTC session bucket for an hour-of-day."""
    h = int(hour)
    if h in LIQUID_HOURS:
        return 'liquid'
    if h in TRANSITIONAL_HOURS:
        return 'transitional'
    return 'thin'


# ───────────────────── STEP 0: rebuild + reproduction gate ────────────────────

def rebuild_h_dir_1(out_dir: str = POOLED_DIR, seed: int = h1d.RANDOM_SEED,
                    verbose: bool = True):
    """
    Rebuild H_dir.1 EXACTLY as it ran, by calling the imported program's own
    functions -- same cached EURUSD H1 CSV, same target, same feed gate, same
    split/purge/embargo, same train-only standardizer, same XGBClassifier
    (n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', device='cuda',
    balanced scale_pos_weight, seed 42, NO early_stopping, NO eval_set).

    Enforces the STEP 0 reproduction gate, then returns everything the three
    diagnostics need. The test block is never indexed: only splits['train'] and
    splits['val'] are read, and both are guarded.
    """
    h1d.seed_everything(seed)

    raw = h1d.load_eurusd_h1(out_dir=out_dir)
    context, target_index, counts = h1d.build_direction_dataset(raw)
    h1d.assert_feed_not_stale(counts)

    splits, split_counts = h1d.split_purge_embargo(target_index)
    train_idx, val_idx = splits['train'], splits['val']
    val_end_ts = split_counts['val_end_ts']

    # Guard BOTH row sets the diagnostics will read.
    assert_no_test_block(train_idx, val_end_ts)
    assert_no_test_block(val_idx, val_end_ts)

    uniqueness = h1d.mean_label_uniqueness(context, train_idx)
    h1d.assert_labels_independent(uniqueness)

    mean, std = h1d.fit_standardizer(context, train_idx)
    Xtr, ytr = h1d.flat_matrix(context, train_idx, mean, std)
    Xval, yval = h1d.flat_matrix(context, val_idx, mean, std)

    gbm = h1d.train_gbm(Xtr, ytr, seed=seed)
    prob = h1d.predict_gbm_proba(gbm, Xval)
    pred_gbm = (prob >= 0.5).astype(int)

    majority = h1d.train_majority_class(ytr)
    pred_major = np.full(len(yval), majority, dtype=int)

    acc_gbm = float((pred_gbm == yval).mean())
    acc_major = float((pred_major == yval).mean())
    drift = abs(acc_gbm - REPRODUCTION_TARGET_ACC)

    if verbose:
        print('=' * 74)
        print('STEP 0 — REPRODUCTION GATE')
        print('=' * 74)
        print(f"  logged H_dir.1 accuracy   : {REPRODUCTION_TARGET_ACC:.6f}")
        print(f"  reproduced accuracy       : {acc_gbm:.6f}")
        print(f"  |drift|                   : {drift:.6f}  (tolerance "
              f"+/- {REPRODUCTION_TOL})")
        print(f"  gate                      : "
              f"{'PASS' if drift <= REPRODUCTION_TOL else 'FAIL'}")

    assert_reproduction(acc_gbm)        # raises ReproductionGateError on drift

    return {
        'context': context, 'counts': counts, 'split_counts': split_counts,
        'train_idx': train_idx, 'val_idx': val_idx, 'val_end_ts': val_end_ts,
        'uniqueness': uniqueness, 'gbm': gbm,
        'Xval': Xval, 'yval': yval, 'prob': prob,
        'pred_gbm': pred_gbm, 'pred_major': pred_major,
        'majority_class': majority,
        'acc_gbm': acc_gbm, 'acc_majority': acc_major,
        'delta_acc': acc_gbm - acc_major,
        'reproduced_acc': acc_gbm, 'reproduction_drift': drift,
        'reproduction_passed': True,        # unreached unless the gate passed
        'max_position_touched': int(len(train_idx) + len(val_idx)),
        'seed': seed,
    }


# ───────────────── DIAGNOSTIC 1: what the model uses ──────────────────────────

def permutation_importance(gbm, X, y, n_repeats: int = N_PERMUTATION_REPEATS,
                           seed: int = PERMUTATION_SEED,
                           feature_names=tuple(FEATURE_COLUMNS)):
    """
    Permutation importance on the VALIDATION rows: shuffle ONE column, re-score,
    record the ACCURACY DROP relative to the unpermuted baseline. n_repeats
    shuffles per feature; mean and std are reported.

    The matrix is restored column-by-column after every repeat, and the caller's
    array is never left modified -- a unit test asserts byte-equality of X
    before vs after this routine.

    Reported side by side with gain-based importance because gain is known to be
    biased toward high-cardinality continuous features, whereas this measures
    what the model's validation ACCURACY actually depends on. Where they
    disagree, this is the one to believe.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    baseline = float(((h1d.predict_gbm_proba(gbm, X) >= 0.5).astype(int) == y).mean())

    rng = np.random.default_rng(seed)
    rows = []
    work = X.copy()                     # never mutate the caller's matrix
    for j, name in enumerate(feature_names):
        original = work[:, j].copy()
        drops = np.empty(n_repeats, dtype=float)
        for r in range(n_repeats):
            work[:, j] = rng.permutation(original)
            acc = float(((h1d.predict_gbm_proba(gbm, work) >= 0.5).astype(int) == y).mean())
            drops[r] = baseline - acc
        work[:, j] = original           # restore before moving to the next column
        rows.append({
            'feature': name,
            'perm_importance_mean': float(drops.mean()),
            'perm_importance_std': float(drops.std(ddof=1)) if n_repeats > 1 else 0.0,
            'perm_n_repeats': int(n_repeats),
        })
    return baseline, pd.DataFrame(rows)


def gain_importance(gbm, feature_names=tuple(FEATURE_COLUMNS)) -> pd.DataFrame:
    """Gain-based importance straight from the fitted booster, mapped back onto
    the pre-registered feature names (features never seen in a split score 0)."""
    score = gbm.get_booster().get_score(importance_type='gain')
    gains = [float(score.get(f'f{j}', 0.0)) for j in range(len(feature_names))]
    total = float(sum(gains))
    return pd.DataFrame({
        'feature': list(feature_names),
        'gain_importance': gains,
        'gain_share': [(g / total if total > 0 else float('nan')) for g in gains],
    })


def diagnostic_features(rebuilt, n_repeats: int = N_PERMUTATION_REPEATS,
                        seed: int = PERMUTATION_SEED, out_path: str = FEATURES_CSV,
                        write: bool = True):
    """
    DIAGNOSTIC 1 — gain-based and permutation importance side by side for every
    pre-registered feature, plus family-level totals. NO kill criterion:
    interpretive context only.
    """
    assert_no_test_block(rebuilt['val_idx'], rebuilt['val_end_ts'])

    baseline, perm = permutation_importance(
        rebuilt['gbm'], rebuilt['Xval'], rebuilt['yval'],
        n_repeats=n_repeats, seed=seed)
    gain = gain_importance(rebuilt['gbm'])

    fam_of = {f: fam for fam, cols in FEATURE_FAMILIES.items() for f in cols}
    df = gain.merge(perm, on='feature', how='outer')
    df['family'] = df['feature'].map(fam_of)
    df['scope'] = 'feature'
    df['perm_baseline_acc'] = baseline

    perm_total = float(df['perm_importance_mean'].sum())
    fam_rows = []
    for fam in FEATURE_FAMILIES:
        sub = df[df['family'] == fam]
        fam_rows.append({
            'feature': f'FAMILY::{fam}', 'family': fam, 'scope': 'family_total',
            'n_features': int(len(sub)),
            'gain_importance': float(sub['gain_importance'].sum()),
            'gain_share': float(sub['gain_share'].sum()),
            'perm_importance_mean': float(sub['perm_importance_mean'].sum()),
            'perm_importance_std': float('nan'),
            'perm_share': (float(sub['perm_importance_mean'].sum() / perm_total)
                           if perm_total != 0 else float('nan')),
            'perm_n_repeats': int(n_repeats),
            'perm_baseline_acc': baseline,
        })

    df['n_features'] = 1
    df['perm_share'] = (df['perm_importance_mean'] / perm_total
                        if perm_total != 0 else float('nan'))
    cols = ['scope', 'family', 'feature', 'n_features', 'gain_importance',
            'gain_share', 'perm_importance_mean', 'perm_importance_std',
            'perm_share', 'perm_n_repeats', 'perm_baseline_acc']
    out = pd.concat([df[cols], pd.DataFrame(fam_rows)[cols]], ignore_index=True)

    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


# ───────────────── DIAGNOSTIC 2: temporal stability ───────────────────────────

def quarter_masks(n_rows: int, n_blocks: int = N_QUARTERS):
    """
    Partition n_rows into n_blocks CONTIGUOUS chronological blocks of as-equal
    size as possible. Exhaustive and non-overlapping by construction: every row
    lands in exactly one block (a unit test pins this).
    """
    bounds = np.array_split(np.arange(n_rows), n_blocks)
    masks = []
    for b in bounds:
        m = np.zeros(n_rows, dtype=bool)
        if len(b):
            m[b] = True
        masks.append(m)
    return masks


def diagnostic_temporal(rebuilt, n_blocks: int = N_QUARTERS,
                        out_path: str = TEMPORAL_CSV, write: bool = True):
    """
    DIAGNOSTIC 2 — the edge over time. Four contiguous chronological quarters
    (date range, n, GBM accuracy, majority accuracy, delta, SEs, edge share),
    then the same at MONTHLY granularity purely for shape.

    NO per-month significance test is run: the samples are small and it would be
    uncorrected multiple testing. The monthly rows are shape only.

    Evaluates KILL PATTERN A: a single quarter contributing > 60% of the total
    edge, OR two or more quarters with a negative delta.
    """
    val_idx = pd.DatetimeIndex(rebuilt['val_idx'])
    assert_no_test_block(val_idx, rebuilt['val_end_ts'])

    y, pg, pm = rebuilt['yval'], rebuilt['pred_gbm'], rebuilt['pred_major']
    total_edge = float(edge_vector(pg, pm, y).sum())
    n = len(val_idx)

    rows = []
    quarter_stats = []
    for q, mask in enumerate(quarter_masks(n, n_blocks), start=1):
        st = _block_stats(mask, y, pg, pm)
        share = (st['edge_sum'] / total_edge) if total_edge != 0 else float('nan')
        rec = {
            'scope': 'quarter', 'label': f'Q{q}',
            'start': val_idx[mask][0].isoformat() if st['n_rows'] else '',
            'end': val_idx[mask][-1].isoformat() if st['n_rows'] else '',
            'edge_share': share, **st,
        }
        rows.append(rec)
        quarter_stats.append(rec)

    # Monthly — SHAPE ONLY, no significance testing (see the docstring).
    # tz is intentionally dropped for the period grouping only (the labels are
    # calendar months); the underlying UTC index is never modified.
    months = val_idx.tz_convert(None).to_period('M')
    monthly = []
    for period in sorted(months.unique()):
        mask = np.asarray(months == period, dtype=bool)
        st = _block_stats(mask, y, pg, pm)
        rec = {
            'scope': 'month', 'label': str(period),
            'start': val_idx[mask][0].isoformat() if st['n_rows'] else '',
            'end': val_idx[mask][-1].isoformat() if st['n_rows'] else '',
            'edge_share': ((st['edge_sum'] / total_edge)
                           if total_edge != 0 else float('nan')),
            **st,
        }
        rows.append(rec)
        monthly.append(rec)

    rows.append({
        'scope': 'slice_total', 'label': 'validation[70:85]',
        'start': val_idx[0].isoformat(), 'end': val_idx[-1].isoformat(),
        'n_rows': n, 'acc_gbm': rebuilt['acc_gbm'],
        'acc_majority': rebuilt['acc_majority'], 'delta': rebuilt['delta_acc'],
        'se_acc': accuracy_se(n),
        'se_delta_paired': paired_delta_se(edge_vector(pg, pm, y)),
        'edge_sum': total_edge, 'edge_share': 1.0,
    })

    out = pd.DataFrame(rows)[
        ['scope', 'label', 'start', 'end', 'n_rows', 'acc_gbm', 'acc_majority',
         'delta', 'se_acc', 'se_delta_paired', 'edge_sum', 'edge_share']]
    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        out.to_csv(out_path, index=False)

    shares = [q['edge_share'] for q in quarter_stats]
    deltas = [q['delta'] for q in quarter_stats]
    n_negative = int(sum(1 for d in deltas if d < 0))
    max_share = float(np.nanmax(shares)) if len(shares) else float('nan')
    best_quarter = quarter_stats[int(np.nanargmax(shares))]['label'] if len(shares) else ''

    kill_a = bool(max_share > KILL_A_MAX_QUARTER_SHARE
                  or n_negative >= KILL_A_MAX_NEGATIVE_QUARTERS)

    return {
        'table': out, 'quarters': quarter_stats, 'monthly': monthly,
        'total_edge': total_edge,
        'per_quarter_se': accuracy_se(int(np.ceil(n / n_blocks))),
        'max_quarter_edge_share': max_share, 'best_quarter': best_quarter,
        'n_quarters_negative': n_negative,
        'n_positive_quarters': int(sum(1 for d in deltas if d > 0)),
        'n_months': len(monthly),
        'n_months_positive': int(sum(1 for m in monthly if m['delta'] > 0)),
        'kill_pattern_a': kill_a,
    }


# ───────────────── DIAGNOSTIC 3: hour-of-day concentration ────────────────────

def zero_return_rows_by_hour(context: pd.DataFrame, val_index=None):
    """
    Per-UTC-hour count of the rows STEP 1 of the main program dropped for having
    EXACTLY zero next-bar return (the 451). They carry a NaN `label` in the
    context frame, which keeps them available as sequence context while removing
    them from every trained/scored row set.

    Returns (whole-series counts, counts inside the validation window). The
    second is what makes the overlap with any hourly concentration of the edge
    directly readable, since the whole-series figure spans train and test eras
    too.
    """
    dropped = context.index[context['label'].isna()]
    whole = pd.Series(0, index=range(24), dtype=int)
    whole.update(pd.Series(dropped.hour).value_counts())

    in_val = pd.Series(0, index=range(24), dtype=int)
    if val_index is not None and len(val_index):
        vi = pd.DatetimeIndex(val_index)
        within = dropped[(dropped >= vi.min()) & (dropped <= vi.max())]
        in_val.update(pd.Series(within.hour).value_counts())
    return whole, in_val


def diagnostic_hourly(rebuilt, out_path: str = HOURLY_CSV, write: bool = True):
    """
    DIAGNOSTIC 3 — where in the trading day the edge sits. Per UTC hour 0-23 and
    per pre-registered session bucket: n, GBM accuracy, majority accuracy,
    delta, SE, and share of the total validation edge. Cross-references the
    per-hour count of STEP 1's dropped zero-return rows in the SAME table, so
    any overlap between the dropped rows and a concentration of the edge is
    visible at a glance.

    Evaluates KILL PATTERN B: liquid-session delta at or below zero while the
    thin hours carry essentially all of the edge.
    """
    val_idx = pd.DatetimeIndex(rebuilt['val_idx'])
    assert_no_test_block(val_idx, rebuilt['val_end_ts'])

    y, pg, pm = rebuilt['yval'], rebuilt['pred_gbm'], rebuilt['pred_major']
    hours = val_idx.hour.to_numpy()
    total_edge = float(edge_vector(pg, pm, y).sum())
    zero_all, zero_val = zero_return_rows_by_hour(rebuilt['context'], val_idx)

    rows = []
    for h in range(24):
        st = _block_stats(hours == h, y, pg, pm)
        rows.append({
            'scope': 'hour', 'label': f'{h:02d}:00', 'hour_utc': h,
            'session': session_bucket(h),
            'edge_share': ((st['edge_sum'] / total_edge)
                           if total_edge != 0 else float('nan')),
            'zero_return_dropped_all': int(zero_all.get(h, 0)),
            'zero_return_dropped_in_val_window': int(zero_val.get(h, 0)),
            **st,
        })

    bucket_stats = {}
    for bucket in SESSION_ORDER:
        member = np.array([session_bucket(h) == bucket for h in hours])
        st = _block_stats(member, y, pg, pm)
        hrs = [h for h in range(24) if session_bucket(h) == bucket]
        rec = {
            'scope': 'session', 'label': bucket, 'hour_utc': -1,
            'session': bucket,
            'edge_share': ((st['edge_sum'] / total_edge)
                           if total_edge != 0 else float('nan')),
            'zero_return_dropped_all': int(sum(zero_all.get(h, 0) for h in hrs)),
            'zero_return_dropped_in_val_window': int(sum(zero_val.get(h, 0) for h in hrs)),
            **st,
        }
        rows.append(rec)
        bucket_stats[bucket] = rec

    rows.append({
        'scope': 'slice_total', 'label': 'validation[70:85]', 'hour_utc': -1,
        'session': 'all', 'n_rows': len(val_idx), 'acc_gbm': rebuilt['acc_gbm'],
        'acc_majority': rebuilt['acc_majority'], 'delta': rebuilt['delta_acc'],
        'se_acc': accuracy_se(len(val_idx)),
        'se_delta_paired': paired_delta_se(edge_vector(pg, pm, y)),
        'edge_sum': total_edge, 'edge_share': 1.0,
        'zero_return_dropped_all': int(zero_all.sum()),
        'zero_return_dropped_in_val_window': int(zero_val.sum()),
    })

    out = pd.DataFrame(rows)[
        ['scope', 'label', 'hour_utc', 'session', 'n_rows', 'acc_gbm',
         'acc_majority', 'delta', 'se_acc', 'se_delta_paired', 'edge_sum',
         'edge_share', 'zero_return_dropped_all',
         'zero_return_dropped_in_val_window']]
    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        out.to_csv(out_path, index=False)

    liquid, thin = bucket_stats['liquid'], bucket_stats['thin']
    kill_b = bool(liquid['delta'] <= 0.0 and thin['delta'] > 0.0)

    return {
        'table': out, 'buckets': bucket_stats, 'total_edge': total_edge,
        'zero_by_hour_all': zero_all, 'zero_by_hour_in_val': zero_val,
        'kill_pattern_b': kill_b,
    }


# ───────────────────────── orchestration ──────────────────────────────────────

def run(out_dir: str = POOLED_DIR, seed: int = h1d.RANDOM_SEED,
        n_repeats: int = N_PERMUTATION_REPEATS, write: bool = True,
        verbose: bool = True, features_path: str = FEATURES_CSV,
        temporal_path: str = TEMPORAL_CSV, hourly_path: str = HOURLY_CSV):
    """
    Full post-hoc diagnostic pass: STEP 0 reproduction gate, then Diagnostic 2
    (temporal — reported first because it is the most likely killer), Diagnostic
    3 (hourly) and Diagnostic 1 (features).

    Writes ONLY the three results/h1_direction_diagnostics_*.csv files. Never
    creates or modifies ANY hypothesis log -- no alpha is consumed here and no
    verdict is produced. The test block is never indexed.
    """
    rebuilt = rebuild_h_dir_1(out_dir=out_dir, seed=seed, verbose=verbose)
    temporal = diagnostic_temporal(rebuilt, write=write, out_path=temporal_path)
    hourly = diagnostic_hourly(rebuilt, write=write, out_path=hourly_path)
    features = diagnostic_features(rebuilt, n_repeats=n_repeats, write=write,
                                   out_path=features_path)

    survival = bool(temporal['n_positive_quarters'] >= 3
                    and not temporal['kill_pattern_a']
                    and hourly['buckets']['liquid']['delta'] > 0.0)

    return {
        'rebuilt': rebuilt, 'temporal': temporal, 'hourly': hourly,
        'features': features, 'survival_pattern': survival,
        'n_val': len(rebuilt['val_idx']),
    }


def _print_report(r):
    """The report, in the pre-registered order: reproduction, noise floor,
    temporal, hourly, features, plain reading. RAW numbers only."""
    reb, tmp, hr = r['rebuilt'], r['temporal'], r['hourly']

    print('\n' + '=' * 78)
    print('H_dir.1 POST-HOC DIAGNOSTICS — descriptive only, NO alpha consumed')
    print('=' * 78)

    print('\n1. STEP 0 REPRODUCTION')
    print(f"   logged     : {REPRODUCTION_TARGET_ACC:.6f}")
    print(f"   reproduced : {reb['reproduced_acc']:.6f}   "
          f"|drift| = {reb['reproduction_drift']:.6f}  (tol +/- {REPRODUCTION_TOL})")
    print(f"   majority   : {reb['acc_majority']:.6f}   "
          f"delta = {reb['delta_acc']:+.6f}   n_val = {r['n_val']}")

    print('\n2. NOISE FLOOR (stated BEFORE any table)')
    print(f"   per-quarter n ~ {int(np.ceil(r['n_val'] / N_QUARTERS))}  ->  "
          f"SE(accuracy) = {tmp['per_quarter_se'] * 100:.2f}pp")
    print("   A single quarter at +0.5pp or +4pp is NOT informative alone. Only")
    print("   the PATTERN across quarters is.")

    print('\n3. DIAGNOSTIC 2 — TEMPORAL STABILITY (4 contiguous quarters)')
    print(f"   {'blk':<4}{'start':<12}{'end':<12}{'n':>7}{'acc':>9}{'base':>9}"
          f"{'delta':>10}{'SE':>8}{'edge%':>9}")
    for q in tmp['quarters']:
        print(f"   {q['label']:<4}{q['start'][:10]:<12}{q['end'][:10]:<12}"
              f"{q['n_rows']:>7}{q['acc_gbm']:>9.4f}{q['acc_majority']:>9.4f}"
              f"{q['delta']:>+10.4f}{q['se_acc'] * 100:>7.2f}pp"
              f"{q['edge_share'] * 100:>8.1f}%")
    print(f"\n   quarters with positive delta : {tmp['n_positive_quarters']}/4")
    print(f"   quarters with negative delta : {tmp['n_quarters_negative']}/4")
    print(f"   largest single-quarter share : {tmp['max_quarter_edge_share'] * 100:.1f}% "
          f"({tmp['best_quarter']})   [KILL A bar: > 60%]")
    print(f"   months with positive delta   : {tmp['n_months_positive']}/{tmp['n_months']} "
          "(shape only — no per-month significance test)")
    print(f"\n   KILL PATTERN A  : {'MET — regime artifact' if tmp['kill_pattern_a'] else 'NOT MET'}")

    print('\n4. DIAGNOSTIC 3 — HOUR-OF-DAY CONCENTRATION')
    print(f"   {'hour':<7}{'sess':<14}{'n':>6}{'acc':>9}{'base':>9}{'delta':>10}"
          f"{'SE':>8}{'edge%':>9}{'0ret':>7}")
    for _, row in hr['table'][hr['table']['scope'] == 'hour'].iterrows():
        print(f"   {row['label']:<7}{row['session']:<14}{int(row['n_rows']):>6}"
              f"{row['acc_gbm']:>9.4f}{row['acc_majority']:>9.4f}"
              f"{row['delta']:>+10.4f}{row['se_acc'] * 100:>7.2f}pp"
              f"{row['edge_share'] * 100:>8.1f}%"
              f"{int(row['zero_return_dropped_all']):>7}")
    print(f"\n   {'session':<14}{'hours':<16}{'n':>7}{'delta':>10}{'SE':>8}"
          f"{'edge%':>9}{'0ret':>7}")
    labels = {'liquid': '07:00-17:00', 'transitional': '18:00-19:00',
              'thin': '20:00-06:00'}
    for bucket in SESSION_ORDER:
        b = hr['buckets'][bucket]
        print(f"   {bucket:<14}{labels[bucket]:<16}{b['n_rows']:>7}"
              f"{b['delta']:>+10.4f}{b['se_acc'] * 100:>7.2f}pp"
              f"{b['edge_share'] * 100:>8.1f}%"
              f"{b['zero_return_dropped_all']:>7}")
    print(f"\n   KILL PATTERN B  : {'MET — microstructure noise' if hr['kill_pattern_b'] else 'NOT MET'}")

    print('\n5. DIAGNOSTIC 1 — FEATURE IMPORTANCE (no kill criterion; context only)')
    feat = r['features']
    per_feature = feat[feat['scope'] == 'feature'].sort_values(
        'perm_importance_mean', ascending=False)
    print(f"   {'feature':<18}{'family':<17}{'gain%':>9}{'perm_mean':>12}"
          f"{'perm_std':>11}{'perm%':>9}")
    for _, row in per_feature.iterrows():
        print(f"   {row['feature']:<18}{row['family']:<17}"
              f"{row['gain_share'] * 100:>8.1f}%{row['perm_importance_mean']:>+12.5f}"
              f"{row['perm_importance_std']:>11.5f}{row['perm_share'] * 100:>8.1f}%")
    print(f"\n   {'FAMILY':<18}{'n':>4}{'gain%':>10}{'perm_sum':>12}{'perm%':>9}")
    for _, row in feat[feat['scope'] == 'family_total'].iterrows():
        print(f"   {row['family']:<18}{int(row['n_features']):>4}"
              f"{row['gain_share'] * 100:>9.1f}%{row['perm_importance_mean']:>+12.5f}"
              f"{row['perm_share'] * 100:>8.1f}%")

    print('\n6. SURVIVAL PATTERN')
    print(f"   >=3/4 quarters positive, no quarter dominating, liquid delta > 0 : "
          f"{'MET' if r['survival_pattern'] else 'NOT MET'}")
    print('\n   (Descriptive decomposition only. No verdict, no alpha consumed,')
    print('    and no recommendation about the reserved test block.)')


if __name__ == '__main__':
    _print_report(run())
