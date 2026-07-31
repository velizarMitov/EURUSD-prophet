"""
H_dir FAMILY — FINAL ONE-SHOT TEST-BLOCK REPORT (irreversible).

WHAT THIS IS
------------
The h1_direction family's reserved test block [85:100%] has never been indexed.
This program spends it ONCE, on two claims BOTH declared before a single
test-block row is read:

  CLAIM A -- H_dir.1 DIRECTION, exactly as committed, scored on the test block.
      Registered validation result: 0.527462 vs 0.504432 train-majority,
      +2.3029pp, block CI [+0.0075, +0.0392], McNemar p = 0.00169.
  CLAIM B -- H_dir.6 MAGNITUDE, a NEW signed-percent-return regressor on the
      SAME rows, SAME features, SAME train slice. Never built before.

AFTER THIS RUN THE BLOCK IS SPENT FOR THIS FAMILY PERMANENTLY. No further H1
direction or magnitude question may be scored on it, whatever the outcome.

NO TRADING FRAME
----------------
Spread, breakeven accuracy, transaction cost, pips-of-profit, P&L, Sharpe,
position sizing and equity curves are ABSENT and must stay absent. Earlier
programs in this project carried a "breakeven accuracy" anchor; that was an
advisor-introduced TRADING criterion inside a purely PREDICTIVE question, and it
is REMOVED here. The owner's target is direction and percentage return, and those
are the only things measured. Pips appear only as a unit of MOVE SIZE alongside
percent, never as profit.

FAMILY RESIZE
-------------
Claim B is a genuinely new hypothesis, so the family grows 5 -> 6 and
alpha becomes 0.05/6 = 0.008333, applied RETROACTIVELY to all six rows including
the direction confirmation. Every prior verdict is re-stated at the new bar.

A tightened alpha WIDENS the bootstrap interval, so a re-statement can only turn
a KEEP into a DROP, never the reverse. For the two rows that already DROPped at a
LOOSER bar (H_dir.2, H_dir.3) that monotonicity settles the question with no
refit. The three rows that KEPT (H_dir.1, H_dir.4, H_dir.5) are genuinely
recomputed: their models are refit on their own already-spent train slices and
their intervals re-cut at the new alpha, using the SAME bootstrap seed so this is
a re-statement of the same experiment rather than a new one.

ORDERING IS ENFORCED, NOT PROMISED
----------------------------------
`TestBlockGuard` records the instant the test block is first accessed and every
fit call site asserts the flag is still unset. Fitting anything after that instant
raises. This is what makes "nothing was refit after seeing test-block numbers" an
assertion rather than a claim.

HARD BOUNDARY -- research only. Writes ONLY
results/h1_direction_hypothesis_log.csv and results/h1_direction_final/.
`src/h1_direction_model.py` is PROTECTED and imported UNCHANGED -- not one
hyperparameter, feature or split rule is redefined here.
"""

import os

import numpy as np
import pandas as pd

# EVERYTHING about the registered model is imported UNCHANGED from the protected
# module. This file adds the magnitude regressor and the test-block scoring; it
# redefines no feature, no hyperparameter and no split rule.
from src.h1_direction_model import (
    BLOCK_LEN, FEATURE_COLUMNS, HYPOTHESIS_LOG, LOG_COLUMNS, N_BOOT, RANDOM_SEED,
    _block_bootstrap_delta, _mcnemar_exact, _roc_auc, arbiter,
    build_direction_dataset, build_sequences, fit_standardizer, flat_matrix,
    load_eurusd_h1, mean_label_uniqueness, assert_feed_not_stale,
    assert_labels_independent, predict_gbm_proba, seed_everything,
    split_purge_embargo, split_purge_embargo_by_dates, train_gbm,
    train_majority_class, REPLICATION_HYPOTHESES,
)
from src.pooled_h1_data import POOLED_DIR
from src.pooled_h1_model import resolve_device
# Pre-registered NY session buckets, imported UNCHANGED -- the session definition
# has exactly one home in this project.
from src.h1_newyork_time import NY_HISTORY_CSV, PIP, SESSION_ORDER, ny_session

OUT_DIR = 'results/h1_direction_final'

# ── The family resize, fixed here ──
FAMILY_SIZE_FINAL = 6
ALPHA_FINAL = 0.05 / FAMILY_SIZE_FINAL              # 0.008333...
PRIOR_ALPHAS = {1: 0.025, 2: 0.025, 3: 0.01, 4: 0.01, 5: 0.01}

# ── The reproduction gate for claim A, fixed before the block is read ──
REPRO_TARGET_ACC = 0.527462
REPRO_TOLERANCE = 0.003

TEST_ARBITER_LABEL = 'h1_direction_TESTBLOCK[85:100]_block_bootstrap'
VAL_ARBITER_LABEL = 'h1_direction_validation[70:85]_block_bootstrap'

SPENT_NOTE = (
    'TEST BLOCK NOW SPENT: the h1_direction family reserved slice [85:100%] was '
    'read ONCE by this program and is PERMANENTLY SPENT. No further H1 direction '
    'or magnitude question may be scored on it, whatever this outcome was. Both '
    'claims were declared before a single test-block row was read, and a runtime '
    'guard (TestBlockGuard) asserts that no model was fitted after the first read.'
)
NO_TRADING_NOTE = (
    'NO TRADING FRAME: spread, breakeven accuracy, transaction cost, P&L, Sharpe, '
    'position sizing and equity curves are deliberately absent. The "breakeven '
    'accuracy" anchor carried by earlier programs in this project was an '
    'advisor-introduced TRADING criterion inside a purely PREDICTIVE question and '
    'is REMOVED. Direction and percentage return are the only things measured; '
    'pips appear only as a unit of move SIZE alongside percent, never as profit.'
)
RESIZE_NOTE = (
    f'FAMILY RESIZED 5 -> 6 (claim B is a genuinely new hypothesis), so '
    f'alpha = 0.05/6 = {ALPHA_FINAL:.6f} applied RETROACTIVELY to all six rows. A '
    'tightened alpha only WIDENS the interval, so a re-statement can turn a KEEP '
    'into a DROP but never the reverse: rows that already DROPped at a looser bar '
    'stay DROPped by monotonicity, and every row that KEPT was genuinely refit and '
    're-cut at the new alpha with the same bootstrap seed.'
)
CLAIM_B_NOTE = (
    "CLAIM B hyperparameters were COPIED VERBATIM from claim A (n_estimators=300, "
    "max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, "
    "reg_lambda=1.0, seed 42, no early_stopping, no eval_set) specifically to "
    "REMOVE TUNING FREEDOM and make the two claims directly comparable. No search "
    "for better hyperparameters was performed. Baseline is the TRAIN-SLICE MEAN of "
    "the signed percent return, a constant -- the honest no-information benchmark "
    "for a noisy zero-mean target. PRIMARY metric is MAE in percent; KEEP requires "
    "the model's MAE below the constant's with the block-bootstrap CI on the MAE "
    "difference entirely below zero. ARCHITECTURE_DOCS.md 4.2.1 documents that this "
    "project's daily return regressor shrinks toward the ~0% conditional mean and "
    "that its MAE approximates the predict-the-mean baseline -- correct behaviour "
    "under a squared/Huber loss on a noisy zero-mean target, not a defect. Claim B "
    "tests whether that also holds at H1."
)


class ReproductionGateError(RuntimeError):
    """Claim A did not reproduce its registered validation accuracy. The test
    block must NOT be spent on a model that is not the registered one."""


def assert_reproduction(acc: float, target: float = REPRO_TARGET_ACC,
                        tol: float = REPRO_TOLERANCE) -> float:
    """
    THE REPRODUCTION GATE. Claim A's refit validation accuracy must land within
    `tol` of the registered value BEFORE the test block is read.

    Two-sided on purpose: a refit that scores materially HIGHER than the
    registered model is just as much "not the registered model" as one that
    scores lower, and would mean the block was about to be spent on something
    other than the committed claim.
    """
    delta = abs(float(acc) - float(target))
    if delta > tol:
        raise ReproductionGateError(
            f'validation accuracy {acc:.6f} differs from the registered '
            f'{target:.6f} by {delta:.6f} > {tol}. STOP -- the test block must not '
            'be spent on a model that is not the registered one.')
    return delta


class FitAfterTestBlockReadError(RuntimeError):
    """Something tried to fit or select AFTER the test block was first read.
    That would invalidate the one-shot discipline, so it raises."""


# ═══════════════════ the ordering guard ═══════════════════════════════════════

class TestBlockGuard:
    """
    Makes "nothing was fitted after the test block was read" an ASSERTION.

    `read()` stamps the instant of first access. Every fit call site calls
    `assert_unread()` first, which raises once the stamp exists. The stamp is set
    exactly once, so re-reading the block for descriptive breakdowns is fine --
    what is forbidden is FITTING afterwards.
    """

    def __init__(self):
        self.read_at = None
        self.fits = []

    def assert_unread(self, what: str):
        if self.read_at is not None:
            raise FitAfterTestBlockReadError(
                f'attempted to fit {what!r} at {pd.Timestamp.utcnow().isoformat()}, '
                f'AFTER the test block was first read at {self.read_at.isoformat()}. '
                'The one-shot discipline forbids this.')
        self.fits.append(what)

    def read(self, obj, label='test block'):
        if self.read_at is None:
            self.read_at = pd.Timestamp.utcnow()
            self.first_read_label = label
        return obj

    @property
    def was_read(self) -> bool:
        return self.read_at is not None


# ═══════════════════ claim B: the magnitude target ════════════════════════════

def build_magnitude_target(df: pd.DataFrame) -> pd.Series:
    """
    Signed next-bar return in PERCENT:

        y[t] = (close[t+1] / close[t] - 1) * 100

    Written literally from `close` rather than derived from the direction
    module's log return, so the unit test can pin the formula itself and show
    that an off-by-one produces a different number. The final bar has no next
    close and is NaN -- dropped, never padded.
    """
    close = df['close'].astype(float)
    return (close.shift(-1) / close - 1.0) * 100.0


def constant_baseline(y_train) -> float:
    """
    The no-information benchmark for claim B: the TRAIN-SLICE MEAN of the signed
    percent return, predicted for every row.

    Deliberately a function of the TRAIN labels ONLY. Computing it from the
    scored slice would be fitting the baseline on the evaluation data, which
    would flatter the baseline and understate the model.
    """
    return float(np.mean(np.asarray(y_train, dtype=float)))


def train_magnitude_gbm(Xtr, ytr, seed: int = RANDOM_SEED, guard: TestBlockGuard = None):
    """
    Claim B's regressor. IDENTICAL hyperparameters to claim A's classifier --
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, seed 42, no early_stopping, no
    eval_set -- differing ONLY in objective ('reg:squarederror') and head.

    Copying them is the point: it removes every degree of tuning freedom and
    makes the two claims directly comparable. Do NOT search for better ones.
    """
    if guard is not None:
        guard.assert_unread('claim B magnitude regressor')
    import xgboost as xgb
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='reg:squarederror', tree_method='hist', device='cuda',
        random_state=seed, n_jobs=0,
    )
    reg.fit(Xtr, ytr)
    return reg


# ═══════════════════ claim B: the arbiter ═════════════════════════════════════

def mae_difference_vector(pred_model, pred_baseline, y_true) -> np.ndarray:
    """
    Row-wise |model error| - |baseline error|.

    SIGN CONVENTION, fixed here and unit-tested: a model STRICTLY BETTER than the
    baseline yields a NEGATIVE mean. KEEP therefore requires the interval to lie
    entirely BELOW zero -- the mirror of claim A's "entirely above".
    """
    y = np.asarray(y_true, dtype=float)
    return (np.abs(np.asarray(pred_model, dtype=float) - y)
            - np.abs(np.asarray(pred_baseline, dtype=float) - y))


def _block_bootstrap_mean(values, block_len, n_boot, alpha, seed=RANDOM_SEED):
    """
    Moving-block (circular) bootstrap CI for the MEAN of a paired row-wise
    difference. Same block machinery and same 24-bar block (one trading day) as
    claim A, so the two claims are judged under the same dependence assumption:
    hourly FX shows volatility clustering, so consecutive rows are not
    independent even when the LABELS are.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    n = len(v)
    point = float(v.mean()) if n else float('nan')
    if n == 0:
        return float('nan'), float('nan'), point
    n_blocks = int(np.ceil(n / block_len))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        means[b] = v[idx].mean()
    return (float(np.percentile(means, 100 * (alpha / 2))),
            float(np.percentile(means, 100 * (1 - alpha / 2))), point)


def magnitude_arbiter(pred_model, pred_baseline, y_true, alpha=ALPHA_FINAL,
                      block_len=BLOCK_LEN, n_boot=N_BOOT, seed=RANDOM_SEED):
    """
    PRIMARY metric: Mean Absolute Error in percent. KEEP iff the model's MAE is
    LOWER than the constant baseline's AND the block-bootstrap CI on the MAE
    difference lies ENTIRELY BELOW zero. No McNemar -- this is not a paired
    classification test.

    RMSE, R-squared and the prediction mean/sd are returned as DESCRIPTIVE
    context only, so prediction shrinkage is VISIBLE rather than inferred.
    """
    y = np.asarray(y_true, dtype=float)
    pm = np.asarray(pred_model, dtype=float)
    pb = np.asarray(pred_baseline, dtype=float)

    diff = mae_difference_vector(pm, pb, y)
    lo, hi, point = _block_bootstrap_mean(diff, block_len, n_boot, alpha, seed)
    ss_res = float(np.sum((y - pm) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    cleared = bool(hi < 0.0)
    return {
        'mae_model': float(np.mean(np.abs(pm - y))),
        'mae_baseline': float(np.mean(np.abs(pb - y))),
        'mae_diff': point, 'mae_diff_ci_low': lo, 'mae_diff_ci_high': hi,
        'rmse_model': float(np.sqrt(np.mean((pm - y) ** 2))),
        'rmse_baseline': float(np.sqrt(np.mean((pb - y) ** 2))),
        'r2_model': float(1.0 - ss_res / ss_tot) if ss_tot else float('nan'),
        'pred_mean': float(pm.mean()), 'pred_sd': float(pm.std()),
        'target_mean': float(y.mean()), 'target_sd': float(y.std()),
        'baseline_constant': float(pb[0]) if len(pb) else float('nan'),
        'block_len': block_len, 'alpha': alpha,
        'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP',
    }


# ═══════════════════ descriptive breakdowns ═══════════════════════════════════

def ny_session_map() -> pd.Series:
    """
    server timestamp -> pre-registered NY session bucket, from the VERIFIED
    mapping in results/pooled_h1/EURUSD_h1_newyork.csv. The bucket boundaries
    are imported unchanged from src/h1_newyork_time.py; nothing is re-derived.
    """
    ny = pd.read_csv(NY_HISTORY_CSV, usecols=['server_timestamp', 'ny_hour'])
    ts = pd.DatetimeIndex(pd.to_datetime(ny['server_timestamp'])).tz_localize('UTC')
    return pd.Series([ny_session(h) for h in ny['ny_hour']], index=ts, name='session')


def breakdown_table(index, y_dir, pred_dir, pred_major, y_pct, pred_mag,
                    base_const, key: pd.Series, order=None) -> pd.DataFrame:
    """One descriptive row per group: direction accuracy and magnitude MAE."""
    g = pd.DataFrame({
        'key': key.to_numpy(), 'y': np.asarray(y_dir, dtype=int),
        'ok_model': (np.asarray(pred_dir, dtype=int) == np.asarray(y_dir, dtype=int)),
        'ok_major': (np.asarray(pred_major, dtype=int) == np.asarray(y_dir, dtype=int)),
        'ae_model': np.abs(np.asarray(pred_mag, dtype=float) - np.asarray(y_pct, dtype=float)),
        'ae_base': np.abs(base_const - np.asarray(y_pct, dtype=float)),
        'abs_move_pct': np.abs(np.asarray(y_pct, dtype=float)),
    }, index=index)

    rows = []
    keys = order if order is not None else sorted(g['key'].unique())
    for k in keys:
        sub = g[g['key'] == k]
        if not len(sub):
            continue
        rows.append({
            'group': k, 'n': int(len(sub)),
            'acc_model': float(sub['ok_model'].mean()),
            'acc_majority': float(sub['ok_major'].mean()),
            'delta_acc_pp': 100.0 * float(sub['ok_model'].mean() - sub['ok_major'].mean()),
            'mae_model_pct': float(sub['ae_model'].mean()),
            'mae_baseline_pct': float(sub['ae_base'].mean()),
            'mae_diff_pct': float(sub['ae_model'].mean() - sub['ae_base'].mean()),
            # DESCRIPTIVE move SIZE only -- never a profit, never net of anything.
            'mean_abs_move_pct': float(sub['abs_move_pct'].mean()),
        })
    return pd.DataFrame(rows)


def sign_agreement(pred_dir, pred_mag) -> dict:
    """
    Do the classifier and the regressor point the same way? The regressor's sign
    is an implied direction call, so disagreement between the two is a coherence
    check on the pair of claims.
    """
    d = np.asarray(pred_dir, dtype=int)
    m = (np.asarray(pred_mag, dtype=float) > 0).astype(int)
    return {'n': int(len(d)), 'agree': int((d == m).sum()),
            'agree_rate': float((d == m).mean()),
            'regressor_pct_up': float(m.mean()),
            'classifier_pct_up': float(d.mean())}


# ═══════════════════ retroactive re-statement at the new alpha ════════════════

def restate_prior_rows(log: pd.DataFrame, recomputed: dict) -> pd.DataFrame:
    """
    Re-state cleared_bar for H_dir.1..H_dir.5 at alpha = 0.05/6.

    Rows whose interval ALREADY spanned zero at a LOOSER alpha are settled by
    monotonicity -- a tighter alpha only widens the interval, so they cannot
    start clearing. Rows that CLEARED are recomputed from refit predictions with
    the same bootstrap seed, because widening genuinely can push a narrow lower
    bound below zero.
    """
    out = log.copy()
    out['alpha'] = ALPHA_FINAL
    notes = []
    for i, row in out.iterrows():
        n = int(row['n'])
        hyp = str(row['hypothesis'])
        if hyp in recomputed:
            r = recomputed[hyp]
            prev = bool(row['cleared_bar']) if str(row['cleared_bar']) != '' else False
            out.at[i, 'delta_acc_ci_low_block'] = round(r['delta_acc_ci_low_block'], 6)
            out.at[i, 'delta_acc_ci_high_block'] = round(r['delta_acc_ci_high_block'], 6)
            out.at[i, 'cleared_bar'] = r['cleared_bar']
            out.at[i, 'verdict'] = r['verdict']
            notes.append({'hypothesis': hyp, 'method': 'REFIT + re-cut at new alpha',
                          'prior_alpha': PRIOR_ALPHAS.get(n), 'prior_cleared': prev,
                          'new_cleared': r['cleared_bar'],
                          'changed': bool(prev != r['cleared_bar']),
                          'ci_low': r['delta_acc_ci_low_block'],
                          'ci_high': r['delta_acc_ci_high_block'],
                          'mcnemar_p': r['mcnemar_p']})
        elif str(row['verdict']) == 'DROP':
            # Already DROPped at a LOOSER bar; a tighter alpha widens the interval
            # and raises the p-value bar, so it cannot start clearing.
            notes.append({'hypothesis': hyp, 'method': 'monotonicity (no refit needed)',
                          'prior_alpha': PRIOR_ALPHAS.get(n), 'prior_cleared': False,
                          'new_cleared': False, 'changed': False,
                          'ci_low': row['delta_acc_ci_low_block'],
                          'ci_high': row['delta_acc_ci_high_block'],
                          'mcnemar_p': row['mcnemar_p']})
    return out, pd.DataFrame(notes)


# ═══════════════════ the log ══════════════════════════════════════════════════
# Claim B is a REGRESSION, so accuracy/McNemar columns do not apply to it and are
# left blank rather than filled with a mislabelled number. The extra columns below
# carry its metrics; prior rows get `metric='accuracy'` and blanks.
MAGNITUDE_COLUMNS = ['metric', 'mae_model_pct', 'mae_baseline_pct', 'mae_diff_pct',
                     'mae_diff_ci_low', 'mae_diff_ci_high', 'rmse_model_pct',
                     'rmse_baseline_pct', 'r2_model', 'pred_mean_pct', 'pred_sd_pct',
                     'target_mean_pct', 'target_sd_pct']
FINAL_LOG_COLUMNS = LOG_COLUMNS[:-1] + MAGNITUDE_COLUMNS + ['notes']


def _blank_row(**kw):
    row = {c: '' for c in FINAL_LOG_COLUMNS}
    row.update(kw)
    return row


# ═══════════════════ orchestration ════════════════════════════════════════════

def run(out_dir: str = POOLED_DIR, log_path: str = HYPOTHESIS_LOG,
        seed: int = RANDOM_SEED, register: bool = True, verbose: bool = True) -> dict:
    """
    The whole one-shot program, in the mandated order: fit everything on TRAIN,
    report both claims on the already-spent VALIDATION slice for context, clear
    the reproduction gate, and only THEN read the test block -- once.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    guard = TestBlockGuard()
    device, dev_info = resolve_device()
    used_seed = seed_everything(seed)

    # ── data, exactly as the protected module builds it ──
    raw = load_eurusd_h1(out_dir=out_dir)
    context, target_index, counts = build_direction_dataset(raw)
    assert_feed_not_stale(counts)
    splits, split_counts = split_purge_embargo(target_index)
    train_idx, val_idx, test_idx = splits['train'], splits['val'], splits['test']

    uniqueness = mean_label_uniqueness(context, train_idx)
    assert_labels_independent(uniqueness)

    # ── claim B's target on the IDENTICAL rows ──
    pct = build_magnitude_target(raw)
    n_dropped_series_end = int(pct.isna().sum())          # the final bar, unpadded
    y_pct_all = pct.reindex(context.index)
    assert np.isfinite(y_pct_all.loc[target_index].to_numpy()).all()

    # ── matrices. TEST IS NOT TOUCHED HERE. ──
    mean, std = fit_standardizer(context, train_idx)
    Xtr, ytr_dir = flat_matrix(context, train_idx, mean, std)
    Xval, yval_dir = flat_matrix(context, val_idx, mean, std)
    ytr_pct = y_pct_all.loc[train_idx].to_numpy(dtype=float)
    yval_pct = y_pct_all.loc[val_idx].to_numpy(dtype=float)

    majority = train_majority_class(ytr_dir)
    base_const = constant_baseline(ytr_pct)

    # ── CLAIM A: refit byte-for-byte, then the REPRODUCTION GATE ──
    guard.assert_unread('claim A direction classifier')
    clf = train_gbm(Xtr, ytr_dir, seed=seed)
    prob_val = predict_gbm_proba(clf, Xval)
    pred_val = (prob_val >= 0.5).astype(int)
    repro_acc = float((pred_val == yval_dir).mean())
    repro_delta = assert_reproduction(repro_acc)

    pred_major_val = np.full(len(yval_dir), majority, dtype=int)
    a_val = arbiter(pred_val, pred_major_val, yval_dir, score_challenger=prob_val,
                    alpha=ALPHA_FINAL, seed=seed)

    # ── CLAIM B: fit on the SAME train rows ──
    reg = train_magnitude_gbm(Xtr, ytr_pct, seed=seed, guard=guard)
    mag_val = reg.predict(Xval)
    b_val = magnitude_arbiter(mag_val, np.full(len(yval_pct), base_const),
                              yval_pct, alpha=ALPHA_FINAL, seed=seed)

    # ── retroactive re-statement: refit the rows that CLEARED, before any test read ──
    recomputed = {'H_dir.1_GBM_vs_train_majority': a_val}
    recomputed.update(_refit_replication_rows(
        context, splits, split_counts, guard, device, seed, out_dir, verbose))

    # ══════════════ THE TEST BLOCK IS READ HERE, ONCE ══════════════
    test_index = guard.read(test_idx, label='h1_direction test block [85:100%]')
    first_read_at = guard.read_at

    Xtest, ytest_dir = flat_matrix(context, test_index, mean, std)
    ytest_pct = y_pct_all.loc[test_index].to_numpy(dtype=float)
    prob_test = predict_gbm_proba(clf, Xtest)
    pred_test = (prob_test >= 0.5).astype(int)
    pred_major_test = np.full(len(ytest_dir), majority, dtype=int)
    mag_test = reg.predict(Xtest)
    base_test = np.full(len(ytest_pct), base_const)

    a_test = arbiter(pred_test, pred_major_test, ytest_dir,
                     score_challenger=prob_test, alpha=ALPHA_FINAL, seed=seed)
    b_test = magnitude_arbiter(mag_test, base_test, ytest_pct,
                               alpha=ALPHA_FINAL, seed=seed)

    # ── descriptive breakdowns (reading, never fitting) ──
    sess = ny_session_map().reindex(test_index)
    n_unmapped = int(sess.isna().sum())
    sess_filled = sess.fillna('UNMAPPED')
    by_session = breakdown_table(test_index, ytest_dir, pred_test, pred_major_test,
                                 ytest_pct, mag_test, base_const, sess_filled,
                                 order=list(SESSION_ORDER) +
                                 (['UNMAPPED'] if n_unmapped else []))
    years = pd.Series(pd.DatetimeIndex(test_index).year.astype(str), index=test_index)
    by_year = breakdown_table(test_index, ytest_dir, pred_test, pred_major_test,
                              ytest_pct, mag_test, base_const, years)
    agree = sign_agreement(pred_test, mag_test)

    result = {
        'device': str(device), 'device_info': dev_info, 'seed': used_seed,
        'guard': guard, 'first_read_at': first_read_at,
        'counts': counts, 'split_counts': split_counts, 'uniqueness': uniqueness,
        'n_dropped_series_end': n_dropped_series_end,
        'repro_acc': repro_acc, 'repro_delta': repro_delta,
        'majority_class': majority, 'baseline_constant': base_const,
        'n_test': int(len(test_index)),
        'a_val': a_val, 'b_val': b_val, 'a_test': a_test, 'b_test': b_test,
        'by_session': by_session, 'by_year': by_year, 'sign_agreement': agree,
        'n_session_unmapped': n_unmapped, 'recomputed': recomputed,
        'test_span': (test_index[0].isoformat(), test_index[-1].isoformat()),
    }

    by_session.to_csv(os.path.join(OUT_DIR, 'testblock_by_ny_session.csv'), index=False)
    by_year.to_csv(os.path.join(OUT_DIR, 'testblock_by_year.csv'), index=False)
    pd.DataFrame({'timestamp': test_index, 'y_direction': ytest_dir,
                  'pred_direction': pred_test, 'proba_direction': prob_test,
                  'y_pct': ytest_pct, 'pred_pct': mag_test}).to_csv(
        os.path.join(OUT_DIR, 'testblock_predictions.csv'), index=False)

    if register:
        result['log'], result['restatement'] = _write_final_log(
            result, log_path, dev_info['device'])
    return result


def _refit_replication_rows(context, splits, split_counts, guard, device, seed,
                            out_dir, verbose):
    """
    Refit H_dir.4 (AUDUSD) and H_dir.5 (CHFUSD) -- the replication rows that
    CLEARED at alpha=0.01 -- and re-cut their intervals at alpha=0.05/6.

    H_dir.2 and H_dir.3 are deliberately NOT refit: both already DROPped at a
    LOOSER bar, and widening an interval that already spans zero cannot make it
    stop spanning zero. That is a proof, not a shortcut, and it is recorded as
    such in the re-statement table.

    Runs BEFORE the test block is read; `guard.assert_unread` enforces it.
    """
    from src.pooled_h1_data import build_pooled_pairs
    pairs = build_pooled_pairs(out_dir=out_dir, write_chfusd=False)
    train_end_ts, val_end_ts = split_counts['train_end_ts'], split_counts['val_end_ts']

    out = {}
    for n, pair in sorted(REPLICATION_HYPOTHESES.items()):
        if n == 3:
            continue                      # DROPped at a looser bar; monotonicity settles it
        ctx, tgt, counts_p = build_direction_dataset(pairs[pair])
        assert_feed_not_stale(counts_p)
        sp, _sc = split_purge_embargo_by_dates(tgt, train_end_ts, val_end_ts)
        p_mean, p_std = fit_standardizer(ctx, sp['train'])
        Xtr_p, ytr_p = flat_matrix(ctx, sp['train'], p_mean, p_std)
        Xval_p, yval_p = flat_matrix(ctx, sp['val'], p_mean, p_std)

        guard.assert_unread(f'H_dir.{n} re-statement refit ({pair})')
        clf_p = train_gbm(Xtr_p, ytr_p, seed=seed)
        prob_p = predict_gbm_proba(clf_p, Xval_p)
        maj_p = train_majority_class(ytr_p)
        res = arbiter((prob_p >= 0.5).astype(int),
                      np.full(len(yval_p), maj_p, dtype=int), yval_p,
                      score_challenger=prob_p, alpha=ALPHA_FINAL, seed=seed)
        out[f'H_dir.{n}_replication_{pair}'] = res
        if verbose:
            print(f'   re-stated H_dir.{n} ({pair}) at alpha={ALPHA_FINAL:.6f}: '
                  f"CI [{res['delta_acc_ci_low_block']:+.5f}, "
                  f"{res['delta_acc_ci_high_block']:+.5f}] -> {res['verdict']}")
    return out


def _write_final_log(r, log_path, device_used):
    """Re-state the five prior rows at the new alpha, then append the test-block
    confirmation and H_dir.6."""
    existing = pd.read_csv(log_path)
    restated, restatement = restate_prior_rows(existing, r['recomputed'])
    for col in MAGNITUDE_COLUMNS:
        if col not in restated.columns:
            restated[col] = ''
    restated['metric'] = 'accuracy'
    restated = restated[FINAL_LOG_COLUMNS]

    a, b, c, s = r['a_test'], r['b_test'], r['counts'], r['split_counts']
    common = dict(
        date=pd.Timestamp.utcnow().date().isoformat(), instrument='EURUSD',
        arbiter=TEST_ARBITER_LABEL, n_rows_total=c['n_feature_valid'],
        n_zero_return_dropped=c['n_zero_return_dropped'],
        zero_return_rate_pct=round(100.0 * c['zero_return_share'], 4),
        longest_identical_close_run=c['longest_identical_close_run'],
        mean_label_uniqueness=round(float(r['uniqueness']), 6),
        n_purged=s['n_purged'], n_embargoed=s['n_embargoed'],
        n_train=s['n_train'], n_val=r['n_test'], block_len=BLOCK_LEN,
        alpha=ALPHA_FINAL, device_used=device_used,
    )

    row_conf = _blank_row(
        n=1, hypothesis='H_dir.1_GBM_vs_train_majority_TESTBLOCK_CONFIRMATION',
        metric='accuracy', acc_challenger=round(a['acc_challenger'], 6),
        acc_reference=round(a['acc_reference'], 6),
        auc_challenger=round(a['auc_challenger'], 6),
        delta_acc=round(a['delta_acc'], 6),
        delta_acc_ci_low_iid=round(a['delta_acc_ci_low_iid'], 6),
        delta_acc_ci_high_iid=round(a['delta_acc_ci_high_iid'], 6),
        delta_acc_ci_low_block=round(a['delta_acc_ci_low_block'], 6),
        delta_acc_ci_high_block=round(a['delta_acc_ci_high_block'], 6),
        mcnemar_b=a['mcnemar_b'], mcnemar_c=a['mcnemar_c'],
        mcnemar_p=round(a['mcnemar_p'], 10),
        cleared_bar=a['cleared_bar'], verdict=a['verdict'],
        notes=(f"ONE-SHOT TEST-BLOCK CONFIRMATION of H_dir.1 (not a 7th hypothesis: "
               f"the family is size 6). Reproduction gate PASSED before the block was "
               f"read: refit validation accuracy {r['repro_acc']:.6f} vs registered "
               f"{REPRO_TARGET_ACC:.6f}, |diff| {r['repro_delta']:.6f} <= "
               f"{REPRO_TOLERANCE}. Test block n={r['n_test']} rows, "
               f"{r['test_span'][0]}..{r['test_span'][1]}; first read at "
               f"{r['first_read_at'].isoformat()}Z. {SPENT_NOTE} {RESIZE_NOTE} "
               f"{NO_TRADING_NOTE}"),
        **common)

    row6 = _blank_row(
        n=6, hypothesis='H_dir.6_magnitude_regressor_vs_train_mean',
        metric='MAE_pct', mae_model_pct=round(b['mae_model'], 8),
        mae_baseline_pct=round(b['mae_baseline'], 8),
        mae_diff_pct=round(b['mae_diff'], 8),
        mae_diff_ci_low=round(b['mae_diff_ci_low'], 8),
        mae_diff_ci_high=round(b['mae_diff_ci_high'], 8),
        rmse_model_pct=round(b['rmse_model'], 8),
        rmse_baseline_pct=round(b['rmse_baseline'], 8),
        r2_model=round(b['r2_model'], 8), pred_mean_pct=round(b['pred_mean'], 8),
        pred_sd_pct=round(b['pred_sd'], 8), target_mean_pct=round(b['target_mean'], 8),
        target_sd_pct=round(b['target_sd'], 8),
        cleared_bar=b['cleared_bar'], verdict=b['verdict'],
        notes=(f"NEW hypothesis, declared before the test block was read. Signed "
               f"next-bar return in PERCENT, y=(close[t+1]/close[t]-1)*100, on the "
               f"IDENTICAL rows, features and train slice as claim A. Baseline = "
               f"train-slice mean constant {r['baseline_constant']:+.8f}%. "
               f"Accuracy/McNemar columns are blank because this is a REGRESSION, not "
               f"a classification -- they are left empty rather than filled with a "
               f"mislabelled number. {CLAIM_B_NOTE} {SPENT_NOTE} {RESIZE_NOTE} "
               f"{NO_TRADING_NOTE}"),
        **common)

    final = pd.concat([restated, pd.DataFrame([row_conf, row6],
                                              columns=FINAL_LOG_COLUMNS)],
                      ignore_index=True)
    final = final.sort_values(['n', 'arbiter'], kind='stable').reset_index(drop=True)
    final.to_csv(log_path, index=False)
    restatement.to_csv(os.path.join(OUT_DIR, 'alpha_restatement.csv'), index=False)
    return final, restatement


# ═══════════════════ the report ═══════════════════════════════════════════════

def _acc_block(res, label):
    return (f"   {label}\n"
            f"     acc model {res['acc_challenger']:.6f}   "
            f"acc train-majority {res['acc_reference']:.6f}   "
            f"delta {res['delta_acc']:+.6f} ({100 * res['delta_acc']:+.4f} pp)\n"
            f"     block CI [{res['delta_acc_ci_low_block']:+.6f}, "
            f"{res['delta_acc_ci_high_block']:+.6f}]  (block_len={res['block_len']} "
            f"bars = 1 trading day) <- governs\n"
            f"     iid CI   [{res['delta_acc_ci_low_iid']:+.6f}, "
            f"{res['delta_acc_ci_high_iid']:+.6f}]  (context only)\n"
            f"     McNemar exact b={res['mcnemar_b']} c={res['mcnemar_c']} "
            f"p={res['mcnemar_p']:.8g}     ROC-AUC {res['auc_challenger']:.6f} "
            f"(descriptive)\n"
            f"     -> {res['verdict']}")


def _mag_block(res, label):
    return (f"   {label}\n"
            f"     MAE model    {res['mae_model']:.6f} %   "
            f"MAE baseline {res['mae_baseline']:.6f} %\n"
            f"     MAE diff (model - baseline) {res['mae_diff']:+.8f} %   "
            f"block CI [{res['mae_diff_ci_low']:+.8f}, {res['mae_diff_ci_high']:+.8f}]\n"
            f"     RMSE model   {res['rmse_model']:.6f} %   "
            f"RMSE baseline {res['rmse_baseline']:.6f} %   "
            f"R^2 {res['r2_model']:+.8f}\n"
            f"     PREDICTIONS  mean {res['pred_mean']:+.8f} %   "
            f"sd {res['pred_sd']:.8f} %\n"
            f"     TARGET       mean {res['target_mean']:+.8f} %   "
            f"sd {res['target_sd']:.8f} %   "
            f"(shrinkage ratio sd_pred/sd_target = "
            f"{res['pred_sd'] / res['target_sd']:.6f})\n"
            f"     -> {res['verdict']}")


def render_report(r) -> str:
    L = []
    add = L.append
    d, c, s = r['device_info'], r['counts'], r['split_counts']

    add('=' * 78)
    add('H_dir FAMILY — FINAL ONE-SHOT TEST-BLOCK REPORT (irreversible)')
    add('=' * 78)

    add('\n1. DEVICE')
    add(f"   CUDA available : {d['cuda_available']}")
    if d['cuda_available']:
        add(f"   CUDA device    : {d['cuda_device_name']}")
    add(f"   device used    : {r['device']}     seed = {r['seed']}")

    add('\n2. REPRODUCTION GATE (claim A) — checked BEFORE the test block was read')
    add(f"   registered validation accuracy : {REPRO_TARGET_ACC:.6f}")
    add(f"   refit validation accuracy      : {r['repro_acc']:.6f}")
    add(f"   |difference|                   : {r['repro_delta']:.6f}   "
        f"tolerance {REPRO_TOLERANCE}")
    add(f"   -> GATE PASSED. The block is being spent on the registered model.")

    add('\n3. ROW COUNTS')
    add(f"   raw H1 bars             : {c['n_bars_raw']}")
    add(f"   feature-valid rows      : {c['n_feature_valid']}")
    add(f"   zero-return dropped     : {c['n_zero_return_dropped']} "
        f"({c['zero_return_share']:.4%})")
    add(f"   dropped at series end   : {r['n_dropped_series_end']} "
        f"(label needs a bar beyond the series; DROPPED, never padded)")
    add(f"   train / validation      : {s['n_train']} / {s['n_val']}")
    add(f"   TEST BLOCK              : {r['n_test']}   "
        f"{r['test_span'][0]} .. {r['test_span'][1]}")
    add(f"   purged {s['n_purged']}, embargoed {s['n_embargoed']} "
        f"(train/val boundary only, exactly as the protected module does; the "
        f"train/test gap is the whole validation slice)")
    add(f"   mean label uniqueness   : {r['uniqueness']:.6f}")

    add('\n4. VALIDATION SLICE [70:85%] — CONTEXT ONLY (already spent, not new evidence)')
    add(_acc_block(r['a_val'], 'CLAIM A  direction, validation'))
    add(_mag_block(r['b_val'], 'CLAIM B  magnitude, validation'))

    add('\n5. *** TEST BLOCK FIRST READ AT '
        f"{r['first_read_at'].isoformat()}Z ***")
    add(f"   Fits performed BEFORE that instant, in order: "
        f"{', '.join(r['guard'].fits)}")
    add('   Nothing was fitted, tuned, thresholded or selected after that instant.')
    add('   Enforced by TestBlockGuard: every fit call site asserts the read flag is')
    add('   still unset, and raises FitAfterTestBlockReadError if it is not.')

    add('\n6. TEST BLOCK — CLAIM A (DIRECTION)')
    add(_acc_block(r['a_test'], f"H_dir.1 on [85:100%], n={r['n_test']}"))

    add('\n7. TEST BLOCK — CLAIM B (MAGNITUDE)')
    add(_mag_block(r['b_test'], f"H_dir.6 on [85:100%], n={r['n_test']}"))
    add(f"   baseline constant = train-slice mean = "
        f"{r['baseline_constant']:+.8f} % per bar")

    add('\n8. DESCRIPTIVE BREAKDOWNS (never a decision path)')
    add('   (a) BY NY SESSION BUCKET — the single most informative line here.')
    add('       On validation the direction edge was ABSENT in the London/NY overlap')
    add('       (+1.00pp against SE 1.39pp). Does that repeat on the test block?')
    add(f"       {'group':10s} {'n':>6s} {'acc':>9s} {'major':>9s} {'delta pp':>9s} "
        f"{'MAE %':>9s} {'base %':>9s} {'|move| %':>9s}")
    for _i, x in r['by_session'].iterrows():
        add(f"       {x['group']:10s} {int(x['n']):6d} {x['acc_model']:9.4f} "
            f"{x['acc_majority']:9.4f} {x['delta_acc_pp']:+9.4f} "
            f"{x['mae_model_pct']:9.6f} {x['mae_baseline_pct']:9.6f} "
            f"{x['mean_abs_move_pct']:9.6f}")
    if r['n_session_unmapped']:
        add(f"       ({r['n_session_unmapped']} test rows had no NY mapping)")
    add('   (b) BY CALENDAR YEAR within the test block')
    for _i, x in r['by_year'].iterrows():
        add(f"       {x['group']:10s} {int(x['n']):6d} {x['acc_model']:9.4f} "
            f"{x['acc_majority']:9.4f} {x['delta_acc_pp']:+9.4f} "
            f"{x['mae_model_pct']:9.6f} {x['mae_baseline_pct']:9.6f} "
            f"{x['mean_abs_move_pct']:9.6f}")
    g = r['sign_agreement']
    add(f"   (c) SIGN AGREEMENT classifier vs regressor: {g['agree']}/{g['n']} = "
        f"{g['agree_rate']:.4f}")
    add(f"       classifier predicts UP {g['classifier_pct_up']:.4f} of the time; "
        f"regressor sign is UP {g['regressor_pct_up']:.4f}")

    add(f'\n9. VERDICTS AT alpha = 0.05/6 = {ALPHA_FINAL:.6f}')
    add(f"   H_dir.1 direction, TEST BLOCK : {r['a_test']['verdict']}")
    add(f"   H_dir.6 magnitude, TEST BLOCK : {r['b_test']['verdict']}")
    add('\n   RETROACTIVE RE-STATEMENT of H_dir.1-H_dir.5 at the new alpha:')
    add(f"       {'hypothesis':38s} {'prior a':>8s} {'was':>6s} {'now':>6s} "
        f"{'changed':>8s}  method")
    for _i, x in r['restatement'].iterrows():
        add(f"       {x['hypothesis']:38s} {x['prior_alpha']:>8} "
            f"{str(x['prior_cleared']):>6s} {str(x['new_cleared']):>6s} "
            f"{str(x['changed']):>8s}  {x['method']}")
    return '\n'.join(L)


if __name__ == '__main__':
    out = run()
    text = render_report(out)
    print(text)
    with open(os.path.join(OUT_DIR, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
