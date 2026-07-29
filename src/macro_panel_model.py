"""
GLOBAL MACRO FX DIRECTION — a NEW hypothesis family (research-only).

THE QUESTION
Do macroeconomic fundamentals, entered as SEPARATE COUNTRY LEVELS rather than
differentials, predict the direction of G10 currencies against the dollar at a
MONTHLY horizon?

WHY THIS FRAMING
The production model carries four macro features, all of them DIFFERENCES (EU
minus US). Differencing discards information: a 3% vs 5% rate configuration is
not the same economic state as 1% vs 3%, though the differential is identical.
This program enters each side separately and lets the model find the interaction.
The horizon is monthly because that is the frequency at which the data actually
updates -- CPI prints 12 times a year, GDP 4, central banks 8. Feeding a monthly
variable into a next-day model asks a slow variable to explain fast noise, and
every previous program in this project has run into that wall.

THE HONEST PRIOR
Meese & Rogoff (1983) established that macro exchange-rate models fail to beat a
random walk out of sample at short horizons, and that result has largely survived
40 years of attack. Mark (1995) found long-horizon predictability; Kilian (1999)
and Faust et al. (2003) contested it. Where modern evidence is least contested is
PANEL estimation across many currencies at LONGER horizons -- which is this
design. A null is expected, and the design exists to make a null informative.

PRE-REGISTERED
* Family    -- NEW and independent: results/macro_panel_hypothesis_log.csv,
               size 2, Bonferroni bar alpha = 0.05/2 = 0.025. Touches no
               existing family's alpha.
* Split     -- chronological on the TIME axis, IDENTICAL boundaries for all nine
               pairs: train [0:70%], validation [70:85%], test [85:100%]
               RESERVED. A per-pair split would place one currency's future
               beside another's present; they co-move, and that is
               cross-sectional leakage.
* Target    -- sign of the pair's log return over the NEXT calendar month.
               Monthly observations are NON-OVERLAPPING, so label uniqueness is
               1.0 by construction -- asserted, and the property every prior H1
               program in this project lacked.
* NO NEURAL NETWORKS. At a few hundred independent observations they are
  excluded by arithmetic, and this project has demonstrated that four times.

H_macro.1 -- L2 logistic regression vs the train-majority baseline.
H_macro.2 -- small GBM vs H_macro.1's predictions on identical rows. The primary
             reference is the SIMPLER MODEL, not the trivial baseline: a GBM that
             beats a coin flip but not a logistic regression has demonstrated
             nothing about nonlinearity, only about the data. Standing project
             convention.

MANDATORY CONTROLS (no alpha):
  1. SHUFFLED-LABEL -- refit on permuted training labels; must land near the
     majority rate or the pipeline leaks and everything is void.
  2. LOOK-AHEAD POSITIVE CONTROL -- re-run with every Tier-B feature shifted one
     month EARLIER than the availability rule allows, deliberately introducing
     look-ahead. Accuracy should IMPROVE. If it does not, the availability dating
     may not be binding and should be inspected. No prior program in this project
     has run a positive control on its anti-look-ahead machinery.

NO P&L, no spread, no position sizing, no backtest, no equity curve. Accuracy and
coefficients only.
"""

import os

import numpy as np
import pandas as pd

from src import macro_panel_data as mpd

HYPOTHESIS_LOG = 'results/macro_panel_hypothesis_log.csv'
ARBITER_LABEL = 'macro_panel_validation[70:85]_month_block_bootstrap'

FAMILY_SIZE = 2
FAMILY_ALPHA = 0.05 / FAMILY_SIZE          # 0.025
N_BOOT = 2000
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85
MIN_INDEPENDENT = 150                      # below this the panel cannot answer

# ── The pre-registered feature list. No time index, no trend, no pair-identity
#    dummy: the model must not be able to memorise which currency or which
#    decade a row is from. `industrial_production_yoy_pct` is absent for the
#    data-availability reason documented in src/macro_panel_data.py.
COUNTRY_BLOCK = ('policy_rate', 'rate_3m', 'yield_10y', 'curve_slope',
                 'cpi_yoy_pct', 'unemployment_rate', 'rate_change_3m')
FEATURE_COLUMNS = (
    [f'for_{c}' for c in COUNTRY_BLOCK]
    + [f'us_{c}' for c in COUNTRY_BLOCK]
    + ['vix_level', 'us_equity_1m_return',
       'own_return_1m', 'own_return_3m', 'own_return_12m']
)
FORBIDDEN_SUBSTRINGS = ('month', 'date', 'pair', 'country', 'index_t', 'trend', 'time')

LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter', 'pairs_used', 'horizon_months',
    'n_train', 'n_val', 'rho_bar', 'k_eff', 'n_independent',
    'mean_label_uniqueness', 'min_detectable_edge_pp', 'tier_c_feature_fraction',
    'mean_abs_revision_us_cpi', 'acc_challenger', 'acc_reference',
    'auc_challenger', 'delta_acc', 'delta_acc_ci_low', 'delta_acc_ci_high',
    'mcnemar_b', 'mcnemar_c', 'mcnemar_p', 'shuffled_label_control_acc',
    'lookahead_positive_control_acc', 'alpha', 'cleared_bar', 'verdict',
    'device_used', 'notes',
]

UNSPENT = 'REGISTERED-UNSPENT'


class UnderpoweredPanelError(RuntimeError):
    """Raised when the panel yields fewer than MIN_INDEPENDENT independent
    observations. Reporting that the panel cannot answer the question is more
    honest than producing a confident-looking null."""


# ───────────────────── panel -> matrices ──────────────────────────────────────

def load_panel(path: str = mpd.PANEL_CSV, horizon: str = '1m') -> pd.DataFrame:
    """Complete cases only: a row enters if every pre-registered feature and its
    label are available under the availability rule."""
    df = pd.read_csv(path, parse_dates=['month_end'])
    need = list(FEATURE_COLUMNS) + [f'y_{horizon}']
    out = df.dropna(subset=need).copy()
    out[f'y_{horizon}'] = out[f'y_{horizon}'].astype(int)
    return out.sort_values(['month_end', 'pair']).reset_index(drop=True)


def assert_no_forbidden_features(columns):
    """No time index, no trend, no pair identity may reach the feature matrix."""
    bad = [c for c in columns
           if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise ValueError(f'forbidden identity/time features in the matrix: {bad}')
    return True


def split_months(panel: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """
    Chronological split on the TIME axis with IDENTICAL boundaries for all nine
    pairs. Returns (train_end_month, val_end_month).
    """
    months = np.sort(panel['month_end'].unique())
    n = len(months)
    return months[int(n * train_frac) - 1], months[int(n * val_frac) - 1]


def slice_panel(panel: pd.DataFrame, train_end, val_end):
    m = panel['month_end']
    return (panel[m <= train_end], panel[(m > train_end) & (m <= val_end)],
            panel[m > val_end])


def standardize(train: pd.DataFrame, other: pd.DataFrame, columns=FEATURE_COLUMNS):
    """
    Standardise using TRAIN-ONLY means and standard deviations. Validation rows
    never influence the fitted statistics -- asserted in a unit test.
    """
    mu = train[list(columns)].mean()
    sd = train[list(columns)].std().replace(0.0, 1.0)
    return ((train[list(columns)] - mu) / sd, (other[list(columns)] - mu) / sd,
            mu, sd)


def label_uniqueness(panel: pd.DataFrame) -> float:
    """Monthly observations do not overlap, so each label occupies exactly one
    month for its own pair: uniqueness is exactly 1.0. Computed, not assumed."""
    counts = panel.groupby(['pair', 'month_end']).size()
    return float((1.0 / counts).mean()) if len(counts) else float('nan')


# ───────────────────── STEP 5: independence + power ───────────────────────────

def independence(panel_train: pd.DataFrame, panel_val: pd.DataFrame,
                 horizon: str = '1m'):
    """
    Nine currencies are not nine independent observations per month -- they share
    a dollar factor. Measures the full 9x9 correlation of contemporaneous monthly
    returns on TRAIN ONLY, then the effective number of independent series.
    """
    wide = panel_train.pivot_table(index='month_end', columns='pair',
                                   values='fwd_logret_1m')
    corr = wide.corr()
    k = corr.shape[0]
    off = corr.to_numpy()[~np.eye(k, dtype=bool)]
    rho_bar = float(np.nanmean(off))
    k_eff = k / (1.0 + (k - 1) * rho_bar) if (1.0 + (k - 1) * rho_bar) else float('nan')
    n_val = int(len(panel_val))
    return {'corr': corr, 'k': k, 'rho_bar': rho_bar, 'k_eff': float(k_eff),
            'n_val_rows': n_val, 'n_independent': int(round(n_val * k_eff / k)),
            'mean_label_uniqueness': label_uniqueness(panel_val)}


def power_statement(n_independent: int, alpha: float = FAMILY_ALPHA):
    """
    Given n_independent, the standard error on accuracy and therefore the
    SMALLEST ACCURACY EDGE this test can resolve. Printed BEFORE any accuracy
    figure. Uses p=0.5, the conservative (largest) variance near chance.
    """
    from scipy.stats import norm
    se = float(np.sqrt(0.25 / n_independent)) if n_independent > 0 else float('nan')
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return {'n_independent': int(n_independent), 'standard_error': se,
            'z_alpha': z, 'min_detectable_edge_pp': 100.0 * z * se, 'alpha': alpha}


# ───────────────────── arbiter ────────────────────────────────────────────────

def month_block_bootstrap(correct_c, correct_r, months, n_boot=N_BOOT,
                          alpha=FAMILY_ALPHA, seed=RANDOM_SEED):
    """
    Paired bootstrap on delta accuracy, resampling BY MONTH and taking all nine
    pairs of a sampled month together. Never resample individual rows: the
    cross-sectional correlation measured in STEP 5 makes rows within a month
    dependent, and row-wise resampling would understate the interval.
    """
    cc = np.asarray(correct_c, dtype=float)
    cr = np.asarray(correct_r, dtype=float)
    months = np.asarray(months)
    uniq = np.unique(months)
    idx_by_month = {m: np.flatnonzero(months == m) for m in uniq}
    rng = np.random.default_rng(seed)
    point = float(cc.mean() - cr.mean())
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_month[m] for m in picked])
        deltas[b] = cc[take].mean() - cr[take].mean()
    return (float(np.percentile(deltas, 100 * (alpha / 2))),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))), point)


def mcnemar_exact(correct_c, correct_r):
    from scipy.stats import binomtest
    a = np.asarray(correct_c).astype(bool)
    r = np.asarray(correct_r).astype(bool)
    b = int(np.sum(a & ~r))
    c = int(np.sum(~a & r))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue)


def roc_auc(y_true, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def arbiter(pred_c, pred_r, y_true, months, score_c=None, alpha=FAMILY_ALPHA,
            seed=RANDOM_SEED):
    """PRIMARY metric: accuracy. ROC-AUC is descriptive context only.
    KEEP iff the CI lies entirely above zero AND McNemar p < alpha."""
    y = np.asarray(y_true).astype(int)
    cc = (np.asarray(pred_c).astype(int) == y).astype(float)
    cr = (np.asarray(pred_r).astype(int) == y).astype(float)
    lo, hi, delta = month_block_bootstrap(cc, cr, months, alpha=alpha, seed=seed)
    b, c, p = mcnemar_exact(cc, cr)
    cleared = bool(lo > 0.0 and p < alpha)
    return {'acc_challenger': float(cc.mean()), 'acc_reference': float(cr.mean()),
            'auc_challenger': roc_auc(y, score_c) if score_c is not None else float('nan'),
            'delta_acc': delta, 'delta_acc_ci_low': lo, 'delta_acc_ci_high': hi,
            'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p, 'alpha': alpha,
            'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP'}


# ───────────────────── models ─────────────────────────────────────────────────

def fit_logistic(Xtr, ytr, seed=RANDOM_SEED):
    """L2 logistic regression, balanced classes, C FIXED at 1.0. No tuning and
    no CV sweep -- a sweep would be a search over the validation slice."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced',
                             max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


def fit_gbm(Xtr, ytr, seed=RANDOM_SEED):
    """Small GBM: depth 3 deliberately -- with this sample a deep tree
    memorises. Fixed hyperparameters, NO early_stopping, NO eval_set."""
    import xgboost as xgb
    pos = float((np.asarray(ytr) == 1).sum())
    neg = float((np.asarray(ytr) == 0).sum())
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        objective='binary:logistic', eval_metric='logloss',
        tree_method='hist', device='cuda',
        scale_pos_weight=(neg / pos) if pos > 0 else 1.0,
        random_state=seed, n_jobs=0)
    clf.fit(Xtr, ytr)
    return clf


def _proba(clf, X):
    return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)


def train_majority(ytr) -> int:
    y = np.asarray(ytr).astype(int)
    return int((y == 1).sum() >= (y == 0).sum())


def shuffled_label_control(Xtr, ytr, Xval, yval, seed=RANDOM_SEED):
    """MANDATORY leakage control (no alpha)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(ytr).astype(int).copy()
    rng.shuffle(y)
    clf = fit_logistic(Xtr, y, seed=seed)
    return float(((_proba(clf, Xval) >= 0.5).astype(int) == np.asarray(yval)).mean())


def lookahead_positive_control(horizon='1m', seed=RANDOM_SEED, verbose=False):
    """
    MANDATORY POSITIVE control on the anti-look-ahead machinery: rebuild the
    panel with every Tier-B feature taken one month LATER than the availability
    rule permits (i.e. deliberately leaking), refit H_macro.1 and score it on the
    real validation rows.

    Accuracy should IMPROVE relative to the honest run. If it does not, the
    availability dating may not be binding and must be inspected.
    """
    leaky = mpd.build_panel(write=False, verbose=verbose, lookahead_tier_b_months=1)
    need = list(FEATURE_COLUMNS) + [f'y_{horizon}']
    leaky = leaky.dropna(subset=need).copy()
    leaky[f'y_{horizon}'] = leaky[f'y_{horizon}'].astype(int)
    if not len(leaky):
        return float('nan')
    tr_end, val_end = split_months(leaky)
    tr, va, _te = slice_panel(leaky, tr_end, val_end)
    if not len(tr) or not len(va):
        return float('nan')
    Xtr, Xva, _mu, _sd = standardize(tr, va)
    clf = fit_logistic(Xtr.to_numpy(), tr[f'y_{horizon}'].to_numpy(), seed=seed)
    pred = (_proba(clf, Xva.to_numpy()) >= 0.5).astype(int)
    return float((pred == va[f'y_{horizon}'].to_numpy()).mean())


# ───────────────────── log ────────────────────────────────────────────────────

def _r(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return round(v, nd) if np.isfinite(v) else v


def _upsert(row, log_path=HYPOTHESIS_LOG):
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


# ───────────────────── orchestration ──────────────────────────────────────────

def run(panel_path: str = mpd.PANEL_CSV, log_path: str = HYPOTHESIS_LOG,
        seed: int = RANDOM_SEED, register: bool = True, verbose: bool = True,
        run_lookahead_control: bool = True):
    """
    STEP 5 (independence + power) is computed and reported BEFORE any model. If
    the panel yields fewer than MIN_INDEPENDENT independent observations the
    program STOPS: the hypotheses are logged REGISTERED-UNSPENT, consuming no
    alpha, rather than producing a confident-looking null.
    """
    from src.pooled_h1_model import resolve_device
    device, dev_info = resolve_device()

    panel = load_panel(panel_path)
    assert_no_forbidden_features(FEATURE_COLUMNS)
    tr_end, val_end = split_months(panel)
    train, val, test = slice_panel(panel, tr_end, val_end)

    ind = independence(train, val)
    power = power_statement(ind['n_independent'])

    # CORROBORATING 3-month arm, sampled NON-OVERLAPPING (quarterly) so
    # independence is preserved. Context only, never a second path to a KEEP;
    # it has its own, weaker power statement.
    q = panel.dropna(subset=['y_3m']).copy()
    q = q[pd.DatetimeIndex(q['month_end']).month % 3 == 0]
    q_tr, q_va, _q_te = slice_panel(q, tr_end, val_end)
    ind_3m = (independence(q_tr, q_va) if len(q_tr) and len(q_va)
              else {'rho_bar': np.nan, 'k_eff': np.nan, 'n_independent': 0,
                    'n_val_rows': 0, 'mean_label_uniqueness': np.nan, 'corr': None})
    power_3m = power_statement(ind_3m['n_independent'])
    revisions = mpd.revision_check(write=False)
    us_cpi_rev = float(revisions.loc[revisions['series_id'] == 'CPIAUCSL',
                                     'mean_abs_revision'].iloc[0]) if len(revisions) else np.nan

    shared = {
        'date': pd.Timestamp.utcnow().date().isoformat(), 'arbiter': ARBITER_LABEL,
        'pairs_used': '|'.join(sorted(panel['pair'].unique())), 'horizon_months': 1,
        'n_train': int(len(train)), 'n_val': int(len(val)),
        'rho_bar': _r(ind['rho_bar']), 'k_eff': _r(ind['k_eff']),
        'n_independent': ind['n_independent'],
        'mean_label_uniqueness': _r(ind['mean_label_uniqueness']),
        'min_detectable_edge_pp': _r(power['min_detectable_edge_pp'], 4),
        'tier_c_feature_fraction': _r(mpd.tier_c_fraction(), 4),
        'mean_abs_revision_us_cpi': _r(us_cpi_rev, 4),
        'alpha': FAMILY_ALPHA, 'device_used': dev_info['device'],
    }

    result = {'panel': panel, 'train': train, 'val': val, 'test': test,
              'split': (tr_end, val_end), 'independence': ind, 'power': power,
              'revisions': revisions, 'device': str(device), 'device_info': dev_info,
              'shared': shared, 'independence_3m': ind_3m, 'power_3m': power_3m,
              'n_val_3m': int(len(q_va)), 'n_train_3m': int(len(q_tr))}

    # ── the STOP gate, evaluated before any model is fitted ──
    if ind['n_independent'] < MIN_INDEPENDENT:
        note = (f"STOPPED BEFORE FITTING: the validation slice yields only "
                f"{ind['n_independent']} independent observations "
                f"({ind['n_val_rows']} rows x k_eff {ind['k_eff']:.3f} / 9), below the "
                f"pre-registered floor of {MIN_INDEPENDENT}. rho_bar={ind['rho_bar']:.4f} "
                f"across the nine pairs. The smallest edge this panel could resolve at "
                f"alpha={FAMILY_ALPHA} is {power['min_detectable_edge_pp']:.2f} percentage "
                f"points, which is larger than any plausible macro effect at a monthly "
                f"horizon. Both hypotheses are therefore REGISTERED-UNSPENT: they "
                f"consumed NO alpha and the family bar is unchanged. Reporting that the "
                f"panel cannot answer the question is more honest than producing a "
                f"confident-looking null. The binding constraint is ALFRED vintage "
                f"coverage: enforcing FIRST PRINTS restricts the usable sample to "
                f"{panel['month_end'].min():%Y-%m}..{panel['month_end'].max():%Y-%m}, "
                f"far shorter than the underlying series.")
        rows = []
        for n, name in ((1, 'H_macro.1_logistic_vs_train_majority'),
                        (2, 'H_macro.2_GBM_vs_logistic')):
            row = {c: '' for c in LOG_COLUMNS}
            row.update({**shared, 'n': n, 'hypothesis': name, 'cleared_bar': '',
                        'verdict': UNSPENT, 'notes': note})
            rows.append(row)
            if register:
                _upsert(row, log_path)
        result.update({'stopped': True, 'rows': rows, 'stop_note': note})
        return result

    # ── matrices ──
    Xtr_df, Xva_df, mu, sd = standardize(train, val)
    Xtr, Xva = Xtr_df.to_numpy(), Xva_df.to_numpy()
    ytr, yva = train['y_1m'].to_numpy(), val['y_1m'].to_numpy()
    months_val = val['month_end'].to_numpy()

    leak = shuffled_label_control(Xtr, ytr, Xva, yva, seed=seed)
    leak_ok = bool(0.40 <= leak <= 0.60)

    logit = fit_logistic(Xtr, ytr, seed=seed)
    p_log = _proba(logit, Xva)
    pred_log = (p_log >= 0.5).astype(int)
    majority = train_majority(ytr)
    pred_major = np.full(len(yva), majority, dtype=int)

    h1 = arbiter(pred_log, pred_major, yva, months_val, score_c=p_log, seed=seed)

    gbm = fit_gbm(Xtr, ytr, seed=seed)
    p_gbm = _proba(gbm, Xva)
    pred_gbm = (p_gbm >= 0.5).astype(int)
    h2 = arbiter(pred_gbm, pred_log, yva, months_val, score_c=p_gbm, seed=seed)
    h2_corr = arbiter(pred_gbm, pred_major, yva, months_val, score_c=p_gbm, seed=seed)

    lookahead = (lookahead_positive_control(seed=seed) if run_lookahead_control
                 else float('nan'))

    coefs = pd.DataFrame({'feature': list(FEATURE_COLUMNS),
                          'coefficient': logit.coef_[0]}).sort_values(
        'coefficient', key=np.abs, ascending=False)
    per_pair = []
    for pair, sub in val.assign(pred=pred_log, correct=(pred_log == yva)).groupby('pair'):
        per_pair.append({'pair': pair, 'n': int(len(sub)),
                         'acc_logistic': float(sub['correct'].mean()),
                         'up_rate': float(sub['y_1m'].mean())})

    result.update({'stopped': False, 'h_macro_1': h1, 'h_macro_2': h2,
                   'h2_corroborating': h2_corr, 'coefficients': coefs,
                   'per_pair': pd.DataFrame(per_pair), 'majority_class': majority,
                   'shuffled_label_control_acc': leak, 'leak_control_sane': leak_ok,
                   'lookahead_control_acc': lookahead})

    common = (f"features are SEPARATE COUNTRY LEVELS, never pre-differenced; no time "
              f"index, trend or pair-identity column reaches the matrix; "
              f"standardisation uses TRAIN-ONLY statistics; split boundaries are "
              f"IDENTICAL across all nine pairs; monthly labels are non-overlapping so "
              f"uniqueness = {ind['mean_label_uniqueness']:.4f}; bootstrap resamples BY "
              f"MONTH (all nine pairs together) because rho_bar={ind['rho_bar']:.4f} "
              f"makes rows within a month dependent; minimum detectable edge "
              f"{power['min_detectable_edge_pp']:.2f}pp. industrial_production dropped "
              f"for all countries (FRED's OECD PRINTO01 family retired; a US-only "
              f"version would make the blocks asymmetric).")

    rows = [
        {**shared, 'n': 1, 'hypothesis': 'H_macro.1_logistic_vs_train_majority',
         'acc_challenger': _r(h1['acc_challenger']), 'acc_reference': _r(h1['acc_reference']),
         'auc_challenger': _r(h1['auc_challenger']), 'delta_acc': _r(h1['delta_acc']),
         'delta_acc_ci_low': _r(h1['delta_acc_ci_low']),
         'delta_acc_ci_high': _r(h1['delta_acc_ci_high']),
         'mcnemar_b': h1['mcnemar_b'], 'mcnemar_c': h1['mcnemar_c'],
         'mcnemar_p': _r(h1['mcnemar_p']), 'shuffled_label_control_acc': _r(leak),
         'lookahead_positive_control_acc': _r(lookahead),
         'cleared_bar': h1['cleared_bar'], 'verdict': h1['verdict'],
         'notes': f"L2 logistic, C=1.0 FIXED (no tuning, no CV sweep), class_weight=balanced; reference = train-majority class {majority}. {common}"},
        {**shared, 'n': 2, 'hypothesis': 'H_macro.2_GBM_vs_logistic',
         'acc_challenger': _r(h2['acc_challenger']), 'acc_reference': _r(h2['acc_reference']),
         'auc_challenger': _r(h2['auc_challenger']), 'delta_acc': _r(h2['delta_acc']),
         'delta_acc_ci_low': _r(h2['delta_acc_ci_low']),
         'delta_acc_ci_high': _r(h2['delta_acc_ci_high']),
         'mcnemar_b': h2['mcnemar_b'], 'mcnemar_c': h2['mcnemar_c'],
         'mcnemar_p': _r(h2['mcnemar_p']), 'shuffled_label_control_acc': _r(leak),
         'lookahead_positive_control_acc': _r(lookahead),
         'cleared_bar': h2['cleared_bar'], 'verdict': h2['verdict'],
         'notes': (f"XGBoost depth3 n_est200 lr.05 balanced, device=cuda, NO early_stopping "
                   f"and NO eval_set; PRIMARY reference = the LOGISTIC model on identical "
                   f"rows (a GBM that beats a coin flip but not a logistic regression has "
                   f"shown nothing about nonlinearity); CORROBORATING vs majority: "
                   f"{h2_corr['acc_challenger']:.4f} vs {h2_corr['acc_reference']:.4f} "
                   f"(delta {h2_corr['delta_acc']:+.4f}, p={h2_corr['mcnemar_p']:.4g}) -- "
                   f"NOT a path to KEEP. {common}")},
    ]
    if register:
        for row in rows:
            _upsert(row, log_path)
    result['rows'] = rows
    return result
