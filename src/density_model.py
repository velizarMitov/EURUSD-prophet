"""
Conditional DENSITY forecasting of next-day EUR/USD log return — family `density`.

PRE-REGISTERED BEFORE THIS FILE EXISTED.
  Protocol : results/density/PRE_REGISTRATION.md
  Registry : results/density_hypothesis_log.csv
  Commit   : 7524cab0237d325a144b7eef1f85ca759643b942

That commit contains the protocol and the two registry rows and NOTHING else —
no model code, no numbers. Everything in this module was written afterwards and
is bound by what that commit already fixed: the metric (CRPS on
validation[70:80], closed form only), the primary rival (Student-t on a FROZEN
calendar scale), the challenger (5-seed MDN ensemble, all-or-nothing), the bar
(family_size=2 -> alpha=0.025, 97.5% CI), the CI method (paired moving-block
bootstrap, block_len=5, 2000 resamples), the decision rule, and the expected
outcome. None of those may be renegotiated here.

WHY THIS IS A NEW ESTIMAND, NOT A REMATCH
The calendar model (models/calendar/, 10 numbers) emits a SCALE — one
non-negative number per day. It has no shape, no skew and no tail index, so it
cannot answer "what is P(r < -1.2%)". Density forecasting asks for the whole
conditional law. The calendar model enters only through a distributional
wrapper, which is what baselines 1-3 are.

ISOLATION CONTRACT (same discipline as src/ti_lstm_h1_experimental.py)
  * NEVER imported by api.py, src/inference.py or _train_pipeline.py.
  * Writes NOTHING under models/. models/calendar/ is read FROZEN — its JSON is
    loaded straight into CalendarVolatilityParams and `fit` is never called.
  * All output lands in results/density/ plus the family's own registry row.
  * Does not touch results/feature_hypothesis_log.csv and does not tighten any
    other family's Bonferroni bar.

Entry point:  python -m src.density_model [--stage validation|test|all]
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import beta as _beta_fn
from scipy.stats import t as student_t

from src.calendar_volatility import (
    CalendarVolatilityModel, CalendarVolatilityParams, build_daily_dataset,
    fit_garch11, garch_sigma_path,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'density')
HYPOTHESIS_LOG = os.path.join(REPO_ROOT, 'results', 'density_hypothesis_log.csv')
CALENDAR_JSON = os.path.join(REPO_ROOT, 'models', 'calendar', 'calendar_volatility.json')

PREREG_COMMIT = '7524cab0237d325a144b7eef1f85ca759643b942'

# ── everything below this line was fixed by the pre-registration commit ──────
SIGMA_FLOOR = 0.05          # percent units; daily EUR/USD sigma ~ 0.53
K_GRID = (2, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
BLOCK_LEN = 5
N_BOOT = 2000
ALPHA = 0.025               # 0.05 / family_size(2)
CI_LEVEL = 1.0 - ALPHA      # 0.975, two-sided
RANDOM_STATE = 42
FAMILY_SIZE = 2
# ── end pre-registered block ─────────────────────────────────────────────────

# Architecture depth was NOT pinned by the pre-registration (it fixed only
# "shared trunk -> K components", the sigma floor, the softplus/log-space
# parameterisation and clipnorm). These are recorded here so the run is
# reproducible and so the unpinned choices are visible rather than buried.
TRUNK_UNITS = (64, 64)
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20
CLIPNORM = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. CRPS — closed forms only. Pure numpy/scipy: importable and testable
#    without keras, which is why the metric layer is kept free of the model.
# ═══════════════════════════════════════════════════════════════════════════

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def _norm_cdf(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _crps_A(m, var):
    """A(m, s^2) = m (2 Phi(m/s) - 1) + 2 s phi(m/s), the Grimit et al. (2006)
    kernel. s = 0 is a degenerate point mass, for which A -> |m|."""
    m = np.asarray(m, dtype=float)
    s = np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0))
    out = np.abs(m)
    nz = s > 0
    if np.any(nz):
        z = np.zeros_like(m)
        np.divide(m, s, out=z, where=nz)
        out = np.where(nz, m * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * s * _norm_pdf(z), out)
    return out


def crps_gaussian_mixture(w, mu, sigma, y):
    """Closed-form CRPS of a Gaussian MIXTURE (Grimit et al. 2006):

        CRPS = sum_i w_i A(y - mu_i, sigma_i^2)
               - 0.5 sum_i sum_j w_i w_j A(mu_i - mu_j, sigma_i^2 + sigma_j^2)

    w, mu, sigma are (n, K); y is (n,). Returns (n,) per-row CRPS.

    NOT estimated by sampling: a Monte-Carlo CRPS would add noise to the exact
    quantity the registered comparison is decided on. With K = 1 this must equal
    src/ltc_data.py::crps_gaussian_np to numerical precision — unit-tested.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if not (w.shape == mu.shape == sigma.shape):
        raise ValueError("crps_gaussian_mixture: w, mu, sigma must share a shape")
    if w.shape[0] != y.shape[0]:
        raise ValueError("crps_gaussian_mixture: y length must match the row count")

    var = sigma ** 2
    term1 = np.sum(w * _crps_A(y - mu, var), axis=1)

    # (n, K, K) pairwise block. K is at most 5 seeds x 5 components = 25.
    dmu = mu[:, :, None] - mu[:, None, :]
    vsum = var[:, :, None] + var[:, None, :]
    wpair = w[:, :, None] * w[:, None, :]
    term2 = 0.5 * np.sum(wpair * _crps_A(dmu, vsum), axis=(1, 2))
    return term1 - term2


def crps_student_t(y, sigma, nu):
    """Closed-form CRPS for a location-0, scale-sigma Student-t (Jordan, Krueger
    and Lerch 2019), by the scaling identity CRPS(y; 0, s, nu) = s * C(y/s; nu):

        C(z) = z (2 F_nu(z) - 1)
               + 2 f_nu(z) (nu + z^2) / (nu - 1)
               - (2 sqrt(nu) / (nu - 1)) * B(1/2, nu - 1/2) / B(1/2, nu/2)^2

    Requires nu > 1 (the mean must exist). Verified in the unit tests against
    numerical integration of int (F(x) - 1{x >= y})^2 dx.
    """
    nu = float(nu)
    if nu <= 1.0:
        raise ValueError("crps_student_t: nu must exceed 1 for CRPS to be finite")
    y = np.asarray(y, dtype=float)
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = y / s
    const = (2.0 * math.sqrt(nu) / (nu - 1.0)) * (
        _beta_fn(0.5, nu - 0.5) / _beta_fn(0.5, nu / 2.0) ** 2)
    c = (z * (2.0 * student_t.cdf(z, nu) - 1.0)
         + 2.0 * student_t.pdf(z, nu) * (nu + z ** 2) / (nu - 1.0)
         - const)
    return s * c


def crps_empirical(sample, y):
    """Exact CRPS of an EMPIRICAL predictive distribution:

        CRPS = E|X - y| - 0.5 E|X - X'|

    The second term does not depend on y, so it is computed once. The first uses
    the sorted-cumsum identity rather than an (n x m) distance matrix. Exact, not
    sampled — the sample IS the distribution here.
    """
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("crps_empirical: empty sample")
    csum = np.concatenate([[0.0], np.cumsum(x)])
    total = csum[-1]

    y = np.asarray(y, dtype=float)
    k = np.searchsorted(x, y, side='right')          # count of x_i <= y
    below = k * y - csum[k]                          # sum over x_i <= y of (y - x_i)
    above = (total - csum[k]) - (n - k) * y          # sum over x_i >  y of (x_i - y)
    e_abs = (below + above) / n

    # Gini mean difference: sum_ij |x_i - x_j| = 2 sum_i (2i - n + 1) x_(i)
    i = np.arange(n, dtype=float)
    gini = 2.0 * np.sum((2.0 * i - n + 1.0) * x) / (n * n)
    return e_abs - 0.5 * gini


# ═══════════════════════════════════════════════════════════════════════════
# 2. PIT / rank histogram — CALIBRATION DIAGNOSTIC ONLY.
#
#    The pre-registration is explicit that this is NOT a second significance
#    test. No p-value is computed from any of it, and a flat PIT may not be used
#    to argue a KEEP the CRPS gap did not earn. A model can be perfectly
#    calibrated and still lose on CRPS — that is why both are carried.
# ═══════════════════════════════════════════════════════════════════════════

def pit_gaussian_mixture(w, mu, sigma, y):
    """F(y) = sum_i w_i Phi((y - mu_i)/sigma_i)."""
    w = np.atleast_2d(np.asarray(w, dtype=float))
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    return np.sum(w * _norm_cdf((y - mu) / np.maximum(sigma, 1e-12)), axis=1)


def pit_student_t(y, sigma, nu):
    return student_t.cdf(np.asarray(y, dtype=float)
                         / np.maximum(np.asarray(sigma, dtype=float), 1e-12), float(nu))


def pit_empirical(sample, y):
    x = np.sort(np.asarray(sample, dtype=float))
    return np.searchsorted(x, np.asarray(y, dtype=float), side='right') / x.size


def rank_histogram(pit, bins: int = 20):
    """Counts and relative frequency per PIT bin. Uniform == calibrated."""
    counts, edges = np.histogram(np.clip(np.asarray(pit, dtype=float), 0.0, 1.0),
                                 bins=bins, range=(0.0, 1.0))
    return counts, edges, counts / max(1, counts.sum())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Data — the volatility family's own rows, features and split. Nothing new.
# ═══════════════════════════════════════════════════════════════════════════

def load_frozen_calendar() -> CalendarVolatilityModel:
    """models/calendar/calendar_volatility.json straight into the model object.

    `fit` is NEVER called. The pre-registration commits to using these 10 numbers
    exactly as the calendar family validated them.
    """
    with open(CALENDAR_JSON, encoding='utf-8') as f:
        params = json.load(f)
    model = CalendarVolatilityModel(use_dow=True)
    model.params = CalendarVolatilityParams(**params)
    return model


def build_density_dataset(config, base_dir: str = ''):
    """Rows, features, target and the frozen calendar scale, all on one index.

    * rows/features/split come from src/volatility.py::build_volatility_matrix —
      price-only PRICE_FEATURE_COLUMNS + lag-PCA, PCA and scaler fit on [0:70%]
      only. No feature is invented here (single-source-of-truth contract).
    * target y = log(close_{t+1}/close_t) * 100, the SIGNED counterpart of
      TARGET_VOLATILITY_COLUMN; |y| equals that column exactly, which is what
      lets the frozen calendar scale enter in its native units.
    * sigma_cal is produced on the calendar family's OWN row set
      (build_daily_dataset) and then aligned onto the feature index by
      timestamp, so the GARCH recursion is the one that model was validated
      with rather than a re-started copy of it. Rows without a calendar value
      (the first, which has no prior return) or without a target (the last) are
      dropped from EVERY model equally.
    """
    from src.volatility import build_volatility_matrix

    feat, x_scaled, split = build_volatility_matrix(config, base_dir=base_dir)
    idx = feat.index.tz_localize(None) if feat.index.tz is not None else feat.index

    daily = build_daily_dataset(os.path.join(base_dir, config['data']['history_csv_path'])
                                if base_dir else config['data']['history_csv_path'])
    cal_idx = daily.index.tz_localize(None)
    sigma_cal = pd.Series(
        load_frozen_calendar().predict(daily['log_return_pct'].to_numpy(float),
                                       daily['dow'].to_numpy(int)),
        index=cal_idx).reindex(idx)

    close = feat['close'].to_numpy(float)
    y = np.full(len(feat), np.nan)
    y[:-1] = np.log(close[1:] / close[:-1]) * 100.0

    keep = np.isfinite(y) & np.isfinite(sigma_cal.to_numpy())
    # Contiguity matters: the split fractions and the moving-block bootstrap
    # both assume a chronological run of rows, so assert the drop is edge-only.
    kept_positions = np.flatnonzero(keep)
    assert kept_positions.size == kept_positions[-1] - kept_positions[0] + 1, \
        "row drops must be at the edges only — an interior gap would break the block bootstrap"

    out = {
        'index': idx[keep],
        'X': x_scaled[keep],
        'y': y[keep],
        'sigma_cal': sigma_cal.to_numpy()[keep],
        'returns_pct': feat['log_return'].to_numpy(float)[keep] * 100.0,
        'dow': np.asarray(idx[keep].dayofweek, dtype=int),
    }
    n = len(out['y'])
    out['train_end'] = int(n * 0.70)
    out['val_end'] = int(n * 0.80)
    out['n'] = n
    return out


def split_masks(d):
    """[0:70] train, [70:80] validation ARBITER, [80:100] test one-shot."""
    n, tr_end, va_end = d['n'], d['train_end'], d['val_end']
    tr = np.zeros(n, dtype=bool)
    tr[:tr_end] = True
    va = np.zeros(n, dtype=bool)
    va[tr_end:va_end] = True
    te = np.zeros(n, dtype=bool)
    te[va_end:] = True
    return tr, va, te


# ═══════════════════════════════════════════════════════════════════════════
# 4. Baselines — all four. Every nuisance parameter is fit on TRAIN rows only.
# ═══════════════════════════════════════════════════════════════════════════

def fit_gaussian_link(y_train, sigma_train) -> float:
    """MLE of c in N(0, c*sigma_cal). Closed form: c = sqrt(mean((y/sigma)^2)).

    sigma_cal is a conditional scale for |y|, not a standard deviation, so the
    link constant is what turns the calendar model's output into a distribution
    parameter. It is a nuisance parameter OF THE BASELINE, not a refit of the
    calendar model — declared as such in the pre-registration, and fitting it is
    what gives the rival its best shot.
    """
    z = np.asarray(y_train, dtype=float) / np.maximum(np.asarray(sigma_train, dtype=float), 1e-12)
    return float(np.sqrt(np.mean(z ** 2)))


def _student_t_nll(params, y, sigma_base):
    log_c, log_nu_excess = params
    c = math.exp(log_c)
    nu = 1.05 + math.exp(log_nu_excess)      # nu > 1 so CRPS stays finite
    s = c * sigma_base
    return float(-np.sum(student_t.logpdf(y / s, nu) - np.log(s)))


def fit_student_t_link(y_train, sigma_train):
    """Joint MLE of (nu, c) for Student-t(nu, 0, c*sigma_base) on TRAIN rows.

    Parameterised as (log c, log(nu - 1.05)) so both stay in their admissible
    range without a constrained optimiser. Returns (nu, c).
    """
    y = np.asarray(y_train, dtype=float)
    s = np.maximum(np.asarray(sigma_train, dtype=float), 1e-12)
    x0 = np.array([math.log(max(fit_gaussian_link(y, s), 1e-6)), math.log(4.0)])
    res = optimize.minimize(_student_t_nll, x0, args=(y, s), method='Nelder-Mead',
                            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 4000})
    log_c, log_nu_excess = res.x
    return float(1.05 + math.exp(log_nu_excess)), float(math.exp(log_c))


def garch_sigma_train_only(returns_pct, train_mask):
    """GARCH(1,1) fit on TRAIN rows only, then its sigma path over all rows.

    A SEPARATE fit from the frozen calendar model (which carries its own GARCH
    fitted on [0:80%] by the calendar family), so baselines 2 and 3 are
    genuinely different arms rather than the same recursion wearing two hats.
    """
    par = fit_garch11(np.asarray(returns_pct, dtype=float), np.asarray(train_mask, dtype=bool))
    return garch_sigma_path(np.asarray(returns_pct, dtype=float), par), par


def baseline_predictions(d, tr, eval_mask):
    """All four baselines, scored on `eval_mask`. Returns {name: dict}."""
    y_tr = d['y'][tr]
    s_cal_tr, s_cal_ev = d['sigma_cal'][tr], d['sigma_cal'][eval_mask]
    y_ev = d['y'][eval_mask]

    out = {}

    c_g = fit_gaussian_link(y_tr, s_cal_tr)
    sg = c_g * s_cal_ev
    out['gaussian_calendar'] = {
        'crps': crps_gaussian_mixture(np.ones((sg.size, 1)), np.zeros((sg.size, 1)),
                                      sg[:, None], y_ev),
        'nll': 0.5 * np.log(2 * np.pi) + np.log(sg) + 0.5 * (y_ev / sg) ** 2,
        'pit': _norm_cdf(y_ev / sg),
        'params': {'c_g': c_g},
    }

    nu_t, c_t = fit_student_t_link(y_tr, s_cal_tr)
    st = c_t * s_cal_ev
    out['student_t_calendar'] = {
        'crps': crps_student_t(y_ev, st, nu_t),
        'nll': -(student_t.logpdf(y_ev / st, nu_t) - np.log(st)),
        'pit': pit_student_t(y_ev, st, nu_t),
        'params': {'nu': nu_t, 'c_t': c_t},
    }

    s_garch_all, garch_par = garch_sigma_train_only(d['returns_pct'], tr)
    nu_h, c_h = fit_student_t_link(y_tr, s_garch_all[tr])
    sh = c_h * s_garch_all[eval_mask]
    out['student_t_garch'] = {
        'crps': crps_student_t(y_ev, sh, nu_h),
        'nll': -(student_t.logpdf(y_ev / sh, nu_h) - np.log(sh)),
        'pit': pit_student_t(y_ev, sh, nu_h),
        'params': {'nu': nu_h, 'c_h': c_h, **{k: float(v) for k, v in garch_par.items()}},
    }

    # Unconditional: the train-block return distribution, no conditioning at all.
    # The floor any conditional model has to clear to have earned its inputs.
    kde_sample = y_tr
    out['empirical_unconditional'] = {
        'crps': crps_empirical(kde_sample, y_ev),
        'nll': np.full(y_ev.size, np.nan),      # a discrete ECDF has no density
        'pit': pit_empirical(kde_sample, y_ev),
        'params': {'n_train_sample': int(kde_sample.size)},
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. The MDN.
# ═══════════════════════════════════════════════════════════════════════════

class SigmaFloorViolation(RuntimeError):
    """Raised when a component sigma reaches the floor or goes non-finite.

    An MDN diverges when one component collapses onto a training point and the
    likelihood runs away. The parameterisation sigma = FLOOR + softplus(raw)
    makes sigma > FLOOR structurally true, so this exception firing means
    something worse has happened — a NaN or an inf in the parameters. Either way
    the run must ABORT LOUDLY rather than quietly produce a number that looks
    like a result.
    """


def _mdn_heads(trunk, k: int, symmetric: bool):
    """logits, mu, raw_sigma. `symmetric` forces one shared mean across all K
    components — the H_den.2 ablation, which removes the mixture's ability to
    express SKEW while leaving its tail/kurtosis freedom intact."""
    import keras
    logits = keras.layers.Dense(k, name='logits')(trunk)
    if symmetric:
        shared = keras.layers.Dense(1, name='mu_shared')(trunk)
        mu = keras.layers.Concatenate(name='mu')([shared] * k) if k > 1 else shared
    else:
        mu = keras.layers.Dense(k, name='mu')(trunk)
    raw_sigma = keras.layers.Dense(k, name='raw_sigma')(trunk)
    return keras.layers.Concatenate(name='theta')([logits, mu, raw_sigma])


def build_mdn(n_features: int, k: int, seed: int, symmetric: bool = False):
    import keras
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input(shape=(n_features,))
    h = inp
    for units in TRUNK_UNITS:
        h = keras.layers.Dense(units, activation='relu')(h)
        h = keras.layers.Dropout(DROPOUT)(h)
    out = _mdn_heads(h, k, symmetric)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                                  clipnorm=CLIPNORM),
                  loss=_make_mdn_nll(k))
    return model


def _make_mdn_nll(k: int):
    """Negative log-likelihood of a Gaussian mixture, in log space throughout.

        log p(y) = logsumexp_i [ log w_i - log sigma_i - 0.5 z_i^2 ] - 0.5 log 2pi

    sigma = SIGMA_FLOOR + softplus(raw) is applied here, so the floor is part of
    the graph and the network can never train its way underneath it.
    """
    from keras import ops

    def loss(y_true, theta):
        logit, mu, raw = theta[:, :k], theta[:, k:2 * k], theta[:, 2 * k:]
        sigma = SIGMA_FLOOR + ops.softplus(raw)
        log_w = logit - ops.logsumexp(logit, axis=1, keepdims=True)
        z = (ops.reshape(y_true, (-1, 1)) - mu) / sigma
        log_comp = log_w - ops.log(sigma) - 0.5 * ops.square(z)
        return -(ops.logsumexp(log_comp, axis=1) - 0.5 * math.log(2 * math.pi))
    return loss


def mixture_params(model, x, k: int):
    """Forward pass -> (w, mu, sigma) as numpy, with the same floor the loss used."""
    theta = np.asarray(model.predict(x, batch_size=4096, verbose=0), dtype=float)
    logit, mu, raw = theta[:, :k], theta[:, k:2 * k], theta[:, 2 * k:]
    w = np.exp(logit - logit.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    sigma = SIGMA_FLOOR + np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0.0)
    return w, mu, sigma


def _make_sigma_guard(model, x_train, k: int):
    """Per-epoch tripwire, as pre-registered: assert min sigma > floor and abort."""
    import keras

    class _Guard(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            _, mu, sigma = mixture_params(model, x_train, k)
            smin = float(np.min(sigma))
            if not np.isfinite(sigma).all() or not np.isfinite(mu).all():
                raise SigmaFloorViolation(
                    f"epoch {epoch + 1}: non-finite mixture parameters — MDN diverged")
            if smin <= SIGMA_FLOOR:
                raise SigmaFloorViolation(
                    f"epoch {epoch + 1}: min sigma {smin:.6g} reached the floor "
                    f"{SIGMA_FLOOR} — component collapse")
            if logs is not None and not np.isfinite(logs.get('loss', 0.0)):
                raise SigmaFloorViolation(f"epoch {epoch + 1}: non-finite training loss")
    return _Guard()


def train_mdn(x_tr, y_tr, x_va, y_va, k: int, seed: int, symmetric: bool = False,
              verbose: int = 0):
    """One MDN. Trains on [0:70%], early-stops on the [70:80%] arbiter (NLL)."""
    import keras
    model = build_mdn(x_tr.shape[1], k, seed, symmetric=symmetric)
    stopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                            restore_best_weights=True)
    model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE, shuffle=True, verbose=verbose,
              callbacks=[stopper, _make_sigma_guard(model, x_tr, k)])
    return model


def ensemble_mixture(per_seed):
    """Equal-weight mixture over seeds -> ONE 5K-component Gaussian mixture.

    This is the validated object, all-or-nothing, exactly as the volatility
    family defines `vol_ready`. A partial ensemble is never scored: the caller
    asserts it holds every seed before this is called.
    """
    if not per_seed:
        raise ValueError("ensemble_mixture: no seeds")
    w = np.concatenate([p[0] for p in per_seed], axis=1) / float(len(per_seed))
    mu = np.concatenate([p[1] for p in per_seed], axis=1)
    sigma = np.concatenate([p[2] for p in per_seed], axis=1)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)
    return w, mu, sigma


# ═══════════════════════════════════════════════════════════════════════════
# 6. Uncertainty — paired moving-block bootstrap, exactly as pre-registered
#    (block_len=5, 2000 resamples, random_state=42, 97.5% CI).
# ═══════════════════════════════════════════════════════════════════════════

def paired_block_bootstrap(crps_rival, crps_challenger, block_len: int = BLOCK_LEN,
                           n_boot: int = N_BOOT, ci_level: float = CI_LEVEL,
                           random_state: int = RANDOM_STATE):
    """CI for mean(rival - challenger) per row. Positive => challenger better.

    Moving-block (circular) resampling of the DIFFERENCE series, so the pairing
    is preserved exactly and serial dependence in adjacent FX days is respected
    — the same reasoning and the same block_len the volatility and calendar
    families use.
    """
    diff = np.asarray(crps_rival, dtype=float) - np.asarray(crps_challenger, dtype=float)
    n = diff.size
    if n == 0:
        raise ValueError("paired_block_bootstrap: empty input")
    block_len = max(1, min(int(block_len), n))
    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_len))

    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % n
        means[b] = diff[idx[:n]].mean()

    tail = (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(means, [100 * tail, 100 * (1 - tail)])
    return {
        'n': int(n), 'point_delta': float(diff.mean()),
        'ci_low': float(lo), 'ci_high': float(hi),
        'ci_level': float(ci_level), 'block_len': int(block_len), 'n_boot': int(n_boot),
        'excludes_zero': bool(lo > 0 or hi < 0),
        'cleared': bool(lo > 0),      # the registered rule: challenger strictly better
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Run
# ═══════════════════════════════════════════════════════════════════════════

def _score_mixture(w, mu, sigma, y):
    log_w = np.log(np.maximum(w, 1e-300))
    z = (y[:, None] - mu) / np.maximum(sigma, 1e-300)
    log_comp = log_w - np.log(np.maximum(sigma, 1e-300)) - 0.5 * z ** 2
    m = log_comp.max(axis=1, keepdims=True)
    log_p = (m[:, 0] + np.log(np.exp(log_comp - m).sum(axis=1))
             - 0.5 * math.log(2 * math.pi))
    return {
        'crps': crps_gaussian_mixture(w, mu, sigma, y),
        'nll': -log_p,
        'pit': pit_gaussian_mixture(w, mu, sigma, y),
    }


def fit_mdn_ensemble(d, tr, va, k: int, symmetric: bool, eval_masks: dict,
                     seeds=SEEDS, verbose: bool = False):
    """Every seed, then the all-or-nothing ensemble mixture on each eval block.

    Returns (per_seed_crps_on_validation, {block: (w, mu, sigma)}).
    """
    x_tr, y_tr = d['X'][tr], d['y'][tr]
    x_va, y_va = d['X'][va], d['y'][va]

    per_seed_params = {name: [] for name in eval_masks}
    per_seed_val_crps = {}
    for seed in seeds:
        model = train_mdn(x_tr, y_tr, x_va, y_va, k, seed, symmetric=symmetric,
                          verbose=1 if verbose else 0)
        for name, mask in eval_masks.items():
            per_seed_params[name].append(mixture_params(model, d['X'][mask], k))
        w, mu, sg = per_seed_params['validation'][-1]
        per_seed_val_crps[seed] = float(np.mean(
            crps_gaussian_mixture(w, mu, sg, d['y'][eval_masks['validation']])))

    assert all(len(v) == len(seeds) for v in per_seed_params.values()), \
        "partial ensemble — the validated object is the FULL seed set, all-or-nothing"
    return per_seed_val_crps, {name: ensemble_mixture(p) for name, p in per_seed_params.items()}


def select_k(d, tr, va, seeds=SEEDS, k_grid=K_GRID, verbose: bool = False):
    """K chosen on the VALIDATION block only, by ensemble CRPS.

    The test mask is not an argument to this function and cannot be reached from
    it — that is the structural form of the pre-registered promise, and
    tests/test_density_model.py asserts it with a tripwire.
    """
    scores = {}
    for k in k_grid:
        _, ens = fit_mdn_ensemble(d, tr, va, k, False,
                                  {'validation': va}, seeds=seeds, verbose=verbose)
        w, mu, sg = ens['validation']
        scores[k] = float(np.mean(crps_gaussian_mixture(w, mu, sg, d['y'][va])))
        print(f"    K={k}: validation ensemble CRPS = {scores[k]:.6f}", flush=True)
    best = min(scores, key=scores.get)
    return best, scores


def _summary_rows(scored: dict, block: str, n_rows: int):
    rows = []
    for name, s in scored.items():
        pit = np.asarray(s['pit'], dtype=float)
        nll = np.asarray(s['nll'], dtype=float)
        rows.append({
            'block': block, 'model': name, 'n_rows': n_rows,
            'crps_mean': float(np.mean(s['crps'])),
            'nll_mean': float(np.nanmean(nll)) if np.isfinite(nll).any() else np.nan,
            'pit_mean': float(np.mean(pit)), 'pit_var': float(np.var(pit)),
            'params': json.dumps(s.get('params', {}), sort_keys=True, default=float),
        })
    return rows


def _score_all_blocks(d, tr, eval_masks, ens, ens_sym, best_k, seeds):
    """Every model on every evaluation block, plus the registered gap tests."""
    all_rows, gaps, pit_rows = [], [], []
    for block, mask in eval_masks.items():
        y = d['y'][mask]
        scored = baseline_predictions(d, tr, mask)
        scored['mdn_ensemble'] = _score_mixture(*ens[block], y)
        scored['mdn_ensemble']['params'] = {'K': best_k, 'seeds': list(seeds),
                                            'components': int(ens[block][0].shape[1])}
        scored['mdn_symmetric_ensemble'] = _score_mixture(*ens_sym[block], y)
        scored['mdn_symmetric_ensemble']['params'] = {'K': best_k, 'shared_mean': True}
        all_rows.extend(_summary_rows(scored, block, int(mask.sum())))

        for name, s in scored.items():
            counts, edges, freq = rank_histogram(s['pit'])
            for i in range(len(counts)):
                pit_rows.append({'block': block, 'model': name, 'bin_lo': edges[i],
                                 'bin_hi': edges[i + 1], 'count': int(counts[i]),
                                 'frequency': float(freq[i])})

        for rival in ('student_t_calendar', 'gaussian_calendar', 'student_t_garch',
                      'empirical_unconditional'):
            bs = paired_block_bootstrap(scored[rival]['crps'], scored['mdn_ensemble']['crps'])
            gaps.append({'block': block, 'hypothesis': 'H_den.1', 'challenger': 'mdn_ensemble',
                         'rival': rival, 'is_registered_primary': rival == 'student_t_calendar',
                         **bs, 'alpha_bonferroni': ALPHA})
        bs = paired_block_bootstrap(scored['mdn_symmetric_ensemble']['crps'],
                                    scored['mdn_ensemble']['crps'])
        gaps.append({'block': block, 'hypothesis': 'H_den.2', 'challenger': 'mdn_ensemble',
                     'rival': 'mdn_symmetric_ensemble', 'is_registered_primary': False,
                     **bs, 'alpha_bonferroni': ALPHA})
    return all_rows, gaps, pit_rows


def run(stage: str = 'validation', seeds=SEEDS, base_dir: str = '', verbose: bool = False):
    """The registered protocol, in the registered order."""
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    cfg_path = os.path.join(base_dir, 'config.json') if base_dir else 'config.json'
    with open(cfg_path, encoding='utf-8') as f:
        config = json.load(f)

    d = build_density_dataset(config, base_dir=base_dir)
    tr, va, te = split_masks(d)
    print(f"pre-registration commit {PREREG_COMMIT}", flush=True)
    print(f"rows n={d['n']}  train={tr.sum()}  validation={va.sum()}  test={te.sum()}",
          flush=True)

    print("selecting K on validation (test block never indexed) ...", flush=True)
    best_k, k_scores = select_k(d, tr, va, seeds=seeds, verbose=verbose)
    print(f"  -> K = {best_k}", flush=True)

    eval_masks = {'validation': va}
    if stage in ('test', 'all'):
        eval_masks['test'] = te

    per_seed_val, ens = fit_mdn_ensemble(d, tr, va, best_k, False, eval_masks,
                                         seeds=seeds, verbose=verbose)
    per_seed_sym, ens_sym = fit_mdn_ensemble(d, tr, va, best_k, True, eval_masks,
                                             seeds=seeds, verbose=verbose)

    all_rows, gaps, pit_rows = _score_all_blocks(d, tr, eval_masks, ens, ens_sym,
                                                 best_k, seeds)

    summary = pd.DataFrame(all_rows)
    gaps_df = pd.DataFrame(gaps)
    seed_df = pd.DataFrame(
        [{'model': 'mdn_ensemble', 'seed': s, 'validation_crps': v}
         for s, v in per_seed_val.items()]
        + [{'model': 'mdn_symmetric_ensemble', 'seed': s, 'validation_crps': v}
           for s, v in per_seed_sym.items()])
    k_df = pd.DataFrame([{'K': k, 'validation_ensemble_crps': v, 'selected': k == best_k}
                         for k, v in k_scores.items()])

    summary.to_csv(os.path.join(OUT_DIR, 'scores.csv'), index=False)
    gaps_df.to_csv(os.path.join(OUT_DIR, 'gaps.csv'), index=False)
    seed_df.to_csv(os.path.join(OUT_DIR, 'per_seed_crps.csv'), index=False)
    k_df.to_csv(os.path.join(OUT_DIR, 'k_selection.csv'), index=False)
    pd.DataFrame(pit_rows).to_csv(os.path.join(OUT_DIR, 'pit_histograms.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'prereg_commit': PREREG_COMMIT, 'stage': stage, 'seeds': list(seeds),
                   'K_selected': best_k, 'K_grid': list(K_GRID), 'alpha': ALPHA,
                   'ci_level': CI_LEVEL, 'block_len': BLOCK_LEN, 'n_boot': N_BOOT,
                   'sigma_floor': SIGMA_FLOOR, 'family_size': FAMILY_SIZE,
                   'n': d['n'], 'train_end': d['train_end'], 'val_end': d['val_end'],
                   'runtime_seconds': round(time.time() - t0, 1)}, f, indent=2)
    return {'summary': summary, 'gaps': gaps_df, 'per_seed': seed_df, 'k': k_df,
            'best_k': best_k, 'd': d}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Pre-registered density family (H_den.1/H_den.2)')
    ap.add_argument('--stage', choices=['validation', 'test', 'all'], default='validation',
                    help="'validation' = the registered arbiter. 'test' adds the ONE-SHOT "
                         "final report and must be run only after the validation verdict "
                         "is recorded.")
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)
    res = run(stage=args.stage, verbose=args.verbose)
    print(res['summary'].to_string(index=False))
    print(res['gaps'].to_string(index=False))
    return res


if __name__ == '__main__':
    main()
