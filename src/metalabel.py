"""
META-LABELING family — power analysis, rules and guards. Research-only.

WHAT THIS IS
------------
The registered protocol is results/metalabel/PRE_REGISTRATION.md (committed
before this file existed, 6082fe5). The question: does disagreement between
sub-models carry information about WHEN a direction call is reliable? Three
conditions were registered — C0 daily variant disagreement, C1 H1 cross-model
disagreement, C2 H1 probability-margin terciles.

WHAT IT FOUND, BEFORE ANY OUTCOME WAS JOINED
--------------------------------------------
Every cell is UNDERPOWERED at the registered bar. C0 needs ~605 years of daily
data to resolve a 3pp effect; C2, the best-powered cell, reaches a 4.46pp
80%-power MDE against a 3pp economic floor; C1 is INFEASIBLE-AS-SPECIFIED (the
"H1->daily ensemble" predicts the next DAY, not the next hour, and is 93.4%
in-sample on the H1 validation window). The registry records ZERO comparisons
spent. This module therefore contains everything needed to state that result
reproducibly — and REFUSES to evaluate a cell, because the pre-registration's
stopping rule says so and the owner's ruling on the refit is NO GO.

HARD BOUNDARY
-------------
* NO model is loaded. NO .fit() is called. NO artifact under models/ is read.
  The served models/h1_direction/ is full-history and in-sample on validation
  (its own meta: "validated_out_of_sample": false); the daily GBM trains on
  [0:80%], which contains the validation slice. The only honest scorer is a
  [0:70%] reconstruction under a reproduction gate — HOLD / NO GO.
* The TEST BLOCKS are NEVER indexed: daily [6884:8605] (dates > 2021-02-01),
  H1 [58946:69349] (bars > 2024-11-19 04:00 UTC). `TestBlockGuard` raises on
  any index that reaches them, and tests/test_metalabel.py asserts it.
* Writes ONLY under results/metalabel/. Never touches models/, src/inference.py,
  src/paper_trading.py, api.py, or any OTHER family's hypothesis log. The
  metalabel registry itself was written at pre-registration and is read here,
  not appended to.

CONVENTIONS MIRRORED
--------------------
* Split arithmetic: src/ablation.py::_canonical_split (daily) and
  src/h1_direction_model.py::split_purge_embargo (H1), reproduced as pure
  functions so the indices can be asserted without importing either module's
  data loaders.
* Bootstrap: paired moving-block (circular), block 24 for H1 / 5 for daily,
  n_boot 2000, seed 42, percentile CI at the FAMILY alpha (passed explicitly —
  h1_direction_model.arbiter's default is the stale FAMILY_ALPHA = 0.025).
  n <= block_len is REFUSED (nan), exactly as
  src/vol_scaled_backtest.py::bootstrap_delta_sharpe does.
* Power: unpaired two-proportion MDE (disjoint cells), p = 0.5, two-sided.
  80%-power governs; just-significant reported alongside, never adjudicated.
* Design effect: MEASURED from the h1_direction registry's own block-vs-iid
  intervals rather than assumed.

Run:  python -m src.metalabel      (writes results/metalabel/power_analysis.csv
                                    and run_meta.json; evaluates nothing)
"""

import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Pre-registered constants (frozen; results/metalabel/PRE_REGISTRATION.md) ──
FEATURE_LOG = 'results/feature_hypothesis_log.csv'
H_DIR_LOG = 'results/h1_direction_hypothesis_log.csv'
HYPOTHESIS_LOG = 'results/metalabel_hypothesis_log.csv'
PRE_REGISTRATION = 'results/metalabel/PRE_REGISTRATION.md'
OUT_DIR = 'results/metalabel'

N_NEW_CONDITIONS = 3                    # C0, C1, C2
FAMILY_SIZE = 12                        # 9 feature-log rows + 3; HELD at 12 (C1 infeasible does not loosen it)
FAMILY_ALPHA = 0.05 / FAMILY_SIZE       # 0.004167
POWER = 0.80
ECONOMIC_FLOOR_PP = 3.0                 # any cell above this at 80% power is NO DECISION
BASE_RATE_P = 0.5                       # variance-maximising p for the MDE

BLOCK_LEN_H1 = 24                       # bars = one trading day (diurnal ACF spike at lag 24)
BLOCK_LEN_DAILY = 5                     # density / calendar_paired_bootstrap convention
N_BOOT = 2000
RANDOM_SEED = 42

TRADING_DAYS_PER_YEAR = 261
EURO_ERA_YEARS = 27                     # 1999 -> 2026

DAILY_TRAIN_FRACTION = 0.80
DAILY_VAL_FRACTION = 0.10
H1_TRAIN_FRAC = 0.70
H1_VAL_FRAC = 0.85                      # END position (test = [0.85:1.0])
H1_EMBARGO_BARS = 24

MIXED_SENTINELS = ('MIXED / LOW CONFIDENCE', 'MIXED / TIE')
DIRECTIONS = ('UP', 'DOWN')

REPRO_TARGET_ACC = 0.527462             # H_dir.1 registered validation accuracy
REPRO_TOLERANCE = 0.003
PIP_TOLERANCE = 5e-5                    # half a pip on EUR/USD

# Registered cell sizes (PRE_REGISTRATION §6). 'assumed' cells depend on model
# output that only a [0:70%] reconstruction (HOLD) could produce.
REGISTERED_CELLS = (
    # (label,                         n1,   n0,   source)
    ('C0_daily_disagree_vs_agree',    353,  508,  'ASSUMED: forward-log rate 20/49 = 40.8% of 861'),
    ('C1_proposal_gbm_vs_lstm',       3585, 6793, 'MEASURED: H_dir.2 McNemar b+c = 1734+1851'),
    ('C2_top_vs_bottom_tercile',      3459, 3459, 'ASSUMED: equal thirds of 10378'),
)
C0_N_VAL = 861


class TestBlockTouchedError(RuntimeError):
    """Raised when any index reaches a SPENT test block. Not a warning."""


class TestBlockGuard:
    """Refuses any index that touches the test block.

    `last_allowed` is the last permitted position or timestamp (inclusive):
    daily `val_end - 1` as a position, or the H1 `val_end_ts` as a UTC
    Timestamp. `check()` is called at every point a slice or index is built;
    a violation raises rather than logging, so it cannot be scrolled past.
    """

    def __init__(self, last_allowed, label=''):
        self.last_allowed = last_allowed
        self.label = label
        self.checks = 0

    def check(self, index):
        self.checks += 1
        idx = pd.Index(index) if not isinstance(index, pd.Index) else index
        if len(idx) == 0:
            return idx
        hi = idx.max()
        if hi > self.last_allowed:
            raise TestBlockTouchedError(
                f'{self.label or "index"} reaches {hi!r} beyond the last permitted '
                f'{self.last_allowed!r}: that is the SPENT test block. Refused.')
        return idx


# ───────────────────────── split arithmetic (pure) ────────────────────────────

def daily_split(n, train_fraction=DAILY_TRAIN_FRACTION, val_fraction=DAILY_VAL_FRACTION):
    """src/ablation.py::_canonical_split, reproduced. `train_end` is the 70%
    mark, `val_end` the 80% mark. Validation = [train_end:val_end];
    test = [val_end:n] and is NEVER indexed."""
    train_end = int(n * (train_fraction - val_fraction))
    val_end = int(n * train_fraction)
    return {'n': n, 'train_end': train_end, 'val_end': val_end,
            'n_val': val_end - train_end, 'n_test_reserved': n - val_end}


def h1_split(n_labelled, train_frac=H1_TRAIN_FRAC, val_frac=H1_VAL_FRAC,
             embargo=H1_EMBARGO_BARS):
    """src/h1_direction_model.py::split_purge_embargo, reproduced positionally.
    Purge drops the final train row; embargo drops the first `embargo`
    validation rows. Returns positions, not timestamps."""
    train_end = int(n_labelled * train_frac)
    val_end = int(n_labelled * val_frac)
    n_train = max(train_end - 1, 0)
    n_val = max(val_end - train_end - embargo, 0)
    return {'n': n_labelled, 'train_end': train_end, 'val_end': val_end,
            'val_start_scored': train_end + embargo, 'n_train': n_train,
            'n_val': n_val, 'n_purged': 1 if train_end else 0,
            'n_embargoed': min(embargo, val_end - train_end),
            'n_test_reserved': n_labelled - val_end}


def validation_positions(split, guard):
    """Positional validation index for either frequency, passed through the
    guard so the test block cannot be reached even by an off-by-one."""
    start = split.get('val_start_scored', split['train_end'])
    idx = np.arange(start, split['val_end'])
    guard.check(idx)
    return idx


# ──────────────────────────────── power ───────────────────────────────────────

def mde_two_proportion(n1, n0, alpha=FAMILY_ALPHA, power=POWER, p=BASE_RATE_P):
    """Unpaired two-proportion minimum detectable effect (fraction, not pp).
    `power=None` returns the just-significant MDE (z_alpha * SE), which is
    ~50% power by construction and is reported but never adjudicated on."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power) if power is not None else 0.0
    se = np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n0))
    return float((z_a + z_b) * se)


def n_required_per_cell(target, alpha=FAMILY_ALPHA, power=POWER, p=BASE_RATE_P):
    """Rows per cell (equal cells) to detect `target` (fraction) at `power`."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil(2 * p * (1 - p) * (z / target) ** 2))


def years_of_daily_data(n_val_now, mde_now, target, trading_days=TRADING_DAYS_PER_YEAR,
                        val_fraction=DAILY_VAL_FRACTION, existing_years=EURO_ERA_YEARS):
    """MDE scales as 1/sqrt(n): n_needed = n_now * (mde_now/target)^2. Reports
    the arbiter-block years and the total-history years at `val_fraction`."""
    factor = (mde_now / target) ** 2
    n_needed = n_val_now * factor
    total_years = n_needed / val_fraction / trading_days
    return {'factor': float(factor), 'n_val_needed': int(round(n_needed)),
            'arbiter_years': float(n_needed / trading_days),
            'total_history_years': float(total_years),
            'shortfall_vs_existing': float(total_years / existing_years)}


def design_effect_from_registered_runs(log_path=H_DIR_LOG, base_dir=''):
    """Block-24 vs i.i.d. half-width ratio on the h1_direction registry's own
    rows — a MEASURED design effect, not an assumed one. Returns the per-row
    SE ratios and their median/max squared."""
    path = os.path.join(base_dir, log_path) if base_dir else log_path
    d = pd.read_csv(path)
    ratios = []
    for _, r in d.iterrows():
        try:
            ii = (float(r['delta_acc_ci_high_iid']) - float(r['delta_acc_ci_low_iid'])) / 2
            bb = (float(r['delta_acc_ci_high_block']) - float(r['delta_acc_ci_low_block'])) / 2
        except (TypeError, ValueError):
            continue
        if np.isfinite(ii) and np.isfinite(bb) and ii > 0:
            ratios.append(bb / ii)
    ratios = np.asarray(ratios, dtype=float)
    if len(ratios) == 0:
        return {'n_rows': 0, 'se_ratios': [], 'deff_median': float('nan'), 'deff_max': float('nan')}
    return {'n_rows': int(len(ratios)), 'se_ratios': ratios.tolist(),
            'deff_median': float(np.median(ratios) ** 2), 'deff_max': float(ratios.max() ** 2)}


def underpowered(mde_pp, floor_pp=ECONOMIC_FLOOR_PP):
    return bool(mde_pp > floor_pp)


def power_table(cells=REGISTERED_CELLS, alpha=FAMILY_ALPHA, power=POWER, deff=1.0,
                floor_pp=ECONOMIC_FLOOR_PP):
    """One row per registered cell: n, just-significant MDE, 80%-power MDE,
    design-effect-corrected MDE, rows required, and the verdict the stopping
    rule forces. All in percentage points."""
    rows = []
    for label, n1, n0, source in cells:
        js = 100 * mde_two_proportion(n1, n0, alpha, power=None)
        p80 = 100 * mde_two_proportion(n1, n0, alpha, power=power)
        p80d = p80 * np.sqrt(deff)
        req = n_required_per_cell(floor_pp / 100, alpha, power)
        rows.append({
            'cell': label, 'n1': n1, 'n0': n0, 'n_source': source,
            'alpha': alpha, 'power': power, 'deff': deff,
            'mde_just_significant_pp': round(js, 4),
            'mde_80pct_power_pp': round(p80, 4),
            'mde_80pct_power_x_deff_pp': round(p80d, 4),
            'n_required_per_cell_for_floor': req,
            'smaller_cell_pct_of_required': round(100 * min(n1, n0) / req, 1),
            'economic_floor_pp': floor_pp,
            'verdict': 'UNDERPOWERED — NO DECISION' if underpowered(p80, floor_pp) else 'POWERED',
        })
    return pd.DataFrame(rows)


# ──────────────────────────────── rules ───────────────────────────────────────

def is_trade(direction):
    """A direction call is a trade only if it is UP or DOWN. The MIXED sentinels
    are 'stand flat' and are EXCLUDED from every cell, never scored wrong."""
    s = pd.Series(direction, dtype=object)
    return s.isin(DIRECTIONS).to_numpy()


def variant_agreement(dir_a, dir_b):
    """The serving definition (src/inference.py): the two consensus STRINGS are
    equal. Note this is True when both are MIXED and False when one is — which
    is why `is_trade` must be applied first to define the scored row set."""
    a = pd.Series(dir_a, dtype=object).to_numpy()
    b = pd.Series(dir_b, dtype=object).to_numpy()
    return a == b


def two_model_agreement(pred_a, pred_b):
    """The FORCED rule for exactly two binary models (PRE_REGISTRATION §4.2):
    agree <=> same direction. There is no unanimous-vs-split distinction at n=2."""
    return np.asarray(pred_a).astype(int) == np.asarray(pred_b).astype(int)


def disagreement_cells_from_mcnemar(b, c, n):
    """With two binary models, opposite calls are exactly the McNemar
    discordant pairs (one right, one wrong), so n_disagree = b + c. Lets the
    C1-proposal cell sizes be read from the registered H_dir.2 row with no refit."""
    n_dis = int(b) + int(c)
    return {'n_disagree': n_dis, 'n_agree': int(n) - n_dis, 'share_disagree': n_dis / float(n)}


def tercile_boundaries(margin_train):
    """(q33, q67) of |p - 0.5| on the TRAIN slice only. Never call this on
    validation; `assign_terciles` applies the frozen boundaries there."""
    m = np.asarray(margin_train, dtype=float)
    return float(np.quantile(m, 1 / 3)), float(np.quantile(m, 2 / 3))


def assign_terciles(margin, q33, q67):
    """0 = bottom, 1 = middle, 2 = top, using boundaries fitted elsewhere."""
    m = np.asarray(margin, dtype=float)
    return np.where(m < q33, 0, np.where(m < q67, 1, 2))


# ────────────────────────────── bootstrap ─────────────────────────────────────

def block_bootstrap_delta(cc, cr, block_len, n_boot=N_BOOT, alpha=FAMILY_ALPHA, seed=RANDOM_SEED):
    """PAIRED moving-block (circular) bootstrap CI on delta accuracy, with the
    percentile bounds set by the FAMILY alpha (passed explicitly).

    REFUSAL, mirroring src/vol_scaled_backtest.py::bootstrap_delta_sharpe: when
    n <= block_len a "block" as long as the sample is a cyclic rotation of every
    value once, so every resample gives the identical delta and the CI collapses
    to a falsely confident point. Returns (nan, nan, point) rather than clamping.
    """
    cc = np.asarray(cc, dtype=float)
    cr = np.asarray(cr, dtype=float)
    n = len(cc)
    if n != len(cr):
        raise ValueError('paired series must have equal length')
    point = float(cc.mean() - cr.mean()) if n else float('nan')
    if n == 0 or n <= block_len:
        return float('nan'), float('nan'), point
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        deltas[b] = cc[idx].mean() - cr[idx].mean()
    return (float(np.percentile(deltas, 100 * (alpha / 2))),
            float(np.percentile(deltas, 100 * (1 - alpha / 2))), point)


# ─────────────────────────── join sanity check ────────────────────────────────

def reconstruct_as_of_close(price_df, as_of_dates, guard=None):
    """Look up `close` for each as-of date from a tz-naive daily price frame
    indexed by date. If `guard` is given the dates are checked against it FIRST,
    so a forward-log date that falls inside the test block is refused before any
    row is read."""
    dates = pd.to_datetime(pd.Index(as_of_dates))
    if getattr(dates, 'tz', None) is not None:
        dates = dates.tz_localize(None)
    if guard is not None:
        guard.check(dates)
    idx = price_df.index
    if getattr(idx, 'tz', None) is not None:
        idx = idx.tz_localize(None)
    close = pd.Series(price_df['close'].to_numpy(), index=idx)
    return close.reindex(dates)


def assert_close_agrees(logged, reconstructed, tol=PIP_TOLERANCE):
    """Sub-pip agreement or the join is rejected. Returns the max abs diff."""
    a = np.asarray(logged, dtype=float)
    b = np.asarray(reconstructed, dtype=float)
    if a.shape != b.shape:
        raise ValueError('logged and reconstructed closes differ in length')
    if np.isnan(b).any():
        raise ValueError('reconstruction has NaN: a date is missing from the price file')
    worst = float(np.max(np.abs(a - b))) if len(a) else 0.0
    if worst > tol:
        raise AssertionError(f'as_of_close disagrees by {worst:.6f} > {tol}: join rejected')
    return worst


# ─────────────────────────────── registry ─────────────────────────────────────

def family_arithmetic(feature_log=FEATURE_LOG, n_new=N_NEW_CONDITIONS, base_dir=''):
    """The bar, with the arithmetic shown. Counts EVERY feature-log row (the
    registered brief's instruction) and adds this family's conditions. Held at
    the registered size even though C1 is infeasible."""
    path = os.path.join(base_dir, feature_log) if base_dir else feature_log
    n_existing = int(len(pd.read_csv(path)))
    size = n_existing + n_new
    return {'feature_log_rows': n_existing, 'new_conditions': n_new, 'family_size': size,
            'alpha': 0.05 / size, 'alpha_if_c1_dropped': 0.05 / (size - 1),
            'held_at_registered_size': True,
            'repo_convention_self_contained_alpha': 0.05 / n_new}


def read_registry(path=HYPOTHESIS_LOG, base_dir=''):
    p = os.path.join(base_dir, path) if base_dir else path
    return pd.read_csv(p)


# ──────────────────────────────── run ─────────────────────────────────────────

def run(base_dir='', write=True):
    """Compute and (optionally) persist the power analysis. EVALUATES NOTHING:
    no model is loaded, no outcome is joined, no cell is scored. Raises if the
    registered arithmetic no longer matches what is on disk."""
    fam = family_arithmetic(base_dir=base_dir)
    if fam['family_size'] != FAMILY_SIZE:
        raise RuntimeError(
            f"family_size on disk {fam['family_size']} != registered {FAMILY_SIZE}: "
            f"the feature log changed after registration. Do not silently re-derive alpha.")

    deff = design_effect_from_registered_runs(base_dir=base_dir)
    table = power_table(deff=deff['deff_median'])
    c0 = table[table['cell'].str.startswith('C0')].iloc[0]
    years = {t: years_of_daily_data(C0_N_VAL, c0['mde_80pct_power_pp'] / 100, t / 100)
             for t in (3.0, 2.0)}

    reg = read_registry(base_dir=base_dir)
    unspent = reg['notes'].str.contains('ZERO COMPARISONS SPENT').all()
    all_blank = reg[['point_delta_acc', 'ci95_dacc_low', 'ci95_dacc_high',
                     'mcnemar_p', 'cleared_bar']].isna().all().all()

    meta = {
        'family': 'metalabel', 'run_at_utc': datetime.now(timezone.utc).isoformat(),
        'pre_registration': PRE_REGISTRATION, 'registry': HYPOTHESIS_LOG,
        'family_arithmetic': fam, 'design_effect': deff,
        'daily_split': daily_split(8605), 'h1_split': h1_split(69349),
        'c0_years_to_resolve': {f'{k:.0f}pp': v for k, v in years.items()},
        'comparisons_spent': 0, 'cells_evaluated': 0, 'models_loaded': 0, 'fits_performed': 0,
        'registry_all_metrics_blank': bool(all_blank), 'registry_all_rows_unspent': bool(unspent),
        'refit_status': 'HOLD / NO GO (owner ruling); precedented path in PRE_REGISTRATION §8',
        'verdict': 'Every cell UNDERPOWERED — NO DECISION; C1 INFEASIBLE-AS-SPECIFIED; zero alpha spent.',
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
    }
    if write:
        out = os.path.join(base_dir, OUT_DIR) if base_dir else OUT_DIR
        os.makedirs(out, exist_ok=True)
        table.to_csv(os.path.join(out, 'power_analysis.csv'), index=False)
        with open(os.path.join(out, 'run_meta.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
    return {'power_table': table, 'meta': meta}


def _print_report(res):
    t = res['power_table']; m = res['meta']; f = m['family_arithmetic']
    print('=' * 78)
    print('METALABEL FAMILY — POWER ANALYSIS. NOTHING EVALUATED.')
    print('=' * 78)
    print(f"family_size = {f['feature_log_rows']} + {f['new_conditions']} = {f['family_size']}  "
          f"alpha = 0.05/{f['family_size']} = {f['alpha']:.6f}  (held; C1 infeasible does not loosen it)")
    print(f"design effect (measured, h1_direction registry): median {m['design_effect']['deff_median']:.3f}  "
          f"max {m['design_effect']['deff_max']:.3f}")
    print()
    cols = ['cell', 'n1', 'n0', 'mde_just_significant_pp', 'mde_80pct_power_pp',
            'mde_80pct_power_x_deff_pp', 'smaller_cell_pct_of_required', 'verdict']
    print(t[cols].to_string(index=False))
    print()
    for k, v in m['c0_years_to_resolve'].items():
        print(f"C0 to resolve {k}: n_val {v['n_val_needed']:,} = {v['arbiter_years']:.0f} yrs in the arbiter "
              f"= {v['total_history_years']:.0f} yrs total history ({v['shortfall_vs_existing']:.0f}x what exists)")
    print()
    print(f"comparisons spent: {m['comparisons_spent']}   models loaded: {m['models_loaded']}   "
          f"fits: {m['fits_performed']}   refit: {m['refit_status']}")
    print(m['verdict'])


if __name__ == '__main__':
    _print_report(run())
