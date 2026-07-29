"""
DIVERGENCE RETEST AT A MULTI-DAY HORIZON — extends the existing family.

WHY THIS EXISTS
---------------
H_div.1/H_div.2 tested a 1-hour horizon (N=4 M15 bars) and DROPped. Measured on
this data EURUSD moves a mean of ~7.3 pips in an hour but ~40.6 pips in a day
and ~71.8 pips in three days, and the reversal a divergence is supposed to
precede is a multi-day phenomenon -- so a 1-hour window was asking a different
question from the one the concept makes.

WHAT THIS HONESTLY IS
---------------------
N=4 DROPped, and the corroborating N=8 and N=12 also straddled zero. This is
therefore the project's THIRD look at the same event set, with a longer horizon
chosen AFTER seeing nulls at shorter ones. The justification is principled
(horizon/claim mismatch, quantified above) rather than exploratory, but it is
still a revision after seeing data and is recorded as such in the log notes, in
those words.

BINDING PRE-COMMITMENT, made at registration time and written into the log:
N=96 and N=192 are the LAST horizons this family will test. If they DROP, the
divergence family is CLOSED -- no further horizon, oscillator, swing-detector or
session variation will be run on it. Without this, horizons could be extended
indefinitely until something cleared by chance.

FAMILY SIZE — A CORRECTION TO THE BRIEF
---------------------------------------
The brief instructed "family grows from 2 to 4, alpha = 0.05/4 = 0.0125", and
registered these arms as H_div.3/H_div.4. That reflects the family as it stood
BEFORE the triggered-entry amendment, which has already been executed: H_div.3
and H_div.4 are TAKEN (triggered-entry event study and its GBM), they were run,
and they consumed alpha. Reusing those slot numbers would silently overwrite two
spent hypotheses and would under-count the family.

So these arms register as **H_div.5 and H_div.6**, the family grows **4 -> 6**,
and the Bonferroni bar is **alpha = 0.05/6 = 0.008333**, applied RETROACTIVELY
to all six rows. That is the same standing rule the brief invoked, applied to
the family's actual size. It is a stricter bar than the brief's 0.0125, never a
looser one.

SCOPE (STEP 2)
--------------
  PRIMARY, decision-bearing -- ALL HOURS, no session filter. The 07:00-10:00 NY
      filter was coherent for a 1-hour hold; it is not coherent for a 1-day hold
      (restricting entry to three hours while holding through twenty-four is
      arbitrary), and it discarded ~83% of events.
  DESCRIPTIVE, no verdict -- the 07:00-10:00 NY subset, reported alongside for
      continuity with H_div.1/H_div.2. NOT a second path to a KEEP.

REUSE (STEP 1)
--------------
src/divergence.py and src/divergence_check.py are imported UNCHANGED and are
sha256-asserted: swing detection, oscillator computation, divergence
classification and the reveal_bar(s2) timing rule are all exactly as they were.
Entry remains close[reveal_bar(s2)]. The structural-entry-trigger variant is NOT
part of this program -- it is H_div.3/H_div.4 and lives in divergence_check.py.
The only things changing here are the forward horizon and the session scope.

NO P&L, no spread subtraction, no equity curve. Pips are a descriptive measure
of move size only.
"""

import hashlib
import os

import numpy as np
import pandas as pd

from src import divergence as dv
from src import divergence_check as dc
from src.h1_horizon_feasibility import uniqueness_from_spans

# ── Family accounting (see the docstring correction) ──
FAMILY_SIZE = 6
FAMILY_ALPHA = 0.05 / FAMILY_SIZE            # 0.008333...
PREVIOUS_FAMILY_ALPHA = 0.05 / 4             # 0.0125, before this resize

# ── STEP 3: horizons, both declared NOW; no other will be added ──
PRIMARY_HORIZON_BARS = 96                    # 24 hours, DECISION-BEARING
CORROBORATING_HORIZON_BARS = 192             # 48 hours, context only

SCOPE_ALL = 'all_hours'                      # PRIMARY
SCOPE_SESSION = 'session_0700_1000_NY'       # DESCRIPTIVE ONLY

MIN_INDEPENDENT_EVENTS = 150                 # below this the test cannot answer
UNDERPOWERED_FRACTION = 0.25                 # of the horizon's mean |move|

HYPOTHESIS_LOG = dc.HYPOTHESIS_LOG
ARBITER_LABEL = 'divergence_m15_validation[70:85]_block_bootstrap'

LOG_COLUMNS = list(dc.LOG_COLUMNS) + [
    'horizon_bars', 'session_scope', 'mean_uniqueness', 'n_independent',
    'min_detectable_edge_pips',
]

HORIZON_REVISION_NOTE = (
    "HORIZON REVISED AFTER SEEING NULLS AT N=4/8/12. Stated in those words: the "
    "1-hour primary horizon and its N=8/N=12 corroborators all straddled zero, and "
    "this is the family's FOURTH look at the SAME event set (horizons N=4, N=8, "
    "N=12, now N=96) with a longer horizon chosen after seeing those nulls; the "
    "triggered-entry arms H_div.3/H_div.4 were a further separate re-examination of "
    "the same events. The justification is principled rather than "
    "exploratory -- EURUSD moves ~7.3 pips in an hour but ~40.6 in a day and ~71.8 "
    "in three days, so a 1-hour window asked a different question from the one the "
    "divergence claim makes -- but it is a revision after seeing data and is logged "
    "as such, not as a fresh pre-registration."
)
PRECOMMITMENT_NOTE = (
    "BINDING PRE-COMMITMENT, made at registration time: N=96 and N=192 are the LAST "
    "horizons this family will test. If they DROP the divergence family is CLOSED -- "
    "no further horizon, oscillator, swing-detector or session variation will be run "
    "on it. Without this commitment horizons could be extended indefinitely until "
    "something cleared by chance."
)
FAMILY_RESIZE_NOTE = (
    "FAMILY RESIZE 4 -> 6, alpha 0.0125 -> 0.05/6 = 0.008333, applied RETROACTIVELY "
    "to all six rows. The brief specified '2 -> 4, alpha 0.05/4' and asked for these "
    "arms to be H_div.3/H_div.4, but that reflected the family BEFORE the "
    "triggered-entry amendment had been executed. H_div.3/H_div.4 were already spent "
    "on the triggered-entry arms, so reusing those slots would have overwritten two "
    "spent hypotheses and under-counted the family. These arms are therefore H_div.5 "
    "and H_div.6 at the stricter bar."
)
SCOPE_NOTE = (
    "PRIMARY scope is ALL HOURS -- no session filter. When the holding horizon (24h) "
    "far exceeds the session (3h), restricting entry hour is arbitrary and it "
    "discarded ~83% of events. The 07:00-10:00 NY subset is reported alongside for "
    "continuity and is DESCRIPTIVE ONLY -- never a second path to a KEEP."
)

PROTECTED_SOURCES = ('src/divergence.py', 'src/divergence_check.py')


class UnderpoweredError(RuntimeError):
    """Raised when the primary arm has fewer than MIN_INDEPENDENT_EVENTS
    independent observations. Better to report that the test cannot answer the
    question than to produce a confident-looking null."""


class ReusedSourceModifiedError(RuntimeError):
    """Raised when a module this program claims to REUSE has been modified."""


# ───────────────────── STEP 1: prove the reuse ────────────────────────────────

def source_hashes(paths=PROTECTED_SOURCES) -> dict:
    """sha256 of the modules this program reuses unchanged."""
    out = {}
    for p in paths:
        with open(p, 'rb') as f:
            out[p] = hashlib.sha256(f.read()).hexdigest()
    return out


def assert_sources_unmodified(expected: dict):
    """Detection logic must be genuinely reused, not quietly edited."""
    now = source_hashes(tuple(expected))
    changed = [p for p in expected if now[p] != expected[p]]
    if changed:
        raise ReusedSourceModifiedError(
            f"these modules must be reused UNCHANGED but differ: {changed}")
    return True


# ───────────────────── STEP 4: independence accounting ────────────────────────

def independence_accounting(events: pd.DataFrame, horizon: int,
                            grid_lo: int, grid_hi: int) -> dict:
    """
    The constraint that has decided every previous program in this project. At
    N=4 labels barely overlapped; at N=96 they do, heavily.

      * inter-event gap distribution (median, IQR) over reveal bars
      * mean label UNIQUENESS in the Lopez de Prado sense, from the ACTUAL label
        spans [reveal, reveal+horizon], reusing
        src/h1_horizon_feasibility.uniqueness_from_spans unchanged
      * n_independent = n_events * mean_uniqueness, as an integer

    The uniqueness grid is the validation bar range, offset to zero, so
    concurrency is counted on the slice the test actually uses.
    """
    n = int(len(events))
    if n == 0:
        return {'n_events': 0, 'median_gap': np.nan, 'q25': np.nan,
                'q75': np.nan, 'iqr': np.nan, 'mean_uniqueness': np.nan,
                'n_independent': 0}

    rev = np.sort(events['reveal_idx'].to_numpy().astype(np.int64))
    gaps = np.diff(rev)
    gaps = gaps[gaps > 0]
    median_gap = float(np.median(gaps)) if len(gaps) else 0.0
    q25 = float(np.percentile(gaps, 25)) if len(gaps) else 0.0
    q75 = float(np.percentile(gaps, 75)) if len(gaps) else 0.0

    starts = rev - int(grid_lo)
    ends = starts + int(horizon)
    grid_len = int(max(grid_hi - grid_lo, ends.max() + 1))
    uniq = float(uniqueness_from_spans(starts, ends, grid_len))

    return {'n_events': n, 'median_gap': median_gap, 'q25': q25, 'q75': q75,
            'iqr': q75 - q25, 'mean_uniqueness': uniq,
            'n_independent': int(round(n * uniq))}


def power_statement(values, n_independent: int, alpha: float = FAMILY_ALPHA) -> dict:
    """
    PRE-REGISTERED POWER STATEMENT, computed and printed BEFORE any result.

    Using the measured standard deviation of signed returns and the INDEPENDENT
    sample size (not the raw event count), state the standard error and hence the
    smallest mean edge in pips this test could resolve at `alpha`.

    A DROP from an underpowered test means "no LARGE edge", not "no edge", and
    the report must say so before the numbers rather than after.
    """
    from scipy.stats import norm
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
    mean_abs = float(np.abs(v).mean()) if len(v) else float('nan')
    se = sd / np.sqrt(n_independent) if n_independent > 0 else float('nan')
    z = float(norm.ppf(1.0 - alpha / 2.0))
    mde = z * se
    return {
        'std_signed_pips': sd, 'mean_abs_move_pips': mean_abs,
        'n_independent': int(n_independent), 'standard_error_pips': se,
        'z_alpha': z, 'min_detectable_edge_pips': mde,
        'mde_as_fraction_of_mean_abs': (mde / mean_abs) if mean_abs else np.nan,
        'underpowered': bool(mean_abs and (mde / mean_abs) > UNDERPOWERED_FRACTION),
        'alpha': alpha,
    }


# ───────────────────── event sets at the long horizons ────────────────────────

def build_horizon_events(ny_df: pd.DataFrame, base_events: pd.DataFrame,
                         horizons=(PRIMARY_HORIZON_BARS, CORROBORATING_HORIZON_BARS)):
    """
    Attach long-horizon forward returns to the EXISTING event table. Detection,
    classification and timing are untouched -- only the forward window changes.

    Windows crossing a weekend/holiday gap or reaching into the reserved test
    block are EXCLUDED, not padded. At 96 bars this excludes considerably more
    events than at 4, which is reported rather than absorbed.
    """
    _train_end, val_end = dv.split_bounds(len(ny_df))
    return dv.attach_forward_returns(base_events, ny_df, horizons=horizons,
                                     test_start_idx=val_end)


def scope_events(events: pd.DataFrame, horizon: int, scope: str,
                 slice_name: str = 'val') -> pd.DataFrame:
    """Regular divergences with a usable window at `horizon`, in one scope."""
    m = (events['is_regular'] & events[f'window_ok_n{horizon}']
         & (events['slice'] == slice_name))
    if scope == SCOPE_SESSION:
        m = m & events['in_session']
    return events[m].copy()


def choose_block_len(median_gap: float, horizon: int) -> int:
    """block_len >= max(median inter-event gap, N) -- overlapping labels must
    stay inside one block or the bootstrap will understate the correlation."""
    return int(max(int(np.ceil(median_gap)), horizon))


# ───────────────────── the declared sensitivity band at N=96 ──────────────────

def band_at_horizon(ny_df: pd.DataFrame, horizon: int, lo: int, hi: int,
                    block_len: int, scope: str = SCOPE_ALL,
                    seed: int = dc.RANDOM_SEED) -> pd.DataFrame:
    """
    The band declared in src/divergence.py, re-run at this horizon. Same binding
    rules: no alpha, no verdict, the DEFAULT parameters alone decide, and a band
    member clearing while the default does not is PARAMETER FRAGILITY, not a KEEP.
    """
    pivots = dv.detect_swings(ny_df)
    n_bars = len(ny_df)
    _train_end, val_end = dv.split_bounds(n_bars)
    rows = []
    for kind, params_list in dv.SENSITIVITY_BAND.items():
        for params in params_list:
            label = dv.oscillator_name(kind, params)
            ev = dv.build_all_events(ny_df, pivots, {label: (kind, params)})
            if not len(ev):
                continue
            ev = dv.attach_session(ev, ny_df.index)
            ev = dv.attach_forward_returns(ev, ny_df, horizons=(horizon,),
                                           test_start_idx=val_end)
            ev = dv.assign_slice(ev, n_bars)
            sub = scope_events(ev, horizon, scope)
            v = sub[f'signed_return_pips_n{horizon}'].to_numpy(dtype=float)
            if not len(v):
                continue
            ci_lo, ci_hi, mean = dc.block_bootstrap_mean(
                v, sub['reveal_idx'].to_numpy(), lo, hi, block_len,
                alpha=FAMILY_ALPHA, seed=seed)
            rows.append({
                'oscillator_kind': kind, 'params': str(params), 'label': label,
                'is_default': label in dv.PRIMARY_OSCILLATORS,
                'n_events': int(len(v)), 'mean_signed_pips': mean,
                'ci_low': ci_lo, 'ci_high': ci_hi,
                'win_rate_pct': float((v > 0).mean() * 100.0),
                'would_have_cleared': bool(np.isfinite(ci_lo) and ci_lo > 0),
            })
    return pd.DataFrame(rows)


# ───────────────────── STEP 6: the log ────────────────────────────────────────

def _write_family_log(rows, log_path: str = HYPOTHESIS_LOG):
    """Write ALL SIX rows in one place, so the family's alpha is consistent by
    construction rather than by a later patch."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log = pd.DataFrame(rows, columns=LOG_COLUMNS).sort_values('n')
    log.to_csv(log_path, index=False)
    return log


# ───────────────────── orchestration ──────────────────────────────────────────

def run(log_path: str = HYPOTHESIS_LOG, seed: int = dc.RANDOM_SEED,
        register: bool = True, verbose: bool = True, run_band: bool = True):
    """
    STEP 1 (reuse + hash) -> STEP 4 (independence + power, BEFORE any result)
    -> H_div.5 / H_div.6 -> band -> N=192 -> log. The test block is never indexed.
    """
    hashes_before = source_hashes()

    base = dc.run(register=False, write_ny=False, verbose=False, run_band=False,
                  seed=seed)
    assert_sources_unmodified(hashes_before)

    ny_df = dv.load_m15_newyork()
    train_end, val_end = dv.split_bounds(len(ny_df))
    lo, hi = train_end, val_end

    events = build_horizon_events(ny_df, base['events'])
    H, HC = PRIMARY_HORIZON_BARS, CORROBORATING_HORIZON_BARS

    sets = {}
    for scope in (SCOPE_ALL, SCOPE_SESSION):
        for h in (H, HC):
            sets[(scope, h, 'val')] = scope_events(events, h, scope, 'val')
            sets[(scope, h, 'train')] = scope_events(events, h, scope, 'train')
    for key, frame in sets.items():
        assert (frame['slice'] != 'test').all(), 'test block must never be indexed'

    # ── STEP 4: independence + power, computed BEFORE any test ──
    accounting, power = {}, {}
    for scope in (SCOPE_ALL, SCOPE_SESSION):
        for h in (H, HC):
            ev = sets[(scope, h, 'val')]
            acc = independence_accounting(ev, h, lo, hi)
            accounting[(scope, h)] = acc
            power[(scope, h)] = power_statement(
                ev[f'signed_return_pips_n{h}'].to_numpy(), acc['n_independent'])

    primary_acc = accounting[(SCOPE_ALL, H)]
    if primary_acc['n_independent'] < MIN_INDEPENDENT_EVENTS:
        raise UnderpoweredError(
            f"primary arm has only {primary_acc['n_independent']} independent "
            f"observations (< {MIN_INDEPENDENT_EVENTS}). This test cannot answer "
            "the question -- STOP and report rather than produce a "
            "confident-looking null.")

    block_len = choose_block_len(primary_acc['median_gap'], H)
    block_len_hc = choose_block_len(accounting[(SCOPE_ALL, HC)]['median_gap'], HC)

    # ── H_div.5 : event study, N=96, ALL HOURS ──
    val_all = sets[(SCOPE_ALL, H, 'val')]
    h5 = dc.event_study(val_all, lo, hi, block_len, horizon=H,
                        alpha=FAMILY_ALPHA, seed=seed)
    h5_breakdown = dc.descriptive_breakdown(val_all, horizon=H)

    # DESCRIPTIVE ONLY: the 07:00-10:00 NY subset, never a path to KEEP
    val_sess = sets[(SCOPE_SESSION, H, 'val')]
    h5_session = dc.event_study(val_sess, lo, hi,
                                choose_block_len(accounting[(SCOPE_SESSION, H)]['median_gap'], H),
                                horizon=H, alpha=FAMILY_ALPHA, seed=seed)

    # CORROBORATING ONLY: N=192
    val_all_hc = sets[(SCOPE_ALL, HC, 'val')]
    h5_corroborating = dc.event_study(val_all_hc, lo, hi, block_len_hc,
                                      horizon=HC, alpha=FAMILY_ALPHA, seed=seed)

    # ── H_div.6 : GBM, N=96, ALL HOURS ──
    h6 = dc.model_test(sets[(SCOPE_ALL, H, 'train')], val_all, lo, hi, block_len,
                       horizon=H, alpha=FAMILY_ALPHA, seed=seed)
    if not h6['leak_control_sane']:
        raise dc.LeakageError(
            f"H_div.6 shuffled-label control = {h6['shuffled_label_control_acc']:.4f}, "
            "outside [0.40, 0.60] -- the pipeline leaks and every number is void.")

    band = (band_at_horizon(ny_df, H, lo, hi, block_len, seed=seed)
            if run_band else pd.DataFrame())

    # ── restate H_div.1..4 at the NEW family alpha ──
    r1 = dc.event_study(base['val_events'], lo, hi, base['block']['block_len'],
                        alpha=FAMILY_ALPHA, seed=seed)
    r2 = dc.model_test(base['train_events'], base['val_events'], lo, hi,
                       base['block']['block_len'], alpha=FAMILY_ALPHA, seed=seed)
    trig_col = f'trig_return_pips_n{dv.PRIMARY_HORIZON}'
    r3 = dc.event_study(base['trig_val'], lo, hi, base['trig_block']['block_len'],
                        alpha=FAMILY_ALPHA, seed=seed, value_col=trig_col)
    r4 = dc.model_test(base['trig_train'], base['trig_val'], lo, hi,
                       base['trig_block']['block_len'], alpha=FAMILY_ALPHA,
                       seed=seed, value_col=trig_col,
                       extra_features=('bars_reveal_to_trigger', 'pips_given_up'),
                       pos_col='trigger_idx')

    restated = {1: r1, 2: r2, 3: r3, 4: r4}
    # Whether the tightening changes anything, checked rather than asserted. Note
    # that "the CIs are unchanged, only the threshold moves" holds for the McNemar
    # p-value but NOT for a bootstrap CI: the interval is CONSTRUCTED at alpha, so
    # it widens. Point estimates are bit-identical (same data, same seed); only the
    # interval widths move. That is a re-evaluation, not a re-run.
    previous_verdicts = {int(row['n']): row['verdict'] for row in base['rows']}
    verdict_changes = {n: (previous_verdicts[n], restated[n]['verdict'])
                       for n in sorted(restated)
                       if previous_verdicts[n] != restated[n]['verdict']}

    rows = []
    for row in base['rows']:
        new = dict(row)
        res = restated[int(row['n'])]
        new['alpha'] = FAMILY_ALPHA
        new['cleared_bar'] = res['cleared_bar']
        new['verdict'] = res['verdict']
        if 'ci_low' in res:
            new['ci_low'] = dc._r(res['ci_low'])
            new['ci_high'] = dc._r(res['ci_high'])
        if 'delta_acc_ci_low' in res:
            new['delta_acc_ci_low'] = dc._r(res['delta_acc_ci_low'])
            new['delta_acc_ci_high'] = dc._r(res['delta_acc_ci_high'])
        new['horizon_bars'] = dv.PRIMARY_HORIZON
        new['session_scope'] = SCOPE_SESSION
        new['mean_uniqueness'] = ''
        new['n_independent'] = ''
        new['min_detectable_edge_pips'] = ''
        new['notes'] = (f"{row['notes']} ALPHA RESTATED at the resized family bar "
                        f"0.05/6=0.008333. {FAMILY_RESIZE_NOTE}")
        rows.append(new)

    p = power[(SCOPE_ALL, H)]
    shared = {
        'date': pd.Timestamp.utcnow().date().isoformat(), 'arbiter': ARBITER_LABEL,
        'oscillators': '|'.join(sorted(dv.PRIMARY_OSCILLATORS)),
        'n_events_raw': base['counts']['n_events_raw'],
        'n_events_after_session_filter': int(len(val_all)),
        'n_events_train': int(len(sets[(SCOPE_ALL, H, 'train')])),
        'n_events_val': int(len(val_all)),
        'median_swing_gap': dc._r(base['median_swing_gap'], 3),
        'median_confirm_lag': dc._r(base['median_confirm_lag'], 3),
        'block_len': block_len, 'alpha': FAMILY_ALPHA,
        'device_used': base['device_info']['device'],
        'horizon_bars': H, 'session_scope': SCOPE_ALL,
        'mean_uniqueness': dc._r(primary_acc['mean_uniqueness']),
        'n_independent': primary_acc['n_independent'],
        'min_detectable_edge_pips': dc._r(p['min_detectable_edge_pips'], 4),
    }
    common_notes = (f"{HORIZON_REVISION_NOTE} {PRECOMMITMENT_NOTE} "
                    f"{FAMILY_RESIZE_NOTE} {SCOPE_NOTE} {dc.POOLING_NOTE} "
                    f"{dc.TIMING_NOTE}")

    rows.append({
        **shared, 'n': 5,
        'hypothesis': 'H_div.5_event_study_N96_all_hours',
        'mean_signed_pips': dc._r(h5['mean_signed_pips']),
        'ci_low': dc._r(h5['ci_low']), 'ci_high': dc._r(h5['ci_high']),
        'acc_challenger': '', 'acc_reference': '', 'delta_acc': '',
        'delta_acc_ci_low': '', 'delta_acc_ci_high': '', 'mcnemar_p': '',
        'shuffled_label_control_acc': dc._r(h6['shuffled_label_control_acc']),
        'cleared_bar': h5['cleared_bar'], 'verdict': h5['verdict'],
        'notes': (f"N=96 M15 bars (24h), ALL HOURS; win rate {h5['win_rate_pct']:.2f}%; "
                  f"mean uniqueness {primary_acc['mean_uniqueness']:.4f} -> "
                  f"n_independent {primary_acc['n_independent']}; MINIMUM DETECTABLE "
                  f"EDGE {p['min_detectable_edge_pips']:.2f} pips "
                  f"({100 * p['mde_as_fraction_of_mean_abs']:.1f}% of the "
                  f"{p['mean_abs_move_pips']:.1f} pip mean absolute move) -- a DROP "
                  f"here rules out an edge of that size and NOTHING SMALLER"
                  f"{'; THIS TEST IS UNDERPOWERED' if p['underpowered'] else ''}. "
                  f"block_len={block_len} (>= max(median gap "
                  f"{primary_acc['median_gap']:.0f}, N={H})). {common_notes}"),
    })
    rows.append({
        **shared, 'n': 6,
        'hypothesis': 'H_div.6_GBM_N96_all_hours_vs_train_majority',
        'n_events_train': h6['n_train'], 'n_events_val': h6['n_val'],
        'mean_signed_pips': dc._r(h5['mean_signed_pips']), 'ci_low': '', 'ci_high': '',
        'acc_challenger': dc._r(h6['acc_challenger']),
        'acc_reference': dc._r(h6['acc_reference']),
        'delta_acc': dc._r(h6['delta_acc']),
        'delta_acc_ci_low': dc._r(h6['delta_acc_ci_low']),
        'delta_acc_ci_high': dc._r(h6['delta_acc_ci_high']),
        'mcnemar_p': dc._r(h6['mcnemar_p']),
        'shuffled_label_control_acc': dc._r(h6['shuffled_label_control_acc']),
        'cleared_bar': h6['cleared_bar'], 'verdict': h6['verdict'],
        'notes': (f"XGBoost hist/device=cuda n_est300 depth4 lr.05 balanced "
                  f"scale_pos_weight, NO early_stopping and NO eval_set; features = "
                  f"STEP-0 continuous measures + one-hot oscillator + one-hot type; "
                  f"target sign(signed_return_pips) at N=96; reference = "
                  f"train-majority class {h6['majority_class']}; McNemar "
                  f"b={h6['mcnemar_b']} c={h6['mcnemar_c']}. {common_notes}"),
    })

    log = _write_family_log(rows, log_path) if register else pd.DataFrame(rows)
    family_closed = bool(not h5['cleared_bar'] and not h6['cleared_bar'])

    assert_sources_unmodified(hashes_before)
    return {
        'base': base, 'events': events, 'sets': sets, 'accounting': accounting,
        'power': power, 'block_len': block_len, 'block_len_hc': block_len_hc,
        'h_div_5': h5, 'h_div_6': h6, 'h5_breakdown': h5_breakdown,
        'h5_session': h5_session, 'h5_corroborating': h5_corroborating,
        'band': band, 'restated': restated, 'log': log,
        'previous_verdicts': previous_verdicts, 'verdict_changes': verdict_changes,
        'family_closed': family_closed, 'source_hashes': hashes_before,
        'device_info': base['device_info'], 'device': base['device'],
        'n_bars': len(ny_df), 'split': {'train_end': train_end, 'val_end': val_end},
    }


def _print_report(r):
    """STEP 8 report order. Independence and power LEAD, before any mean or CI."""
    H, HC = PRIMARY_HORIZON_BARS, CORROBORATING_HORIZON_BARS
    d = r['device_info']
    h5, h6 = r['h_div_5'], r['h_div_6']

    print('\n' + '=' * 82)
    print('DIVERGENCE AT A MULTI-DAY HORIZON — RESULTS (raw)')
    print('=' * 82)

    print('\n1. DEVICE')
    print(f"   CUDA available : {d['cuda_available']}")
    if d['cuda_available']:
        print(f"   CUDA device    : {d['cuda_device_name']}")
    print(f"   device used    : {r['device']}")
    print(f"   reused UNCHANGED (sha256 verified): "
          + ', '.join(PROTECTED_SOURCES))

    print('\n2. EVENT COUNTS (regular divergences; gap + test-block windows excluded)')
    print(f"   raw regular divergences        : {r['base']['counts']['n_events_raw']}")
    print(f"   {'scope':<24}{'N':>5}{'train':>9}{'val':>8}")
    for scope in (SCOPE_ALL, SCOPE_SESSION):
        for h in (H, HC):
            print(f"   {scope:<24}{h:>5}"
                  f"{len(r['sets'][(scope, h, 'train')]):>9}"
                  f"{len(r['sets'][(scope, h, 'val')]):>8}")
    print(f"   (for contrast, the N=4 primary arm had "
          f"{len(r['base']['val_events'])} validation events in the session scope)")

    print('\n3. INDEPENDENCE ACCOUNTING AND POWER  — READ BEFORE ANY RESULT')
    print(f"   {'scope':<24}{'N':>5}{'events':>8}{'med gap':>9}{'IQR':>16}"
          f"{'uniq':>8}{'n_indep':>9}")
    for scope in (SCOPE_ALL, SCOPE_SESSION):
        for h in (H, HC):
            a = r['accounting'][(scope, h)]
            iqr = f"[{a['q25']:.0f}, {a['q75']:.0f}]"
            print(f"   {scope:<24}{h:>5}{a['n_events']:>8}{a['median_gap']:>9.0f}"
                  f"{iqr:>16}{a['mean_uniqueness']:>8.4f}{a['n_independent']:>9}")

    p = r['power'][(SCOPE_ALL, H)]
    print(f"\n   PRE-REGISTERED POWER STATEMENT (primary arm: {SCOPE_ALL}, N={H})")
    print(f"     std of signed returns      : {p['std_signed_pips']:.2f} pips")
    print(f"     mean ABSOLUTE move         : {p['mean_abs_move_pips']:.2f} pips")
    print(f"     independent observations   : {p['n_independent']} "
          f"(from {r['accounting'][(SCOPE_ALL, H)]['n_events']} events x uniqueness "
          f"{r['accounting'][(SCOPE_ALL, H)]['mean_uniqueness']:.4f})")
    print(f"     standard error             : {p['standard_error_pips']:.2f} pips")
    print(f"     z at alpha={p['alpha']:.6f}      : {p['z_alpha']:.3f}")
    print(f"     >>> THIS TEST CAN DETECT A MEAN EDGE OF "
          f"{p['min_detectable_edge_pips']:.2f} PIPS PER EVENT AND NOTHING SMALLER")
    print(f"     that is {100 * p['mde_as_fraction_of_mean_abs']:.1f}% of the mean "
          f"absolute move at this horizon")
    if p['underpowered']:
        print("     >>> UNDERPOWERED (>25% of the typical move). A DROP here will")
        print("         mean 'NO LARGE EDGE', not 'no edge'. Read the verdict that way.")
    else:
        print("     >>> adequately powered by the pre-registered 25% criterion")

    print('\n4. SHUFFLED-LABEL CONTROL (leakage gate, no alpha)')
    print(f"   accuracy = {h6['shuffled_label_control_acc']:.4f}  -> "
          f"{'SANE — near chance' if h6['leak_control_sane'] else 'ANOMALOUS — VOID'}")

    print(f"\n5. H_div.5 — EVENT STUDY, N={H} (24h), ALL HOURS  "
          f"(alpha = {h5['alpha']:.6f})")
    print(f"   n events        : {h5['n_events']}")
    print(f"   mean signed     : {h5['mean_signed_pips']:+.4f} pips")
    print(f"   median signed   : {h5['median_signed_pips']:+.4f} pips")
    print(f"   std             : {h5['std_signed_pips']:.4f} pips")
    print(f"   win rate        : {h5['win_rate_pct']:.2f}%")
    print(f"   block_len       : {r['block_len']} bars")
    print(f"   block CI        : [{h5['ci_low']:+.4f}, {h5['ci_high']:+.4f}]  <- governs")
    print(f"   VERDICT         : {h5['verdict']}")

    print('\n   BREAKDOWN — DESCRIPTIVE ONLY, carries no verdict')
    for _, row in r['h5_breakdown'].iterrows():
        print(f"     {row['breakdown']:<11} {row['group']:<18} n={int(row['n']):<6} "
              f"mean {row['mean_signed_pips']:+.4f} pips   win {row['win_rate_pct']:.2f}%")
    s = r['h5_session']
    print(f"\n   07:00-10:00 NY SUBSET — DESCRIPTIVE ONLY, NOT a path to KEEP")
    print(f"     n={s['n_events']}  mean {s['mean_signed_pips']:+.4f} pips  "
          f"CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  win {s['win_rate_pct']:.2f}%")

    print(f"\n6. H_div.6 — GBM, N={H}, ALL HOURS  (alpha = {h6['alpha']:.6f})")
    print(f"   n train / val   : {h6['n_train']} / {h6['n_val']}   "
          f"(zero-move dropped: {h6['n_zero_dropped_train']} / {h6['n_zero_dropped_val']})")
    print(f"   class balance   : train {h6['train_class_balance_pct1']:.2f}% up, "
          f"val {h6['val_class_balance_pct1']:.2f}% up")
    print(f"   acc challenger  : {h6['acc_challenger']:.4f}")
    print(f"   acc reference   : {h6['acc_reference']:.4f}  "
          f"(train-majority class {h6['majority_class']})")
    print(f"   delta           : {h6['delta_acc']:+.4f}")
    print(f"   block CI        : [{h6['delta_acc_ci_low']:+.4f}, "
          f"{h6['delta_acc_ci_high']:+.4f}]  <- governs")
    print(f"   McNemar exact   : b={h6['mcnemar_b']} c={h6['mcnemar_c']} "
          f"p={h6['mcnemar_p']:.6g}")
    print(f"   VERDICT         : {h6['verdict']}")

    if len(r['band']):
        print(f"\n7. DECLARED SENSITIVITY BAND AT N={H} — descriptive, no alpha, no verdict")
        print(f"   {'label':<16}{'n':>7}{'mean pips':>12}{'CI low':>10}{'CI high':>10}"
              f"{'win%':>8}  default")
        for _, row in r['band'].iterrows():
            print(f"   {row['label']:<16}{int(row['n_events']):>7}"
                  f"{row['mean_signed_pips']:>+12.4f}{row['ci_low']:>+10.4f}"
                  f"{row['ci_high']:>+10.4f}{row['win_rate_pct']:>8.2f}"
                  f"   {'<- DEFAULT' if row['is_default'] else ''}")
        same = int((np.sign(r['band']['mean_signed_pips'])
                    == np.sign(h5['mean_signed_pips'])).sum())
        cleared = int(r['band']['would_have_cleared'].sum())
        print(f"   {same}/{len(r['band'])} members share the default's sign; "
              f"{cleared} would have cleared.")
        if cleared and not h5['cleared_bar']:
            print('   The default DROPped, so any member that would have cleared is')
            print('   PARAMETER FRAGILITY, not a KEEP.')

    c = r['h5_corroborating']
    print(f"\n8. N={HC} (48h) — CORROBORATING ONLY, NOT a path to KEEP")
    print(f"   n={c['n_events']}  mean {c['mean_signed_pips']:+.4f} pips  "
          f"CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  win {c['win_rate_pct']:.2f}%")
    pc = r['power'][(SCOPE_ALL, HC)]
    print(f"   n_independent {pc['n_independent']}, minimum detectable edge "
          f"{pc['min_detectable_edge_pips']:.2f} pips")

    print('\n9. VERDICTS  (family bar alpha = 0.05/6 = 0.008333, retroactive)')
    for n, name, res in ((1, 'H_div.1 pattern-only N=4', r['restated'][1]),
                         (2, 'H_div.2 GBM pattern-only', r['restated'][2]),
                         (3, 'H_div.3 triggered entry', r['restated'][3]),
                         (4, 'H_div.4 GBM triggered', r['restated'][4]),
                         (5, f'H_div.5 event study N={H}', h5),
                         (6, f'H_div.6 GBM N={H}', h6)):
        print(f"   {name:<32}: {res['verdict']}")
    ch = r['verdict_changes']
    print("\n   verdict changes at 0.008333 vs 0.0125: "
          + (', '.join(f"{k}: {a} -> {b}" for k, (a, b) in ch.items()) if ch
             else 'NONE — all four were already DROP at 0.0125 and remain DROP'))
    print('   (bootstrap CIs are CONSTRUCTED at alpha, so they widen; McNemar p is')
    print('    unchanged and only its threshold moves. Point estimates are identical.)')

    if r['family_closed']:
        print('\n   PRE-COMMITMENT HONOURED: both N=96 arms DROPped, so the')
        print('   DIVERGENCE FAMILY IS NOW CLOSED. No further horizon, oscillator,')
        print('   swing-detector or session variation will be run on it.')
    else:
        print('\n   At least one arm cleared; the family is not closed by the')
        print('   pre-commitment. See the report for what that does and does not mean.')


if __name__ == '__main__':
    _print_report(run())
