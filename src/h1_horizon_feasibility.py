"""
H1 label-geometry FEASIBILITY SCAN — a design calculation, NOT a hypothesis test.

WHY THIS EXISTS
---------------
The pooled-H1 program (results/pooled_h1_hypothesis_log.csv, both DROP) surfaced
the real constraint: on the 120-bar triple-barrier design the mean label
uniqueness is ~0.008, so 192,560 pooled training rows carry only ~1,600
INDEPENDENT labels — three orders of magnitude below where sequential DL is
shown to work on intraday data. Pooling adds rows, not independence.

This scan answers ONE design question with ARITHMETIC, not models: does any
(horizon H, target-multiplier m) cell simultaneously yield (a) enough
independent labels to train on AND (b) a target barrier wide enough that a
1.5-pip cost does not eat it? If no cell satisfies both, the H1 triple-barrier
direction approach is closed on arithmetic and we stop testing it model by model.

HARD METHODOLOGICAL GUARDRAILS (enforced here)
----------------------------------------------
* NOT a hypothesis test. Writes NO hypothesis log, consumes NO Bonferroni alpha.
* Computes NO predictive metric (no accuracy/AUC/MAE, not even a majority
  baseline). Output is restricted to LABEL GEOMETRY: uniqueness, barrier width,
  resolution time, class balance, cost ratio.
* TRAIN SLICE ONLY — exactly the [0:70%] global chronological train slice of the
  intersected window that src/pooled_h1_model.py established. Every labelled
  entry's full resolution horizon is required to stay strictly inside the train
  slice (an entry is dropped otherwise), so the scan NEVER indexes the
  validation slice [70:85%] and never touches the reserved test block.
* No model, no GPU, no MT5 refetch — reuses the already-cached
  results/pooled_h1/*.csv. Missing file => STOP.

BOUNDARY: research only. Modifies nothing under models/, and none of
_train_pipeline.py, src/inference.py, src/features.py, src/paper_trading.py,
config.json, results/eurusd_h1.csv, src/triple_barrier.py, src/pooled_h1_model.py,
src/pooled_h1_data.py. New files only: this module,
results/h1_horizon_feasibility.csv, and tests.

THE NOMINAL-vs-ACTUAL UNIQUENESS DISTINCTION (the whole point)
-------------------------------------------------------------
src/pooled_h1_model.py computed its logged mean_label_uniqueness (0.008275) from
each label's NOMINAL max horizon (entry+H) — every label spans exactly H bars,
so concurrency ~= H and uniqueness ~= 1/H. THIS scan instead uses each label's
ACTUAL resolution time t1 (early barrier hits resolve sooner than H), which is
the honest information content. `uniqueness_from_spans` is validated against the
logged 0.008275 by feeding it the SAME nominal spans (STEP 4 anchor), then the
scan proper uses actual-t1 spans.
"""

import os

import numpy as np
import pandas as pd

from src.triple_barrier import ewma_log_return_std, triple_barrier_label
from src.pooled_h1_data import POOLED_PAIRS, PIP_SIZE, build_pooled_pairs
from src.pooled_h1_model import build_pooled_dataset, global_split_boundaries, EWMA_SPAN

# ── Pre-registered grid (STEP 1) ──
HORIZONS = (6, 12, 24, 48, 120, 240)
TARGET_MULTS = (0.5, 1.0, 1.5, 2.0)
STOP_MULT = 1.0                       # fixed throughout; the grid isolates target width
COST_PRICE = 1.5 * PIP_SIZE           # 1.5 pips = 0.00015 price units

OUT_CSV = 'results/h1_horizon_feasibility.csv'
ANCHOR_CELL = (120, 1.5)
ANCHOR_LOGGED_UNIQUENESS = 0.008275
ANCHOR_TOL = 0.0005

# ── Pre-registered thresholds (STEP 3) — written before running ──
IND_NN = 50_000        # plausible for a sequential NN
IND_GBM = 5_000        # plausible for a GBM over ~14 features
IND_MARGINAL = 1_000   # marginal — small regularised linear model at best
COST_COMFORTABLE = 10.0
COST_FATAL = 25.0
# VIABLE iff n_independent >= IND_GBM AND median cost_ratio <= COST_FATAL.

CSV_COLUMNS = [
    'horizon_bars', 'target_mult', 'stop_mult', 'scope', 'n_rows',
    'mean_uniqueness', 'n_independent', 'mean_resolution_bars',
    'median_resolution_bars', 'pct_resolved_target', 'pct_resolved_stop',
    'pct_resolved_time', 'median_target_width_pips', 'p25_target_width_pips',
    'cost_ratio_median_pct', 'cost_ratio_p25_pct', 'class_balance_pct_label1',
    'tier_independence', 'tier_cost', 'viability',
]


def price_to_pips(price_move: float) -> float:
    """Convert a raw price move to pips (1 pip = 0.0001 = PIP_SIZE). Guards the
    classic FX unit error: 0.00015 -> 1.5 pips, 0.00300 -> 30.0 pips."""
    return price_move / PIP_SIZE


# ───────────────────────── uniqueness (validated, span-based) ─────────────────

def uniqueness_from_spans(starts, ends, grid_len):
    """
    Lopez de Prado mean label uniqueness for labels occupying integer bar spans
    [start_i, end_i] (inclusive) on a shared position grid of length `grid_len`.
    concurrency c_t = number of labels live at bar t; a label's uniqueness is the
    average of 1/c_t over its own span; the return is the mean over labels.

    Vectorised: a +1/-1 difference array builds concurrency in O(L+grid); a
    cumulative sum of 1/concurrency gives each label's span-average in O(1).
    Does NOT test itself — validated by a hand-computed unit test and by the
    STEP 4 anchor.
    """
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    if len(starts) == 0:
        return float('nan')

    conc = np.zeros(grid_len + 1, dtype=np.float64)
    np.add.at(conc, starts, 1.0)
    np.add.at(conc, ends + 1, -1.0)
    conc = np.cumsum(conc)[:grid_len]
    conc[conc == 0.0] = 1.0

    inv = 1.0 / conc
    csum = np.concatenate([[0.0], np.cumsum(inv)])   # prefix sums of 1/c
    span_sum = csum[ends + 1] - csum[starts]
    span_len = (ends - starts + 1).astype(np.float64)
    return float(np.mean(span_sum / span_len))


# ───────────────────────── per-cell barrier geometry ─────────────────────────

def label_cell(full: pd.DataFrame, entry_pos: np.ndarray, horizon_vol: np.ndarray,
               H: int, m: float):
    """
    Vectorised triple-barrier first-crossing for one (H, m) cell on one
    instrument's FULL OHLC arrays, for the given entry positions (all guaranteed
    to have entry+H strictly inside the train slice). Long direction (+1), stop
    fixed at 1.0*horizon_vol, target at m*horizon_vol; same-bar tie resolves to
    the STOP and the time barrier uses the cost-aware rule — identical to
    src/triple_barrier.triple_barrier_label (verified on a sample in `run`).

    Returns a dict of per-entry arrays: label, t1 (actual resolution bars),
    outcome ('target'/'stop'/'time'), target_width_price.
    """
    high = full['high'].to_numpy(float)
    low = full['low'].to_numpy(float)
    close = full['close'].to_numpy(float)

    entry_close = close[entry_pos]
    hv = horizon_vol[entry_pos]
    target = entry_close * np.exp(m * hv)
    stop = entry_close * np.exp(-STOP_MULT * hv)

    E = len(entry_pos)
    resolved = np.zeros(E, dtype=bool)
    t1 = np.full(E, H, dtype=np.int64)
    label = np.zeros(E, dtype=np.int64)
    outcome = np.full(E, 'time', dtype=object)

    for k in range(1, H + 1):
        active = ~resolved
        if not active.any():
            break
        aidx = np.where(active)[0]
        kp = entry_pos[aidx] + k
        hit_stop = low[kp] <= stop[aidx]
        hit_target = high[kp] >= target[aidx]
        res_stop = hit_stop                      # stop priority (tie -> stop)
        res_target = hit_target & ~hit_stop
        si = aidx[res_stop]
        ti = aidx[res_target]
        t1[si] = k; label[si] = 0; resolved[si] = True; outcome[si] = 'stop'
        t1[ti] = k; label[ti] = 1; resolved[ti] = True; outcome[ti] = 'target'

    un = ~resolved
    if un.any():
        exit_close = close[entry_pos[un] + H]
        signed = exit_close - entry_close[un]    # direction +1
        win = signed > COST_PRICE
        label[un] = win.astype(np.int64)
        t1[un] = H
        outcome[un] = 'time'

    return {
        'label': label, 't1': t1, 'outcome': outcome,
        'target_width_price': target - entry_close,
        'entry_pos': entry_pos,
    }


def _cell_metrics(res, grid_len, use_nominal_uniqueness=False, H=None):
    """Geometry metrics for one cell/scope from a label_cell result (or a merged
    multi-instrument result). `use_nominal_uniqueness` spans each label over the
    NOMINAL H (the pooled program's definition, for the anchor); otherwise the
    ACTUAL t1 (the scan's definition)."""
    label = res['label']; t1 = res['t1']; outcome = res['outcome']
    width_price = res['target_width_price']; entry_pos = res['entry_pos']
    n = len(label)

    span_end = entry_pos + (H if use_nominal_uniqueness else t1)
    mean_uniq = uniqueness_from_spans(entry_pos, span_end, grid_len)

    width_pips = width_price / PIP_SIZE
    med_w = float(np.median(width_pips))
    p25_w = float(np.percentile(width_pips, 25))

    return {
        'n_rows': int(n),
        'mean_uniqueness': mean_uniq,
        'n_independent': int(round(n * mean_uniq)),
        'mean_resolution_bars': float(np.mean(t1)),
        'median_resolution_bars': float(np.median(t1)),
        'pct_resolved_target': 100.0 * np.mean(outcome == 'target'),
        'pct_resolved_stop': 100.0 * np.mean(outcome == 'stop'),
        'pct_resolved_time': 100.0 * np.mean(outcome == 'time'),
        'median_target_width_pips': med_w,
        'p25_target_width_pips': p25_w,
        'cost_ratio_median_pct': 1.5 / med_w * 100.0 if med_w > 0 else float('inf'),
        'cost_ratio_p25_pct': 1.5 / p25_w * 100.0 if p25_w > 0 else float('inf'),
        'class_balance_pct_label1': 100.0 * float(np.mean(label == 1)),
    }


# ───────────────────────── tiering (STEP 3, mechanical) ───────────────────────

def _tier_independence(n_ind):
    if n_ind >= IND_NN:
        return 'NN_plausible'
    if n_ind >= IND_GBM:
        return 'GBM_plausible'
    if n_ind >= IND_MARGINAL:
        return 'marginal'
    return 'not_trainable'


def _tier_cost(cost_ratio_median):
    if cost_ratio_median <= COST_COMFORTABLE:
        return 'comfortable'
    if cost_ratio_median <= COST_FATAL:
        return 'marginal'
    return 'fatal'


def _viability(n_ind, cost_ratio_median):
    if n_ind >= IND_GBM and cost_ratio_median <= COST_FATAL:
        return 'VIABLE'
    if n_ind >= IND_MARGINAL and cost_ratio_median <= COST_FATAL:
        return 'MARGINAL'
    return 'CLOSED'


# ───────────────────────── orchestration ─────────────────────────────────────

def _prepare_inputs(out_dir='results/pooled_h1'):
    """Reproduce the EXACT intersected common window + train boundary that
    src/pooled_h1_model established, and return per-instrument full OHLC arrays,
    causal per-bar EWMA std, and the train-entry positions (whose full horizon
    stays inside the train slice, per the max horizon in the grid)."""
    data = build_pooled_dataset(out_dir=out_dir)
    common = data['common']
    train_end_ts, _val_end_ts = global_split_boundaries(common)

    pairs = build_pooled_pairs(out_dir=out_dir, write_chfusd=False)
    common_set = set(common)

    inputs = {}
    for inst in POOLED_PAIRS:
        full = pairs[inst].sort_index()
        idx = full.index
        per_bar_std = ewma_log_return_std(full['close'].to_numpy(float), EWMA_SPAN)
        # first full-series position at/after the train/val boundary
        train_end_pos = int(np.searchsorted(idx.values, np.datetime64(train_end_ts)))
        in_common = np.array([ts in common_set for ts in idx])
        before_boundary = idx.values < np.datetime64(train_end_ts)
        finite_vol = np.isfinite(per_bar_std) & (per_bar_std > 0)
        base_eligible = in_common & before_boundary & finite_vol
        inputs[inst] = {
            'full': full, 'idx': idx, 'per_bar_std': per_bar_std,
            'train_end_pos': train_end_pos, 'base_eligible': base_eligible,
            'grid_len': len(idx),
        }
    return inputs, common, train_end_ts


def _eligible_entries(inp, H):
    """Entry positions eligible for horizon H: base-eligible AND entry+H strictly
    before the train/val boundary position (so the whole resolution window stays
    in the train slice -- validation is never indexed)."""
    n = len(inp['idx'])
    pos = np.arange(n)
    horizon_in_train = (pos + H) < inp['train_end_pos']
    mask = inp['base_eligible'] & horizon_in_train
    return pos[mask]


def _verify_against_triple_barrier(inp, entry_pos, H, m, sample=200, seed=0):
    """Sanity: the vectorised first-crossing must agree with the UNCHANGED
    src.triple_barrier.triple_barrier_label on a random sample of entries."""
    rng = np.random.default_rng(seed)
    if len(entry_pos) == 0:
        return True
    pick = rng.choice(len(entry_pos), size=min(sample, len(entry_pos)), replace=False)
    high = inp['full']['high'].to_numpy(float)
    low = inp['full']['low'].to_numpy(float)
    close = inp['full']['close'].to_numpy(float)
    hv = inp['per_bar_std'] * np.sqrt(H)
    res = label_cell(inp['full'], entry_pos[pick], hv, H, m)
    for j, gi in enumerate(pick):
        i = int(entry_pos[gi])
        lab, _out = triple_barrier_label(
            high, low, close, entry_idx=i, direction=1,
            horizon_vol=float(hv[i]), horizon_bars=H, cost_price=COST_PRICE,
            target_mult=m, stop_mult=STOP_MULT,
        )
        if lab is None or int(lab) != int(res['label'][j]):
            return False
    return True


def run(out_dir='results/pooled_h1', out_csv=OUT_CSV, write=True, verbose=True,
        enforce_anchor=True):
    """
    Full feasibility scan. Returns (rows_df, anchor). Writes
    results/h1_horizon_feasibility.csv and prints a compact pooled grid.
    Touches no hypothesis log and no protected file.

    `enforce_anchor` (default True) makes the STEP 4 anchor a HARD GATE on real
    data. It is only set False by the file-integrity unit tests, which feed tiny
    synthetic inputs that cannot reproduce the logged 0.008275 and are checking
    write-path safety, not the anchor.
    """
    inputs, common, train_end_ts = _prepare_inputs(out_dir)

    # STEP 4 anchor FIRST: reproduce logged 0.008275 at (120,1.5) from NOMINAL
    # spans; also report the ACTUAL-t1 value the scan uses.
    Hc, mc = ANCHOR_CELL
    anchor_inst = 'EURUSD'
    ap = _eligible_entries(inputs[anchor_inst], Hc)
    assert _verify_against_triple_barrier(inputs[anchor_inst], ap, Hc, mc), \
        "vectorised first-crossing disagrees with triple_barrier_label"
    ahv = inputs[anchor_inst]['per_bar_std'] * np.sqrt(Hc)
    ares = label_cell(inputs[anchor_inst]['full'], ap, ahv, Hc, mc)
    grid_len = inputs[anchor_inst]['grid_len']
    anchor_nominal = uniqueness_from_spans(ap, ap + Hc, grid_len)
    anchor_actual = uniqueness_from_spans(ap, ap + ares['t1'], grid_len)
    anchor = {
        'nominal_uniqueness': anchor_nominal,
        'actual_uniqueness': anchor_actual,
        'logged': ANCHOR_LOGGED_UNIQUENESS,
        'nominal_reproduces': abs(anchor_nominal - ANCHOR_LOGGED_UNIQUENESS) <= ANCHOR_TOL,
    }
    if verbose:
        print('=' * 72)
        print('STEP 4 ANCHOR — (H=120, m=1.5), EURUSD train slice')
        print(f"  logged (pooled_h1_model, NOMINAL horizon) = {ANCHOR_LOGGED_UNIQUENESS}")
        print(f"  this scan, NOMINAL spans                  = {anchor_nominal:.6f}  "
              f"(|diff|={abs(anchor_nominal-ANCHOR_LOGGED_UNIQUENESS):.6f}, tol={ANCHOR_TOL})")
        print(f"  -> implementation validated: {anchor['nominal_reproduces']}")
        print(f"  this scan, ACTUAL-t1 spans (what the scan uses) = {anchor_actual:.6f}")
        print('=' * 72)

    if enforce_anchor and not anchor['nominal_reproduces']:
        raise RuntimeError(
            f"ANCHOR FAILED: nominal uniqueness {anchor_nominal:.6f} does not reproduce "
            f"logged {ANCHOR_LOGGED_UNIQUENESS} within {ANCHOR_TOL}. Uniqueness "
            "implementation is wrong; stopping per STEP 4."
        )

    rows = []
    for H in HORIZONS:
        for m in TARGET_MULTS:
            per_inst_res = {}
            for inst in POOLED_PAIRS:
                inp = inputs[inst]
                ep = _eligible_entries(inp, H)
                hv = inp['per_bar_std'] * np.sqrt(H)
                res = label_cell(inp['full'], ep, hv, H, m)
                per_inst_res[inst] = (res, inp['grid_len'])
                mt = _cell_metrics(res, inp['grid_len'], H=H)
                rows.append(_row(H, m, inst, mt))

            # pooled scope: temporal uniqueness = rows-weighted mean of the
            # per-instrument (within-instrument) uniqueness -- matches the pooled
            # program's own accounting (192,560 rows * single-pair uniqueness);
            # cross-sectional dependence is the separate rho_bar/k_eff story.
            pooled_mt = _pooled_metrics(per_inst_res, H)
            rows.append(_row(H, m, 'pooled', pooled_mt))

            if verbose:
                print(f"  scanned H={H:>3} m={m:>3}  pooled n_ind={pooled_mt['n_independent']:>7}  "
                      f"cost_med={pooled_mt['cost_ratio_median_pct']:.1f}%  {pooled_mt['viability']}")

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if write:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)

    if verbose:
        _print_pooled_grid(df)

    return df, anchor


def _row(H, m, scope, mt):
    cost_med = mt['cost_ratio_median_pct']
    ti = _tier_independence(mt['n_independent'])
    tc = _tier_cost(cost_med)
    via = _viability(mt['n_independent'], cost_med)
    return {
        'horizon_bars': H, 'target_mult': m, 'stop_mult': STOP_MULT, 'scope': scope,
        'n_rows': mt['n_rows'], 'mean_uniqueness': round(mt['mean_uniqueness'], 6),
        'n_independent': mt['n_independent'],
        'mean_resolution_bars': round(mt['mean_resolution_bars'], 3),
        'median_resolution_bars': round(mt['median_resolution_bars'], 1),
        'pct_resolved_target': round(mt['pct_resolved_target'], 2),
        'pct_resolved_stop': round(mt['pct_resolved_stop'], 2),
        'pct_resolved_time': round(mt['pct_resolved_time'], 2),
        'median_target_width_pips': round(mt['median_target_width_pips'], 3),
        'p25_target_width_pips': round(mt['p25_target_width_pips'], 3),
        'cost_ratio_median_pct': round(cost_med, 3),
        'cost_ratio_p25_pct': round(mt['cost_ratio_p25_pct'], 3),
        'class_balance_pct_label1': round(mt['class_balance_pct_label1'], 2),
        'tier_independence': ti, 'tier_cost': tc, 'viability': via,
    }


def _pooled_metrics(per_inst_res, H):
    """Merge the four instruments' entries into one pooled geometry summary.
    Uniqueness is the rows-weighted mean of per-instrument uniqueness (see run
    docstring); all distributional metrics pool every entry."""
    labels, t1s, outcomes, widths = [], [], [], []
    uniq_weighted_num, total_rows = 0.0, 0
    for inst, (res, grid_len) in per_inst_res.items():
        labels.append(res['label']); t1s.append(res['t1'])
        outcomes.append(res['outcome']); widths.append(res['target_width_price'])
        u = uniqueness_from_spans(res['entry_pos'], res['entry_pos'] + res['t1'], grid_len)
        uniq_weighted_num += u * len(res['label'])
        total_rows += len(res['label'])

    label = np.concatenate(labels); t1 = np.concatenate(t1s)
    outcome = np.concatenate(outcomes); width_pips = np.concatenate(widths) / PIP_SIZE
    mean_uniq = uniq_weighted_num / total_rows if total_rows else float('nan')
    med_w = float(np.median(width_pips)); p25_w = float(np.percentile(width_pips, 25))

    return {
        'n_rows': int(total_rows), 'mean_uniqueness': mean_uniq,
        'n_independent': int(round(total_rows * mean_uniq)),
        'mean_resolution_bars': float(np.mean(t1)),
        'median_resolution_bars': float(np.median(t1)),
        'pct_resolved_target': 100.0 * np.mean(outcome == 'target'),
        'pct_resolved_stop': 100.0 * np.mean(outcome == 'stop'),
        'pct_resolved_time': 100.0 * np.mean(outcome == 'time'),
        'median_target_width_pips': med_w, 'p25_target_width_pips': p25_w,
        'cost_ratio_median_pct': 1.5 / med_w * 100.0 if med_w > 0 else float('inf'),
        'cost_ratio_p25_pct': 1.5 / p25_w * 100.0 if p25_w > 0 else float('inf'),
        'class_balance_pct_label1': 100.0 * float(np.mean(label == 1)),
        'viability': _viability(int(round(total_rows * mean_uniq)),
                                1.5 / med_w * 100.0 if med_w > 0 else float('inf')),
    }


def _print_pooled_grid(df):
    pooled = df[df['scope'] == 'pooled']
    print('\n' + '=' * 72)
    print('POOLED SUMMARY GRID — n_independent / cost_ratio_median% / viability')
    print('=' * 72)
    header = 'H \\ m  ' + '  '.join(f"{m:>22}" for m in TARGET_MULTS)
    print(header)
    for H in HORIZONS:
        cells = []
        for m in TARGET_MULTS:
            r = pooled[(pooled['horizon_bars'] == H) & (pooled['target_mult'] == m)].iloc[0]
            cells.append(f"{r['n_independent']:>7}/{r['cost_ratio_median_pct']:>6.1f}%/{r['viability'][:4]:>4}")
        print(f"{H:>5}  " + '  '.join(f"{c:>22}" for c in cells))
    print('=' * 72)


if __name__ == '__main__':
    run()
