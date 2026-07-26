"""
Fractal-breakout DRIFT/CONTINUATION event-study — its OWN hypothesis family.

Genuinely different question from hypothesis #7 (`results/feature_hypothesis_log.csv`,
DROPped): #7 asked whether `fractal_breakout_up`/`fractal_breakout_down` help predict
the SINGLE NEXT day's direction as an input feature alongside everything else. This
asks the classic breakout-MOMENTUM thesis instead: conditional on a confirmed breakout
today, does price keep moving in that direction over the next few days? A forward
event-study, not a same-day feature-addition test — different target horizon, different
mechanism, so it gets its own family log (`results/fractal_breakout_driftcheck_hypothesis_log.csv`)
and does NOT touch `feature_hypothesis_log.csv`, `volatility_hypothesis_log.csv`,
`cot_weekly_hypothesis_log.csv`, or `harmonic_pattern_hypothesis_log.csv`.

Research-only, regardless of outcome: no model, no feature, no serving change at this
stage. A KEEP-signal here is only the TRIGGER to design a proper dedicated
event-conditional model later (mirroring how the harmonic-pattern H1 model was built,
`src/harmonic_event_check.py`) -- never an automatic feature/serving change itself.

Data & reuse
------------
Daily `results/eurusd_features.csv`, the same series hypothesis #7 used.
`confirmed_high_low_levels()` / `add_fibonacci_features()` from
`src.fibonacci_fractals` are reused UNCHANGED to get `fractal_breakout_up` /
`fractal_breakout_down` per day -- fractal detection itself is not rebuilt, and the
confirmation-lag look-ahead guard (a fractal is only knowable 2 bars after it forms)
is already baked into those columns by construction; no additional lag is added here.

PRE-REGISTERED design (fixed before looking at any result; run ONCE, no iterating)
-----------------------------------------------------------------------------------
Event         day t is an event iff `fractal_breakout_up[t]==1` XOR
              `fractal_breakout_down[t]==1`. `event_direction = +1` for a confirmed
              up-breakout, `-1` for a confirmed down-breakout. The rare day where BOTH
              flags fire simultaneously (direction ambiguous -- happens 3 times in the
              full 1971-2026 daily history) is excluded and counted separately, never
              arbitrarily assigned a sign.
Statistic     For each event day t and horizon N in {2, 3, 5}:
              `forward_return_N = log(close[t+N] / close[t])`
              `signed_continuation_N = event_direction * forward_return_N`
              (positive = the move continued in the breakout's direction).
Split         SAME chronological daily convention as every other family (from
              `config.json`: `train_fraction=0.80`, `val_fraction=0.10`) ->
              train[0:70%], validation[70%:80%], test[80%:100%] RESERVED and never
              read here, exactly like `src.ablation`.
Boundary      An event's forward window must stay INSIDE the slice it is being
              evaluated in. For a validation-slice event, `t+N` landing at or past
              `val_end` would read into the reserved test block even though the
              underlying CSV physically has more rows there (this is daily EURUSD
              history extending to the present, not a short series) -- such an event
              is excluded for that horizon, same as genuine "insufficient forward
              history" at the true end of the series. One boundary parameter
              (`max_idx`) in `compute_signed_continuation` enforces both cases
              identically -- see its docstring and `tests/test_unit.py`.
Test          PRIMARY (governs the verdict): mean(signed_continuation_3) over event
              days in the validation[70:80] slice, paired bootstrap (2000 resamples),
              95% CI. KEEP-signal ONLY if the CI is entirely > 0.
              CORROBORATING (context only, never a second path to KEEP): the same
              statistic for N=2 and N=5. If N=3 is null but N=2 or N=5 look
              significant, the verdict is still DROP -- same anti-cherry-pick
              convention as every other bundled hypothesis in this project (e.g.
              `harmonic_h1_2_mlp_vs_h1_1_primary`).
Alpha         0.05 -- the first (and, at this pass, only) hypothesis of this brand-new
              family. A future test in this family becomes #2 and tightens the bar.
Power         Report the raw validation-slice event count (up/down separately) before
              interpreting anything; if thin, state the caveat honestly rather than
              over-claim (mirrors the weekly-COT "~100 validation weeks -> only
              |rho|>0.2 detectable" precedent).

Run:  python -m src.fractal_breakout_driftcheck
"""
import os

import numpy as np
import pandas as pd

from src.fibonacci_fractals import add_fibonacci_features

DRIFTCHECK_LOG = "results/fractal_breakout_driftcheck_hypothesis_log.csv"
DRIFTCHECK_LOG_COLUMNS = [
    "n", "date", "hypothesis", "arbiter",
    "n_val_events_up", "n_val_events_down", "n_val_events_both_excluded",
    "n_used_2", "mean_signed_continuation_2", "ci_2_low", "ci_2_high",
    "n_used_3", "mean_signed_continuation_3", "ci_3_low", "ci_3_high",
    "n_used_5", "mean_signed_continuation_5", "ci_5_low", "ci_5_high",
    "alpha", "cleared_bar", "verdict", "notes",
]
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
HISTORY_CSV = "results/eurusd_features.csv"
HORIZONS = (2, 3, 5)
PRIMARY_HORIZON = 3


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def _canonical_split(n, train_fraction=0.80, val_fraction=0.10):
    """Identical formula to `src.ablation._canonical_split` / the project-wide
    chronological convention: train[0:70%], validation[70%:80%], test[80%:100%]."""
    train_end = int(n * (train_fraction - val_fraction))
    val_end = int(n * train_fraction)
    return train_end, val_end


def compute_signed_continuation(close, breakout_up, breakout_down,
                                horizons=HORIZONS, max_idx=None):
    """Per-event-day signed continuation statistics.

    An event day is any t where `breakout_up[t]` XOR `breakout_down[t]` is truthy;
    `event_direction = +1` for up, `-1` for down. Days where BOTH fire are excluded
    (direction undefined) and counted separately in the returned `both_excluded`.

    For each horizon N, `signed_continuation_N = event_direction * log(close[t+N]/close[t])`,
    computed ONLY if `t+N < max_idx` (defaults to `len(close)`, i.e. plain
    "insufficient forward history" at the true end of the series). Passing a
    validation slice's `val_end` as `max_idx` additionally guarantees the forward
    window never reads into the reserved test block, even when the underlying array
    has more rows beyond it — the SAME exclusion rule handles both cases so there is
    only one place look-ahead-safety can be gotten wrong.

    Returns (events: DataFrame indexed by event day position, with columns
    `event_direction` and `signed_continuation_{N}` per horizon — NaN where excluded
    for that horizon —, both_excluded: int).
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    if max_idx is None:
        max_idx = n
    up = np.asarray(breakout_up).astype(bool)
    down = np.asarray(breakout_down).astype(bool)
    both = up & down
    event_mask = (up | down) & ~both
    event_idx = np.where(event_mask)[0]
    direction = np.where(up[event_idx], 1, -1)

    rows = []
    for t, d in zip(event_idx, direction):
        row = {"idx": int(t), "event_direction": int(d)}
        for horizon in horizons:
            t_fwd = t + horizon
            if t_fwd < max_idx and t_fwd < n:
                row[f"signed_continuation_{horizon}"] = d * float(np.log(close[t_fwd] / close[t]))
            else:
                row[f"signed_continuation_{horizon}"] = np.nan
        rows.append(row)
    cols = ["idx", "event_direction"] + [f"signed_continuation_{h}" for h in horizons]
    events = pd.DataFrame(rows, columns=cols).set_index("idx")
    return events, int(both.sum())


def _bootstrap_mean_ci(x, n_boot=BOOTSTRAP_RESAMPLES, random_state=42):
    """Point mean + percentile 95% CI over paired bootstrap resamples of `x`
    (NaNs already dropped by the caller)."""
    x = np.asarray(x, dtype=float)
    point = float(np.mean(x))
    rng = np.random.default_rng(random_state)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = np.mean(x[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def build_validation_events(base_dir="", history_csv=HISTORY_CSV, horizons=HORIZONS):
    """Load the daily history, reuse `add_fibonacci_features` UNCHANGED, and return
    the validation-slice event table plus bookkeeping (train_end, val_end, n,
    raw up/down validation-slice event counts, both-flags-excluded count).

    `max_idx=val_end` is passed to `compute_signed_continuation` so a
    validation-slice event's forward window can never read into the reserved test
    block, per the pre-registered boundary rule."""
    daily = pd.read_csv(_p(base_dir, history_csv), index_col="time", parse_dates=True).sort_index()
    feat = add_fibonacci_features(daily)
    n = len(feat)
    train_end, val_end = _canonical_split(n)

    close = feat["close"].to_numpy(float)
    up = feat["fractal_breakout_up"].to_numpy()
    down = feat["fractal_breakout_down"].to_numpy()

    events, both_excluded = compute_signed_continuation(
        close, up, down, horizons=horizons, max_idx=val_end
    )
    val_events = events[(events.index >= train_end) & (events.index < val_end)]

    up_bool = up.astype(bool)
    down_bool = down.astype(bool)
    both_bool = up_bool & down_bool
    val_slice = slice(train_end, val_end)
    n_val_up = int((up_bool[val_slice] & ~both_bool[val_slice]).sum())
    n_val_down = int((down_bool[val_slice] & ~both_bool[val_slice]).sum())
    n_val_both = int(both_bool[val_slice].sum())

    return {
        "feat": feat, "n": n, "train_end": train_end, "val_end": val_end,
        "val_events": val_events,
        "n_val_up": n_val_up, "n_val_down": n_val_down, "n_val_both_excluded": n_val_both,
    }


def run(base_dir="", out_log=DRIFTCHECK_LOG, random_state=42, register=True):
    ctx = build_validation_events(base_dir=base_dir)
    val_events = ctx["val_events"]

    print("=" * 78)
    print("FRACTAL-BREAKOUT DRIFT/CONTINUATION EVENT-STUDY — new, own family (n=1)")
    print(f"  daily rows: {ctx['n']:,}  |  split: train[0:{ctx['train_end']}]  "
          f"val[{ctx['train_end']}:{ctx['val_end']}]  test[{ctx['val_end']}:{ctx['n']}] RESERVED")
    print(f"  validation-slice raw event counts: breakout_up={ctx['n_val_up']}  "
          f"breakout_down={ctx['n_val_down']}  (both-flags-same-day excluded={ctx['n_val_both_excluded']})")
    n_val_total = ctx['n_val_up'] + ctx['n_val_down']
    print(f"  total usable validation events (direction well-defined): {n_val_total}")
    if n_val_total < 100:
        print("  POWER CAVEAT: fewer than 100 validation events -- a null here is weak "
              "evidence of absence, not proof of no effect.")
    print(f"  alpha = {FAMILY_ALPHA} (first hypothesis of this family)")
    print("=" * 78)

    stats = {}
    for horizon in HORIZONS:
        col = f"signed_continuation_{horizon}"
        x = val_events[col].dropna().to_numpy()
        point, lo, hi = _bootstrap_mean_ci(x, random_state=random_state)
        excl0_positive = lo > 0
        stats[horizon] = {"n_used": len(x), "mean": point, "ci_low": lo, "ci_high": hi,
                          "signal": excl0_positive}
        tag = "PRIMARY" if horizon == PRIMARY_HORIZON else "corroborating"
        print(f"\n--- N={horizon} ({tag}) --- n_used={len(x)}")
        print(f"  mean(signed_continuation_{horizon}) = {point:+.6f}  "
              f"95% CI[{lo:+.6f}, {hi:+.6f}]  "
              f"{'CI ENTIRELY > 0' if excl0_positive else 'CI does not exclude/straddle-or-below 0'}")

    primary = stats[PRIMARY_HORIZON]
    if primary["signal"]:
        verdict = ("KEEP-signal — primary N=3 signed-continuation 95% CI is entirely above 0; "
                   "TRIGGER to design a proper dedicated event-conditional breakout-momentum "
                   "model later (mirrors src/harmonic_event_check.py), NOT an automatic "
                   "feature/serving change")
    else:
        verdict = ("DROP — no continuation edge: primary N=3 CI does not clear the bar "
                   "(N=2/N=5 are corroborating context only, not a second path to KEEP "
                   "if N=3 is null)")
    print(f"\n  VERDICT: {verdict}")
    if n_val_total < 100:
        print(f"  (power caveat: only {n_val_total} validation events — a null is weak "
              "evidence of absence)")

    row = {
        "n": 1, "date": pd.Timestamp.utcnow().date().isoformat(),
        "hypothesis": "fractal_breakout_continuation_3day",
        "arbiter": "validation[70:80]",
        "n_val_events_up": ctx["n_val_up"], "n_val_events_down": ctx["n_val_down"],
        "n_val_events_both_excluded": ctx["n_val_both_excluded"],
        "n_used_2": stats[2]["n_used"],
        "mean_signed_continuation_2": round(stats[2]["mean"], 6),
        "ci_2_low": round(stats[2]["ci_low"], 6), "ci_2_high": round(stats[2]["ci_high"], 6),
        "n_used_3": stats[3]["n_used"],
        "mean_signed_continuation_3": round(stats[3]["mean"], 6),
        "ci_3_low": round(stats[3]["ci_low"], 6), "ci_3_high": round(stats[3]["ci_high"], 6),
        "n_used_5": stats[5]["n_used"],
        "mean_signed_continuation_5": round(stats[5]["mean"], 6),
        "ci_5_low": round(stats[5]["ci_low"], 6), "ci_5_high": round(stats[5]["ci_high"], 6),
        "alpha": FAMILY_ALPHA,
        "cleared_bar": bool(primary["signal"]),
        "verdict": verdict,
        "notes": (f"event=confirmed fractal breakout (fractal_breakout_up XOR "
                  f"fractal_breakout_down, from src.fibonacci_fractals.add_fibonacci_features, "
                  f"unchanged), event_direction=+1/-1, signed_continuation_N=direction*"
                  f"log(close[t+N]/close[t]) for N in {{2,3,5}}. Different question from "
                  f"hypothesis #7 (feature_hypothesis_log.csv, DROPped, next-single-day input "
                  f"feature) -- this is a forward multi-day event-study, own family. "
                  f"Validation-slice events whose forward window would cross into the reserved "
                  f"test block are excluded (not padded/estimated), same rule as genuine "
                  f"insufficient-history exclusion. PRIMARY=N=3 mean signed continuation, "
                  f"paired bootstrap ({BOOTSTRAP_RESAMPLES} resamples) 95% CI, decision = CI "
                  f"entirely > 0. N=2/N=5 corroborating only, anti-cherry-pick. Research-only: "
                  f"no model/feature/serving change regardless of outcome; a KEEP-signal is "
                  f"only the trigger to design a dedicated event-conditional model later."),
    }
    out_path = _p(base_dir, out_log)
    new = pd.DataFrame([row], columns=DRIFTCHECK_LOG_COLUMNS)
    if register:
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        new.to_csv(out_path, index=False)
        print(f"\nLogged: {out_path}")
    return row


if __name__ == "__main__":
    run()
