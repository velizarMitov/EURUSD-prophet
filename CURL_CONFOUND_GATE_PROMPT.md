# Prompt — confound gate (paste into Claude in VS Code)

---

The Day-1 battery is running. In parallel, we are closing the confound leak that would
otherwise make the primary hypothesis a liquidity proxy in a Hodge costume.

Two new functions are already implemented and tested in `src/curl_mt5.py`:

- `causal_residualise_against_confounds` — refits the confound regression on a **trailing
  window** (`max_train` lookback, `refit_every` blocks) instead of one static fit on
  `[0:70%]`. Quantile edges are recomputed per block from the trailing window only.
  Design = linear terms + per-variable quantile dummies + a 2-D (volatility × activity)
  grid + hour/day dummies. The 2-D grid matters: the null is a function of
  `variance / tick count`, so mis-specification leaves a *joint* residual that no additive
  model removes.
- `confound_gate` — measures what survives, on `validation[70:80]` only. Reports marginal
  correlations per confound, hour-of-day eta, and **joint R²** of the residual on the full
  confound design. Joint R² is the number that matters; marginals can each look small while
  the confounds jointly explain the index.

Measured on synthetic data with a 3.5× tick-level drift imposed:

| design | worst \|corr\| | joint R² |
|---|---|---|
| static additive fit on `[0:70%]` | 0.272 | 0.234 |
| causal trailing refit + 2-D grid | 0.152 | 0.070 |

## The thing you must not do

**Do not chase joint R² → 0.01.** That threshold was my initial guess and it is almost
certainly unreachable. On synthetic data where the asynchronicity null is **true by
construction**, the causally residualised index still shows joint R² ≈ 0.07. Some confound
structure is inherent to the estimator: the excess-curl distribution depends on volatility
and tick count through the chi-square shape, the truncation regime, and the Parkinson
discreteness bias, none of which a linear-in-P calibration captures, and the rolling smooth
amplifies the systematic part. Tuning against zero would mean over-fitting the
residualiser until it destroys real signal along with the artifact.

Judge against what a true null produces — the same logic as the permutation control in
`staleness_exponent_test`.

## Task A — establish the floor (do this first, before touching real data)

The floor is itself noisy. Across three seeds I measured joint R² = 0.070, 0.106, 0.066.
One draw is not a reference.

Write `src/curl_confound_reference.py`:

1. Generate **at least 8 seeds** of synthetic M1 via `curl_null_simulation.simulate_bars`,
   matched to the real feed: same bar count as the real common index (or ≥300k), tick rates
   set so per-pair median tick counts match the real medians (EURUSD 44, EURJPY 100,
   EURGBP 40, GBPUSD 56, GBPJPY 100, USDJPY 53), bid-side half-spreads matched to the real
   medians (0.46–1.31 bp), and `activity_trend` matching the observed per-year drift.
2. For each seed run `calculate_causal_excess_curl` → `causal_residualise_against_confounds`
   → `confound_gate`, with the **same parameters you will use on real data**.
3. Write `results/curl/confound_reference.csv` and report the distribution. Use the **90th
   percentile** of joint R² as the reference, not the mean — we want an upper envelope.

The synthetic feed must be generated with the null TRUE (no `dislocated_pair`). That is the
whole point: it tells us what "nothing there" looks like through our own estimator.

## Task B — run the gate on real data

```python
res, calib = m5.calculate_causal_excess_curl(base, ...)
orth = m5.causal_residualise_against_confounds(res["stress"], base,
                                               min_train=..., refit_every=..., max_train=...)
table, summary = m5.confound_gate(orth, base, reference_joint_r2=REFERENCE_P90)
```

Sizing for ~2.96M bars: `min_train=250_000`, `refit_every=20_000`, `max_train=250_000`
(≈135 refits; the 300k-bar test ran in ~12s, so expect a couple of minutes). Report wall
time — if it is far off that, something is quadratic and I want to know.

Write `results/curl/confound_gate.csv` and print the table and summary verbatim.

## Decision rules

- **Real joint R² within the synthetic-null budget** → the residual index is as clean as
  this estimator can be. Proceed to pre-register the primary, recording both the real figure
  and the reference in `results/curl_hypothesis_log.csv`.
- **Real joint R² materially above the budget** → genuine confound structure beyond the
  estimator's floor. Do not score the primary. Report which confound dominates (my last run
  had `hour_of_day` eta at 0.152 as the worst single axis, above `log_range` at 0.108).
- **Real joint R² far below the floor** → suspicious, not good news. Check that the
  residualiser is not simply annihilating the index; report the variance of `stress` before
  and after residualisation.

## Constraints

- Everything causal. Quantile edges, coefficients, calendar means — trailing window only.
  There is a `test_causal_excess_curl_has_no_lookahead` pattern in
  `tests/test_curl_stress.py`; add the equivalent for
  `causal_residualise_against_confounds` (poison future rows, assert earlier values
  unchanged).
- The gate is a **diagnostic, not a hypothesis**. It spends no slot from
  `results/curl_hypothesis_log.csv`. Run it as often as needed.
- Do not tune `grid_bins`, `n_tick_bins`, `smooth` or `max_train` against the real gate
  result. That is the residualiser eating the signal. Fix them from the synthetic reference
  and leave them alone.
- Show the numbers before the narrative. All 28 tests must still pass.

Start with Task A and show me the eight-seed distribution before running anything on the
real data.
