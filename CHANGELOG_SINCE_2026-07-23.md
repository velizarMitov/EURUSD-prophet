# Changes since the first exam submission (23 July 2026)

**Baseline commit:** `64f6285` (21 July 2026) — the state submitted on 23 July 2026.
**Current head:** 9 August 2026. **33 commits**, 360 files changed.

This document is the auditable detail behind **Section 22** of
`notebooks/01_data_preparation.ipynb`. The headline is not that more models were added.
It is that a **pre-registered, Bonferroni-corrected hypothesis registry** was applied to
34 new research claims, and **27 of them were rejected and recorded as rejected**.

---

## 1. Summary table

| Metric | 23 Jul 2026 | 9 Aug 2026 | Δ |
|---|---:|---:|---|
| Hypothesis families (registry files) | 4 | 12 | +8 |
| Registered hypotheses (all-time) | 15 | 50 | +35 |
| …of which recorded as DROP | — | 34 | — |
| Python modules in `src/` | 14 | 53 | +39 |
| Test files | 4 | 13 | +9 |
| Test functions | 73 | 424 | +351 |
| `IMPROVEMENT_LOG.md` (lines) | 799 | 2,025 | +1,226 |
| `ARCHITECTURE_DOCS.md` (lines) | 1,150 | 1,892 | +742 |

The DROP count is the point. A research pipeline that only records its successes is not
measuring anything; this one records every claim it ever spent.

---

## 2. Eight new hypothesis families

Each family carries its own registry CSV and its own Bonferroni α-ladder, so a new claim
in one family does not silently loosen the bar in another. Verdicts as recorded.

| Family | Registry file | New | KEEP | DROP |
|---|---|---:|---:|---:|
| H1 next-bar direction | `h1_direction_hypothesis_log.csv` | 8 | 4 | 4 |
| RSI/price divergence | `divergence_hypothesis_log.csv` | 6 | 0 | 6 |
| Harmonic patterns (H1 + M15) | `harmonic_pattern_hypothesis_log.csv` | 6 | 0 | 6 |
| G10 macro panel | `macro_panel_hypothesis_log.csv` | 2 | 2 | 0 |
| Tier-A macro / carry | `macro_tier_a_hypothesis_log.csv` | 2 | 0 | 2 |
| Pooled multi-instrument H1 | `pooled_h1_hypothesis_log.csv` | 2 | 0 | 2 |
| H1 multi-day reversion | `h1_multiday_hypothesis_log.csv` | 2 | 0 | 2 |
| Fractal-breakout drift | `fractal_breakout_driftcheck_hypothesis_log.csv` | 1 | 0 | 1 |

Plus 3 new ADD-tests in the existing direction family (`fibonacci_retracement_block`,
`vix_regime_block`, `volatility_forecast_block` — all DROP, tightening the family bar to
α = 0.05/9 ≈ 0.00556) and 2 in the volatility family.

---

## 3. The most important single result: the volatility re-verification

The 23 July submission reported the 5-seed multi-task LSTM volatility ensemble as the
project's one CI-confirmed edge over GARCH(1,1).

On **2026-08-07** that claim was re-verified out-of-sample on 1,712 rows with frozen
models. The edge **replicated** — dMAE +0.01333, CI95 [+0.01039, +0.01612], ahead in every
year 2021–2026. But the decomposition showed **the mechanism is calendar, not volatility
skill**:

- the entire aggregate edge sits in **Friday** rows (dMAE +0.09112), where the Fri→Mon
  target averages 0.0877 against 0.31–0.38 on other weekdays;
- on Mon/Tue/Wed, plain GARCH is slightly **ahead**;
- a GARCH(1,1) multiplied by a **six-number day-of-week lookup table**, fitted on the
  training block only, **beats the neural ensemble outright** (MAE 0.20521 vs 0.21925).

The entry stands as originally measured, with the mechanism recorded alongside it. This
is a negative result about our own headline claim, found by our own verification, and
written into the log rather than quietly dropped. Data: `results/volatility_verification/`.

---

## 4. External-model comparison (Kronos)

A third-party pre-trained financial time-series foundation model was integrated behind a
vendored, commit-pinned loader (`src/external/kronos/`, `requirements-kronos.txt`) and
used for the project's **first cross-model comparison**.

Result (`volatility_hypothesis_log.csv` #9, 2026-08-06): on the pre-declared primary,
Kronos's `pred_abs_move_pct` carries **incremental** information beyond our ensemble
(residual correlation +0.1667, CI99.44% [+0.0354, +0.3090], robust to block length, seed
and rank transform). But the secondary forecast-improvement test **includes zero**, and
Kronos does not beat GARCH(1,1) on those rows. Recorded as: signal is real, does not yet
convert into a distinguishable forecast gain.

Contamination was handled explicitly — the comparison window was forced to
2024-07 → 2026-06 by Kronos's own training cutoff, and the note records that this is the
spent test block rather than the usual validation arbiter.

---

## 5. Methodology hardening

- **Walk-forward validation** (`src/walk_forward_validation.py`) — robustness check
  against the single fixed chronological split.
- **Triple-barrier labelling** (`src/triple_barrier.py`) — path-dependent event labels
  replacing naive fixed-horizon targets for the event-conditional families.
- **Replication requirement** — `H_dir.1` triggered mandatory replication on GBPUSD,
  AUDUSD and CHFUSD (`H_dir.3`–`H_dir.5`) before it could be believed.
- **Label-geometry feasibility scan** (`src/h1_horizon_feasibility.py`) — a design
  calculation run *before* spending a hypothesis, to check the label geometry can support
  a detectable effect at all.
- **Protected-artifact checksums** (`tests/fixtures/h1_production_protected_sha256.json`)
  — production artifacts are hash-pinned so a retrain cannot silently alter them.
- **Owner-override documentation** — where a model was put into production *against* its
  DROP verdict (H1 TI-LSTM, 2026-07-18), that override is recorded as an override, with
  the contrary evidence next to it.

---

## 6. Work in progress (no hypothesis spent)

**Discrete-Hodge curl on the FX currency graph** — `src/curl_stress.py`,
`src/curl_mt5.py`, `src/curl_null_simulation.py`, `CURL_EXPERIMENT_PLAN.md`.

Treats the six EUR/USD/JPY/GBP pairs as a 1-cochain on a graph, where no-arbitrage makes
the cochain exact and every triangle curl identically zero — so the observed curl is a
pure measurement-error channel whose *amplitude* estimates volatility × illiquidity.

Currently validated **only on synthetic data**, where the truth is known:

- exact simultaneity gives curl < 1e-12 (orientation algebra correct);
- the closed-form asynchronicity null matches realised curl variance to 0.87–0.98;
- curl variance is flat across M1/M5/M15/H1 as theory predicts, while return variance
  grows linearly;
- an injected 2 bp dislocation is separated with AUC 0.996.

A pre-registered STOP GATE is written down *before* the real-data run, with the expected
outcome stated in advance as "nothing there". **No hypothesis has been registered and no
production code is touched.** Included in the submission as a demonstration of instrument
validation, not as a result.

---

## 7. Commit list

```
2026-08-09  Update confound_design with optional detrending/binning parameters
2026-08-09  Enhance curl_null_simulation with activity trend parameter
2026-08-09  Add confound gate prompt and documentation for curl_mt5.py
2026-08-08  Add discrete-Hodge curl measurement module for FX currency graph
2026-08-08  Add initial draft of research frontier ideas
2026-08-08  Enhance retrain logging and status handling
2026-08-07  Add volatility verification results and yearly decomposition data
2026-08-05  Update model checksums and enhance test coverage for Kronos volatility
2026-08-05  Refactor tests and update protected file checks
2026-08-04  Add new hourly data files for various currency pairs and summary CSV
2026-08-03  Add JSON file for external Kronos control and frequency metrics
2026-08-02  Add tests and results for Kronos integration
2026-07-31  Add Kronos external model integration and testing framework
2026-07-31  Add unit tests for TIER-A MACRO FX DIRECTION program
2026-07-29  Add macro panel model for G10 currency direction
2026-07-29  Add H1 label-geometry feasibility scan module and tests
2026-07-29  Add post-hoc diagnostics for H_dir.1 model performance
2026-07-28  Add walk-forward validation module
2026-07-28  Add comprehensive tests for pooled multi-instrument H1 data
2026-07-28  fix: drop_incomplete_bars handles tz-aware indices, drops weekend bars
2026-07-26  Add Fibonacci and VIX features for direction/return model
2026-07-26  Add ZigZag swing pivot detector and fractal breakout driftcheck
2026-07-26  Add harmonic pattern detection and triple barrier event labeling
2026-07-26  Add volatility-scaled position-sizing backtest module
2026-07-26  feat: Add volatility forecast feature, integrate into direction/return model
2026-07-26  feat: Implement raw PyTorch MLP for H1.2
2026-07-24  Update prediction history and logs; adjust yield differentials
```
(plus 6 refactor/maintenance commits)
