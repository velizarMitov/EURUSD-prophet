# Tasks

## 1. Scaffolding and dependencies

- [x] 1.1 Declare `torch` in `requirements.txt` with a comment naming the CUDA 12.8 wheel index, and verify `pip install -r requirements.txt --dry-run` resolves in `.venv` without changing the installed torch.
- [x] 1.2 Create `src/fx_fib/` with `__init__.py` and `fx_fib_config.json`. The config holds:
  - pairs `["EURUSD", "GBPUSD"]`, `k_atr_d1 = 1.5`, `k_atr_h1 = 2.0`;
  - barrier `m = 1.0`, `H = 24`;
  - spiral `k = 0.306349`, `H = 5`, start `A`, cap 10;
  - the fold settings, the MLP and LightGBM parameters, the seeds `[11, 22, 33, 44, 55]` and `allow_cpu = false`;
  - the scoring settings: alpha `0.0125`, block 24, 10 000 resamples, seed 20261121.

  Verify by loading the config in a test.
- [x] 1.3 Add `tests/test_fx_fib_protected.py`. It digests `models/`, `config.json`, `results/pooled_h1/`, `results/eurusd_h1.csv`, `src/features.py`, `src/inference.py` and `src/paper_trading.py` before and after a no-op family run. Verify it passes.

## 2. Module 1a: market data

- [x] 2.1 Implement `data_loader.py`:
  - an MT5 refresh into `results/fx_fib/data/<PAIR>_h1.csv`, OHLC only, append-only;
  - symbol resolution imported from `src/pooled_h1_data.py`, stopping for all pairs on an unresolved symbol;
  - no other vendor.

  Verify with a fake-MT5 test covering append, unreachable and ambiguous-symbol cases.
- [x] 2.2 Add the Berlin weekly-open check, using `src/forecast_eval/m15_data.py` helpers. A mismatching series is refused. Verify with tests on a normal week, a March DST-mismatch week and a shifted-clock series.
- [x] 2.3 Run the first real refresh for both pairs, and verify each cache starts by 2015-05, has no duplicate labels, and has a weekly-open match rate printed in the run log.

## 3. Module 1b: D1 bars and the causal aligner

- [x] 3.1 Implement D1 aggregation by Berlin label date in `mtf_aligner.py`, with Sunday bars merged into Monday and the D1 close time set to the end of the last H1 bar. Verify with tests for the Sunday stub and for identical rebuilds.
- [x] 3.2 Implement the close-time `merge_asof` join of D1 values onto H1 rows. Verify with tests:
  - the 14:00 bar sees the previous day;
  - the 23:00 bar sees its own day;
  - altering every bar after `t` leaves rows ≤ `t` unchanged.
- [x] 3.3 Implement the causal Wilder ATR(14) on H1 and D1 by reusing `src/h1_features.py::wilder_atr`. Verify ATR at bar `t` is unchanged by truncating after `t`.

## 4. Module 1c: the Fischer swing detector

- [x] 4.1 Implement the single-pass state machine in `swing_detector.py`, emitting `(kind, extreme_idx, extreme_price, confirm_idx)`. Verify with tests for:
  - close below the high bar's low;
  - a counter-move of at least `min_swing`;
  - a retrace of at least 38.2 %;
  - a leg too small;
  - the extreme bar never confirming itself.
- [x] 4.2 Add the 3-bar spacing and the per-timeframe thresholds. Verify with tests that pivots 2 bars apart are not confirmed, and that changing `k_atr_d1` changes only the D1 pivots.
- [x] 4.3 Add `visible_pivots(t)` and the 15-bar D1 fallback for TGD only. Verify with tests:
  - a pivot at bar 100 confirmed at bar 104 is invisible at bars 100–103;
  - the fallback starts after 15 quiet D1 bars and stops at the next pivot.
- [x] 4.4 Write `results/fx_fib/README.md`, covering the data, alignment and swing rules, the gross-of-cost status and the commands. Verify the commands in it run as written.
- [x] 4.5 Owner review checkpoint. Stop here, show Module 1's test results and a pivot plot for one month of each pair, and wait for the owner's go-ahead before group 5.

## 5. Structure features

- [x] 5.1 Implement `fib_features.py`: `swing_amp_atr`, `swing_bars` (the owner's addition), `retracement_pct`, `dist_to_fib_382`, `_500` and `_618`, `dist_to_ext_1618_w3`, `dist_to_target2_0618` and `price_band_width_atr`, on D1 (joined) and H1. Verify with the spec's worked examples and the warm-up exclusion test.
- [x] 5.2 Implement the breakout, alignment and risk features: `h1_breakout_{1,2,4}`, `h1_close_vs_extreme_bar` (2 columns), `d1_alignment_flag`, `price_square_{low,high}_atr` and `rr_to_50pct_target`. Verify with tests including the 0.1 ATR divisor floor.
- [x] 5.3 Implement `tgd.py`: `d1_bars_to_nearest_tgd`, `d1_tgd_cluster_count` and `price_time_confluence`. Verify with the 100/110 → 126 projection test and the confluence-maximum test.
- [x] 5.4 Implement `spiral_engine.py` with the two spiral sets (`spiral306`: k 0.306349, rings 0–2; `spiral077`: k 0.076587, rings 2–4; owner's choice after the coverage check), normalised coordinates, both directions, the ring id, the crossing flag and the cap. Verify with tests:
  - a negative `k` is refused;
  - a known geometry gives hand-computed distances;
  - two pairs with the same normalised shape match.
- [x] 5.5 Implement the `features.py` assembler with the ordered column list from the config. Verify with tests:
  - column drift fails the build;
  - truncation invariance holds over all features at 20 random cut points per pair;
  - scaling OHLC by a constant leaves the features unchanged.

## 6. Barrier labels

- [x] 6.1 Implement `labeling.py`: vectorised first-touch labels from `t+1`, ties set to 0 and counted, no partial labels, horizons counted in bars. Verify with tests for upper first, time out, the same-bar tie, the newest bars unlabelled and a weekend inside the horizon.
- [x] 6.2 Write the label report `results/fx_fib/labels.md`. It covers the train-row class balance, the tie counts and the mean uniqueness per pair, under the gross-of-cost header. Verify that no class is below 10 %; otherwise amend the config and log it before group 7.

## 7. Splits and models

- [x] 7.1 Implement `splits.py`: 6-month folds from 2019-07-01 on shared dates, the inner 15 % validation tail, and a purge plus embargo of `H` at each boundary. Verify the ordering tests, the overlap-removed test and the gap-size test.
- [x] 7.2 Implement the per-fold scaler, fit on fit rows only. Verify its statistics equal those of the fit rows alone.
- [x] 7.3 Implement `model_mlp_cuda.py` with:
  - the exact architecture, device-resident tensors and a seeded device generator;
  - class-weighted cross-entropy, AdamW and ReduceLROnPlateau;
  - early stopping that restores the best epoch;
  - the CUDA check, which refuses unless `allow_cpu` is set, and the run metadata.

  Verify with tests for output shape and row sums, and for the best-epoch restore.
- [x] 7.4 Add determinism settings. Verify the rerun test: same fold and seed twice, maximum probability difference ≤ 1e-5.
- [x] 7.5 Implement the all-or-nothing 5-seed ensemble and the `research_models/fx_fib/` writer with `artifact_manifest.csv` digests. Verify a missing or tampered member blocks the ensemble.
- [x] 7.6 Implement `baselines.py`: LightGBM with the fixed config parameters on the identical rows, features and validation slice, plus the descriptive 64-32 MLP and L2 logistic regression. Verify the identical-inputs test.

## 8. The walk-forward

- [x] 8.1 Implement `walk_forward_backtest.py`. It runs every fold for both pairs and reports, per fold and pooled:
  - log-loss, macro F1, Brier, ECE over 10 bins and the reliability table;
  - the class-prior baseline.

  Verify a run completes on the GPU and the run log records the device.
- [x] 8.2 Implement the fixed gross rule: act on class 1 or 2, one position per pair, barrier or time exit. Compute profit factor, Sharpe on daily sums (√252), maximum drawdown, win rate and expectancy in percent. Verify with a hand-built 10-bar test, including a skipped overlapping signal.
## 9. Scoring and registry

- [x] 9.1 Implement `score.py`. It pools each pair's out-of-sample rows, computes the per-row log-loss difference for the four cells, and runs the one-sided circular block bootstrap (block 24, 10 000 resamples, fixed seed). It records KEEP or DROP with the interval, the config, feature-list and artifact digests, and the observed minimum detectable effect. Verify with tests: a synthetic clear improvement is KEEP, an interval reaching zero is DROP, and the same inputs give the same interval.
- [x] 9.2 Write the registry `results/fx_fib/fx_fib_hypothesis_log.csv`. Each note states the decision basis, "gross predictive result, not a trading edge", and for EURUSD the overlap with the H1 direction family's 2024-11..2026-07 block. Verify the tests for second-scoring refusal and verdict wording, and that the other families' logs are byte-identical before and after.
- [x] 9.3 Add the guard: label, training and scoring stages refuse when the config or feature-list digest differs from the one recorded with existing verdicts. Verify with a test that changes `m`.
- [x] 9.4 Generate `results/fx_fib/walk_forward_report.md` with the gross-of-cost header, the fold and pooled metrics, the gross rule table and the four verdicts. Verify it is identical across two builds apart from the timestamp.
- [x] 9.5 Add an ARCHITECTURE_DOCS.md section for the fx-fibonacci family, and a short CLAUDE.md entry naming the family, its alpha `0.05/4`, its gross-of-cost status and that it is decided by its own walk-forward. Verify the files and paths both name exist.

## 10. Integration check

- [x] 10.1 Run the full pipeline end to end on both pairs: refresh, features, labels, walk-forward, scoring and the report. Then run `python -m pytest -q tests/test_fx_fib_*.py` and the protected-set test. Verify all pass and the protected digests are unchanged.

## Workflow follow-up

- Archive the change with `/opsx:archive` once groups 1–10 are complete and reviewed.
