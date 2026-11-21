# Design

## Context

See `proposal.md` (Why) and the five specs under `specs/fx-fibonacci/` for the requirements.

### Step 0 findings (inspected 2026-11-21, read-only)

| Item | Finding |
|---|---|
| Project | Python only (VS Code). There is no `.sln`, `.csproj` or `.pyproj`. The interpreter is `.venv`, Python 3.13.5. |
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8 188 MiB, driver 616.56. `nvcc` 12.4 is on PATH but not needed, because the torch wheel bundles its own CUDA runtime. |
| PyTorch | `torch 2.11.0+cu128`. `torch.cuda.is_available()` is `True`. It is installed but **not declared** in `requirements.txt`. |
| Baselines | `lightgbm 4.6.0`, `scikit-learn 1.9.0` and `xgboost 3.3.0` are installed and declared. |
| TensorFlow | 2.21 has no GPU on native Windows. It is irrelevant here. |
| Data | `results/pooled_h1/{EURUSD,GBPUSD}_h1.csv` hold 70 000 H1 OHLC bars each, from 2015-04-27 to 2026-07-28, on Berlin labels. They are pinned by the provenance fixtures and must not be rewritten. `results/eurusd_h1.csv` feeds production. |
| Prior results | Daily Fibonacci bundle #7: DROP. `src/fibonacci_fractals.py` is dormant. H1 direction family: EURUSD KEEP at about 53 % gross, GBPUSD DROP, and it read the EURUSD H1 block from 2024-11 to 2026-07 on 2026-07-30. By owner decision these do not constrain this family; they are recorded as facts only. |
| Reusable helpers | `src/h1_features.py::wilder_atr`, the MT5 symbol resolution and bar fetch in `src/pooled_h1_data.py`, and the weekly-open checks in `src/forecast_eval/m15_data.py`. All are imported read-only. |

**Stack recommendation.** Use Python and PyTorch-CUDA in the existing `.venv`, with
LightGBM on CPU for the baseline. Nothing needs installing; `torch` only has to be
declared. C# options (TorchSharp, or ONNX Runtime GPU) were rejected: there is no .NET
code to integrate with, and they would add a second toolchain for no gain.

## Goals / Non-Goals

**Goals:**
- Module 1, the causal aligner and swing detector, ships first, with no-look-ahead
  tests. Each later module plugs in without editing it.
- Every stage is deterministic from the cached bars plus the config.
- A null result is cheap to accept, and a positive one is hard to fake.

**Non-Goals:**
- Pooling pairs into one model. Each pair has its own model, because each pair is its
  own registered cell.
- Hyperparameter search of any kind, for the MLP, LightGBM, `k_atr` or the barriers.
- Live use of predictions, any UI page, alerts or MT5 chart output.
- Net-of-cost evaluation. That is explicitly excluded by the owner.

## Decisions

### D1. Package layout

`src/fx_fib/` holds the following modules:
- `fx_fib_config.json`, named so it cannot be confused with production's root `config.json`;
- `data_loader.py`, `mtf_aligner.py`, `swing_detector.py`;
- `fib_features.py`, `tgd.py`, `spiral_engine.py`;
- `features.py`, the assembler and column check;
- `labeling.py`, `splits.py`;
- `model_mlp_cuda.py`, `baselines.py`, `walk_forward_backtest.py`;
- `score.py`.

Outputs go to `results/fx_fib/` and frozen artifacts to `research_models/fx_fib/`. This
mirrors the gold family's isolation.

### D2. Own cache, same provenance

The family keeps its own cache, seeded from MT5 for the full history
(`results/fx_fib/data/<PAIR>_h1.csv`, OHLC only). It does not reuse
`results/pooled_h1/`, because those files are frozen evidence for completed programs and
are digest-pinned. The family imports the pooled module's symbol resolution, so both use
identical broker symbols. There is no yfinance fallback, for the same provenance reason as
the pooled program.

### D3. D1 from H1, joined by close time

D1 is resampled from H1 by Berlin label date, with Sunday stubs merged into Monday. One
source avoids a mismatch with MT5's own D1 bars around Sunday and DST weeks. The join uses
`merge_asof` on close times, backward with exact matches allowed: the H1 close is
`label + 1 h`, and the D1 close is the end of its last H1 bar. One mechanism means one
place to test.

Alternative rejected: shifting D1 features by one day. That is wrong for the 23:00 bar,
which may legitimately see the day that just closed, and it is fragile around holidays.

### D4. The swing detector is a single-pass state machine

The detector runs once over the bars, in O(n). It emits pivot events
`(kind, extreme_idx, extreme_price, confirm_idx)`. On each bar it applies these steps in
order:
1. Update the candidate extreme with the bar's high (or low). A new extreme resets the
   candidate to this bar.
2. Test confirmation, using this bar's close and range against a candidate from an
   **earlier** bar only. The bar that sets the extreme can never confirm it.
3. Apply the 3-bar spacing, counting both pivot bars, so extreme indices must differ by
   at least 2. The owner chose this on 2026-11-21 after the checkpoint. The stricter
   reading, a difference of 3, dropped about a quarter of the D1 swings, including
   confirmed 150-pip V-turns. A qualifying candidate that is too close to the previous
   pivot stays pending and keeps updating.

`min_swing` uses ATR at the confirming bar, which is known then. The defaults
`k_atr_d1 = 1.5` and `k_atr_h1 = 2.0` match the project's existing zigzag convention.
They are fixed and never searched.

Features read the visible pivots at bar `t`, meaning those with `confirm_idx ≤ t`. Swing
amplitude and leg direction come from the same list.

### D5. How the owner's P0…P3 notation maps to levels

The spec fixes this mapping:
- **Extension.** `P0 → P1` is the newest confirmed leg, and the level is
  `P1 + s·1.618·|P1 − P0|`. This is the owner's formula taken literally.
- **Target 2.** It uses the last four pivots, oldest to newest. `s'` is the direction of
  `P0 → P3`.

Fischer also describes a wave-3 projection from the end of wave 2,
`P2 + 1.618·(P1 − P0)`. It was not chosen, because the owner wrote the formula from `P1`.
The mapping must be settled before the first fold trains. After that it is frozen.

### D6. Spiral evaluation

1. Normalise the coordinates around `B`, with `H = 5` D1 bars and `V = ATR_D1(t_B)`.
   Time is a trading-day index, and an H1 bar counts as `day + bars_elapsed/24`.
2. Take `R0` and `θ0` from the start point. `A` is the default, and `C` is selectable in
   the config.
3. Sample each ring at 720 points per turn, for each `DIR`. Two sets are built, by the
   owner's choice of 2026-11-21:
   - `spiral306`: `k = 0.306349`, rings 0–2;
   - `spiral077`: `k = 0.076587`, rings 2–4.

   The spec's original `k = 0.306349` on rings 2–4 put those rings 25–175 ATR away. The
   distance was capped on about 98 % of rows, with no crossings. The choice was made on
   feature coverage alone, never on labels or model results.
4. Find where the branch crosses the vertical line at `x_now`, and refine each crossing
   by linear interpolation.
5. Report the nearest `|y_now − y_branch|`.

D1 anchors change rarely, so the samples are cached per anchor. This needs no numerical
root-finder library.

### D7. Labels are vectorised

Labels use sliding windows of length `H` over the high and low arrays (`n × 24`), with
first-touch indices found by `argmax`. A same-bar tie is when both first-touch indices are
equal. The defaults `m = 1.0` and `H = 24` were chosen a priori. Over one trading day, a
±1 daily-ATR barrier is touched often enough that no class should be rare. The train-row
class balance is reported before any model trains. If a class falls under 10 %, the config
is amended before any model trains and the amendment is logged.

### D8. Folds

| Setting | Value |
|---|---|
| Evaluation folds | 6 months each |
| First fold | Starts 2019-07-01, after about 4 years of training |
| Last fold | Ends at the cache end |
| Inner validation | The last 15 % of each training span |
| Purge and embargo | `H = 24` bars at each boundary |

Every pair uses the same calendar boundaries.

### D9. Training on the GPU

Each fold's tensors go to the GPU once. They are about 13 MB per pair, so batches are
index slices drawn by a seeded `torch.Generator` on the device; `DataLoader` and
`pin_memory` are not needed.

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr `1e-3`, weight decay `1e-4` |
| Batch | 1024 |
| Epochs | At most 200 |
| Early stopping | Patience 15 on validation log-loss |
| Learning rate | ReduceLROnPlateau, factor 0.5, patience 5 |

Determinism uses `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The model uses no cuDNN ops.

This network is tiny, so GPU speed-up per model is modest. The benefit is volume: about
5 seeds × 14 folds × 2 pairs, plus the variants.

### D10. Fixed LightGBM parameters

`multiclass`, `num_leaves=31`, `learning_rate=0.03`, up to 2000 rounds with 100-round
early stopping on the same validation slice, `min_child_samples=200`,
`feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0`,
balanced class weights, `deterministic=True`, `seed=42`, on CPU.

### D11. Gross of cost

The owner wrote "don't account for spread and slippage for edge". The same request also
mentions net-of-cost targets and metrics. The explicit instruction wins, and the
contradiction is recorded here.

For scale: the barrier is about 1 ATR_D1, roughly 60–90 pips on EUR/USD, and the typical
spread is about 1 pip. Cost would therefore be a small share of each barrier move, but it
is still unmeasured. So no number from this family may be read as a trading result.

### D12. Decision from the walk-forward

The owner decided on 2026-11-21 that this family is decided by its own walk-forward. There
is no forward phase and no pre-registration file.

The out-of-sample rows of all folds, from 2019-07-01 to the cache end, are pooled per pair.
Each cell's statistic is the mean per-row log-loss difference, model minus reference. A
one-sided circular block bootstrap uses block 24, 10 000 resamples and seed 20261121. All
of these are fixed in the config before the first fold trains.

`score.py` runs once, after the walk-forward completes. It writes the four verdicts with
the config, feature-list and artifact digests, and it refuses to overwrite an existing
verdict. Re-running the walk-forward with a different config is a new hypothesis, which
grows the family and tightens alpha.

Each verdict's notes state as facts:
- the decision basis;
- that the result is gross of cost;
- for EURUSD, the overlap of its 2024-11..2026-07 rows with the H1 direction family's
  read block.

The report also gives the minimum detectable effect implied by the observed variance, so
a DROP can be read as "no effect" or as "too noisy to tell".

## Risks / Trade-offs

- **[The history was seen before.]** EURUSD rows from 2024-11 to 2026-07 were read by
  another family, and Fibonacci ideas were tested on daily data. → By owner decision this
  does not block a verdict. It is stated in every registry row, so a reader can weigh it.
- **[Re-running after seeing results would be data-snooping.]** → Each cell is scored
  once. A changed config is a new hypothesis that grows the family and tightens alpha.
- **[Overlapping labels shrink the effective sample.]** The ~60k rows per pair amount to
  only a few thousand independent samples. → A small network with dropout, weight decay
  and early stopping, the LightGBM cell, and a block bootstrap with block `H`.
- **[Fischer's prose leaves rule details open: notation, `k_atr`, spiral start.]** → All
  are fixed in config and in D4–D6 before the first fold trains, recorded by digest with
  the verdicts, and never searched.
- **[GPU non-determinism.]** → Deterministic flags and the 1e-5 tolerance test. If that
  fails, the config's CPU switch is used, and the device is recorded in the run log.
- **[Gross results invite over-reading.]** → Every report header and registry note says
  "gross — not a trading result". The family has no path to the API, the ledgers or the
  execution code.
- **[A large feature set (~45) gives room to overfit.]** → The list is fixed by config
  and never pruned by performance. The history report is descriptive.

## Migration Plan

The work ships in phases, and each phase is usable on its own:
1. Module 1 (data, aligner, swings).
2. Features.
3. Labels.
4. Models.
5. Walk-forward and scoring.

**Rollback.** Delete `src/fx_fib/`, `results/fx_fib/` and `research_models/fx_fib/`.
Nothing in production depends on them.
