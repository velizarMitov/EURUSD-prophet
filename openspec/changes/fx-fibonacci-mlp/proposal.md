# Proposal

## Why

The owner wants to test, for EUR/USD and GBP/USD, whether Robert Fischer's quantitative
Fibonacci rules (swing filters, 0.382/0.5/0.618/1.618 levels, Time Goal Days, the
corrected logarithmic spiral), with daily context feeding an hourly decision, carry
information that a compact neural network can learn. The test must be built so a
false positive cannot reach live money.

By owner decision (2026-11-21) this is a new, self-contained research family, decided
by its own walk-forward on the full history. Earlier project results do not constrain
it. These are the daily Fibonacci bundle #7, rejected 2026-07-26, and the H1 direction
family's test block, read 2026-07-30. Overlap with them is recorded as a fact in the
registry and never blocks a verdict.

## What Changes

- New research package `src/fx_fib/`, Python in the existing `.venv`, PyTorch on CUDA.
  The machine's RTX 4070 Laptop (8 GB), driver 616.56, is visible to
  `torch 2.11+cu128`, and `lightgbm 4.6` is installed. The repo has no .NET project,
  so there is no C#, TorchSharp or ONNX path.
- An MT5 OHLC H1 cache for `EURUSD` and `GBPUSD`, kept in the family's own tree. The
  pair list lives in config, so adding a pair is a config change. Daily bars are built
  from H1 on the broker-label (Berlin) calendar.
- A causal H1→D1 aligner. Each H1 bar sees only D1 bars that closed at or before its
  own close.
- A Fischer swing detector, with an ATR-adaptive threshold, the confirmation rules, the
  3-bar rule and the 15-bar rule. A pivot is recorded only at the bar that confirms it.
- ATR-normalised feature blocks:
  - Fibonacci retracements and extensions;
  - H1 breakout and alignment features;
  - risk/target geometry;
  - D1 Time Goal Days;
  - the logarithmic spiral with the **positive** constant `k = (2/π)·ln φ`, with
    `k` configurable.
- Triple-barrier labels on H1, in three classes: neutral, upper barrier first, and
  lower barrier first. **Labels and every P&L-style metric are gross of spread and
  slippage, by owner decision (2026-11-21).**
- Models:
  - a 5-seed ensemble of the specified MLP (128-64-32, LayerNorm, Dropout), trained
    on the GPU;
  - a fixed-hyperparameter LightGBM baseline on identical features and folds;
  - a smaller 64-32 MLP and an L2 logistic regression, both descriptive only.
- Chronological walk-forward with a purge and embargo of at least the label horizon.
  Scalers are fit per train fold. `shuffle` is never used.
- A one-family registry in `results/fx_fib/`, with its own alpha `0.05/4`. Its four
  cells (per pair: MLP vs class prior, MLP vs LightGBM) are scored once, from the pooled
  out-of-sample walk-forward rows, with a block bootstrap.

Out of scope:
- any broker, order, position-sizing or execution code;
- any change to `models/`, `config.json`, `FEATURE_COLUMNS`, the served API, the
  paper-trading ledgers or the `0.05/9` bar;
- any UI page.

## Capabilities

### New Capabilities
- `fx-fibonacci/market-data`: the MT5 H1 cache for the configured pairs, the D1 bars
  built from it on the broker calendar, and the causal H1→D1 alignment contract.
- `fx-fibonacci/structure-features`: the Fischer swing detector and every derived
  feature (Fibonacci levels, breakouts, risk geometry, TGD, spiral), each causal and
  ATR-normalised.
- `fx-fibonacci/barrier-labels`: the triple-barrier target on H1. It is gross of cost,
  its parameters are fixed in config, and same-bar ties are resolved deterministically.
- `fx-fibonacci/models`: the CUDA MLP seed ensemble, the LightGBM baseline and the
  descriptive variants, with the training controls the family requires.
- `fx-fibonacci/evaluation`: walk-forward with purge and embargo, metrics, the
  registry, and the one-shot scoring of the four cells from the walk-forward.

### Modified Capabilities
- None. No existing spec's requirements change.

## Impact

- **New code:** `src/fx_fib/` (about 11 modules) and `tests/test_fx_fib_*.py`.
- **Config:** `src/fx_fib/config.json`. Its digest is recorded with the verdicts.
- **Data and results:** `results/fx_fib/` (cache, reports, registry).
- **Artifacts:** frozen models in the gitignored `research_models/fx_fib/`, with digests
  in `results/fx_fib/artifact_manifest.csv`.
- **Dependencies:** `torch` is installed but undeclared, so it must be added to
  `requirements.txt` with the CUDA wheel note. No other new packages.
- **Operations:** none. No scheduled task.
- **Docs:** a new `ARCHITECTURE_DOCS.md` section and a short CLAUDE.md entry naming
  the family, its alpha, its gross-of-cost status and how it is decided.
- **Untouched and asserted by test:** `models/`, `config.json`, `results/pooled_h1/`,
  `results/eurusd_h1.csv`, `src/features.py`, `src/inference.py`,
  `src/paper_trading.py`.
