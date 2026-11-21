# fx-fibonacci research family

This family applies Robert Fischer's Fibonacci structure, with a compact MLP on CUDA, to
EUR/USD and GBP/USD. The H1 bar is the decision grid, and D1 bars are causal context.

The OpenSpec change is `fx-fibonacci-mlp`, and the code lives in `src/fx_fib/`.

> **Gross of spread and slippage — not a trading result.** This is the owner's decision
> of 2026-11-21. Nothing here is broker, order or execution code. Nothing here feeds the
> served API, `models/` or the paper-trading ledgers.

## Status

**Decided 2026-11-21: all four cells DROP** (`fx_fib_hypothesis_log.csv`,
`walk_forward_report.md`).
- On log-loss, the MLP ensemble is about 0.38 worse than the class prior and 0.12–0.14
  worse than LightGBM, on both pairs and in every fold.
- Part of that margin is calibration: class-balanced training spreads the probabilities,
  so the top-class confidence sits at 0.3–0.5 while 77 % of rows are neutral.
- Macro F1, which ignores calibration, also shows no information: MLP 0.29–0.30 against
  the prior's 0.29.
- The gross fixed rule has a profit factor of 0.91–1.04.
- A recalibrated or unweighted variant would be a new hypothesis under a new config.

## Commands

Run every command from the repo root, with `.venv` active.

```powershell
# Refresh both pairs' H1 caches from the running MT5 terminal (append-only).
python -m src.fx_fib.data_loader

# Label report (class balance, ties, uniqueness) -> labels.md
python -m src.fx_fib.labeling

# Walk-forward on the GPU: every fold, every model -> oos_predictions.csv.gz, walk_forward_run.json
python -m src.fx_fib.walk_forward_backtest

# One-shot scoring of the four cells -> fx_fib_hypothesis_log.csv (refuses a second run)
python -m src.fx_fib.score

# Report -> walk_forward_report.md
python -m src.fx_fib.report

# All fx_fib tests.
python -m pytest -q tests/test_fx_fib_config.py tests/test_fx_fib_protected.py tests/test_fx_fib_data.py tests/test_fx_fib_aligner.py tests/test_fx_fib_swings.py tests/test_fx_fib_features.py tests/test_fx_fib_labels.py tests/test_fx_fib_models.py tests/test_fx_fib_walkforward.py tests/test_fx_fib_score.py
```

## Data — `data/`

| File | Content |
|---|---|
| `<PAIR>_h1.csv` | MT5 H1 bars from 2015-01-01: `time,open,high,low,close`. There is no tick volume and no spread. |
| `refresh_log.csv` | One row per pair per refresh: status, rows appended, range, weekly-open match rate. |

- **Bar labels are on the broker server's clock, `Europe/Berlin`. They are not UTC.**
  They are stored UTC-localised as-is, like every cache in this repo.
- A refresh is refused for a pair whose weekly-open labels stop matching the Berlin open
  hour, checked week by week, including the March and October DST-mismatch weeks.
- Bars are read from MT5 position 1, so the bar still forming is never cached. The
  pooled cache `results/pooled_h1/<PAIR>_h1.csv` has a partial last row from 2026-07-28
  15:00 for exactly this reason. Its other 69 999 rows match this cache exactly.
- If symbols cannot be resolved for every pair, nothing is fetched. If MT5 is
  unreachable, the caches are left as they are. No other data vendor is ever used.

## H1 → D1 alignment (`src/fx_fib/mtf_aligner.py`)

- **D1 bars are built from H1 by trading date.** A date whose first bar is labelled
  22:00 or later holds only a market reopen, so its bars join the next date. This
  covers every Sunday evening and the reopen evenings of 25 December and 1 January.
  The result is Monday–Friday D1 bars with no one-hour stub days.
- **A D1 bar closes at the midnight that ends its date.** That depends only on the date,
  so a history cut off mid-day can never make a half-built day look closed.
- **An H1 bar labelled `t` closes at `t + 1 h`.** It uses only D1 bars closed at or
  before that time:

  | H1 bar | Newest D1 bar it may use |
  |---|---|
  | Wednesday 14:00 | Tuesday |
  | Wednesday 23:00 | Wednesday |
  | Friday 22:00 | Thursday |
  | Sunday 23:00 | Friday |

- ATR is Wilder ATR(14), causal, computed on each timeframe separately.

## Swing detector (`src/fx_fib/swing_detector.py`)

The threshold is `min_swing = k_atr × ATR(14)`, with `k_atr_d1 = 1.5` and
`k_atr_h1 = 2.0` (fixed, never searched).

A swing high is the highest bar since the last low pivot. A later bar confirms it when
all three of these hold:
1. the leg is at least `min_swing`;
2. any close since then is below the high bar's low, OR the counter-move is at least
   `min_swing`, OR the counter-move retraces at least 38.2 % of the leg;
3. the two pivots span at least 3 bars counting both pivot bars, so their extreme bars are
   at least 2 apart (Fischer's 3-bar rule, the owner's reading B of 2026-11-21).

Lows mirror this.

- **A pivot is visible only from its confirmation bar.** The bar that makes the extreme
  never confirms itself.
- **A new extreme beyond the last pivot moves that pivot.** This is recorded as a
  replacement event at that bar, so earlier bars keep the old pivot.
- **Fischer's 15-bar rule.** After 15 quiet D1 bars, the TGD features use the 15-bar
  extremes. No other feature uses this rule.

Measured on the real caches:

| Pair | H1 | D1 |
|---|---|---|
| EURUSD | 6 402 pivots, one about every 9 bars | 411 pivots, one about every 6 days |
| GBPUSD | 6 296 pivots, one about every 9 bars | 381 pivots, one about every 6 days |

The median confirmation lag is 1 bar. The stricter reading of the 3-bar rule (extreme bars
at least 3 apart) kept only 312 and 298 D1 pivots. It dropped fast V-turns such as EURUSD's
157-pip rally from 14 to 16 October 2026, which reading B confirms on 19 October.

## Features (`src/fx_fib/features.py`)

There are 42 columns, listed and ordered in `src/fx_fib/fx_fib_config.json`. Every one
is causal and scale-free.

| Block | Columns |
|---|---|
| D1 leg, joined to H1 | amplitude, duration in bars, retracement, distances to the 0.382, 0.5 and 0.618 levels, the 1.618 extension, target 2, band width |
| H1 leg | the same eight columns |
| H1 breakouts | 1, 2 and 4 bars; the close against the leg's high-bar low and low-bar high |
| D1 alignment flag | the 1-bar breakout direction inside the D1 0.618 or 1.618 zone |
| Risk | distances to the 5-bar price square; reward:risk to the H1 leg's 50 % level |
| D1 Time Goal Days | `C = B + 1.618·(B − A)`; days to the nearest TGD, cluster count, price-time confluence |
| Spirals | `spiral306` (k = 0.306349, rings 0–2) and `spiral077` (k = 0.076587, rings 2–4); per direction: distance, ring, crossing |

Distances divide by ATR of their own timeframe. Warm-up rows, those missing pivots or ATR,
are excluded, leaving about 99 % of rows usable.

On the real caches, k = 0.306349 on rings 2–4 was capped on about 98 % of rows. Both sets
now have a median distance of 3–4 ATR and about 3 000 crossings each.

## How this family is decided

The family is decided by its own purged walk-forward. There are four cells: for each
pair, MLP vs class prior and MLP vs LightGBM, on log-loss. The bar is
`alpha = 0.05/4`, by owner decision of 2026-11-21.

Each cell is scored once. A changed config is a new hypothesis. Earlier project results
are recorded in the registry as facts, never as blockers.
