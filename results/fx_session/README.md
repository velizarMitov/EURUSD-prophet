# fx-session research family

The family makes one forecast per day for EUR/USD and GBP/USD, after the US data release
and the New York open.
- **Features** are known at the close of the H1 bar labelled 14:00 Berlin (16:00 Sofia).
- **The target** is whether the mid price is higher at the close of the bar labelled
  21:00 Berlin (23:00 Sofia).

The OpenSpec change is `fx-session-ny`, and the code lives in `src/fx_session/`.

> **Gross of spread and slippage — not a trading result.** This is the owner's decision
> of 2026-11-21. Nothing here is broker, order or execution code. Nothing here feeds the
> served API, `models/` or the paper-trading ledgers.

## Status

**Decided 2026-11-21.** Both LightGBM cells are DROP and both MLP cells are NOT TRIGGERED
(`fx_session_hypothesis_log.csv`, `walk_forward_report.md`).

| Pair | LightGBM log-loss | Prior log-loss | LightGBM accuracy | Wilson 95 % |
|---|---|---|---|---|
| EURUSD | 0.69597 | 0.69509 | 51.27 % | 49.6–52.9 % |
| GBPUSD | 0.69488 | 0.69350 | 50.65 % | 49.0–52.3 % |

The test could detect accuracy of about 52.6 %, so an edge of that size would have shown.

## Data — `data/`

| File | Content |
|---|---|
| `<PAIR>_h1.csv` | MT5 H1 bars: `time,open,high,low,close,spread`. Prices are bid; the spread is in points. |
| `symbols.json` | Each pair's broker symbol and point size. |
| `refresh_log.csv` | One row per pair per refresh: status, rows appended, range, weekly-open match rate, point size. |

**The real refresh of 2026-11-21:**

| Pair | First bar | Rows | Weekly-open match |
|---|---|---|---|
| EURUSD | 2012-01-02 08:00 | 92 557 | 96.6 % of 790 weeks |
| GBPUSD | 2012-01-02 08:00 | 92 081 | 96.7 % of 787 weeks |
| AUDUSD | 2012-01-02 08:00 | 92 558 | 96.6 % of 790 weeks |

AUDUSD is only an input to the dollar factor. Its spread is never used, and it contains
one bar with a broken 300 000-point spread.

### Why history starts in 2012, not 2010

MT5 reached back to October 2010 for all three pairs. Two problems were found there.

- **The broker's GBPUSD bars from December 2010 to November 2011 are one hour off the
  clock of EURUSD and AUDUSD.**
  - In that span, the two pairs' hourly returns correlate at about 0.6 only after
    GBPUSD is shifted back one hour. At lag 0 the correlation is near 0.
  - From December 2011 they correlate at lag 0 (0.6–0.7).
  - The project's weekly-open check still passed GBPUSD at 91 %, against a 90 %
    threshold. A whole wrong year fits inside that threshold.
  - The owner decided to start every pair on 2012-01-01. The shifted bars are dropped,
    never corrected.
- **GBPUSD March 2014 holds one bar per day**, the same on every fetch.
  - It is declared in the config as a known broker gap: it is cached as returned, left
    out of the clock check, and never filled from another source.
  - Any other hole still refuses the refresh.
  - D1 bars built from fewer than 12 H1 bars count as missing.

## Clock, entry and exit

- **Bar labels are the broker's Berlin wall clock.** They are stored UTC-localised
  as-is. Sofia is Berlin + 1 h all year.
- **Entry and exit are fixed label hours, all year.**
  - Entry is the close of the 14:00 bar; exit is the close of the 21:00 bar.
  - In the US/EU DST-mismatch weeks, the 08:30 ET data release falls in the 13:00 bar,
    and the exit lands at the 17:00 New York rollover. The `dst_mismatch` feature flags
    those days.
- **A day is eligible only if both of these hold.**
  - EURUSD and GBPUSD both have every bar from 14:00 to 21:00.
  - Every pair's weekly open for that week is on the Berlin clock: 23:00, or 22:00 in a
    mismatch week. Before 2017, the broker's mismatch weeks opened at 23:00, so those
    weeks are excluded.

On the real caches there are 3 875 weekdays from 2012. Of those, 3 717 are eligible:

| Excluded as | Days |
|---|---|
| Unverified clock week | 116 |
| Incomplete session | 42 |

From the first evaluation fold (2013-01-01) on, 3 479 days are eligible.

## Commands

Run every command from the repo root, with `.venv` active.

```powershell
python -m src.fx_session.data_loader                 # refresh EURUSD, GBPUSD, AUDUSD H1 + spread (all-or-nothing)
python -m src.fx_session.labeling                    # labels.md: counts, up-rates, parity, power (no model)
python -m src.fx_session.walk_forward --stage lgbm   # prior + LightGBM, every fold -> oos_lgbm.csv.gz, run_lgbm.json
python -m src.fx_session.score --stage lgbm          # score the two LightGBM cells ONCE
python -m src.fx_session.walk_forward --stage mlp    # MLP only for pairs whose LightGBM cell is KEEP (CUDA)
python -m src.fx_session.score --stage mlp           # score the MLP cells ONCE, or record NOT TRIGGERED
python -m src.fx_session.report                      # walk_forward_report.md
python -m pytest -q tests/test_fx_session_config.py tests/test_fx_session_protected.py tests/test_fx_session_data.py tests/test_fx_session_features.py tests/test_fx_session_labels.py tests/test_fx_session_models.py tests/test_fx_session_walkforward.py tests/test_fx_session_report.py
```

## Features (`src/fx_session/features.py`)

There are 20 columns, listed and ordered in `src/fx_session/fx_session_config.json`.
Every value is known at the entry close. A test cuts the history right after the entry
bar and checks that the row does not change.

| Block | Columns |
|---|---|
| Pre-session moves | Asia (bars 00–08), London (09–13) and the data hour (14), each divided by ATR(14) of the last valid D1 bar |
| Ranges | London and data-hour range against their 20-day means; the entry close's position in the London range |
| Yesterday | move, range and close location; the move from yesterday's close to the entry |
| Breaks | whether today, up to 14:00, traded above yesterday's high or below yesterday's low |
| Volatility | ATR(14) of D1 as a fraction of the entry close |
| Dollar factor | the mean of EURUSD, GBPUSD and AUDUSD normalised moves from 00:00 to entry, the same for the data hour, and the pair's residual |
| Calendar | day of week; NFP; scheduled FOMC day from `results/fomc_dates.csv`; DST mismatch |

**NFP is the first-Friday rule, a declared approximation.** The BLS sometimes publishes
on the second Friday. The rule is never corrected by hand, so that no edit can be guided
by the outcomes.

## Target (`src/fx_session/labeling.py`)

- **The target is 1 when the mid close of the 21:00 bar is above that of the 14:00 bar,
  and 0 when it is below.** Mid is bid plus half the bar's spread.
- **Ties are dropped and counted.** There are no barriers and no class weights.
- **Each target covers only bars 15:00–21:00 of its own day.**
- **The same sign on bid closes is the parity check.** A gap above 1.0 pp would flag a
  spread artifact. On the real data it is 0.00–0.03 pp.

## How this family is decided

**Four cells at `alpha = 0.05/4`** are recorded in `fx_session_hypothesis_log.csv`. Each
is scored once.

| Cell | Runs when |
|---|---|
| `EURUSD_lightgbm_vs_prior`, `GBPUSD_lightgbm_vs_prior` | always |
| `EURUSD_mlp_vs_lightgbm`, `GBPUSD_mlp_vs_lightgbm` | only after that pair's LightGBM cell is KEEP; otherwise NOT TRIGGERED, with the alpha left unspent and never reassigned |

- **The test.** It is a one-sided circular block bootstrap, with blocks of 5 days, on the
  per-day log-loss difference. KEEP requires the upper bound to be below zero.
- **A changed config is a new hypothesis.** The label, walk-forward and score stages all
  refuse a config that differs from the one recorded with the verdicts.
- **Earlier results are recorded as facts in every registry note, not blockers.** These
  are the horizon-study session cells at 4 h and 6.5 h (about 49 %), the H1 direction
  family, and the fx-fibonacci DROPs.
