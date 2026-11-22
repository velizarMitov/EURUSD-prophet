# fx-bb-reversion — Bollinger Bands (20, 2 SD) mean reversion on D1

> **Gross of spread and slippage — not a trading result.**

Openspec change `fx-bb-reversion`. Code `src/fx_bb/`, tests `tests/test_fx_bb_*.py`, docs
`ARCHITECTURE_DOCS.md §3.17`.

## Result (scored once, 2026-11-22): both cells DROP

| Cell | Question | Events (dates) | Statistic | One-sided bound | Verdict |
|---|---|---|---|---|---|
| `bb_rule_mean_reversion` | mean counter-trend 3-day outcome > 0 ATR | 1 262 (928), 2012-02-14..2026-11-12 | +0.0139 ATR | 2.5 % quantile −0.0624 | **DROP** |
| `bb_lightgbm_vs_base_rate` | pooled LightGBM log-loss below the base rate | 1 048 (749), 2015-01-02..2026-11-12 | +0.0035 (0.69967 vs 0.69616) | 97.5 % quantile +0.0097 | **DROP** |

alpha = 0.05/2 per cell. Each cell was tested with a circular block bootstrap over
business days: 5-day blocks, 10 000 resamples, seed 20261124. The minimum detectable
effects were stated in `labels.md` before any model:
- Cell 1: about 0.09 ATR on events and 0.11 ATR on dates.
- Cell 2: about 54.3 % on events and 55.1 % on dates.

A DROP means there is no reversion and no filter value of that size.

What the descriptive tables in `walk_forward_report.md` show:
- **The rule.** The win rate is 50.7 % (Wilson on dates, 47.5–53.9 %). The gross profit
  factor is 1.013, and 0.989 after each pair's mean closing spread.
- **Mid against bid.** The bid statistic is +0.0145 ATR, a gap of 0.0007. That is within
  the 0.02 tolerance, so no parity flag.
- **Pairs and years.** These scatter both ways: USDCAD −0.105 ATR and AUDUSD +0.094; the
  years run from −0.22 to +0.30 ATR. No sub-group was tested, and none is a verdict.
- **The filter.** Events with P(win) > 0.5 look better, at 55.9 % on 189 dates. That
  subset was chosen after the model, and the registered log-loss test is DROP. Calibration
  fails too: the 0.60+ bin won 27.8 % of the time. **Do not read it as an edge.**

Everything here is **gross of spread and slippage and not a trading result.** The spread
appears only in the descriptive after-spread columns.

## Files

| File | What it is |
|---|---|
| `labels.md`, `labels.json` | events, exclusions and the power table, before any model; no outcome is read |
| `fx_bb_hypothesis_log.csv` | the two verdict rows |
| `cell1_events.csv.gz` | Cell 1's mid and bid events with outcomes, written when Cell 1 was scored |
| `oos_events.csv.gz`, `run.json` | Cell 2's out-of-sample events with `p_prior` and `p_lightgbm`; the folds and the six cache digests |
| `artifact_manifest.csv` | the digests of the frozen fold models in `research_models/fx_bb/POOLED/` |
| `walk_forward_report.md` | the descriptive report; `python -m src.fx_bb.report` |
| `data/` | the USDJPY, USDCAD and USDCHF H1 caches, `symbols.json`, `refresh_log.csv` |

Stages, in order: `python -m src.fx_bb.data_loader` (refresh), `python -m src.fx_bb.labels`,
`python -m src.fx_bb.walk_forward`, `python -m src.fx_bb.score --cell rule`, then
`--cell model`, then `python -m src.fx_bb.report`. Every stage refuses a config whose digest
differs from the one recorded by an earlier stage or verdict.

## Market data (task 2.3, real refresh 2026-11-22)

`python -m src.fx_bb.data_loader` refreshed the family's own caches from MT5. It wrote
nothing for the fx_session pairs.

| Pair | Source | Point | First bar used | Last bar | H1 bars | Weekly-open match | Weeks |
|---|---|---|---|---|---|---|---|
| EURUSD | `results/fx_session/data` (read-only) | 0.00001 | 2012-01-02 08:00 | 2026-11-20 21:00 | 92 557 | 96.6 % | 790 |
| GBPUSD | `results/fx_session/data` (read-only) | 0.00001 | 2012-01-02 08:00 | 2026-11-20 21:00 | 92 081 | 96.7 % | 787 |
| USDJPY | `results/fx_bb/data` | 0.001 | 2012-01-02 08:00 | 2026-11-20 21:00 | 92 554 | 96.5 % | 790 |
| AUDUSD | `results/fx_session/data` (read-only) | 0.00001 | 2012-01-02 08:00 | 2026-11-20 21:00 | 92 558 | 96.6 % | 790 |
| USDCAD | `results/fx_bb/data` | 0.00001 | 2012-01-02 08:00 | 2026-11-20 21:00 | 92 427 | 96.6 % | 785 |
| USDCHF | `results/fx_bb/data` | 0.00001 | 2015-02-01 23:00 | 2026-11-20 21:00 | 73 462 | 98.1 % | 623 |

Labels are the broker's Berlin wall clock. The refresh log is
`results/fx_bb/data/refresh_log.csv`, and the point sizes are in
`results/fx_bb/data/symbols.json`.

### Lag check

For each calendar year, the check takes the absolute correlation of the pair's hourly log
returns with EURUSD's, at lags −1, 0 and +1. It must peak at lag 0. The check runs on every
refresh and on every load.

| Pair | Years | Peak at lag 0 | Weakest lag-0 year | Largest off-lag value |
|---|---|---|---|---|
| GBPUSD | 2012–2026 | 15 / 15 | 0.45 (2017) | 0.06 |
| USDJPY | 2012–2026 | 15 / 15 | 0.04 (2012) | 0.04 |
| AUDUSD | 2012–2026 | 15 / 15 | 0.34 (2016) | 0.05 |
| USDCAD | 2012–2026 | 15 / 15 | 0.30 (2016) | 0.05 |
| USDCHF | 2015–2026 | 12 / 12 | 0.65 (2024) | 0.03 |

**USDJPY 2012 passes, but only by a little: 0.04 at lag 0 against 0.03 off-lag.** I checked
it month by month.
- Against EURUSD, AUDUSD and USDCAD separately, 10 of 12 months of 2012 peak at lag 0.
- The low yearly value comes from the USDJPY–EURUSD co-movement changing sign within 2012,
  not from a shifted clock.

**USDCHF before its start would fail.** I read this from MT5 without caching it:

| Year | Lag −1 | Lag 0 | Lag +1 | Peak |
|---|---|---|---|---|
| 2012 | 0.058 | 0.029 | 0.915 | +1 |
| 2013 | 0.016 | 0.050 | 0.771 | +1 |
| 2014 | 0.012 | 0.175 | 0.755 | +1 |
| 2015 | 0.004 | 0.371 | 0.018 | 0 |

Its 2012–2014 bars are one hour off, yet they match 96 % of the weekly opens. The broker
also has no USDCHF bars from 2015-01-15 19:00 to 2015-01-18 23:00 (76 h), around the SNB
floor removal. USDCHF therefore starts on 2015-02-01. No bar is ever shifted.

### Corrupt spreads (more than 50 pips)

| Pair | Bars | Labels | Effect |
|---|---|---|---|
| AUDUSD | 19 | 2014-01-07 05:00 … 23:00 Berlin, spread 300 000 points (3.0 in price) | the 2014-01-07 D1 mid close is void |
| USDJPY | 19 | 2014-01-07, spread 3 000 points (3 yen, 300 pips) | the 2014-01-07 D1 mid close is void |

- **Every other spread** is at most 240 points: 90 for EURUSD, 162 for GBPUSD, 170 for
  USDCAD and 240 for USDCHF.
- **The USDJPY defect was not known before this refresh.** The same rule catches it.
- **Earlier families are unaffected.** fx_session read AUDUSD only through its bid prices,
  and it never read USDJPY.

### Declared broker gaps

Nothing in these gaps is filled. Their days simply have no D1 bar, and any event whose
window touches one is excluded as a pair gap.

| Pair | Month | What the cache holds |
|---|---|---|
| GBPUSD | 2014-03 | 72 H1 bars; 2 D1 days reach 12 bars |
| USDCAD | 2013-05 | no bars from 2013-05-03 22:00 to 2013-05-13 14:00 (232 h); 17 D1 days |

### D1 bars

D1 bars close at 17:00 New York. They are built from the Berlin labels, need at least 12
H1 bars, and use the mid close.

| Pair | D1 days | With a valid mid close |
|---|---|---|
| EURUSD | 3 861 | 3 861 |
| GBPUSD | 3 840 | 3 840 |
| USDJPY | 3 861 | 3 860 |
| AUDUSD | 3 861 | 3 860 |
| USDCAD | 3 855 | 3 855 |
| USDCHF | 3 062 | 3 062 |
