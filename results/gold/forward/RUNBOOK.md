# Gold forward logger — runbook

Operating notes for the forward arbiter of the gold research family (openspec change
`gold-forecasting`, capability `gold-forecasting/forward-arbiter`). Rows logged here are
the **only** data a forward gold claim can be decided on. Everything is **gross of
cost**: spread is used nowhere, so no row or report holds pips, P&L or a ledger.

## What runs

| Scheduled task | When | Does |
|---|---|---|
| `EURUSDProphet-GoldLogger` | hourly at :03 | refreshes `results/gold/gold_{d1,h1}.csv` from MT5, forecasts the newest closed H1 bar and day, writes gap records for missed bars, settles matured forecasts |

The task is independent of the EUR/USD tasks. It has its own lock
(`research_models/gold/forward.lock`), its own files and its own log
(`research_models/gold/EURUSDProphet-GoldLogger.log`).

## Install / remove

```powershell
python -m src.gold.study --stage final                              # frozen artifacts + manifest, once
powershell -ExecutionPolicy Bypass -File scripts\gold\install_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\gold\uninstall_task.ps1
python -m src.gold.forward_logger                                   # one pass by hand
```

Install refuses until `results/gold/artifact_manifest.csv` exists.

## One pass, in order

1. **Lock.** A second pass while one is running exits with "another gold logger pass
   holds the lock". A lock older than 30 minutes counts as stale.
2. **Data.** Both caches are refreshed from MT5. They are append-only, so a bar once
   used is never rewritten. **If MT5 is not reachable, nothing is forecast.** The
   cache's newest bar may be hours old, so its target could already have closed. A
   failure row says "no live MT5 data", and the bars missed become gaps on the next
   live pass.
3. **Clock.** The run stops if the weekly opens stop landing on the predicted
   Europe/Berlin hour: below 95 % over the whole history, or below 60 % over the last
   26 weeks. The broker's clock already drifted once (14 weeks in winter 2016/17).
4. **Artifacts.** Every file under `research_models/gold/<cell>/` must match its
   SHA-256 in `results/gold/artifact_manifest.csv`. A changed file fails only that cell.
   Models are frozen for the registered window and never refit quietly.
5. **Forecast.** Per cadence, every cell forecasts the newest closed bar. Its target is
   still open by construction.
6. **Gaps.** Every as-of bar since the last logged one that was not forecast gets a gap
   record. A forecast written now would come after its outcome, so it is never
   back-filled.
7. **Settle.** Every forecast whose next bar now exists is settled with the study's own
   targets. H1 rows that border a market break settle as `eligible = False` and are
   never scored.

## Files (`results/gold/forward/`)

| file | one row per |
|---|---|
| `predictions.csv` | cell × as-of bar: `p_up`, `ret_pct`, `vol_pct`, `implied_price`, `model_digest`, `phase` |
| `settlements.csv` | settled forecast: realised direction / return / \|return\|, `eligible` |
| `gaps.csv` | missed as-of bar and why |
| `failures.csv` | failed cell or pass and the error |

## Phases

Every row is stamped `dry_run` until `results/gold/forward/PRE_REGISTRATION.md` is
committed and unmodified, and `registered` after that. Dry-run rows count for coverage
only and are **never scored** (task 8.3 runs the dry run for at least two weeks).

## What the logger never does

It never trades, sizes or simulates P&L. It never writes under `models/` or
`results/forward_eval/`, and it never refits a model.
