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

## Staying alive (logger health)

This section comes from openspec change `gold-forecasting-v2`, spec
`gold-forecasting/logger-health`.

**Task settings.** `install_task.ps1` sets the task to:
- **WakeToRun**: wake the PC to run;
- **StartWhenAvailable**: run once as soon as possible after a missed start;
- run on battery, and keep running if the PC goes onto battery;
- never run two passes at once;
- give up after 20 minutes.

**Wake timers.** A task can wake the PC only if the active power plan allows wake
timers. The installer prints the setting and never changes it. If it says "disabled" or
"important wake timers only", enable *Allow wake timers* under Power Options > Sleep.

**A catch-up pass never back-fills.** It forecasts the newest closed bar, and every hour
it missed becomes a gap.

**Health log.** Every pass, including a locked, crashed or MT5-less one, appends one row
to `results/gold/forward/health.csv`:

`started_at, ended_at, outcome, mt5, newest_h1, forecasts, settled, gaps, note`

`outcome` is one of `ok`, `no live data`, `clock mismatch`, `locked` or `error`.

**Stall rule** (`src/gold/health.py`, `GET /api/gold/health`). It applies only while the
metals market is open (New York, Sunday 18:00 to Friday 17:00, with a daily break at
17:00–18:00). `/gold` shows a red warning when either:
- the last successful pass is more than 2 hours old; or
- more than 3 market hours of the last 24 have no H1 forecast.

Exchange holidays are not modelled. A holiday can raise a false warning, but can never
hide a real stall.

## Round-two additions (change `gold-forecasting-v2`)

All round-two output goes to its own files. Round one's `predictions.csv`,
`settlements.csv` and `gaps.csv` keep their schema byte for byte.

**Ranges.**
- **Files.** Each pass writes `ranges.csv`, holding an 80 % and a 95 % range for each
  h1_lstm or d1_lstm forecast made **in that pass**. Older forecasts never get one
  back-filled. Each range's `range_settlements.csv` row records whether the exit close
  fell `inside`.
- **Calibration.** It is adaptive conformal inference, replayed on every pass from a
  seed. The seed is the newest W eligible rows of `results/gold/test_predictions_*.csv`
  (W = 500 for H1, 250 for D1). Those come from the test-stage fit of the same cell, not
  from the final refit, because no saved predictions of the final refit exist.
- **Correction.** The calibration then moves with every settled live bar, which
  corrects that seed within W bars.
- **Validation result.** Calibrated exactly, but not significantly sharper than GARCH:
  see `RESULTS.md`, 2026-10-23.

**Large-move alerts.**
- **Threshold.** It is the 90th percentile of the frozen h1_lstm's own forecasts
  (±0.151 %, calibrated 2026-10-23, `src/gold/gold_config_v2.json` → `alerts`).
- **On each alert,** a new h1_lstm forecast at or above the threshold:
  - writes one `alerts.csv` row, deduplicated by bar;
  - raises one Windows notification, Bulgarian then English, which names the size only
    and says the direction is unknown;
  - shows a banner on `/gold` while that hour is open.
- **Settlement.** `alert_settlements.csv` records the realised move.
- **Testing and recalibrating.** `python -m src.gold.alerts --test-notification` raises
  a sample toast. Recalibrate only after a promotion, with `--calibrate`.

**Monthly refit and shadow version.**
- **The refit.** `EURUSDProphet-GoldRetrain` runs on the 1st at 02:30
  (`install_retrain_task.ps1` / `uninstall_retrain_task.ps1`). It fits every forward
  cell on all bars available into `research_models/gold/versions/<YYYYMMDD>/`, with its
  own `manifest.csv`, and makes that version the single active shadow
  (`versions/active.json`).
- **Shadow forecasting.** Each pass also forecasts with the shadow into
  `predictions_v2.csv` with `shadow=1`, and settles it into `settlements_v2.csv`. Shadow
  rows never enter `predictions.csv`, so no registered forward verdict counts them. The
  leaderboard shows them.
- **Registered models stay frozen.** A refit never touches `research_models/gold/<cell>/`
  or `results/gold/artifact_manifest.csv`. Concurrent refits are prevented by
  `versions/retrain.lock`, and a skipped refit is recorded in `health.csv`.
- **Promotion is manual:**

  `python -m src.gold.versions promote <YYYYMMDD> --reason "..."`

  It refuses while the registered forward window is open (until the cap, or until every
  forward verdict is final). Otherwise it:
  - copies the version into the registered layout and rewrites the manifest;
  - records the old and new digests and the reason in `RESULTS.md`;
  - retires the forward registration, so a new window must be registered before
    scoring.

  `python -m src.gold.versions status` shows the active shadow and the window.

## Reading the history

The *History* tab on `/gold` reads these files without changing them, through
`GET /api/gold/history?source=live`. It pairs each prediction with its own settlement
by `key`, and lists every `gaps.csv` row as a missed bar. The `validation` and `test`
sources show the study's saved predictions, labelled as past research. The test block
is shown for reference and is never used to choose a model.

Its hit rates and errors are **descriptive, not a verdict**. Forward verdicts come only
from `python -m src.gold.forward_logger --report`, after the forward pre-registration
below. A run of missed bars usually means the PC was asleep or MT5 was closed during
those hours: the logger records what it could not forecast instead of back-filling it.

## Phases

Every row is stamped `dry_run` until `results/gold/forward/PRE_REGISTRATION.md` is
committed and unmodified, and `registered` after that. Dry-run rows count for coverage
only and are **never scored** (task 8.3 runs the dry run for at least two weeks).

## What the logger never does

It never trades, sizes or simulates P&L. It never writes under `models/` or
`results/forward_eval/`, and it never refits a model.
