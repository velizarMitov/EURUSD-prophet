# Forward evaluation — runbook

Operating notes for the forward arbiter of the horizon study
(`openspec/changes/horizon-study-forward-evaluation/`). The logger records what
every challenger predicts at every horizon, as each bar closes. Those rows are
the **only** data a horizon claim can be decided on.

## What runs

| Scheduled task | When | Does |
|---|---|---|
| `EURUSDProphet-ForwardLogger` | every hour at :01 | predicts the newest closed H1 bar (and the newest closed day, once per day), writes gap records for bars it missed, settles matured forecasts |
| `EURUSDProphet-Refit` | Saturdays 06:00 | refits every model on the **first weekend of the month** only; otherwise it does nothing |

Both call `scripts/forecast_eval/run_module.cmd`. Their console output is
appended to `research_models/forward_eval/<task name>.log`.

## Install / remove

```powershell
scripts\forecast_eval\install_tasks.ps1      # creates both tasks
scripts\forecast_eval\uninstall_tasks.ps1    # removes them; logged rows stay
schtasks /Query /TN EURUSDProphet-ForwardLogger
```

## Before it can log anything

- **The MT5 terminal must be open and logged in.** The logger attaches
  read-only to whatever account the terminal holds; today that is a real-money
  ActivTrades account. It reads bars, spreads, swap rates and the server name.
  It never records the login, and no code in `src/forecast_eval/` can place,
  change or inspect an order (a test enforces this).
- **Keep AutoTrading off** in the terminal. Nothing here needs it.
- **Disable sleep** on this machine (or move the logger to an always-on host).
  Every hour the machine sleeps is an hour of data that is lost for good:
  missed bars are recorded as gaps and **never** predicted afterwards.
- The study must have produced `results/horizon_study/artifact_manifest.csv`.
  Until then, only the pinned Kronos channels can log.

## The files

All of them are append-only. A written row is never changed.

| File | One row per |
|---|---|
| `predictions.csv` | model × horizon × as-of bar: the forecast, model version, price source, server, the bar's spread, swap rates, phase |
| `settlements.csv` | prediction whose exit bar has closed: the realised move, whether the forecast was right, and whether the row is **scorable** |
| `gaps.csv` | bar close the logger missed (machine off, terminal closed) |
| `failures.csv` | model that could not predict at a bar, and why. `*` rows mean MT5 itself was unreachable |
| `refit_failures.csv` | monthly refit that failed. The previous version stays live |

## Reading it

- **Coverage** = predicted bars ÷ (predicted + gap bars). The power table in
  the pre-registration assumes the coverage measured during the dry run.
  Coverage well below it lengthens every time-to-decision in proportion.
- **`scorable = False`** has a reason in `exclusion_reason`: dry-run phase, a
  price source other than MT5, or a server other than the registered one. Such
  rows never enter a verdict.
- **`feature code changed since this version was trained`** in `failures.csv`
  means a production feature module was edited. Models refuse to predict on
  shifted features until the next refit trains on the new code.

## Phases

1. **dry run** (`phase = dry_run`): the first two weeks or more. Rows are logged
   and settled but never scorable. The dry run measures real coverage and
   flushes out operational faults before any alpha exists.
2. **scoring**: begins only after `results/forward_eval/PRE_REGISTRATION.md`
   is committed. The logger checks this by itself. Verdicts are issued only at
   the registered sample size. Running figures before that are labelled
   *interim, not adjudicating*.

## If something is wrong

- **Terminal logged into another account or server:** rows keep logging with
  the server name, and rows from a server other than the registered one are
  excluded from scoring. Log back into the registered server.
- **Stop everything:** run `uninstall_tasks.ps1`. Nothing in serving depends on
  these tasks.
