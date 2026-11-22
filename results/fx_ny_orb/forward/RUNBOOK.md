# fx-ny-orb-hot — runbook

## Once

1. **Commit the pre-registration.** Commit `results/fx_ny_orb/PRE_REGISTRATION.md` and `.json`
   unchanged. Check with `python -m src.fx_ny_orb.prereg --check`. Forward days count from the
   commit.
2. **Install the indicator.** Run `powershell -ExecutionPolicy Bypass -File
   scripts\fx_ny_orb\install_mt5_indicator.ps1`. In MT5, open Navigator > Indicators >
   EURUSDProphet > ProphetNYOrb and drag it onto EURUSD and GBPUSD M15 charts.
3. **Install the tasks.** Run `powershell -ExecutionPolicy Bypass -File
   scripts\fx_ny_orb\install_tasks.ps1`. Add `-DryRun` to only print the definitions.
4. **Score the historical cells once.** Run `python -m src.fx_ny_orb.score --family historical`,
   then `python -m src.fx_ny_orb.report`.

## Every trading day (automatic)

| Sofia time | Task | Does |
|---|---|---|
| 16:05 | `EURUSDProphet-NYOrbForecast` | logs the range forecast, the hot flag and the opening range; rebuilds the chart file |
| 16:01–20:01 every 15 min | `EURUSDProphet-NYOrbSignal` | logs the rule's first entry (whatever the gate); rebuilds the chart file |
| 23:05 | `EURUSDProphet-NYOrbSettle` | records the realised range and the simulated trade, and settles the signals into the ledger |

The logs are in `research_models\fx_ny_orb\<task>.log`.

## What the chart shows

- **Every day.**
  - A blue box for the opening range, with its high and low lines to 21:00 Berlin.
  - A dashed band for the forecast size, red on a hot day. It shows size, not direction.
  - After 23:00 Sofia, a dotted box for the realised range.
- **Past days.** The simulated trade: entry arrow, SL/TP lines and an X at the exit, coloured by net R.
- **Today.** An entry arrow with SL/TP lines **only after a historical KEEP** of `orb_hot_net_r`, and
  only on a hot day. Until then the corner says so.

## If something is wrong

- **"No ny_orb.tsv yet" on the chart.** Run `python -m src.fx_ny_orb.export`.
- **A pass prints "too early", "weekend" or "after the entry cut-off".** That is normal; passes check
  the clock themselves.
- **The PC was off.** Missed days are never backfilled. The settle pass marks a missed signal as
  `missed` in the ledger.
- **`FrozenChanged`.** A frozen file was altered. Restore it from the digests; never refit silently.

## Remove

Run `scripts\fx_ny_orb\uninstall_tasks.ps1` and `scripts\fx_ny_orb\uninstall_mt5_indicator.ps1`. They
remove only this family's tasks, indicator files and `ny_orb.tsv`.
