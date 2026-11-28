# gold-nyrange-forward — runbook

## Once

1. **Commit the pre-registration.** Commit `results/gold_nyfwd/PRE_REGISTRATION.md` and `.json`
   unchanged. Check with `python -m src.gold_nyfwd.prereg --check`. Forward days count from the day
   after the commit.
2. **Install the tasks.** Run `powershell -ExecutionPolicy Bypass -File scripts\gold_nyfwd\install_tasks.ps1`.
   Add `-DryRun` to only print the definitions.
3. **Install the indicator.** Run
   `powershell -ExecutionPolicy Bypass -File scripts\gold_nyfwd\install_mt5_indicator.ps1`. In MT5,
   open Navigator > Indicators > EURUSDProphet > ProphetGoldNYRange and drag it onto a GOLD H1 (or
   lower) chart.

## Every trading day (automatic)

| Sofia time | Task | Does |
|---|---|---|
| 16:08 | `EURUSDProphet-GoldNYForecast` | logs the expected size of today's NY session (or the reason there is none); rebuilds the chart file |
| 23:08 | `EURUSDProphet-GoldNYSettle` | records the realised range and both squared errors; rebuilds the chart file |

The logs are in `research_models\gold_nyfwd\<task>.log`. A pass run by hand at another time is safe: it
checks the clock itself and logs each day at most once.

## Early December: the dollar-index contract

`USDIndDec26` expires on 2026-12-11 and stops covering days from 2026-12-10. **Check MT5 once a week
from 2026-12-01** (Market Watch > Symbols, search `USDInd`) for the next contract, e.g. `USDIndMar27`.

- When it is listed, nothing needs doing: the next forecast pass records it and checks it against EURUSD.
- Until it is listed, days from 2026-12-10 are logged `no dollar data` and never count. The horizon is
  not extended.

## What the chart shows

- From 16:08 Sofia, a dashed band from 16:00 to 23:00 Sofia (15:00–22:00 on the chart's Berlin clock),
  centred on the 14:00 Berlin close, as tall as the expected range. A dotted line marks the centre.
- After 23:08 Sofia, a dotted box with the realised high and low.
- The corner shows the forward days so far out of 60 and that there is no verdict yet.
- It shows size only: no direction and nothing to act on.

## If something is wrong

- **"No gold_ny_range.tsv yet" on the chart.** Run `python -m src.gold_nyfwd.export`.
- **"This indicator is for GOLD only".** It was put on another symbol; move it to a GOLD chart.
- **A pass prints "too early", "weekend" or "already logged".** That is normal.
- **`NOT READY` / `PreregNotCommitted`.** The pre-registration is not committed, or was edited.
  Restore it with `git checkout -- results/gold_nyfwd`, never re-write it.
- **`ConfigChanged`.** `src/gold_nyfwd/gold_nyfwd_config.json` was edited. Restore it; a changed
  config is a new hypothesis.
- **`FrozenChanged`.** gold_bbnr's `ols.json` was altered. Restore it; never refit.
- **`symbol(s) ... not listed` or `MT5 terminal is not reachable`.** Start the MT5 terminal at
  `C:\Users\test\Desktop\mt5app`. That day's pass logged nothing; it is not backfilled.
- **The PC was off.** Missed days are never backfilled. `python -m src.gold_nyfwd.report` counts them.

## Scoring (once)

When the report shows 60 forward days, or on 2027-04-30, run `python -m src.gold_nyfwd.score`, then
`python -m src.gold_nyfwd.report`. A second scoring refuses.

## Remove

Run `scripts\gold_nyfwd\uninstall_tasks.ps1` and `scripts\gold_nyfwd\uninstall_mt5_indicator.ps1`. They
remove only this family's two tasks, the indicator files and `gold_ny_range.tsv`. The logs stay.
