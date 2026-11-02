# External foundation models — runbook

openspec change `tsfm-second-opinion`. Two external, open-weight time-series models,
**Chronos-2** (Amazon) and **TimesFM 2.5** (Google), forecast the size of the next H1
move of gold and EUR/USD. They are a pinned **second opinion**: their own registered
family (`external_tsfm`, 4 cells at α = 0.05/4), never an input to the in-house models.
Research only, gross of cost, no trade.

## What runs where

| piece | environment | writes |
|---|---|---|
| `src/external/tsfm/worker.py` (the models) | `.venv-tsfm` (Python 3.11, `requirements-tsfm.txt`) | nothing but its reply |
| everything else (`forward_logger`, `view`, `replay`, …) | the app's `.venv` | `results/external_tsfm/`, `research_models/external_tsfm/` |

The app's `.venv` and `requirements.txt` never change. Weights are pinned by Hugging Face
revision and SHA-256 (`artifact_manifest.csv`) and verified before every load.

## Install (once)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\external_tsfm\install.ps1      # .venv-tsfm, pinned packages, weights, smoke test
python -m src.external.tsfm.vendor --verify                                       # every manifest entry: digest or MISMATCH
powershell -ExecutionPolicy Bypass -File scripts\external_tsfm\install_task.ps1  # the hourly task
```

`install.ps1` ends with `OK: both models return 9 increasing deciles.` and writes the
resolved package versions to `results/external_tsfm/env_freeze.txt`.

## Before registration (once, in this order)

```powershell
python -m src.external.tsfm.finetune --freeze     # pre-window H1 history -> results\external_tsfm\finetune\
python -m src.external.tsfm.finetune --train      # LoRA, 2 models x 2 instruments, GPU, a few hours
python -m src.external.tsfm.finetune --check      # reload every adapter: "identical"
python -m src.external.tsfm.baseline --fit        # frozen EUR/USD GARCH x hour-of-day
python -m src.external.tsfm.replay --freeze       # clean-window bars -> results\external_tsfm\replay\
python -m src.external.tsfm.replay                # 8 configurations per model x instrument, GPU
python -m src.external.tsfm.prereg --write        # PRE_REGISTRATION.md / .json
```

The replay is **descriptive**: it chooses one configuration per model and instrument by
the lowest first-half MAE and gives no verdict. Review `replay/REPLAY.md` and
`PRE_REGISTRATION.md`, then commit both registration files. From the next pass on, rows
are `registered`; before that they are `dry_run` and never scored. `prereg --write`
refuses once the registration is committed.

## Each hour

| time (Sofia) | task | does |
|---|---|---|
| :01 | `EURUSDProphet-ExternalLogger` | forecasts the newest closed H1 bar of gold and EUR/USD with both models; gaps; settlement; one health row |
| :03 | `EURUSDProphet-GoldLogger` | the gold pass; its MT5 export carries the second-opinion line |

One worker call per model per pass, with a 120 s limit. The worker uses the GPU when at
least 2 GB is free, and the CPU otherwise. A pass takes about 30 s.

```powershell
python -m src.external.tsfm.forward_logger                                        # one pass by hand
Get-ScheduledTask -TaskName EURUSDProphet-ExternalLogger | Select-Object State
Get-Content research_models\external_tsfm\EURUSDProphet-ExternalLogger.log -Tail 5
```

## Files (`results/external_tsfm/forward/`, append-only)

- `predictions.csv`: model, instrument, as-of bar and close, configuration, the 9 deciles
  of the next return in %, `size_pct` (median of |return|, the MAE-optimal point),
  `p_up` (descriptive, never judged), the EUR/USD baseline, digests, `phase`.
- `settlements.csv`: realised return and |return| with each instrument's own targets,
  `eligible` (the market-break rule), the baseline (gold's `h1_garch_hod` from the gold
  logger, EUR/USD's frozen GARCH × hour), the in-house forecast (`h1_lstm` / `h1_gbm`).
  A gold row waits up to 6 h for the gold logger's baseline.
- `gaps.csv`: bars missed while no pass ran; never forecast late.
- `failures.csv`: per model and instrument; one model's failure never stops the other.
- `health.csv`: one row per pass: `ok`, `partial`, `mt5 unreachable`,
  `external environment unavailable` or `failed`.

```powershell
Import-Csv results\external_tsfm\forward\health.csv | Select-Object -Last 3 | Format-Table run_at, status, forecasts, gaps, settled, failures
```

## Scoring (once per cell)

```powershell
python -m src.external.tsfm.forward_report --score
```

It refuses before registration. Each cell is scored once, at 1,500 paired eligible
registered rows or on 2027-04-30 with what it has (fewer than 500 rows reads
`UNDERPOWERED — NO DECISION`). A two-sided block bootstrap is applied to
|baseline error| − |model error| (block 24, 2,000 resamples, seed 42, α = 0.0125).
KEEP when the lower bound is above 0. Verdicts are appended to `verdicts.json` and never
re-scored.

## Where to see it

- `/second-opinion` (More menu): both instruments, both models, agreement, the record.
- `/gold`: the "Second opinion" card.
- MT5 gold panel (H1 and lower): one line, see `results/gold/forward/RUNBOOK.md`.

## Troubleshooting

- **health `external environment unavailable`**: `.venv-tsfm` is missing. Run `install.ps1`.
- **failure `... differs from the manifest digest` / `missing`**: a pinned file changed or
  was deleted. Re-run `install.ps1` (the fetch refuses anything but the pinned revision).
  A fine-tuned adapter cannot be re-fetched: restore it, or register anew.
- **failure `worker timed out after 120 s`**: the PC was busy. The next pass forecasts the
  newest bar; the missed one becomes a gap.
- **health `mt5 unreachable`**: MT5 is closed or not logged in. Nothing is forecast.
- **`GoldClockMismatch` / `ClockMismatch` in failures**: the broker's bar-label clock no
  longer matches Europe/Berlin. That instrument is refused until the clock is established
  again (CLAUDE.md invariant 6).

## Rollback

```powershell
powershell -ExecutionPolicy Bypass -File scripts\external_tsfm\uninstall_task.ps1
Remove-Item -Recurse -Force .venv-tsfm, research_models\external_tsfm
```

Nothing else depends on them. `/second-opinion` then shows "no external forecasts" and the
MT5 line disappears. Logged rows under `results/external_tsfm/` stay.
