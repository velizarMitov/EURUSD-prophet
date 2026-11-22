# fx-ny-orb-hot — NY opening-range breakout on hot days, the frozen range forecast and the MT5 chart

> **Net of the recorded spread plus 0.4 bps slippage (no swap: intraday) — not a trading result.**

Openspec change `fx-ny-orb-hot`. Code `src/fx_ny_orb/`, tests `tests/test_fx_ny_orb_*.py`, indicator
`mql5/Indicators/ProphetNYOrb.mq5`, scripts `scripts/fx_ny_orb/`, docs `ARCHITECTURE_DOCS.md §3.20`,
runbook `results/fx_ny_orb/forward/RUNBOOK.md`.

## Three parts

| Part | What | Decided by |
|---|---|---|
| [B] historical | The owner's breakout on M15 mid bars, 2018-08-08 .. 2026-08-07, on days the walk-forward HAR-X (fx_nyvol, read-only) calls hot. Non-hot days are the control. | `fx_ny_orb_hypothesis_log.csv`: `orb_hot_net_r`, `orb_hot_minus_cold`, alpha 0.05/2 each, scored once, **only after `PRE_REGISTRATION.md` and `.json` are committed** |
| [A] forward | Frozen OLS HAR-X and HAR (fit to 2026-11-20, `research_models/fx_ny_orb/frozen_*.json`, digests in `forward/frozen_manifest.csv`), forecast live every trading day | `forward/fx_ny_orb_forward_hypothesis_log.csv`: `EURUSD_harx_vs_har`, `GBPUSD_harx_vs_har`, alpha 0.05/2 each, once, at 270 settled days per pair or 2028-01-31 |
| [C] chart | `Common\Files\EURUSDProphet\ny_orb.tsv`, drawn by `ProphetNYOrb.mq5` | display only. A live entry sign appears only when `orb_hot_net_r` is KEEP |

## The rule (pre-registered, never searched)

- **NY opening range.** The M15 bars labelled 14:00, 14:15, 14:30 and 14:45 Berlin (15:00–16:00 Sofia).
- **Entry.** The first M15 close outside the range from 15:00, then the next bar's mid open. The entry
  needs the range width ≤ 0.50 F, the breakout ≤ 0.25 F beyond the level, and an entry before 19:00.
  Otherwise there is no trade that day.
- **Exits.** SL is 0.25 F and TP 0.60 F from the entry. A bar touching both counts as SL. The forced
  exit is the close of the 20:45 bar. One trade per day per pair.
- **F** is the HAR-X forecast of the session range, `exp(p_harx) × close14`.

## Power, from `labels.md`, before any outcome

| Side | Days | Trades | Dates | Minimum detectable mean (SD 1.5 R) |
|---|---|---|---|---|
| hot | 415 | **170** | 121 | 0.32 R on trades, 0.38 R on dates |
| non-hot | 3 701 | 1 951 | 1 290 | 0.10 R on trades, 0.12 R on dates |

**On hot days the range-width condition removes 235 of the 415 days (57 %).** A hot first hour is
itself wide, and the forecast already reflects that hour. The historical test can therefore see only a
large edge. This is stated before any outcome.

## Data checks

- **M15 mid bars.** Aggregated from the M1 parquets with each bar's median spread. Nothing is written to
  the horizon-study cache.
- **`close14`.** The 14:45 M15 mid close matches fx_nyvol's H1 close within 1 pip on at least 99 % of
  days.
- **Frozen models.** Our OLS reproduces fx_nyvol's last-fold HAR-X within 1e-6. Both are fitted on
  3 692 days per pair, 2012-02-06 .. 2026-11-20.

## The indicator compiles cleanly

`scripts/fx_ny_orb/install_mt5_indicator.ps1` ran on 2026-11-22. It compiled the indicator in all three
local terminals: MT5ICmarket, ActivTrades EU MetaTrader 5 and mt5app. Each log reads
`Result: 0 errors, 0 warnings`.

## Clock

The live passes use the **real time on the Berlin label clock**, never MT5's last tick. A smoke test on
Sunday 2026-11-22 found that the stale Friday tick made Friday look like "today" and backfilled a
forecast. That was fixed before any forward day, and a regression test now guards it. The stray rows
were deleted.

## Files

| File | What it is |
|---|---|
| `labels.md`, `labels.json` | entry decisions and power; no exit or result read |
| `PRE_REGISTRATION.md`, `.json` | the rule, cells, window, config digest, power and facts; the owner commits them |
| `fx_ny_orb_hypothesis_log.csv` | the two historical verdict rows, once scored |
| `hist_trades.csv.gz` | every simulated day, written only by the scoring after the gate |
| `report.md` | the descriptive report |
| `live/` | the family's own MT5 H1 (3 pairs) and M15 (2 pairs) caches, append-only |
| `forward/forecasts.csv`, `signals.csv`, `settlements.csv`, `signal_ledger.csv` | the live logs, append-only and never backfilled |
| `forward/frozen_manifest.csv`, `forward/fx_ny_orb_forward_hypothesis_log.csv` | the frozen-model digests and the forward verdicts |
