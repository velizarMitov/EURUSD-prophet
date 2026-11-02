# Proposal

## Why

The owner watches gold in MetaTrader 5, but the gold signals live only on `/gold` in the
browser: the h1_lstm direction and expected move, and the large-move alert. Switching
between the two windows means a signal is read late or not at all. The owner asked on
2026-11-02 to see the gold signals on the MT5 chart itself, as arrows with a description.

This puts research output on the screen where real-money trades are placed. Each mark must
carry the same honesty as `/gold`:
- h1_lstm direction cleared the one-shot test gross of cost (51.3 % vs 48.5 %), but its
  sign strategy has no trading edge (DSR 0.054);
- the large-move alert is the one forecast with proven skill, and it has no direction.

## What Changes

- **A read-only MT5 indicator** (MQL5 source tracked in the repo) draws on a gold chart
  (`GOLD` / `XAUUSD`, with broker suffixes) at H1 and lower timeframes:
  - an up or down arrow on each hour the h1_lstm forecast, by the same rule as `/gold`
    (up when P(up) > 0.5);
  - a separate two-headed marker on each hour a large-move alert covers. It never points
    up or down;
  - a short text beside each mark: the probability or the expected size, and the outcome
    once the hour has settled (right or wrong, and the realised move);
  - a full description on hover: the hour in Sofia time, the probability, the expected
    move in % and $, the outcome, and the "research, no proven edge, gross of cost"
    label;
  - a corner panel with the current forecast, any open alert, the 5-day count of right
    and wrong arrows (descriptive, never a test), and the freshness of the data.
- **History:** the forecasts and alerts of the last 5 trading days, each with its result.
  Older marks are removed.
- **A signal export.** Each gold logger pass rebuilds one file in the MT5 shared
  `Common\Files` folder from the forward logs it has just written. A failed export is
  recorded and never costs the pass. A CLI rebuilds the file on demand.
- **An installer and an uninstaller** copy the indicator into every MT5 terminal found on
  the PC and compile it there. The owner attaches it to a gold chart once.
- **Unchanged:**
  - the forward logs, the alert threshold, the alert text rules, and `/gold`;
  - the registries and every verdict;
  - h1_gbm, the D1 cells and the round-two/shadow rows. These are not drawn: the owner
    chose h1_lstm only.

## Capabilities

### New Capabilities

- `gold-forecasting/mt5-chart-signals`: the gold signals on the MT5 chart. It covers which
  signals are drawn and how (arrows, alert marker, descriptions, panel), the 5-day history
  with outcomes, the export file and its freshness, placement on the chart's own clock,
  installation, and the guarantee that the chart shows signals and never trades.

### Modified Capabilities

None. `gold-forecasting/move-alerts` already requires alert text to name size and never
direction. The chart marker follows that requirement as written.

## Impact

- **Code.**
  - New `src/gold/mt5_export.py`: builds the file from `predictions.csv`,
    `settlements.csv`, `alerts.csv` and `alert_settlements.csv`.
  - `src/gold/forward_logger.py`: one guarded export step after the alerts step, and the
    result in the health note.
  - New `mql5/Indicators/ProphetGoldSignals.mq5`.
  - New `scripts/gold/install_mt5_indicator.ps1` and `uninstall_mt5_indicator.ps1`.
- **Isolation contract (`src/gold/__init__.py`).** It gains one named exception: gold may
  write the single export file under the MT5 `Common\Files\EURUSDProphet\` folder, outside
  the repo. Nothing in the repo reads that file back.
- **Outside the repo.** The installer writes the indicator into each terminal's
  `MQL5\Indicators\EURUSDProphet\` folder.
- **Tests.**
  - New `tests/test_gold_mt5_export.py`.
  - Additions to `tests/test_gold_forward_logger.py` and `tests/test_gold_isolation.py`:
    the export path, no trading or network code in the indicator, and no buy, sell,
    entry, stop or target wording in any chart text.
- **Docs.** `ARCHITECTURE_DOCS.md` §3.12, `results/gold/forward/RUNBOOK.md` (install and
  check), and the gold bullet in `CLAUDE.md`.
- **Not touched.**
  - `models/`, `config.json`, `gold_config.json`, `gold_config_v2.json` and the registries;
  - `api.py` and every page;
  - the EUR/USD side.
- **Load.** One small file written per hourly pass. The indicator re-reads it only when its
  modified time changes, checking every 30 s.
