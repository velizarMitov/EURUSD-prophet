# Proposal

## Why

Starting the program is the owner's least favourite part. `start.bat` keeps a
black console window open for as long as the server runs, and closing it kills
the app. `scripts/forecast_eval/view_forecasts.cmd` starts a second server that
nobody needs any more, because `/forecasts` lives inside the app. And after
every code update the owner is told to "restart `python api.py`", which is easy
to forget and leaves the browser on stale code.

Two loose ends from earlier work are also bundled here:

- **A display bug in the forecasts view.**
  `report.latest_forecasts` uses pandas `groupby().last()`, which skips blank
  cells. When the newest forecast for a cell has no probability, `/forecasts`
  shows an older forecast's probability as if it were current.
- **The `friendly-home-page` change is not finished.**
  - The `DATA.md` row-count test fails because the live H1 cache grew.
  - The wording review (task 5.2) is still open. The owner asked for it to be
    done on their behalf.

## What Changes

- **One desktop icon, "EUR/USD Prophet".** Double-click it and:
  - if the server is not running, it starts hidden, with no console window, and
    the browser opens once the server answers;
  - if the server is already running, the browser simply opens.

  There is no stop control (owner's choice). The server runs until log-off or
  shutdown.
- **Automatic restart on stale code.** If any Python file the server loads
  (`api.py`, `src/**/*.py`) changed after the running server started, the icon
  restarts the server before opening the browser. "Restart `python api.py`"
  disappears from the instructions. Changes to HTML pages need no restart,
  because they are served from disk.
- **Visible failure.** If the server does not come up within 3 minutes, a plain
  message box (in Bulgarian) names the log file to look at. Server output goes to
  `research_models/server/server.log`, which is gitignored.
- **Fewer files.**
  - **BREAKING (for anyone double-clicking it):** `start.bat` is removed.
  - `scripts/forecast_eval/view_forecasts.cmd` and its test are removed. The
    viewer remains reachable as `python -m src.forecast_eval.report --serve`.
  - The 4 background-task files stay as they are. The owner never touches them,
    and they already run hidden.
  - New: `scripts/launch.vbs` (the icon's target) and
    `scripts/install_shortcut.ps1` (run once to create the icon).
- **Forecasts view fix.** The newest forecast per cell is taken as a whole row,
  so a missing probability shows as "no answer", never as an older value.
- **Finish `friendly-home-page`.**
  - Update `DATA.md` to the committed size of `results/eurusd_h1.csv`. The
    decision and its reason are recorded in that change.
  - Apply the wording review. The most important fix: the "Утре / Tomorrow"
    tile is mislabelled in the morning, when it shows today's session.
  - Close tasks 5.1 and 5.2.

## Capabilities

### New Capabilities
- `app-launcher`: how the owner starts the application. One icon; hidden start;
  reuse of a running server; restart when its code is stale; a visible failure
  message; no console window.
- `forecast-view`: what the `/forecasts` headline may show as current. The
  newest forecast per cell is taken whole. A finished window is never shown as
  current (this records the 2026-10-07 fix already in `report.py`).

### Modified Capabilities
- None. `openspec/specs/` holds no archived capabilities yet. The home-page
  wording edits change labels, not `home-overview` requirements.

## Impact

- **New:** `scripts/launch.vbs`, `scripts/install_shortcut.ps1`,
  `tests/test_launcher.py`, and an `.ico` file for the icon.
- **Removed:** `start.bat`, `scripts/forecast_eval/view_forecasts.cmd`, and
  `test_the_launcher_script_points_at_the_viewer`.
- **Edited:**
  - `src/forecast_eval/report.py` (`latest_forecasts`);
  - `tests/test_forecast_eval_report.py`;
  - `static/home.html` (labels only);
  - `DATA.md`;
  - `README.md`, `HOW_TO_RUN.md`, `results/forward_eval/RUNBOOK.md` and
    `CLAUDE.md` (start instructions).
- **Untouched:** `api.py` (its stale `start.bat` comment stays, because the file
  is additive-only), `src/inference.py`, `src/paper_trading.py`, `models/`,
  `static/index.html`, and the scheduled-task scripts. No retrain.
