# Proposal

## Why

The first gold study (change `gold-forecasting`, 2026-10-21) found:
- **The size of the next hour's move can be predicted:** the hourly LSTM beat GARCH on
  both validation and the one-shot test.
- **Direction and price cannot:** the models do about as well as a coin flip or as
  "no change".

The owner wants to improve the predictions and add features that make the app more
useful. The research points to three places that could help:
- **Calibrated price ranges built on the proven volatility forecast.** Adaptive conformal
  prediction is reported to give sharp, well-covered intervals on gold.
- **Gold's documented macro drivers:** real interest rates, the dollar, risk appetite,
  silver and oil.
- **Honest new attempts at direction:** abstaining when unsure, and labels based on
  barriers instead of the next close.

The first round's hourly logger also showed operational gaps:
- 8 hours were missed overnight while the PC slept;
- nothing tells the owner when the logger stops;
- there is no way to compare models on live data or to refresh them without breaking
  the frozen forward test.

## What Changes

- **A second research round with its own pre-registration.**
  - Every new hypothesis is registered before its first score, in new families with
    their own Bonferroni bar.
  - The first round's families, registries and alphas stay as they are.
  - Validation `[70:80]` is the arbiter. The page and docs disclose that the owner has
    already seen first-round results on that block.
  - The test block, `[80:100]`, is spent and is not used again.
  - The final judge is forward data.
- **Price ranges (new research and a new display).**
  - For the next hour and the next day, gold is shown as "between X and Y" at 80 % and
    95 %.
  - The ranges are built from the volatility forecast and calibrated by adaptive
    conformal inference.
  - They are scored on coverage and on interval score, against a GARCH-based range.
  - Live coverage is tracked in the forward log.
- **Macro drivers (new research).** These are daily features, each lagged so it was
  known at the as-of close:
  - the US 10-year real yield (FRED DFII10);
  - the broad dollar index (DTWEXBGS);
  - VIX (VIXCLS);
  - silver and Brent returns from MT5.

  They are tested as an add-on to the daily models, for volatility, return and ranges.
  Daily direction stays underpowered by the registered power rule.
- **Direction, new approach (new research):**
  - a selective model that abstains when unsure, with a coverage floor fixed in
    advance;
  - triple-barrier labels on H1;
  - a 4-hour horizon.

  Each cell must meet the first round's power rule (a 3 % edge at 80 % power), and any
  that does not is registered "UNDERPOWERED — NO DECISION".
- **Live model leaderboard (new feature).** A *Leaderboard* tab on `/gold` and
  `GET /api/gold/leaderboard` rank every model and version on live data:
  - hit rate against a coin;
  - move error against its GARCH baseline;
  - range coverage.

  The ranking is descriptive and is never a verdict.
- **A reliable logger (new feature):**
  - The scheduled task wakes the PC, runs a missed hour as soon as it can, and runs on
    battery.
  - A health record tracks the last run, the last success, missed hours in the last
    24 h and MT5 reachability.
  - `/gold` shows a warning when the logger has stalled.
- **Large-move alerts (new feature).**
  - When the hourly expected move exceeds a threshold fixed from training data, the
    logger writes an alert row, the page shows a banner, and Windows shows a
    notification.
  - The alert names the size of the move, never its direction.
- **Scheduled retraining with versions (new feature).**
  - A monthly task refits the gold models into a new, digest-pinned version.
  - The new version forecasts **in shadow** next to the frozen registered models and
    never replaces them during a registered forward window.
  - Every row and score is kept per version. Promotion is a manual, recorded owner
    decision.

## Capabilities

### New Capabilities
- `gold-forecasting/registration-amendments`: how a later research round adds gold
  hypothesis families, with pre-registration, disclosure of what has already been seen,
  and an unchanged power rule.
- `gold-forecasting/price-ranges`: calibrated next-hour and next-day price intervals,
  their scoring and their live coverage.
- `gold-forecasting/macro-drivers`: daily gold macro features, their no-look-ahead
  timing, and their add-on test.
- `gold-forecasting/direction-research`: selective, barrier-label and 4-hour direction
  cells, and their power gate.
- `gold-forecasting/leaderboard`: the live, descriptive comparison of gold models and
  versions.
- `gold-forecasting/logger-health`: keeping the hourly logger running, and making a
  stall visible.
- `gold-forecasting/move-alerts`: alerts when the hourly expected move is unusually
  large.
- `gold-forecasting/model-versions`: scheduled refits as shadow versions, and how a
  version is promoted.

### Modified Capabilities
- None in `openspec/specs/`.
- `gold-forecasting/gold-history` (main spec) keeps its requirements. Its existing
  per-digest grouping already separates versions.
- The first-round gold specs (`study`, `forward-arbiter`, `gold-view`) live in the
  unarchived change `gold-forecasting` and are not rewritten. This change relies on
  their rule that models stay frozen for the registered window.

## Impact

- **New modules under `src/gold/`:** `ranges`, `macro`, `direction2`, `prereg2`,
  `leaderboard`, `health`, `alerts` and `versions`.
- **Extended:**
  - `forward_logger`: range columns, alerts, health, and shadow versions;
  - `study`: second-round stages;
  - `gold_config.json`;
  - `static/gold.html`: the Leaderboard tab, range display and warning banners;
  - `scripts/gold/`: task settings and the retrain task.
- **New files:**
  - `results/gold/PRE_REGISTRATION_V2.{md,json}`;
  - three new registries: `gold_{range,macro,direction2}_hypothesis_log.csv`;
  - `results/gold/macro/*.csv` caches;
  - `results/gold/forward/{alerts,health}.csv`;
  - `research_models/gold/versions/`.
- **`api.py`:** additive only, with `GET /api/gold/leaderboard` and
  `GET /api/gold/health`. `src/app_shell.py` and the pinned EUR/USD modules are
  untouched.
- **Production:** nothing under `models/`, no `config.json`, and no EUR/USD registry
  changes. The gold isolation tests still hold.
- **Data:** FRED is reached through the existing generic `src.macro_data.fetch_fred_feature`,
  with no API key needed. `SILVER` and `Brent` come from MT5 and are confirmed present.
- **Dependencies:** none added. The Windows notification uses PowerShell's built-in
  toast API.
- **Order:** the first round's forward pre-registration (`gold-forecasting` task 9.1,
  2026-11-04 at the earliest) is independent of this change and goes ahead first. New
  models start their own forward dry run.
