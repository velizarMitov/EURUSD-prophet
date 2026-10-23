# Proposal

## Why

The app has grown into ten pages that were built one change at a time, and it shows:

- Each page has its own styling. Some are server-rendered pages with a different look.
- There is no shared navigation. Cross-links are a line of footer text, and some pages
  are reachable only by typing their URL.
- The owner cannot move around the app or read it comfortably.

Gold has a second gap. The hourly logger already keeps every forecast and every outcome,
append-only. But `/gold` shows only the newest forecast per model, so the owner cannot see
what each model predicted against what gold actually did, or compare one model with
another.

## What Changes

- **One app shell for every page.** It adds:
  - shared design tokens (light and dark), typography and components;
  - a top navigation bar that marks the current page;
  - one BG/EN switch and one light/dark/system theme switch, both remembered across pages.

  The shell is two local files under `static/shell/`: no external scripts or fonts, so
  the app still works offline on loopback.
- **Redesigned owned pages.** The home page (`/`) and the gold page (`/gold`) are rebuilt
  on the shell. Every existing requirement for them still holds: read-only, bilingual,
  phone width, honesty notices, and no cost figures on the home page or the gold page.
- **The shell on the other pages.** It reaches pages whose source must not change:
  - pages served from pinned modules (`/history` and `/paper-trading`, rendered by the
    byte-pinned `src/tracking.py` and `src/paper_trading.py`);
  - the other server-rendered views (`/h1-direction`, `/kronos-volatility`,
    `/kronos-direction`, `/forecasts`).

  A response layer in `api.py` injects the shell into their HTML, and `api.py` only gains
  lines. The same layer serves the research console `/advanced`, so `static/index.html`
  stays byte-identical.
- **Gold history.** A new read-only endpoint, `GET /api/gold/history`, pairs each logged
  gold forecast with its settled outcome:
  - predicted against actual direction, price and size of move;
  - pending forecasts marked as pending;
  - rows grouped by model digest, so a refit never blends two models.

  It also serves the saved validation and test-block predictions as **past research**,
  labelled as such, so there is a long history from day one.
- **A History view on `/gold`.**
  - a predicted-against-actual table, filterable and paged;
  - every model's forecast for the same bar side by side;
  - an inline-SVG chart of predicted against actual price and expected against realised
    move;
  - descriptive summaries for each model: hit rate with a Wilson 95 % interval against
    a coin, and the volatility MAE against its GARCH baseline on the same rows;
  - a CSV download of what is on screen.
- **Not in this change:**
  - Pages are not translated: their own text stays English. Only the shell's labels
    are bilingual.
  - Nothing changes in the pinned modules, models, the gold logger or the gold
    registries.
  - There are no new hypothesis tests. History summaries are descriptive; gold verdicts
    still come only from the gold study and the forward pre-registration.

## Capabilities

### New Capabilities
- `app-shell`: the shared look, top navigation, language and theme switches, and how
  every HTML page in the app receives them, including pages whose source is pinned.
- `gold-forecasting/gold-history`: the read-only record of gold forecasts against
  outcomes, live and past research, and how the owner compares models with it.

### Modified Capabilities
- None. `openspec/specs/` holds only `app-launcher` and `forecast-view`. This change keeps
  their behaviour, and the URLs and content of `/forecasts` stay the same.
- The home-page and gold-view requirements live in changes that have not been archived:
  `friendly-home-page` (`home-overview`) and `gold-forecasting` (`gold-forecasting/gold-view`).
  The `app-shell` spec requires that those requirements keep holding. It does not
  rewrite them.

## Impact

- **New files:**
  - `static/shell/shell.css` and `static/shell/shell.js`
  - `src/app_shell.py`
  - `src/gold/history.py`
  - `tests/test_app_shell.py` and `tests/test_gold_history.py`
- **Rewritten:** `static/home.html` and `static/gold.html`.
- **Additive only:** `api.py`, which gains the HTML injection layer and the
  `/api/gold/history` route.
- **Untouched:** `static/index.html`, `src/tracking.py`, `src/paper_trading.py`,
  `src/inference.py`, everything under `models/`, and the gold logger and registries.
- **Data read:** `results/gold/forward/{predictions,settlements,gaps}.csv` and
  `results/gold/{validation,test}_predictions_{d1,h1}.csv`. Nothing is written.
- **Docs:** `ARCHITECTURE_DOCS.md` (gold section and a UI section) and the
  front-door bullet in `CLAUDE.md`.
- **Dependencies:** none added.
