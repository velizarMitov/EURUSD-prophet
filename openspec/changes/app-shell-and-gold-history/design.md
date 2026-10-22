# Design

## Context

See proposal.md for why. The constraints that shape the approach:

- **Ten HTML routes, three kinds of source.**
  - Owned static files: `static/home.html` and `static/gold.html`.
  - An additive-only static file: `static/index.html`, served at `/advanced`.
    `tests/test_forecast_eval_report.py` asserts it, and `api.py`, lose no line.
  - Server-rendered pages:
    - `/history` and `/paper-trading` come from `src/tracking.py` and
      `src/paper_trading.py`. Both are byte-pinned against HEAD and by the protected-set
      fixtures.
    - `/h1-direction`, `/kronos-volatility` and `/forecasts` come from their own
      modules.
    - `/kronos-direction` is an inline string in `api.py`.
- **Static files are mounted at `/ui`.** The page routes serve the HTML; `/ui/...` serves
  assets.
- **The home and gold pages already share tokens.** Both define the same `:root` palette
  with a dark override and a `data-theme` hook, and both use the language key
  `eurusd.home.lang`. The gold page reuses the home page's style but keeps its own copy.
- **Gold history already exists on disk.** The logger appends to
  `results/gold/forward/predictions.csv`, `settlements.csv` and `gaps.csv`, using
  `key = cell|cadence|as_of_bar`. The study saved one wide file per stage and cadence,
  `results/gold/{validation,test}_predictions_{d1,h1}.csv`, with these columns:
  - `time`, `close`, `eligible`, `y_dir`, `y_ret`, `y_vol`;
  - one `<cell>__p_up`, `<cell>__ret_pct` and `<cell>__vol_pct` per cell.

  Validation came from fits on `[0:70]` and test from fits on `[0:80]`, per
  `src/gold/splits.py`.
- **Gold isolation.** `tests/test_gold_isolation.py` restricts what `src/gold/*` may
  import: no EUR/USD serving modules and no `config.json`.
- **Bar-label clock.** Gold times are Europe/Berlin wall-clock labels stored with a
  `+00:00` suffix (`src/gold/summary.py::label_clock_now`). The history must show them as
  labels, not convert them as if they were UTC.

## Goals / Non-Goals

**Goals:**
- One source for the navigation markup, so ten pages cannot drift apart again.
- No edit to a pinned module, and zero removed lines in `api.py` and `static/index.html`.
- History endpoints that stay fast on the H1 test block (about 14,000 rows) and are
  never slower than the files they read.

**Non-Goals:**
- Restyling the internals of the research console or of the server-rendered pages beyond
  a low-specificity base layer. Their own inline styles keep winning.
- Translating server-rendered page text.
- A charting library, a build step, or any npm or pip dependency.

## Decisions

### D1. Inject the shell with an ASGI layer in `api.py`, from one Python source

`src/app_shell.py` owns the shell: the navigation items, their bilingual labels, and the
markup they render to. A small pure-ASGI middleware in `api.py` uses it for responses
that meet three conditions:
- the path is on an explicit allowlist of page routes;
- the content type is `text/html`;
- the body does not already contain the shell marker `data-ps-shell`.

For each such response, the middleware:
- buffers the body, which works for both `HTMLResponse` and `FileResponse`;
- inserts the theme-boot snippet, `<link href="/ui/shell/shell.css">` and
  `<script defer src="/ui/shell/shell.js">` before `</head>`;
- inserts the `<nav data-ps-shell>` markup right after `<body…>`, with
  `aria-current="page"` set from the path;
- adds `ps-legacy` to `<html>` for pages not built on the shell;
- recomputes `Content-Length`.

Alternatives considered:
- *Wrap each route's return value*, as `_with_h1_nav` does. The routes' return lines
  would have to change, which deletes lines in `api.py`. It also misses `FileResponse`
  pages.
- *`BaseHTTPMiddleware`*. It changes streaming and exception behaviour for every route,
  including `/api/predict`. A pure ASGI layer that only touches allowlisted paths is
  narrower.
- *Render the navigation in JS only.* Then the page has no navigation without JS, and the
  spec's "landmark exactly once" cannot be checked on the HTML.

Owned pages (`home.html`, `gold.html`) get the shell from the same layer, so there is one
code path. Their source contains no navigation markup.

`static/index.html` needs no tag added, because the layer serves `/advanced` like any other
page. The file itself stays byte-identical.

**This narrows an existing contract.** `friendly-home-page` promised that `/advanced`
serves `index.html` byte for byte, and `tests/test_home_overview.py::
test_advanced_is_the_old_dashboard_byte_for_byte` enforces that. The owner chose to include
the research console in the redesign, so the response now differs by the injected shell.
The test is rewritten as follows:
- `/advanced` must equal `index.html` once the shell's three injected segments are
  removed;
- `index.html` on disk must be unchanged.

The console's content, scripts and behaviour stay exactly as they were; only the
navigation and base layer are added. If the owner wants `/advanced` left out of the
redesign, the path is dropped from the allowlist and the test stays as it is. Nothing
else in the change depends on it.

### D2. `shell.css`: tokens and components for owned pages, a `:where()` base layer for legacy pages

- **Tokens.** One token set lives on `:root`, with dark values under
  `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and
  again under `:root[data-theme="dark"]`. It is the existing home and gold palette made
  canonical, plus spacing, radius, elevation and type-scale tokens.
- **Components for owned pages:** card, stat, pill, segmented control, tabs, table,
  notice, skeleton and button.
- **The `.ps-legacy` base layer** is written entirely inside `:where(...)` selectors, so
  its specificity is zero:
  - background and ink;
  - font stack;
  - table rules with tabular numbers;
  - link colour;
  - focus outlines.

  A pinned page's inline `style=` attributes and `<style>` blocks therefore always win.
  The page gains readability and keeps its layout.

Alternative considered: *a full reset on legacy pages.* That risks breaking a pinned
page's layout that nobody can edit.

### D3. `shell.js`: language, theme and the phone menu

- **Language.**
  - The key is `prophet.lang`. If it is missing, the script reads the old
    `eurusd.home.lang` once.
  - Every storage access is wrapped in try/catch, with Bulgarian as the fallback.
  - On a switch it sets `<html lang>`, updates the navigation's `data-bg`/`data-en`
    labels, and dispatches a `ps:lang` event.
  - Owned pages listen for the event and re-render from their own dictionaries.
- **Theme.**
  - The key is `prophet.theme` (`system` | `light` | `dark`).
  - An inline boot snippet in `<head>` sets `data-theme` before first paint.
  - `shell.js` only wires the control.
- **Phone menu.** Below 720 px the navigation becomes a disclosure button with
  `aria-expanded`. Escape closes it and focus returns to the button.

### D4. Gold history is a pure read module with an mtime cache

`src/gold/history.py`, gold-isolated, with this outline:

- `live(cfg)` joins predictions to settlements on `key`, a left join from predictions.
  Gap records are appended as `status = missed`. Each row's status is `pending`,
  `settled`, `not_scored` (ineligible) or `missed`.
- `study(cfg, stage, cadence)` melts the wide file into the same long schema:
  - one row per `(time, cell)`;
  - `implied_price = close·exp(ret_pct/100)`;
  - `actual_price = close·exp(y_ret/100)`;
  - the realised direction, return and move come from `y_dir`, `y_ret` and `y_vol`;
  - `status = settled` or `not_scored` from `eligible`;
  - the digest comes from `artifact_manifest.csv`, with a fixed tag per stage.
- `wide(rows)` pivots to one line per `as_of_bar` with a column group per cell and one
  realised column set.
- `summaries(rows)` groups by `(cell, model_digest, source)` over `settled` rows:
  - the count;
  - the hit rate with a Wilson 95 % interval, implemented here because importing
    `src/home_summary.py` would break gold isolation;
  - for `*_lstm` cells, the mean absolute volatility error and the same metric for the
    paired GARCH cell, on the intersection of their `as_of_bar`s.

  Below 30 rows, a summary reports only the count. Every summary carries
  `"kind": "descriptive — not a verdict"`.
- Parsed frames are cached by the files' `(path, mtime, size)`. A reload after the
  logger appends picks up the new rows. The H1 test block is parsed once per process.

Live and study rows share one schema, so the page renders both with one table and one
chart.

The fields `predicted_dir`/`probability` follow `src/gold/summary.py`: the direction is
up when `p_up ≥ 0.5`, and the probability is that of the chosen side. Rows are taken
whole, and blanks stay blank.

### D5. `GET /api/gold/history` parameters, validated with a 400 on bad input

| Parameter | Values |
|---|---|
| `source` | `live`, `validation` or `test` (default `live`) |
| `cadence` | `H1` or `D1` (default `H1`) |
| `cell` | optional, one of that cadence's cells |
| `start` and `end` | optional label-clock dates |
| `layout` | `long` or `wide` (default `long`) |
| `offset` | 0 or more |
| `limit` | 1 to 1000 (default 200) |

- Rows come newest first.
- `summaries` are computed over the whole filtered set, not the page.
- The response carries `total`, `source_notice` and the allowed values.

`api.py` gains this one route, lazily importing `src.gold.history`, as `/api/gold` does.

CSV download is client-side, from the rows on screen. That needs no second endpoint and
cannot drift from what the page shows.

### D6. `/gold` layout

- **Tabs:** *Now* (today's content, restyled) and *History*.
- **History controls:** source, cadence, cell, date range and layout (by model / side by
  side).
- **Below the controls:**
  - the summary cards;
  - an inline-SVG chart: a price line for direction and return cells, and an
    expected-vs-realised move chart for volatility cells. It shows the newest 500 points
    of the filtered set, with a note when it is truncated;
  - the paged table, with a hit or miss icon plus text, never colour alone.
- **Source notices.** The study sources show a permanent notice, using the wording
  required in the spec. The live source shows the dry-run phase when the rows carry it.
- **Charts.** They follow the dataviz skill's method: the palette is validated in both
  themes and the axes use tabular numbers.

### D7. Home page

The home page is restyled on the shell's tokens and components. Its DOM ids, dictionary
keys, fetch calls and copy stay as they are, so `tests/test_home_overview.py` keeps
passing without edits. `tests/test_home_overview.py` asserts several things about the page source:
- `<html lang="bg">` is present;
- `let lang = 'bg';` is present;
- the page's own `localStorage` calls are guarded by try/catch.

So the home page keeps its language variable and its guarded storage calls. Its BG/EN
buttons move out of its header, and the page listens for `ps:lang` from the shell's
switch instead. It reads the shell's `prophet.lang`, falling back to `eurusd.home.lang`.

## Risks / Trade-offs

- **[The ASGI layer breaks a response]** → It acts only on allowlisted paths with an HTML
  content type, it passes everything else byte for byte, and on any exception it sends the
  original body. A test hits every API route and compares bodies with and without the
  layer.
- **[A legacy page still looks dated]** → Accepted. The pinned modules cannot change, so
  the zero-specificity base layer is the honest ceiling. The navigation and tokens still
  unify the app.
- **[The `_H1_NAV` footer duplicates the new navigation]** → Accepted for now. Removing it
  deletes lines in `api.py`. A later change that lifts the additive-only rule can drop it.
- **[The owner reads descriptive live hit rates before the forward pre-registration]** →
  The registration is mechanical: its window and n come from coverage, per
  `forward_report.build_registration`, and peeking cannot change the frozen models. Every
  summary is labelled descriptive. The forward verdict stays with `forward_report`.
- **[The test block on screen invites choosing a model]** → The notice says it was not
  used to choose. The study's verdicts are already recorded, so nothing on this page can
  change them.
- **[Copilot auto-commits half-finished work]** → Each task group ends in a state where
  the suite passes. Ask the owner to turn off auto-commit before applying.

## Migration Plan

- The change is additive. Deploying it means restarting the server through the launcher,
  which restarts it automatically when `.py` files are newer.
- Rollback: revert the change's commit. No data file is written and no schema changes, so
  nothing needs migrating.
