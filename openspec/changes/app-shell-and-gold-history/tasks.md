# Tasks

Before starting, ask the owner to turn off VS Code Copilot auto-commit and auto-push
(memory: project-copilot-auto-commits). Each group ends with the full suite in the state
it started in or better, so a commit made at any group boundary is safe.

## 1. Shell source and assets

- [ ] 1.1 Create `src/app_shell.py` with the navigation items and their bilingual labels:
  Home `/`, Gold `/gold`, Forecasts `/forecasts`, History `/history`, Paper trading
  `/paper-trading`, Research console `/advanced`, and a More menu holding `/h1-direction`
  and `/kronos-volatility`. Add a `render_nav(path)` that marks the current item with
  `aria-current="page"` and carries the `data-ps-shell` marker, plus the head snippet: the
  theme boot, `<link>` and `<script defer>`. Verify with unit tests in
  `tests/test_app_shell.py`: one marker, the right item current, no `/kronos-direction`,
  and every label non-empty in both languages.
- [ ] 1.2 Create `static/shell/shell.css`:
  - one token set (canonical light and dark, the `:root:not([data-theme="light"])` guard
    and `[data-theme="dark"]`);
  - the owned-page components (card, stat, pill, segmented control, tabs, table, notice,
    skeleton, button);
  - the `.ps-legacy` base layer, written only inside `:where()`;
  - the navigation and the phone menu below 720 px;
  - a visible focus ring.

  Verify with a test that every `.ps-legacy` rule sits inside `:where(` and that the file
  has no `http://`, `https://` or `@import`.
- [ ] 1.3 Create `static/shell/shell.js`:
  - the language switch (`prophet.lang`, read once from `eurusd.home.lang`, try/catch on
    every storage call, Bulgarian fallback, `<html lang>` updated, `ps:lang` dispatched);
  - the theme control (`prophet.theme`);
  - the phone menu (`aria-expanded`, Escape closes it and returns focus).

  Verify with a test that every `localStorage` call sits inside try/catch and that the
  file loads nothing external.

## 2. Inject the shell into every page

- [ ] 2.1 Add the pure-ASGI shell layer to `api.py` (additions only). It acts on the
  allowlisted page paths with a `text/html` content type and no existing marker. It
  buffers the body, inserts the head snippet before `</head>` and the navigation after
  `<body…>`, adds `ps-legacy` to `<html>` for non-shell pages, recomputes
  `Content-Length`, and sends the original body on any exception. Verify:
  - every route in the spec's list carries the shell stylesheet, script and marker exactly
    once;
  - `/api/gold`, `/api/home` and `/api/provenance` bodies are unchanged with and without
    the layer;
  - `git diff --numstat HEAD -- api.py` shows 0 removed lines.
- [ ] 2.2 Rewrite `tests/test_home_overview.py::test_advanced_is_the_old_dashboard_byte_for_byte`
  per design D1: the `/advanced` response with the three injected segments removed must
  equal `static/index.html`, and `static/index.html` must be unchanged against HEAD. Add a
  test that `src/tracking.py`, `src/paper_trading.py`, `src/inference.py` and
  `static/index.html` are byte-identical to HEAD. Verify with
  `python -m pytest -q tests/test_home_overview.py tests/test_app_shell.py tests/test_forecast_eval_report.py tests/test_h1_production.py`.
- [ ] 2.3 Document the shell in `ARCHITECTURE_DOCS.md`: a new UI section covering the
  injection layer, the allowlist, the `:where()` legacy layer and why pinned pages are
  reached this way. Update the front-door bullet in `CLAUDE.md`. Verify that both name
  `src/app_shell.py` and the `/ui/shell/` assets.

## 3. Gold history data

- [ ] 3.1 Create `src/gold/history.py` with `live(cfg)`:
  - predictions are left-joined to settlements on `key`;
  - gap records become `status = missed`;
  - each row's status is `pending`, `settled`, `not_scored` or `missed`;
  - direction and probability are derived as in `src/gold/summary.py`;
  - blanks stay blank.

  Verify in `tests/test_gold_history.py`, using tmp-path fixture CSVs, these cases:
  - the h1_lstm "down, 53 %" miss from the spec;
  - a blank volatility value staying blank;
  - a pending newest hour;
  - a gap listed as missed;
  - no settlement joined to the wrong key.
- [ ] 3.2 Add `study(cfg, stage, cadence)`. It melts the wide prediction files into the
  long schema, with `implied_price = close·exp(ret_pct/100)`,
  `actual_price = close·exp(y_ret/100)`, the status taken from `eligible`, and the
  fit-block notice per stage. Verify:
  - the spec's 2,306.03 → 2,305.76 / 2,305.53 row;
  - the row count of `results/gold/test_predictions_h1.csv` round-trips;
  - the test-stage notice contains "not used to choose".
- [ ] 3.3 Add `wide(rows)` and `summaries(rows)`:
  - `wide` gives one line per `as_of_bar`;
  - `summaries` groups by `(cell, model_digest, source)` over settled rows. It gives the
    count, the hit rate with a Wilson 95 % interval, and the LSTM volatility MAE against
    the paired GARCH cell on the intersection of their bars. Below 30 rows it reports
    only the count. Every summary is tagged `descriptive — not a verdict`.

  Verify:
  - two digests give two summaries;
  - the Wilson interval matches the textbook value for 55 of 100;
  - 29 rows give "too few";
  - the GARCH pairing uses only shared bars.
- [ ] 3.4 Add the mtime/size cache and confirm read-only and isolation. Verify:
  - a test that building every source writes no file under the repo;
  - `tests/test_gold_isolation.py` still passes, so no EUR/USD or model import is added;
  - appending a row to a fixture CSV shows up on the next call.

## 4. Gold history endpoint

- [ ] 4.1 Add `GET /api/gold/history` to `api.py` (additions only, lazy import), with:
  - parameters `source`, `cadence`, `cell`, `start`, `end`, `layout`, `offset` and
    `limit`;
  - newest first;
  - summaries over the full filtered set;
  - `total`, `source_notice` and the allowed values in the response;
  - 400 on bad input.

  Verify with the test client:
  - the H1 test block at `limit=200` returns 200 rows, the total and the right next page;
  - an unknown source or cell returns 400 with the allowed values;
  - the payload contains none of spread, pips, P&L, profit or equity.

## 5. Redesigned gold page

- [ ] 5.1 Rebuild `static/gold.html` on the shell with the tabs *Now* and *History*. *Now*
  keeps today's content and wording, still uses `fetch('/api/gold'` and keeps both verdict
  labels in both languages. Verify that `tests/test_gold_view.py` passes unchanged.
- [ ] 5.2 Build the History tab:
  - controls: source, cadence, cell, date range and layout;
  - summary cards marked "descriptive — not a verdict";
  - the permanent study-source notices;
  - the paged table, with hit and miss shown as an icon plus text;
  - the side-by-side layout from `layout=wide`;
  - a client-side CSV download of the rows shown.

  Verify with tests on the page source: every new label is in both languages, it has no
  external URL and no cost words, and it contains a `layout=wide` fetch.
- [ ] 5.3 Draw the inline-SVG chart, following the dataviz skill's method:
  - predicted against actual price for direction and return cells;
  - expected against realised move for volatility cells;
  - the newest 500 points, with a truncation note;
  - a palette validated in light and dark;
  - tabular-number axes.

  Verify in the browser (`/browse`) on the live and test sources for h1_lstm and d1_gbm.
- [ ] 5.4 Document the history in `ARCHITECTURE_DOCS.md §3.12` and
  `results/gold/forward/RUNBOOK.md`: the endpoint, the sources, the descriptive-only
  rule, and why the test block is shown but never used to choose. Verify that the docs
  name `/api/gold/history` and the three sources.

## 6. Redesigned home page

- [ ] 6.1 Restyle `static/home.html` on the shell's tokens and components, and move its
  BG/EN buttons to the shell switch (`ps:lang` listener, `prophet.lang` with the
  `eurusd.home.lang` fallback). Keep the DOM ids, the `I18N` block, `<html lang="bg">`,
  `let lang = 'bg';` and the guarded storage calls. Verify that `tests/test_home_overview.py`
  passes, apart from the test rewritten in 2.2, which also passes.

## 7. Integration checks

- [ ] 7.1 Run a browser QA pass with `/browse` over every page in the allowlist:
  - at 360 px and 1280 px;
  - in light and dark;
  - with BG and EN.

  Check that there is no horizontal scroll, that the current item is marked, that the
  keyboard can open and close the menu with Escape, that the theme does not flash, and
  that a language choice made on `/gold` carries to `/`. Record screenshots and any fixes.
- [ ] 7.2 Run the full suite (`python -m pytest -q`). Verify that it shows no new failure
  compared with the baseline taken before group 1: the protected-set and `DATA.md`
  failures from the undeclared retrain are known and outside this change.
- [ ] 7.3 Restart the server through the launcher and open `/gold`. Confirm that History
  shows the live rows logged since 2026-10-21 and the validation and test sources. Report
  the result to the owner with the page link.
