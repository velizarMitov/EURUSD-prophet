# Design

## Context

- `/` currently serves `static/index.html`, a 695-line research console. It
  fetches on button press only (`POST /api/predict`). Its text is the evidence
  narrative: CIs, ECE, GARCH deltas, DROP verdicts. Several tests pin its
  content (links, card text, the provenance banner).
- `api.py` and `static/index.html` are under an **additive-only** contract,
  checked by `tests/test_external_kronos.py` (against `6319df2`, with exact
  counted deletion exemptions) and by
  `tests/test_forecast_eval_report.py` (against `HEAD`). `src/inference.py` and
  `src/paper_trading.py` must stay byte-identical.
- The data the home page needs already exists:
  - `results/prediction_log.csv`: one row per as-of day, written by
    `POST /api/predict`, which is idempotent per day. It holds the `baseline_*`
    consensus, `variant_agreement` and `vol_pred_pct`.
  - `src/paper_trading.build_ledger(..., direction_column='baseline_direction')`:
    the read-only ledger (`build_all_ledgers` persists CSVs, and its `win` is net
    of spread, a cost verdict). A call is right when its signed `gross_pips > 0`.
    It joins the log to realised closes and can be slow or fail when no price
    source is reachable.
  - `src/forecast_eval/report.py`: `session_state()`, `latest_forecasts()` and
    `admitted_cells()` over `results/forward_eval/*.csv`. They are pure readers.

## Goals / Non-Goals

**Goals:**
- Put one thin aggregation layer (`src/home_summary.py`) between the logs and the
  page, so the wording rules and the reliability states are unit-testable in
  Python rather than buried in browser JS.
- Change as few existing lines as possible. Every existing page and its tests
  keep their meaning.

**Non-Goals:**
- No new charts library, no build step, no framework.
- No change to what any model predicts, or to when predictions are logged.
- No restyle of the other pages beyond what moving `/` implies.

## Decisions

1. **New file `static/home.html`. The old console moves to `/advanced` untouched.**
   Rewriting `index.html` would break about 10 content tests and the
   additive-only check, and would throw away the console the owner still uses
   for retraining and provenance.
   *Alternative:* a `/home` route that leaves `/` alone. Rejected, because the
   owner asked for the front door itself to be understandable.

2. **`read_root` serves `home.html`; a new `/advanced` route serves `index.html`.**
   The change to `read_root` is purely additive: new lines check for `home.html`
   before the existing `index.html` check, which stays as the fallback. `api.py`
   therefore loses zero lines. Both additive-only guards keep their previous
   counts, with no exemption, and both were confirmed to still fail on a stray
   deletion.
   *Alternatives:* replacing the `index.html` lines, which would have needed a
   counted exemption; or registering a second `@app.get("/")` above the old
   one, which would be two handlers on one path. Both rejected.

3. **The home data comes from `GET /api/home`, which is read-only. Refresh is
   the existing POST.** Loading the page must not run models or write logs. The
   console's discipline (on-demand only, every call logs) stays. Refresh calls
   `POST /api/predict`, whose log write is idempotent per day, then re-fetches
   `/api/home`.
   *Alternative:* call `POST /api/predict` on every page load. Rejected, because
   it is slow (TF, LSTM, H1) and logs on every visit.

4. **Each section is isolated.** `/api/home` returns `{tomorrow, movement,
   session, reliability, generated_at}`. Each block is built in its own
   try/except and degrades to `{available: false, reason}`. The reliability
   block, which needs realised prices, is fetched in a **second call**
   (`/api/home?part=reliability`). The fast tiles therefore paint immediately
   and a slow price source never blocks them.

5. **The wording lives in Python, the translation lives in the page.**
   `home_summary` emits stable keys (`no_signal`, `slightly_up`,
   `slightly_down`, `too_early`, `coin_range`, `above_coin`). `home.html` holds
   one `{key: {bg, en}}` dictionary. Tests can then assert the rules (0.52
   threshold, N < 30, 95 % Wilson bound) in Python, and assert dictionary
   completeness by parsing the HTML.
   The threshold is imported as `CONFIDENCE_THRESHOLD`, never re-typed.

6. **The reliability statistic is the 95 % Wilson score lower bound on
   `n_wins / n_positions`, with a minimum of 30.** It is descriptive, for the
   owner's orientation only. It is not a pre-registered verdict, and the page
   says "not yet proven" even above the line. The `0.05/9` feature bar and the
   forward-eval families are untouched, and this strip never counts as a
   hypothesis test.

7. **Times are shown in Europe/Sofia, computed server-side.** Server-side `now`
   keeps one clock. The session block uses `report.session_state()`, which
   already encodes the Berlin-label to Sofia mapping. The page never re-derives
   session hours.

8. **Visual language.** There are three large tiles. Each has an icon, a
   headline word and a one-line plain sentence. Colours are muted green, red
   and neutral grey: neutral for "no clear signal", and never a saturated
   "go" colour. The reliability strip is a horizontal bar with a fixed 50 %
   tick and a numeric label. A bar, not a gauge, because a gauge implies
   precision these numbers do not have, the same reasoning as in the console.
   Colours are CSS custom properties with a `prefers-color-scheme: dark` set.
   There are no external assets.

## Risks / Trade-offs

- [Bookmarks to `/` now land on the simplified page] → A prominent "Advanced /
  technical view" link. `/advanced` is documented in README, HOW_TO_RUN and the
  RUNBOOK.
- [Plain words could read as advice ("slightly up" → buy)] → A permanent
  honesty notice. The banned-words test also covers buy/sell/strong. Headlines
  never use imperative verbs.
- [A hit-rate strip can look like proof once it creeps above 50 %] → The fixed
  "not yet proven" suffix, N ≥ 30 before any non-neutral state, and a link to
  the ledgers that actually decide.
- [The paper-trading readers fetch prices (MT5/yfinance)] → They run in the
  separate reliability call with a timeout-tolerant fallback to
  `available: false`. The tiles never wait on them.

## Migration Plan

1. Ship `home.html`, `home_summary.py`, the two routes and the tests in one
   commit. No `models/` change, so no `RETRAIN:` line.
2. The owner restarts `python api.py` once. A running server does not reload.
3. Rollback: revert the commit. `/` serves `index.html` again, and nothing on
   disk needs undoing because the endpoint writes nothing.

## Open Questions

- The exact Bulgarian phrasing of the honesty notice and the tile sentences. The
  owner reviews the dictionary after the first render. Keys and rules do not
  change.
