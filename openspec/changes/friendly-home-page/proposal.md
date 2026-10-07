# Proposal

## Why

The page at `/` is a research console. It has a "Fetch Live Market Data & Predict
Tomorrow" button, two side-by-side model committees, test statistics ("ΔAUC CI",
"ECE 0.364", "MAE vs GARCH"), and a retrain button. Five of its links open
technical ledgers. The owner, who built it, says they cannot read it. A future
buyer could read it even less. The owner wants one screen that answers, in plain
words and in their language, three questions:

1. What do the models say?
2. How big a move is expected?
3. How far can I trust it so far?

The owner chose to build it for themselves first and for buyers later. This change
builds that screen. It keeps every existing page reachable, and it keeps the
project's honesty: there is no proven directional edge yet, and the forward
ledgers decide that, not the page.

## What Changes

- **New home page at `/`.** One screen in plain language with three tiles:
  - *Tomorrow* — the daily direction, written as words such as "slightly up" or
    "no clear signal", not as a raw probability.
  - *Today's session* — the latest 15-minute to 4-hour forecasts for the owner's
    15:30–23:00 Sofia session, taken from the forward-evaluation logs.
  - *Expected movement* — the next-day volatility forecast as "± X %".
- **A "How reliable is this so far?" strip.** It shows the forward hit rate next
  to the 50 % coin-flip line, with the sample size and an explicit "too early to
  tell" state while the sample is small.
- **A permanent honesty notice.** "No proven edge yet. Not financial advice.
  Simulated tracking only." It cannot be dismissed, because buyers will see this
  page.
- **A Bulgarian / English switch.** Every label exists in both languages. The
  choice is remembered per browser. The default is Bulgarian.
- **A read-only `GET /api/home` endpoint.** It assembles the home data from
  existing logs and runs no model. It is cheap and never writes. A refresh button
  on the page calls the existing `POST /api/predict`, which is already idempotent
  per day.
- **The current dashboard moves to `/advanced`, byte-identical.** The home page
  links to it as "Advanced / technical view". All other routes are unchanged.
- **BREAKING (for bookmarks only):** `/` stops serving the research console. The
  console now lives at `/advanced`. Back-links on other pages that say
  "dashboard" now land on the new home page, which links onward to `/advanced`.

## Capabilities

### New Capabilities
- `home-overview`: the plain-language home page, the bilingual labels, the
  confidence-wording rules, the reliability strip, the honesty notice, and the
  read-only `/api/home` summary endpoint it renders from.

### Modified Capabilities
- None. `openspec/specs/` has no existing capabilities.

## Impact

- **New files:** `static/home.html` (page, inline CSS/JS, bilingual dictionary)
  and `src/home_summary.py` (read-only aggregation). New tests go in
  `tests/test_home_overview.py`.
- **`api.py`:** new `/api/home` and `/advanced` routes. `read_root` gains a
  check that serves `home.html` first and keeps the existing `index.html` line as
  the fallback. This is additive only (0 deleted lines), so the guards in
  `tests/test_external_kronos.py` and `tests/test_forecast_eval_report.py` need no
  exemption.
- **Unchanged, byte-identical:** `static/index.html` (now at `/advanced`),
  `src/inference.py`, `src/paper_trading.py`, `src/tracking.py`, everything
  under `models/` and `src/forecast_eval/`. No retrain and no `RETRAIN:` line.
- **Reads (never writes):** `results/prediction_log.csv`, the paper-trading
  ledgers built by the existing `build_all_ledgers`, and
  `results/forward_eval/*.csv` through the existing `report` helpers.
- **Out of scope:** payments, user accounts, marketing or landing pages, a
  public deployment, redesigning the other 7 pages, and any broker or order code.
  Selling the service is a later decision. It is gated on the forward ledgers
  showing an edge and on a check of whether selling signals in the EU counts as
  regulated investment advice (MiFID II).
