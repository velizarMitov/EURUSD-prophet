# Tasks

The code already exists in commit `fe3520b`, which was made unrequested. Each task checks
it against the spec, fills any gap and only then is ticked.

## 1. Notification that waits (spec: An alert is logged, shown and notified once)

- [x] 1.1 Check that `src/gold/alerts.py` `_TOAST_PS` builds a `ToastGeneric` toast with
  `scenario="reminder"`, the reminder sound and a system `dismiss` action. The texts must
  still enter through `CreateTextNode` from environment variables. Verify that
  `tests/test_gold_alerts.py::test_notification_stays_until_dismissed` and the existing
  failed-notification test both pass.
- [ ] 1.2 Run `python -m src.gold.alerts --test-notification`. It must return `(True, '')`,
  and the owner must confirm the toast stays on screen until they press its button.
- [x] 1.3 Update `results/gold/forward/RUNBOOK.md` (Large-move alerts) to say the
  notification is a reminder that stays until dismissed. Verify by reading the section back.

## 2. Page re-reads the alert (spec: An open page picks up a new alert)

- [x] 2.1 Check that `static/gold.html` re-reads `/api/gold` and the logger health every
  60 s with no visibility gate, and at once on `visibilitychange` to visible. Check that
  `renderAlert` hides the banner when the payload has no open alert. Verify with
  `tests/test_gold_alerts.py::test_page_rereads_alerts_while_open`.
- [x] 2.2 Drive the live page with `/browse`. Open `/gold` while `/api/gold` has no open
  alert, then wait through the next :03 logger pass that writes an alert, without
  reloading. Verify the banner appears within a minute of the pass. If no alert fires
  during the session, record that and verify the hide path instead: an open banner must
  disappear within a minute after its hour closes.
  - *2026-10-28, 07:27–07:31 Sofia.* No real alert was open, and `alerts.csv` is
    append-only, so no row was faked. Instead a page-side `fetch` wrapper added an alert
    to the real `/api/gold` answer, with nothing written to disk:
    - **Shown:** the banner appeared 59.9 s after the wrapper went in, with no reload, and
      the title read "⚠ Голямо движение ±0,20 %".
    - **Hidden:** the real `fetch` was restored, so the API again had no alert. The banner
      hid 49.2 s later and the title returned to plain.
    - **Tab comes back:** a `visibilitychange` event made the page request `/api/gold` and
      `/api/gold/health` at once.
    - **Real alert:** the 2026-10-27 18:00–19:00 banner was checked live in the session
      before.

## 3. Banner and tab title (specs: The banner names the hour and the size; The tab title carries the alert)

- [x] 3.1 Check the banner text:
  - percent with two decimals;
  - dollars only when the h1_lstm tile's `as_of_bar` equals the alert's;
  - the hour as `window_end − 1 h` to `window_end`, shifted to Sofia (+1 h);
  - "direction unknown", in BG and EN.

  Verify with `/browse` against a live or recorded open alert. For as-of bar 16:00 at
  close 4063.36 it must read ±0.20 %, ≈ ±7.94 $ and 18:00–19:00.
- [x] 3.2 Add a test to `tests/test_gold_alerts.py` that the BG and EN texts of every
  `bigmove*` key contain none of the `BANNED` words. Verify that it passes.
- [x] 3.3 Add a test that the page sets `document.title` with the `bigmove_tab` text while
  an alert is shown and restores the base title when it is hidden (a source check, like
  the existing page tests). Verify that it passes. Then confirm with `/browse` that
  `document.title` begins "⚠" while an alert is open.
- [x] 3.4 Check the banner contrast in both themes against the D5 figures in design.md.
  Use `/browse` screenshots at 390 px width in light and dark, with no horizontal scroll.
  Check that the pulse is off under `prefers-reduced-motion`.
  - *Measured in the page on 2026-10-28:*
    - light theme `rgb(154,107,0)` / white: 4.69:1;
    - dark theme `rgb(229,184,92)` / `rgb(27,20,0)`: 9.89:1;
    - at 390 px, `scrollWidth` 390 = `innerWidth` 390.
  - *Phone screenshots:* both themes at 390 px were taken on 2026-10-27.
  - *Reduced motion:* `/browse` refuses `Emulation.setEmulatedMedia`. CSSOM shows the
    parsed rule `(prefers-reduced-motion: reduce) => .ps-notice.bigmove { animation: none }`.
- [x] 3.5 Update the large-move alert line in `ARCHITECTURE_DOCS.md` §3.12. It must say the
  page re-reads every minute, the banner names the Sofia hour and the size in % and $,
  and the toast stays until dismissed. Verify by reading the paragraph back.

## 4. Integration

- [x] 4.1 Run `python -m pytest -q tests/test_gold_alerts.py tests/test_gold_view.py
  tests/test_gold_summary.py tests/test_gold_forward_logger.py tests/test_gold_gvz.py`.
  Verify that all pass and that nothing under `models/`, `gold_config.json` or the gold
  registries changed (`git status`).
- [x] 4.2 Run `openspec validate gold-alert-visibility --strict` and verify that it passes.
