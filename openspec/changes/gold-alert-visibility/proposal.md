# Proposal

## Why

On 2026-10-27, gold rose about $55 in the owner's afternoon session. The large-move
alert fired three times: 15:03, 16:03 and 17:03 Sofia. Before the first alert the
h1_lstm expected ±0.17 %; the next hour moved +0.56 %. The owner saw none of the three
alerts:

- `/gold` read `/api/gold` once, on load. A page opened at 13:00 never showed an alert
  written at 15:03.
- The banner was a thin gold line that looked like the standing research notice under it.
- The banner gave the end of the hour on the bar-label clock (Berlin), so "until 16:00"
  meant 17:00 in Sofia.
- The Windows toast was delivered (`notified=yes`) but left the screen after about five
  seconds.

The alert repeats the one gold forecast with skill: the size of the next hour's move. An
alert that is not seen is worth nothing.

The fix was made on 2026-10-27 and committed, unrequested, as `fe3520b`. This change records the
requirement and verifies the implementation against it.

## What Changes

- **An open `/gold` page re-reads the alert.**
  - It re-reads `/api/gold` and the logger health every minute, including in a
    background tab.
  - It re-reads at once when the tab becomes visible again.
- **The banner is hard to miss.**
  - It is a solid gold block with a warning sign and a short pulse when it appears.
  - It shows the expected size in percent and, when the alerted close is known, in dollars.
  - It names the alerted hour in Sofia time, e.g. "18:00–19:00 (Sofia time)".
  - The direction stays unknown in the text.
- **The browser tab title carries the alert**, e.g. "⚠ Large move ±0.20 %". It is
  visible while the owner looks at another tab or MT5.
- **The Windows notification stays on screen until it is dismissed.** It is a reminder
  toast with a sound and a dismiss button. It is still raised only once per alert.
- **Unchanged:**
  - the threshold (90th percentile, ±0.151 %), when an alert fires, the alert and
    settlement logs, and the leaderboard comparison;
  - the direction and probability tiles on `/gold`. The owner asked on 2026-10-27 to
    keep them.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `gold-forecasting/move-alerts`:
  - The banner requirement now covers a page that is already open.
  - The banner names the hour in Sofia time and the size in percent and dollars.
  - The tab title carries the alert.
  - The notification stays until dismissed.

## Impact

- **Code.**
  - `static/gold.html`: banner markup, style, Sofia-time helper, tab title, one-minute
    re-read and re-read when the tab is visible.
  - `src/gold/alerts.py`: the toast XML (reminder scenario, sound, dismiss action). It
    still takes no extra dependency.
- **Tests.** `tests/test_gold_alerts.py`: the bilingual keys, the re-read and the
  reminder toast.
- **Docs.** The large-move alert line in `ARCHITECTURE_DOCS.md` §3.12.
- **Not touched.**
  - `models/`, `gold_config.json`, `cfg2['study']` and the registries.
  - `/api/gold`: the payload is unchanged, and the dollar size is computed in the page
    from the h1_lstm tile.
- **Load.** Each open `/gold` tab makes two small loopback requests per minute.
