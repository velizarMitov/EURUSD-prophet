# Design

## Context

See proposal.md, Why. The alert pipeline itself works: the hourly logger writes
`alerts.csv` at :03, raises the toast and `/api/gold` exposes the open alert through
`alerts.open_alert`. What failed is delivery to the owner's eyes.

Constraints:

- `/gold` is a static page behind the app shell. It loads no external scripts and must work
  offline on loopback.
- `/api/gold` (`src/gold/summary.py`) is read-only. It reads files only and imports no model library.
- The toast goes through PowerShell's Windows Runtime API. The texts arrive in environment
  variables, so no quoting can break the command. The project takes no extra dependency
  for this.
- Bar labels are the Berlin wall clock stored as UTC. Sofia is Berlin + 1 h all year.

The implementation already exists in commit `fe3520b`. That commit was made unrequested,
together with logger rows. This design records the choices so that apply can verify them.

## Goals / Non-Goals

**Goals:**
- An alert written while `/gold` is open reaches the owner within a minute, on the page,
  in the tab title and as a notification that waits for them.

**Non-Goals:**
- No change to when an alert fires, its threshold or its logs.
- No alert outside `/gold`. Examples are the app-shell top bar, the home page `/` and a
  phone push.
- No sound in the browser. Autoplay rules make it unreliable, and the toast already
  carries a sound.

## Decisions

### D1. Poll `/api/gold` every minute, not push

- **Choice.** The page calls `load()` and `loadHealth()` every 60 s. It calls them again
  on `visibilitychange` when the tab becomes visible.
- **Background tabs too.** The tab title must update while the owner is elsewhere.
  Browsers throttle background timers to about once a minute, which is still enough.
- **Alternatives:**
  - Server-sent events or a WebSocket. They need a long-lived endpoint and a file watcher
    in `api.py`. The source changes once an hour, so this is wasted machinery.
  - Polling only at :04 each hour. A sleeping laptop or a skewed clock would miss the
    alert, and a fixed minute buys nothing over 60 s on loopback.

### D2. Compute the dollar size in the page

- **Choice.** Dollars = `vol_pct × as_of_close / 100`. The close comes from the h1_lstm
  tile, and only when its `as_of_bar` equals the alert's.
- **Why.** `/api/gold` stays unchanged, and `open_alert` keeps reading only `alerts.csv`.
- **Alternative:** add `as_of_close` to the alert block. That is a payload change for one
  number the page already has.
- **When the bars differ** (for example a newer forecast, or a gap), the page shows
  percent only. It never shows a dollar figure from another bar.

### D3. Name the hour in Sofia time, locally in the banner

- **Choice.** A small `sofiaHour` helper adds 1 h to the label time. The banner shows
  `window_end − 1 h` to `window_end`.
- **Scope.** The rest of the page keeps showing label times, and its footer says so.
  Converting every timestamp on the page is a wider change than this fix needs.

### D4. A reminder toast instead of a plain one

- **Choice.** A `ToastGeneric` XML with `scenario="reminder"`, the
  `Notification.Reminder` sound and a system `dismiss` action.
- **Why.** Windows keeps a reminder on screen until a button is pressed, and a reminder
  needs at least one button to behave that way. The text still enters through
  `CreateTextNode`, so it is never parsed as XML.
- **Alternatives:**
  - `duration="long"`: about 25 s, still easy to miss.
  - `scenario="alarm"`: a looping sound meant for alarm clocks. Too loud for an hourly
    research signal.
  - The browser Notification API: it needs a permission prompt and works only while a
    tab is open. The logger's toast works with no tab at all.

### D5. Banner colours from the shell's gold token

- **Choice.** The background is `--ps-gold`. The text uses a local `--bigmove-ink`:
  - light theme: white on `#9a6b00`, 4.7:1;
  - dark theme: `#1b1400` on `#e5b85c`, 9.9:1.
- **Why.** Dark text on the light-theme gold measured about 3.9:1, which fails AA.
- **Motion.** The pulse runs four times and is turned off under
  `prefers-reduced-motion`.

## Risks / Trade-offs

- [Focus assist or Do Not Disturb hides the toast] → Reminders can still be suppressed by
  Windows. The page banner and the tab title do not depend on Windows.
- [The owner keeps `/gold` closed] → Only the toast reaches them. This is acceptable, and
  an alert in the app shell is a possible follow-up (non-goal here).
- [Polling load] → Two small loopback requests per minute per open tab. `/api/gold` reads
  a few CSVs; this is negligible.
- [The reminder toast stays until dismissed] → Up to one toast per hour can pile up in a
  long volatile session. Each one is still a distinct alerted hour, which is what the
  spec asks for.
- [Unrequested commit] → `fe3520b` bundled this fix with forward-log rows. History is not
  rewritten. Apply verifies the committed code against the spec rather than trusting the
  commit message.

## Migration Plan

- **Deploy.** Nothing to deploy. The page picks up the HTML on its next load, and the
  logger uses the new toast on its next hourly pass.
- **Rollback.** Revert the three files. No data or config depends on them.
