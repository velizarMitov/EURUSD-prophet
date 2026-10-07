# Spec Delta

## Purpose

Gives the owner, and later a non-technical buyer, a single plain-language screen.
It shows what the models currently say, how big a move is expected, and how far
the forecasts have been reliable so far. It never overstates the evidence.

## ADDED Requirements

### Requirement: Home page is the application's front door
`GET /` SHALL serve the plain-language home page. The previous research dashboard
SHALL remain reachable, unchanged, at `GET /advanced`. The home page SHALL link
to it as the advanced or technical view.

#### Scenario: Root serves the home page
- **WHEN** a browser requests `/`
- **THEN** the response is the home page with the three tiles and the honesty notice, not the research dashboard

#### Scenario: Research dashboard still reachable
- **WHEN** a browser requests `/advanced`
- **THEN** the response is byte-for-byte the previous dashboard file, and all of its buttons keep working

#### Scenario: Other pages unaffected
- **WHEN** `/history`, `/paper-trading`, `/forecasts`, `/h1-direction` or `/kronos-volatility` is requested
- **THEN** each responds exactly as before this change

### Requirement: Read-only home summary endpoint
`GET /api/home` SHALL return, as JSON, everything the home page renders. It SHALL
build this only from existing on-disk logs and the existing paper-trading and
forward-evaluation readers. It SHALL NOT run any model and SHALL NOT write any
file.

#### Scenario: No side effects
- **WHEN** `/api/home` is called any number of times
- **THEN** no file under `results/` or `models/` is created or modified, and no model inference runs

#### Scenario: One source fails, the rest still render
- **WHEN** one source (for example the forward-evaluation logs or the realised-price fetch behind the hit rate) raises or is missing
- **THEN** the response still returns 200, that section carries `available: false` with a reason, and the other sections are populated

### Requirement: Tomorrow tile in plain words
The Tomorrow tile SHALL show the price-only (baseline) daily consensus as words,
not as a raw probability or return. Below `CONFIDENCE_THRESHOLD` (0.52) it SHALL
read "no clear signal". At or above it, it SHALL read "slightly up" or "slightly
down". Words implying strength ("strong", "sure", "buy", "sell") SHALL never
appear.

#### Scenario: Near-chance forecast
- **WHEN** the latest baseline consensus confidence is 0.515
- **THEN** the tile reads "no clear signal" (BG: "няма ясен сигнал") with a neutral colour and no arrow

#### Scenario: Directional forecast
- **WHEN** the latest baseline consensus is UP with confidence 0.53
- **THEN** the tile reads "slightly up" (BG: "леко нагоре") with an up colour, and shows the forecast date

#### Scenario: Variants disagree
- **WHEN** the latest logged row has `variant_agreement` false
- **THEN** the tile adds the note "the two model versions disagree today" without changing the headline wording rule

### Requirement: Freshness is always visible
Every tile SHALL show the time its data refers to, in the owner's clock
(Europe/Sofia). A daily forecast whose forecast date is already in the past SHALL
be marked outdated. The page SHALL offer a refresh action that calls the existing
`POST /api/predict` and then re-renders.

#### Scenario: Stale daily forecast
- **WHEN** the newest logged daily forecast is for a date before today's trading session
- **THEN** the Tomorrow tile is greyed, labelled "outdated", and the refresh button is highlighted

#### Scenario: Refresh
- **WHEN** the owner presses the refresh button
- **THEN** `POST /api/predict` is called once, a loading state is shown, and the tiles re-render from `/api/home` afterwards; on failure a plain-language error replaces the loading state

### Requirement: Session tile from the forward logs
The Today's-session tile SHALL show the latest forecasts of the registered
session cells (15 minutes to 4 hours) from the forward-evaluation logs. Each
forecast SHALL be shown as a direction word and the clock window it covers.
Outside the 15:30–23:00 Europe/Sofia session it SHALL say the session is closed
and when it next opens.

#### Scenario: Session open
- **WHEN** the session is open and fresh forecasts exist
- **THEN** each registered cell shows "up"/"down", its window (for example "16:00–16:30"), and a link to `/forecasts` for detail

#### Scenario: Session closed
- **WHEN** the current Sofia time is outside 15:30–23:00 or it is a weekend
- **THEN** the tile shows "Session closed — opens Mon 15:30" (or the next weekday) and no stale direction

### Requirement: Expected-movement tile
The Expected-movement tile SHALL show the latest next-day volatility forecast as
"± X %". It SHALL state in one plain sentence that this is the size of the move,
not its direction.

#### Scenario: Volatility available
- **WHEN** the latest logged row has a volatility forecast of 0.42
- **THEN** the tile reads "± 0.42 %" and "how much EUR/USD may move tomorrow, not which way"

#### Scenario: Volatility missing
- **WHEN** no volatility forecast is logged
- **THEN** the tile reads "not available" and does not show 0

### Requirement: Reliability strip compares against a coin flip
The page SHALL show the forward hit rate of the daily baseline forecasts as
"X of N right (Y %)", drawn against a 50 % coin-flip reference line. It SHALL be
in exactly one state: "too early to tell" while N < 30; otherwise "within
coin-flip range" or "above coin-flip so far — not yet proven", by the 95 % Wilson
lower bound vs 50 %. It SHALL never say proven, guaranteed or profitable.

#### Scenario: Small sample
- **WHEN** 12 daily forecasts have settled
- **THEN** the strip shows "7 of 12 right" and the state "too early to tell", with no green styling

#### Scenario: Indistinguishable from a coin
- **WHEN** 60 have settled with 33 right
- **THEN** the state reads "within coin-flip range"

#### Scenario: Above the line
- **WHEN** the 95 % Wilson lower bound on the settled hit rate exceeds 50 %
- **THEN** the state reads "above coin-flip so far — not yet proven" and links to the paper-trading ledgers

### Requirement: No cost or profit figures on the home page
The home page and `/api/home` SHALL NOT show spread, breakeven, pips, net P&L,
profit or return-on-capital figures. Those stay on the advanced pages, consistent
with the owner's 2026-10-02 decision for the forecast view.

#### Scenario: Guard against cost words
- **WHEN** the rendered home page text, in either language, is scanned for cost and profit terms
- **THEN** none are present

### Requirement: Permanent honesty notice
The page SHALL always show a notice saying: no proven edge yet, not financial
advice, simulated tracking only. It SHALL be visible without scrolling on a phone
and on a desktop, and SHALL NOT be dismissible.

#### Scenario: Notice present in both languages
- **WHEN** the page is shown in Bulgarian or in English
- **THEN** the notice is present in that language above the fold, with no close control

### Requirement: Bulgarian and English with a remembered switch
Every user-visible label on the home page SHALL exist in Bulgarian and English. A
visible BG/EN switch SHALL change the language without reloading. The choice
SHALL be remembered in the browser. The default SHALL be Bulgarian, and a missing
or blocked memory SHALL fall back to Bulgarian without error.

#### Scenario: Switch language
- **WHEN** the owner clicks "EN"
- **THEN** every label changes to English immediately, and a later visit opens in English

#### Scenario: Complete dictionary
- **WHEN** the label dictionary is checked
- **THEN** every key has a non-empty Bulgarian and English value

#### Scenario: Storage blocked
- **WHEN** browser storage throws (private mode)
- **THEN** the page renders in Bulgarian and the switch still works for the visit

### Requirement: Readable on phone and desktop
The home page SHALL work from 360 px wide upwards with no horizontal scroll. It
SHALL respect the system light or dark preference. It SHALL load no external
scripts, so it works offline on loopback.

#### Scenario: Phone width
- **WHEN** the page is rendered 360 px wide
- **THEN** the tiles stack in one column and nothing overflows horizontally
