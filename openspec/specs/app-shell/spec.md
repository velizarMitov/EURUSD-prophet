# app-shell Specification

## Purpose

Gives every page of the application one look, one navigation bar and one language and
theme choice, so the owner can move around the app and read it comfortably. Pages whose
source is pinned get it without being edited.

## Requirements

### Requirement: Every HTML page carries the shell
Every HTML page the application serves SHALL show the shared top navigation and use the
shared design tokens. JSON responses, CSV responses and static assets SHALL NOT be altered.

#### Scenario: Each page route
- **WHEN** `/`, `/gold`, `/advanced`, `/history`, `/paper-trading`, `/h1-direction`, `/kronos-volatility`, `/kronos-direction` or `/forecasts` is requested
- **THEN** the HTML contains the shell stylesheet, the shell script and the navigation landmark, each exactly once

#### Scenario: API response
- **WHEN** `/api/gold`, `/api/home` or `/api/provenance` is requested
- **THEN** the body is byte-identical to what the route returned before the shell existed

### Requirement: Pinned page sources stay byte-identical
The shell SHALL reach pages rendered by pinned modules without those modules changing.
`src/tracking.py`, `src/paper_trading.py`, `src/inference.py` and `static/index.html` SHALL
stay byte-identical, and `api.py` SHALL only gain lines.

#### Scenario: Diff after the change
- **WHEN** the working tree is diffed against the commit before this change
- **THEN** the four files show no difference and `api.py` shows zero removed lines

### Requirement: The research console differs only by the shell
`/advanced` SHALL serve `static/index.html` unchanged apart from the injected shell, and
`static/index.html` SHALL stay byte-identical. Its content, scripts and behaviour SHALL
not change.

#### Scenario: Shell removed
- **WHEN** the shell's injected segments are removed from the `/advanced` response
- **THEN** the result equals `static/index.html` byte for byte

### Requirement: Navigation shows where the owner is
The navigation SHALL link to Home, Gold, Forecasts, History, Paper trading and the
research console. It SHALL mark the current page as current, and the less-used views
(H1 direction, Kronos volatility) SHALL be under one "More" menu. A retired view SHALL
not be listed.

#### Scenario: On the gold page
- **WHEN** `/gold` is open
- **THEN** the Gold item is marked `aria-current="page"` and every other item is a plain link

#### Scenario: Retired view
- **WHEN** the navigation is rendered
- **THEN** it has no link to `/kronos-direction`

### Requirement: Navigation works on a phone and from the keyboard
At 360 px wide the navigation SHALL collapse into a menu button with no horizontal page
scroll. Every navigation control SHALL be reachable and operable by keyboard, with a
visible focus outline.

#### Scenario: Phone width
- **WHEN** any page is rendered 360 px wide
- **THEN** the navigation shows a menu button and the page does not scroll sideways

#### Scenario: Keyboard
- **WHEN** the owner tabs through the navigation and presses Enter on the menu button
- **THEN** the menu opens, focus moves into it, and Escape closes it and returns focus to the button

### Requirement: One remembered language choice
The shell SHALL offer a BG/EN switch that applies across pages and is remembered in the
browser. The default SHALL be Bulgarian, and an earlier choice made on the home page
SHALL be honoured. Pages with their own dictionary SHALL follow the switch without a reload.

#### Scenario: Choice carries over
- **WHEN** the owner picks EN on `/gold` and then opens `/`
- **THEN** the home page opens in English

#### Scenario: Earlier home-page choice
- **WHEN** the browser holds only the home page's earlier stored language "en"
- **THEN** the shell opens in English

#### Scenario: Storage blocked
- **WHEN** browser storage throws
- **THEN** pages render in Bulgarian and the switch still works for the visit

### Requirement: One remembered theme choice
The shell SHALL offer System, Light and Dark. System SHALL follow the operating-system
preference. The choice SHALL be remembered in the browser and applied before the first
paint, so a page never flashes the other theme.

#### Scenario: Dark chosen
- **WHEN** the owner picks Dark and opens another page
- **THEN** that page renders dark from its first paint

### Requirement: The shell works offline
The shell and every page built on it SHALL load no script, stylesheet or font from
outside the application.

#### Scenario: External references
- **WHEN** the shell files, `static/home.html` and `static/gold.html` are searched for `http://` or `https://` in `src`, `href` or `@import`
- **THEN** none is found

### Requirement: Redesigned pages keep their contracts
The redesigned home and gold pages SHALL keep every behaviour their existing
requirements describe:
- read-only, and the endpoints they read;
- the bilingual dictionary;
- the honesty notices and verdict labels;
- no spread, pips or P&L;
- the phone-width layout.

#### Scenario: Existing page tests
- **WHEN** the home-page and gold-view tests run against the redesigned pages
- **THEN** they pass unchanged, except the `/advanced` byte-for-byte test, which is rewritten to the "research console differs only by the shell" requirement
