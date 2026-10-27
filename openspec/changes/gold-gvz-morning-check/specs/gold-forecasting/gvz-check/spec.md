# Spec Delta

## Purpose

Shows the owner the options market's own expectation of gold volatility (GVZ) before the
13:00 Sofia session. It is a market reading beside the model's forecast, never a model
claim, and it logs the two daily, unscored, so the owner can compare them over time.

## ADDED Requirements

### Requirement: GVZ history comes from FRED
The system SHALL acquire the daily GVZ close from FRED series `GVZCLS`, through the
project's shared FRED fallback chain (API, then public CSV, then the on-disk cache), from
the configured history start. It SHALL store the values in a gold-owned cache, with one
value per US session date.

#### Scenario: FRED reachable
- **WHEN** the refresh runs and FRED answers
- **THEN** every published `GVZCLS` value from the history start is in the cache, dated by its US session

#### Scenario: FRED unreachable with a cache
- **WHEN** FRED cannot be reached and the cache exists
- **THEN** the cached values are used unchanged and the failure is recorded

### Requirement: Unpublished days are topped up from yfinance
For US session dates after the newest FRED value, the system SHALL take the `^GVZ` daily
close from yfinance. A yfinance value SHALL never replace a FRED value for the same date.
Once FRED publishes a date, the FRED value SHALL be served for it.

#### Scenario: FRED two sessions behind
- **WHEN** FRED's newest value is Friday 2026-10-23 and yfinance has Monday 2026-10-26 at 23.55
- **THEN** the served value is 23.55 dated 2026-10-26, with source yfinance

#### Scenario: FRED catches up
- **WHEN** FRED later publishes 2026-10-26 with a different value
- **THEN** the served value for 2026-10-26 is FRED's, and its source is FRED

### Requirement: Only completed US sessions are stored
A GVZ value SHALL be stored only if its date is strictly before the current date in
America/New_York. An intraday value for the current US session SHALL be ignored.

#### Scenario: Intraday bar
- **WHEN** the refresh runs at 17:00 Sofia on 2026-10-27 and yfinance returns a 2026-10-27 bar
- **THEN** that bar is not stored, and the newest stored date stays 2026-10-26

### Requirement: The gold logger refreshes GVZ without risk to the pass
Each gold logger pass SHALL refresh GVZ only while the newest stored date is older than
the last completed US session. A GVZ failure SHALL be appended to the gold failures log
and named in the pass's health note. It SHALL never stop the pass or change any
round-one, round-two, range, alert or shadow row.

#### Scenario: Already fresh
- **WHEN** a pass runs at 14:03 Sofia on Tuesday and the newest stored value is Monday's
- **THEN** no GVZ source is contacted

#### Scenario: Both sources fail
- **WHEN** FRED and yfinance both raise
- **THEN** a failure row names GVZ, the health note says the GVZ refresh failed, and the pass's forecasts and settlements are written as usual

### Requirement: The API serves GVZ from files only
`GET /api/gold` SHALL include a `gvz` block built only from stored files. The block SHALL
carry the value, its date, its source, the percentile, the implied move in percent and in
dollars, the gold close used, and the stale flag. Building it SHALL contact no network
source and write no file.

#### Scenario: Read-only build
- **WHEN** `/api/gold` is requested while the network is unavailable
- **THEN** the `gvz` block is built from the caches, and no file changes

#### Scenario: No GVZ stored yet
- **WHEN** no GVZ value has ever been stored
- **THEN** the `gvz` block reports no data, and the rest of the payload is unchanged

### Requirement: The percentile is against the stored history
The percentile SHALL be the share of stored values that are strictly below the served
value, times 100. The values counted SHALL run from the configured history start (2013-01-01)
up to and including the served value's date.

#### Scenario: High reading
- **WHEN** the served value is greater than 91 % of the stored values since 2013-01-01
- **THEN** the percentile reads 91

### Requirement: The implied daily move is arithmetic only
The implied daily move SHALL be the served GVZ divided by √252, in percent, as one
standard deviation over one full trading day. Its dollar value SHALL be that percent of
the newest gold H1 close the logger recorded. Neither number SHALL come from a fitted
model.

#### Scenario: Conversion
- **WHEN** GVZ is 25.2 and the newest gold close is 4,000.00
- **THEN** the implied move is ±1.59 % and ±63.50 $

#### Scenario: No gold close yet
- **WHEN** no gold H1 close has been recorded
- **THEN** the percent is shown and the dollar value is absent

### Requirement: A stale GVZ value is flagged
The `gvz` block SHALL mark its value stale when the value's date is more than 5 calendar
days before the current America/New_York date.

#### Scenario: Normal weekend
- **WHEN** it is Monday 2026-10-26 and the newest value is from Friday 2026-10-23
- **THEN** the value is not stale

#### Scenario: Sources silent for a week
- **WHEN** it is Thursday 2026-10-29 and the newest value is from Friday 2026-10-23
- **THEN** the value is marked stale

### Requirement: The morning-check card shows size only
`/gold` SHALL show a morning-check card at the top of the "Now" tab, with:
- the GVZ value, its date and its source;
- the percentile;
- the implied daily move in percent and in dollars;
- the current h1_lstm next-hour move, where one exists.

The card SHALL say the direction is unknown, and that GVZ is the options market's 30-day
expectation, not a model forecast.

#### Scenario: Normal day
- **WHEN** the owner opens `/gold` at 13:00 Sofia with a fresh value
- **THEN** the card shows the value with its date, the percentile, the implied move, and the h1_lstm move

#### Scenario: Stale or missing value
- **WHEN** the value is stale, or there is none
- **THEN** the card says so with the date of the last value, or says that no value is stored yet, and the rest of the tab renders as before

### Requirement: The card is bilingual and free of trade and cost wording
The card's text SHALL follow the page's BG/EN switch. It SHALL contain no direction, no
buy, sell, entry, stop or target wording, and none of the page's forbidden cost words.

#### Scenario: Language switch
- **WHEN** the owner switches the shell to EN
- **THEN** every card label switches to English without a reload

#### Scenario: Forbidden words
- **WHEN** the rendered `/gold` page is scanned
- **THEN** it contains none of `spread`, `pips`, `p&l`, `profit`, `pnl` or `sharpe`

### Requirement: One observation row per weekday after 13:00 Sofia
On each Monday to Friday, the first gold logger pass at or after 13:00 Europe/Sofia SHALL
append one row to `results/gold/gvz_log.csv`. The row SHALL hold GVZ as served at that
moment, its percentile and implied move, the gold close, and the newest h1_lstm forecast,
each with its date. A field unknown at that moment SHALL stay blank and never be filled
later.

#### Scenario: First afternoon pass
- **WHEN** passes run at 12:03, 13:03 and 14:03 Sofia on a Tuesday
- **THEN** exactly one row is written for that date, at the 13:03 pass

#### Scenario: Missing forecast
- **WHEN** MT5 was closed and no h1_lstm forecast exists at 13:03
- **THEN** the row is written with the forecast fields blank, and they stay blank

### Requirement: Each observation is settled with the realised afternoon range
Once the gold H1 history holds a bar labelled after 21:00 on the observation date, the
system SHALL append one row for that date to `results/gold/gvz_settlements.csv`. Logged
rows are never rewritten. The range is the high minus the low of the H1 bars labelled
12:00–21:00 (13:00–23:00 Sofia), as a percent of the first bar's open, with the number of
bars used. Fewer than 9 bars SHALL leave it blank.

#### Scenario: Full session
- **WHEN** all 10 bars from 13:00 to 23:00 Sofia exist and a later bar has arrived
- **THEN** one settlement row holds the range and 10 bars, and the observation row is unchanged

#### Scenario: Thin session
- **WHEN** only 7 of the bars exist once a later bar has arrived
- **THEN** the settlement row's range is blank and its bar count reads 7

#### Scenario: Session not over
- **WHEN** a pass runs at 20:03 Sofia
- **THEN** that day is not settled

### Requirement: The observation log is never scored
The GVZ observation and settlement logs SHALL NOT be:
- registered;
- written to any gold hypothesis registry;
- given an alpha or a verdict;
- shown on the leaderboard;
- read by the forward scoring report.

#### Scenario: Registries unchanged
- **WHEN** a month of observation rows has been logged
- **THEN** every gold registry, registration and manifest is byte-identical to before

### Requirement: GVZ settings stay outside the pinned study settings
The GVZ settings SHALL live in their own top-level `gvz` block in
`src/gold/gold_config_v2.json`. Adding or editing them SHALL leave the round-two study
digest and `src/gold/gold_config.json` unchanged.

#### Scenario: Digest check
- **WHEN** the `gvz` block is added
- **THEN** the study digest recorded in `PRE_REGISTRATION_V2.json` still matches the configuration, and `gold_config.json` is unchanged
