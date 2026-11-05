# Spec Delta

## Purpose

Supplies the gold analytics with honest, correctly timed data. Covers read-only MT5
prices and bid–ask ticks, a per-day spread cache, session and news-hour definitions on
the right clocks, and a rule that volume is never shown or approximated.

## ADDED Requirements

### Requirement: MT5 is read only
The analytics SHALL read gold from MT5 only through the history and tick functions:
- the live tick;
- bars at M1, M15, H1 and D1;
- ticks in a range.

It SHALL never send, modify or query orders or positions. When MT5 is unreachable, it
SHALL fall back to the newest cached data and mark the result stale with its time.

#### Scenario: Terminal closed
- **WHEN** a request arrives and MT5 cannot be initialised
- **THEN** the response uses the cached data, sets `stale: true` and gives the time of the newest bar used

#### Scenario: No trading calls
- **WHEN** the package's source is scanned
- **THEN** it contains no call to `order_send`, `order_check`, `positions_get` or `orders_get`

### Requirement: Labels follow the repo clock and are shown in Sofia time
Bar labels SHALL be treated as Europe/Berlin wall time, as every cache in the repo
stores them. Every time shown to the owner SHALL be Europe/Sofia, which is Berlin + 1 h
all year. Converting a label to an instant SHALL use the Berlin zone with daylight
saving.

#### Scenario: Display
- **WHEN** a bar labelled 14:00 is shown
- **THEN** it is shown as 15:00 Sofia

### Requirement: Sessions are defined in their own time zones
Sessions SHALL be configurable, each with a start, an end and an IANA time zone. The
defaults are:
- Asia: 09:00–15:00 Asia/Tokyo;
- London: 08:00–16:30 Europe/London;
- New York: 08:20–13:30 America/New_York;
- the London–New York overlap: the intersection of London and New York on each day.

For each trading day, a session's interval SHALL be computed in its own zone and
converted, so daylight-saving changes in Europe and the US are each handled on their own
dates.

#### Scenario: US/EU daylight-saving mismatch
- **WHEN** the day is in the March weeks when the US is on summer time and Europe is not
- **THEN** New York 08:20 maps to 14:20 Sofia instead of the usual 15:20

#### Scenario: Custom session
- **WHEN** the config defines a session 07:00–10:00 Europe/London
- **THEN** that session is computed and shown like the defaults

### Requirement: The spread comes from ticks, aggregated per day
A refresh command SHALL pull MT5 ticks one trading day at a time. For each day it SHALL
store, per 5-minute bucket of the Berlin label clock:
- the median and 90th-percentile spread (ask − bid) in USD;
- the tick count, used only to judge coverage and never shown as volume.

Days already cached SHALL be skipped. Page requests SHALL read only this cache and SHALL
never pull more than the current day's ticks.

#### Scenario: Incremental refresh
- **WHEN** the refresh runs and 249 of the last 250 trading days are cached
- **THEN** it pulls ticks for the one missing day only

#### Scenario: Request
- **WHEN** the time-of-day section asks for the spread over 250 days
- **THEN** it reads the cache and pulls no historical ticks

### Requirement: News hours come from release calendars
NFP and CPI release dates SHALL come from the FRED release-dates service, cached on
disk. Each release is at 08:30 America/New_York. FOMC statement dates SHALL come from
`results/fomc_dates.csv`, at 14:00 America/New_York. When FRED is unreachable, the cache
SHALL be used, and the section SHALL say how old it is.

#### Scenario: CPI in winter and summer
- **WHEN** a CPI release falls in January and another in July
- **THEN** they map to 15:30 and 15:30 Sofia, and in the March and October mismatch weeks to 14:30 Sofia

### Requirement: No volume
The analytics SHALL never show, chart, rank or weight by volume. That means no VWAP,
volume profile, POC, VAH, VAL, HVN, LVN, PWP or volume metric. MT5 tick volume and real
volume SHALL not be read into any analytic. Tick counts used for spread coverage SHALL
never be labelled or shown as volume.

#### Scenario: Payload
- **WHEN** any analytics endpoint responds
- **THEN** no field name or label contains "volume", "vwap", "poc" or "profile"
