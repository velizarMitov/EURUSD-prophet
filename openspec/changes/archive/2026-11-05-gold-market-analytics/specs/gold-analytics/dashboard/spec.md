# Spec Delta

## Purpose

Gives the owner one read-only, bilingual page, `/gold-analytics`, showing how gold
behaves: price, sessions, time of day, spread, news hours, volatility, history and
time-based benchmarks. It describes the past and never suggests a trade.

## ADDED Requirements

### Requirement: The page is read-only and descriptive
`/gold-analytics` and its endpoints SHALL:
- run no model;
- write nothing except the analytics' own cache;
- place, size or simulate no order.

Every section SHALL carry the label "descriptive: how gold behaved, not a forecast or a
recommendation" in Bulgarian and English. No text SHALL say buy, sell, купи or продай.

#### Scenario: Wording
- **WHEN** the page and every endpoint's payload are rendered in both languages
- **THEN** none contains buy, sell, купи or продай, and every section has the descriptive label

### Requirement: Overview shows the live state
The overview SHALL show:
- the live XAU/USD bid and ask, and the daily change in $ and %;
- the daily high, low and open, and the previous close;
- the current session, or "between sessions";
- historical volatility (20-day, annualised) and intraday volatility (today's realised
  so far).

The day SHALL be the Berlin label day. With MT5 closed, the overview SHALL show the last
known values marked stale.

#### Scenario: Between sessions
- **WHEN** the time is outside every configured session
- **THEN** the overview names the next session and when it starts, in Sofia time

### Requirement: Price chart in eight timeframes
The chart SHALL offer 1m, 5m, 15m, 30m, 1h, 4h, D1 and W1:
- 1m comes from M1;
- 5m and 30m are resampled from M1, or from M15 for 30m beyond the M1 history;
- 15m comes from M15, 1h from H1 and D1 from D1;
- 4h is resampled from H1, aligned to Sofia midnight;
- W1 is resampled from D1, with weeks starting Monday.

The chart SHALL state the earliest bar available for the chosen timeframe.

#### Scenario: 4h bar
- **WHEN** 4h is chosen
- **THEN** each bar covers 00:00–04:00, 04:00–08:00 Sofia and so on, built from H1 open, max high, min low and last close

#### Scenario: Short history
- **WHEN** 1m is chosen and MT5 keeps 70 days of M1
- **THEN** the chart shows those 70 days and states the first available bar

### Requirement: Session statistics
For each session and a chosen period, the section SHALL give:
- per session: open, close, high, low, range in $ and %, % return, and realised
  volatility;
- across the period: directional bias (the share of sessions closing above their open,
  with a Wilson 95 % interval), breakout frequency (the share that trade beyond the
  previous same session's high or low) and trend persistence (the share whose return has
  the sign of the previous same session's).

Sessions SHALL be compared side by side, with an "overnight against day" summary.

#### Scenario: Directional bias interval
- **WHEN** the London session closed above its open in 33 of 60 sessions
- **THEN** the section shows 55 % with its Wilson 95 % interval and does not call it an edge

### Requirement: Time-of-day statistics
For a chosen interval (1, 5, 15, 30 or 60 minutes) and period, the section SHALL show, by
Sofia time of day:
- mean return and mean absolute move;
- mean range and acceleration (the change in absolute move from the previous interval);
- momentum (the correlation of the first and last hour of the window);
- the distribution of the times of the day's high and low;
- the median and 90th-percentile spread from the tick cache.

The owner's window 15:30–23:00 Sofia SHALL be shaded.

#### Scenario: Spread by hour
- **WHEN** 60 minutes and 250 days are chosen
- **THEN** each hour shows its median and 90th-percentile spread in $ and as % of price

#### Scenario: Interval beyond history
- **WHEN** 1 minute is chosen for 250 days but M1 covers 70
- **THEN** the section uses the 70 days and says so

### Requirement: News-hour statistics
For NFP, CPI and FOMC in a chosen period, the section SHALL show, for the hours around
each release:
- the mean absolute move and the median spread;
- the same hours on non-release days, for comparison;
- the count of releases.

It SHALL state that the direction depends on the surprise against forecasts, which this
page does not have.

#### Scenario: NFP hour
- **WHEN** NFP is selected for the last 24 months
- **THEN** the hour from 15:30 Sofia shows its mean absolute move on the 24 release days against the other Fridays' same hour

### Requirement: Volatility and regime
The section SHALL show:
- ATR(14) on D1;
- realised volatility over 5, 20 and 60 days, annualised;
- the regime as low, normal or high: today's 20-day realised volatility in the bottom,
  middle or top third of its trailing 250-day distribution;
- today's range against its 20-day average.

#### Scenario: Regime boundaries
- **WHEN** today's 20-day realised volatility is above the 67th percentile of the last 250 days
- **THEN** the regime reads high

### Requirement: Historical periods
For 5, 20, 60, 120 or 250 trading days, or a custom range, the section SHALL give:
- the average daily return, average daily range and ATR;
- realised volatility and maximum drawdown;
- trend persistence;
- a return histogram;
- session statistics;
- the comparison of the current session, volatility and range with the period average.

#### Scenario: Custom range
- **WHEN** a range of 2025-01-01 to 2025-06-30 is requested
- **THEN** every statistic uses only trading days in that range and states the day count

### Requirement: Time benchmarks without volume
The section SHALL compute:
- TWAP over a chosen interval: the mean of the M1 (or M15) closes in it;
- the current price against TWAP, in $ and %;
- for a chosen start time, the arrival price (the close of the bar at that time) and the
  move after it, in $ and %, at 15, 30, 60 and 120 minutes.

It SHALL be described as a hypothetical benchmark, not a trade.

#### Scenario: Arrival
- **WHEN** a start of 16:00 Sofia yesterday is chosen
- **THEN** the arrival price is the close of the bar ending at 16:00 Sofia, with the move at the four horizons

### Requirement: Endpoints validate their parameters
Each section SHALL have one GET endpoint under `/api/gold-analytics/`. An unknown
section, timeframe, interval, session, period or a malformed date SHALL answer 400, with
the allowed values. Responses SHALL carry `stale` and `as_of`.

#### Scenario: Bad timeframe
- **WHEN** `/api/gold-analytics/chart?timeframe=2h` is requested
- **THEN** the answer is 400 and lists 1m, 5m, 15m, 30m, 1h, 4h, D1 and W1

### Requirement: The page works offline and on a phone
The page SHALL load no external script, style or font, and SHALL draw its charts as
inline SVG. At 360 px wide it SHALL have no horizontal page scroll. Every control SHALL
be operable by keyboard. Its chart colours SHALL pass the dataviz palette validator in
both themes.

#### Scenario: Phone width
- **WHEN** the page is rendered at 360 px
- **THEN** the sections stack in one column and the page does not scroll horizontally
