# fx-session/market-data Specification

## Purpose

Defines the fx-session family's own H1 market data: MT5 bars with their spread back to
2010, the verified broker clock, the mid price, and the trading-day calendar that fixes one
entry bar and one exit bar per day on the owner's Sofia/Berlin clock.

## Requirements

### Requirement: Own H1 caches with spread
The family SHALL keep its own H1 caches for EURUSD, GBPUSD and AUDUSD, separate from every
other cache in the repo. Each row SHALL hold the bar label, bid open, high, low and close,
and the bar's spread in points. AUDUSD SHALL be used only as a dollar-factor input, never as
a scored pair.

#### Scenario: Cache content
- **WHEN** a cache file for a pair is read
- **THEN** it has the columns time, open, high, low, close and spread, and no other family's cache was written

### Requirement: History from 2012
A refresh SHALL cache history from 2012-01-01. If any pair's first bar is later than
2012-01-31, the refresh SHALL stop without writing and report the start dates, so the owner
decides.

The start is the owner's decision of 2026-11-21. The broker's GBPUSD H1 bars from December
2010 to November 2011 sit one hour off the clock of the other pairs.

#### Scenario: Shallow broker history
- **WHEN** MT5 returns GBPUSD bars only from 2012-03-01
- **THEN** no cache is written and the report names each pair's earliest bar

### Requirement: Declared broker gaps
A refresh SHALL refuse any interior hole or thin month, except months the config declares
as known broker gaps for that pair. Declared months SHALL be cached as MT5 returns them,
never filled from any other source. They SHALL be left out of the clock check.

#### Scenario: GBPUSD March 2014
- **WHEN** GBPUSD's March 2014 holds one bar per day and the month is declared
- **THEN** the refresh succeeds, the month is cached unchanged, and its days fail the session eligibility rule

#### Scenario: Undeclared hole
- **WHEN** a pair has a thin month that the config does not declare
- **THEN** the refresh is refused and nothing is written for any pair

### Requirement: Verified clock weeks
A day SHALL be eligible only if, for every loaded pair, its week's weekend open falls on the
Berlin label hour that the project's clock rule predicts for that Sunday. Failing weeks SHALL
be excluded and counted as `unverified clock week`. A week opening at Monday 00:00 fails, as
does a DST-mismatch week opening at 23:00.

#### Scenario: Pre-2017 mismatch week
- **WHEN** the weekly open of a March DST-mismatch week in 2014 is labelled 23:00
- **THEN** every day of that week is excluded as an unverified clock week

### Requirement: Closed bars only and append-only refresh
Bars SHALL be read from MT5 position 1, so the bar still forming is never cached. A refresh
SHALL append only bars newer than the cache's last row. It SHALL count overlapping rows
whose values differ, and record every refresh in a refresh log.

#### Scenario: Forming bar
- **WHEN** a refresh runs mid-hour
- **THEN** the newest cached bar is the last completed hour

### Requirement: Broker clock verification
Every refresh SHALL check, week by week, that the weekly-open labels match the Berlin wall
clock, including the March and October/November DST-mismatch weeks. A pair whose open-label
match rate falls below the threshold used by the project's existing clock check SHALL be
refused.

#### Scenario: Clock drift
- **WHEN** a pair's weekly opens stop matching the Berlin open hour
- **THEN** that refresh is refused and nothing is appended for any pair

### Requirement: Mid price
The mid price of a bar SHALL be its bid price plus half its spread converted to price units
with the symbol's point size. The mid SHALL be computed for the open and the close.

#### Scenario: Mid from spread
- **WHEN** a EURUSD bar has bid close 1.08000, a spread of 6 points and a point of 0.00001
- **THEN** its mid close is 1.08003

### Requirement: Fixed Sofia/Berlin session clock
Each trading day SHALL use the Berlin bar labels: the entry bar is the one labelled 14:00
(close at 16:00 Sofia), and the exit bar is the one labelled 21:00 (close at 23:00 Sofia).
These label hours SHALL NOT move in the weeks when US and European daylight saving time
differ.

#### Scenario: DST-mismatch week
- **WHEN** a day falls in the March week when New York is already on summer time and Berlin is not
- **THEN** the entry is still the 14:00 Berlin bar and the exit is still the 21:00 Berlin bar

### Requirement: Eligible trading days
A trading day SHALL be eligible only if it is Monday to Friday and every H1 bar labelled
14:00 through 21:00 exists for both scored pairs. A day missing any of those bars SHALL be
excluded and counted, with its reason.

#### Scenario: Early close
- **WHEN** 24 December has no bars after 19:00 Berlin
- **THEN** that day is excluded and counted as "incomplete session"
