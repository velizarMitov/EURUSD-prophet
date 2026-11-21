# Spec Delta

## Purpose

Defines the fx-rates family's data: the official daily 2-year and 10-year yield caches
from FRED, the Bundesbank and the Bank of England. It also defines the FX D1 bars and the
causal rule that lets each day see only yields from the previous FX trading day.

## ADDED Requirements

### Requirement: Official daily yield sources
The family SHALL cache four daily series, each from its official source only:
- US 2Y: FRED `DGS2`;
- US 10Y: FRED `DGS10`;
- DE 2Y: the Bundesbank Svensson term structure at a residual maturity of 2.0 years;
- UK 2Y: the Bank of England nominal spot curve at 2.0 years.

No other vendor SHALL be used. Requests SHALL carry a generic client name and no personal
data.

#### Scenario: Source unavailable
- **WHEN** the Bundesbank request fails during a refresh
- **THEN** the DE 2Y cache is left unchanged, the failure is logged, and no other source is tried for that series

### Requirement: Bank of England 2-year point
The UK 2Y value SHALL be the 2.0-year column of the "spot curve" sheet. It SHALL be read
from the archive of nominal daily files plus the current-month file. Where both cover a
date, the current-month value SHALL be used.

#### Scenario: Archive lag
- **WHEN** the archive ends on the last business day of the previous month
- **THEN** the current month's days come from the current-month file

### Requirement: Append-only yield caches
Each yield cache SHALL keep the columns date and value. A refresh SHALL append only dates
newer than the cache's last date. It SHALL count overlapping dates whose values differ, and
log every refresh per series.

#### Scenario: Revised value
- **WHEN** a source returns a different value for a date already cached
- **THEN** the cached value is kept and the revision is counted in the refresh log

### Requirement: FX D1 bars
The FX D1 bars for EURUSD and GBPUSD SHALL be built read-only from the fx_session H1
caches from 2012-01-01, by trading date. A D1 bar built from fewer than 12 H1 bars SHALL
count as missing. The D1 mid close SHALL be the last H1 bar's bid close plus half its
spread.

#### Scenario: One-bar day
- **WHEN** a trading date holds a single H1 bar
- **THEN** that date has no D1 bar

### Requirement: Yields lag one FX trading day
For the decision at the close of D1 bar `t`, every yield value SHALL come from a calendar
date at or before the FX trading date before `t`. No yield value dated `t` or later SHALL
enter day `t`'s features.

#### Scenario: Same-day yield
- **WHEN** DGS2 is published for date `t`
- **THEN** it enters features only from the next FX trading day on

### Requirement: Limited forward-fill and stale flag
When a series has no value on the lagged date, the most recent earlier value SHALL be
used, at most 3 business days old. `yield_stale_flag` SHALL be 1 when any series was
filled. A day whose fill would exceed 3 business days SHALL have no feature row. Values
SHALL never be taken from a later date.

#### Scenario: US holiday
- **WHEN** DGS2 has no value for the lagged date because of a US holiday, but has one the business day before
- **THEN** that earlier value is used and `yield_stale_flag` is 1

#### Scenario: Long outage
- **WHEN** a series' latest value before the lagged date is 5 business days old
- **THEN** the day has no feature row and is counted as "stale yields"
