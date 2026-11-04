# Spec Delta

## Purpose

Provides an independent gold price history from before 2013, with bid and ask on the
repo's Berlin label clock. Its prices stay unread until a registration that uses them is
committed, so one rule fixed in advance can be backtested once on years nobody has seen.

## ADDED Requirements

### Requirement: The archive holds M1 bid and ask candles in a gitignored folder
The downloader SHALL fetch XAU/USD one-minute bid and ask candles, one trading day per
request, into a gitignored folder. It SHALL be resumable, SHALL skip days already stored
with a matching digest, and SHALL be throttled. A failed day SHALL be recorded as a gap
and never filled by interpolation. Nothing SHALL be written under `models/` or into a
tracked results file except the data manifest.

#### Scenario: Resume after interruption
- **WHEN** the download is run again after stopping partway
- **THEN** stored days with a matching digest are skipped and only missing days are fetched

#### Scenario: Day not available
- **WHEN** a trading day cannot be fetched after the retry limit
- **THEN** it is listed as a gap with the error, and no candle is invented for it

### Requirement: The archive is pinned by digest
Each stored year SHALL have a SHA-256 digest, a first and last timestamp, a day count and
a gap count in `results/gold/window/data_manifest.csv`. Any later stage SHALL refuse to
run if a stored year's digest no longer matches the manifest.

#### Scenario: Changed file
- **WHEN** a stored year file is modified after the manifest was written
- **THEN** every stage that reads prices refuses with the year and both digests

### Requirement: Timestamps are converted to the Berlin label clock
Source timestamps SHALL be treated as UTC. They SHALL be converted to Europe/Berlin wall
time with daylight saving, and stored as wall-clock labels in the repo's convention, where
the label is not UTC. H1, M15 and D1 bars SHALL be built from M1 on that label clock, with
the D1 day being the Berlin calendar day.

#### Scenario: Summer and winter
- **WHEN** a source candle is stamped 13:00 UTC on a July day and another on a January day
- **THEN** their labels are 15:00 and 14:00 respectively

### Requirement: The clock and scale are verified on the overlap with MT5
On the overlap 2013-08-08 to 2014-12-31, the archive's H1 mid closes SHALL be compared
with the MT5 cache:
- the lag that maximises the correlation of hourly returns SHALL be 0;
- that correlation SHALL be at least 0.95;
- the median absolute close difference SHALL be at most $1.00.

The weekly open labels over the whole archive SHALL also be tabulated. Any failed check
SHALL refuse registration.

#### Scenario: Clock off by one hour
- **WHEN** the best-correlated lag on the overlap is +1 or −1 hour
- **THEN** verification fails and the registration cannot be written

#### Scenario: Verification passes
- **WHEN** lag 0 is best, the correlation is at least 0.95 and the median difference is at most $1.00
- **THEN** the results are written into the data manifest and the registration may be written

### Requirement: Pre-overlap prices stay unread until registration
Before the window-strategy registration is committed and clean, only these parts of the
archive before 2013-08-08 SHALL be readable:
- timestamps;
- bar counts;
- gap counts.

Any request for its prices, returns or spreads SHALL refuse.

#### Scenario: Early read refused
- **WHEN** a stage asks for 2009 mid prices while the registration is uncommitted
- **THEN** it refuses and reads nothing

#### Scenario: After registration
- **WHEN** the registration files are committed and unmodified
- **THEN** the confirmation stage may read the pre-overlap prices
