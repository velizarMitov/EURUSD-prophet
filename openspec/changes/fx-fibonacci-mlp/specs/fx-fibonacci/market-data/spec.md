# Spec Delta

## Purpose

Supplies the fx-fibonacci research family with MT5 hourly bars for each configured
currency pair, daily bars built from them on the broker calendar, and the causal rule
that decides which daily bars an hourly bar may see.

## ADDED Requirements

### Requirement: Pairs come from the family config
The family SHALL process exactly the pairs listed in its config. Phase 1 lists `EURUSD` and
`GBPUSD`. Adding a pair SHALL require only a config entry and a data refresh, with no code
change. Every pair SHALL be processed with identical rules and parameters.

#### Scenario: A third pair is added
- **WHEN** `AUDUSD` is appended to the config's pair list and the data refresh runs
- **THEN** its cache, features, labels and reports are produced with no source edit

### Requirement: The hourly cache holds MT5 OHLC only
The family SHALL keep one H1 cache per pair in its own results tree. Each cache SHALL hold
the label time and open, high, low and close from a live MT5 terminal. It SHALL hold no
tick volume and no spread. It SHALL never rewrite `results/pooled_h1/` or
`results/eurusd_h1.csv`.

#### Scenario: Refresh appends new bars
- **WHEN** the refresh runs while MT5 is reachable
- **THEN** bars newer than the cache's last label are appended, and no existing row changes

#### Scenario: MT5 unreachable
- **WHEN** the refresh runs while MT5 is not reachable
- **THEN** the cache is left unchanged, the run reports the failure, and no other vendor is tried

#### Scenario: Symbol cannot be resolved
- **WHEN** a configured pair matches no broker symbol, or matches several
- **THEN** the refresh stops for every pair and names the unresolved pair

### Requirement: Bar labels are on the broker clock
Bar labels SHALL be treated as the broker server's wall clock, which is `Europe/Berlin`,
not as UTC. A refresh SHALL refuse to write a cache whose weekly-open labels stop matching
the expected Berlin open hour.

#### Scenario: Clock drift detected
- **WHEN** the weekly opens of a fetched series do not match the expected open hour
- **THEN** the refresh writes nothing for that pair and reports the mismatch

#### Scenario: DST-mismatch week
- **WHEN** a week falls in the March or late-October gap between the US and EU clock changes
- **THEN** its weekly open is checked against that week's own expected hour, not the usual one

### Requirement: Daily bars are built from hourly bars
D1 bars SHALL be aggregated from the H1 cache by broker-calendar date: first open, maximum
high, minimum low, last close. A bar labelled on a Sunday SHALL be merged into the next
Monday's D1 bar. A D1 bar SHALL close at the end of its last H1 bar.

#### Scenario: Sunday stub
- **WHEN** the H1 cache holds a bar labelled Sunday 23:00
- **THEN** no Sunday D1 bar exists, and that bar's prices are part of Monday's D1 bar

#### Scenario: Reproducible aggregation
- **WHEN** D1 bars are built twice from the same H1 cache
- **THEN** both results are identical

### Requirement: An hourly bar sees only closed daily bars
For an H1 bar labelled `t`, which closes at `t + 1 h`, every D1-derived value SHALL come
only from D1 bars whose close time is at or before `t + 1 h`. A D1 bar that is still open
SHALL never contribute to an H1 bar's features.

#### Scenario: Midday bar
- **WHEN** features are computed for the H1 bar labelled Wednesday 14:00
- **THEN** the newest D1 bar it uses is Tuesday's

#### Scenario: Last bar of the day
- **WHEN** features are computed for the H1 bar labelled Wednesday 23:00
- **THEN** Wednesday's D1 bar is complete at that bar's close, so the bar may use it

#### Scenario: Future bars changed
- **WHEN** every H1 bar after label `t` is altered or removed and the features are recomputed
- **THEN** the features of every bar up to and including `t` are unchanged
