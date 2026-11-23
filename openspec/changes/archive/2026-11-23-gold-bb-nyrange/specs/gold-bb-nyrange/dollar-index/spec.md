# Spec Delta

## Purpose

Defines the US dollar index inputs of gold-bb-nyrange: an hourly series from the broker's MT5
terminal for the New York range features, and a daily series from Yahoo for the Bollinger
features. Each is fetched once into the family's own cache and pinned by digest.

## ADDED Requirements

### Requirement: One-time hourly DXY cache from MT5
A fetch stage SHALL read the broker's US Dollar Index CFD H1 bars (OHLC only) from MT5 and write
them to `results/gold_bbnr/data/dxy_h1.csv`. The symbol is set in the config. The stage SHALL
use only market-data calls, never place or query orders, and keep only bars from 2017-01-03
through 2026-11-20.

#### Scenario: Read-only terminal use
- **WHEN** the fetch stage runs
- **THEN** it calls no MT5 order, position or account-trading function

### Requirement: Hourly DXY clock check
Every stage that reads `dxy_h1.csv` SHALL check, for each calendar year, that the correlation
of DXY and EURUSD H1 log returns is most negative at lag 0 among the lags −1, 0 and +1. EURUSD
comes from fx-session's cache. The stage SHALL refuse to run when any year fails.

#### Scenario: Shifted series
- **WHEN** a year of DXY labels is shifted by one hour
- **THEN** that year's correlation peaks at a lag other than 0 and the stage stops with an error

### Requirement: One-time daily DXY cache from Yahoo
A fetch stage SHALL download the daily `DX-Y.NYB` closes from 2014-06-01 through 2026-11-20
into `results/gold_bbnr/data/dxy_d1.csv`. It SHALL send a generic client name and no personal
data.

#### Scenario: No personal data sent
- **WHEN** the Yahoo request is made
- **THEN** its headers carry only the generic client name

### Requirement: Pinned caches
The SHA-256 digests of both DXY caches SHALL be recorded in the pre-registration. Every gated
stage SHALL refuse a cache whose digest differs. No stage after the pre-registration SHALL
refetch either cache.

#### Scenario: Cache changed
- **WHEN** `dxy_d1.csv` is edited after the pre-registration was committed
- **THEN** every gated stage refuses to run

### Requirement: Previous-day use of daily DXY
A Part 1 feature for gold day `t` SHALL use only daily DXY closes dated strictly before `t`.

#### Scenario: Same-day close excluded
- **WHEN** the features of a breach on 2020-03-12 are computed
- **THEN** the latest DXY close they use is dated 2020-03-11 or earlier
