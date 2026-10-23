# gold-forecasting/leaderboard Specification

## Purpose

Shows the owner, on live data, how every gold model and version is doing next to its
baseline, as a description that never replaces a registered verdict.

## Requirements

### Requirement: One row per model version on live data
`GET /api/gold/leaderboard` and the *Leaderboard* tab on `/gold` SHALL list one row per
cell and model digest, built only from settled live rows. Each row SHALL show:
- the count of settled forecasts;
- the hit rate with a Wilson 95 % interval, where the cell forecasts direction;
- the move error next to its GARCH baseline on shared bars, where it forecasts
  volatility;
- range coverage at 80 % and 95 %, where it forecasts ranges;
- the phase (dry run or registered) and whether the row comes from a shadow version.

#### Scenario: Shadow version listed
- **WHEN** the frozen h1_lstm and a shadow refit both have settled rows
- **THEN** they appear as two rows with their digests, and the shadow one is labelled shadow

### Requirement: The board is descriptive
Every row SHALL be labelled "descriptive — not a verdict". The board SHALL never show
KEEP or DROP. Below 30 settled rows, a row SHALL show only its count and "too few to
say".

#### Scenario: Ordering
- **WHEN** the board is sorted
- **THEN** it sorts by the chosen metric among rows with at least 30 settled forecasts, and the rest are listed after them unsorted

### Requirement: The board is read-only
Building the board SHALL run no model, fetch no market data and write no file.

#### Scenario: Board opened
- **WHEN** `/api/gold/leaderboard` is requested
- **THEN** no file under the repository changes
