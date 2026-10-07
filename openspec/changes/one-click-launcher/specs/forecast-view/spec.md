# Spec Delta

## Purpose

Defines what the `/forecasts` view may present as a current forecast, so the
owner never acts on a stale or borrowed number.

## ADDED Requirements

### Requirement: The newest forecast per cell is taken whole
For each model and horizon, the view SHALL use the single newest logged row
exactly as recorded. A blank field in that row SHALL NOT be filled from an older
row.

#### Scenario: Newest forecast has no probability
- **WHEN** a cell's 10:00 forecast has `p_up` 0.70 and its newer 10:15 forecast has a blank `p_up`
- **THEN** the cell shows the 10:15 forecast with direction "—" (no answer), and 0.70 appears nowhere as current

#### Scenario: Newest forecast is complete
- **WHEN** the newest row of a cell has `p_up` 0.41
- **THEN** the cell shows "−" with 41 % for up

### Requirement: A finished window is not shown as current
The headline SHALL omit any forecast whose window ended at or before the current
time, measured on the bar-label clock.

#### Scenario: Morning after a session
- **WHEN** the newest 15-minute forecast covers 23:00–23:15 the previous evening and the page is opened at 09:34
- **THEN** the "след 15 минути" block is absent, and only forecasts whose window has not ended are shown
