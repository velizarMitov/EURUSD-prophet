# Spec Delta

## Purpose

Shows the owner the newest gold forecasts and their recorded verdicts without running any
model, so research output is visible but never presented as a proven or tradable signal.

## ADDED Requirements

### Requirement: The view is read-only
`GET /api/gold` and the `/gold` page SHALL only read the gold logger's and study's
recorded files. They SHALL run no model, fetch no market data and write no file.

#### Scenario: Page opened
- **WHEN** `/gold` is requested
- **THEN** the response is built from recorded files only and no file under the repository changes

#### Scenario: Nothing logged yet
- **WHEN** no gold forecast has been logged
- **THEN** the page says no gold forecast exists yet and shows no number

### Requirement: The newest forecast per cell is taken whole
For each gold cell, the view SHALL show the single newest logged row exactly as recorded.
A blank field SHALL never be filled from an older row.

#### Scenario: Newest row has no direction probability
- **WHEN** the newest row of a cell has a blank direction probability and an older row has 0.62
- **THEN** the cell shows no direction and 0.62 appears nowhere as current

### Requirement: A finished window is not shown as current
The view SHALL omit, as a current forecast, any row whose target bar has already closed
on the bar-label clock.

#### Scenario: Logger stopped yesterday
- **WHEN** the newest H1 row targets a bar that closed hours ago
- **THEN** the H1 block shows no current forecast and states when the last one ended

### Requirement: Each forecast shows direction, return, implied price and expected move
A current forecast SHALL show the direction with its probability, the expected return in
percent, the implied price (close times e to the forecast return) and the expected
absolute move in percent where a volatility forecast exists. It SHALL show no spread,
pips or P&L.

#### Scenario: Complete forecast
- **WHEN** a D1 row has p_up 0.53, return 0.04 %, volatility 0.81 % and as-of close 4,131.00
- **THEN** the page shows "up, 53 %", "+0.04 %", the implied price 4,132.65 and an expected move of ±0.81 %

### Requirement: Every forecast is labelled with its recorded verdict
Each cell SHALL display its family verdict exactly as recorded, or "UNDERPOWERED — NO
DECISION". The page SHALL carry the label "research — no proven edge" for every cell
without a forward KEEP.

#### Scenario: Validation CLEARED but no forward verdict
- **WHEN** a cell CLEARED on validation and has no forward verdict
- **THEN** it is still labelled "research — no proven edge" and the validation result is shown as history

#### Scenario: Forward KEEP
- **WHEN** a cell has a forward KEEP
- **THEN** it reads "statistical skill, gross of cost; not a trading result" and never "edge" or "profitable"
