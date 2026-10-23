# gold-forecasting/logger-health Specification

## Purpose

Keeps the hourly gold logger running through sleep and battery changes, and makes any
stall visible to the owner instead of leaving silent holes.

## Requirements

### Requirement: The task wakes the PC and catches up
The installed `EURUSDProphet-GoldLogger` task SHALL be set to:
- wake the computer to run;
- start as soon as possible after a missed start;
- run on battery and keep running on battery;
- never run two instances at once.

The installer SHALL report whether wake timers are enabled in the active power plan.

#### Scenario: Install on a laptop
- **WHEN** the task is installed
- **THEN** its settings show WakeToRun, StartWhenAvailable, battery start allowed, and MultipleInstances IgnoreNew, and the installer prints the wake-timer state

#### Scenario: Late start after sleep
- **WHEN** the PC wakes at 08:20 after missing the 03:03 to 08:03 runs
- **THEN** one pass runs promptly, forecasts the newest closed bar, and records the missed bars as gaps without back-filling

### Requirement: Every pass leaves a health record
Each logger pass SHALL append one row to `results/gold/forward/health.csv`, with:
- its start and end time and its outcome;
- whether MT5 was reachable;
- the newest bar it saw;
- the forecasts, settlements and gaps it wrote.

#### Scenario: MT5 closed
- **WHEN** MT5 is not running during a pass
- **THEN** the health row records MT5 unreachable and zero forecasts, and the pass exits cleanly

### Requirement: A stall is shown to the owner
`GET /api/gold/health` SHALL report:
- the last pass;
- the last successful pass;
- the missed hours in the last 24 hours.

`/gold` SHALL show a warning when the last successful pass is more than two hours old
during market hours, or when more than three hours were missed in the last 24.

#### Scenario: Logger stopped
- **WHEN** the last successful pass was five hours ago on a weekday afternoon
- **THEN** `/gold` shows a warning naming the time of the last success

#### Scenario: Weekend
- **WHEN** the market is closed for the weekend
- **THEN** no stall warning is shown
