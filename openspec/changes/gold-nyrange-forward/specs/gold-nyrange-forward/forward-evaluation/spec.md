# Spec Delta

## Purpose

Defines how the gold NY-range forward test is decided: the pinned frozen coefficients, the
pre-registration that must be committed first, the single forward cell and its one-time scoring,
and the plain report.

## ADDED Requirements

### Requirement: Pinned coefficients
The pre-registration SHALL copy the HAR-X (15) and HAR (4) coefficients from gold_bbnr's frozen
`ols.json`, with its digest. Every pass SHALL use the copied coefficients. It SHALL refuse to run if
gold_bbnr's `ols.json` exists and its digest differs from the recorded one.

#### Scenario: Altered ols.json
- **WHEN** gold_bbnr's `ols.json` is edited after the pre-registration
- **THEN** the next pass refuses to run

### Requirement: Pre-registration gate
`results/gold_nyfwd/PRE_REGISTRATION.md` and `.json` SHALL be committed before any forward day.
Each pass SHALL refuse unless:
- both files are tracked by git and unchanged from HEAD;
- the JSON matches the family's config digest.

The pre-registration SHALL state the cell, alpha, the horizon and the expected power.

#### Scenario: Uncommitted pre-registration
- **WHEN** a pass runs while `PRE_REGISTRATION.json` is untracked
- **THEN** it refuses and logs nothing

### Requirement: Forward days
A forward day SHALL be a settled `ok` day whose date is after the pre-registration's commit date on
the label clock. Days settled before that SHALL never count.

#### Scenario: Commit day
- **WHEN** the pre-registration is committed at 10:00 Sofia on 2026-12-01
- **THEN** 2026-12-01 is not a forward day, and 2026-12-02 is the first one possible

### Requirement: The forward cell
One cell, `gold_harx_vs_har_forward`, SHALL be registered at `alpha = 0.05` in
`results/gold_nyfwd/gold_nyfwd_hypothesis_log.csv`. It SHALL test the per-day difference of the
HAR-X and HAR squared errors with a one-sided 5-day block bootstrap. It SHALL be KEEP only if the
upper bound of the mean difference is below 0.

#### Scenario: Planted improvement
- **WHEN** synthetic forward days give HAR-X squared errors clearly below HAR's
- **THEN** the cell is KEEP

#### Scenario: No difference
- **WHEN** synthetic forward days give equal squared errors in expectation
- **THEN** the cell is DROP

### Requirement: Scored once at 60 days or 2027-04-30
Scoring SHALL run once, either when the cell has 60 forward days or on 2027-04-30. With fewer than
60 forward days on 2027-04-30, the cell SHALL be recorded as `UNDERPOWERED — NO DECISION`. Earlier
scoring SHALL refuse and write nothing. A second scoring SHALL refuse and leave the registry
unchanged.

#### Scenario: Early scoring
- **WHEN** scoring runs on 2027-02-15 with 40 forward days
- **THEN** it refuses and the registry is unchanged

#### Scenario: Re-scoring
- **WHEN** the cell is already scored and scoring runs again
- **THEN** it refuses and the registry is byte-identical

### Requirement: Stated low power
The pre-registration and the report SHALL state the expected power at 60 days: about 35 % if the
true effect equals the historical one. They SHALL state that a DROP is not evidence that the model
fails.

#### Scenario: Power stated
- **WHEN** the pre-registration is written
- **THEN** it states the expected power and that a DROP is not evidence of failure

### Requirement: Separate and gross
The cell SHALL be a gross predictive result on bid prices, never a trading result. The family SHALL
never touch the `0.05/9` bar, the gold, gold_bbnr or fx_ny_orb registries, `models/` or
`config.json`.

#### Scenario: Protected paths
- **WHEN** both daily passes and the scoring run
- **THEN** every other registry, `models/` and `config.json` are byte-identical before and after

### Requirement: Plain report
`results/gold_nyfwd/report.md` SHALL show:
- the forward day count;
- days without a forecast, by reason;
- the dollar-index contracts used.

Before scoring, it SHALL show no squared error and no MSE, and it SHALL say `not scored yet`. After
scoring, it SHALL show the verdict, MSE by month and HAR-X's MSE relative to HAR's. Building it
twice SHALL give the same content apart from the generated-at line.

#### Scenario: Before scoring
- **WHEN** the report is built with 30 forward days and no score
- **THEN** it shows the count and `not scored yet`, and no error figure
