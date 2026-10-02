# Spec Delta

## Purpose

Measures, for every model family in the project, how forecasting skill and cost-net
value change with the forecast horizon. It produces a descriptive horizon curve per
model that later forward evaluation can build on.

## ADDED Requirements

### Requirement: Fixed horizon grid
The study SHALL use a horizon grid fixed before the first model is fitted. H1-cadence
models use {1, 2, 4, 6, 12, 24, 48, 120} H1 bars. Daily-cadence models (direction and
volatility) use {1, 2, 5} trading days. Changing the grid after the first fit SHALL
require a new, separately recorded study.

#### Scenario: The grid is recorded before fitting
- **WHEN** the study starts
- **THEN** the grid is written to the study record before any model is fitted

#### Scenario: The grid cannot change mid-study
- **WHEN** a horizon is added or removed after the first fit
- **THEN** the run is refused unless it is started as a new study with its own record

### Requirement: A challenger for every model type
The study SHALL implement its own challenger for each model type in the project, at
its native cadence: daily GBM and daily LSTM, each on the price-only and the
with-macro feature sets, the H1→daily ensemble, TI-LSTM H1, the H1 next-bar direction
GBM and the volatility multi-task LSTM ensemble. Kronos direction and Kronos
volatility SHALL be run as-is, without training.

#### Scenario: Inventory is complete
- **WHEN** the study plan is produced
- **THEN** every model type above appears with its cadence and its horizons, or with a stated reason it could not be fitted

#### Scenario: One family failing does not stop the rest
- **WHEN** one family fails to fit at one horizon
- **THEN** that cell is recorded as failed with its reason, and every other cell still runs

### Requirement: Standalone from production code
The challengers SHALL be trained by the study's own code. The study SHALL import the
project's feature computation read-only, so that features are identical in both
systems. It SHALL NOT modify any checksum-pinned file or re-baseline any fixture.

#### Scenario: Pinned files untouched
- **WHEN** the study has been built and run
- **THEN** every file pinned by `tests/fixtures/*_protected_sha256.json` is byte-identical and the checksum tests pass without a re-baseline

#### Scenario: Features match production
- **WHEN** the study computes features for a given bar
- **THEN** they equal the values the production feature code computes for that bar

### Requirement: Configuration declared once and frozen across horizons
Each challenger's architecture, hyperparameters and feature set SHALL be declared in
the study record before its first fit, and SHALL be identical at every horizon. Only
the target horizon changes. Kronos SHALL run at its pinned generation settings, with
the forecast length set to the horizon.

#### Scenario: Configuration does not vary by horizon
- **WHEN** a challenger is fitted at any two horizons
- **THEN** its recorded configurations are identical except for the target horizon

### Requirement: One pre-declared tuning rule per cell
A tuned variant SHALL be produced only by nested purged cross-validation over a search
grid declared before fitting. It SHALL replace the frozen configuration only if a
decision rule, also declared in advance, says so on development data. Exactly one
configuration per cell SHALL leave the study.

#### Scenario: Tuning cannot enlarge the family
- **WHEN** a cell has both a frozen and a tuned variant
- **THEN** the declared rule selects exactly one, both variants are counted in the trial log, and only the selected one is passed on

#### Scenario: Undeclared search is refused
- **WHEN** a hyperparameter value outside the declared grid is used
- **THEN** the run is refused

### Requirement: Horizon targets defined without look-ahead
The direction target at horizon h SHALL be sign(close at t+h minus close at t). The
return target SHALL be the percent change over h. The volatility target SHALL be
|log return over h| × 100. Features at time t SHALL use only bars at or before t.
Scalers and decompositions SHALL be fitted inside each training fold only.

#### Scenario: Zero-move rows are handled by a stated rule
- **WHEN** close at t+h equals close at t
- **THEN** the row is excluded from directional accuracy, and the excluded count is reported per cell

#### Scenario: No future information in features
- **WHEN** features are built for time t
- **THEN** shifting all data after t changes no feature value at t

### Requirement: Cost breakeven per horizon
For each instrument and horizon, the study SHALL compute the breakeven directional
accuracy, 0.5 + c / (2 · E|r_h|), from development data only. It SHALL be computed at
the measured spread and at the config round trip, and recorded per cell.

#### Scenario: Breakeven recorded for every cell
- **WHEN** the horizon curve is produced
- **THEN** each cell shows its breakeven at both cost levels, next to the model's accuracy

#### Scenario: Below-breakeven skill is labelled
- **WHEN** a cell's accuracy exceeds 50 % but not its breakeven
- **THEN** the report labels it "predictive, not cost-viable" rather than as an edge

### Requirement: Descriptive horizon curve
For each model, the study SHALL report per horizon: accuracy, AUC, gross and net mean
return per trade, breakeven, the Pesaran–Timmermann p-value, the CPCV path
distribution, PBO and Deflated Sharpe. Every result SHALL be labelled descriptive and
never KEEP or DROP.

#### Scenario: Curve is complete and labelled
- **WHEN** the study finishes
- **THEN** one curve per model covers every horizon in its grid, and every file states that it is descriptive

### Requirement: Research artifacts kept apart from production
Fitted study models SHALL be written outside `models/`, each with its SHA-256 digest in
a tracked manifest. No serving path SHALL load them.

#### Scenario: Production is untouched
- **WHEN** the full study has run
- **THEN** no file under `models/` has changed, and the provenance digest check still reports nothing undeclared

#### Scenario: Every artifact is traceable
- **WHEN** a study model file exists
- **THEN** the manifest lists its path, digest, family, horizon, configuration and training date range

### Requirement: Declared use of historical data
Reading the full EUR/USD history, including the spent test blocks, SHALL be limited to
development use, because no verdict is drawn from it. Every report SHALL state the
exact date range read. No study output SHALL be written to any existing hypothesis log.

#### Scenario: Date range is stated
- **WHEN** any study output is written
- **THEN** it states the first and last bar read

#### Scenario: Existing registries untouched
- **WHEN** the study completes
- **THEN** every existing `*_hypothesis_log.csv` is byte-identical to its state before the study
