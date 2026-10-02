# Spec Delta

## Purpose

Measures, for every model family in the project, how forecasting skill and cost-net
value change with the forecast horizon. It produces a descriptive horizon curve per
model that later forward evaluation can build on.

## ADDED Requirements

### Requirement: Fixed horizon grid
The study SHALL use a horizon grid fixed before the first model is fitted. M15-cadence
models use {1, 2, 4, 8, 16, 26} M15 bars. H1-cadence models use {1, 2, 4, 6, 12, 24,
48, 120} H1 bars. Daily-cadence models (direction and volatility) use {1, 2, 5} trading
days. Changing the grid after the first fit SHALL require a new, separately recorded
study.

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

### Requirement: Session challengers on the M15 cadence
The study SHALL add two challengers on the M15 session cadence: a gradient-boosting
model and a sequence model. Their features SHALL be built from M15 mid bars and MAY
include intrabar movement, tick volume, the bar's own spread and the bar's position
within the session. Their configurations SHALL be declared once and frozen across
horizons, as for every other challenger.

#### Scenario: Both session challengers appear with their grid
- **WHEN** the study plan is produced
- **THEN** both M15 challengers appear with cadence M15 and the full M15 horizon grid

#### Scenario: The session restricts forecasts, never the features' direction of time
- **WHEN** an M15 challenger is fitted
- **THEN** the session window decides only which as-of bars are forecast, and no feature reads a bar later than its as-of bar

### Requirement: Mid-price series for sub-hourly cadences
Sub-hourly bars SHALL be aggregated from one-minute history that carries a per-bar
spread. Every sub-hourly target, accuracy and cost figure SHALL be computed on the mid
price, defined as the bid close plus half the spread. Each aggregated bar SHALL retain
its own spread.

#### Scenario: Targets are built on mid, not on the bid close
- **WHEN** a sub-hourly direction target is built
- **THEN** it is the sign of the change in mid price, and the bar's own spread is recorded beside it

#### Scenario: A source without a spread is refused
- **WHEN** the one-minute source for a sub-hourly bar carries no spread field
- **THEN** the aggregation is refused rather than assuming a spread

### Requirement: The bar-label clock is established by measurement
The wall clock that labels the bars SHALL be established from the data, never assumed,
and recorded in the study record before the first fit. The weekly market open and
close labels, including the weeks in which US and EU daylight saving disagree, SHALL
be the evidence. Every session rule SHALL be expressed against the established clock.

#### Scenario: The clock is evidenced, not assumed
- **WHEN** any session rule is applied
- **THEN** the study record states the established label clock and the weekly-boundary evidence for it

#### Scenario: A clock change is refused, not absorbed
- **WHEN** the weekly open label stops matching the recorded clock
- **THEN** the run is refused, rather than silently shifting every session rule

### Requirement: Session-restricted evaluation window for sub-hourly cells
Sub-hourly cells SHALL forecast only inside the owner's declared trading session,
15:30 to 23:00 Europe/Sofia on weekdays. A trade SHALL be eligible at horizon h only
when its as-of bar and its target bar both fall inside that session on the same session
day. Any other row SHALL be recorded with that exclusion reason and left out of
scoring.

#### Scenario: Out-of-session bars produce no forecast
- **WHEN** an as-of bar falls outside the declared session
- **THEN** no sub-hourly forecast is made for it, and its absence is not recorded as a gap

#### Scenario: A trade that would end outside the session is not eligible
- **WHEN** a forecast's target bar would close after the session end
- **THEN** the row is recorded with that exclusion reason and is left out of every accuracy figure and verdict

#### Scenario: The session does not drift with daylight saving
- **WHEN** the session is resolved in January and again in July
- **THEN** both resolve to the same bar-label time of day, because the owner's zone and the bar-label zone share one daylight-saving rule

### Requirement: Bid-versus-mid parity check on every sub-hourly result
Every sub-hourly cell SHALL be scored twice, on mid prices and on bid closes, and both
figures SHALL be reported with their difference. A cell whose two accuracies differ by
more than a tolerance declared before fitting SHALL be labelled a spread artifact and
SHALL NOT be reported as skill.

#### Scenario: A spread-driven result is labelled, never presented as an edge
- **WHEN** a cell's bid and mid accuracies differ by more than the declared tolerance
- **THEN** the cell is labelled "spread artifact" and its bid figure is never reported as skill

#### Scenario: Both price definitions reach the report
- **WHEN** the horizon curve is produced
- **THEN** every sub-hourly cell shows its accuracy on mid, its accuracy on bid, and the difference

#### Scenario: A small sample is not called an artifact on noise alone
- **WHEN** a cell's two accuracies differ by more than the declared tolerance but by less than the sampling noise of that difference
- **THEN** the cell is not labelled a spread artifact, and the noise floor is reported beside the difference

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

#### Scenario: A sub-hourly breakeven uses the session's own spread
- **WHEN** a sub-hourly cell's breakeven is computed
- **THEN** the measured level is the median spread of the bars inside the declared session, not the all-hours median

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
