# Spec Delta

## Purpose

Builds and judges gold price, direction and volatility models from the broker's MT5 bars
under the same methodology as EUR/USD: chronological splits, a validation-only arbiter,
pre-registered Bonferroni families and a one-shot test report, isolated from production.

## ADDED Requirements

### Requirement: Gold bars come from MT5 as OHLC only
Gold history SHALL be read from the live MT5 terminal's gold symbol at D1 and H1, with an
on-disk cache as the only offline fallback. Stored bars SHALL carry open, high, low and
close only. Tick volume, real volume and spread SHALL be dropped at acquisition and never
stored.

#### Scenario: Volume and spread never reach disk
- **WHEN** gold bars are fetched and cached
- **THEN** the cache's columns are exactly the bar time plus open, high, low and close

#### Scenario: Terminal unavailable
- **WHEN** no MT5 terminal is reachable and a cache exists
- **THEN** the run uses the cache and records that source; no other vendor's prices are used

#### Scenario: No terminal and no cache
- **WHEN** no MT5 terminal is reachable and no cache exists
- **THEN** the run stops with a clear error and writes no result

### Requirement: Runs refuse an unverified bar clock
Before any feature is built, a run SHALL check that the gold weekly market opens fall on
the measured broker-clock pattern (Monday 00:00, or Sunday 23:00 in US/EU daylight-saving
mismatch weeks). A run whose matching share falls below the registered threshold SHALL
stop.

#### Scenario: Clock pattern holds
- **WHEN** at least the registered share of weekly opens match the pattern
- **THEN** the run proceeds and records the measured share

#### Scenario: Broker changes its clock
- **WHEN** the share of matching weekly opens falls below the threshold
- **THEN** the run stops before producing any feature, prediction or score

### Requirement: Gold features are scale-free and price-only
Every model input SHALL be invariant to the price level: returns, ranges and distances
normalised by price or volatility, rolling volatility, and calendar encodings. No raw
price level, moving-average level, unnormalised range, volume, spread or external macro
series SHALL be a model input.

#### Scenario: Level columns are rejected
- **WHEN** a feature set containing a raw open, high, low, close or moving-average level is passed to a gold challenger
- **THEN** the challenger refuses to fit

#### Scenario: Same features at any price level
- **WHEN** an entire OHLC history is multiplied by a constant
- **THEN** every gold feature value is unchanged

### Requirement: Targets are next-bar return, direction and volatility
The targets SHALL be the next bar's log return in percent, its sign, and its absolute
value in percent, each built only from bars after the as-of bar. A price forecast SHALL
be derived as the as-of close times the exponential of the forecast return and SHALL
never be a training target.

#### Scenario: No target leaks into features
- **WHEN** every bar after an as-of bar is altered
- **THEN** that bar's features are unchanged and only its targets change

#### Scenario: Price is derived, not modelled
- **WHEN** a challenger forecasts a return of 0.10 % from a close of 4,000.00
- **THEN** the reported price forecast is 4,000.00 × e^0.001 and no model was trained on price levels

### Requirement: Unified chronological split
Each cadence SHALL split chronologically at the same fractions as EUR/USD: training
[0:70 %], validation [70:80 %] and test [80:100 %]. Gradient-boosted models SHALL fit on
training only during validation scoring. Sequence models SHALL early-stop on the last
part of training. Every scaler and transform SHALL be fit on training rows only.

#### Scenario: Preprocessing sees no validation row
- **WHEN** validation rows are changed arbitrarily
- **THEN** every fitted scaler and model parameter used for validation scoring is unchanged

#### Scenario: Labels do not straddle a boundary
- **WHEN** a target window crosses a split boundary
- **THEN** that row is purged from the earlier block

### Requirement: H1 eligibility excludes bars that border a market gap
An H1 row SHALL be scored only if its as-of bar and target bar are consecutive hours with
no gap between them, and neither is the first bar after a gap. This replaces the
bid-versus-mid check that cannot run without spread.

#### Scenario: Daily break
- **WHEN** the as-of bar is the last bar before the daily break
- **THEN** that row is excluded from fitting targets and from every score

#### Scenario: First bar of the week
- **WHEN** the target bar is the first bar after the weekend
- **THEN** that row is excluded

### Requirement: Each challenger is scored against its own honest baseline
Every reported number SHALL be paired with its baseline on the same rows:
- direction against the training-majority class and 50 %;
- return against the zero-return forecast;
- volatility against GARCH(1,1) fitted on training only (D1), or GARCH with
  hour-of-day seasonality fitted on training only (H1).

#### Scenario: Direction report
- **WHEN** a direction challenger is scored
- **THEN** the report shows its accuracy, the majority-class accuracy and the Pesaran–Timmermann statistic on the same rows

#### Scenario: Volatility report
- **WHEN** a volatility challenger is scored
- **THEN** the report shows its MAE, the GARCH baseline's MAE and a block-bootstrap confidence interval of the difference

### Requirement: Neural challengers are seed ensembles, all or nothing
Each neural challenger SHALL be the mean of five independently seeded fits. If any seed
fails to train or load, that challenger SHALL produce no prediction rather than a partial
ensemble.

#### Scenario: One seed missing
- **WHEN** one of five seed models is unavailable
- **THEN** the challenger is reported as failed for that run and no prediction is emitted

### Requirement: Three gold families with registered composition
Direction, return and volatility SHALL each form their own family with its own Bonferroni
alpha. Composition SHALL be fixed in the pre-registration from power arithmetic: a cell
whose validation rows fall short of the required n is registered UNDERPOWERED — NO
DECISION, is still reported, and spends no alpha.

#### Scenario: Daily direction is not adjudicated
- **WHEN** a D1 direction cell has fewer validation rows than the registered required n
- **THEN** it is registered UNDERPOWERED — NO DECISION and excluded from the direction family's alpha

#### Scenario: Gold families stay separate from EUR/USD
- **WHEN** any gold verdict is recorded
- **THEN** no EUR/USD registry or bar changes, including the feature bar of 0.05/9

#### Scenario: A failing cell does not loosen the bar
- **WHEN** an admitted cell becomes unrunnable after registration
- **THEN** the family size and alpha stay as registered and the cell is recorded as unspent

### Requirement: Pre-registration precedes the first validation score
A gold pre-registration SHALL be committed before any validation score exists. It fixes:
- the cells and their families;
- the frozen configurations and any inner-train tuning grid;
- the metrics and baselines;
- the alpha, CI method and verdict rule;
- the eligibility rule;
- an expected outcome for each cell.

Corrections SHALL go in a separate results document.

#### Scenario: Scoring before registration
- **WHEN** validation scoring is started and no committed, unmodified gold pre-registration exists
- **THEN** scoring refuses to run

#### Scenario: Correction after registration
- **WHEN** the registered protocol needs a correction
- **THEN** the pre-registration file is unchanged and the correction appears in the results document with its reason

### Requirement: Validation is the only arbiter
KEEP or DROP for any gold cell SHALL be decided on validation [70:80 %] only, at its
family's alpha. Every validation verdict SHALL be appended to that family's registry
with its point estimate, CI and alpha. A recorded row SHALL never be rewritten.

#### Scenario: Verdict recorded
- **WHEN** a cell's validation CI excludes zero in its favour at the family alpha
- **THEN** its registry row reads CLEARED and records the CI bounds and alpha

#### Scenario: Verdict not reached
- **WHEN** the CI includes zero
- **THEN** the row reads DROP

### Requirement: The test block is scored exactly once
The test block SHALL be scored only after every validation verdict is committed, in one
run that refits on [0:80 %] and records results as a one-shot report. A second test run
SHALL be refused. Test results SHALL never change a validation verdict.

#### Scenario: Second test run
- **WHEN** a test run is started and a committed one-shot test report already exists
- **THEN** the run refuses and changes nothing

#### Scenario: Test disagrees with validation
- **WHEN** a cell CLEARED on validation but scores at chance on test
- **THEN** both results are reported as recorded and the validation verdict is not edited

### Requirement: The study is isolated from production
The gold study SHALL write only under the gold results folder and the gitignored
research model folder. It SHALL never write under the production models folder. Serving,
training and paper-trading code SHALL never import it. No pinned protected file SHALL
change.

#### Scenario: Full study run
- **WHEN** the complete gold study runs
- **THEN** every protected-set fixture still matches and nothing under the production models folder changed
