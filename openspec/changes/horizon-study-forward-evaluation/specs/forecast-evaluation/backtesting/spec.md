# Spec Delta

## Purpose

Scores any model's forecasts on historical data with leakage-free splits, realistic
costs and random-walk benchmarks. It quantifies how much of an apparent edge is
overfitting, and never decides whether a model is kept.

## ADDED Requirements

### Requirement: Purged and embargoed walk-forward splits
The harness SHALL produce chronological walk-forward splits for a horizon h. Any
training row whose label window overlaps a test row's label window SHALL be removed
(purge). A gap of at least h bars SHALL follow each test block before training rows
resume (embargo).

#### Scenario: Overlapping labels are purged
- **WHEN** splits are built for h = 24 H1 bars
- **THEN** no training row's [t, t+24] window overlaps any test row's [t, t+24] window

#### Scenario: Embargo scales with the horizon
- **WHEN** splits are built for h = 1 and then for h = 48
- **THEN** the embargo after each test block is at least 1 bar and at least 48 bars respectively

#### Scenario: Training never sees the future
- **WHEN** any walk-forward split is produced
- **THEN** every training row precedes every test row of that split in time

### Requirement: Combinatorial purged cross-validation
The harness SHALL provide combinatorial purged cross-validation (CPCV): the data is
cut into N contiguous groups, every combination of k test groups is evaluated with
purge and embargo applied, and the results are reassembled into backtest paths.
Every observation appears exactly once per path.

#### Scenario: Path count matches the combinatorics
- **WHEN** CPCV runs with N = 6 groups and k = 2 test groups
- **THEN** 15 splits and 5 complete out-of-sample paths are produced

#### Scenario: A path covers every observation once
- **WHEN** any single reassembled path is inspected
- **THEN** each development observation has exactly one out-of-sample forecast in it

### Requirement: Uniqueness weighting for overlapping labels
When labels overlap (h > 1), the harness SHALL weight each training observation by its
average label uniqueness, so that overlapping labels are not counted as independent.
It SHALL report the effective number of independent observations.

#### Scenario: Non-overlapping labels are unweighted
- **WHEN** h = 1 with one label per bar
- **THEN** every uniqueness weight equals 1 and the effective n equals the row count

#### Scenario: Overlap reduces effective n
- **WHEN** h = 24 on hourly bars
- **THEN** the reported effective n is far smaller than the row count, and both numbers are reported

### Requirement: Cost model from measured spreads
Net results SHALL charge a cost on every position change. The cost SHALL come from the
measured median spread for the instrument, and every net figure SHALL also be reported
at the `config.json` round-trip cost as a sensitivity case. Holding an unchanged
position SHALL incur no new spread.

#### Scenario: Gross and net are both reported
- **WHEN** a strategy result is produced
- **THEN** the gross return, the net return at the measured spread, and the net return at the config round trip are all present

#### Scenario: No cost while a position is unchanged
- **WHEN** consecutive signals keep the same direction
- **THEN** no cost is charged for those bars

#### Scenario: Missing spread data stops the run
- **WHEN** no measured spread exists for the instrument
- **THEN** the harness refuses to produce net figures and names the missing input

### Requirement: Random-walk benchmarks
Every evaluation SHALL be reported against three random-walk nulls: a coin-flip
direction forecast (50 %) alongside the train-majority class; a zero-return forecast
(driftless random walk) for return predictions; and a distribution of random-sign
strategies with the same turnover as the model.

#### Scenario: Benchmarks accompany every result
- **WHEN** any model is scored
- **THEN** the report shows the model's result next to all three nulls, computed on the same rows

#### Scenario: Random-sign distribution matches turnover
- **WHEN** the random-sign null is generated for a model that changes position on 30 % of bars
- **THEN** each random strategy changes position on 30 % of bars, and the model's percentile in that distribution is reported

### Requirement: Directional accuracy test
Directional accuracy SHALL be tested with the Pesaran–Timmermann test, and with a
moving-block bootstrap confidence interval whose block length is at least the horizon h.

#### Scenario: Horizon-aware block length
- **WHEN** accuracy is bootstrapped at h = 12
- **THEN** the block length used is at least 12 and is reported

#### Scenario: Too little data is refused
- **WHEN** the scored sample has no more rows than one block length
- **THEN** no interval or p-value is reported, and the result says it was refused

### Requirement: Return forecast test against the random walk
A return forecast SHALL be compared with the zero-return random walk using the
Clark–West test, because the random walk is nested in the model. Diebold–Mariano SHALL
be used only for comparisons between non-nested models.

#### Scenario: Nested comparison uses Clark–West
- **WHEN** a model's return forecast is compared with the zero forecast
- **THEN** the Clark–West statistic and p-value are reported, not Diebold–Mariano

### Requirement: Family-wise error control across cells
When several model × horizon cells are compared, the harness SHALL report p-values
adjusted with Bonferroni and with the Romano–Wolf step-down procedure, so that the
correlation between cells is visible.

#### Scenario: Both adjustments reported
- **WHEN** a set of cells is evaluated together
- **THEN** each cell shows its raw, Bonferroni-adjusted and Romano–Wolf-adjusted p-value

### Requirement: Probability of backtest overfitting
For any set of candidate configurations evaluated on the same data, the harness SHALL
estimate the Probability of Backtest Overfitting (PBO) by combinatorially symmetric
cross-validation, and report it with the set it was computed over.

#### Scenario: PBO reported with its candidate set
- **WHEN** PBO is computed
- **THEN** the report lists every configuration in the candidate set and the PBO value

### Requirement: Deflated Sharpe ratio with a cumulative trial count
Every reported Sharpe ratio SHALL be accompanied by a Deflated Sharpe Ratio. Its trial
count SHALL be the cumulative number of model × horizon × configuration combinations
ever evaluated in this harness, read from a persistent append-only trial log.

#### Scenario: Trials accumulate across runs
- **WHEN** the harness has evaluated 40 combinations in earlier runs and evaluates 10 more
- **THEN** the Deflated Sharpe for the new runs uses a trial count of 50

#### Scenario: Trial log is append-only
- **WHEN** a run finishes
- **THEN** its combinations are appended to the trial log, and no earlier entry is removed or changed

### Requirement: The harness never adjudicates
Harness outputs SHALL be labelled descriptive. The harness SHALL NOT write a KEEP, DROP
or CLEARED verdict to any hypothesis log, and SHALL NOT consume Bonferroni alpha in any
family.

#### Scenario: No verdict is written
- **WHEN** any harness run completes, whatever the results
- **THEN** no hypothesis log in the repository has changed, and every output file carries a descriptive label

### Requirement: Deterministic results
With identical inputs and seeds, the harness SHALL produce identical outputs. Every
output SHALL record the seeds and the date range of data it read.

#### Scenario: Re-run reproduces
- **WHEN** a run is repeated with the same inputs and seeds
- **THEN** every metric in the output is identical, and the output names the seeds and date range used
