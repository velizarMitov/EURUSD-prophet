# fx-fibonacci/evaluation Specification

## Purpose

Defines how the fx-fibonacci family is evaluated: a chronological walk-forward over the
full history with purge and embargo, whose out-of-sample results decide a small
hypothesis family once, at a fixed corrected alpha.

## Requirements

### Requirement: Expanding walk-forward folds
The history SHALL be split into consecutive evaluation folds of the configured length. Each
fold's training data SHALL be all eligible rows before that fold. The first fold SHALL begin
only after the configured minimum training span. Fold boundaries SHALL be the same calendar
dates for every pair.

#### Scenario: Fold boundaries
- **WHEN** folds are built for EURUSD and GBPUSD
- **THEN** both pairs use the same fold start and end dates

### Requirement: Purge and embargo between slices
A row SHALL be purged from an earlier slice when its label horizon reaches the start of the
next slice. A further embargo of at least `H` bars SHALL separate consecutive slices. This
SHALL hold between fit and validation rows and between validation and evaluation rows.

#### Scenario: Overlap removed
- **WHEN** a fit row's horizon `t+1…t+H` contains the first validation bar
- **THEN** that row is not used for fitting

#### Scenario: Gap size
- **WHEN** the slices are inspected
- **THEN** the last used fit row's horizon ends at least `H` bars before the first validation row

### Requirement: Out-of-sample metrics
Each fold, and the pooled folds, SHALL report:
- log-loss, macro F1, Brier score and expected calibration error over 10 bins;
- a reliability table;
- the class-prior baseline's log-loss, using train-fold class frequencies.

These SHALL be reported for the MLP ensemble, LightGBM and the descriptive variants.

#### Scenario: Baseline always shown
- **WHEN** a model's log-loss is reported
- **THEN** the class-prior log-loss for the same rows is shown beside it

### Requirement: Gross performance from a fixed rule
A fixed rule SHALL act when the predicted class is 1 or 2, holding one position per pair at
a time and exiting at the first barrier or the time limit. From it the report SHALL show
profit factor, Sharpe from daily-summed returns (√252), maximum drawdown, win rate and
expectancy, in percent and gross of cost.

#### Scenario: Labelled gross
- **WHEN** the gross performance table is rendered
- **THEN** it is titled "gross of spread and slippage — not a trading result"

#### Scenario: One position at a time
- **WHEN** a signal arrives while that pair's previous position is still open
- **THEN** the signal is skipped and counted

### Requirement: A separate hypothesis family
The family SHALL keep its own registry at `results/fx_fib/fx_fib_hypothesis_log.csv` with
exactly four cells: for each of EURUSD and GBPUSD, "MLP ensemble beats class prior" and "MLP
ensemble beats LightGBM", on pooled walk-forward out-of-sample log-loss. Alpha SHALL be
`0.05/4`. The family SHALL never touch other families' bars.

#### Scenario: Other bars unchanged
- **WHEN** this family's registry is written
- **THEN** `results/feature_hypothesis_log.csv`, the gold registries and `results/h1_direction_hypothesis_log.csv` are unchanged

### Requirement: Decisions come from this family's walk-forward
Each cell SHALL be decided only from this family's own pooled out-of-sample walk-forward
rows, by owner decision of 2026-11-21. Earlier project results, such as hypothesis #7 or
the H1 direction family's spent block, SHALL NOT block or change a verdict. Each registry
row SHALL list overlapping prior use of its period as a fact.

#### Scenario: Prior result present
- **WHEN** scoring runs for EURUSD
- **THEN** the verdict depends only on this family's out-of-sample rows, and the notes list the overlap with the H1 direction family's 2024-11..2026-07 block

### Requirement: The scoring test
Each cell's test SHALL be a one-sided circular block bootstrap with block length `H` on the
per-row log-loss difference, model minus reference. A cell SHALL be KEEP only when the whole
interval at its alpha is below zero, and DROP otherwise. The bootstrap seed and resample
count SHALL be fixed in the config.

#### Scenario: Clear improvement
- **WHEN** the upper bound of the interval on the mean difference is below zero
- **THEN** the cell is KEEP

#### Scenario: Interval reaches zero
- **WHEN** the interval includes zero
- **THEN** the cell is DROP

### Requirement: Each cell is scored once
Scoring SHALL record the config digest, the feature-list digest and the fold-artifact
digests with each verdict. A cell that already holds a verdict SHALL NOT be rescored. A
changed config SHALL be a new hypothesis that enlarges the family.

#### Scenario: Second scoring refused
- **WHEN** scoring is invoked after the registry already holds a verdict for a cell
- **THEN** it refuses and changes nothing

### Requirement: The walk-forward report
The walk-forward results, the four verdicts and their intervals SHALL be written to
`results/fx_fib/walk_forward_report.md`. A report rebuilt from the same caches and
artifacts SHALL be identical apart from the generation timestamp.

#### Scenario: Reproducible report
- **WHEN** the report is generated twice from the same caches and artifacts
- **THEN** both are identical apart from the generation timestamp

### Requirement: No verdict is a trading result
Every verdict, report and log of this family SHALL state that it is gross of spread and
slippage. The family SHALL contain no broker, order, position-sizing or execution code, and
SHALL feed neither the served API nor the paper-trading ledgers.

#### Scenario: Verdict wording
- **WHEN** a KEEP is recorded
- **THEN** the registry row's notes say it is a gross predictive result, not a trading edge

### Requirement: Production stays untouched
The family SHALL leave unchanged `models/`, `config.json`, `results/pooled_h1/`,
`results/eurusd_h1.csv`, `src/features.py`, `src/inference.py` and `src/paper_trading.py`.
A unit test SHALL assert this.

#### Scenario: Protected set
- **WHEN** the family's full pipeline has run
- **THEN** every protected path has the same digest as before the run
