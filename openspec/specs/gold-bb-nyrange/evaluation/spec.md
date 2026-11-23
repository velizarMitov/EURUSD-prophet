# gold-bb-nyrange/evaluation Specification
## Purpose

Defines how gold-bb-nyrange is decided: a label and power report before any outcome, a
committed pre-registration that gates every outcome, three registries with seven cells
scored once against Bonferroni bars, and a plain report.

## Requirements
### Requirement: Label and power report before any outcome
A label stage SHALL write `results/gold_bbnr/labels.md` and `labels.json` with event, trade
and eligible-day counts (overall and per year), exclusions by reason including `thin fold`,
distinct dates, and the minimum detectable effect at 80 % power for each cell. It SHALL use no
outcome, no target value of a scored day and no model output.

#### Scenario: Outcomes not read
- **WHEN** the label stage runs
- **THEN** it never calls the Part 1a outcome, the Part 1b exit scan or the Part 2 target for scored days

### Requirement: Declared dispersions for power
The power calculation SHALL use the SD of the 3-day ATR move over all valid days for Part 1a,
a declared 1.0 R for Part 1b, and a declared SD of the per-day squared-error difference fixed
in the config for Part 2.

#### Scenario: Part 1b MDE
- **WHEN** Part 1b has 100 scored trades at alpha 0.025 and 80 % power
- **THEN** its MDE is `(z(0.975) + z(0.8)) × 1.0 / √100` ≈ 0.28 R

### Requirement: Pre-registration gate
A pre-registration stage SHALL write `results/gold_bbnr/PRE_REGISTRATION.md` and `.json`. They
SHALL hold:
- the rules, the cells and the alphas;
- the data window and the config digest;
- the power table from the label report;
- the prior facts.

Every stage that computes an outcome, a target of a scored day or a verdict SHALL first check
that both files are tracked by git, are unchanged since HEAD, and carry the current config
digest. It SHALL refuse to run otherwise.

#### Scenario: Not committed
- **WHEN** `PRE_REGISTRATION.json` exists but is not committed
- **THEN** the walk-forward and scoring stages stop with an error and compute no outcome

#### Scenario: Config changed after registration
- **WHEN** the config digest differs from the one in `PRE_REGISTRATION.json`
- **THEN** every gated stage refuses to run

### Requirement: Part 1a registry
`results/gold_bbnr/gold_bb_hypothesis_log.csv` SHALL hold two cells at alpha 0.05/2:
`gold_bb_rule_mean_reversion` (mean `r` > 0) and `gold_bb_lightgbm_vs_base_rate` (mean
log-loss of the model below that of the training-fold base rate).

#### Scenario: Two rows
- **WHEN** Part 1a is fully scored
- **THEN** its registry has exactly these two rows

### Requirement: Part 1b registry
`results/gold_bbnr/gold_bb_sma_hypothesis_log.csv` SHALL hold two cells at alpha 0.05/2:
`gold_bb_sma_rule_mean_r` (mean R > 0) and `gold_bb_sma_lightgbm_vs_base_rate` (mean
log-loss below the training-fold base rate).

#### Scenario: Two rows
- **WHEN** Part 1b is fully scored
- **THEN** its registry has exactly these two rows

### Requirement: Part 2 registry
`results/gold_bbnr/gold_nyvol_hypothesis_log.csv` SHALL hold three independent cells at alpha
0.05/3, each on mean squared error of `y`: `gold_harx_vs_har`, `gold_lgbm_vs_harx` and
`gold_mlp_vs_harx`. No cell SHALL wait on another.

#### Scenario: MLP scored without a LightGBM KEEP
- **WHEN** `gold_lgbm_vs_harx` is DROP
- **THEN** `gold_mlp_vs_harx` is still scored

### Requirement: Isolation from other registries
The family's cells SHALL never touch the `0.05/9` bar or any gold or fx registry.

#### Scenario: Other registries unchanged
- **WHEN** any cell is scored
- **THEN** every other hypothesis log in the repository is byte-identical before and after

### Requirement: Day-block bootstrap verdicts
Each cell SHALL be decided by a bootstrap of whole business days from the first to the last
scored date, with 10 000 resamples and a fixed seed. Blocks SHALL be 5 days for Part 1a and
Part 2, and 10 days for Part 1b. A "greater" cell SHALL be KEEP when the alpha-quantile of
the bootstrap means is above 0. A "below" cell SHALL be KEEP when the (1 − alpha)-quantile of
the mean difference is below 0. Every other result SHALL be DROP.

#### Scenario: Rule cell straddles zero
- **WHEN** the alpha-quantile of the bootstrap mean R is −0.05
- **THEN** `gold_bb_sma_rule_mean_r` is DROP

### Requirement: Scored once
Each cell SHALL be scored once. A second attempt SHALL raise an error and leave the registry
unchanged. A row SHALL record:
- the statistic, the bound and the sample size;
- the minimum detectable effect, the verdict and the date;
- the config digest;
- a gross-of-cost note.

#### Scenario: Re-scoring refused
- **WHEN** `gold_harx_vs_har` is already in its registry and scoring is run again
- **THEN** it raises an error and the registry file is unchanged

### Requirement: Report contents
`results/gold_bbnr/report.md` SHALL show the seven verdicts and the power table. For Parts 1a
and 1b it SHALL show counts, win rates, mean outcome by year and by side, and exit-reason
shares. For Part 2 it SHALL show each model's MSE relative to HAR, by year.

#### Scenario: Report before scoring
- **WHEN** the report is built before any cell is scored
- **THEN** it shows the power table and "not scored yet", with no outcome

### Requirement: Report wording and determinism
The report SHALL say plainly that only the registry rows are verdicts, that the results are
gross on bid prices, and that none is a trading result. Building it twice SHALL give the same
content apart from its timestamp.

#### Scenario: Two builds
- **WHEN** the report is built twice with no new data
- **THEN** the two files differ only in the generated-at line
