# fx-ny-orb-hot/evaluation Specification

## Purpose

Defines how the fx-ny-orb-hot breakout is decided on history: the power report written
before any outcome, the pre-registration gate, the two cells, their day-block bootstrap,
the one-shot registry and the config guard.

## Requirements

### Requirement: Historical window
The historical test SHALL use every eligible fx_nyvol day of both pairs that has
walk-forward HAR-X values and complete M15 bars, from 2018-08-08 to 2026-08-07. The window
SHALL follow from the data rules alone and be recorded with the verdicts.

#### Scenario: Day outside the M15 data
- **WHEN** an eligible day falls before 2018-08-08
- **THEN** it is not part of the historical test

### Requirement: Power report before any outcome
Before any outcome is computed, the label report SHALL give, per pair and pooled, hot and
non-hot:
- the eligible days, the days with a trade, and the days without one, by reason (no range,
  range too wide, breakout too far, too late, no breakout);
- the minimum detectable mean net R at alpha 0.025 one-sided and 80 % power, with a
  declared SD of 1.5 R.

It SHALL read no exit, price after entry, or result.

#### Scenario: Outcome function never called
- **WHEN** the label report is generated while the function that computes exits raises if called
- **THEN** the report succeeds

### Requirement: Pre-registration gate
`results/fx_ny_orb/PRE_REGISTRATION.md` and `PRE_REGISTRATION.json` SHALL record:
- the rule, the costs and the cells;
- the window and the config digest.

No outcome SHALL be computed and no cell scored until both files are committed to git
unchanged. The scoring SHALL refuse when either differs from its committed version.

#### Scenario: Uncommitted pre-registration
- **WHEN** scoring runs while PRE_REGISTRATION.json is untracked or modified
- **THEN** it refuses and computes no outcome

### Requirement: Two registered cells
The family SHALL register two historical cells at `alpha = 0.025` (0.05/2) in
`results/fx_ny_orb/fx_ny_orb_hypothesis_log.csv`, pooled over both pairs:
- `orb_hot_net_r`: the mean net R of hot-day trades is above zero;
- `orb_hot_minus_cold`: the mean net R of hot-day trades exceeds that of non-hot trades.

They SHALL never touch the `0.05/9` bar or any other registry.

#### Scenario: Other registries unchanged
- **WHEN** both cells are scored
- **THEN** every other registry file is byte-identical before and after

### Requirement: Day-block bootstrap
Both cells SHALL use a one-sided circular block bootstrap over business days:
- 5-day blocks, 10 000 resamples and a fixed seed;
- the same blocks for every daily sum and count;
- a recomputed trade-weighted mean.

For the difference cell, each resample SHALL compute the hot mean minus the non-hot mean
from the same blocks. A cell SHALL be KEEP only when its alpha-quantile is above zero.

#### Scenario: Positive but uncertain
- **WHEN** the hot mean is +0.10 R and the bootstrap alpha-quantile is −0.08
- **THEN** the cell is DROP

### Requirement: One-shot scoring and config guard
Each cell SHALL be scored once: a second scoring SHALL be refused and leave the registry
byte-identical. Every stage SHALL refuse a config whose digest differs from the one
recorded with the pre-registration, an earlier stage or a verdict.

#### Scenario: Changed target multiple
- **WHEN** the target multiple is changed from 0.60 to 0.70 after the pre-registration
- **THEN** the label, simulation and scoring stages refuse to run

### Requirement: Registry content and note
Each row SHALL record:
- the trade and date counts, the window, the statistic, its bound, `mde_80` and the
  verdict;
- the hot and non-hot means;
- the digests and a note.

The note SHALL say three things:
- the result is net of the recorded spread plus 0.4 bps slippage, and not a trading result;
- HAR-X was chosen after its range MSE was seen;
- the breakout outcomes were not seen before the pre-registration.

#### Scenario: Note text
- **WHEN** a row is written
- **THEN** its note names the cost model, the HAR-X selection fact and the pre-registration

### Requirement: Descriptive trade report
The report SHALL give, per pair, hot and non-hot, and per year:
- the trades, the shares of target, stop and forced exit, and the win rate with a Wilson
  interval on dates;
- the mean net R, the pips gross and net, and the profit factor gross and net;
- the number of same-bar conflicts;
- five example trades chosen by a fixed rule.

Only the registry rows SHALL be verdicts. A cell's numbers SHALL appear only after that cell
has its row.

#### Scenario: Before scoring
- **WHEN** the report is built before the cells are scored
- **THEN** it shows the counts and the power table only
