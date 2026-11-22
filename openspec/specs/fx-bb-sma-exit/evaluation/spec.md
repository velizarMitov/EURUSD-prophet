# fx-bb-sma-exit/evaluation Specification

## Purpose

Defines how the fx-bb-sma-exit family is decided: the power report written before any
model, the rule cell and the LightGBM filter cell, their day-block bootstrap tests, the
walk-forward, and the one-shot registry and config guard.

## Requirements

### Requirement: Power report before any model
Before any model trains, the label report SHALL give, per pair and pooled, overall, from
2015 and per year:
- the first breaches, the trades opened, the breaches skipped because a trade was open,
  the exclusions by reason, and the distinct entry dates;
- the minimum detectable effects of both cells.

It SHALL show no exit reason, exit price or result.

#### Scenario: Report generated before training
- **WHEN** the label report is generated on a checkout with no predictions
- **THEN** it succeeds and contains the trade counts and the power table, and no exit reason or result appears in it

### Requirement: Minimum detectable effects
At alpha 0.025 one-sided and 80 % power:
- Cell 1's minimum detectable mean SHALL be computed with a declared standard deviation of
  1.0 R. A take profit and a stop are worth about ±1 R gross.
- Cell 2's minimum detectable accuracy SHALL be `0.5 + 2.80 · 0.5 / √n`.

Both SHALL be given on the trade count and on the distinct-date count.

#### Scenario: Thousand trades
- **WHEN** there are 1 000 eligible trades on 750 distinct dates
- **THEN** Cell 1's minimum detectable mean reads 0.089 R on trades and 0.102 R on dates

### Requirement: Two registered cells
The family SHALL register exactly two cells at `alpha = 0.025` (0.05/2) in its own
registry: `sma_exit_rule_net_r` (Cell 1) and `sma_exit_lightgbm_vs_base_rate` (Cell 2). It
SHALL never touch the `0.05/9` bar or any other registry.

#### Scenario: Other registries unchanged
- **WHEN** both cells are scored
- **THEN** every other registry file in the repository is byte-identical before and after

### Requirement: Rule cell
Cell 1 SHALL be scored on every eligible trade, from each pair's first complete bands to
the last trade that has exited, with no fitted parameter. Its statistic SHALL be the mean
net R per trade. It SHALL be KEEP only when the alpha-quantile of the bootstrap means is
above zero.

#### Scenario: Small positive mean
- **WHEN** the mean net R is +0.03 and the bootstrap alpha-quantile is −0.04
- **THEN** Cell 1 is DROP

### Requirement: Day-block bootstrap with blocks covering a trade
Both cells SHALL use a one-sided circular block bootstrap over business days, keyed by
entry date: 10-business-day blocks, 10 000 resamples and a fixed seed. Each resample SHALL
draw the same blocks for the daily sums and the daily trade counts and recompute the
trade-weighted mean.

#### Scenario: Equivalence with the project bootstrap
- **WHEN** every business day carries exactly one trade
- **THEN** the bootstrap distribution equals the project's circular block bootstrap of the mean for the same seed and block

### Requirement: Filter cell walk-forward
Cell 2 SHALL use expanding 12-month folds from 2015-01-01. A fold SHALL train only on
trades whose exit date is more than 10 business days before the fold's first date. Early
stopping SHALL use the last 15 % of those trades by entry date, with the same 10-day gap
before it. Every out-of-sample trade SHALL be predicted once.

#### Scenario: Long trade near the fold start
- **WHEN** a trade enters on 2019-12-10 and exits by time stop on 2019-12-24, and a fold starts on 2020-01-01
- **THEN** the trade is not in that fold's training data

### Requirement: Filter cell model and target
Cell 2 SHALL use the ten fx-bb-reversion features, computed at the close of `t`. One
LightGBM model SHALL be pooled over the six pairs, with fixed settings and no class
weights. Its target SHALL be 1 when the net result is above zero and 0 otherwise. The
reference SHALL be the fit slice's win rate.

#### Scenario: Losing after spread
- **WHEN** a trade's gross result is +0.3 pips and its net result is −0.7 pips
- **THEN** its target is 0

### Requirement: Filter cell verdict
Cell 2 SHALL be tested on the per-trade log-loss difference (model minus base rate), with
the same bootstrap. It SHALL be KEEP only when the (1 − alpha)-quantile is below zero.

#### Scenario: Model no better than the base rate
- **WHEN** the model's mean log-loss is above the base rate's
- **THEN** Cell 2 is DROP

### Requirement: One-shot scoring and config guard
Each cell SHALL be scored once: a second scoring SHALL be refused and leave the registry
byte-identical. Every stage SHALL refuse a config whose digest differs from the one
recorded by an earlier stage or verdict.

#### Scenario: Changed time stop
- **WHEN** the time stop is changed from 10 to 15 days after the label report
- **THEN** the walk-forward and scoring stages refuse to run

### Requirement: Registry content
Each row SHALL record:
- the trade and distinct-date counts and the first and last entry dates;
- the statistic, its bound, `mde_80` and the verdict;
- for Cell 1, the bid/ask control mean and a parity flag when it differs from the net mid
  result by more than 0.05 R;
- the digests and a note.

The note SHALL say the result is net of the recorded spread only, and that the events were
seen in fx-bb-reversion.

#### Scenario: Note text
- **WHEN** a row is written
- **THEN** its note names the spread-only cost model, the earlier fx-bb-reversion DROP, and that this is not a trading result

### Requirement: Protected paths
The family SHALL never write under `models/`, `config.json`, any other family's results,
caches or configs, or any other registry. Its frozen artifacts SHALL be kept in a
gitignored research directory with their digests recorded.

#### Scenario: Earlier families pinned
- **WHEN** the family's tests run
- **THEN** the fx_bb, fx_rates, fx_session and fx_fib configs match the digests recorded with their verdicts
