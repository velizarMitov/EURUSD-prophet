# Spec Delta

## Purpose

Defines the fx-bb-sma-exit report: a plain-language summary of how the simulated trades
ended and what they earned, five example trades, and descriptive breakdowns. All of it is
shown only after the matching cell is scored.

## ADDED Requirements

### Requirement: Built from saved inputs after the verdicts
The report SHALL be built only from saved files: the registry, the label report, the
scored trades and the out-of-sample predictions. A cell's results SHALL appear only after
that cell has its registry row. Two builds from the same files SHALL differ only in the
generation line.

#### Scenario: Before scoring
- **WHEN** the report is built before Cell 1 is scored
- **THEN** it shows the trade counts and the power table, and says the results are not shown yet

#### Scenario: Two identical builds
- **WHEN** the report is built twice from the same saved files
- **THEN** the two files differ only in the generation line

### Requirement: Cost statement
The report's first content line SHALL state three things: results are net of the recorded
bar spread only; slippage, swap and commission are not included; and it is not a trading
result.

#### Scenario: Header
- **WHEN** any fx-bb-sma-exit report is generated
- **THEN** its first content line carries that statement

### Requirement: Plain-language trade summary
The report SHALL contain a section written in plain Bulgarian, without statistical jargon.
Its table SHALL give:
- the number of trades;
- the share that ended at the take profit, at the stop loss and at the 10-day time stop;
- the win rate;
- the average win and the average loss in pips;
- the profit factor gross and net of spread;
- the total result in R, gross and net.

#### Scenario: Exit shares add up
- **WHEN** the summary is generated
- **THEN** the take-profit, stop-loss and time-stop shares add up to 100 % of the scored trades

### Requirement: Five example trades
The summary SHALL list five real trades chosen by a fixed rule, never by their result: the
most recent take-profit exit, the most recent stop-loss exit, the most recent time-stop
exit, then the most recent long and the most recent short not already listed. Each SHALL
show the pair, the entry date and price, the take profit, the stop, the exit date and
reason, and the result in pips gross and net.

#### Scenario: Every exit reason shown
- **WHEN** the scored trades include all three exit reasons
- **THEN** the five examples include at least one trade of each reason and at least one long and one short

### Requirement: Pip size and profit factor
Pips SHALL use 0.01 for USDJPY and 0.0001 for the other pairs. The profit factor SHALL be
the sum of winning results divided by the absolute sum of losing results. It SHALL be shown
gross and net.

#### Scenario: USDJPY pips
- **WHEN** a USDJPY short enters at 150.40 and exits at its take profit of 149.90
- **THEN** it is reported as +50 pips gross

### Requirement: Descriptive breakdowns
The report SHALL break the Cell 1 trades down per pair, per year and per direction. For
each group it SHALL give the trades, the exit-reason shares, the win rate with a Wilson
interval on distinct entry dates, the mean net R, and the pips. It SHALL also give:
- the mean holding time;
- the bid/ask control line;
- for Cell 2, log-loss and Brier per fold, a reliability table, and the trades above and
  below a predicted win probability of 0.5.

Only the registry rows are verdicts.

#### Scenario: Breakdown labelled descriptive
- **WHEN** the breakdowns are generated
- **THEN** they are introduced as descriptive and none carries a verdict
