# Spec Delta

## Purpose

Tests, once and net of cost, whether a few fixed, model-gated rules make money on gold
inside the owner's trading window (15:30–23:00 Sofia). The test runs on years nobody has
looked at, and the result is reported in a way that never reads as a trade instruction.

## ADDED Requirements

### Requirement: A reproducible descriptive report on the seen history
The system SHALL generate `results/gold/window/DESCRIPTIVE.md` from the MT5 caches:
- log return and mean |move| by Sofia hour, session, weekday and month, per span;
- the share of the total return from the best days;
- next-day |move| by quintile of today's.

Every section SHALL be labelled descriptive, never evidence, and SHALL name the bid-price
limitation around the daily break.

#### Scenario: Report regenerated
- **WHEN** the report command runs twice on the same caches
- **THEN** both reports are byte-identical apart from the generation timestamp

### Requirement: Exactly four cells at a Bonferroni bar
The family `gold_window` SHALL have exactly four cells, `h1_gated_direction`,
`continuation`, `us_open_breakout` and `d1_gated_direction`, at
alpha = 0.05 / 4 = 0.0125. Its rows SHALL go to its own registry. It SHALL never change
another gold registry, registration or config, or the EUR/USD 0.05/9 bar.

#### Scenario: Other registries untouched
- **WHEN** the family is registered and scored
- **THEN** every other gold registry, registration and `gold_config.json` is byte-identical to before

### Requirement: Every trade opens and closes inside the owner's window
A trade SHALL be entered no earlier than 15:30 Sofia and exited no later than 23:00 Sofia
on the same weekday. That is the label window 14:30–22:00 Berlin, because Sofia is
Berlin + 1 h all year. No position SHALL be held across a rollover. A day missing any M15
bar inside the window SHALL be excluded and counted.

#### Scenario: Exit time
- **WHEN** a cell is in a position at 23:00 Sofia
- **THEN** it is closed at that bar's close, and no swap is charged

#### Scenario: Gap in the window
- **WHEN** a weekday lacks the M15 bar labelled 18:00 Berlin
- **THEN** that day has no trade for any cell and is listed as excluded

### Requirement: The h1_gated_direction rule
The cell SHALL act at the close of each H1 bar labelled 14:00–20:00 Berlin. It SHALL take
the `h1_lstm` call (up when P(up) > 0.5) for exactly one hour, only when that bar's
`h1_lstm` size forecast is at or above the 80th percentile of the cell's own in-window
forecasts over the previous 250 trading days.

#### Scenario: Gate closed
- **WHEN** the size forecast is below the trailing 80th percentile
- **THEN** no trade is taken for that hour

#### Scenario: Gate open
- **WHEN** the size forecast is at the 85th trailing percentile and P(up) is 0.47
- **THEN** a one-hour short is simulated from that bar's close

### Requirement: The continuation rule
The cell SHALL take the sign of the mid move from the close of the bar labelled 21:00
Berlin on the previous trading day to the close of the bar labelled 14:00 Berlin today.
It SHALL hold that sign from 16:00 to 23:00 Sofia, only when the D1 GARCH forecast for
today is at or above its median over the previous 250 trading days. A zero move SHALL
mean no trade.

#### Scenario: Up into the window
- **WHEN** the mid rose from 23:00 yesterday to 16:00 Sofia today and the GARCH gate is open
- **THEN** a long is simulated from 16:00 to 23:00 Sofia

### Requirement: The us_open_breakout rule
The cell SHALL take the high and low of M15 mid bars from 15:30 to 16:30 Sofia. It SHALL
enter in the direction of the first M15 close beyond that range from 16:45 to 20:00 Sofia,
at most once a day, and exit at 23:00 Sofia. It SHALL trade only when the same GARCH gate
as `continuation` is open.

#### Scenario: No break
- **WHEN** no M15 close from 16:45 to 20:00 Sofia lies outside the range
- **THEN** the day has no trade

#### Scenario: Break down first
- **WHEN** the first close beyond the range is below its low at 17:15 Sofia
- **THEN** a short is simulated from that close to 23:00 Sofia, and later breaks that day are ignored

### Requirement: The d1_gated_direction rule
The cell SHALL use the `d1_lstm` call from the last D1 bar that closed before the trading
day. It SHALL hold that call from 16:00 to 23:00 Sofia, only when that bar's `d1_lstm` size
forecast is at or above its median over the previous 250 trading days.

#### Scenario: Monday
- **WHEN** the trading day is a Monday
- **THEN** the call comes from the D1 bar of the previous Friday or a later weekend bar, whichever closed last

### Requirement: Model predictions are walk-forward and complete
On the archive, `h1_lstm`, `d1_lstm` and the D1 GARCH SHALL predict a calendar year only
from fits on data that ended before that year:
- each fit is refit once a year on an expanding window;
- the first three calendar years are fit-only;
- the frozen gold settings are used unchanged.

A neural prediction SHALL exist only when all five seed models of that year are present.

#### Scenario: Year boundary
- **WHEN** predictions for 2010 are produced
- **THEN** every model used was fit only on bars labelled before 2010-01-01

#### Scenario: Missing seed
- **WHEN** one of the five seed models of a year is missing
- **THEN** that year has no neural predictions, and its cells trade nothing that year

### Requirement: Gates use only the past
Every gate threshold SHALL be computed from forecasts dated strictly before the
trading day. A gate SHALL stay closed until 250 prior trading days of forecasts exist.

#### Scenario: Warm-up
- **WHEN** fewer than 250 prior trading days of a cell's forecasts exist
- **THEN** the gate is closed and no trade is taken

### Requirement: Results are net of cost
Each simulated trade SHALL fill at the ask when buying and the bid when selling. Its cost
SHALL be the larger of:
- the realised bid–ask cost from the archive;
- 0.0062 % of the price per round trip, the broker's window median on 2026-11-04.

A stress result SHALL also be reported at 0.0122 %. Spread and volume SHALL never be a
model or gate input.

#### Scenario: Archive tighter than the broker
- **WHEN** the archive's round-trip cost on a trade is 0.004 % of price
- **THEN** 0.0062 % is charged

### Requirement: The registration is committed before any archive price is read
The registration SHALL be written as `results/gold/window/PRE_REGISTRATION.md` and
`.json`. It SHALL record:
- the four rules with their parameters, the alpha, the cost rule and the trade minimum;
- the confirmation span and the archive digests;
- the frozen model settings;
- a power statement computed from the seen MT5 span only.

It SHALL refuse to write if archive verification failed. Walk-forward fitting and scoring
SHALL refuse until the registration is committed and clean.

#### Scenario: Fit before commit
- **WHEN** walk-forward fitting starts while the registration is uncommitted
- **THEN** it refuses and reads no archive price

#### Scenario: Write after commit
- **WHEN** the registration writer runs after the files are committed
- **THEN** it refuses and leaves them unchanged

### Requirement: Each cell is scored once on the confirmation span
The confirmation span SHALL run from the first walk-forward year through 2013-07-31.
For each cell:
- the statistic is the mean daily net return over eligible weekdays, with zero on days
  without a trade;
- the interval is two-sided (1 − 0.0125), a moving-block bootstrap with block 20,
  2,000 resamples and seed 42.

A cell SHALL be CLEARED only if the lower bound is above 0. Below 300 trades it SHALL
read "UNDERPOWERED — NO DECISION". Otherwise it SHALL be a DROP. Verdicts SHALL be
appended once and never re-scored.

#### Scenario: Positive but uncertain
- **WHEN** a cell's mean is positive and its interval includes 0
- **THEN** the cell is a DROP

#### Scenario: Second run
- **WHEN** scoring runs again after verdicts exist
- **THEN** nothing is appended and the existing verdicts are reported

### Requirement: The seen-span replay is descriptive and comes after registration
After the registration is committed, the same four rules SHALL be replayed on November
2022 to 2026 with the existing honest gold predictions and MT5 prices. The replay SHALL be
written to `results/gold/window/replay_seen.md`, labelled "already seen, never evidence".
It SHALL never change a verdict.

#### Scenario: Replay before registration
- **WHEN** the replay is requested while the registration is uncommitted
- **THEN** it refuses

### Requirement: Simulated only, never an instruction
The family SHALL contain no order, broker, position-sizing or stop-loss code. Its reports
and any view SHALL describe results as simulated history. They SHALL never say to buy or
sell, and SHALL never name a trade for the current day.

#### Scenario: Report wording
- **WHEN** a confirmation report is rendered
- **THEN** it states "simulated, net of the stated costs" and contains no instruction to buy or sell
