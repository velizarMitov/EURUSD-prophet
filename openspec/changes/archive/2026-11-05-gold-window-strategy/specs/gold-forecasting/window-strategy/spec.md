# Spec Delta

## Purpose

Shows, descriptively and net of cost, how a few fixed, model-gated rules would have done
on gold inside the owner's trading window (15:30–23:00 Sofia), with the models refit on the
past only, and never presents the result as a proven edge or a trade instruction.

## ADDED Requirements

### Requirement: A reproducible descriptive report on the seen history
The system SHALL generate `results/gold/window/DESCRIPTIVE.md` from the MT5 caches:
- log return and mean absolute move by Sofia hour, session, weekday and month, per span;
- the share of the total return from the best days;
- next-day absolute move by quintile of today's.

Every section SHALL be labelled descriptive, never evidence, and SHALL name the bid-price
limitation around the daily break.

#### Scenario: Report regenerated
- **WHEN** the report command runs twice on the same caches
- **THEN** both reports are byte-identical apart from the generation timestamp

### Requirement: Exactly four fixed rules
The family SHALL have exactly four rules: `h1_gated_direction`, `continuation`,
`us_open_breakout` and `d1_gated_direction`. Their parameters SHALL be read from the
family's config and SHALL never be searched or tuned. No other variant SHALL be added to
the backtest report.

#### Scenario: Parameters fixed
- **WHEN** the backtest runs
- **THEN** it uses exactly the config's rule parameters, and the report lists them

### Requirement: Every trade opens and closes inside the owner's window
A trade SHALL be entered no earlier than 15:30 Sofia and exited no later than 23:00 Sofia
on the same weekday. That is the label window 14:30–22:00 Berlin, because Sofia is
Berlin + 1 h all year. No position SHALL be held across a rollover. A day missing any
window bar the rule needs SHALL be excluded and counted:
- the M15 bars labelled 14:30–21:45 where M15 is used;
- otherwise, the H1 bars labelled 14:00–21:00.

#### Scenario: Exit time
- **WHEN** a rule is in a position at 23:00 Sofia
- **THEN** it is closed at that bar's close, and no swap is charged

#### Scenario: Gap in the window
- **WHEN** a weekday lacks the H1 bar labelled 18:00 Berlin
- **THEN** that day has no trade for the H1 rules and is listed as excluded

### Requirement: The h1_gated_direction rule
The rule SHALL act at the close of each H1 bar labelled 14:00–20:00 Berlin. It SHALL take
the `h1_lstm` call (up when P(up) > 0.5) for exactly one hour, only when that bar's
`h1_lstm` size forecast is at or above the 80th percentile of the rule's own in-window
forecasts over the previous 250 trading days.

#### Scenario: Gate closed
- **WHEN** the size forecast is below the trailing 80th percentile
- **THEN** no trade is taken for that hour

#### Scenario: Gate open
- **WHEN** the size forecast is at the 85th trailing percentile and P(up) is 0.47
- **THEN** a one-hour short is simulated from that bar's close

### Requirement: The continuation rule
The rule SHALL take the sign of the move from the close of the bar labelled 21:00 Berlin
on the previous trading day to the close of the bar labelled 14:00 Berlin today. It SHALL
hold that sign from 16:00 to 23:00 Sofia, only when the D1 GARCH forecast for today is at
or above its median over the previous 250 trading days. A zero move SHALL mean no trade.

#### Scenario: Up into the window
- **WHEN** the price rose from 23:00 yesterday to 16:00 Sofia today and the GARCH gate is open
- **THEN** a long is simulated from 16:00 to 23:00 Sofia

### Requirement: The us_open_breakout rule
The rule SHALL take the high and low of the M15 bars from 15:30 to 16:30 Sofia. It SHALL
enter in the direction of the first M15 close beyond that range from 16:45 to 20:00 Sofia,
at most once a day, and exit at 23:00 Sofia. It SHALL trade only when the same GARCH gate
as `continuation` is open. It SHALL run only on days with MT5 M15 bars, and the report
SHALL state that span.

#### Scenario: No break
- **WHEN** no M15 close from 16:45 to 20:00 Sofia lies outside the range
- **THEN** the day has no trade

#### Scenario: Break down first
- **WHEN** the first close beyond the range is below its low at 17:15 Sofia
- **THEN** a short is simulated from that close to 23:00 Sofia, and later breaks that day are ignored

### Requirement: The d1_gated_direction rule
The rule SHALL use the `d1_lstm` call from the last D1 bar that closed before the trading
day. It SHALL hold that call from 16:00 to 23:00 Sofia, only when that bar's `d1_lstm` size
forecast is at or above its median over the previous 250 trading days.

#### Scenario: Monday
- **WHEN** the trading day is a Monday
- **THEN** the call comes from the last D1 bar labelled before Monday

### Requirement: Model predictions are walk-forward and complete
`h1_lstm`, `d1_lstm` and the D1 GARCH SHALL predict a calendar year only from fits on data
that ended before that year:
- each is refit once a year on an expanding window;
- the first three calendar years of each cache are fit-only;
- the frozen gold settings are used unchanged.

A neural prediction SHALL exist only when all five seed models of that year are present.

#### Scenario: Year boundary
- **WHEN** predictions for 2020 are produced
- **THEN** every model used was fit only on bars labelled before 2020-01-01

#### Scenario: Missing seed
- **WHEN** one of the five seed models of a year is missing
- **THEN** that year has no neural predictions, and its rules trade nothing that year

### Requirement: Gates use only the past
Every gate threshold SHALL be computed from forecasts dated strictly before the trading
day. A gate SHALL stay closed until 250 prior trading days of forecasts exist.

#### Scenario: Warm-up
- **WHEN** fewer than 250 prior trading days of a rule's forecasts exist
- **THEN** the gate is closed and no trade is taken

### Requirement: Results are net of cost
Prices SHALL be MT5 bid closes. A buy SHALL fill at the bid plus that bar's MT5 spread,
and a sell at the bid. Each round trip SHALL be charged at least 0.0062 % of the entry
price, the broker's window median on 2026-11-04. A stress result SHALL also be reported
at 0.0122 %. Spread and volume SHALL never be a model or gate input.

#### Scenario: Missing spread
- **WHEN** a bar's MT5 spread is 0 (as for 2015–2017)
- **THEN** the trade is charged the 0.0062 % floor

### Requirement: The backtest report is descriptive
The system SHALL write `results/gold/window/backtest.md`. For each rule it SHALL give, net
and at the stress cost:
- the trade count and the span;
- the mean daily return and a 95 % block-bootstrap interval (block 20, 2,000 resamples,
  seed 42);
- the total, maximum drawdown and hit rate;
- a per-year table;
- the same rule with its gate removed;
- an always-long-in-the-window baseline;
- the share of time-shifted gates doing at least as well.

Every section SHALL be labelled "descriptive: rules chosen after seeing these years,
simulated, net of the stated costs". The report SHALL never use the words CLEARED or DROP,
and SHALL write no registry row.

#### Scenario: Labels
- **WHEN** the report is rendered
- **THEN** every section carries the descriptive label, and the text contains no verdict word

### Requirement: Simulated only, never an instruction
The family SHALL contain no order, broker, position-sizing or stop-loss code. Its reports
SHALL describe results as simulated history. They SHALL never say to buy or sell, and
SHALL never name a trade for the current day.

#### Scenario: Report wording
- **WHEN** a backtest report is rendered
- **THEN** it contains no instruction to buy or sell

### Requirement: Other gold files stay untouched
Running the family SHALL leave byte-identical:
- every gold registry and registration;
- `gold_config.json`;
- the gold research models under `research_models/gold/` outside `window/`;
- everything under `models/`.

#### Scenario: After a full run
- **WHEN** the walk-forward and the backtest have run
- **THEN** `git status` shows no change to those files
