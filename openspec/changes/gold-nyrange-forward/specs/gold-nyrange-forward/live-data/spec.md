# Spec Delta

## Purpose

Defines the live inputs of the gold NY-range forward test: read-only MT5 gold and dollar-index H1
bars in the family's own caches, the real-time label clock, and the dollar-index contract roll.

## ADDED Requirements

### Requirement: Read-only MT5 access
The family SHALL read gold (`GOLD`) and dollar-index H1 bars from the MT5 terminal using
market-data calls only. It SHALL never send, modify or query orders or positions. It SHALL never
write the gold family's caches or any other family's files.

#### Scenario: No trading call
- **WHEN** the family's source is searched for order or position calls
- **THEN** none is found

### Requirement: Own append-only caches
Each pass SHALL append newly closed H1 bars of gold and the dollar index to the family's own caches
under `research_models/gold_nyfwd/`. A bar already cached SHALL never be rewritten. Only closed
bars SHALL be cached, never the bar in progress.

#### Scenario: Repeated top-up
- **WHEN** two passes run within the same hour
- **THEN** the caches hold each closed bar exactly once and no earlier bar changed

### Requirement: Real-time label clock
"Now" for every pass SHALL be real time converted to the Berlin label clock. It SHALL never be the
time of the last MT5 tick. A pass that runs on a Saturday or Sunday of the label clock SHALL log
nothing.

#### Scenario: Stale weekend tick
- **WHEN** a pass runs on Saturday and MT5's last tick is from Friday 22:59
- **THEN** the pass logs nothing and reports `weekend`

### Requirement: Gold clock check
Before a forecast, the pass SHALL check the gold cache with the gold family's weekly-open clock
check. A day whose week fails the check SHALL be logged as `unverified clock week` with no
forecast.

#### Scenario: Shifted week
- **WHEN** the current week's gold weekly open is one hour off the Berlin clock
- **THEN** the day is logged `unverified clock week` without a forecast

### Requirement: Dollar-index contract roll
Each day the family SHALL use the listed `USDInd` contract with the nearest expiry at least two
trading days after the day. It SHALL record that contract with the day's forecast. If no listed
contract qualifies, the day SHALL be logged `no dollar data` with no forecast.

#### Scenario: Roll past Dec26
- **WHEN** the day is 2026-12-10 and `USDIndDec26` (expiry 2026-12-11) and `USDIndMar27` are listed
- **THEN** the day uses `USDIndMar27`

#### Scenario: No successor listed
- **WHEN** the day is 2026-12-14 and no `USDInd` contract expiring after it is listed
- **THEN** the day is logged `no dollar data` and no forecast is made

### Requirement: Dollar-index lag check
Before a contract is first used, its cached H1 closes SHALL be checked against EURUSD H1 closes on
the same label hours. The negative correlation of hourly returns SHALL peak at lag 0. A contract
that fails SHALL not be used, and the days it would cover SHALL be logged `no dollar data`.

#### Scenario: One-hour offset
- **WHEN** a new contract's returns best match EURUSD's at a lag of one hour
- **THEN** that contract is refused and those days are logged `no dollar data`

### Requirement: Scale-free dollar feature across a roll
The dollar feature SHALL be computed exactly as in gold_bbnr: the pre-session range of 00:00–14:00
divided by the mean of the previous 20 days' pre-session ranges, missing with fewer than 10 bars.
Ranges from different contracts SHALL be mixed in that mean without adjustment.

#### Scenario: Mean spans the roll
- **WHEN** the previous 20 days include 8 days of `USDIndDec26` and 12 of `USDIndMar27`
- **THEN** the feature uses all 20 days' ranges as recorded, without adjustment
