# Spec Delta

## Purpose

Defines the live, forward part of fx-ny-orb-hot: a frozen HAR-X range forecast made every
trading day at 16:00 Sofia, its settlement after the session, and the forward test that
decides whether HAR-X beats HAR on data no model has seen.

## ADDED Requirements

### Requirement: Frozen models
HAR-X (an OLS on fx_nyvol's 14 features) and HAR (an OLS on its three history features)
SHALL be fitted once per pair, on fx_nyvol's eligible days up to 2026-11-20. Their
coefficients SHALL be stored with a recorded digest. Every live pass SHALL refuse
coefficients whose digest differs.

#### Scenario: Altered coefficients
- **WHEN** a frozen coefficient file is edited
- **THEN** the next live pass refuses to run

### Requirement: Daily forecast pass
After the close of the bar labelled 14:00 Berlin (16:00 Sofia), a pass SHALL:
- read the H1 and M15 bars of EURUSD, GBPUSD and AUDUSD from MT5, read-only, into the
  family's own cache;
- build fx_nyvol's 14 features for the day;
- log the forecast `p`, `F`, the hot threshold, the hot flag and the opening range per pair.

A day that is not eligible SHALL be logged with its reason and no forecast.

#### Scenario: Forecast logged once
- **WHEN** the forecast pass runs twice on the same day
- **THEN** the log holds one forecast row per pair for that day, unchanged by the second run

### Requirement: Causality of the live forecast
The forecast pass SHALL use only bars labelled 14:00 Berlin or earlier on the day. A pass
that starts before the bar labelled 14:00 has closed SHALL log nothing for the day.

#### Scenario: Early start
- **WHEN** the pass starts at 14:50 Berlin
- **THEN** it logs no forecast for the day

### Requirement: Intraday signal passes
Every 15 minutes, from 15:00 Berlin until the bar labelled 19:00 has opened, a pass SHALL:
- apply the orb-rule to the current day's closed M15 bars from MT5, with the logged
  forecast;
- rebuild the export.

A pass SHALL never use a bar that has not closed. The one exception is the open of the bar
in progress, which is final from its first tick and serves as the entry price.

#### Scenario: Forming bar ignored
- **WHEN** a pass runs at 15:52 Berlin
- **THEN** it uses bars up to the one labelled 15:30 only

### Requirement: Live signal log
When the rule enters, the signal pass SHALL log one live signal per pair per day: the
direction, entry bar, entry price, stop, target and hot flag. It SHALL log it whether or
not the chart gate is open. A logged signal SHALL never change.

#### Scenario: Signal logged once
- **WHEN** the rule enters at the bar labelled 15:45 and later passes run
- **THEN** exactly one signal row exists for that pair and day, unchanged

### Requirement: Live signal ledger
The settlement pass SHALL settle every logged live signal of the day under the orb-rule
exits on the live M15 bars. It SHALL record the exit, the reason and the net R in an
append-only ledger, `results/fx_ny_orb/forward/signal_ledger.csv`. The ledger SHALL be
descriptive. It SHALL never be scored by this change, and it SHALL never be backfilled for
a missed day.

#### Scenario: Missed signal pass
- **WHEN** the PC was off from 15:00 to 18:00 Berlin
- **THEN** no signal is logged for that day and the day is marked missed in the ledger

### Requirement: Settlement pass
After the close of the bar labelled 21:00 Berlin, a pass SHALL record:
- the realised log range of the session;
- the HAR-X and HAR squared errors;
- the simulated trade of the day under the orb-rule, if any, with its exit and net R.

A settled row SHALL never be rewritten.

#### Scenario: Settled once
- **WHEN** the settlement pass runs twice
- **THEN** the second run changes no settled row

### Requirement: Forward cells
A forward family SHALL register two cells at `alpha = 0.025` (0.05/2) in
`results/fx_ny_orb/forward/fx_ny_orb_forward_hypothesis_log.csv`:
`EURUSD_harx_vs_har` and `GBPUSD_harx_vs_har`. Each SHALL test the per-day squared-error
difference with a 5-day block bootstrap and be scored once:
- after 270 settled eligible forward days for that pair; or
- on 2028-01-31, when the pair is labelled `UNDERPOWERED — NO DECISION` if it has fewer.

Forward days SHALL start after the pre-registration commit.

#### Scenario: Early scoring refused
- **WHEN** scoring runs with 150 settled EURUSD forward days before 2028-01-31
- **THEN** it refuses and writes nothing

### Requirement: Gross of cost and separate from production
The forward cells SHALL be gross predictive results. The family SHALL never write under
`models/`, `config.json`, another family's caches or results, or any other registry. It
SHALL never write the production volatility family.

#### Scenario: Protected paths
- **WHEN** both daily passes run
- **THEN** every protected path is byte-identical before and after
