# fx-ny-orb-hot/chart-display Specification

## Purpose

Defines what fx-ny-orb-hot draws on the owner's MT5 chart: the New York opening range, the
forecast session range and the hot flag every day as lines and shapes, the realised range
after the session, and the past simulated trades. A live entry sign appears only after the
rule's historical cell is KEEP. The chart never places an order.

## Requirements

### Requirement: One exported file
Each daily pass SHALL rebuild one file,
`%APPDATA%\MetaQuotes\Terminal\Common\Files\EURUSDProphet\ny_orb.tsv`. It SHALL hold the
last 60 trading days and the current day, per pair, with every price, every time as a
Berlin bar label, and every text shown, in Bulgarian and English. Nothing in the repository
SHALL read the file back.

#### Scenario: Rebuilt atomically
- **WHEN** a pass rebuilds the file
- **THEN** it writes a temporary file and replaces the old one in one step, so the indicator never reads a partial file

### Requirement: Opening range drawn
For each day with an opening range, the indicator SHALL draw a rectangle from 14:00 to
15:00 Berlin, between the range low and high. It SHALL draw horizontal lines at the range
high and low from 15:00 to 21:00 Berlin.

#### Scenario: Today after 15:00
- **WHEN** the chart is open at 15:30 Berlin on a day with an opening range
- **THEN** today's range rectangle and its two lines are visible

### Requirement: Forecast band drawn
For each forecast day, the indicator SHALL draw two dashed lines from 15:00 to 22:00
Berlin, the forecast session, at `close14 − F/2` and `close14 + F/2`. They SHALL be red on a hot day and grey
otherwise, labelled with the forecast size in pips and the hot flag. The label SHALL state
that the band shows size, not direction.

#### Scenario: Hot day band
- **WHEN** today is hot with `F` = 0.0045
- **THEN** a red band 45 pips wide is drawn around the 14:00 close and labelled as a size

### Requirement: Realised range after settlement
After a day is settled, the indicator SHALL draw an outlined rectangle from 15:00 to
22:00 Berlin between the realised mid low and high of the session's H1 bars (labelled
15:00–21:00), the same bars the forecast target uses.

#### Scenario: Forecast versus reality
- **WHEN** a past day is shown
- **THEN** its forecast band and its realised range rectangle are both visible

### Requirement: Past simulated trades
For settled days with a simulated trade, the indicator SHALL draw:
- an entry marker;
- stop and target lines from the entry to the exit;
- an exit marker.

It SHALL colour them green when the net R is positive and red otherwise. Every label SHALL
call it a simulation. Hot-day trades SHALL be drawn by default and non-hot ones only by a
display switch.

#### Scenario: Yesterday's trade
- **WHEN** yesterday had a hot-day trade that hit its stop
- **THEN** its entry, stop and target lines and exit are drawn in red and labelled as a simulation

### Requirement: Live entry sign gated by the historical verdict
For the current day, the export SHALL carry the rule's live signal only when the historical
cell `orb_hot_net_r` has a KEEP row, today is hot and the rule has entered. The signal is
the direction, the entry bar time and price, the stop and the target. The indicator SHALL
then draw an entry arrow (up for long, down for short), and lines for the stop and target
from the entry to 21:00 Berlin. While that cell is DROP or not scored, no current-day entry,
stop or target SHALL be exported or drawn.

#### Scenario: Gate closed
- **WHEN** `orb_hot_net_r` is not scored or is DROP and today's rule enters at 16:15 Berlin
- **THEN** no arrow, stop or target is drawn for today

#### Scenario: Gate open
- **WHEN** `orb_hot_net_r` is KEEP, today is hot, and the rule enters at the bar labelled 15:30 Berlin
- **THEN** within two minutes of that bar opening, an arrow and the stop and target lines are drawn, labelled as the rule's signal with its historical verdict and its forward record

#### Scenario: Non-hot day
- **WHEN** the gate is open but today is not hot
- **THEN** no current-day arrow is drawn

### Requirement: Status line and wording
The indicator SHALL show a status line from the export: the historical verdicts (or "not
scored yet"), the forecast's forward test ("N of about 270 days"), and the live signal
record so far (signals, wins, mean net R). All text SHALL come from the export.
- A live signal SHALL be worded as the rule's signal, with direction "long" or "short".
- Text SHALL never contain an order instruction, a lot size, or a promise of profit.
- Past trades SHALL be worded as simulations.

#### Scenario: Wording check
- **WHEN** the export is built
- **THEN** a test confirms no exported text contains an order instruction, a lot size or a promise of profit, and every live signal names its historical verdict

### Requirement: Installer
A script SHALL install the indicator in every MT5 terminal on the PC, compiling it with
that terminal's own compiler and reporting each result, as the gold installer does. An
uninstall script SHALL remove it. Neither SHALL change anything else in a terminal.

#### Scenario: Install report
- **WHEN** the install script runs
- **THEN** it prints one line per terminal: compiled, or FAILED with the compiler log
