# Spec Delta

## Purpose

Turns hourly and daily bars into Robert Fischer's swing structure, and into causal,
ATR-normalised Fibonacci, time and spiral features that carry over between currency pairs.

## ADDED Requirements

### Requirement: Every feature is causal
Each feature value for an H1 bar labelled `t` SHALL depend only on H1 bars up to and
including `t`, and on D1 bars allowed by the market-data alignment rule. The full feature
set SHALL be identical whether it is computed on the whole history or on a history cut
off at `t`.

#### Scenario: Truncation invariance
- **WHEN** features are computed on the full cache and on the cache truncated at a random label `t`
- **THEN** every row up to and including `t` is identical in both results, column for column

### Requirement: Every price feature is ATR-normalised
Every price distance or amplitude SHALL be divided by Wilder ATR(14) of its own timeframe.
D1 quantities SHALL use ATR_D1 and H1 quantities SHALL use ATR_H1. No raw price level SHALL
be a feature. Each ATR SHALL be causal.

#### Scenario: Scale-free check
- **WHEN** a pair's OHLC is multiplied by a constant and features are recomputed
- **THEN** every feature is unchanged, within floating-point tolerance

### Requirement: The adaptive swing threshold
The swing threshold SHALL be `min_swing = k_atr × ATR(14)` on each timeframe. The factors
`k_atr_d1` and `k_atr_h1` SHALL be set separately in the config.

#### Scenario: Threshold per timeframe
- **WHEN** `k_atr_d1` is changed and `k_atr_h1` is not
- **THEN** only the D1 pivots can change

### Requirement: A swing high is confirmed by Fischer's rules
A swing high SHALL be the highest bar since the last confirmed low, provided it rose at
least `min_swing` above that low. A later bar SHALL confirm it when it closes below the
high bar's low, moves down at least `min_swing`, or retraces at least 38.2 % of the leg.
Swing lows SHALL mirror this.

#### Scenario: Close below the high bar's low
- **WHEN** a leg rose by at least `min_swing` and a later bar closes below the low of the leg's highest bar
- **THEN** that highest bar becomes a confirmed swing high

#### Scenario: Leg too small
- **WHEN** a leg's rise from the last confirmed low is smaller than `min_swing`
- **THEN** no swing high is confirmed, whatever the later bars do

### Requirement: A pivot is recorded at its confirmation bar
A confirmed pivot SHALL keep the index and price of its extreme bar. It SHALL become
visible to features only from the bar that confirmed it. No feature SHALL use a pivot on
any bar before its confirmation bar.

#### Scenario: Delayed visibility
- **WHEN** a high made at bar 100 is confirmed at bar 104
- **THEN** features at bars 100 to 103 do not use it, and features from bar 104 do

### Requirement: Fischer's 3-bar rule
Two consecutive confirmed pivots on the same timeframe SHALL span at least 3 bars, counting
both pivot bars. Their extreme-bar indices SHALL therefore differ by at least 2. A candidate
closer than that SHALL NOT be confirmed. The rule SHALL apply on both H1 and D1. This is the
owner's choice of 2026-11-21.

#### Scenario: Adjacent pivots
- **WHEN** a high's extreme bar directly follows the previous low's extreme bar
- **THEN** that high is not confirmed

#### Scenario: Fast V-turn kept
- **WHEN** a high's extreme bar is 2 bars after the previous low's, and the leg and reversal conditions hold
- **THEN** the high is confirmed

### Requirement: Fischer's 15-bar rule
When at least 15 D1 bars have closed since the last confirmed D1 pivot, the TGD features
SHALL use the highest high and lowest low of the last 15 closed D1 bars, in time order, as
their swing points. This SHALL last until a new pivot is confirmed. No other feature SHALL
use this fallback.

#### Scenario: Quiet market
- **WHEN** 15 D1 bars close with no new confirmed pivot
- **THEN** the TGD features switch to the 15-bar extremes, and the Fibonacci features keep the last confirmed pivots

### Requirement: Fibonacci level features
On each timeframe, with the last two confirmed pivots `P0` and `P1` (`P1` newest), the
family SHALL compute `swing_amp_atr`, `retracement_pct` and `dist_to_fib_382`, `_500` and
`_618`, as `(close − level) / ATR`. Each D1 feature SHALL be joined causally to its H1 bar.

#### Scenario: Retracement of an up-leg
- **WHEN** `P0` is a low at 1.1000, `P1` a high at 1.1100, and the H1 close is 1.1050
- **THEN** `retracement_pct` is 0.50, and `dist_to_fib_500` is 0

### Requirement: Swing duration features
On each timeframe, `swing_bars` SHALL be the newest confirmed leg's length in that
timeframe's bars, from `P0`'s extreme bar to `P1`'s. This lets a model tell fast V-turns
from slow swings. It is the owner's addition of 2026-11-21.

#### Scenario: Fast V-turn
- **WHEN** `P0` sits at D1 index 3055 and `P1` at 3057
- **THEN** `d1_swing_bars` is 2

### Requirement: Extension and price-band features
With the newest leg `P0 → P1`, `dist_to_ext_1618_w3` SHALL measure the distance to
`P1 + s·1.618·|P1 − P0|`, where `s` is the leg's direction. With the last four pivots
`P0`…`P3`, `dist_to_target2_0618` SHALL use `P3 + s'·0.618·|P3 − P0|`.
`price_band_width_atr` SHALL be the gap between the two levels.

#### Scenario: Not enough pivots
- **WHEN** fewer than four confirmed pivots exist on a timeframe
- **THEN** that row is in the warm-up and is excluded from labels and training

### Requirement: Hourly breakout features
`h1_breakout_n` for `n` in {1, 2, 4} SHALL be +1 if the H1 close is above the highest high
of the previous `n` bars, −1 if it is below their lowest low, and 0 otherwise.
`h1_close_vs_extreme_bar` SHALL be two features: the H1 close against the low of the
leg's highest bar, and against the high of its lowest bar, each in ATR_H1.

#### Scenario: One-bar breakout up
- **WHEN** the H1 close exceeds the previous bar's high
- **THEN** `h1_breakout_1` is +1

### Requirement: The alignment flag
`d1_alignment_flag` SHALL equal `h1_breakout_1` when the H1 close is within the
configured zone width (in ATR_D1) of the D1 0.618 retracement level or the D1 1.618
extension level. Otherwise it SHALL be 0.

#### Scenario: Outside the zones
- **WHEN** the H1 close is farther than the zone width from both D1 levels
- **THEN** `d1_alignment_flag` is 0, whatever the breakout

### Requirement: Risk geometry features
`price_square_low_atr` and `price_square_high_atr` SHALL measure the distance from the H1
close to the lowest low and the highest high of the last 5 H1 bars. `rr_to_50pct_target`
SHALL divide the distance to the last H1 leg's 50 % level by the distance to the opposite
5-bar extreme. That divisor SHALL be floored at 0.1 ATR_H1.

#### Scenario: Zero divisor
- **WHEN** the close equals the opposite 5-bar extreme
- **THEN** `rr_to_50pct_target` uses 0.1 ATR_H1 as its divisor and stays finite

### Requirement: D1 Time Goal Days
From the last two confirmed D1 highs `A`, `B` and the last two confirmed D1 lows, each
projection SHALL be `C = B + 1.618·(B − A)` in D1 bar indices. The features SHALL be
`d1_bars_to_nearest_tgd`, `d1_tgd_cluster_count` (inside ±`tgd_band` days) and
`price_time_confluence`.

#### Scenario: Projection arithmetic
- **WHEN** the last two confirmed D1 highs sit at D1 indices 100 and 110
- **THEN** a TGD is projected at index 126.18, which rounds to 126

#### Scenario: Confluence
- **WHEN** the H1 close is at the D1 0.618 level on the projected TGD day
- **THEN** `price_time_confluence` is 1, its maximum

### Requirement: The logarithmic spiral uses the positive constant
The spiral SHALL be `R(θ) = R0·e^{k·θ}`. The family SHALL build two spiral feature sets:
`spiral306`, with `k = (2/π)·ln φ ≈ 0.306349` on rings 0–2, and `spiral077`, with
`k = ln φ/(2π) ≈ 0.076587` on rings 2–4. A negative or zero `k`, such as the misprinted
`log(1/φ)`, SHALL be rejected.

#### Scenario: Ring choice from coverage
- **WHEN** `k = 0.306349` is used on rings 2–4
- **THEN** the distance is capped on about 98 % of rows, which is why the owner chose rings 0–2 for that constant on 2026-11-21

#### Scenario: Wrong constant refused
- **WHEN** the config sets `k` to a negative value
- **THEN** feature building stops with an error naming the expected positive constants

### Requirement: The spiral is fixed in scale
Coordinates SHALL be `x = (t − t_B)/H` and `y = (P − y_B)/V` around the pivot `B` of the
last confirmed D1 A–B–C swing. `H` SHALL be a fixed number of D1 bars (default 5). `V`
SHALL be ATR_D1 at `t_B`. Time SHALL be counted in trading days, with H1 bars as
fractions of a day.

#### Scenario: Same geometry on both pairs
- **WHEN** two pairs have swings of the same shape in ATR and day units
- **THEN** their spiral features are identical

### Requirement: Spiral features
For each spiral set and each direction (`ccw` = +1, `cw` = −1), the family SHALL compute
`<set>_dist_<dir>_atr`: the distance from the H1 close to the nearest branch of that set's
rings. Ring `n` SHALL be the turn `φ ∈ [2π(n−1), 2πn)` from the start, so ring 0 lies just
inside it. `<set>_ring_id_<dir>` and `<set>_crossed_<dir>` SHALL name the ring and flag a
crossing.

#### Scenario: No branch crosses
- **WHEN** no ring 2–4 branch exists at the current `x`
- **THEN** the distance is the configured cap, and that set's ring id is 0

### Requirement: The feature list is fixed by config
The ordered feature list SHALL be declared in the config, and its digest SHALL be recorded
with the verdicts. A build whose columns differ from the declared list SHALL fail.

#### Scenario: Column drift
- **WHEN** code adds a column not in the declared list
- **THEN** the build fails and names the extra column
