# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-ny-orb-hot/*`.

**What exists, all read-only for this family:**
- **fx_nyvol** (`src/fx_nyvol/`, archived change 2026-11-22-fx-ny-volatility).
  - Eligibility, the target and the 14 features.
  - `results/fx_nyvol/oos_lgbm.csv.gz`: walk-forward HAR-X predictions `p_harx` and the
    causal Q75 `hot_threshold`, for 3 475 days per pair from 2013 to 2026-11-20.
- **M15 bars with spread exist only from the M1 parquets.** `results/curl/raw/EURUSD_M1.parquet`
  and `GBPUSD_M1.parquet` run from 2018-08-08 to 2026-08-07.
  - `src/forecast_eval/m15_data.aggregate_m15` builds M15 bars with `spread_price`.
  - `m15_bars()` writes a cache into `research_models/horizon_study/`, which belongs to
    another family. This family calls `aggregate_m15(load_m1(path))` directly and never
    writes that cache.
  - MT5 itself serves M15 only from 2022-11, because its `maxbars` is 100 000.
- **Hot days in the M15 window**, from a read-only count with no outcomes: 230 EURUSD and
  185 GBPUSD (11.2 % and 9.0 % of 2 057 eligible days).
- **Bar sizes.** The median H1 bar in the session is 11.6 bps, against a stop of about
  10 bps. That is why the test uses M15, not H1.
- **The gold chart precedent.**
  - `src/gold/mt5_export.py` writes `Common\Files\EURUSDProphet\gold_signals.tsv`.
  - `mql5/Indicators/ProphetGoldSignals.mq5` draws from it.
  - `scripts/gold/install_mt5_indicator.ps1` compiles it with each terminal's own
    `metaeditor64.exe`.
  - The chart time is the broker's server time, which is the Berlin label clock (CLAUDE.md
    invariant 6), so the label times in the export map straight onto the chart.

## Goals / Non-Goals

**Goals:**
- One pre-registered, net-of-cost test of the owner's New York breakout, gated by the range
  forecast, against 0 and against non-hot days.
- A frozen HAR-X whose forward record decides whether it beats HAR.
- A display-only chart layer showing the opening range, the forecast band and the
  realised range every day, and past simulated trades.

**Non-Goals:**
- Any search over the window, the multiples, the cut-off times or the hot threshold.
- An EA, orders, position sizing, or a current-day trade on the chart.
- Retraining fx_nyvol, or changing any of its files.
- Using LightGBM or the MLP: the owner chose OLS HAR-X.

## Decisions

### D1. One package, two registries

**Layout.** `src/fx_ny_orb/` holds:
- `fx_ny_orb_config.json`;
- the modules `m15`, `rule`, `labels`, `prereg`, `score`, `report`, `frozen`, `live`,
  `export` and `registry`.

**Two registries for two families:**
- **historical (B):** `results/fx_ny_orb/fx_ny_orb_hypothesis_log.csv`, two cells at 0.05/2;
- **forward (A):** `results/fx_ny_orb/forward/fx_ny_orb_forward_hypothesis_log.csv`, two
  cells at 0.05/2.

**Alternative rejected:** one family of four cells at 0.05/4. They answer different
questions on different data at different times. Mixing them would make the historical test
wait for the forward one.

### D2. M15 bars

**Historical.** For each pair, `aggregate_m15(load_m1(results/curl/raw/<PAIR>_M1.parquet))`
gives bid OHLC plus `spread_price`. Mid OHLC is the bid OHLC plus `spread_price / 2`.
- The clock is checked with `m15_data.verify_clock` (Berlin), which refuses on a mismatch.
- A bar whose spread is above 50 pips voids its day.

**Live.** `mt5.copy_rates_from_pos(sym, M15, 1, n)`, starting from position 1, so the
forming bar is never used.
- The spread is the rates field times the point.
- The bars go into `results/fx_ny_orb/live/<PAIR>_m15.csv`, append-only.
- H1 bars of the three pairs go into `live/<PAIR>_h1.csv`, for fx_nyvol's features.

### D3. The rule as one pure function

`rule.simulate_day(m15_day, F, hot, cfg)` returns either no trade with a reason, or a trade
row. Every price and time comes from that day's M15 bars.
- **Order of checks:**
  1. the range (4 bars);
  2. the width ≤ 0.50 F;
  3. the first close outside the range from 15:00;
  4. its distance ≤ 0.25 F;
  5. the entry bar label < 19:00;
  6. the scan from the entry bar through 20:45.
- **The entry bar is scanned.** The entry is at its open, so its own high and low count.
- **Stop fills** at the level or the worse open. The open of the entry bar is the entry, so
  it cannot gap.
- **Costs.**
  - Half the entry bar's spread at entry.
  - Half the exit bar's spread at exit. For a forced exit, the exit bar is the 20:45 bar.
  - Slippage of 0.4 bps × the entry price per trade.
  - R = net / (0.25 F).
- **Why not fixed costs.** The owner proposed a fixed 0.46 bps spread. That is EURUSD's
  median, and GBPUSD's is 0.65 bps. Each bar's own spread is honest for both pairs and
  catches news-hour widening.

**Timing in DST-mismatch weeks.** The 08:30 New York release falls in the bar labelled
13:30 Berlin, which is before the range. The rule stays on fixed Berlin labels, as the
owner's window is fixed in Sofia time. The report breaks down mismatch days.

### D4. Historical hot flag

`p` and the threshold come from fx_nyvol's saved walk-forward values, never refitted.
`F = exp(p_harx) × close14`, where `close14` is the mid close of the bar labelled 14:45
M15. That is the close of the 14:00 H1 bar, so the same instant. The fx_nyvol target uses
the H1 close; the two agree to within the M1-versus-H1 aggregation, and a test checks it.

### D5. Power and pre-registration

**The label report** (`labels.py`) runs the rule only up to the entry decision:
`rule.entries` stops before the exit scan. It counts trades and non-trade reasons per pair,
hot and non-hot. A test makes the exit scan raise and requires the report to succeed.

**The declared SD is 1.5 R.**
- A stop is −1 R and a target +2.4 R. With a win rate of 0.3, the SD of that two-point mix
  is 3.4 × √(0.21) ≈ 1.56.
- Forced exits shrink it.
- With about 250 hot trades, the minimum detectable mean is about 2.80 × 1.5 / √250 ≈
  0.27 R. `labels.md` prints the exact figure before any outcome.

**Pre-registration.** `prereg.py` writes `PRE_REGISTRATION.md` and `.json`:
- the rule, the costs, the cells, the window, the config digest, the power table, and the
  facts about HAR-X selection.

**The git gate.** A file is committed when `git ls-files --error-unmatch` finds it and
`git diff --quiet HEAD -- <file>` reports no change. The simulation of outcomes and both
scorings check that gate first. The owner makes the commit, as always.

### D6. Bootstrap for the two cells

**Per business day d:**
- `S_hot`, `N_hot` for hot trades;
- `S_cold`, `N_cold` for non-hot trades;
- days without trades are zeros.

**Resampling.** The same circular 5-day blocks for all four series, 10 000 resamples, and
a fixed seed.
- Cell 1 is `ΣS_hot* / ΣN_hot*`.
- Cell 2 is that minus `ΣS_cold* / ΣN_cold*`.
- A resample with no hot trade is dropped and counted. If more than 1 % are dropped, the
  scoring refuses.

**Code.** fx_bb's `ratio_block_means` covers cell 1 (read-only). The difference needs four
series on shared draws, so a small new function mirrors its draw order. With `N_cold`
always zero, it must reproduce cell 1 exactly; a test checks that.

### D7. The frozen models and the live passes

**Frozen models.**
- `frozen.fit()` fits the HAR-X and HAR OLS per pair on fx_nyvol's eligible days up to
  2026-11-20.
- It writes `research_models/fx_ny_orb/frozen_<PAIR>.json` and records their SHA-256 in
  `results/fx_ny_orb/forward/frozen_manifest.csv`.

**The forecast pass,** `python -m src.fx_ny_orb.live --pass forecast`.
- It refuses unless the bar labelled 14:00 Berlin has closed, judged by the latest MT5
  server time.
- It tops up the live H1 and M15 caches and builds the fx_nyvol dataset on them.
- It logs, per pair, the eligibility, `p_harx`, `p_har`, `F`, the threshold, the hot flag,
  the range high and low, and `close14`.
- It rebuilds the export.

**The settle pass,** `--pass settle`.
- It refuses unless the bar labelled 21:00 has closed.
- It records the realised `y`, the squared errors, and the day's simulated trade, using
  `rule.simulate_day` on the live M15 bars.
- It rebuilds the export.

**The signal pass,** `--pass signal`. It runs at :01, :16, :31 and :46 from 16:01 to 20:01
Sofia.
- It refuses before the forecast row exists or after the entry cut-off.
- It reads closed M15 bars only (from position 1).
- It runs `rule.entries` on today's bars with the logged `F`.
- It logs the first entry to `forward/signals.csv`, once and append-only, whatever the hot
  flag or the gate.
- It rebuilds the export.

The arrow can appear up to about 2 minutes after the bar opens. That is the latency the
owner accepts. The historical rule assumes an entry exactly at the open.

**The settle pass** also settles each logged signal with `rule.simulate_day` from its logged
entry. It appends the exit, the reason and the net R to `forward/signal_ledger.csv`. A day
with no signal pass before 19:00 is marked `missed` and never backfilled.

**Logs.**
- `results/fx_ny_orb/forward/forecasts.csv`, `signals.csv`, `settlements.csv` and
  `signal_ledger.csv`, all append-only.
- A day already present is never rewritten.

**Scheduled tasks.** `scripts/fx_ny_orb/install_tasks.ps1` registers three tasks, Monday to
Friday, in local (Sofia) time:
- `EURUSDProphet-NYOrbForecast` at 16:05;
- `EURUSDProphet-NYOrbSignal` every 15 minutes from 16:01 to 20:01;
- `EURUSDProphet-NYOrbSettle` at 23:05.

Each pass checks the clock itself, so a wrong local time zone produces a refusal, not a bad
row.

**Forward days** begin on the first trading day after the pre-registration commit date.

### D8. The export and the indicator

**`export.py`** writes `ny_orb.tsv` through a temporary file and `os.replace`. It holds:
- a header block: the status line, the verdicts or "not scored yet", and the text in
  Bulgarian and English;
- one row per pair per day for the last 60 trading days plus today: the range box,
  `close14`, `F`, the hot flag, the realised high and low once settled, and the trade
  fields once settled.

**The gate (owner decision, option 1).** The current day carries signal fields only when
the historical registry has `orb_hot_net_r` = KEEP, today is hot, and `signals.csv` has
today's entry. The export builder enforces this, and tests check all three closed-gate
cases.

**When the gate is open,** the indicator draws:
- an `OBJ_ARROW` up or down at the entry bar and price;
- dashed `OBJ_TREND` lines for the stop and target from the entry to 21:00;
- a label from the export, for example "Сигнал на правилото: long · исторически KEEP ·
  forward 7 сигнала, средно +0.4 R".

**`ProphetNYOrb.mq5`** reads the file every 60 s. It draws only on EURUSD and GBPUSD
charts at M5 to H1, with objects prefixed `NYORB_`:

| Object | What it shows |
|---|---|
| `OBJ_RECTANGLE` (filled, faint) | the opening range, 14:00–15:00 |
| `OBJ_TREND` | the range high and low, 15:00–21:00 |
| `OBJ_TREND` (dashed, red or grey) | the forecast band |
| `OBJ_RECTANGLE` (outline) | the realised range |
| `OBJ_ARROW` | the entry and exit of past simulated trades |
| `OBJ_TREND` | the stop and target of past simulated trades |
| `OBJ_TEXT`, `OBJ_LABEL` | the labels and the status corner |

- **Inputs.** Show non-hot trades (default false), and the language (BG or EN).
- **Housekeeping.** It deletes its own objects on deinit.

**Alternative rejected:** computing the range in MQL5 from the chart's own bars. That would
duplicate the rule in a second language. Python computes everything, and the indicator only
draws, as in the gold design.

### D9. Wording

All text comes from Python and is tested. It contains no order instruction, no lot size and
no promise of profit.
- A live signal is worded "сигнал на правилото / rule signal", with long or short, its
  historical verdict and its forward record.
- Past trade drawings say "симулация / simulation".
- The band label says "размер, не посока / size, not direction".
- The status line names the forward test and its day count.

## Risks / Trade-offs

- **[About 250 hot trades: only about 0.27 R per trade can be detected]** → stated in
  `labels.md` and the pre-registration before any outcome. A DROP means no edge of that
  size.
- **[HAR-X was chosen after its range MSE was seen]** → recorded as a fact in every
  registry note. The breakout outcomes themselves are unseen until the pre-registration is
  committed.
- **[M15 cannot order a target and a stop inside 15 minutes]** → the stop wins, and the
  report counts the conflicts.
- **[The recorded spread may understate the fill spread at news times; 0.4 bps of slippage
  is a guess]** → labelled "net of the recorded spread plus 0.4 bps". This is not a
  trading result.
- **[The live MT5 feed differs from the M1-derived history]** → the forward cells use only
  live bars on both sides of the comparison. The historical cells use only the parquets.
  The two are never mixed in one statistic.
- **[The owner's PC is off at 16:05 or 23:05]** → a missed day is logged as missing at the
  next pass and never backfilled from later bars. That keeps the forward record honest.
- **[Live arrows are trade signals]** → they appear only behind the historical KEEP that
  the owner set as the gate. Every live signal is logged before its outcome and settled in
  a ledger the owner can see on the chart. The ledger is descriptive. Using it to decide
  real trading is a later, separate decision.
- **[Arrow latency of up to 2 minutes against the historical open-price entry]** → the
  settle pass uses the logged entry bar's open, the same as history. The ledger shows any
  drift between the two.

## Migration Plan

Nothing to migrate. Rolling back means:
- running `scripts/fx_ny_orb/uninstall_tasks.ps1` and `uninstall_mt5_indicator.ps1`;
- deleting `src/fx_ny_orb/`, `tests/test_fx_ny_orb_*.py`, `results/fx_ny_orb/`,
  `research_models/fx_ny_orb/` and `mql5/Indicators/ProphetNYOrb.mq5`;
- deleting the doc entries and the exported `ny_orb.tsv`.
