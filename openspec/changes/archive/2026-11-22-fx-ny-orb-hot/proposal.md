# Proposal

## Why

fx-ny-volatility (2026-11-22) gave the first fx KEEPs, and its descriptive winner is a plain
linear model.
- **The KEEPs.** Pre-session information predicts the size of the New York session
  (16:00–23:00 Sofia) beyond its own history.
- **The winner.** HAR-X, an OLS on the same 14 features, cuts the HAR MSE by 15–20 %.

A size forecast is not a profit. The owner's question is whether a pure New York
opening-range breakout makes money net of cost on the days the model calls hot, and only
those. The owner also wants to see the forecast and the range on the MT5 chart every day.

**The breakout's results on these days have never been looked at.** HAR-X was chosen
after its range MSE was seen, and that is recorded as a fact.

## What Changes

- **A new research family, `fx_ny_orb`, for EURUSD and GBPUSD.** It has its own package
  `src/fx_ny_orb/`, config, registries and results directory. It reuses fx_nyvol's features
  and its walk-forward HAR-X predictions read-only.
- **[B] The breakout rule, fixed now and never searched.**
  - **NY opening range (OR).** The high and low of the four M15 bars labelled
    14:00–14:45 Berlin (15:00–16:00 Sofia).
  - **Hot day.** The HAR-X forecast, available at the close of the bar labelled 14:00
    Berlin, is above the causal Q75 of the previous 60 days.
  - **Entry.** The first M15 close outside the OR after 15:00 Berlin; entry at the next
    bar's open. It requires:
    - OR width ≤ 0.50 × the forecast;
    - the breakout close within 0.25 × the forecast of the broken level;
    - entry before 19:00 Berlin.

    If the first breakout fails a condition, the day has no trade.
  - **Exits.**
    - SL at 0.25 × the forecast and TP at 0.60 × the forecast, from the entry.
    - When one M15 bar touches both, the SL counts.
    - Forced exit at the close of the bar labelled 20:45 Berlin.
    - One trade per day per pair.
  - **Costs.** Half the recorded spread at entry and half at exit, plus a fixed 0.4 bps of
    slippage per trade. Results are in R, where R is the SL distance.
- **The historical test.** It runs on M15 mid bars aggregated from the M1 parquets
  (2018-08-08 → 2026-08-07, both pairs, with spread), read-only. Two cells at
  alpha = 0.05/2, each scored once:
  1. the mean net R on hot days is above 0;
  2. hot days beat non-hot days, which run the same rule as the control.
- **Pre-registration first.** `results/fx_ny_orb/PRE_REGISTRATION.md` and `.json` record
  the rule, the cells and the power. No outcome is computed and nothing is scored until
  they are committed.
- **[A] A frozen HAR-X forward check.**
  - The OLS HAR-X and HAR are fitted once on the fx_nyvol eligible days up to 2026-11-20
    and pinned by digest.
  - A daily task forecasts at 16:0x Sofia and settles at 23:0x Sofia.
  - A forward family of two cells (HAR-X vs HAR per pair, alpha = 0.05/2) is scored once
    at about 270 eligible forward days per pair or on 2028-01-31, whichever comes first.
- **[C] Drawing on the MT5 chart, for display only.** Each daily pass writes
  `Common\Files\EURUSDProphet\ny_orb.tsv`, and the indicator
  `mql5/Indicators/ProphetNYOrb.mq5` draws from it. It draws real lines and shapes:
  - an OR box with lines along its high and low;
  - the forecast band, coloured by the hot flag;
  - the realised session range after settlement;
  - for past days, the simulated trade: entry, SL and TP lines, and exit, coloured by net R.

  **A live entry sign is drawn only after the historical cell `orb_hot_net_r` is KEEP**
  (owner decision, option 1): an arrow with SL and TP lines on hot days when the rule
  enters. Until then, and after a DROP, the current day shows only the range, the forecast
  and the flag. It is installed by a script like the gold one.
- **Intraday signal passes every 15 minutes, 16:00–20:00 Sofia.** They apply the rule live.
  Every live signal is logged whether or not the chart gate is open, and settled into a
  descriptive forward ledger (`signal_ledger.csv`). That record is never backfilled.
- **Data.** The live part reads MT5 H1 and M15 bars read-only into its own small cache,
  and never writes another family's caches.

## Capabilities

### New Capabilities
- `fx-ny-orb-hot/orb-rule`: the opening range, the hot-day gate, the entry conditions,
  the exits and the cost model on M15 mid bars.
- `fx-ny-orb-hot/evaluation`: the power report before any outcome, the pre-registration
  gate, the two historical cells, the day-block bootstrap, the registry and the config
  guard.
- `fx-ny-orb-hot/range-forward`: the frozen HAR-X and HAR, the daily forecast and
  settlement passes, the forward log, and the two forward cells.
- `fx-ny-orb-hot/chart-display`: the TSV export, the MT5 indicator's drawn objects and
  wording, and the installer.

### Modified Capabilities
(none)

## Impact

- **New code and results.**
  - Code in `src/fx_ny_orb/`, tests in `tests/test_fx_ny_orb_*.py`.
  - The indicator `mql5/Indicators/ProphetNYOrb.mq5`, and scripts in `scripts/fx_ny_orb/`
    (indicator install and uninstall, task install and uninstall).
  - Results in `results/fx_ny_orb/`, frozen models in `research_models/fx_ny_orb/`.
- **Outside the repo.** One file, `Common\Files\EURUSDProphet\ny_orb.tsv`, and three
  scheduled tasks: `EURUSDProphet-NYOrbForecast`, `EURUSDProphet-NYOrbSignal` (every 15
  minutes, 16:00–20:00 Sofia) and `EURUSDProphet-NYOrbSettle`. The owner installs them with
  a script. MT5 is used read-only.
- **Read-only inputs.**
  - From fx_nyvol: the features, the target and `oos_lgbm.csv.gz` (walk-forward HAR-X).
  - The M1 parquets in `results/curl/raw/`.
  - The M15 aggregation in `src/forecast_eval/m15_data.py`, used without its cache.
  - From fx_fib: the bootstrap.

  A test pins the configs of fx_nyvol, fx_bb_sma, fx_bb, fx_rates, fx_session and fx_fib.
- **No execution.** The family has no order, broker, sizing or execution code.
  - A live entry sign exists only behind a historical KEEP.
  - It is the owner's chosen exception to the "no signal before a supported edge" practice:
    a historical KEEP is that edge, and the signal ledger keeps a forward record from day
    one.
  - Slippage is modelled at a flat 0.4 bps, and swap does not apply to intraday trades. No
    verdict is a trading result.
- **Docs.** `ARCHITECTURE_DOCS.md §3.20` and a CLAUDE.md invariant entry.
- **Dependencies.** None new.
