# Design

## Context

See proposal.md for why. The constraints that shape the approach:

- **Round one is fixed.** `results/gold/PRE_REGISTRATION.{md,json}` (committed in
  `ab420d3`) and the three registries hold its verdicts. Its test block is spent
  (`e279411`). The forward pre-registration of round-one models (change
  `gold-forecasting`, task 9.1) is still open, and its rule that models stay frozen for
  the registered window must hold.
- **Saved study predictions already exist.** `results/gold/{validation,test}_predictions_{d1,h1}.csv`
  hold the h1/d1 LSTM volatility and the GARCH forecasts for every validation and test
  bar. Range calibration can be studied on validation **without refitting anything**.
- **The forward logs have a fixed schema.** `predictions.csv` and `settlements.csv`
  have 12 fields each. The round-one forward registration and `src/gold/history.py` read
  them. Adding columns would break the append-only CSVs and the frozen comparison.
- **Data sources.** MT5 has `GOLD`, `SILVER` and `Brent` (checked 2026-10-23), but no
  dollar index. FRED is reached through the generic
  `src.macro_data.fetch_fred_feature(series_ids, start, end, combine, cache_path)`,
  which reads no `config.json`. Gold isolation allows importing production code
  read-only. It forbids naming `models/`, `config.json` or `forward_eval`.
- **Logger task settings.** The task was installed with default settings:
  DisallowStartIfOnBatteries is True, and there is no wake. That explains the 8 missed
  overnight hours of 2026-10-21/22.
- **Power rule.** `prereg.direction_family` with `power.edge` 0.03 and power 0.80 gives
  about 2,640 rows at α = 0.025. Changing that edge would loosen the bar, which memory
  `feedback-infeasible-must-not-loosen-the-bar` forbids.

## Goals / Non-Goals

**Goals:**
- Every round-two number is pre-registered, reproducible from saved files, and labelled
  as needing forward confirmation.
- No round-one file, schema or verdict changes.
- The logger keeps running through sleep, and a stall is visible within two hours.

**Non-Goals:**
- A new volatility model. The owner did not choose that direction, so ranges reuse the
  existing frozen LSTM and GARCH forecasts.
- Trading, sizing, stops or targets of any kind.
- Weekly direction. With about 70 validation weeks, the power rule cannot admit it, so
  it is not built.
- Foundation models such as Chronos or TimesFM. The research finds them rarely better
  than a tuned HAR or GARCH, so they are left for a later round.

## Decisions

### D1. Families, cells and alphas (registered in `PRE_REGISTRATION_V2`)

| Family | Cells | α | Metric / baseline |
|---|---|---|---|
| range | `h1_range`, `d1_range` | 0.05/2 = 0.025 | Winkler interval score at 80 %, against a GARCH-based static range on the same rows. The 80 % coverage tolerance is ±3 points. The 95 % level is reported |
| macro | `d1_lstm_macro:vol`, `d1_lstm_macro:ret`, `d1_gbm_macro:ret`, `d1_range_macro` | 0.05/4 = 0.0125 | paired error difference for vol and range, Clark–West for return, each against the same model without macro, on the same rows |
| direction2 | `h1_selective`, `h1_barrier`, `h4_dir`, and `d1_*_macro:dir` as UNDERPOWERED | from the power fixed point, as in round one | covered-bar accuracy against the training majority and a coin; both CIs must exclude 0 |

- Bootstrap: a moving-block bootstrap with `n_boot` 2000, reusing round one's
  `study.score_*` helpers where they fit.
- The registration lists:
  - for each cell, the validation count and the required count;
  - the statement that validation was seen in round one;
  - the expected outcomes: range likely to clear on H1, macro unlikely to clear, and
    direction2 unlikely to clear.

Alternative considered: *re-scoring round-one cells with new settings.* That would be
data snooping on cells with recorded verdicts. Rejected.

### D2. Ranges: σ from the frozen volatility forecast, adaptive conformal calibration

The models forecast `E|r|` in percent. With `σ = E|r| · √(π/2)`, each standardised
error is `z_t = r_t / σ_t`. The range is `close · exp(±q_t · σ_t / 100)`, centred on
the as-of close, since return has no proven skill.

`q_t` uses adaptive conformal inference (Gibbs & Candès), applied per level:
- the miscoverage target is `α_t+1 = α_t + γ(α − err_t)`, with γ = 0.005;
- `q_t` is the `(1 − α_t)` quantile of |z| over the previous W settled bars, where W is
  500 for H1 and 250 for D1.

There are three warm-ups:
- **Validation:** the first W validation rows warm up and are not scored. The GARCH
  baseline (`q` = normal quantile, σ from GARCH) is scored on the same rows.
- **Live:** the logger seeds the window from the newest W saved study rows of the same
  frozen model's fit. The final refit is on 100 %, so seed rows are in-sample for the
  live model. This is stated in RUNBOOK and affects only the first W live ranges.

Alternative considered: *a static Gaussian q.* Gold's |z| has fat tails, so a static q
under-covers in stress. ACI tracks that. It stays in the study as a registered
comparison row, not a cell.

### D3. New forward files, not new columns

- **New files:**
  - `ranges.csv`: key, level, lo, hi, q, α_t, and on settlement `inside`;
  - `predictions_v2.csv`: the macro, direction2 and shadow-version forecasts, with
    `version` and `shadow` fields;
  - `settlements_v2.csv`;
  - `alerts.csv`;
  - `health.csv`.
- **Round-one files** (`predictions.csv`, `settlements.csv`, `gaps.csv`) keep their
  schema and writers byte for byte.
- **`src/gold/history.py`** gains the v2 files as an extra source, `live_v2`, without
  changing its round-one behaviour.

Alternative considered: *adding columns to `predictions.csv`.* That breaks append-only
CSV readers and the frozen round-one comparison. Rejected.

### D4. Macro data: `src/gold/macro.py`

- **FRED:** DFII10, DTWEXBGS and VIXCLS come through `fetch_fred_feature`, with
  identity combine functions and caches `results/gold/macro/{dfii10,dtwexbgs,vixcls}.csv`.
- **MT5:** `SILVER` and `Brent` D1 bars come through the same append-only cache
  discipline as `src/gold/data.py`, factored into a shared helper, with caches
  `results/gold/macro/{silver,brent}_d1.csv`.
- **Alignment:** for the gold D1 bar labelled day d, each FRED value is the newest one
  dated ≤ the previous US business day, and each MT5 value is the newest bar labelled
  ≤ d. Forward-fill only.
- **Features**, all scale-free and re-checked by `assert_scale_free`:
  - the real-yield change over 1 and 5 days;
  - the dollar log-return over 1 and 5 days;
  - the VIX level and its 5-day change;
  - silver and Brent log-returns over 1 and 5 days;
  - the gold/silver ratio as a 60-day z-score.
- **Variants:** `d1_lstm_macro` and `d1_gbm_macro` append these to the 22 D1 features,
  with the same hyperparameters as round one. No new tuning, so no new trials beyond
  the registered cells.

### D5. Direction2: `src/gold/direction2.py`

- **`h1_selective`.** The round-one GBM classifier gets a confidence threshold τ on
  |p − 0.5|.
  - τ is chosen on inner walk-forward folds of the training block as the largest τ that
    keeps coverage ≥ the registered floor.
  - The floor is 0.45, chosen so that 0.45 × 6,392 ≈ 2,876 covered rows, which is at or
    above the power requirement. If expected coverage falls below what the power rule
    needs, the cell is UNDERPOWERED before scoring.
- **`h1_barrier`.** k = 6 hours, and the barriers sit at ±1.0 × the EWMA(20) volatility
  of hourly returns at the as-of bar. EWMA is causal, which avoids using a fitted
  volatility model inside the label.
  - The label is up, down or time-out. Time-outs count as abstentions.
  - The classifier is the round-one GBM grid, re-tuned on the training block only.
  - Training rows whose 6-hour window crosses the training end are purged.
- **`h4_dir`.** These are non-overlapping 4-hour windows starting on label hours 0, 4,
  8, 12, 16 and 20. A window with any missing hour is ineligible. It uses the H1
  feature set at the window's last hour and the GBM classifier. With about 1,600
  eligible validation windows, the power rule is expected to mark it UNDERPOWERED, and
  the code records whatever the table says.

### D6. Leaderboard: `src/gold/leaderboard.py`

It is built from `history.live()` plus `live_v2`, `ranges.csv` and `alerts.csv`, with
one row per (cell, digest). It reuses `history.wilson` and the shared-bar baseline
logic. The endpoint `GET /api/gold/leaderboard` is additive in `api.py` with a lazy
import, and the gold isolation test allowlist gains `src.gold.leaderboard` and
`src.gold.health`.

### D7. Logger health and task settings

- **Task settings.** `install_task.ps1` uses `New-ScheduledTaskSettingsSet` with:
  - `-WakeToRun` and `-StartWhenAvailable`;
  - `-AllowStartIfOnBatteries` and `-DontStopIfGoingOnBatteries`;
  - `-MultipleInstances IgnoreNew` and `-ExecutionTimeLimit 00:20`.

  It reads the active power plan's "Allow wake timers" setting with `powercfg /q` and
  prints it. It never changes the power plan.
- **Health rows.** `forward_logger.run_once` is wrapped so that every pass, including
  failures, appends a `health.csv` row.
- **Market hours.** These come from `src/gold/clock.py`'s weekly-open logic. Gold trades
  from Sunday's weekly open to Friday 23:00 on the label clock, with a daily one-hour
  break.
- **The stall rule.** It lives in `src/gold/health.py` and is unit-tested with fixed
  clocks.

### D8. Alerts: `src/gold/alerts.py`

- **Threshold.** `python -m src.gold.alerts --calibrate` computes the 90th percentile of
  the frozen h1_lstm's forecasts on its own fit rows (in-sample by design: it defines
  "unusually large for this model"). It writes the percentile and value to
  `gold_config.json` under `alerts`. The logger refuses to alert without them.
- **Notification.**
  - The logger runs a PowerShell one-liner through `subprocess` with a 10 s timeout,
    using `Windows.UI.Notifications.ToastNotificationManager`, with the BG text first
    and then the EN text. No new dependency.
  - A failed notification is recorded in `health.csv`.
  - The notification is deduplicated by as-of bar.
- **Page.** It shows a banner while the alert's target hour is open, through
  `/api/gold` gaining an `alert` block (additive field).

### D9. Versions: `src/gold/versions.py` and a monthly task

- **The refit.** `EURUSDProphet-GoldRetrain` runs monthly, on the 1st at 02:30, which
  is outside market hours. It calls the study's `final` path with an output root of
  `research_models/gold/versions/<YYYYMMDD>/` and writes `manifest.csv` there.
- **Shadow forecasting.**
  - `research_models/gold/versions/active.json` names at most one active shadow
    version, the newest. Older ones are retired, which keeps hourly compute bounded.
  - The logger forecasts with it into `predictions_v2.csv` with `shadow=1`.
- **Promotion.** `python -m src.gold.versions promote <id> --reason "..."`:
  - checks the forward window end from `results/gold/forward/PRE_REGISTRATION.json` and
    refuses while it is open;
  - copies the version's artifacts into the registered layout and rewrites
    `artifact_manifest.csv`;
  - records the old and new digests and the reason in `RESULTS.md`;
  - starts a new window by marking the forward registration superseded. A new forward
    registration must then be committed before scoring, following round one's rule.
- **Lock.** The refit uses its own `research_models/gold/retrain.lock`.

## Risks / Trade-offs

- **[Validation reuse inflates round-two results]** → The reuse is disclosed. There are
  new families with their own Bonferroni bars, no new tuning on validation, and the
  test block stays spent. Every round-two verdict reads "needs forward confirmation".
- **[The range warm-up seeds live calibration with in-sample residuals]** → This
  affects only the first W live ranges, is stated in the RUNBOOK, and ACI corrects it
  online. Live coverage is tracked from the first row.
- **[FRED publishes late or skips US holidays]** → The one-business-day lag plus
  forward-fill handles holidays. A missing value at as-of time makes the macro cells
  record a gap, never a zero.
- **[A wake timer is disabled by the power plan or by IT policy]** → The installer
  prints the state, and the stall banner surfaces the result either way.
- **[Two model sets double the hourly compute]** → Only one shadow version is active at
  a time. LSTM inference on one bar is under a second per seed.
- **[Toast notifications differ across Windows builds]** → A failure is logged and
  never blocks the pass. The page banner is the reliable channel.
- **[Copilot auto-commits half-finished work (memory: project-copilot-auto-commits)]**
  → Each task group ends with the suite passing. Ask the owner to turn off auto-commit
  before applying.

## Migration Plan

1. Land groups 1–4 (registration, data, ranges, macro and direction2 study code). Commit
   `PRE_REGISTRATION_V2` on its own, then run the validation stage once.
2. Land the logger extensions, health and alerts. Reinstall the task with the new
   settings using `scripts/gold/install_task.ps1`.
3. Land the leaderboard and versions, then install the monthly retrain task.
4. Rollback: uninstall the retrain task and reinstall the logger task. The new forward
   files can be left in place. Round-one files are untouched, so nothing needs to be
   restored.
