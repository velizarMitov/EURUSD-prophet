# Reading the status pass

`status.py` prints `signals` and `performance`.

- **Signals** answer one question: "is the machine working?". Each is `green`,
  `yellow`, `red` or `unknown`. `overall` is the worst of them, ordered green <
  unknown < yellow < red.
- **Performance** entries are `info` and never take part in `overall`.

When `/api/*` does not answer, the three endpoint signals read `unknown`. Read
`server` first in that case: the cause is usually the server, not the signals.

Each section below says what the level means, what is normal and needs no
action, where the owner can see it, and who acts. "Owner" means that Claude
reports and asks. "Claude, after a yes" means that Claude may do it once the
owner confirms that specific action.

## server

The server signal follows the launcher's own rules. `references/launch.md` maps
each server state to what to do.

- **Normal:** none. If it is red or yellow, there is something to say.
- **Where:** the address bar, and `research_models/server/server.log`.
- **Who acts:** Claude may start a stopped server. Anything that restarts or
  stops one needs a yes.

## forecast

- **What the level means:**
  - **yellow:** the newest daily forecast is outdated. Nobody has pressed the
    refresh button since the last session closed.
  - **red:** there is no forecast at all.
- **Normal:** none. The app's `outdated` flag is already weekend-aware: a Friday
  forecast for Monday is not outdated on Saturday.
- **Where:** the home page `/`, the "tomorrow" tile and its refresh button.
- **Who acts:** the owner. A forecast is made by `POST /api/predict`. That call
  runs every model and appends a row to `results/prediction_log.csv`, and the
  forward ledgers are built from that log. Making one is an entry in the owner's
  forward record, so Claude does not press it on the owner's behalf.
- **Details:** `variants_disagree: true` means that the price-only and macro
  variants pointed different ways. Say so when it is true.

## provenance

- **What the level means:** red. The `models/` artifacts being served differ
  from the ones pinned in `tests/fixtures/*_protected_sha256.json`. Read it as
  "a decision is waiting", not "the app is broken". The app's own wording is
  "undeclared for N days", and `src/artifact_provenance.py` explains why it
  avoids the word "error".
- **Normal:** a fresh retrain. A run from the dashboard's Retrain button rewrites
  `models/` at once. When `undeclared_since` matches the `retrain` signal's date,
  that run is the explanation, so say so.
- **Where:** the provenance banner on `/advanced`, and `GET /api/provenance`. The
  `details.diverged` field lists the paths.
- **Who acts:** the owner, with one of two outcomes:
  - **Declare it.** Re-pin the fixtures and commit with a `RETRAIN:` line. The
    rule is `CLAUDE.md` → *Production Methodology*, item 4.
  - **Restore the declared artifacts** from git.

  Either outcome is its own task. This skill never re-pins, commits or restores.
  The longer it stays red, the more forward-ledger rows come from models that no
  validation result describes. When `undeclared_days` is large, say that plainly.

## gold

- **What the level means:**
  - **red:** the app reports the hourly gold logger as `stalled`. Its rule is
    no successful pass for more than two market hours, or more than three
    missed hours in 24.
  - **yellow:** some hours were missed, but not enough to count as a stall.
- **Normal:** a weekend, when the app does not count market-closed hours. A few
  hours of overnight sleep also show up as missed hours.
- **Where:** the warning on `/gold`. For each pass, `results/gold/forward/health.csv`
  records whether MT5 was reachable (`mt5`). `details.missed_bars` lists the
  hours.
- **Who acts:** the owner, who needs to keep the PC awake and MT5 running. The
  logger does not back-fill the missed hours. Background:
  `ARCHITECTURE_DOCS.md` §3.12 and the `gold-forecasting/logger-health` spec.

## retrain

This signal is read from `results/retrain_state.json` and `results/retrain.log`,
never from `/api/retrain/status`, which can hot-reload the models.

- **What the level means:**
  - **green:** the last run completed. Its date often explains a red
    `provenance`.
  - **yellow:** a run is in progress. Do not restart the server, because that
    would kill the run.
  - **red:** the run failed (non-zero `rc`), or it stopped without its exit
    marker. In that case the artifacts in `models/` may be a partial set.
- **Where:** `results/retrain.log`. Earlier runs are kept in
  `results/retrain_logs/`.
- **Who acts:** the owner, who decides whether to run again. Starting a retrain
  is never part of this skill.

## tasks

The four scheduled tasks: `EURUSDProphet-ForwardLogger` (every 15 min),
`-GoldLogger` (hourly), `-GoldRetrain` and `-Refit` (weekly).

- **What the level means:**
  - **red:** a task is missing or disabled, so its logging has stopped.
  - **yellow:** a task's last run ended with a failure code.
    - `0x800710E0`: Task Scheduler refused to start the task. Usually a
      condition was not met: the PC was asleep, on battery, or the user was
      logged off.
    - `0x1`: the job itself failed. For the Refit task, look in
      `results/forward_eval/refit_failures.csv`.
- **Normal:** `0x41301` (running) and `0x41303` (never run yet) are not failures,
  and the pass already treats them that way.
- **Where:** Task Scheduler. The forward logger's guide is
  `results/forward_eval/RUNBOOK.md` → *If something is wrong*.
- **Who acts:** the owner. Reinstalling a task with `scripts\forecast_eval\` or
  `scripts\gold\` changes the machine, so Claude does it only after a yes.

## forward_logger

- **What the level means:**
  - **red:** nothing has been logged to `results/forward_eval/predictions.csv`
    for more than 2 h on a weekday, between Monday 00:00 and Friday 21:00 UTC.
  - **yellow:** failures were logged in the last 24 h.
- **Normal:** `MT5Unavailable ... IPC timeout`, which means the MT5 terminal was
  closed or not logged in when the task ran. Overnight this is common and costs
  only those passes.
- **Where:** `/forecasts`, `results/forward_eval/failures.csv` and the RUNBOOK.
- **Who acts:** the owner, who keeps MT5 open. Read the failures' text before
  guessing.

## server_log

- **What the level means:** yellow. A traceback or an `ERROR` line is in the last
  200 lines of `research_models/server/server.log`. Usually one request failed
  and the server kept serving.
- **Normal:** the log is append-only and its lines carry no timestamps, so an old
  traceback stays until 200 newer lines push it out. Check the request lines
  around it before calling it current.
- **Who acts:** Claude may read and explain it. A fix is a code change, which
  belongs in a normal change, not in this skill.

## Performance (`info`)

The entries:
- `ledger_<variant>`: the simulated paper-trading ledgers, read from
  `results/paper_trading_log_*.csv`. `rate` is the win rate **net of spread**,
  and `file_built` says when the app last rebuilt the file.
- `direction_hit_rate`: whether the direction was right, **before cost**, which
  is what the home page shows.

How to read them:
- `too_early: true` (N < 30) means no conclusion. Say only "too early".
- **When `wilson_95` contains 0.5,** the result cannot be told apart from a coin.
  Say so.
- **When `wilson_95` lies entirely below or above 0.5,** describe it, and say
  that it is descriptive, not a test. The app's reliability strip uses the same
  wording.
- The ledgers are a simulation. Which variant earns its keep is decided over
  months of forward data. That is `CLAUDE.md` → *Production Methodology*,
  item 3.

What to report:
- In a status reply, give the hit rate in one line.
- Give pips and P&L only when the owner asks about the ledgers or paper trading.
  The home page shows no cost figure by the owner's decision.
- Never turn any of this into buy, sell, hold, size or timing advice.
