# Forward evaluation — runbook

Operating notes for the forward arbiter of the horizon study
(`openspec/changes/horizon-study-forward-evaluation/`). The logger records what
every challenger predicts at every horizon, as each bar closes. Those rows are
the **only** data a horizon claim can be decided on.

## What runs

| Scheduled task | When | Does |
|---|---|---|
| `EURUSDProphet-ForwardLogger` | every 15 min at :01 / :16 / :31 / :46 | predicts the newest closed M15 bar (in session only), H1 bar and day, writes gap records for bars it missed, settles matured forecasts |
| `EURUSDProphet-Refit` | Saturdays 06:00 | refits every model on the **first weekend of the month** only; otherwise it does nothing |

One pass handles all three cadences; the ones with no new bar report
`nothing new`, so the 15-minute schedule only makes H1 and D1 coverage more
robust. A run lock keeps two passes from overlapping.

## The trading session (M15 cells only)

Sub-hourly cells are logged **only inside the owner's trading session**:

| | |
|---|---|
| owner's clock | **15:30 – 23:00 Europe/Sofia**, Monday to Friday |
| bar labels | 14:30 – 22:00 (the labels are Europe/Berlin wall clock, not UTC) |
| New York | 08:30 – 16:00, except the ~5 daylight-saving mismatch weeks |
| eligible trade | the as-of bar **and** the target bar both inside the session, same day |

Outside that window an M15 bar close produces **neither a prediction nor a gap**
— the logger was never going to forecast it, so it is not missing coverage.

Why the window stops at 23:00 Sofia: one hour later the broker's spread
quintuples for the New York 17:00 rollover, and a bid-only series reads a
spurious 72 % "accuracy" there. M15 rows are therefore scored on the **mid**
price built from each bar's own recorded spread, and a bar with no spread is
excluded rather than given a guessed mid.

### Expected volume per session day

| Cadence | Bar closes logged | Rows per close | Rows per day |
|---|---|---|---|
| M15 | 30 (in session) | 12 (2 models × 6 horizons) | ~360 |
| H1 | 24 | 17 | ~408 |
| D1 | 1 | 21 | ~21 |

A session day well below ~360 M15 rows means the machine slept, the terminal was
closed, or the session window moved — check `gaps.csv` and `failures.csv`.
A `ClockMismatch` failure row means the broker's server clock no longer matches
Europe/Berlin: **stop and re-establish the clock before trusting any session
rule**, because every session boundary rests on it.

Both call `scripts/forecast_eval/run_module.cmd`. Their console output is
appended to `research_models/forward_eval/<task name>.log`.

## Install / remove

```powershell
scripts\forecast_eval\install_tasks.ps1      # creates both tasks
scripts\forecast_eval\uninstall_tasks.ps1    # removes them; logged rows stay
schtasks /Query /TN EURUSDProphet-ForwardLogger
```

## Before it can log anything

- **The MT5 terminal must be open and logged in.** The logger attaches
  read-only to whatever account the terminal holds; today that is a real-money
  ActivTrades account. It reads bars, spreads, swap rates and the server name.
  It never records the login, and no code in `src/forecast_eval/` can place,
  change or inspect an order (a test enforces this).
- **Keep AutoTrading off** in the terminal. Nothing here needs it.
- **Disable sleep** on this machine (or move the logger to an always-on host).
  Every hour the machine sleeps is an hour of data that is lost for good:
  missed bars are recorded as gaps and **never** predicted afterwards. During the
  session that now costs four M15 bars an hour, not one H1 bar.
- The study must have produced `results/horizon_study/artifact_manifest.csv`.
  Until then, only the pinned Kronos channels can log.

## Where to look

Open **[`dashboard.html`](dashboard.html)** in a browser. It is a single
self-contained file, rewritten at the end of every logging run, so it is never
more than 15 minutes old (it also refreshes itself every 5 minutes). It shows,
in **your** clock:

- the latest forecast per model and horizon — `+` or `−`, the probability, and
  the window the move is measured over;
- `★` on the cells of the registered family, the only ones that can ever reach a
  verdict;
- the running accuracy per cell, always labelled *interim* with the number of
  forecasts still needed;
- whether the session is open, which phase the logger is in, and any gaps or
  failures.

**It carries no spread, breakeven or net-profit figure.** You account for
trading cost yourself (your decision, 2026-10-02). Those columns stay in
`predictions.csv` and in the pre-registration, because the project's methodology
refuses to produce net figures on an assumed-free spread — they simply decide
nothing and are not put in front of you. A guard in `report.py` fails the write
if such a figure ever reaches the page.

Regenerate it by hand with `python -m src.forecast_eval.report`. It reads the
logs only; it is not part of the dashboard served by `api.py`.

## The files

All of them are append-only. A written row is never changed.

| File | One row per |
|---|---|
| `predictions.csv` | model × horizon × as-of bar: the forecast, model version, price source, server, the bar's spread, swap rates, phase |
| `settlements.csv` | prediction whose exit bar has closed: the realised move, whether the forecast was right, and whether the row is **scorable** |
| `gaps.csv` | bar close the logger missed (machine off, terminal closed) |
| `failures.csv` | model that could not predict at a bar, and why. `*` rows mean MT5 itself was unreachable |
| `refit_failures.csv` | monthly refit that failed. The previous version stays live |
| `dashboard.html` | the operator view, rewritten from the four logs above on every run. Not append-only: it is a rendering, safe to delete |

## Reading it

- **Coverage** = predicted bars ÷ (predicted + gap bars). The power table in
  the pre-registration assumes the coverage measured during the dry run.
  Coverage well below it lengthens every time-to-decision in proportion.
- **`scorable = False`** has a reason in `exclusion_reason`: dry-run phase, a
  price source other than MT5, a server other than the registered one, a missing
  spread (so no mid price), or a sub-hourly trade that would end outside the
  session. Such rows never enter a verdict.
- **`feature code changed since this version was trained`** in `failures.csv`
  means a production feature module was edited. Models refuse to predict on
  shifted features until the next refit trains on the new code.

## Phases

1. **dry run** (`phase = dry_run`): the first two weeks or more. Rows are logged
   and settled but never scorable. The dry run measures real coverage and
   flushes out operational faults before any alpha exists.
2. **scoring**: begins only after `results/forward_eval/PRE_REGISTRATION.md`
   is committed. The logger checks this by itself. Verdicts are issued only at
   the registered sample size. Running figures before that are labelled
   *interim, not adjudicating*.

## If something is wrong

- **Terminal logged into another account or server:** rows keep logging with
  the server name, and rows from a server other than the registered one are
  excluded from scoring. Log back into the registered server.
- **Stop everything:** run `uninstall_tasks.ps1`. Nothing in serving depends on
  these tasks.
