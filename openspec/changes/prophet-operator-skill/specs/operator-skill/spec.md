# Spec Delta

## Purpose

Lets the owner ask Claude, in Bulgarian or English, how EUR/USD Prophet is
doing, or to start or diagnose it, and get one honest report built without
changing anything in the system being reported on.

## ADDED Requirements

### Requirement: The skill loads on operating requests
The repository SHALL contain a Claude Code project skill named `prophet`. Claude SHALL load it when the owner, in Bulgarian or English, asks how the app is doing, asks to start, open or run it, or asks why it is not working. The built-in `/run` skill SHALL find it as the project's launch skill.

#### Scenario: Status question in Bulgarian
- **WHEN** the owner writes "как е Prophet днес?"
- **THEN** Claude loads the `prophet` skill before answering

#### Scenario: Start request in English
- **WHEN** the owner writes "start the app" or runs `/run`
- **THEN** Claude uses the `prophet` skill's launch instructions instead of guessing a command

### Requirement: The skill stays out of code and research work
The skill SHALL NOT load for requests to change code, add a feature or a hypothesis, retrain, or analyse research results.

#### Scenario: Feature request
- **WHEN** the owner asks to add a new feature to `src/features.py`
- **THEN** the `prophet` skill is not loaded

### Requirement: The status pass changes nothing
The status pass SHALL make only GET requests, and only to `/api/home`, `/api/provenance` and `/api/gold/health`. It SHALL write no file anywhere in the repository. It SHALL NOT call:
- `/api/paper-trading`, which rewrites the ledger CSVs;
- `/api/retrain/status`, which can hot-reload the models;
- `/api/predict` or `/api/retrain`;
- any page route.

#### Scenario: Repository unchanged
- **WHEN** the status pass runs against the live app
- **THEN** `git status --porcelain` and the modification times of every file under `results/` and `models/` are the same before and after

#### Scenario: Request allowlist
- **WHEN** the status pass runs
- **THEN** every HTTP request it makes is a GET to one of the three allowed paths

### Requirement: One report with a level per signal
The status pass SHALL print one JSON document. Every entry SHALL have an id, a level and a one-line summary. A health signal's level SHALL be `green`, `yellow`, `red` or `unknown`. The document SHALL carry an overall level, which is the worst health level present.

#### Scenario: Mixed signals
- **WHEN** one signal is `red` and the others are `green`
- **THEN** the overall level is `red`

#### Scenario: A source cannot be read
- **WHEN** one source raises an error or times out
- **THEN** that signal is `unknown` with the reason, and every other signal is still reported

### Requirement: The pass completes when the app is down
When nothing answers on `127.0.0.1:8000`, the status pass SHALL still complete. Every signal that needs the server SHALL be `unknown`. The signals read from files, scheduled tasks and the process list SHALL still be reported. The pass SHALL NOT start the server.

#### Scenario: Server stopped
- **WHEN** the server is not running and the status pass runs
- **THEN** the server signal is `red`, the forecast, provenance and gold signals are `unknown`, the scheduled-task signal is reported normally, and no server process is started

### Requirement: Server process signal
The server signal SHALL be:
- `red` when nothing answers on port 8000, or when a process the launcher does not own holds the port;
- `yellow` when the project's server is running but `api.py` or any `src/**/*.py` file is newer than its start time;
- `green` otherwise.

The project's server is identified the same way as by the launcher.

#### Scenario: Stale code
- **WHEN** `src/home_summary.py` was saved after the running server started
- **THEN** the server signal is `yellow` and says that the launcher will restart it

### Requirement: Forecast freshness signal
The forecast signal SHALL be:
- `red` when `/api/home` reports no forecast;
- `yellow` when it marks the latest forecast as outdated;
- `green` otherwise.

The summary SHALL name the date the forecast is for.

#### Scenario: No forecast today
- **WHEN** `/api/home` returns `tomorrow.outdated: true` for 2026-10-28 on 2026-10-29
- **THEN** the forecast signal is `yellow` and names 2026-10-28

### Requirement: Provenance signal
The provenance signal SHALL be `red` when `/api/provenance` reports served `models/` artifacts that differ from the declared set. The summary SHALL give their number and the oldest modification time. Otherwise the signal SHALL be `green`.

#### Scenario: Undeclared artifacts
- **WHEN** `/api/provenance` returns `undeclared: true` with 7 diverged paths, the oldest modified 2026-10-14
- **THEN** the provenance signal is `red` and its summary names 7 artifacts and 2026-10-14

### Requirement: Gold logger signal
The gold signal SHALL be:
- `red` when `/api/gold/health` reports `stalled`;
- `yellow` when it is not stalled but missed hours in the last 24;
- `green` otherwise.

#### Scenario: Stalled logger
- **WHEN** `/api/gold/health` returns `stalled: true` and `missed_24h: 7`
- **THEN** the gold signal is `red` and its summary names 7 missed hours

### Requirement: Retrain signal from files
The retrain signal SHALL be built from `results/retrain_state.json` and the exit marker in `results/retrain.log`, never from the API. It SHALL be:
- `red` when the last run failed, or stopped without a marker while its log has been silent for 300 seconds or more;
- `yellow` while a run is in progress;
- `green` otherwise.

#### Scenario: Completed run
- **WHEN** the log ends with `=== RETRAIN EXIT rc=0 ... at=2026-10-14 08:17:59 ===`
- **THEN** the retrain signal is `green` and names 2026-10-14

#### Scenario: Failed run
- **WHEN** the log ends with a marker carrying a non-zero `rc`
- **THEN** the retrain signal is `red`

### Requirement: Scheduled-task signal
The scheduled-task signal SHALL cover four tasks: `EURUSDProphet-ForwardLogger`, `EURUSDProphet-GoldLogger`, `EURUSDProphet-GoldRetrain` and `EURUSDProphet-Refit`. It SHALL be:
- `red` when any of them is missing or disabled;
- `yellow` when any of them ended its last run with a failure code;
- `green` otherwise.

#### Scenario: Task removed
- **WHEN** `EURUSDProphet-GoldLogger` is not registered
- **THEN** the scheduled-task signal is `red` and names it

### Requirement: Forward-logger signal
The forward-logger signal SHALL be built from `results/forward_eval/`. It SHALL be:
- `red` when the newest logged prediction is more than two hours old while the FX market is open;
- `yellow` when failures were logged in the last 24 hours;
- `green` otherwise.

The summary SHALL give the newest logged time and the number of recent failures.

#### Scenario: MT5 timeouts
- **WHEN** two `MT5Unavailable` failures were logged in the last 24 hours and predictions are current
- **THEN** the forward-logger signal is `yellow` and names two failures

### Requirement: Server-log signal
The server-log signal SHALL be:
- `yellow` when the last 200 lines of `research_models/server/server.log` contain a traceback or an `ERROR` line;
- `unknown` when the file is missing;
- `green` otherwise.

#### Scenario: Traceback in the log
- **WHEN** the log tail contains `Traceback (most recent call last)`
- **THEN** the server-log signal is `yellow` and quotes the exception line

### Requirement: Performance is reported, never graded
Paper-ledger results and the direction hit rate SHALL be reported at level `info` and SHALL NOT count toward the overall level. The paper-ledger results are read from the persisted ledger CSVs, together with when each file was last built.

Each performance entry SHALL give:
- its N;
- the Wilson 95 % interval of its rate;
- whether N is below 30.

#### Scenario: Losing ledger
- **WHEN** the baseline ledger shows 53 positions, a 35.9 % win rate and −464.8 net pips
- **THEN** the entry is `info`, gives N = 53 and the Wilson interval, and the overall level is not changed by it

### Requirement: No trading advice
The skill SHALL NOT recommend, size, time or rank any trade. It SHALL NOT turn a forecast or a ledger into an instruction to buy, sell or hold. When asked for one, it SHALL say that the app's forward ledgers are a simulation and are not a supported edge.

#### Scenario: Owner asks what to trade
- **WHEN** the owner asks "да купя ли EUR/USD днес?"
- **THEN** Claude reports what the app shows and its reliability, and gives no buy or sell instruction

### Requirement: Starting is allowed, stopping needs consent
When asked to start or open the app, the skill SHALL use the project's launcher. It SHALL NOT stop or restart a running server, or start a retrain or a prediction, unless the owner confirms that specific action. It SHALL never stop a process the launcher does not own.

#### Scenario: Start when stopped
- **WHEN** the owner asks to open the app and nothing listens on port 8000
- **THEN** Claude runs the launcher and confirms once `/` answers 200

#### Scenario: Restart on stale code
- **WHEN** the status pass reports stale code
- **THEN** Claude asks before restarting the server, and restarts only after the owner says yes

### Requirement: The answer leads with what is wrong
Claude's reply SHALL be in the language the owner wrote in. It SHALL list `red` signals first, then `yellow`, then a one-line all-clear for the rest. Each problem SHALL name the page or file where the owner can see it. Each problem SHALL also say what fixes it, or that it needs the owner's decision.

#### Scenario: Two problems
- **WHEN** the provenance signal is `red`, the forecast signal is `yellow` and the rest are `green`
- **THEN** the reply opens with provenance and its banner on the dashboard, then the missing forecast and the refresh button on `/`, and ends with one line saying the rest is fine

### Requirement: Project rules are linked, not copied
The skill SHALL NOT restate the rules in `CLAUDE.md` or `ARCHITECTURE_DOCS.md`. Where one applies, it SHALL name the rule and where it lives.

#### Scenario: Undeclared models
- **WHEN** the provenance signal is `red`
- **THEN** the skill points to the `RETRAIN:` rule in `CLAUDE.md` and does not reproduce its text
