# Design

## Context

The motivation is in `proposal.md` (Why); the required behaviour is in
`specs/operator-skill/spec.md`. This section records only the facts that shape
the design.

- **Project skills live in `.claude/skills/<name>/` and are tracked in git.**
  The six `openspec-*` skills there are tracked. Claude Code lists each skill by
  its frontmatter `description` and loads its body only when it decides that the
  skill applies. The description therefore decides whether the skill is ever
  used.
- **Some endpoints change the system when you read them.**
  - `GET /api/paper-trading` calls `build_all_ledgers`, which rewrites
    `results/paper_trading_log_{baseline,macro,...}.csv`.
  - `GET /api/retrain/status`, after a completed run, replaces the live
    `PredictionService` and invalidates the provenance cache.
  - `GET /history` rewrites `results/prediction_history.html`.
- **Some endpoints are built to be read-only.**
  - `src/home_summary.py` states the rule "NOTHING HERE WRITES OR PREDICTS".
    It builds its reliability block with `build_ledger`, which only reads.
  - `/api/provenance` re-hashes in memory only.
  - `/api/gold/health` is marked READ-ONLY.
- **The retrain state is on disk.** `results/retrain_state.json` holds the pid,
  start, finish and return code. `results/retrain.log` ends with
  `=== RETRAIN EXIT rc=<n> elapsed=<s> at=<local time> ===`. `api.py`
  `_resolve_state` uses these two files plus a 300-second log-silence rule to
  tell "running" from "interrupted".
- **The launcher's ownership rule** (`scripts/launch.vbs`): a `python.exe` whose
  executable is under the repo's `.venv\` and whose command line contains
  `api.py`. Its `CreationDate` is the server's start time. The code is stale when
  `api.py` or any `src/**/*.py` file outside `__pycache__` is newer than that.
  HTML never counts.
- **The environment** is Windows PowerShell 5.1. The console codepage is cp1252,
  so non-ASCII on stdout raises `UnicodeEncodeError` unless
  `PYTHONIOENCODING=utf-8` is set. `.venv\Scripts\python.exe` exists. `psutil`
  is not a dependency.

## Goals / Non-Goals

**Goals:**
- One command gives Claude every health fact as data, so its prose cannot drift
  away from the evidence.
- The pass is safe to run any number of times, including while a retrain is
  running and while the server is down.
- `SKILL.md` stays short. The detailed knowledge sits in references that load
  only when a request needs them.

**Non-Goals:**
- No new app endpoint, page or dashboard tile. The app's own pages stay the
  owner's view.
- No scheduling and no notification. A pass runs only when someone asks.
- No Codex or `.github/skills/` copy. `AGENTS.md` mirrors `CLAUDE.md` and is not
  touched.
- No changes to, or rewrite of, the old Copilot-era `.github/skills/`.

## Decisions

### D1. One skill, `prophet`, with progressive disclosure

The skill has this layout:

```
.claude/skills/prophet/
  SKILL.md                  <= ~120 lines: when to use, routing table, hard limits
  scripts/status.py         the read-only status pass (stdlib only)
  references/launch.md      start, open, restart, port conflicts, server.log
  references/reading-status.md   per-signal meaning, where to look, what fixes it
  evals/evals.json          trigger and behaviour evals for skill-creator
```

`SKILL.md` maps the request to an action:
- status → run the pass, read `reading-status.md`, answer;
- start or open → `launch.md`;
- "it won't start" → `launch.md` diagnosis, and the pass if the server is up;
- trade question → the no-advice rule.

*Alternatives:*
- Three skills (status, launch, diagnose). They share the server-identity and
  port facts, and three descriptions would compete for the same phrasings.
- One long `SKILL.md`. Every status question would then pay for the launch
  material, and the reverse.

### D2. The pass lives in the skill, not in `src/` or behind an endpoint

`status.py` is operator tooling, so it ships with the skill that uses it. The app
never imports it, and it never imports the app. This keeps it working when the
server or its imports are broken. Importing `src.*` would pull in TensorFlow and
the model loaders, and those are exactly the things that fail.

*Alternative:* `GET /api/status` in `api.py`. Rejected for three reasons:
- it cannot report "server down";
- it needs a server restart to change;
- it adds app surface for a Claude-only need.

### D3. Each fact comes from a source that cannot write

| Signal | Source | Why this source |
|---|---|---|
| server | PowerShell `Get-CimInstance Win32_Process` + a plain TCP connect to the port + file mtimes | Same ownership rule as `launch.vbs`. Works with the server down. |
| forecast | `GET /api/home` (`tomorrow`) | Read-only by design; the app owns "outdated". |
| provenance | `GET /api/provenance` | In-memory re-hash only; gives `undeclared_since` and `undeclared_days`. |
| gold | `GET /api/gold/health` | READ-ONLY; the app owns `stalled`. |
| retrain | `results/retrain_state.json` + the marker line in `results/retrain.log` + log mtime | `/api/retrain/status` can hot-reload the models. |
| tasks | PowerShell `Get-ScheduledTask` + `Get-ScheduledTaskInfo` for the four names | No app involvement. |
| forward-logger | `results/forward_eval/predictions.csv` (newest `logged_at`) + `failures.csv` (last 24 h) | Append-only logs written by the task. |
| server-log | last 200 lines of `research_models/server/server.log` | Launcher output. |
| ledgers (info) | `results/paper_trading_log_baseline.csv`, `..._macro.csv` + file mtime | `/api/paper-trading` rewrites these files. |
| reliability (info) | `GET /api/home?part=reliability` | Read-only `build_ledger`; slow, so it gets its own timeout. |

The three allowed paths are a module-level constant. All HTTP goes through one
function that refuses any method other than GET and any path outside the
constant. Tests pin both.

### D4. Thresholds the app owns are taken from the app, not computed again

Where the app already decides, the pass uses its answer:
- `tomorrow.outdated` (weekend-aware) for the forecast;
- `stalled` and `missed_24h` for gold;
- `undeclared` for provenance.

Where no app rule exists, the thresholds are:
- **Retrain:** `api._resolve_state`, mirrored from the files.
  - A marker with `rc=0` is green; a marker with any other `rc` is red.
  - With no marker, a log modified less than 300 s ago means "running" (yellow).
  - With no marker and an older log, the run is "interrupted" (red).
  - With no state file, nothing has run (green).
- **Forward logger:** red when the newest `logged_at` is more than 2 h old while
  the FX market is open.
  - "Open" means Monday 00:00 UTC to Friday 21:00 UTC. This is narrower than the
    real session, so a Sunday evening or late Friday never reads red falsely.
  - Yellow when `failures.csv` has rows in the last 24 h.
- **Tasks:** these result codes are not failures:
  - `0`;
  - `0x41301` (running);
  - `0x41303` (never run);
  - `0x41325` (queued).

  Any other `LastTaskResult` is yellow. A task that is missing or `Disabled` is
  red.

### D5. Health levels and performance never mix

A level answers "is the machine working", never "is the model right".
Performance entries are `info`, with N, the rate, the Wilson 95 % interval and a
`too_early` flag when N is below 30. The Wilson code follows
`home_summary.wilson_lower`. It is not imported, per D2.

The reasons:
- a green ledger would read as "trade it";
- a red ledger would read as "the app is broken", and the owner would try to fix
  a working system;
- the project's rule is that the forward ledgers decide, over months.

### D6. Output: ASCII JSON on stdout; failures are data

`json.dumps(..., ensure_ascii=True)`, so the cp1252 console can never crash the
pass. The document has this shape:

```
{ "checked_at": ISO,
  "overall": level,
  "signals": [ {id, level, summary, details{...}, see} ],
  "performance": [ {id, level: "info", ...} ] }
```

`see` names the page or file where the owner can look, for example
`/advanced (provenance banner)`.

- The exit code is 0 whenever the JSON was printed. It is 2 only on an internal
  crash.
- Each collector is wrapped, so one exception becomes
  `{level: "unknown", summary: "<Type>: <msg>"}`.
- Flags:
  - `--fast` skips the slow reliability call;
  - `--repo <path>` sets the repository root, for tests;
  - `--base-url` sets the server address, for tests.

### D7. One PowerShell call collects every process and task fact

PowerShell takes about 1 s to start. One `powershell -NoProfile -EncodedCommand`
call emits one JSON object with:
- the `python.exe` processes;
- the four tasks' state, last result and last run time.

Ownership follows `launch.vbs` exactly. Ours is a `python.exe` under the repo's
`.venv\` whose command line contains `api.py`. The port is foreign when it
answers and no such process exists. Whether the port answers is a plain TCP
connect, which is not an HTTP request, so the allowlist is untouched.

*Rejected while implementing:* finding the port's owning process with
`Get-NetTCPConnection` or `netstat`. On this machine the tool sandbox hides the
listening socket. Outside the sandbox, the listener's `OwningProcess` reads `0`,
because the venv relaunch hands the socket to a child process. Checking
connections and owners also took the probe from about 2 s to 9 s.

### D8. The description is written to trigger, then measured

The description names the app ("EUR/USD Prophet", "Prophet"). It lists owner
phrasings in Bulgarian and English:
- "как е приложението", "статус", "пусни го", "не тръгва";
- "status", "start", "run", "open", "why isn't it working", "health check".

It says outright that this is the project's launch skill, which is what `/run`
looks for. It also says when not to use it: code changes, features, research.

`evals/evals.json` holds both kinds of case:
- should-trigger prompts: today's real phrasings, plus `/run`;
- should-not-trigger prompts: adding a feature, explaining the volatility
  family, gold pre-registration questions.

The `skill-creator` description loop is run against it. The final pass rates
are recorded in the change's tasks.

### D9. `references/reading-status.md` explains; it does not copy rules

For each signal it gives four things:
- what the level means in practice;
- the normal causes that need no action (a weekend gap, MT5 closed overnight);
- where the owner sees it;
- who acts: the owner, Claude after a yes, or nobody.

Where a project rule applies, it links the rule instead of copying it:
- `RETRAIN:` → `CLAUDE.md` "Production Methodology" item 4;
- ledger interpretation → item 3;
- gold logger → `ARCHITECTURE_DOCS.md §3.12`.

A test fails if signature lines from those documents appear in the skill.

## Risks / Trade-offs

- **[Risk] An endpoint payload changes shape** → the classifier reports
  `unknown` with the missing key and does not crash. The tests pin today's
  payloads, which were captured live on 2026-10-29.
- **[Risk] Someone later adds a writing call to the pass** → the allowlist is
  enforced in code, and a test asserts that `status.py` contains no other API
  path. A second test runs the pass against a temporary repo copy and asserts
  that no file changed.
- **[Risk] Provenance stays red every day until the owner acts** → this is
  intended. The signal is true, and the reply says that it is the owner's
  decision: declare the retrain with a `RETRAIN:` commit, or restore the
  declared artifacts. The skill never does either.
- **[Risk] The reliability call is slow because it fetches realised prices** →
  it gets a 60 s timeout. `--fast` skips it, and SKILL.md uses `--fast` for "is
  it up?" questions.
- **[Trade-off] Thresholds are duplicated outside the app** (retrain 300 s,
  forward 2 h, Wilson). Importing them would break D2. The tests compare the
  copied values with `api._STALL_SECONDS` and `home_summary.MIN_SETTLED` by
  reading those files as text, so any drift fails the build.
- **[Risk] The skill triggers on research questions and adds noise** → the
  description says when not to use it, and the evals include
  should-not-trigger cases.

## Migration Plan

Adding the folder deploys the skill, and the next Claude Code session finds it.
Deleting `.claude/skills/prophet/` and `tests/test_prophet_skill.py` rolls it
back. No app restart, data change or retrain is involved, so the commit does not
touch `models/` and needs no `RETRAIN:` line.

## Open Questions

- Should `CLAUDE.md` get a one-line pointer to the skill? Skills are listed
  automatically, so this is optional. It can be decided after the evals show
  whether discovery needs help.
