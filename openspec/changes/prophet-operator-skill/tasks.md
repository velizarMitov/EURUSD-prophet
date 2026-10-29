# Tasks

## 1. The status pass: skeleton and the read-only guard

- [x] 1.1 Create `.claude/skills/prophet/scripts/status.py` (stdlib only).
  - Add `--fast`, `--repo` and `--base-url` flags, and a module-level
    `ALLOWED_PATHS` holding `/api/home`, `/api/provenance` and `/api/gold/health`.
  - Route all HTTP through one GET-only fetch function that refuses any other
    method or path.
  - Add a collector wrapper that turns an exception into
    `{level: "unknown", summary}`.
  - Print ASCII JSON with `overall` set to the worst health level and `info`
    excluded from it. The exit code is 0, or 2 on an internal crash.
  - Verify with new tests in `tests/test_prophet_skill.py`:
    - POST is refused, and a non-allowlisted path is refused;
    - a raising collector reads `unknown` while the others still report;
    - `overall` ignores `info`;
    - the output encodes under cp1252.

## 2. Signals

- [x] 2.1 Implement the forecast, provenance and gold collectors over
  `/api/home`, `/api/provenance` and `/api/gold/health`, with D4's rules.
  - Verify with tests on fixture payloads captured from the live app on
    2026-10-29:
    - outdated for 2026-10-28 → `yellow`;
    - 7 undeclared since 2026-10-14 → `red`;
    - `stalled` with 7 missed hours → `red`;
    - a connection error → `unknown` for all three.
- [x] 2.2 Implement the retrain collector from `results/retrain_state.json` and the
  `=== RETRAIN EXIT` marker in `results/retrain.log`. It never calls the API.
  - Verify with tests for:
    - completed rc=0 → `green` naming the date;
    - rc≠0 → `red`;
    - no marker with a log younger than 300 s → `yellow`;
    - no marker with an older log → `red`;
    - no state file → `green`.
  - A test also reads `api.py` as text and asserts that `_STALL_SECONDS = 300`
    still matches.
- [x] 2.3 Implement the forward-logger collector (newest `logged_at` in
  `results/forward_eval/predictions.csv`, plus `failures.csv` rows from the last
  24 h, with D4's market window) and the server-log collector (last 200 lines;
  a traceback or `ERROR` line → `yellow`, quoting the exception line; a missing
  file → `unknown`).
  - Verify with tests for:
    - stale on a Wednesday → `red`;
    - stale on a Saturday → `green`;
    - two `MT5Unavailable` rows → `yellow`;
    - a traceback in the tail → `yellow`.
- [x] 2.4 Implement the single PowerShell probe (D7) and a pure parser for its JSON.
  - The probe collects:
    - the `python.exe` processes with their creation time;
    - the four `EURUSDProphet-*` tasks' state, last result and last run.
  - Whether the port answers comes from a TCP connect.
  - Server signal: red when down, or when the port answers but no `.venv`
    `api.py` process exists (the launcher's rule); yellow when `api.py` or
    `src/**/*.py` (excluding `__pycache__`) is newer than the start time.
  - Task signal: red when a task is missing or disabled; yellow on a failure
    result. `0`, `0x41301`, `0x41303` and `0x41325` are not failures.
  - Verify with parser tests on recorded probe JSON:
    - current → `green`;
    - a stale `src/home_summary.py` → `yellow`;
    - the port answers with no `.venv` `api.py` process → `red` (foreign);
    - the venv relaunch child (base interpreter, same command line) is not
      taken for ours, and the `.venv` parent is;
    - `GoldLogger` missing → `red`, naming it;
    - `0x41301` → not a failure.
- [x] 2.5 Implement the performance entries at `info`.
  - Per-variant ledgers come from the persisted
    `results/paper_trading_log_{baseline,macro}.csv` (N, win rate, net pips,
    last forecasting date and file mtime).
  - Reliability comes from `/api/home?part=reliability` with a 60 s timeout,
    skipped under `--fast`.
  - Every entry carries N, its Wilson 95 % interval and `too_early` when N < 30.
  - Verify with tests:
    - the 53-position, 35.9 %, −464.8-pip ledger is `info` and leaves `overall`
      unchanged;
    - Wilson agrees with `src.home_summary.wilson_lower` within 1e-9 on several
      (hits, n) pairs;
    - `api.py`/`home_summary.py` read as text still give `MIN_SETTLED = 30`.

## 3. The read-only guarantee, end to end

- [x] 3.1 Add an end-to-end test that runs the pass as a subprocess with `--repo` set
  to a temporary fixture tree and `--base-url` set to a local `http.server`
  thread that serves the fixture JSON and records every request. Stub the
  PowerShell probe with recorded JSON. Verify that every recorded request is a
  GET to an allowlisted path, and that no file in the fixture tree changes:
  same set of files, same mtimes, same bytes.
- [x] 3.2 Run the pass once against the live app. Verify that `git status --porcelain`
  and the mtimes of every file under `results/` and `models/` are the same
  before and after. Record the observed levels in this task line.
  - Done 2026-10-29 at 14:40 UTC, full pass including reliability. No file
    changed and `git status` was identical.
  - Overall `red`.
    - **red:** provenance (13 artifacts undeclared since 2026-10-14); gold
      (stalled, 7 missed hours).
    - **yellow:** forecast (2026-10-28, outdated); tasks (Refit `0x800710E0`);
      forward_logger (1 MT5 failure).
    - **green:** server, retrain, server_log.

## 4. The skill documents

- [x] 4.1 Write `.claude/skills/prophet/references/launch.md`. It covers:
  - start/open with `wscript.exe "<repo>\Start EUR-USD Prophet.vbs"`, which works
    from any cwd;
  - confirming that `/` answers 200;
  - the stale-code restart, done only after the owner says yes;
  - a foreign process on port 8000, which is never killed;
  - failure, with `research_models\server\server.log` as the first place to look;
  - the dev path `python -m uvicorn api:app --reload`.

  Verify with a test that the port, the 180 s start budget and the log path
  named in the file match the constants already pinned in `tests/test_launcher.py`.
- [x] 4.2 Write `.claude/skills/prophet/references/reading-status.md` with one
  section per signal id emitted by `status.py`. Each section covers:
  - what the level means;
  - the normal causes that need no action;
  - where the owner sees it;
  - who acts.

  It also has a performance section: info only, N < 30 means no conclusion, a
  simulation and not a supported edge. Project rules are linked, not copied.
  Verify with tests:
  - every signal id in `status.py` has a heading;
  - none of a list of signature lines from `CLAUDE.md` appears in any skill file
    (for example the `0.05/9` bar text and the `RETRAIN:` hook explanation).
- [x] 4.3 Write `.claude/skills/prophet/SKILL.md`. It contains:
  - frontmatter with `name: prophet` and a bilingual description written per D8,
    under 1024 characters;
  - the request → action routing table;
  - the hard limits: no predict, no retrain, no restart without a yes, no foreign
    kill, no trading advice;
  - the reply format: owner's language, red then yellow, one all-clear line,
    `see` for each problem;
  - links to both references and to the run command
    `.venv\Scripts\python.exe .claude\skills\prophet\scripts\status.py [--fast]`.

  Verify with tests:
  - the frontmatter parses, `name` equals the folder, and the description is
    1024 characters or fewer, contains Cyrillic and English, and mentions
    starting the app;
  - every linked file exists;
  - the body is 150 lines or fewer.

## 5. Evals and the description loop

- [x] 5.1 Write the skill's evals in the format the `skill-creator` skill expects.
  - The format is not `evals.json`. That is skill-creator's older format, which
    it says to read but not write.
  - Behaviour cases go in case folders,
    `evals/<case>/{prompt.md, case.yaml, graders/NN-*.md}`.
  - Trigger queries for the description loop go in
    `evals/trigger-queries.json`: 10 should-trigger and 10 should-not-trigger.
  - Changes from the list below:
    - `/run` is not a trigger query, because `claude -p "/run"` runs the run
      skill directly instead of testing the description. A start-the-app query
      covers it.
    - Following skill-creator's advice, the negatives are near-misses (a pytest
      for `home_summary`, a home-page UI tweak, a Wilson question) instead of a
      general Python question.
  - The provenance-first ordering case runs on live state as
    `status-leads-with-what-is-wrong`.
  - Should-trigger cases (at least 8):
    - "как е Prophet днес?", "пусни приложението", "защо не тръгва", "status",
      "start the app", `/run`;
    - a stale-code question;
    - a gold-logger question.
  - Should-not-trigger cases (at least 6):
    - add a feature to `src/features.py`;
    - explain the volatility family;
    - write a gold pre-registration;
    - retrain the models;
    - review a diff;
    - a general Python question.
  - Behaviour cases with expectations:
    - a trade question gets no buy or sell instruction;
    - a reply with provenance red and the forecast yellow puts provenance first;
    - stale code → asks before restarting.

  Verify that the file parses and that a test checks the case counts.
- [ ] 5.2 Run the `skill-creator` eval and description-optimization loop on the skill.
  Adjust only the description and `SKILL.md` wording until no
  should-not-trigger case triggers and at least 90 % of should-trigger cases
  do. Record the final rates in this task line, and verify that
  `python -m pytest -q tests/test_prophet_skill.py` still passes.

## 6. Integration

- [ ] 6.1 Run `python -m pytest -q`. Verify that the full suite passes and that
  `git status` shows changes only under `.claude/skills/prophet/`,
  `tests/test_prophet_skill.py` and this change's folder. Nothing under
  `models/` is touched, so no `RETRAIN:` line is needed.
- [ ] 6.2 In a fresh Claude Code session, ask "как е Prophet днес?" and run `/run`.
  - Verify that both load the `prophet` skill.
  - Verify that the status reply leads with the live `red` signals, names where
    to see each one, and gives no trading advice.
  - Record the outcome in this task line.
