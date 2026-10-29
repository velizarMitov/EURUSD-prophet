# Proposal

## Why

To know how the app is doing, the owner opens four pages and reads them side by
side. Claude, asked the same question, has no project skill to follow. It
guesses which endpoints to call and how to read them, and the built-in `/run`
skill finds no project skill that says how this app is started.

On 2026-10-29 four calls made by hand turned up three problems that no single
page shows together:
- today's forecast was missing (`/api/home` `tomorrow.outdated`);
- models were served undeclared since 2026-10-14 (`/api/provenance`);
- the gold logger had stalled with 7 missed hours (`/api/gold/health`).

The same probe also showed that the obvious calls are not safe.
`GET /api/paper-trading` rewrites the per-variant ledger CSVs.
`GET /api/retrain/status` hot-reloads the models when a run has completed.
A skill that "just checks status" through them would change the system it
reports on.

## What Changes

- **A project skill, `.claude/skills/prophet/`, tracked in git.** Claude loads it
  when the owner asks, in Bulgarian or English, how the app is doing, or to start,
  open or diagnose it. `/run` also finds it as the project's launch skill.
- **One read-only status pass, `scripts/status.py`.** It produces a single JSON
  report from these sources:
  - the server process;
  - `/api/home`;
  - `/api/provenance`;
  - `/api/gold/health`;
  - the retrain state files;
  - the four `EURUSDProphet-*` scheduled tasks;
  - the forward-eval logs;
  - the tail of `server.log`.

  Each health signal gets `green`, `yellow`, `red` or `unknown`. Performance
  numbers are reported as `info` and never colored. The pass calls no endpoint
  that writes or predicts, and it writes no file.
- **Two references that load only when they are needed.**
  - `references/launch.md` covers starting the app and diagnosing it: the
    launcher, a stale-code restart, a foreign process on port 8000, and
    `server.log`.
  - `references/reading-status.md` explains how to read each signal and what
    to say about it.
- **Hard limits in the skill.**
  - It never calls `POST /api/predict` or `POST /api/retrain`.
  - It never stops or restarts a running server without asking.
  - It never gives trading advice.
  - It never restates a `CLAUDE.md` rule; it links to it.
- **Tests** pin the skill's metadata and the endpoint allowlist, check that
  nothing is written, and check how each signal is classified.
- **Trigger evals** check that the skill's description fires on the owner's
  real phrasings and stays quiet on unrelated requests.
- **Not changed:**
  - `api.py`, `src/`, `static/`, `models/` and `results/`. The app gains no
    endpoint and no retrain is involved.
  - `scripts/launch.vbs` and the `app-launcher` spec.
  - The old Copilot-era `.github/skills/`, which describe a different project
    stage.

## Capabilities

### New Capabilities
- `operator-skill`: what the project's Claude Code skill must do when the owner
  asks about, starts or diagnoses the running app. This covers when it triggers,
  the read-only status report and its signal levels, how performance is
  reported, and the actions it may never take.

### Modified Capabilities
- None. The skill uses the existing launcher and the existing read-only endpoints
  as they are.

## Impact

- **New:**
  - `.claude/skills/prophet/SKILL.md`, `scripts/status.py`,
    `references/launch.md` and `references/reading-status.md`;
  - the skill's `evals/evals.json`;
  - `tests/test_prophet_skill.py`.
- **Untouched:** every file the app runs. There is no new dependency:
  `status.py` uses only the standard library and PowerShell, which the launcher
  already relies on.
- **Owner action:** none. The skill is discovered from the repository on the
  next Claude Code session.
