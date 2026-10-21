# Design

## Context

`scripts\launch.vbs` holds all the launch behaviour: hidden start, reuse, the
stale-code restart, the foreign-port refusal and the Bulgarian failure box. It
finds the repo from its own path (`WScript.ScriptFullName`, two levels up), so
it works no matter who calls it. `scripts\install_shortcut.ps1` writes a single
`.lnk`, `Desktop\EUR-USD Prophet.lnk`. That shortcut targets
`wscript.exe //B //Nologo "<repo>\scripts\launch.vbs"` and uses the icon
`static\prophet.ico`.

Two constraints come from `one-click-launcher`. The launcher spec, together with
`tests/test_launcher.py::test_no_redundant_launch_files_are_tracked`, forbids any
tracked `.bat` file and any `.cmd` file other than `run_module.cmd`. The
`.gitattributes` rule also re-encodes `scripts/launch.vbs` to UTF-16, so that
wscript can read its Cyrillic text.

## Goals / Non-Goals

**Goals:**
- One obvious file to double-click in the project folder.
- The app is findable through Windows search.
- Launch behaviour keeps a single source, `scripts\launch.vbs`.

**Non-Goals:**
- Any change to `launch.vbs` behaviour, the port, the log or `api.py`.
- A stop or restart control, a tray icon, or autostart at log-on.
- Replacing VBScript as the launcher host (see Risks).

## Decisions

**D1. Put a thin `.vbs` wrapper in the repo root, not a copy of the launcher.**
`Start EUR-USD Prophet.vbs` does three things:
1. It finds its own folder from `WScript.ScriptFullName`.
2. It runs `wscript.exe //B //Nologo "<folder>\scripts\launch.vbs"`, with the
   window style hidden (`0`) and without waiting (`False`).
3. It exits.

It holds no logic of its own, so the wrapper can never drift from the icon.

Alternatives considered:
- **A tracked `.lnk` in the repo.** A shortcut stores absolute paths. It would
  break as soon as the repo is cloned or moved, and git cannot keep it portable.
- **A `.cmd` or `.bat` file.** The spec forbids these, and each one flashes a
  console window, which the "no console window" requirement rules out.
- **Moving `launch.vbs` to the root.** Its repo-root detection, its tests, the
  `.gitattributes` rule and the existing desktop shortcut would all have to
  change, and the root would gain a 200-line script.

**D2. Keep the wrapper plain ASCII.** It needs no Bulgarian text, because every
message comes from `launch.vbs`. As a result, it needs no UTF-16
`working-tree-encoding` rule. `.gitattributes` gets an `eol=crlf` line for it,
to match the other Windows scripts.

**D3. Create the Start-menu shortcut with
`[Environment]::GetFolderPath('Programs')`.** This is the per-user folder
`%APPDATA%\Microsoft\Windows\Start Menu\Programs`, so no administrator rights are
needed. The installer loops over the desktop folder and the Programs folder and
writes the same shortcut fields into each. A fixed file name means a re-run
overwrites each shortcut and never adds a second one. The installer prints each
path with `Write-Output`.

**D4. Spec bookkeeping.** The `app-launcher` capability exists only as a delta
inside `one-click-launcher`. That change has all 11 tasks done but is not
archived, so `openspec/specs/` is empty. This change adds requirements only and
never rewrites the existing ones, so it uses `## ADDED Requirements` with no
`## Purpose`. Archive `one-click-launcher` first, so that `app-launcher` exists
when this delta is applied. Otherwise the main spec would be created with a TBD
Purpose.

**D5. Tests stay static, like the existing launcher tests.**
`tests/test_launcher.py` reads the files and does not execute them. It checks
these points:
- the root file exists, is tracked and is pure ASCII;
- it refers to `scripts\launch.vbs` through `ScriptFullName`;
- it contains no absolute `C:\` path;
- the installer refers to the `Programs` folder and writes both shortcuts;
- the opening lines of `README.md` name the root file.

Double-clicking is checked by hand (see `tasks.md`).

## Risks / Trade-offs

- **[Risk] VBScript is a deprecated Windows feature.** Microsoft has announced
  that it will be removed in a future Windows release. On a machine without it,
  the wrapper, the icon and the scheduled tasks would all fail together.
  → Mitigation: this risk is already present and not new, and replacing the host
  is out of scope. The opening section of `HOW_TO_RUN.md` keeps the terminal
  fallback `.venv\Scripts\python.exe api.py` (as a one-line step).
- **[Risk] Two entry points could be mistaken for two apps.** → Mitigation: both
  call the same `launch.vbs`, which reuses a running server. Double-clicking both
  never starts a second server, and the spec scenario "Already running" covers
  this.
- **[Trade-off] The Start-menu entry appears only after the installer runs
  again.** It cannot be made automatically from git. → The README section names
  the command, and the root file works even without any shortcut.
- **[Risk] Antivirus software may block `.vbs` files in user folders.** →
  Mitigation: the desktop icon already depends on `.vbs`, so nothing new is
  exposed. If the file is blocked, the terminal fallback still works.

## Migration Plan

1. Merge the change.
2. Run `powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1`
   once. It refreshes the desktop shortcut and adds the Start-menu shortcut.

Rollback: delete `Start EUR-USD Prophet.vbs` and
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\EUR-USD Prophet.lnk`. The
desktop icon keeps working throughout.
