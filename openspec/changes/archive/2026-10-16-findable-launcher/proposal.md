# Proposal

## Why

The owner opens the project folder and cannot find anything there that starts
the app. The only launcher is the desktop icon `EUR-USD Prophet`, made by
`scripts\install_shortcut.ps1`. `start.bat` was removed in
`one-click-launcher`. Nothing in the repo folder points to the icon, and Windows
search does not find it, because the shortcut exists only on the desktop. If the
icon is lost or hidden under other icons, the owner has no obvious way to start
the app.

## What Changes

- **A launcher in the project folder.** Add a new tracked file at the repo root,
  `Start EUR-USD Prophet.vbs`. Double-clicking it does exactly what the desktop
  icon does: it calls `scripts\launch.vbs` with the same hidden start, reuse,
  stale-code restart and failure message. The behaviour has a single source.
- **A Start-menu entry.** `scripts\install_shortcut.ps1` also creates
  `EUR-USD Prophet.lnk` under the user's Start-menu Programs folder, next to the
  desktop icon. Pressing **Win** and typing "Prophet" then finds the app. Running
  the installer again still produces exactly one desktop shortcut and one
  Start-menu shortcut.
- **The installer says where the shortcuts are.** It prints both shortcut paths.
- **Instructions lead with how to start the app.** The first lines of
  `README.md` and `HOW_TO_RUN.md` say, in Bulgarian and English: double-click
  `Start EUR-USD Prophet.vbs` in the project folder, or use the desktop or
  Start-menu icon. Run the installer once to create the icons.
- **Not changed:** `scripts\launch.vbs`, `api.py`, the port, the log path, and
  the scheduled-task scripts. No `.bat` or `.cmd` file is added, because the
  launcher spec forbids them. No stop control is added.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `app-launcher`: adds three entry points to the same launcher: one in the project
  folder, one in the Start menu, and the shortcut paths the installer prints.
  The existing requirements stay as they are. The capability is defined by the
  still-unarchived `one-click-launcher` change, so this delta is written as
  ADDED requirements on that path (see design, D4).

## Impact

- **New:** `Start EUR-USD Prophet.vbs` (repo root, plain ASCII).
- **Edited:**
  - `scripts/install_shortcut.ps1`, which now creates the Start-menu shortcut and
    prints both shortcut paths;
  - `tests/test_launcher.py`, which gains new tests;
  - `README.md`, `HOW_TO_RUN.md` and `CLAUDE.md`, whose start instructions
    change.
- **Untouched:** `scripts/launch.vbs`, `api.py`, `src/`, `static/`, `models/`
  and `results/`. No retrain.
- **Owner action, once:** run
  `powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1` again
  to add the Start-menu entry.
