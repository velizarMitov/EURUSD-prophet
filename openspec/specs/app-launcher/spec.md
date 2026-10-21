# app-launcher Specification

## Purpose
Lets the owner start EUR/USD Prophet from one desktop icon, with no console
window and no manual restarts after code updates.

## Requirements

### Requirement: One icon starts or opens the application
A desktop shortcut named "EUR/USD Prophet" SHALL be the only thing the owner needs
to open the app. When no server answers on `127.0.0.1:8000`, it SHALL start one.
When a current server answers, it SHALL only open the browser.

#### Scenario: Cold start
- **WHEN** the owner double-clicks the icon and nothing listens on port 8000
- **THEN** the server starts and the default browser opens `http://127.0.0.1:8000/` once `/` answers 200, not before

#### Scenario: Already running
- **WHEN** the owner double-clicks the icon while a current server answers
- **THEN** only the browser opens, and no second server process is started

### Requirement: No console window
Neither the icon nor the server it starts SHALL show a console window at any
point. Server output SHALL be appended to `research_models/server/server.log`,
which is not tracked by git.

#### Scenario: Hidden start
- **WHEN** the icon starts the server
- **THEN** no cmd or PowerShell window appears, and the server's startup lines appear in `research_models/server/server.log`

### Requirement: Restart when the running code is stale
On each double-click, the launcher SHALL compare the running server's start time
with the newest modification time of `api.py` and `src/**/*.py`. If any of those
files is newer, it SHALL stop that server and start a fresh one before opening
the browser. Changes to HTML pages SHALL NOT trigger a restart.

#### Scenario: Code changed since start
- **WHEN** `src/forecast_eval/report.py` was saved after the running server started, and the owner double-clicks the icon
- **THEN** the old server process stops, a new one starts, and the browser opens on the new code

#### Scenario: Only a page changed
- **WHEN** only `static/home.html` changed since the server started
- **THEN** the server is not restarted and the browser opens immediately

#### Scenario: Foreign process on the port
- **WHEN** port 8000 is held by a process the launcher did not start
- **THEN** the launcher does not kill it, and it shows a message saying that port 8000 is busy with another program

### Requirement: Failure is visible in plain words
If the server does not answer within 180 seconds of being started, the launcher
SHALL show a message box in Bulgarian. The message SHALL say that the program did
not start and name the log file path. It SHALL NOT open the browser.

#### Scenario: Server crashes on start
- **WHEN** the server process exits with an error during startup
- **THEN** a message box names `research_models\server\server.log`, and no browser tab opens on a dead port

### Requirement: One-time icon install
A single command SHALL create or replace the desktop shortcut with the project's
icon, pointing at the launcher. Running it again SHALL be harmless.

#### Scenario: Install twice
- **WHEN** `scripts\install_shortcut.ps1` is run two times
- **THEN** exactly one "EUR/USD Prophet" shortcut exists on the desktop, and it targets `wscript.exe` with the launcher script

### Requirement: Redundant launchers are gone
`start.bat` and `scripts/forecast_eval/view_forecasts.cmd` SHALL NOT exist. The
background-task scripts (`install_tasks.ps1`, `uninstall_tasks.ps1`,
`run_hidden.vbs`, `run_module.cmd`) SHALL remain unchanged.

#### Scenario: Repository check
- **WHEN** the repository's tracked files are listed
- **THEN** there is no `.bat` file, the only `.cmd` is `run_module.cmd`, and the scheduled-task tests still pass unchanged

### Requirement: A launcher sits in the project folder
The repository root SHALL contain a tracked file named
`Start EUR-USD Prophet.vbs`. Double-clicking it in Windows Explorer SHALL behave
exactly as the desktop icon does. It SHALL run the same launcher, with no console
window, and SHALL depend on nothing outside the repository except
`.venv\Scripts\python.exe`.

#### Scenario: Start from the folder
- **WHEN** the owner double-clicks `Start EUR-USD Prophet.vbs` in the project folder and nothing listens on port 8000
- **THEN** the server starts hidden, and the browser opens `http://127.0.0.1:8000/` once `/` answers 200

#### Scenario: Already running
- **WHEN** the owner double-clicks `Start EUR-USD Prophet.vbs` while a current server answers
- **THEN** only the browser opens, and no second server process is started

#### Scenario: Works without the icons
- **WHEN** neither the desktop shortcut nor the Start-menu shortcut exists
- **THEN** `Start EUR-USD Prophet.vbs` still starts or opens the app

#### Scenario: Started from another working directory
- **WHEN** the file is run as `wscript.exe "<repo>\Start EUR-USD Prophet.vbs"` from a different current directory
- **THEN** it still finds the repository's launcher and starts the same server

### Requirement: The app is findable from the Start menu
The one-time installer SHALL create, or replace, a shortcut named
`EUR-USD Prophet` in the current user's Start-menu Programs folder. It SHALL use
the same target, arguments and icon as the desktop shortcut.

#### Scenario: Search finds it
- **WHEN** the installer has run and the owner presses Win and types "Prophet"
- **THEN** Windows search lists "EUR-USD Prophet", and opening it behaves like the desktop icon

#### Scenario: Install twice
- **WHEN** `scripts\install_shortcut.ps1` is run two times
- **THEN** exactly one `EUR-USD Prophet.lnk` exists on the desktop, and exactly one exists in the Start-menu Programs folder

### Requirement: The installer reports where the shortcuts are
After it succeeds, the installer SHALL print the full path of every shortcut it
wrote.

#### Scenario: Installer output
- **WHEN** `scripts\install_shortcut.ps1` completes
- **THEN** its output contains the desktop shortcut's path and the Start-menu shortcut's path

### Requirement: Instructions start with how to open the app
The opening section of `README.md` and the opening section of `HOW_TO_RUN.md`
SHALL each say, in Bulgarian and in English, how to start the app. The three ways
are `Start EUR-USD Prophet.vbs` in the project folder, the desktop icon, and the
Start-menu entry. The section SHALL also give the one command that creates the
icons.

#### Scenario: Reading the README
- **WHEN** the owner opens `README.md`
- **THEN** the opening section names `Start EUR-USD Prophet.vbs` and `scripts\install_shortcut.ps1`, before any training, notebook or API detail
