# Spec Delta

## ADDED Requirements

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
