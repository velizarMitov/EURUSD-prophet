# Tasks

## 1. Spec bookkeeping

- [x] 1.1 Archive `one-click-launcher` (all of its tasks are done), so that `openspec/specs/app-launcher/spec.md` exists before this change's ADDED delta is applied (design D4). Verify that `openspec list --specs` shows `app-launcher`, and that `openspec validate findable-launcher` passes.

## 2. Launcher in the project folder

- [x] 2.1 Create `Start EUR-USD Prophet.vbs` at the repo root, following design D1 and D2:
  - it is plain ASCII and holds a short header comment;
  - it gets its folder from `WScript.ScriptFullName` and runs `wscript.exe //B //Nologo "<folder>\scripts\launch.vbs"` with `Run(..., 0, False)`;
  - it contains no absolute path;
  - add a `.gitattributes` line, `"Start EUR-USD Prophet.vbs" text eol=crlf`.

  Verify by hand. With the server stopped, double-click the file in Explorer: no window appears, and the browser opens once `/` answers. Double-click it again: only the browser opens, and the count of WMI `python.exe` processes running `api.py` stays the same.
- [x] 2.2 Run `wscript.exe "C:\Users\test\Desktop\eurusdprophet\Start EUR-USD Prophet.vbs"` from a PowerShell prompt whose current folder is `C:\`. Verify that it opens the app, which covers the scenario "Started from another working directory".
- [x] 2.3 Add these tests to `tests/test_launcher.py`:
  - `test_project_folder_launcher_is_tracked_and_delegates`: the file is listed by `git ls-files`, its bytes are ASCII, and it contains `ScriptFullName`, `scripts\launch.vbs`, `//B` and `, 0, False`;
  - `test_project_folder_launcher_has_no_absolute_path`: no `C:\` or `Users\`.

  Verify that `python -m pytest -q tests/test_launcher.py` passes, including the unchanged `test_no_redundant_launch_files_are_tracked`.

## 3. Start-menu entry

- [x] 3.1 Change `scripts/install_shortcut.ps1` to follow design D3:
  - it writes the same shortcut fields to `[Environment]::GetFolderPath('Desktop')` and to `[Environment]::GetFolderPath('Programs')`, each as `EUR-USD Prophet.lnk`;
  - it prints each path;
  - it updates its header comment.

  Run it twice. Verify that there is exactly one `.lnk` in each folder, and that each one targets `wscript.exe` with `//B` and `launch.vbs`. Press Win, type "Prophet", and check that "EUR-USD Prophet" is listed and opens the app.
- [x] 3.2 Extend `test_shortcut_installer_targets_wscript_batch_mode` (or add a sibling test) to assert that the installer mentions both `'Desktop'` and `'Programs'`, and that it prints the shortcut paths. Verify with `python -m pytest -q tests/test_launcher.py`.

## 4. Instructions

- [x] 4.1 Add a short section titled "Как да стартирам / How to start" as the first section after the title in `README.md`, and again in `HOW_TO_RUN.md`. It has 3 to 5 lines in Bulgarian and English and covers:
  - double-click `Start EUR-USD Prophet.vbs` in the project folder;
  - or use the desktop or Start-menu icon "EUR-USD Prophet";
  - run `powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1` once to create the icons;
  - if all of these fail, run `.venv\Scripts\python.exe api.py`.

  In the `CLAUDE.md` Commands block, update the "Owner's way" comment to name the root file and the Start-menu entry. Verify that `grep -n "Start EUR-USD Prophet.vbs" README.md HOW_TO_RUN.md CLAUDE.md` finds each file.
- [x] 4.2 Add `test_readme_opens_with_how_to_start` to `tests/test_launcher.py`. It checks that `Start EUR-USD Prophet.vbs` and `install_shortcut.ps1` appear in `README.md` before the `## Quick start for reviewers` heading. Verify with `python -m pytest -q tests/test_launcher.py`.

## 5. Integration check

- [x] 5.1 Run `python -m pytest -q` from the repo root. Verify that the result matches the pre-change baseline plus the new launcher tests, with no new failures. Check `git status` and confirm that nothing under `models/` is staged by this change.
