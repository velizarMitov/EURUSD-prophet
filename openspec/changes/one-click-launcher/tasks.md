# Tasks

## 1. Forecasts view: take the newest row whole

- [x] 1.1 In `src/forecast_eval/report.py::latest_forecasts`, replace `groupby(...).last()` with `drop_duplicates(['model', 'horizon'], keep='last')` after sorting by `as_of`. Add `test_newest_forecast_with_no_probability_is_not_filled_from_an_older_one` to `tests/test_forecast_eval_report.py` (10:00 `p_up` 0.70, 10:15 blank → direction "—", "70" absent from the headline and the cell). Verify the new test fails before the change and passes after, and that `python -m pytest -q tests/test_forecast_eval_report.py tests/test_home_overview.py` is green.

## 2. The launcher

- [x] 2.1 Write `scripts/launch.vbs` per design Decisions 1–5: the WMI ownership and staleness check, the decision table, the hidden start with log append and separator, the 180 s poll, the Bulgarian `MsgBox` on failure or on a foreign process holding the port, and the `/entry:` test override. Verify the cold-start case by hand: with nothing on :8000, double-click it. There should be no window, the log should gain startup lines, and the browser should open only after `/` answers.
- [x] 2.2 Verify the remaining launcher rows by hand and record the results in this task line. **Results 2026-10-07 (cscript //B):** the cold start answered in 33 s; (a) took 1 s with the same PIDs; (b) restarted the owner's stale start.bat server, and after touching home_summary.py gave new PIDs in 39 s; (c) kept the same PIDs; (d) left the foreign http.server alive, started nothing, and blocked on the message box; (e) logged "can't open file missing.py", opened no browser, and ended via the error box. A venv python.exe re-launches the base interpreter as a child that holds the port, so StopTree ends the whole tree.
  - (a) Already running → the browser opens and the WMI process count stays at 1.
  - (b) `touch src/home_summary.py`, then double-click → a new PID, and the browser opens.
  - (c) `touch static/home.html` only → same PID.
  - (d) Run `python -m http.server 8000` and double-click → the "port busy" message, and the http.server process survives.
  - (e) `/entry:missing.py` → the failure message names the log, and no browser opens.
- [x] 2.3 Generate `static/prophet.ico` (16/32/48/256 px) with Pillow, and write `scripts/install_shortcut.ps1` per Decision 6. Run it twice. Verify that one `EUR-USD Prophet.lnk` exists on the desktop, with target `wscript.exe`, arguments containing `//B` and `launch.vbs`, and the icon set. Double-clicking it behaves like 2.1.
- [x] 2.4 Add `tests/test_launcher.py`. It statically checks that `launch.vbs` runs hidden (`.Run(..., 0, False)`), probes `127.0.0.1:8000`, polls up to 180 s, logs to `research_models\server\server.log`, terminates only processes matched by `.venv` and `api.py` (no bare `taskkill`), and has the Bulgarian failure text. It also checks that `install_shortcut.ps1` targets `wscript.exe //B` and `launch.vbs`, and that `research_models/` is gitignored. Verify with `python -m pytest -q tests/test_launcher.py`.

## 3. Remove the redundant launchers

- [x] 3.1 Delete `start.bat` and `scripts/forecast_eval/view_forecasts.cmd`, and remove `test_the_launcher_script_points_at_the_viewer`. Add to `tests/test_launcher.py` a check that `git ls-files` lists no `.bat` and that `run_module.cmd` is the only `.cmd`. Verify that `test_scheduled_tasks_run_without_a_console_window` still passes unchanged.
- [x] 3.2 Update the start instructions in `README.md`, `HOW_TO_RUN.md` (Option A becomes the icon: install once with `scripts\install_shortcut.ps1`; the terminal command stays as the debug path; "restart the server" notes become "double-click the icon", and the log path is listed), `results/forward_eval/RUNBOOK.md` (the viewer section and the restart note) and `CLAUDE.md` (Commands). Verify that `grep -rn "start.bat\|view_forecasts" --include=*.md .` finds nothing outside `openspec/` and `Claude outputs/`.

## 4. Finish `friendly-home-page`

- [x] 4.1 Apply the four label changes from design Decision 8 in `static/home.html`. Verify that `tests/test_home_overview.py` passes (dictionary complete, no banned words), and check by eye at `/` that the tile reads "Дневна прогноза" with its date.
- [x] 4.2 Update the `results/eurusd_h1.csv` row count in `DATA.md` to the file's current size. Record the decision (keep the real bars, update the count) in `openspec/changes/friendly-home-page/tasks.md` task 5.1, and tick tasks 5.1 and 5.2 there. Verify that `python -m pytest -q tests/test_input_data_provenance.py` passes.

## 5. Integration check

- [x] 5.1 (2026-10-07: 933 passed, 0 failed; the protected set is clean, and api.py has no diff against HEAD 944c139.) Run `python -m pytest -q`. Expect everything green except the known POSIX-shell skip. Confirm that `git diff --quiet HEAD -- api.py src/inference.py src/paper_trading.py static/index.html models/ scripts/forecast_eval/install_tasks.ps1 scripts/forecast_eval/run_hidden.vbs scripts/forecast_eval/run_module.cmd` is clean apart from the earlier uncommitted `friendly-home-page` lines in `api.py`.
- [x] 5.2 (2026-10-07: with every server stopped, the real desktop icon started the server at 10:33:18 and / served "Дневна прогноза". /forecasts at 10:38 showed only the 10:00-11:00/12:00/14:00 windows, and no finished window.) From a cold machine state (no server running), double-click the desktop icon. The home page should open with no window. Then open `/forecasts` and confirm that no finished window is shown as current.
