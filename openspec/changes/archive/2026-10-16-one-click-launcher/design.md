# Design

## Context

- Today `start.bat` sets UTF-8, picks `.venv`, kills whatever holds port 8000
  (any process, ours or not), polls `/` from a hidden PowerShell, and runs
  `python api.py` in the foreground. The window is the server, so closing it
  stops the app.
- The scheduled forward-logger already runs hidden through
  `wscript.exe //B run_hidden.vbs`. That pattern is proven on this machine and
  tested in `test_scheduled_tasks_run_without_a_console_window`.
- `api.py` is under an additive-only guard. It cannot be edited to write a PID
  file or a start timestamp, and does not need to be (see Decision 2).
- `.venv\Scripts\python.exe` exists. Every launcher here assumes the repo's
  `.venv`, as the task scripts do.

## Goals / Non-Goals

**Goals:**
- One double-click, no window, and the browser lands on live code.
- The launcher never kills a process it cannot prove is this app's server.

**Non-Goals:**
- No system-tray app, Windows service or auto-start at logon. The owner chose
  the icon.
- No stop control (owner's choice). The server ends at log-off or shutdown.
- No change to the scheduled-task scripts or to how the forward logger runs.

## Decisions

1. **The launcher is a VBScript run by `wscript.exe`
   (`scripts/launch.vbs`).** `wscript` is a GUI-subsystem host, so no console
   window ever appears, the same reason `run_hidden.vbs` exists. It uses
   `MSXML2.ServerXMLHTTP` to probe `http://127.0.0.1:8000/` and WMI
   (`Win32_Process`) to find processes.
   *Alternatives:* a PowerShell script, which flashes a window when started from
   a shortcut unless wrapped (and the wrapper would be a VBS anyway); a frozen
   Python exe, which adds a build step and a large binary.

2. **"Is it ours, and is it stale?" comes from WMI, not from a state file.**
   The server is the `Win32_Process` whose `ExecutablePath` lies under the repo's
   `.venv` and whose `CommandLine` ends in `api.py`. Its `CreationDate` is the
   start time. Staleness means the newest `DateLastModified` of `api.py` and
   `src\**\*.py` is later than that `CreationDate`.
   *Alternative:* a PID or timestamp file. Rejected: it can go stale after a
   crash or reboot, and writing it from the server would need an `api.py` edit.
   A server started by hand (`python api.py` from a terminal) is recognised the
   same way, which is intended.

3. **Decision table on each double-click:**

   ```
   port answers? | our process? | stale? | action
   --------------+--------------+--------+----------------------------------
   no            | no           |   -    | start, wait for 200, open browser
   no            | yes          |   -    | (still starting) wait, open browser
   yes           | yes          | no     | open browser
   yes           | yes          | yes    | terminate ours, start, wait, open
   yes           | no           |   -    | message: port 8000 busy, do nothing
   ```

   Only processes matched by Decision 2 are ever terminated. This is narrower
   than `start.bat`, which killed whatever held the port.

4. **Starting the server:**
   ```
   cmd /c "set PYTHONIOENCODING=utf-8 && .venv\Scripts\python.exe -u api.py
           >> research_models\server\server.log 2>&1"
   ```
   It is run with window style 0 and does not wait. `-u` keeps the log current,
   for the same reason the retrain runner uses it. Before each start, the
   launcher appends a dated separator line to the log, so one file stays
   readable across starts. `research_models/` is already gitignored.

5. **Waiting and failing.** The launcher polls `/` once a second for up to 180 s.
   This is the same budget `start.bat` used, because a cold TensorFlow start
   takes 25–60 s. If the process exits early or the timer runs out, it shows a
   `MsgBox` in Bulgarian with the log path and opens no browser. An optional
   argument `/entry:<file>` replaces `api.py`, for the manual failure test only.

6. **The shortcut.** `scripts/install_shortcut.ps1` uses `WScript.Shell`
   `CreateShortcut` and writes `Desktop\EUR-USD Prophet.lnk`. Its target is
   `%SystemRoot%\System32\wscript.exe`, its arguments are
   `//B //Nologo "<repo>\scripts\launch.vbs"`, its working directory is the
   repo, and its icon is `static\prophet.ico`. Running it again overwrites the
   same `.lnk`. The icon is generated once with Pillow (already installed via
   matplotlib) and committed. It is a small arrow-on-chart glyph, not a brand
   mark.

7. **The `latest_forecasts` fix.** Replace
   `sort_values('as_of').groupby([...]).last()`, which takes the last non-null
   value per column, with
   `sort_values('as_of').drop_duplicates(['model', 'horizon'], keep='last')`,
   which takes the last whole row. No other caller relies on the fill-in
   behaviour. `home_summary.session_block` also reads this function, so the home
   page inherits the fix.

8. **Finishing `friendly-home-page`.**
   - **`DATA.md`:** update the H1 row count to the size of the
     `results/eurusd_h1.csv` actually committed. The cache is append-only real
     market data, and reverting it would throw away real bars. The guard stays
     as it is.
   - **Wording pass (task 5.2), done for the owner:**

     | key | now | becomes |
     |---|---|---|
     | `tomorrow_title` | Утре / Tomorrow | Дневна прогноза / Daily forecast |
     | `link_kronos` | Kronos волатилност (наблюдение) | Kronos: размах на движението (наблюдение) / Kronos: size of moves (observational) |
     | `fallback_variant` | Показана е версията с макро данни (по-стар запис). | Показана е по-старата версия на модела (запис отпреди двете версии). / Showing the older model version (record from before the two versions). |
     | `rel_basis` | …ценовия модел… | Броят се само дневните прогнози; дните „няма ясен сигнал“ не влизат. / Only daily forecasts count; "no clear signal" days are left out. |

     The tile title changes because the tile is wrong in the morning: at 09:16
     it showed today's session under "Утре". The date line already names the
     day, so a neutral title is correct at every hour.

## Risks / Trade-offs

- [WMI queries can be slow on the first call, around 1 s] → This is acceptable
  for a double-click, and it runs once per click.
- [A hung server still answering 200 after a code update is "current" by port
  but stale by time] → Covered by Decision 3 (yes / ours / stale → restart).
- [A hung server that stops answering but still exists] → Row 2 waits 180 s, then
  shows the failure message with the log path. The owner can then end the Python
  process in Task Manager. This is rare and a real fault, so it is left manual
  rather than auto-killed.
- [The owner loses the console view of errors] → The log file, plus the message
  box that names it. `HOW_TO_RUN` documents
  `python api.py` in a terminal for debugging.
- [Removing `start.bat` breaks old habits and links in docs] → All docs updated
  in the same change. `HOW_TO_RUN` keeps the terminal command as the manual
  path.

## Migration Plan

1. Run `scripts\install_shortcut.ps1` once. The icon appears on the desktop.
2. Close any old `start.bat` window. From then on, use the icon.
3. Rollback: `git revert`. `start.bat` returns, and the shortcut can be deleted
   by hand.
