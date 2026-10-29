# Starting, opening and diagnosing the app

The app is one FastAPI server on `http://127.0.0.1:8000`. It serves the home
page `/`, `/advanced`, `/gold`, `/forecasts` and the rest. The owner starts it
with the launcher. Its behaviour is pinned in the `app-launcher` spec
(`openspec/specs/app-launcher/spec.md`) and in `tests/test_launcher.py`.

## First: is it up?

```
.venv\Scripts\python.exe .claude\skills\prophet\scripts\status.py --fast
```

Read the `server` signal and pick the matching section:

| `server` says | Do |
|---|---|
| green | It is up. Tell the owner the address. Nothing to start. |
| red, "not running" | [Start it](#start-it). This is allowed without asking. |
| yellow, "older than what is on disk" | [Stale code](#stale-code). Ask first: starting it restarts it. |
| red, "port 8000 answers, but not from the project server" | [Foreign process](#foreign-process-on-port-8000). Never kill it. |
| red, "process exists but port 8000 does not answer" | [It will not start](#it-will-not-start). |

## Start it

The launcher sits in the repo root and works from any current directory:

```
wscript.exe //B //Nologo "C:\Users\test\Desktop\eurusdprophet\Start EUR-USD Prophet.vbs"
```

It is the same launcher as the desktop icon and the Start-menu entry "EUR-USD
Prophet". It:
1. starts `.venv\Scripts\python.exe -u api.py` hidden, with no console window;
2. appends the output to `research_models\server\server.log`;
3. waits up to 180 seconds for `/` to answer 200;
4. then opens the owner's browser.

Startup loads TensorFlow and every model, so 30 to 90 seconds is normal.

To confirm, run `status.py --fast` again: the `server` signal should be green.
Or check that `GET http://127.0.0.1:8000/` returns 200. Then give the owner the
address.

Do not start it with `python api.py` or `uvicorn` from a Claude shell. That
process dies with the shell, and the launcher will treat it as a stranger, as
described in the next section.

## Stale code

`api.py` or a `src/**/*.py` file was saved after the server started, so the
server runs old code. HTML pages do not count, because they are read from disk
on each request.

The next launcher run stops the old server and starts a fresh one. That
interrupts anyone using the app, and a running `POST /api/retrain` child is
stopped with it. Check the `retrain` signal first: if it is yellow ("running"),
say so and do not restart.

Ask the owner, for example: "Сървърът върви със стар код (`<file>` е променен
след старта). Да го рестартирам ли?" Run the launcher only after a yes.

## Foreign process on port 8000

Something answers on port 8000, but it is not a `.venv` `python.exe` running
`api.py`. The launcher then shows "Порт 8000 е зает от друга програма" and quits
with code 3. It never kills that process, and neither does Claude.

The usual cause is a dev server, `python -m uvicorn api:app --reload`. Its
command line says `api:app`, not `api.py`, so the launcher does not own it. To
show the owner what it is:

```
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
```

Name the process, and leave the decision to the owner.

## It will not start

The launcher shows a Bulgarian message box that says the program did not start
and names `research_models\server\server.log`. No browser opens. Look in this
order:

1. **The end of `research_models\server\server.log`** (the last 80 lines). A
   traceback there is the answer in most cases. A failed import names the
   missing package; install it in `.venv` only after the owner agrees.
2. **The missing interpreter message**, "Не е намерен Python в папката на
   проекта". `.venv\Scripts\python.exe` is gone, so the venv needs to be
   recreated (`HOW_TO_RUN.md` §3–4). That is the owner's call.
3. **A process that exists but never answers.** If it is still loading, wait out
   the 180 s. If it hangs past that, report the pid and the log tail, and ask
   before stopping it.

## Stopping (only after the owner says yes)

Stop only the project's own server, which is the `.venv` `python.exe` running
`api.py` plus its child:

```
$repo = 'C:\Users\test\Desktop\eurusdprophet'
$ours = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.ExecutablePath -like "$repo\.venv\*" -and $_.CommandLine -like '*api.py*' }
foreach ($p in $ours) {
  Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.ProcessId)" | ForEach-Object { Stop-Process -Id $_.ProcessId }
  Stop-Process -Id $p.ProcessId
}
```

## Not the server's job

- **The background loggers** are scheduled tasks and run with or without the
  server: `EURUSDProphet-ForwardLogger` (every 15 min), `EURUSDProphet-GoldLogger`
  (hourly), `EURUSDProphet-GoldRetrain` and `EURUSDProphet-Refit` (weekly). Their
  installers are in `scripts\forecast_eval\` and `scripts\gold\`. Reinstalling one
  changes the owner's machine, so ask first.
- **The desktop and Start-menu icons** are made once by
  `powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1`. That is
  an owner action.
