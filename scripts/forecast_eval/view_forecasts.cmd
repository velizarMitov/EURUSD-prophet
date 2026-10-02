@echo off
rem Start the forecast viewer and open it in a browser (openspec change
rem horizon-study-forward-evaluation, task 15.3).
rem
rem A SEPARATE read-only server on 127.0.0.1:8001. It does not touch api.py and
rem it imports no part of the serving application, so the dashboard on
rem 127.0.0.1:8000 is unaffected. Close this window to stop it.
setlocal
set REPO=%~dp0..\..
set PY=%REPO%\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo virtualenv python not found: %PY%
  pause
  exit /b 1
)
cd /d "%REPO%"
set PYTHONIOENCODING=utf-8
title EUR/USD prognozi - 127.0.0.1:8001
"%PY%" -m src.forecast_eval.report --serve --open --port 8001
pause
