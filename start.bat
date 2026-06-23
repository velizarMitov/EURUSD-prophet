@echo off
REM ============================================================================
REM  EUR/USD Prophet - one-click launcher (Windows)
REM  Double-click this file, or run `start.bat` from a terminal.
REM  It picks the project's virtual environment if present, otherwise the
REM  system Python, then starts the FastAPI app at http://127.0.0.1:8000
REM ============================================================================
setlocal
cd /d "%~dp0"

REM Avoid cp1252 UnicodeEncodeError on non-ASCII console output.
set PYTHONIOENCODING=utf-8

REM Prefer a local virtual environment if one exists (.venv, then venv).
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Using interpreter: %PY%

REM Free port 8000 if a previous server is still holding it (avoids WinError 10048).
for /f "tokens=5" %%P in ('netstat -ano ^| findstr "127.0.0.1:8000" ^| findstr LISTENING') do (
    echo Port 8000 is busy ^(PID %%P^) - stopping the old server...
    taskkill /PID %%P /F >nul 2>&1
)

echo Starting EUR/USD Prophet at http://127.0.0.1:8000  (close this window to stop)

REM Open the dashboard in the default browser once the server has had a moment
REM to load (TensorFlow + model artifacts take a few seconds). This runs in a
REM separate window so it does not block the server below.
start "" cmd /c "timeout /t 10 /nobreak >nul & start "" http://127.0.0.1:8000"

"%PY%" api.py

REM Keep the window open if the server exits with an error.
if errorlevel 1 (
    echo.
    echo The server stopped with an error. See the messages above.
    pause
)
endlocal
