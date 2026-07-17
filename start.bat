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
echo The dashboard opens in your browser automatically once the server is ready
echo (first start loads TensorFlow + all model artifacts: ~30-60 seconds).

REM Open the dashboard only when the server actually answers. A fixed delay is
REM wrong here: the cold start takes ~25-60 s, so a 10 s timer opened the
REM browser on a dead port ("This site can't be reached") and made the app look
REM broken. This helper polls hidden in a separate process (never blocks the
REM server below) and gives up after ~3 minutes if the server never comes up,
REM leaving the real error visible in this window.
start "" powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:8000'; break}catch{Start-Sleep -Seconds 1}}"

"%PY%" api.py

REM Keep the window open if the server exits with an error.
if errorlevel 1 (
    echo.
    echo The server stopped with an error. See the messages above.
    pause
)
endlocal
