@echo off
rem Runner for the rts_signals scheduled task (schtasks /TR is capped at 261 characters).
rem Usage: run_module.cmd <logname>
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
if not exist "research_models\rts_signals" mkdir "research_models\rts_signals"
".venv\Scripts\python.exe" -m src.rts_signals.run >> "research_models\rts_signals\%1.log" 2>&1
