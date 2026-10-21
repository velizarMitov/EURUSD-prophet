@echo off
rem Runner for the gold forward logger's scheduled task (schtasks /TR is capped at 261
rem characters, so the task calls this file). Usage: run_module.cmd <module> <logname>
rem Logs go to research_models\gold\ -- never to the EUR/USD forward_eval folder.
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
set TF_CPP_MIN_LOG_LEVEL=3
if not exist "research_models\gold" mkdir "research_models\gold"
".venv\Scripts\python.exe" -m %1 >> "research_models\gold\%2.log" 2>&1
