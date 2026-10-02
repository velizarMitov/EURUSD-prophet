@echo off
rem Runner for the forward-evaluation scheduled tasks (schtasks /TR is capped at 261
rem characters, so the task calls this file). Usage: run_module.cmd <module> <logname>
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
if not exist "research_models\forward_eval" mkdir "research_models\forward_eval"
".venv\Scripts\python.exe" -m %1 >> "research_models\forward_eval\%2.log" 2>&1
