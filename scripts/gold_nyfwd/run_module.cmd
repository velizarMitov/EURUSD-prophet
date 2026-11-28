@echo off
rem Runner for the gold-nyrange-forward scheduled tasks (schtasks /TR is capped at 261 characters).
rem Usage: run_module.cmd <pass> <logname>   where <pass> is forecast or settle.
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
if not exist "research_models\gold_nyfwd" mkdir "research_models\gold_nyfwd"
".venv\Scripts\python.exe" -m src.gold_nyfwd.live --pass %1 >> "research_models\gold_nyfwd\%2.log" 2>&1
