@echo off
rem Runner for the fx-ny-orb-hot scheduled tasks (schtasks /TR is capped at 261 characters).
rem Usage: run_module.cmd <pass> <logname>   where <pass> is forecast, signal or settle.
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
if not exist "research_models\fx_ny_orb" mkdir "research_models\fx_ny_orb"
".venv\Scripts\python.exe" -m src.fx_ny_orb.live --pass %1 >> "research_models\fx_ny_orb\%2.log" 2>&1
