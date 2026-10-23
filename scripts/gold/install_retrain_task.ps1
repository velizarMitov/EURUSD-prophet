# Install the monthly gold refit (openspec change gold-forecasting-v2, spec
# gold-forecasting/model-versions).
#
#   EURUSDProphet-GoldRetrain  1st of every month at 02:30 -- fit every gold forward
#                              cell on all bars available into a NEW version
#                              research_models\gold\versions\<YYYYMMDD>\ and make it
#                              the single active SHADOW version. The registered
#                              models and results\gold\artifact_manifest.csv are
#                              never touched; promotion is a manual owner decision
#                              (python -m src.gold.versions promote <id> --reason "...").
#
# Same wake and battery settings as the logger. Output goes to
# research_models\gold\EURUSDProphet-GoldRetrain.log. Remove with
# uninstall_retrain_task.ps1. See results\gold\forward\RUNBOOK.md.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'research_models\gold') | Out-Null

$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
$name = 'EURUSDProphet-GoldRetrain'
$cmd = "`"$wscript`" //B //Nologo `"$runner`" src.gold.retrain_task $name"
if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
& schtasks.exe /Create /F /TN $name /TR $cmd /SC MONTHLY /D 1 /ST 02:30 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }

$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null
Write-Output "installed $name (monthly, 1st at 02:30; wake to run, catch up after a missed start)"
