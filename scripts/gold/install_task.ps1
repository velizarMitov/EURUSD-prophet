# Install the gold forward logger's scheduled task (openspec change
# gold-forecasting, task 8.2).
#
#   EURUSDProphet-GoldLogger  hourly at :03 -- refresh the gold OHLC caches,
#                             forecast the newest closed H1 bar and day with the
#                             frozen artifacts in research_models\gold\, write gap
#                             records for missed bars, settle matured rows.
#
# Install only AFTER `python -m src.gold.study --stage final` has written the
# artifacts and results\gold\artifact_manifest.csv: before that every pass logs
# one failure per cell and forecasts nothing.
#
# Runs as the current user, only while that user is logged on: the MT5 terminal
# lives in the user's session and is read, never traded. Independent of the
# EUR/USD tasks (own lock, files and log folder research_models\gold\).
# Remove with uninstall_task.ps1. See results\gold\forward\RUNBOOK.md.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
$manifest = Join-Path $repo 'results\gold\artifact_manifest.csv'
if (-not (Test-Path $manifest)) { throw "no artifact manifest yet: run 'python -m src.gold.study --stage final' first" }
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'research_models\gold') | Out-Null

$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
$name = 'EURUSDProphet-GoldLogger'
$cmd = "`"$wscript`" //B //Nologo `"$runner`" src.gold.forward_logger $name"
if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
& schtasks.exe /Create /F /TN $name /TR $cmd /SC HOURLY /MO 1 /ST 00:03 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }

# Reliability (openspec change gold-forecasting-v2, spec gold-forecasting/logger-health).
# Installed with defaults the task neither woke the PC nor ran on battery, and the
# night of 2026-10-21/22 lost 8 hours. Wake to run, catch up once after a missed
# start (the logger records missed bars as gaps, it never back-fills), run and keep
# running on battery, never two at once, and give up after 20 minutes.
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null
Write-Output "installed $name (wake to run, catch up after a missed start, runs on battery)"

# A task can only wake the PC if the active power plan allows wake timers. Report
# the setting; never change the owner's power plan.
$q = (& powercfg.exe /q SCHEME_CURRENT SUB_SLEEP RTCWAKE) -join "`n"
$states = @{ '0x00000000' = 'disabled'; '0x00000001' = 'enabled'; '0x00000002' = 'important wake timers only' }
$ac = if ($q -match 'Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)') { $states[$Matches[1].ToLower()] } else { 'unknown' }
$dc = if ($q -match 'Current DC Power Setting Index:\s*(0x[0-9a-fA-F]+)') { $states[$Matches[1].ToLower()] } else { 'unknown' }
Write-Output "wake timers in the active power plan: plugged in = $ac, on battery = $dc"
if ($ac -ne 'enabled') {
    Write-Output "  -> the PC will not wake for the logger while plugged in. Enable 'Allow wake timers' in Power Options > Sleep."
}
