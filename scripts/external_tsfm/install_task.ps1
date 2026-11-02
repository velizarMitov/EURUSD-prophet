# Install the external models' forward logger as a scheduled task (openspec change
# tsfm-second-opinion, task 5.3).
#
#   EURUSDProphet-ExternalLogger  hourly at :01 -- Chronos-2 and TimesFM 2.5 forecast the
#                                 newest closed H1 bar of gold and EUR/USD, gaps are
#                                 recorded, matured rows are settled, one health row.
#
# :01 is before the gold pass at :03, so the gold chart export already carries this
# hour's second opinion. Install only after scripts\external_tsfm\install.ps1 has built
# .venv-tsfm and fetched the pinned weights: until then every pass records "external
# environment unavailable" and forecasts nothing.
#
# Runs as the current user, only while that user is logged on: the MT5 terminal lives
# in the user's session and is read, never traded. Own lock, files and log folder
# (research_models\external_tsfm\). Remove with uninstall_task.ps1. See
# results\external_tsfm\RUNBOOK.md.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
if (-not (Test-Path (Join-Path $repo '.venv-tsfm\Scripts\python.exe'))) {
    throw 'the side environment is missing: run scripts\external_tsfm\install.ps1 first'
}
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'research_models\external_tsfm') | Out-Null

$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
$name = 'EURUSDProphet-ExternalLogger'
$cmd = "`"$wscript`" //B //Nologo `"$runner`" src.external.tsfm.forward_logger $name"
if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
& schtasks.exe /Create /F /TN $name /TR $cmd /SC HOURLY /MO 1 /ST 00:01 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }

# The same reliability settings as the gold logger: wake to run, catch up once after a
# missed start (missed bars become gaps, never late forecasts), run on battery, never two
# at once, give up after 20 minutes.
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Set-ScheduledTask -TaskName $name -Settings $settings | Out-Null
Write-Output "installed $name (hourly at :01, wake to run, catch up after a missed start, runs on battery)"
