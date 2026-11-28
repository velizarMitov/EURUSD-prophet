# Install the rts_signals scheduled task (openspec change rts-signals, design D9).
#
#   EURUSDProphet-RTSInbox  every hour at :20 local (Sofia), every day
#
# The other loggers run at :01, :03, :05 and :08, so :20 is free. Each pass fetches new RTS reports from
# Gmail (read-only, once authorised) and the drop folder, tops up the M1 caches (MT5 read only),
# simulates settled sessions, updates the forward ledger and rebuilds the chart file. The 4 pm New York
# report arrives about 22:00-23:00 Sofia, so its levels are on the chart before the 01:00 Sofia open.
# Output: research_models\rts_signals\<task>.log. Run as the current user while logged on (the MT5
# terminal lives in the user's session). -DryRun prints the definition and registers nothing.
# Remove with uninstall_tasks.ps1. See results\rts_signals\RUNBOOK.md.

param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'

$name = 'EURUSDProphet-RTSInbox'
$cmd = "`"$wscript`" //B //Nologo `"$runner`" $name"
if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
$args = @('/Create', '/F', '/TN', $name, '/TR', $cmd, '/SC', 'HOURLY', '/MO', '1', '/ST', '00:20')
if ($DryRun) {
    Write-Output ("DRY RUN ({0} chars) schtasks.exe {1}" -f $cmd.Length, ($args -join ' '))
    return
}
& schtasks.exe @args | Out-Null
if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }
Write-Output "installed $name"
