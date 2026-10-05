# Install the forward-evaluation scheduled tasks (openspec change
# horizon-study-forward-evaluation, tasks 9.6, 10.2 and 14.10).
#
#   EURUSDProphet-ForwardLogger  every 15 minutes at :01/:16/:31/:46 --
#                                predict / gap / settle for M15, H1 and D1 in
#                                one pass. One minute after each bar close, as
#                                the H1 design already did; the cadences that
#                                have no new bar return "nothing new", so the
#                                finer schedule only makes H1 and D1 coverage
#                                more robust. The run lock keeps passes apart.
#   EURUSDProphet-Refit          every Saturday 06:00 -- refits only on the
#                                first weekend of the month, otherwise no-op
#
# Both run as the current user, only while that user is logged on: the MT5
# terminal lives in the user's session, and the logger attaches READ-ONLY to
# whatever account it holds. Output goes to research_models\forward_eval\.
# Remove with uninstall_tasks.ps1. See results\forward_eval\RUNBOOK.md.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
$logDir = Join-Path $repo 'research_models\forward_eval'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'

function New-ProphetTask($name, $module, $schedule) {
    # schtasks caps /TR at 261 characters, so the task calls a short runner.
    # The runner is a .vbs run by wscript.exe, NOT run_module.cmd directly: cmd.exe
    # is a console program and flashed a window every 15 minutes. wscript //B has none.
    $cmd = "`"$wscript`" //B //Nologo `"$runner`" $module $name"
    if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
    $args = @('/Create', '/F', '/TN', $name, '/TR', $cmd) + $schedule
    & schtasks.exe @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }
    Write-Output "installed $name"
}

New-ProphetTask 'EURUSDProphet-ForwardLogger' 'src.forecast_eval.forward_logger' @('/SC', 'MINUTE', '/MO', '15', '/ST', '00:01')
New-ProphetTask 'EURUSDProphet-Refit' 'src.forecast_eval.refit' @('/SC', 'WEEKLY', '/D', 'SAT', '/ST', '06:00')
