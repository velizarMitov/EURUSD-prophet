# Install the forward-evaluation scheduled tasks (openspec change
# horizon-study-forward-evaluation, tasks 9.6 and 10.2).
#
#   EURUSDProphet-ForwardLogger  every hour at :01 -- predict / gap / settle
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

$runner = Join-Path $PSScriptRoot 'run_module.cmd'

function New-ProphetTask($name, $module, $schedule) {
    # schtasks caps /TR at 261 characters, so the task calls a short runner.
    $cmd = "`"$runner`" $module $name"
    $args = @('/Create', '/F', '/TN', $name, '/TR', $cmd) + $schedule
    & schtasks.exe @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }
    Write-Output "installed $name"
}

New-ProphetTask 'EURUSDProphet-ForwardLogger' 'src.forecast_eval.forward_logger' @('/SC', 'HOURLY', '/MO', '1', '/ST', '00:01')
New-ProphetTask 'EURUSDProphet-Refit' 'src.forecast_eval.refit' @('/SC', 'WEEKLY', '/D', 'SAT', '/ST', '06:00')
