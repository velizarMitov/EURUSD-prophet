# Install the fx-ny-orb-hot scheduled tasks (openspec change fx-ny-orb-hot, design D7).
#
#   EURUSDProphet-NYOrbForecast  Mon-Fri 16:05 local (Sofia) -- after the bar labelled 14:00 Berlin closes
#   EURUSDProphet-NYOrbSignal    Mon-Fri every 15 minutes from 16:01 to 20:01 -- one minute after each M15 close
#   EURUSDProphet-NYOrbSettle    Mon-Fri 23:05 -- after the bar labelled 21:00 Berlin closes
#
# Each pass reads MT5 read-only and checks the broker's own clock, so a wrong local time zone
# gives a refusal, never a bad row. Output: research_models\fx_ny_orb\<task>.log.
# Run as the current user while logged on (the MT5 terminal lives in the user's session).
# -DryRun prints the definitions and registers nothing. Remove with uninstall_tasks.ps1.

param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
$days = @('/SC', 'WEEKLY', '/D', 'MON,TUE,WED,THU,FRI')

function New-NYOrbTask($name, $pass, $schedule) {
    $cmd = "`"$wscript`" //B //Nologo `"$runner`" $pass $name"
    if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
    $args = @('/Create', '/F', '/TN', $name, '/TR', $cmd) + $schedule
    if ($DryRun) {
        Write-Output ("DRY RUN schtasks.exe " + ($args -join ' '))
        return
    }
    & schtasks.exe @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }
    Write-Output "installed $name"
}

New-NYOrbTask 'EURUSDProphet-NYOrbForecast' 'forecast' ($days + @('/ST', '16:05'))
New-NYOrbTask 'EURUSDProphet-NYOrbSignal' 'signal' ($days + @('/ST', '16:01', '/RI', '15', '/DU', '0004:01'))
New-NYOrbTask 'EURUSDProphet-NYOrbSettle' 'settle' ($days + @('/ST', '23:05'))
