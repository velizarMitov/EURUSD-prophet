# Install the gold-nyrange-forward scheduled tasks (openspec change gold-nyrange-forward, design D8).
#
#   EURUSDProphet-GoldNYForecast  Mon-Fri 16:08 local (Sofia) -- after the bar labelled 14:00 Berlin closes
#   EURUSDProphet-GoldNYSettle    Mon-Fri 23:08 local (Sofia) -- after the bar labelled 21:00 Berlin closes
#
# The other loggers run at :01, :03 and :05, so these minutes are free. Each pass checks the Berlin
# label clock itself, so a wrong local time zone gives a refusal, never a bad row. MT5 is read only.
# Output: research_models\gold_nyfwd\<task>.log. Run as the current user while logged on (the MT5
# terminal lives in the user's session). -DryRun prints the definitions and registers nothing.
# Remove with uninstall_tasks.ps1. Install only after PRE_REGISTRATION.md and .json are committed.

param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
$runner = Join-Path $PSScriptRoot 'run_hidden.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
$days = @('/SC', 'WEEKLY', '/D', 'MON,TUE,WED,THU,FRI')

function New-GoldNYTask($name, $pass, $schedule) {
    $cmd = "`"$wscript`" //B //Nologo `"$runner`" $pass $name"
    if ($cmd.Length -gt 261) { throw "task command is $($cmd.Length) characters; schtasks allows 261" }
    $args = @('/Create', '/F', '/TN', $name, '/TR', $cmd) + $schedule
    if ($DryRun) {
        Write-Output ("DRY RUN ({0} chars) schtasks.exe {1}" -f $cmd.Length, ($args -join ' '))
        return
    }
    & schtasks.exe @args | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "schtasks failed for $name (exit $LASTEXITCODE)" }
    Write-Output "installed $name"
}

New-GoldNYTask 'EURUSDProphet-GoldNYForecast' 'forecast' ($days + @('/ST', '16:08'))
New-GoldNYTask 'EURUSDProphet-GoldNYSettle' 'settle' ($days + @('/ST', '23:08'))
