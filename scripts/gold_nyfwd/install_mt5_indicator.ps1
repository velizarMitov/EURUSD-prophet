# Install the MT5 chart indicator for the gold NY session size (openspec change gold-nyrange-forward,
# spec gold-nyrange-forward/chart-display).
#
# For every MT5 terminal on this PC (each %APPDATA%\MetaQuotes\Terminal\<id>\origin.txt):
#   copy mql5\Indicators\ProphetGoldNYRange.mq5 to <data>\MQL5\Indicators\EURUSDProphet\,
#   compile it there with that terminal's own metaeditor64.exe,
#   print one line: compiled, or FAILED followed by the compiler log.
# Then rebuild the chart file once (python -m src.gold_nyfwd.export).
#
# Display only: the indicator never places an order. Changes nothing else in a terminal (the other
# indicators in the same folder are left alone). Afterwards, in MT5: Navigator > Indicators >
# EURUSDProphet > ProphetGoldNYRange, drag it onto a GOLD H1 (or lower) chart.
# Remove with uninstall_mt5_indicator.ps1. See results\gold_nyfwd\RUNBOOK.md.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$source = Join-Path $repo 'mql5\Indicators\ProphetGoldNYRange.mq5'
if (-not (Test-Path $python)) { throw "virtualenv python not found: $python" }
if (-not (Test-Path $source)) { throw "indicator source not found: $source" }

$root = Join-Path $env:APPDATA 'MetaQuotes\Terminal'
$terminals = @(Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName 'origin.txt') })
if ($terminals.Count -eq 0) { Write-Host "no MT5 terminal found under $root" }

$failed = 0
foreach ($t in $terminals) {
    $install = (Get-Content (Join-Path $t.FullName 'origin.txt') -Raw).Trim()
    $label = "$install [$($t.Name.Substring(0, 8))]"
    $editor = Join-Path $install 'metaeditor64.exe'
    if (-not (Test-Path $editor)) {
        Write-Host "FAILED   $label : metaeditor64.exe not found"
        $failed++
        continue
    }
    $dir = Join-Path $t.FullName 'MQL5\Indicators\EURUSDProphet'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $mq5 = Join-Path $dir 'ProphetGoldNYRange.mq5'
    $ex5 = Join-Path $dir 'ProphetGoldNYRange.ex5'
    $log = Join-Path $dir 'ProphetGoldNYRange.compile.log'
    Copy-Item $source $mq5 -Force
    Remove-Item $ex5, $log -ErrorAction SilentlyContinue
    # MetaEditor's exit code is not reliable: the .ex5 and the log's error count decide.
    Start-Process -FilePath $editor -ArgumentList "/compile:`"$mq5`"", "/log:`"$log`"" -Wait -WindowStyle Hidden
    $text = if (Test-Path $log) { Get-Content $log -Raw -Encoding Unicode } else { '(no compiler log)' }
    if ((Test-Path $ex5) -and $text -match 'Result: 0 errors') {
        Write-Host "compiled $label"
    } else {
        Write-Host "FAILED   $label"
        Write-Host $text
        $failed++
    }
}

$env:PYTHONIOENCODING = 'utf-8'
Push-Location $repo
try { & $python -m src.gold_nyfwd.export } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw 'the chart file could not be written: see the error above' }
if ($failed -gt 0) { exit 1 }
