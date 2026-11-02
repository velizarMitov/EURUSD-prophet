# Remove what install_mt5_indicator.ps1 added (openspec change gold-mt5-chart-signals):
# each terminal's MQL5\Indicators\EURUSDProphet\ folder and the signals folder
# Common\Files\EURUSDProphet\. Nothing else in a terminal is touched.
#
# Take the indicator off open charts first: a chart keeps a loaded indicator
# until it is removed or the terminal restarts.

$ErrorActionPreference = 'Stop'
$root = Join-Path $env:APPDATA 'MetaQuotes\Terminal'
$terminals = @(Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName 'origin.txt') })
foreach ($t in $terminals) {
    $install = (Get-Content (Join-Path $t.FullName 'origin.txt') -Raw).Trim()
    $dir = Join-Path $t.FullName 'MQL5\Indicators\EURUSDProphet'
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "removed  $install [$($t.Name.Substring(0, 8))]"
    } else {
        Write-Host "absent   $install [$($t.Name.Substring(0, 8))]"
    }
}

$common = if ($env:MT5_COMMON_FILES) { $env:MT5_COMMON_FILES } else { Join-Path $root 'Common\Files' }
$signals = Join-Path $common 'EURUSDProphet'
if (Test-Path $signals) {
    Remove-Item $signals -Recurse -Force
    Write-Host "removed  $signals"
} else {
    Write-Host "absent   $signals"
}
