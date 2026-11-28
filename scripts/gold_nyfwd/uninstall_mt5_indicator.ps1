# Remove the gold NY session-size indicator (openspec change gold-nyrange-forward) from every MT5
# terminal on this PC, and its chart file. Removes ONLY ProphetGoldNYRange.* and gold_ny_range.tsv:
# the shared EURUSDProphet folders, ProphetNYOrb, ProphetGoldSignals and their files stay.

$ErrorActionPreference = 'Stop'
$root = Join-Path $env:APPDATA 'MetaQuotes\Terminal'
$terminals = @(Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
               Where-Object { Test-Path (Join-Path $_.FullName 'origin.txt') })
foreach ($t in $terminals) {
    $install = (Get-Content (Join-Path $t.FullName 'origin.txt') -Raw).Trim()
    $dir = Join-Path $t.FullName 'MQL5\Indicators\EURUSDProphet'
    $files = @(Get-ChildItem $dir -Filter 'ProphetGoldNYRange.*' -ErrorAction SilentlyContinue)
    if ($files.Count) {
        $files | Remove-Item -Force
        Write-Host "removed  $install [$($t.Name.Substring(0, 8))]"
    } else {
        Write-Host "absent   $install [$($t.Name.Substring(0, 8))]"
    }
}

$common = if ($env:MT5_COMMON_FILES) { $env:MT5_COMMON_FILES } else { Join-Path $root 'Common\Files' }
$file = Join-Path $common 'EURUSDProphet\gold_ny_range.tsv'
if (Test-Path $file) {
    Remove-Item $file -Force
    Write-Host "removed  $file"
} else {
    Write-Host "absent   $file"
}
