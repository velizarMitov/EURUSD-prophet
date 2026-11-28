# Remove the gold-nyrange-forward scheduled tasks, and only these two. The logs and results stay.
$names = @('EURUSDProphet-GoldNYForecast', 'EURUSDProphet-GoldNYSettle')
foreach ($name in $names) {
    & schtasks.exe /Query /TN $name 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /Delete /F /TN $name | Out-Null
        Write-Output "removed $name"
    } else {
        Write-Output "not installed: $name"
    }
}
