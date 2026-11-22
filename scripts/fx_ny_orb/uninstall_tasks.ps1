# Remove the fx-ny-orb-hot scheduled tasks. The logs and results stay.
$names = @('EURUSDProphet-NYOrbForecast', 'EURUSDProphet-NYOrbSignal', 'EURUSDProphet-NYOrbSettle')
foreach ($name in $names) {
    & schtasks.exe /Query /TN $name 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /Delete /F /TN $name | Out-Null
        Write-Output "removed $name"
    } else {
        Write-Output "not installed: $name"
    }
}
