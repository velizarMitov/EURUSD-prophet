# Remove the monthly gold refit task. Versions already written stay on disk, and
# the active shadow version keeps forecasting until versions\active.json is removed.
# Safe to run when the task does not exist.

$name = 'EURUSDProphet-GoldRetrain'
& schtasks.exe /Query /TN $name 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /F /TN $name | Out-Null
    Write-Output "removed $name"
} else {
    Write-Output "not installed: $name"
}
