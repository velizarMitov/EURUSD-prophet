# Remove the external models' scheduled task. Logged rows stay on disk; only the
# schedule is removed. Safe to run when the task does not exist.

$name = 'EURUSDProphet-ExternalLogger'
& schtasks.exe /Query /TN $name 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    & schtasks.exe /Delete /F /TN $name | Out-Null
    Write-Output "removed $name"
} else {
    Write-Output "not installed: $name"
}
