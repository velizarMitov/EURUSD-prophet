# Remove the rts_signals scheduled task, and only this one. The logs and results stay.
$names = @('EURUSDProphet-RTSInbox')
foreach ($name in $names) {
    & schtasks.exe /Query /TN $name 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /Delete /F /TN $name | Out-Null
        Write-Output "removed $name"
    } else {
        Write-Output "not installed: $name"
    }
}
