# One-time, read-only Gmail consent for the rts_signals family (openspec change rts-signals).
#
# Before running: save the Google Cloud OAuth client (Desktop app) JSON as
#   research_models\rts_signals\secrets\client_secret.json
# A browser opens; sign in and allow "View your email messages and settings" (gmail.readonly).
# The token is written next to it, in the gitignored secrets folder. Nothing is printed.
#
#   powershell -ExecutionPolicy Bypass -File scripts\rts_signals\authorize_gmail.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\rts_signals\authorize_gmail.ps1 -Status

param([switch]$Status)

$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$py = Join-Path $repo '.venv\Scripts\python.exe'
Push-Location $repo
try {
    if ($Status) {
        & $py -m src.rts_signals.gmail --status
    } else {
        New-Item -ItemType Directory -Force (Join-Path $repo 'research_models\rts_signals\secrets') | Out-Null
        & $py -m src.rts_signals.gmail --authorize
    }
} finally {
    Pop-Location
}
