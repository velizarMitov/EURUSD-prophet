# Install the side environment for the external foundation models
# (openspec change tsfm-second-opinion, design D1/D2). Safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\external_tsfm\install.ps1
#
# 1. creates .venv-tsfm with Python 3.11 (the app's .venv is never touched)
# 2. installs requirements-tsfm.txt and freezes the result to
#    results\external_tsfm\env_freeze.txt
# 3. reports whether the GPU is visible
# 4. fetches the pinned weights (python -m src.external.tsfm.vendor --fetch)
# 5. smoke test: each model returns 9 increasing deciles for a synthetic series
param([switch]$SkipFetch)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $repo
$venv = Join-Path $repo '.venv-tsfm'
$side = Join-Path $venv 'Scripts\python.exe'
$main = Join-Path $repo '.venv\Scripts\python.exe'

Write-Host '[1/5] side environment'
if (-not (Test-Path $side)) {
    & py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'could not create .venv-tsfm with Python 3.11 (py -3.11)' }
}

Write-Host '[2/5] pinned packages'
& $side -m pip install --disable-pip-version-check -q -r (Join-Path $repo 'requirements-tsfm.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install -r requirements-tsfm.txt failed' }
New-Item -ItemType Directory -Force (Join-Path $repo 'results\external_tsfm') | Out-Null
$freeze = & $side -m pip freeze --disable-pip-version-check
[System.IO.File]::WriteAllLines((Join-Path $repo 'results\external_tsfm\env_freeze.txt'), [string[]]$freeze)

Write-Host '[3/5] GPU'
& $side -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

if (-not $SkipFetch) {
    Write-Host '[4/5] pinned weights'
    & $main -m src.external.tsfm.vendor --fetch
    if ($LASTEXITCODE -ne 0) { throw 'fetching the pinned weights failed' }
}

Write-Host '[5/5] smoke test'
$weights = @{
    chronos2  = (Join-Path $repo 'research_models\external_tsfm\chronos2')
    timesfm25 = (Join-Path $repo 'research_models\external_tsfm\timesfm25')
}
$reqFile = Join-Path $repo 'research_models\external_tsfm\smoke_request.json'
[System.IO.File]::WriteAllText($reqFile, (@{ op = 'smoke'; weights = $weights } | ConvertTo-Json -Compress))
$out = & $side -I (Join-Path $repo 'src\external\tsfm\worker.py') --request $reqFile
Write-Host $out
$reply = $out | ConvertFrom-Json
if (-not $reply.ok) { throw "smoke test failed: $($reply.error)" }
foreach ($m in 'chronos2', 'timesfm25') {
    if (-not $reply.$m.increasing) { throw "$m did not return 9 increasing deciles" }
}
Write-Host 'OK: both models return 9 increasing deciles.'
