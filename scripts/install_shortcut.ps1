# Create (or replace) the "EUR-USD Prophet" desktop icon (openspec change
# one-click-launcher). Run once:
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
#
# The icon runs scripts\launch.vbs through wscript.exe //B. wscript is a GUI
# host, so no console window appears. Double-clicking starts the server hidden,
# or just opens the browser when it is already running on current code.
# Running this script again overwrites the same shortcut; it never makes a second one.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$launcher = Join-Path $PSScriptRoot 'launch.vbs'
$icon = Join-Path $repo 'static\prophet.ico'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
foreach ($p in @($launcher, $icon, (Join-Path $repo '.venv\Scripts\python.exe'))) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

$desktop = [Environment]::GetFolderPath('Desktop')
$path = Join-Path $desktop 'EUR-USD Prophet.lnk'
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($path)
$lnk.TargetPath = $wscript
$lnk.Arguments = "//B //Nologo `"$launcher`""
$lnk.WorkingDirectory = $repo
$lnk.IconLocation = "$icon,0"
$lnk.Description = 'EUR/USD Prophet: starts the app hidden and opens it in the browser'
$lnk.Save()
Write-Output "shortcut: $path"
