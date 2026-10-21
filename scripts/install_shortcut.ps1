# Create (or replace) the "EUR-USD Prophet" shortcuts: one on the desktop and one
# in the Start menu, so Win + "Prophet" finds the app (openspec changes
# one-click-launcher, findable-launcher). Run once:
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
#
# Both shortcuts run scripts\launch.vbs through wscript.exe //B. wscript is a GUI
# host, so no console window appears. Double-clicking starts the server hidden,
# or just opens the browser when it is already running on current code.
# Running this script again overwrites the same two shortcuts; it never makes more.
# Without any shortcut, "Start EUR-USD Prophet.vbs" in the project folder does the same.

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$launcher = Join-Path $PSScriptRoot 'launch.vbs'
$icon = Join-Path $repo 'static\prophet.ico'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'
foreach ($p in @($launcher, $icon, (Join-Path $repo '.venv\Scripts\python.exe'))) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

# 'Programs' is the per-user Start-menu folder, so no administrator rights are needed.
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @('Desktop', 'Programs')) {
    $path = Join-Path ([Environment]::GetFolderPath($folder)) 'EUR-USD Prophet.lnk'
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath = $wscript
    $lnk.Arguments = "//B //Nologo `"$launcher`""
    $lnk.WorkingDirectory = $repo
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = 'EUR/USD Prophet: starts the app hidden and opens it in the browser'
    $lnk.Save()
    Write-Output "shortcut: $path"
}
