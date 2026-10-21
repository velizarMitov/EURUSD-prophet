' EUR/USD Prophet - double-click me to start the app (openspec change findable-launcher).
'
' This file only hands off to scripts\launch.vbs, the same launcher the desktop and
' Start-menu icons use: hidden start, reuse of a running server, restart on stale
' code, and the error messages all live there. Keep this file plain ASCII and free
' of logic so it can never drift from the icons.
Option Explicit
Dim fso, sh, here
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
' 0 = hidden window, False = do not wait: launch.vbs reports its own failures.
sh.Run """" & sh.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\wscript.exe"" //B //Nologo """ & _
       here & "\scripts\launch.vbs""", 0, False
