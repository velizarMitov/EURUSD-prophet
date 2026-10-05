' Run a forward-evaluation module with NO console window.
'
' The scheduled tasks used to call run_module.cmd directly. cmd.exe is a console
' program, so Task Scheduler opened a window for it on every run -- a flash of
' PowerShell/cmd every 15 minutes. wscript.exe is a GUI-subsystem host and shows
' nothing; window style 0 below keeps the child hidden as well.
'
' Usage (from the task):  wscript.exe //B //Nologo run_hidden.vbs <module> <logname>
Option Explicit
Dim fso, sh, here, cmd
If WScript.Arguments.Count < 2 Then WScript.Quit 2
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & here & "\run_module.cmd"" " & WScript.Arguments(0) & " " & WScript.Arguments(1)
' 0 = hidden window, True = wait, so Task Scheduler sees the real exit code
WScript.Quit sh.Run(cmd, 0, True)
