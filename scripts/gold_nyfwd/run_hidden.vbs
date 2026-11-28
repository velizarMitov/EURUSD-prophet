' Run one gold-nyrange-forward pass with NO console window (wscript is a GUI-subsystem host).
' Usage (from the task):  wscript.exe //B //Nologo run_hidden.vbs <pass> <logname>
Option Explicit
Dim fso, sh, here, cmd
If WScript.Arguments.Count < 2 Then WScript.Quit 2
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & here & "\run_module.cmd"" " & WScript.Arguments(0) & " " & WScript.Arguments(1)
' 0 = hidden window, True = wait, so Task Scheduler sees the real exit code
WScript.Quit sh.Run(cmd, 0, True)
