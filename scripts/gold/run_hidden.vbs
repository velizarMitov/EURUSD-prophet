' Run a gold module with NO console window (same reason as
' scripts\forecast_eval\run_hidden.vbs: a console host would flash a window every
' hour; wscript.exe is a GUI host and window style 0 keeps the child hidden).
' Python itself appends the output to research_models\gold\<logname>.log
' (src\gold\run_task.py), so no .cmd wrapper is needed.
'
' Usage (from the task):  wscript.exe //B //Nologo run_hidden.vbs <module> <logname>
Option Explicit
Dim fso, sh, env, repo, python
If WScript.Arguments.Count < 2 Then WScript.Quit 2
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
repo = fso.GetParentFolderName(fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName)))
python = repo & "\.venv\Scripts\python.exe"
Set env = sh.Environment("PROCESS")
env("PYTHONIOENCODING") = "utf-8"
env("TF_CPP_MIN_LOG_LEVEL") = "3"
sh.CurrentDirectory = repo
' 0 = hidden window, True = wait, so Task Scheduler sees the real exit code
WScript.Quit sh.Run("""" & python & """ -m src.gold.run_task " & WScript.Arguments(0) & " " & WScript.Arguments(1), 0, True)
