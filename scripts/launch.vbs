' EUR/USD Prophet - the desktop icon's target (openspec change one-click-launcher).
'
' One double-click:
'   nothing on port 8000            -> start the server hidden, open the browser when it answers
'   our server, current code        -> just open the browser
'   our server, code changed since  -> stop it, start a fresh one, open the browser
'   another program holds port 8000 -> say so, touch nothing
'
' Runs under wscript.exe, a GUI host, so no console window ever appears. That is
' the same reason scripts\forecast_eval\run_hidden.vbs exists.
'
' "Ours" is decided from WMI, never from a state file: a python.exe under this
' repo's .venv whose command line runs api.py. Its CreationDate is the server's
' start time. Only such processes, and their children, are ever terminated.
' The old start.bat killed whatever held the port.
'
' Saved as UTF-16 (see .gitattributes) so wscript reads the Bulgarian messages.
'
' Usage: wscript.exe //B //Nologo launch.vbs [/entry:<file>]
'        /entry is for the manual failure test only; it replaces api.py.
Option Explicit

Const URL = "http://127.0.0.1:8000/"
Const START_BUDGET_S = 180
Const TITLE = "EUR/USD Prophet"

Dim fso, sh, repo, entry, logDir, logPath, py, wmi
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
repo = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
entry = "api.py"
If WScript.Arguments.Named.Exists("entry") Then entry = WScript.Arguments.Named("entry")
logDir = repo & "\research_models\server"
logPath = logDir & "\server.log"
py = repo & "\.venv\Scripts\python.exe"
Set wmi = GetObject("winmgmts:\\.\root\cimv2")

Main

Sub Main()
    Dim ours, answers
    If Not fso.FileExists(py) Then
        Fail "Не е намерен Python в папката на проекта:" & vbCrLf & py
    End If
    Set ours = OurServer()
    answers = PortAnswers()

    If answers And ours Is Nothing Then
        MsgBox "Порт 8000 е зает от друга програма." & vbCrLf & vbCrLf & _
               "EUR/USD Prophet не я спира. Затвори другата програма и опитай пак.", _
               vbExclamation, TITLE
        WScript.Quit 3
    End If

    If Not (ours Is Nothing) Then
        If answers And Not IsStale(ours) Then
            OpenBrowser
            WScript.Quit 0
        End If
        If answers Then
            ' Running on code older than what is on disk: restart it.
            StopTree ours.ProcessId
            WaitPortFree 20
            StartServer
        End If
        ' Not answering yet: it is still starting; just wait below.
    Else
        StartServer
    End If

    WaitAndOpen
End Sub

' ── ownership and staleness ────────────────────────────────────────────────

Function OurServer()
    ' The .venv launcher process. On Windows a venv python.exe re-launches the
    ' base interpreter as its child, so StopTree also ends the children.
    Dim procs, p, exe, cmdl, prefix
    Set OurServer = Nothing
    prefix = LCase(repo & "\.venv\")
    Set procs = wmi.ExecQuery("SELECT ProcessId, CommandLine, ExecutablePath, CreationDate " & _
                              "FROM Win32_Process WHERE Name = 'python.exe'")
    For Each p In procs
        exe = LCase("" & p.ExecutablePath)
        cmdl = LCase("" & p.CommandLine)
        If Left(exe, Len(prefix)) = prefix And InStr(cmdl, LCase(entry)) > 0 Then
            Set OurServer = p
            Exit Function
        End If
    Next
End Function

Function IsStale(proc)
    Dim started
    started = WmiDate(proc.CreationDate)
    IsStale = NewestSource() > started
End Function

Function NewestSource()
    ' api.py and every .py under src\. HTML is served from disk on each request,
    ' so a page change never needs a restart and is not counted.
    Dim newest
    newest = fso.GetFile(repo & "\api.py").DateLastModified
    NewestSource = NewestIn(fso.GetFolder(repo & "\src"), newest)
End Function

Function NewestIn(folder, newest)
    Dim f, sub_
    For Each f In folder.Files
        If LCase(fso.GetExtensionName(f.Name)) = "py" Then
            If f.DateLastModified > newest Then newest = f.DateLastModified
        End If
    Next
    For Each sub_ In folder.SubFolders
        If LCase(sub_.Name) <> "__pycache__" Then newest = NewestIn(sub_, newest)
    Next
    NewestIn = newest
End Function

Function WmiDate(s)
    ' "20261007093412.123456+180" -> local Date (WMI CreationDate is local time).
    WmiDate = DateSerial(CInt(Mid(s, 1, 4)), CInt(Mid(s, 5, 2)), CInt(Mid(s, 7, 2))) + _
              TimeSerial(CInt(Mid(s, 9, 2)), CInt(Mid(s, 11, 2)), CInt(Mid(s, 13, 2)))
End Function

Sub StopTree(pid)
    Dim kids, k
    Set kids = wmi.ExecQuery("SELECT ProcessId FROM Win32_Process WHERE ParentProcessId = " & pid)
    For Each k In kids
        StopTree k.ProcessId
    Next
    On Error Resume Next
    Dim me_
    For Each me_ In wmi.ExecQuery("SELECT * FROM Win32_Process WHERE ProcessId = " & pid)
        me_.Terminate
    Next
    On Error GoTo 0
End Sub

' ── the port ───────────────────────────────────────────────────────────────

Function HttpStatus()
    ' 0 when nothing answers.
    Dim http
    HttpStatus = 0
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    http.setTimeouts 1000, 1000, 2000, 2000
    http.Open "GET", URL, False
    http.Send
    If Err.Number = 0 Then HttpStatus = http.Status
    On Error GoTo 0
End Function

Function PortAnswers()
    PortAnswers = HttpStatus() <> 0
End Function

Sub WaitPortFree(seconds)
    Dim i
    For i = 1 To seconds
        If Not PortAnswers() Then Exit Sub
        WScript.Sleep 1000
    Next
End Sub

' ── start, wait, open ──────────────────────────────────────────────────────

Sub StartServer()
    Dim log_
    If Not fso.FolderExists(repo & "\research_models") Then fso.CreateFolder repo & "\research_models"
    If Not fso.FolderExists(logDir) Then fso.CreateFolder logDir
    Set log_ = fso.OpenTextFile(logPath, 8, True)
    log_.WriteLine ""
    log_.WriteLine "===== launch " & Now & " (" & entry & ") ====="
    log_.Close
    sh.Environment("Process")("PYTHONIOENCODING") = "utf-8"
    sh.CurrentDirectory = repo
    ' 0 = hidden window, False = do not wait: the server keeps running.
    sh.Run "cmd /c """"" & py & """ -u " & entry & " >> """ & logPath & """ 2>&1""", 0, False
End Sub

Sub WaitAndOpen()
    Dim i
    For i = 1 To START_BUDGET_S
        If HttpStatus() = 200 Then
            OpenBrowser
            WScript.Quit 0
        End If
        ' Give the process a few seconds to appear before treating its absence
        ' as a crash.
        If i > 5 Then
            If OurServer() Is Nothing Then Exit For
        End If
        WScript.Sleep 1000
    Next
    Fail "Програмата не успя да стартира." & vbCrLf & vbCrLf & _
         "Причината е записана в:" & vbCrLf & logPath
End Sub

Sub OpenBrowser()
    sh.Run URL, 1, False
End Sub

Sub Fail(msg)
    MsgBox msg, vbCritical, TITLE
    WScript.Quit 1
End Sub
