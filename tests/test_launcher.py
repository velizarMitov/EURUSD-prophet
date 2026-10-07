"""The one-click launcher (openspec change `one-click-launcher`).

The launcher itself is VBScript driving WMI and HTTP; its behaviour was verified
by hand (tasks 2.1-2.2). These checks pin the properties that, if lost, would
quietly bring back the old problems: a console window, killing a process that
is not ours, a browser opened on a dead port, or a pile of launch files.
"""
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'scripts')


def _launcher() -> str:
    raw = open(os.path.join(SCRIPTS, 'launch.vbs'), 'rb').read()
    # wscript reads non-ASCII only from UTF-16, so the working-tree copy is
    # UTF-16LE with a BOM (.gitattributes working-tree-encoding).
    assert raw[:2] == b'\xff\xfe', 'launch.vbs must be UTF-16LE with BOM for the Bulgarian messages'
    return raw[2:].decode('utf-16-le')


def test_server_starts_hidden_and_does_not_block():
    s = _launcher()
    assert re.search(r'sh\.Run "cmd /c .*?, 0, False', s), 'window style 0, no wait'
    assert '-u " & entry' in s, 'unbuffered, so the log is current'


def test_probes_the_local_port_and_waits_up_to_three_minutes():
    s = _launcher()
    assert 'Const URL = "http://127.0.0.1:8000/"' in s
    assert 'Const START_BUDGET_S = 180' in s
    assert 'HttpStatus() = 200 Then' in s, 'the browser opens only on a real 200'


def test_logs_to_the_gitignored_server_log():
    s = _launcher()
    assert '\\research_models\\server' in s and 'server.log' in s
    ignored = subprocess.run(['git', 'check-ignore', '-q', 'research_models/server/server.log'],
                             cwd=REPO).returncode
    assert ignored == 0, 'the server log must not be tracked'


def test_only_our_own_server_is_ever_terminated():
    """start.bat killed whatever held port 8000. The launcher may end only a
    python.exe under this repo's .venv running the entry file, and its children."""
    s = _launcher()
    assert 'taskkill' not in s.lower()
    assert 'prefix = LCase(repo & "\\.venv\\")' in s
    assert 'InStr(cmdl, LCase(entry))' in s
    assert 'StopTree ours.ProcessId' in s
    assert s.count('.Terminate') == 1, 'one termination site, inside StopTree'


def test_a_foreign_program_on_the_port_is_reported_not_killed():
    s = _launcher()
    assert 'If answers And ours Is Nothing Then' in s
    block = s.split('If answers And ours Is Nothing Then', 1)[1].split('End If', 1)[0]
    assert 'Порт 8000 е зает от друга програма' in block
    assert 'StopTree' not in block and 'StartServer' not in block


def test_stale_code_triggers_a_restart_but_html_does_not():
    s = _launcher()
    assert 'repo & "\\api.py"' in s and 'repo & "\\src"' in s
    assert '= "py" Then' in s, 'only .py files count'
    assert '__pycache__' in s
    assert 'static' not in s.split('Function NewestSource()', 1)[1].split('End Function', 1)[0]


def test_failure_message_is_bulgarian_and_names_the_log():
    s = _launcher()
    assert 'Програмата не успя да стартира.' in s
    assert 'Причината е записана в:" & vbCrLf & logPath' in s


def test_shortcut_installer_targets_wscript_batch_mode():
    s = open(os.path.join(SCRIPTS, 'install_shortcut.ps1'), encoding='utf-8').read()
    assert "System32\\wscript.exe" in s
    assert '"//B //Nologo `"$launcher`""' in s
    assert "'launch.vbs'" in s and 'static\\prophet.ico' in s
    assert "'EUR-USD Prophet.lnk'" in s, 'a fixed name, so a rerun replaces it'
    assert os.path.getsize(os.path.join(REPO, 'static', 'prophet.ico')) > 0


def test_no_redundant_launch_files_are_tracked():
    files = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True,
                           text=True).stdout.splitlines()
    present = [f for f in files if os.path.exists(os.path.join(REPO, f))]
    assert not [f for f in present if f.lower().endswith('.bat')], 'no .bat files'
    cmds = [f for f in present if f.lower().endswith('.cmd')]
    assert cmds == ['scripts/forecast_eval/run_module.cmd'], cmds
