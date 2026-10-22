"""The app shell (openspec change app-shell-and-gold-history, capability app-shell)."""
import os
import re

import pytest

from src import app_shell as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_DIR = os.path.join(REPO, 'static', 'shell')


def _read(*parts):
    with open(os.path.join(REPO, *parts), encoding='utf-8') as fh:
        return fh.read()


# ── 1. The navigation source ────────────────────────────────────────────────

def test_nav_carries_one_marker_and_marks_the_current_page():
    nav = S.render_nav('/gold')
    assert nav.count(S.MARKER) == 1
    assert nav.count('aria-current="page"') == 1
    assert re.search(r'href="/gold" aria-current="page"', nav)


def test_more_menu_marks_its_current_item():
    nav = S.render_nav('/kronos-volatility')
    assert re.search(r'href="/kronos-volatility" aria-current="page"', nav)
    assert 'ps-more-current' in nav


def test_retired_view_is_not_listed():
    for path in S.PAGE_PATHS:
        assert '/kronos-direction' not in S.render_nav(path)


def test_every_label_exists_in_both_languages():
    for _, text in S.NAV + S.MORE:
        assert text['bg'].strip() and text['en'].strip()
    for key, text in S.CONTROLS.items():
        assert text['bg'].strip() and text['en'].strip(), key


def test_inject_is_idempotent_and_strip_is_its_inverse():
    page = '<!doctype html><html><head><title>x</title></head><body class="a"><p>hi</p></body></html>'
    once = S.inject(page, '/history')
    assert S.inject(once, '/history') == once
    assert once.count(S.MARKER) == 1
    assert once.index('data-ps-css') < once.index('</head>') < once.index(S.MARKER)
    assert S.strip(once, '/history') == page


def test_inject_handles_a_bare_fragment():
    page = '<h1>H1</h1><table></table>'
    out = S.inject(page, '/h1-direction')
    assert out.startswith(S.head_snippet()) and out.endswith(page)
    assert S.strip(out, '/h1-direction') == page


# ── 2. The shell assets ─────────────────────────────────────────────────────

EXTERNAL = re.compile(r'''(?:src|href)\s*=\s*["']https?://|@import|url\(\s*["']?https?://''', re.I)


def test_shell_assets_load_nothing_external():
    for name in ('shell.css', 'shell.js'):
        assert not EXTERNAL.search(_read('static', 'shell', name)), name
    assert not EXTERNAL.search(S.head_snippet() + S.render_nav('/'))


def test_base_layer_has_zero_specificity():
    """The base layer that every page gets sits in a cascade layer AND inside
    :where(), so a pinned page's own unlayered styles always win."""
    css = re.sub(r'/\*.*?\*/', '', _read('static', 'shell', 'shell.css'), flags=re.S)
    blocks = css.split('@layer ps-base {')[1:]
    assert blocks, 'shell.css has no ps-base layer'
    selectors = []
    for block in blocks:
        selectors += [s.strip() for s in re.findall(r'([^{}]+)\{', block.split('\n}\n', 1)[0])]
    assert selectors
    for sel in selectors:
        assert sel.startswith(':where('), sel


def test_shell_script_guards_every_storage_call():
    js = _read('static', 'shell', 'shell.js')
    calls = [m.start() for m in re.finditer(r'localStorage\.(?:get|set)Item', js)]
    assert calls
    for at in calls:
        assert 'try {' in js[max(0, at - 120):at], js[max(0, at - 120):at + 40]


def test_shell_script_honours_the_old_home_language_key():
    js = _read('static', 'shell', 'shell.js')
    assert "'prophet.lang'" in js and "'eurusd.home.lang'" in js
    assert "'ps:lang'" in js


# ── 3. Every page route gets the shell, nothing else is touched ────────────

@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


@pytest.mark.parametrize('path', S.PAGE_PATHS)
def test_every_page_route_carries_the_shell_once(client, path):
    r = client.get(path)
    assert r.status_code == 200, path
    html_ = r.text
    for part in (S.MARKER, 'data-ps-css', 'data-ps-js', 'data-ps-boot'):
        assert html_.count(part) == 1, (path, part, html_.count(part))
    assert int(r.headers['content-length']) == len(r.content)
    current = re.findall(r'href="([^"]+)" aria-current="page"', html_)
    assert current == ([path] if path != '/kronos-direction' else []), current


@pytest.mark.parametrize('path', ['/api/gold', '/api/home', '/api/provenance', '/ui/shell/shell.css'])
def test_non_page_responses_carry_no_shell(client, path):
    r = client.get(path)
    assert r.status_code == 200, path
    assert S.MARKER not in r.text and 'data-ps-css' not in r.text


def test_layer_passes_non_html_bodies_through_byte_for_byte():
    """A JSON body on a page path (and every non-page path) leaves exactly as sent."""
    import asyncio
    import api

    payload = b'{"a": "<body></head>"}'

    async def inner(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'application/json'),
                                (b'content-length', str(len(payload)).encode())]})
        await send({'type': 'http.response.body', 'body': payload})

    for path in ('/gold', '/api/gold'):
        sent = []

        async def send(message):
            sent.append(message)

        asyncio.run(api.ShellLayer(inner)({'type': 'http', 'method': 'GET', 'path': path}, None, send))
        assert b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body') == payload


def test_layer_sends_the_original_page_when_injection_fails(monkeypatch):
    import asyncio
    import api

    page = '<html><head></head><body>ok</body></html>'.encode()

    async def inner(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'text/html; charset=utf-8')]})
        await send({'type': 'http.response.body', 'body': page})

    def boom(*a, **k):
        raise RuntimeError('broken shell')

    monkeypatch.setattr(S, 'inject', boom)
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(api.ShellLayer(inner)({'type': 'http', 'method': 'GET', 'path': '/history'}, None, send))
    assert sent[-1]['body'] == page


# ── 4. Pinned sources stay pinned ───────────────────────────────────────────

def _git_diff(*args):
    import subprocess
    out = subprocess.run(['git', '-C', REPO, 'diff', *args], capture_output=True, text=True)
    if out.returncode not in (0, 1):
        pytest.skip('git unavailable')
    return out


def test_pinned_page_sources_are_byte_identical():
    for rel in ('src/tracking.py', 'src/paper_trading.py', 'src/inference.py', 'static/index.html'):
        assert _git_diff('--quiet', 'HEAD', '--', rel).returncode == 0, f'{rel} changed'


def test_api_py_only_gains_lines():
    out = _git_diff('--numstat', 'HEAD', '--', 'api.py').stdout.strip()
    for line in out.splitlines():
        added, removed, path = line.split('\t')
        assert removed == '0', f'{path} deleted {removed} lines'


def test_the_console_differs_only_by_the_shell(client):
    r = client.get('/advanced')
    with open(os.path.join(REPO, 'static', 'index.html'), 'rb') as fh:
        on_disk = fh.read()
    assert S.strip(r.text, '/advanced').encode('utf-8') == on_disk
