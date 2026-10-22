"""The app shell: one navigation bar, one look, one language and theme choice
(openspec change app-shell-and-gold-history, capability app-shell).

This module is the ONLY source of the navigation markup. api.py's ShellLayer
injects it into every page route's HTML as the response leaves, which is how
pages rendered by byte-pinned modules (src/tracking.py, src/paper_trading.py)
and the byte-identical research console (static/index.html) get the shell
without a line of their source changing.

Exactly two segments are injected, and `strip` removes exactly those two, so a
test can prove a page is unchanged apart from the shell:
    head_snippet() before </head> (or at the very start of a page with no head)
    the <nav>     right after <body ...> (or after the head snippet)

The shell loads nothing from outside the application: its assets are served
from /ui/shell/ by the existing static mount.
"""

from __future__ import annotations

import hashlib
import html
import os
import re

MARKER = 'data-ps-shell'

# Page routes the layer decorates. An explicit list, never a prefix match, so a
# JSON or CSV route can never be rewritten by accident.
PAGE_PATHS = ('/', '/gold', '/advanced', '/history', '/paper-trading', '/forecasts',
              '/h1-direction', '/kronos-volatility', '/kronos-direction')

BRAND = {'bg': 'EUR/USD Prophet', 'en': 'EUR/USD Prophet'}

NAV = (
    ('/', {'bg': 'Начало', 'en': 'Home'}),
    ('/gold', {'bg': 'Злато', 'en': 'Gold'}),
    ('/forecasts', {'bg': 'Прогнози', 'en': 'Forecasts'}),
    ('/history', {'bg': 'История', 'en': 'History'}),
    ('/paper-trading', {'bg': 'Симулация', 'en': 'Paper trading'}),
    ('/advanced', {'bg': 'Конзола', 'en': 'Research console'}),
)

# Less-used views. /kronos-direction is retired: reachable by URL, never listed.
MORE = (
    ('/h1-direction', {'bg': 'H1 посока', 'en': 'H1 direction'}),
    ('/kronos-volatility', {'bg': 'Kronos волатилност', 'en': 'Kronos volatility'}),
)

CONTROLS = {
    'more': {'bg': 'Още', 'en': 'More'},
    'menu': {'bg': 'Меню', 'en': 'Menu'},
    'language': {'bg': 'Език', 'en': 'Language'},
    'theme': {'bg': 'Тема', 'en': 'Theme'},
    'system': {'bg': 'Системна', 'en': 'System'},
    'light': {'bg': 'Светла', 'en': 'Light'},
    'dark': {'bg': 'Тъмна', 'en': 'Dark'},
    'main_nav': {'bg': 'Основна навигация', 'en': 'Main navigation'},
}

SHELL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'shell')


def asset_version() -> str:
    """A short tag that changes whenever shell.css or shell.js changes on disk,
    so a browser that cached an older copy fetches the new one. Two stat calls
    per page, no read."""
    parts = []
    for name in ('shell.css', 'shell.js'):
        try:
            st = os.stat(os.path.join(SHELL_DIR, name))
            parts.append(f'{st.st_mtime_ns}:{st.st_size}')
        except OSError:
            parts.append('-')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:10]


def head_snippet() -> str:
    """Sets data-theme before the first paint so a page never flashes the other
    theme (storage can throw in private mode: System it is), then loads the
    shell's two assets."""
    v = asset_version()
    return ('<script data-ps-boot>(function(){try{var t=localStorage.getItem("prophet.theme");'
            'if(t==="light"||t==="dark"){document.documentElement.setAttribute("data-theme",t);}}'
            'catch(e){}})();</script>'
            f'<link rel="stylesheet" href="/ui/shell/shell.css?v={v}" data-ps-css>'
            f'<script defer src="/ui/shell/shell.js?v={v}" data-ps-js></script>')


def _label(tag: str, text: dict, attrs: str = '') -> str:
    """An element whose text the shell script swaps per language."""
    return (f'<{tag}{attrs} data-bg="{html.escape(text["bg"])}" data-en="{html.escape(text["en"])}">'
            f'{html.escape(text["bg"])}</{tag}>')


def _link(href: str, text: dict, path: str) -> str:
    current = ' aria-current="page"' if href == path else ''
    return _label('a', text, f' class="ps-nav-link" href="{href}"{current}')


def render_nav(path: str) -> str:
    """The navigation for `path`. Bulgarian is the default text, so a page with
    scripting off still reads; shell.js switches the labels."""
    links = ''.join(f'<li>{_link(h, t, path)}</li>' for h, t in NAV)
    more_current = any(h == path for h, _ in MORE)
    more_items = ''.join(f'<li>{_link(h, t, path)}</li>' for h, t in MORE)
    more = (f'<li class="ps-more{" ps-more-current" if more_current else ""}">'
            f'<button type="button" class="ps-more-btn" aria-expanded="false" aria-controls="ps-more-list">'
            f'{_label("span", CONTROLS["more"])}<span aria-hidden="true" class="ps-caret">▾</span></button>'
            f'<ul id="ps-more-list" class="ps-more-list" hidden>{more_items}</ul></li>')
    c = CONTROLS
    lang = (f'<div class="ps-seg" role="group" aria-label="{c["language"]["bg"]} / {c["language"]["en"]}">'
            '<button type="button" data-ps-lang="bg" aria-pressed="true">BG</button>'
            '<button type="button" data-ps-lang="en" aria-pressed="false">EN</button></div>')
    theme = (f'<div class="ps-seg ps-theme" role="group" aria-label="{c["theme"]["bg"]} / {c["theme"]["en"]}">'
             + ''.join(_label('button', c[k], f' type="button" data-ps-theme="{k}" aria-pressed="{"true" if k == "system" else "false"}"')
                       for k in ('system', 'light', 'dark'))
             + '</div>')
    # The outer div stays in the page flow and reserves the bar's height; the bar
    # itself is fixed, so it spans the window whatever width or margin the
    # page's own <body> has.
    return (f'<div class="ps-shell" {MARKER}>'
            f'<nav class="ps-nav" aria-label="{c["main_nav"]["bg"]} / {c["main_nav"]["en"]}">'
            '<div class="ps-nav-inner">'
            f'<a class="ps-brand" href="/"><span class="ps-brand-mark" aria-hidden="true"></span>{BRAND["bg"]}</a>'
            '<button type="button" class="ps-menu-btn" aria-expanded="false" aria-controls="ps-menu">'
            f'<span class="ps-burger" aria-hidden="true"></span>{_label("span", c["menu"])}</button>'
            f'<div class="ps-menu" id="ps-menu"><ul class="ps-links">{links}{more}</ul>'
            f'<div class="ps-controls">{lang}{theme}</div></div>'
            '</div></nav></div>')


_BODY = re.compile(r'<body\b[^>]*>', re.I)
_HEAD_END = re.compile(r'</head\s*>', re.I)


def inject(page: str, path: str) -> str:
    """`page` with the shell added. A page that already carries it is returned
    unchanged, so injection is idempotent."""
    if MARKER in page:
        return page
    nav = render_nav(path)
    head = _HEAD_END.search(page)
    snippet = head_snippet()
    if head:
        page = page[:head.start()] + snippet + page[head.start():]
    body = _BODY.search(page)
    if body:
        return page[:body.end()] + nav + page[body.end():]
    if head:                                     # a head but no body tag
        at = head.end() + len(snippet)
        return page[:at] + nav + page[at:]
    return snippet + nav + page                   # a bare fragment (/h1-direction)


def strip(page: str, path: str) -> str:
    """Remove exactly what `inject` added: the inverse, for tests."""
    return page.replace(head_snippet(), '', 1).replace(render_nav(path), '', 1)
