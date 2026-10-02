"""Safety guards for src/forecast_eval/ (openspec change horizon-study-forward-evaluation).

The forward logger attaches to whatever account the MT5 terminal holds -- today a
REAL-money account. These guards make two things structurally impossible rather
than merely unintended: (1) any call that could place, change or inspect a trade,
and (2) any write into the production models/ directory. Each guard is shown to
BITE on a probe before it is trusted to pass on the package.
"""
import ast
import glob
import hashlib
import importlib
import json
import os
import pkgutil
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(REPO, 'src', 'forecast_eval')

# MetaTrader5 functions that trade, or read the account's trading state.
TRADE_EXACT = {'order_send', 'order_check', 'order_calc_margin', 'order_calc_profit'}
TRADE_PREFIXES = ('positions_', 'orders_', 'history_orders_', 'history_deals_')


def _is_trade_name(name):
    return name in TRADE_EXACT or name.startswith(TRADE_PREFIXES)


def trade_references(source):
    """Every reference to a trade function: attribute access, bare name,
    import alias, or a string constant (catches getattr(mt5, "order_send"))."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and _is_trade_name(node.attr):
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and _is_trade_name(node.id):
            hits.append(node.id)
        elif isinstance(node, ast.alias) and _is_trade_name(node.name.split('.')[-1]):
            hits.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and _is_trade_name(node.value.strip()):
            hits.append(node.value)
    return hits


_MODELS_LITERAL = re.compile(r'^(\.[\\/])?models([\\/].*)?$')


def models_dir_literals(source):
    """String constants that name the production models/ directory. A path
    joined from 'models' or written as 'models/...' is how every production
    artifact path in this repo is spelled."""
    return [n.value for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and _MODELS_LITERAL.match(n.value.strip())]


def _package_sources():
    paths = sorted(glob.glob(os.path.join(PKG_DIR, '**', '*.py'), recursive=True))
    assert paths, 'src/forecast_eval/ has no modules'
    return [(p, open(p, encoding='utf-8').read()) for p in paths]


# ── 1.2 no trade capability ───────────────────────────────────────────────

@pytest.mark.parametrize('probe', [
    'import MetaTrader5 as mt5\nmt5.order_send(req)\n',
    'mt5.order_check(req)\n',
    'p = mt5.positions_get()\n',
    'o = mt5.orders_total()\n',
    'h = mt5.history_orders_get(a, b)\n',
    'h = mt5.history_deals_get(a, b)\n',
    'f = getattr(mt5, "order_send")\n',
    'from MetaTrader5 import order_send\n',
], ids=['order_send', 'order_check', 'positions', 'orders', 'history_orders',
        'history_deals', 'getattr-string', 'from-import'])
def test_trade_guard_fires_on_a_probe(probe):
    assert trade_references(probe), f'guard missed: {probe!r}'


def test_trade_guard_ignores_read_only_calls():
    ok = ('mt5.initialize()\nmt5.copy_rates_from_pos(s, tf, 0, 10)\n'
          'mt5.symbol_info(s)\nmt5.account_info().server\nmt5.shutdown()\n')
    assert trade_references(ok) == []


def test_package_references_no_trade_function():
    for path, src in _package_sources():
        hits = trade_references(src)
        assert not hits, f'{os.path.relpath(path, REPO)} references trade functions: {hits}'


# ── 1.3 no production models/ writes, no pinned file touched ───────────────

@pytest.mark.parametrize('probe', [
    "open('models/baseline/x.pkl', 'wb')\n",
    "os.path.join('models', 'volatility')\n",
    "p = 'models\\\\h1_gbm.pkl'\n",
    "p = './models/x'\n",
])
def test_models_guard_fires_on_a_probe(probe):
    assert models_dir_literals(probe), f'guard missed: {probe!r}'


def test_models_guard_allows_research_paths():
    ok = "os.path.join('research_models', 'horizon_study')\np = 'research_models/x'\nm = 'model'\n"
    assert models_dir_literals(ok) == []


def test_package_names_no_production_models_path():
    for path, src in _package_sources():
        hits = models_dir_literals(src)
        assert not hits, f'{os.path.relpath(path, REPO)} names the production models/ dir: {hits}'


def _pinned_digests():
    pinned = {}
    for f in glob.glob(os.path.join(REPO, 'tests', 'fixtures', '*_protected_sha256.json')):
        for rel, dig in json.load(open(f)).items():
            if isinstance(dig, str):
                pinned[rel] = dig
    return pinned


def test_importing_the_package_changes_no_pinned_file():
    pinned = _pinned_digests()
    assert len(pinned) > 40
    def snapshot():
        out = {}
        for rel in pinned:
            p = os.path.join(REPO, rel)
            out[rel] = hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else None
        return out
    before = snapshot()
    import src.forecast_eval as pkg
    for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
        importlib.import_module(mod.name)
    assert snapshot() == before
