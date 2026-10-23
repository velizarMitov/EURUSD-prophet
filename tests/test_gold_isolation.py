"""Isolation contract of the gold research family (openspec change gold-forecasting,
spec gold-forecasting/study: "The study is isolated from production").

The protected-set SHA-256 fixtures (tests/test_h1_production.py and friends) pin
models/, config.json and the production code; these tests add the gold-specific
half: production never imports src.gold, and src.gold never names a production
write target or the EUR/USD forward arbiter's files.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(REPO, 'src', 'gold')

PRODUCTION = ['src/inference.py', '_train_pipeline.py', 'src/paper_trading.py', 'src/features.py']
GOLD_IMPORT = re.compile(r'^\s*(from\s+src\.gold\b|from\s+src\s+import\s+gold\b|import\s+src\.gold\b)', re.M)


def _read(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as fh:
        return fh.read()


def _code(src):
    """Source without docstrings and comments: the isolation contract is stated
    in prose in those, and only executable text can break it."""
    src = re.sub(r'(\'\'\'|""").*?\1', '', src, flags=re.S)
    return '\n'.join(l.split('#', 1)[0] for l in src.splitlines())


def _gold_sources():
    for root, _dirs, files in os.walk(GOLD):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, encoding='utf-8') as fh:
                    yield os.path.relpath(path, REPO).replace('\\', '/'), _code(fh.read())


def test_production_never_imports_gold():
    for rel in PRODUCTION:
        assert not GOLD_IMPORT.search(_read(rel)), f'{rel} imports src.gold'


def test_api_imports_only_the_read_only_summary():
    """api.py may import the two read-only views and nothing else of src.gold:
    the summary (/api/gold) and the history (/api/gold/history, openspec change
    app-shell-and-gold-history). Both run no model and write nothing."""
    hits = [m.group(0).strip() for m in re.finditer(r'^\s*(from\s+src\.gold[\w.]*\s+import[^\n]*|import\s+src\.gold[\w.]*)',
                                                   _read('api.py'), re.M)]
    for h in hits:
        assert h.startswith(('from src.gold.summary import', 'from src.gold.history import',
                             'from src.gold.health import', 'from src.gold.leaderboard import')), \
            f'api.py imports more than the read-only views: {h}'


def test_gold_names_no_production_write_target():
    for rel, src in _gold_sources():
        assert not re.search(r"""['"]models[/\\'"]""", src), f'{rel} names the production models/ folder'
        assert not re.search(r'(?<![\w_])config\.json', src), f'{rel} names config.json'
        assert 'forward_eval' not in src, f'{rel} names the EUR/USD forward arbiter'


def test_gold_never_reads_spread_or_volume_as_data():
    """Spread and volume are dropped at acquisition: data.py whitelists OHLC, so
    no module may address either as a column or record field (an exact quoted
    name, or attribute access). The feature guard's rejection pattern may still
    mention the words -- naming what is refused is not reading it."""
    access = re.compile(r"""['"](tick_volume|real_volume|spread|spread_points|spread_price)['"]"""
                        r"""|\.(tick_volume|real_volume|spread)\b""")
    for rel, src in _gold_sources():
        m = access.search(src)
        assert not m, f'{rel} addresses {m.group(0)}'
