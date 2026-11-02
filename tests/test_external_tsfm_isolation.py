"""Isolation contract of the external foundation models (openspec change
tsfm-second-opinion, spec "The models run apart from the application" and "A
failure here costs nothing elsewhere").

Each check is a function over source text, so every test also proves the check
fires on an injected violation.
"""
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'src', 'external', 'tsfm')

PRODUCTION = ['src/inference.py', '_train_pipeline.py', 'src/paper_trading.py', 'src/features.py']
TSFM_IMPORT = re.compile(r'^\s*(from\s+src\.external\.tsfm\b|from\s+src\.external\s+import\s+tsfm\b'
                         r'|import\s+src\.external\.tsfm\b)', re.M)
WORKER_ALLOWED = {'numpy', 'torch', 'chronos', 'timesfm', 'peft', 'transformers', 'safetensors'}
# The production config.json is the repo-root one; a model folder's own
# config.json (vendor.PINS) is a different file. Likewise `models/` is the
# production folder as a PATH; a payload key 'models' is not.
FOREIGN = re.compile(r'results/gold|results/forward_eval|results/horizon_study'
                     r"""|['"]models[/\\]|(?:REPO|repo)\s*,\s*['"]models['"]"""
                     r"""|(?:REPO\s*,\s*|repo_path\(\s*)['"]config\.json""")
WRITES = re.compile(r"""open\([^)]*['"][wax]b?\+?['"]|\.to_csv\(|os\.replace\(|json\.dump\(|np\.save\("""
                    r"""|\.write_text\(|\.write_bytes\(|shutil\.|os\.remove\(|os\.makedirs\(""")


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _code(src):
    """Source without docstrings and comments."""
    src = re.sub(r'(\'\'\'|""").*?\1', '', src, flags=re.S)
    return '\n'.join(l.split('#', 1)[0] for l in src.splitlines())


def _package():
    out = {}
    for f in sorted(os.listdir(PKG)):
        if f.endswith('.py'):
            out[f] = _code(_read(os.path.join(PKG, f)))
    return out


# ── checks ──────────────────────────────────────────────────────────────────

def production_importers(texts: dict) -> list:
    return [rel for rel, src in texts.items() if TSFM_IMPORT.search(src)]


def repo_imports(src: str) -> list:
    """Top-level modules a source imports that are neither stdlib nor allowed."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad.append('.' * node.level + (node.module or ''))
                continue
            names = [node.module or '']
        for n in names:
            top = n.split('.')[0]
            if top not in sys.stdlib_module_names and top not in WORKER_ALLOWED:
                bad.append(n)
    return bad


def foreign_namers(sources: dict) -> list:
    """Modules other than the read-only sources.py that name another family's
    files or production targets."""
    return sorted(f for f, src in sources.items() if f != 'sources.py' and FOREIGN.search(src))


def writes_in(src: str) -> list:
    return WRITES.findall(src)


# ── the real code ───────────────────────────────────────────────────────────

def test_production_never_imports_the_external_package():
    texts = {rel: _read(os.path.join(REPO, rel)) for rel in PRODUCTION}
    assert production_importers(texts) == []


def test_worker_imports_no_repo_module():
    assert repo_imports(_read(os.path.join(PKG, 'worker.py'))) == []


def test_only_sources_names_another_familys_files():
    assert foreign_namers(_package()) == []


def test_sources_never_writes():
    assert writes_in(_package()['sources.py']) == []


def test_api_imports_only_the_read_only_view():
    hits = re.findall(r'^\s*(?:from|import)\s+src\.external\.tsfm[\w.]*[^\n]*', _read(os.path.join(REPO, 'api.py')), re.M)
    for h in hits:
        assert h.strip().startswith('from src.external.tsfm.view import'), h


def test_main_requirements_untouched():
    req = _read(os.path.join(REPO, 'requirements.txt')).lower()
    for pkg in ('chronos', 'timesfm', 'peft'):
        assert pkg not in req, f'{pkg} leaked into requirements.txt'


# ── each check fires on an injected violation ───────────────────────────────

def test_check_flags_a_production_import():
    assert production_importers({'src/inference.py': 'from src.external.tsfm.view import payload\n'}) == \
        ['src/inference.py']
    assert production_importers({'x.py': 'import src.external.tsfm.client\n'}) == ['x.py']


def test_check_flags_a_repo_import_in_the_worker():
    assert repo_imports('import numpy\nfrom src.gold import data\n') == ['src.gold']
    assert repo_imports('from . import forecast\n') != []
    assert repo_imports('from .forecast import size\n') == ['.forecast']
    assert repo_imports('import pandas\n') == ['pandas']


def test_check_flags_a_foreign_path_outside_sources():
    assert foreign_namers({'forward_logger.py': "OUT = 'results/gold/forward/x.csv'"}) == ['forward_logger.py']
    assert foreign_namers({'replay.py': "p = os.path.join(REPO, 'models/', 'x')"}) == ['replay.py']
    assert foreign_namers({'replay.py': "p = os.path.join(REPO, 'models', 'x')"}) == ['replay.py']
    assert foreign_namers({'view.py': "block['models'].append(entry)"}) == []
    assert foreign_namers({'view.py': "open('results/horizon_study/study_record.json')"}) == ['view.py']
    assert foreign_namers({'prereg.py': "cfg = repo_path('config.json')"}) == ['prereg.py']
    assert foreign_namers({'vendor.py': "{'config.json': ('blob', 'abc')}"}) == []
    assert foreign_namers({'sources.py': "X = 'results/gold/gold_h1.csv'"}) == []


def test_check_flags_a_write_in_sources():
    assert writes_in("with open(p, 'w') as fh: pass") != []
    assert writes_in('df.to_csv(p)') != []
    assert writes_in("with open(p, encoding='utf-8') as fh: pass") == []
