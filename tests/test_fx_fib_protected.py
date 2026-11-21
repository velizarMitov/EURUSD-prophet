"""fx-fibonacci-mlp: the family never changes the production or frozen-evidence paths."""
import hashlib
import importlib
import os
import pkgutil
import re

import src.fx_fib as FX
from src.fx_fib import REPO, load_config

PROTECTED = ['models', 'config.json', 'results/pooled_h1', 'results/eurusd_h1.csv',
             'src/features.py', 'src/inference.py', 'src/paper_trading.py']


def _digest(rel):
    path = os.path.join(REPO, *rel.split('/'))
    h = hashlib.sha256()
    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs)
    for f in files:
        h.update(os.path.relpath(f, REPO).replace('\\', '/').encode())
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    return h.hexdigest()


def protected_digests():
    return {rel: _digest(rel) for rel in PROTECTED}


def test_importing_and_configuring_the_family_changes_no_protected_path():
    before = protected_digests()
    load_config()
    for mod in pkgutil.iter_modules(FX.__path__):
        importlib.import_module(f'src.fx_fib.{mod.name}')
    assert protected_digests() == before


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FX.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/pooled_h1|results/eurusd_h1\.csv)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_configured_output_paths_stay_inside_the_family_trees():
    paths = load_config()['paths']
    assert paths['results'] == 'results/fx_fib' and paths['artifacts'] == 'research_models/fx_fib'
    assert paths['data'].startswith('results/fx_fib/')
