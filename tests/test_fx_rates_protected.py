"""fx-rates-2y, group 1: the family never changes production, frozen evidence or other families."""
import glob
import hashlib
import importlib
import json
import os
import pkgutil
import re

import src.fx_rates as FS
from src.fx_rates import REPO, load_config

PROTECTED = ['models', 'config.json', 'results/pooled_h1', 'results/eurusd_h1.csv', 'results/fomc_dates.csv',
             'results/fx_fib', 'src/fx_fib', 'results/fx_session', 'src/fx_session', 'results/macro_panel', 'results/macro_tier_a', 'results/yield_differential.csv', 'src/features.py', 'src/inference.py', 'src/paper_trading.py']


def other_registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'fx_rates' not in p]


def _digest(rel):
    path = os.path.join(REPO, *rel.split('/'))
    h = hashlib.sha256()
    if not os.path.exists(path):
        return None
    files = [path] if os.path.isfile(path) else sorted(
        os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if '__pycache__' not in d)
    for f in files:
        h.update(os.path.relpath(f, REPO).replace('\\', '/').encode())
        with open(f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    return h.hexdigest()


def protected_digests():
    return {rel: _digest(rel) for rel in PROTECTED + other_registries()}


def test_importing_and_configuring_the_family_changes_no_protected_path():
    before = protected_digests()
    load_config()
    for mod in pkgutil.iter_modules(FS.__path__):
        importlib.import_module(f'src.fx_rates.{mod.name}')
    assert protected_digests() == before
    assert len(other_registries()) >= 5


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FS.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/pooled_h1|results/eurusd_h1\.csv|results/fx_fib|results/fx_session)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_configured_output_paths_stay_inside_the_family_trees():
    paths = load_config()['paths']
    assert paths['results'] == 'results/fx_rates' and paths['artifacts'] == 'research_models/fx_rates'
    assert paths['data'].startswith('results/fx_rates/') and paths['fx_h1'] == 'results/fx_session/data'


def test_fx_fib_config_still_matches_its_verdicts():
    from src.fx_fib import config_digest as fib_digest, load_config as fib_config
    run = os.path.join(REPO, 'results', 'fx_fib', 'walk_forward_run.json')
    with open(run, encoding='utf-8') as fh:
        recorded = json.load(fh)['config_digest']
    assert fib_digest(fib_config()) == recorded


def test_fx_session_config_still_matches_its_verdicts():
    import csv
    from src.fx_session import config_digest as s_digest, load_config as s_config
    with open(os.path.join(REPO, 'results', 'fx_session', 'fx_session_hypothesis_log.csv'), newline='', encoding='utf-8') as fh:
        recorded = {r['config_digest'] for r in csv.DictReader(fh)}
    assert recorded == {s_digest(s_config())}
