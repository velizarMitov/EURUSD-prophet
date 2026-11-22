"""fx-bb-reversion, group 1: the family never changes production, frozen evidence or other families."""
import csv
import glob
import hashlib
import importlib
import json
import os
import pkgutil
import re

import src.fx_bb as FB
from src.fx_bb import REPO, load_config

PROTECTED = ['models', 'config.json', 'results/pooled_h1', 'results/eurusd_h1.csv', 'results/fomc_dates.csv',
             'results/fx_fib', 'src/fx_fib', 'results/fx_session', 'src/fx_session', 'results/fx_rates',
             'src/fx_rates', 'results/macro_panel', 'results/macro_tier_a', 'results/yield_differential.csv',
             'src/features.py', 'src/inference.py', 'src/paper_trading.py']


def other_registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'fx_bb' not in p]


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
    for mod in pkgutil.iter_modules(FB.__path__):
        importlib.import_module(f'src.fx_bb.{mod.name}')
    assert protected_digests() == before
    assert len(other_registries()) >= 6


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FB.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/pooled_h1|results/eurusd_h1\.csv|results/fx_fib|"""
                         r"""results/fx_rates)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_the_fx_session_caches_are_only_named_in_the_config_as_a_read_source():
    src_dir = os.path.dirname(FB.__file__)
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                assert 'results/fx_session' not in fh.read(), name


def test_configured_output_paths_stay_inside_the_family_trees():
    cfg = load_config()
    paths = cfg['paths']
    assert paths['results'] == 'results/fx_bb' and paths['artifacts'] == 'research_models/fx_bb'
    assert paths['data'] == 'results/fx_bb/data' and paths['fx_session_data'] == 'results/fx_session/data'
    assert {p for p, s in cfg['sources'].items() if s == 'fx_bb'} == {'USDJPY', 'USDCAD', 'USDCHF'}


def test_fx_fib_config_still_matches_its_verdicts():
    from src.fx_fib import config_digest as fib_digest, load_config as fib_config
    with open(os.path.join(REPO, 'results', 'fx_fib', 'walk_forward_run.json'), encoding='utf-8') as fh:
        recorded = json.load(fh)['config_digest']
    assert fib_digest(fib_config()) == recorded


def _registry_digests(rel):
    with open(os.path.join(REPO, *rel.split('/')), newline='', encoding='utf-8') as fh:
        return {r['config_digest'] for r in csv.DictReader(fh)}


def test_fx_session_config_still_matches_its_verdicts():
    from src.fx_session import config_digest as s_digest, load_config as s_config
    assert _registry_digests('results/fx_session/fx_session_hypothesis_log.csv') == {s_digest(s_config())}


def test_fx_rates_config_still_matches_its_verdicts():
    from src.fx_rates import config_digest as r_digest, load_config as r_config
    assert _registry_digests('results/fx_rates/fx_rates_hypothesis_log.csv') == {r_digest(r_config())}
