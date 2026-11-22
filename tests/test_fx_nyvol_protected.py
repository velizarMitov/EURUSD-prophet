"""fx-ny-volatility, group 1: the family never changes production, frozen evidence or other families."""
import csv
import importlib
import json
import os
import pkgutil
import re

import src.fx_nyvol as FN
from src.fx_nyvol import REPO, load_config
from tests.test_fx_bb_protected import _digest

PROTECTED = ['models', 'config.json', 'results/volatility_hypothesis_log.csv', 'results/volatility_seed_ensemble.csv',
             'results/pooled_h1', 'results/fomc_dates.csv', 'results/fx_fib', 'src/fx_fib', 'results/fx_session',
             'src/fx_session', 'results/fx_rates', 'src/fx_rates', 'results/fx_bb', 'src/fx_bb', 'results/fx_bb_sma',
             'src/fx_bb_sma', 'src/features.py', 'src/inference.py', 'src/volatility.py']


def registries():
    import glob
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'fx_nyvol' not in p]


def digests():
    return {rel: _digest(rel) for rel in PROTECTED + registries()}


def test_importing_and_configuring_the_family_changes_no_protected_path():
    before = digests()
    load_config()
    for mod in pkgutil.iter_modules(FN.__path__):
        importlib.import_module(f'src.fx_nyvol.{mod.name}')
    assert digests() == before
    assert len(registries()) >= 8


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FN.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/volatility|results/pooled_h1|results/fx_fib|"""
                         r"""results/fx_rates|results/fx_session|results/fx_bb)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_configured_output_paths_stay_inside_the_family_trees():
    p = load_config()['paths']
    assert p['results'] == 'results/fx_nyvol' and p['artifacts'] == 'research_models/fx_nyvol'
    assert p['fx_session_data'] == 'results/fx_session/data'


def _recorded(rel):
    with open(os.path.join(REPO, *rel.split('/')), newline='', encoding='utf-8') as fh:
        return {r['config_digest'] for r in csv.DictReader(fh)}


def test_fx_fib_fx_session_and_fx_rates_configs_match_their_verdicts():
    from src import fx_fib, fx_rates, fx_session
    assert _recorded('results/fx_session/fx_session_hypothesis_log.csv') == \
        {fx_session.config_digest(fx_session.load_config())}
    assert _recorded('results/fx_rates/fx_rates_hypothesis_log.csv') == {fx_rates.config_digest(fx_rates.load_config())}
    with open(os.path.join(REPO, 'results', 'fx_fib', 'walk_forward_run.json'), encoding='utf-8') as fh:
        assert json.load(fh)['config_digest'] == fx_fib.config_digest(fx_fib.load_config())
