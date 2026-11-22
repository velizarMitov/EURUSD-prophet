"""fx-ny-orb-hot, group 1: the family never changes production, frozen evidence or other families."""
import csv
import glob
import importlib
import json
import os
import pkgutil
import re

import src.fx_ny_orb as FO
from src.fx_ny_orb import REPO, load_config
from tests.test_fx_bb_protected import _digest

PROTECTED = ['models', 'config.json', 'results/fx_nyvol', 'src/fx_nyvol', 'results/fx_bb_sma', 'src/fx_bb_sma',
             'results/fx_bb', 'src/fx_bb', 'results/fx_rates', 'src/fx_rates', 'results/fx_session', 'src/fx_session',
             'results/fx_fib', 'src/fx_fib', 'research_models/horizon_study', 'results/curl/raw',
             'src/forecast_eval', 'results/volatility_hypothesis_log.csv', 'src/features.py', 'src/inference.py']


def registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv', 'results/*/*/*hypothesis_log.csv']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'fx_ny_orb' not in p]


def digests():
    return {rel: _digest(rel) for rel in PROTECTED + registries()}


def test_importing_and_configuring_the_family_changes_no_protected_path():
    before = digests()
    load_config()
    for mod in pkgutil.iter_modules(FO.__path__):
        importlib.import_module(f'src.fx_ny_orb.{mod.name}')
    assert digests() == before
    assert len(registries()) >= 9


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FO.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/fx_nyvol|results/fx_bb|results/fx_rates|"""
                         r"""results/fx_session|results/fx_fib|results/curl|research_models/horizon_study)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_configured_output_paths_stay_inside_the_family_trees():
    p = load_config()['paths']
    assert p['results'] == 'results/fx_ny_orb' and p['artifacts'] == 'research_models/fx_ny_orb'


def _recorded(rel):
    with open(os.path.join(REPO, *rel.split('/')), newline='', encoding='utf-8') as fh:
        return {r['config_digest'] for r in csv.DictReader(fh)}


def test_earlier_families_still_match_their_verdicts():
    from src import fx_bb, fx_bb_sma, fx_fib, fx_nyvol, fx_rates, fx_session
    for mod, rel in ((fx_nyvol, 'results/fx_nyvol/fx_nyvol_hypothesis_log.csv'),
                     (fx_bb_sma, 'results/fx_bb_sma/fx_bb_sma_hypothesis_log.csv'),
                     (fx_bb, 'results/fx_bb/fx_bb_hypothesis_log.csv'),
                     (fx_rates, 'results/fx_rates/fx_rates_hypothesis_log.csv'),
                     (fx_session, 'results/fx_session/fx_session_hypothesis_log.csv')):
        assert _recorded(rel) == {mod.config_digest(mod.load_config())}, rel
    with open(os.path.join(REPO, 'results', 'fx_fib', 'walk_forward_run.json'), encoding='utf-8') as fh:
        assert json.load(fh)['config_digest'] == fx_fib.config_digest(fx_fib.load_config())
