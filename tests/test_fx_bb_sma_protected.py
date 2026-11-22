"""fx-bb-sma-exit, group 1: the family never changes production, frozen evidence or other families."""
import csv
import importlib
import os
import pkgutil
import re

import src.fx_bb_sma as FS
from src.fx_bb_sma import REPO, load_config
from tests.test_fx_bb_protected import _digest, other_registries

PROTECTED = ['models', 'config.json', 'results/pooled_h1', 'results/fomc_dates.csv', 'results/fx_fib', 'src/fx_fib',
             'results/fx_session', 'src/fx_session', 'results/fx_rates', 'src/fx_rates', 'results/fx_bb', 'src/fx_bb',
             'results/macro_panel', 'results/macro_tier_a', 'src/features.py', 'src/inference.py',
             'src/paper_trading.py']


def registries():
    return [r for r in other_registries() + ['results/fx_bb/fx_bb_hypothesis_log.csv'] if 'fx_bb_sma' not in r]


def digests():
    return {rel: _digest(rel) for rel in PROTECTED + registries()}


def test_importing_and_configuring_the_family_changes_no_protected_path():
    before = digests()
    load_config()
    for mod in pkgutil.iter_modules(FS.__path__):
        importlib.import_module(f'src.fx_bb_sma.{mod.name}')
    assert digests() == before
    assert len(registries()) >= 7


def test_family_sources_never_name_a_protected_write_target():
    src_dir = os.path.dirname(FS.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/pooled_h1|results/eurusd_h1\.csv|results/fx_fib|"""
                         r"""results/fx_rates|results/fx_session|results/fx_bb/)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)


def test_configured_output_paths_stay_inside_the_family_trees():
    cfg = load_config()
    assert cfg['paths'] == {'results': 'results/fx_bb_sma', 'artifacts': 'research_models/fx_bb_sma'}


def _recorded(rel):
    with open(os.path.join(REPO, *rel.split('/')), newline='', encoding='utf-8') as fh:
        return {r['config_digest'] for r in csv.DictReader(fh)}


def test_earlier_fx_configs_still_match_their_verdicts():
    import json
    from src import fx_bb, fx_fib, fx_rates, fx_session
    assert _recorded('results/fx_bb/fx_bb_hypothesis_log.csv') == {fx_bb.config_digest(fx_bb.load_config())}
    assert _recorded('results/fx_rates/fx_rates_hypothesis_log.csv') == {fx_rates.config_digest(fx_rates.load_config())}
    assert _recorded('results/fx_session/fx_session_hypothesis_log.csv') == \
        {fx_session.config_digest(fx_session.load_config())}
    with open(os.path.join(REPO, 'results', 'fx_fib', 'walk_forward_run.json'), encoding='utf-8') as fh:
        assert json.load(fh)['config_digest'] == fx_fib.config_digest(fx_fib.load_config())
