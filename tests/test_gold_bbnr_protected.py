"""gold-bb-nyrange, task 1.2: the family never changes production, the gold family or any other family."""
import copy
import glob
import importlib
import os
import pkgutil
import re

import src.gold_bbnr as GB
from src.gold_bbnr import REPO, load_config
from tests.test_fx_bb_protected import _digest

PROTECTED = ['models', 'config.json', 'results/gold/gold_h1.csv', 'results/gold/gold_h1.meta.json',
             'src/gold/gold_config.json', 'src/gold/gold_config_v2.json', 'results/fx_session/data',
             'results/fomc_dates.csv', 'src/features.py', 'src/inference.py']


def registries():
    found = glob.glob(os.path.join(REPO, 'results', '**', '*hypothesis_log.csv'), recursive=True)
    return sorted(os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'gold_bbnr' not in p)


def digests():
    return {rel: _digest(rel) for rel in PROTECTED + registries()}


def test_label_stage_on_the_real_inputs_changes_no_protected_path(tmp_path):
    from src.gold_bbnr import labels as LB
    before = digests()
    for mod in pkgutil.iter_modules(GB.__path__):
        importlib.import_module(f'src.gold_bbnr.{mod.name}')
    cfg = copy.deepcopy(load_config())
    cfg['paths']['results'] = str(tmp_path / 'results')
    LB.write_report(cfg, 'T')
    assert digests() == before
    assert len(registries()) >= 12


def test_family_sources_never_name_a_foreign_write_target():
    src_dir = os.path.dirname(GB.__file__)
    pattern = re.compile(r"""['"](models/|config\.json|results/gold/|results/fx_|research_models/(?!gold_bbnr))""")
    allowed = {'results/gold/gold_h1.csv', 'results/fx_session/data'}           # read-only inputs, in the config only
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = pattern.findall(fh.read())
            assert not hits, (name, hits)
    cfg = load_config()
    assert set(cfg['paths_read'].values()) - {'results/fomc_dates.csv'} == allowed


def test_configured_output_paths_stay_inside_the_family_trees():
    p = load_config()['paths']
    assert p == {'results': 'results/gold_bbnr', 'data': 'results/gold_bbnr/data', 'artifacts': 'research_models/gold_bbnr'}


def test_no_order_or_broker_code():
    src_dir = os.path.dirname(GB.__file__)
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            text = open(os.path.join(src_dir, name), encoding='utf-8').read()
            for word in ('order_send', 'positions_get', 'TRADE_ACTION', 'lot_size'):
                assert word not in text, (name, word)
