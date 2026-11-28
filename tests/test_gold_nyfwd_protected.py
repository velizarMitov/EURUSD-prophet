"""gold-nyrange-forward, group 1: the family never changes production, frozen evidence or other families.

results/gold/ and results/fx_ny_orb/ are covered except the files their own scheduled loggers append to
(gold_h1/d1 caches, forward/, live/, macro/, the GVZ logs), which may change during a test run for reasons
of their own; those loggers' registries and pre-registrations are covered."""
import glob
import importlib
import os
import pkgutil
import re

import pandas as pd

import src.gold_nyfwd as GN
from src.gold_nyfwd import REPO, load_config
from src.gold_nyfwd import live as LV
from src.gold_nyfwd import prereg as PR
from src.gold_nyfwd import score as SC
from tests._gold_nyfwd_fakes import FakeFeed, cfg_in, common_files_in_tmp, listed, market, no_gate  # noqa: F401
from tests.test_fx_bb_protected import _digest

LIVE_WRITERS = re.compile(r'^results/(gold/(forward|macro|accel|window)/|gold/gold_[hd]1|gold/gvz_|fx_ny_orb/(forward|live)/)')
PROTECTED_TREES = ['models', 'config.json', 'results/gold_bbnr', 'research_models/gold_bbnr', 'src/gold_bbnr',
                   'src/gold/gold_config.json', 'src/gold/gold_config_v2.json', 'src/fx_ny_orb', 'src/gold']


def files_under(rel):
    base = os.path.join(REPO, *rel.split('/'))
    for d, _, fs in os.walk(base):
        for f in fs:
            r = os.path.relpath(os.path.join(d, f), REPO).replace('\\', '/')
            if not LIVE_WRITERS.match(r):
                yield r


def registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv', 'results/*/*/*hypothesis_log.csv']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'gold_nyfwd' not in p]


def digests():
    paths = PROTECTED_TREES + registries() + list(files_under('results/gold')) + list(files_under('results/fx_ny_orb'))
    return {rel: _digest(rel) for rel in paths}


def test_forecast_settlement_and_scoring_change_no_protected_path(tmp_path):
    before = digests()
    load_config()
    for mod in pkgutil.iter_modules(GN.__path__):
        importlib.import_module(f'src.gold_nyfwd.{mod.name}')
    c = cfg_in(tmp_path)
    PR.write(c)
    bars = market()
    day = pd.Timestamp('2026-07-15', tz='UTC')
    assert LV.run_forecast(c, FakeFeed(bars, listed(), day + pd.Timedelta(hours=15, minutes=8)), gate=no_gate)['status'] == 'logged'
    assert LV.run_settle(c, FakeFeed(bars, listed(), day + pd.Timedelta(hours=22, minutes=8)), gate=no_gate)['status'] == 'settled'
    SC.score(c, today='2027-04-30', gate=no_gate)
    assert digests() == before
    assert len(registries()) >= 15


def test_no_source_names_an_order_or_position_call():
    src_dir = os.path.dirname(GN.__file__)
    bad = re.compile(r'order_send|order_check|positions_get|orders_get|TRADE_ACTION|order_calc')
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                assert not bad.findall(fh.read()), name


def test_no_module_names_a_write_target_outside_the_family_trees():
    src_dir = os.path.dirname(GN.__file__)
    target = re.compile(r"""['"](models/|config\.json|results/(?!gold_nyfwd)[a-z_]+|research_models/(?!gold_nyfwd)[a-z_]+)""")
    for name in os.listdir(src_dir):
        if name.endswith('.py'):
            with open(os.path.join(src_dir, name), encoding='utf-8') as fh:
                hits = target.findall(fh.read())
            assert not hits, (name, hits)
    p = load_config()['paths']
    assert p == {'results': 'results/gold_nyfwd', 'artifacts': 'research_models/gold_nyfwd'}


def test_gold_bbnr_frozen_model_still_matches_its_manifest():
    from src.gold_bbnr import load_config as gbc
    from src.gold_bbnr.frozen import verify
    assert set(verify(gbc())) == {'ols.json', 'lightgbm.txt', 'mlp.pt'}
