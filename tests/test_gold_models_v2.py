"""Round-two models logged for observation (openspec change gold-forecasting-v2, task 6.2)."""
import copy
import csv
import hashlib
import os
import sys

import numpy as np
import pytest

from src.gold import forward_logger as FL
from src.gold import load_config
from src.gold import macro as M
from src.gold import models_v2 as M2
from src.gold import prereg2 as P2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gold_forward_logger import FakeModel, _frames, _h1_weeks  # noqa: E402


@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg['paths']['forward'] = str(tmp_path / 'forward')
    os.makedirs(cfg['paths']['forward'])
    cfg2 = copy.deepcopy(P2.load_config_v2())
    cfg2['paths']['v2_artifacts'] = str(tmp_path / 'v2')
    cfg2['paths']['v2_manifest'] = str(tmp_path / 'artifact_manifest_v2.csv')
    rows = []
    for cell, spec in M2.CELLS.items():
        d = tmp_path / 'v2' / cell
        d.mkdir(parents=True)
        (d / 'model.bin').write_bytes(cell.encode())
        rows.append({'cell': cell, 'cadence': spec['cadence'], 'path': str(d / 'model.bin'),
                     'sha256': hashlib.sha256(cell.encode()).hexdigest()})
    with open(cfg2['paths']['v2_manifest'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['cell', 'cadence', 'path', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    return cfg, cfg2


def _loader(cell):
    return FakeModel(p=0.52 if cell == 'h1_selective' else 0.6), {'tau': 0.05}


def _macro(df):
    return np.zeros((len(df), len(M.MACRO_COLUMNS)))


def test_v2_cells_are_logged_with_digests_and_round_one_stays_untouched(env):
    cfg, cfg2 = env
    h1 = _h1_weeks()
    fr = _frames(h1, 400)
    status = FL.log_v2(cfg, fr, {'H1': True, 'D1': True}, 'dry_run', 'now', cfg2, loader=_loader, macro_fn=_macro)
    rows = FL.read_rows(FL.out_path(cfg, 'predictions_v2'))
    assert status.startswith('4 forecast') and {r['cell'] for r in rows} == set(M2.CELLS)
    assert all(r['version'] == 'v2' and r['shadow'] == '0' and len(r['model_digest']) == 16 for r in rows)
    sel = [r for r in rows if r['cell'] == 'h1_selective'][0]
    assert sel['p_up'] == ''                                  # |0.52 - 0.5| < tau: abstained
    assert not os.path.exists(FL.out_path(cfg, 'predictions'))
    again = FL.log_v2(cfg, fr, {'H1': True, 'D1': True}, 'dry_run', 'now', cfg2, loader=_loader, macro_fn=_macro)
    assert again.startswith('0 forecast')                      # one row per bar


def test_a_macro_gap_skips_only_the_macro_cells_and_names_the_series(env):
    cfg, cfg2 = env

    def missing(df):
        raise M.MacroUnavailable('FRED DFII10 (dfii10) is unreachable and not cached')
    status = FL.log_v2(cfg, _frames(_h1_weeks(), 400), {'H1': True, 'D1': True}, 'dry_run', 'now', cfg2,
                       loader=_loader, macro_fn=missing)
    cells = {r['cell'] for r in FL.read_rows(FL.out_path(cfg, 'predictions_v2'))}
    assert cells == {'h1_selective', 'h1_barrier'} and 'DFII10' in status
    assert any('DFII10' in r['error'] for r in FL.read_rows(FL.out_path(cfg, 'failures')))


def test_barrier_rows_settle_on_their_label_after_k_hours_and_the_rest_on_the_next_bar(env):
    cfg, cfg2 = env
    h1 = _h1_weeks()
    FL.log_v2(cfg, _frames(h1, 400), {'H1': True, 'D1': True}, 'dry_run', 'now', cfg2, loader=_loader, macro_fn=_macro)
    FL.settle_v2(cfg, _frames(h1, 401), 'later', cfg2)
    settled = {r['key'].split('|')[0] for r in FL.read_rows(FL.out_path(cfg, 'settlements_v2'))}
    assert 'h1_selective' in settled and 'h1_barrier' not in settled   # its 6-hour window is still open
    FL.settle_v2(cfg, _frames(h1, 410), 'much later', cfg2)
    rows = {r['key'].split('|')[0]: r for r in FL.read_rows(FL.out_path(cfg, 'settlements_v2'))}
    b = rows['h1_barrier']
    k = cfg2['study']['direction2']['barrier']['k_hours']
    assert b['exit_bar'] == str(h1.index[399 + k])


def test_no_v2_models_means_nothing_is_logged(env, tmp_path):
    cfg, cfg2 = env
    cfg2['paths']['v2_manifest'] = str(tmp_path / 'missing.csv')
    assert FL.log_v2(cfg, {}, {}, 'dry_run', 'now', cfg2) == 'no v2 models'
