"""Gold model versions (openspec change gold-forecasting-v2, spec gold-forecasting/model-versions)."""
import copy
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.gold import forward_logger as FL
from src.gold import load_config
from src.gold import prereg2 as P2
from src.gold import study as ST
from src.gold import versions as V

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gold_forward_logger import FakeModel, _frames, _h1_weeks  # noqa: E402

CELLS = [c for cells in FL.FORWARD_CELLS.values() for c in cells]


def _write_manifest(path, root, tag):
    rows = []
    for cell in CELLS:
        d = os.path.join(root, cell)
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, 'model.bin')
        open(f, 'wb').write(f'{cell}-{tag}'.encode())
        rows.append({'cell': cell, 'cadence': 'H1', 'path': f, 'sha256': hashlib.sha256(f'{cell}-{tag}'.encode()).hexdigest(),
                     'bytes': 1, 'config_hash': 'c', 'fit_rows': 1, 'last_bar': 'x', 'written_at': 'x'})
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=ST.MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg['paths']['artifacts'] = str(tmp_path / 'registered')
    cfg['paths']['manifest'] = str(tmp_path / 'artifact_manifest.csv')
    cfg['paths']['forward'] = str(tmp_path / 'forward')
    cfg['paths']['results_md'] = str(tmp_path / 'RESULTS.md')
    open(cfg['paths']['results_md'], 'w').write('# results\n')
    os.makedirs(cfg['paths']['forward'])
    _write_manifest(cfg['paths']['manifest'], cfg['paths']['artifacts'], 'registered')
    cfg2 = copy.deepcopy(P2.load_config_v2())
    cfg2['paths']['versions'] = str(tmp_path / 'versions')
    return cfg, cfg2


def _saver(frames, vcfg):
    return _write_manifest(vcfg['paths']['manifest'], vcfg['paths']['artifacts'], 'refit'), []


def _sha_tree(path):
    out = {}
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            out[p] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def test_a_refit_creates_a_version_and_never_touches_the_registered_models(env):
    cfg, cfg2 = env
    before = (_sha_tree(cfg['paths']['artifacts']), open(cfg['paths']['manifest'], 'rb').read())
    out = V.refit(cfg, cfg2, frames={}, today=datetime(2026, 12, 1, tzinfo=timezone.utc), saver=_saver)
    assert out == {'version': '20261201', 'artifacts': len(CELLS), 'active': True}
    assert (_sha_tree(cfg['paths']['artifacts']), open(cfg['paths']['manifest'], 'rb').read()) == before
    assert V.active(cfg2) == '20261201'
    assert os.path.exists(os.path.join(cfg2['paths']['versions'], '20261201', 'manifest.csv'))
    with pytest.raises(V.VersionRefused, match='already exists'):
        V.refit(cfg, cfg2, frames={}, today=datetime(2026, 12, 1, tzinfo=timezone.utc), saver=_saver)


def test_an_overlapping_refit_exits_and_records_the_skip(env):
    cfg, cfg2 = env
    os.makedirs(cfg2['paths']['versions'])
    with FL.RunLock(os.path.join(cfg2['paths']['versions'], 'retrain.lock')):
        out = V.refit(cfg, cfg2, frames={}, saver=_saver)
    assert out['version'] is None and 'skipped' in out
    from src.gold import health as HL
    [row] = list(csv.DictReader(open(HL.health_path(cfg), encoding='utf-8')))
    assert row['outcome'] == 'locked' and row['note'].startswith('refit: skipped')


def test_promotion_is_refused_while_the_forward_window_is_open(env):
    cfg, cfg2 = env
    V.refit(cfg, cfg2, frames={}, today=datetime(2026, 12, 1, tzinfo=timezone.utc), saver=_saver)
    json.dump({'created_at': '2026-11-04T10:00:00+00:00', 'cap_years': 3.0},
              open(os.path.join(cfg['paths']['forward'], 'PRE_REGISTRATION.json'), 'w'))
    with pytest.raises(V.VersionRefused, match='open until 2029-11-0'):
        V.promote('20261201', 'monthly refit', cfg, cfg2, now='2026-12-02')


def test_promotion_after_the_window_swaps_the_manifest_and_records_it(env):
    cfg, cfg2 = env
    V.refit(cfg, cfg2, frames={}, today=datetime(2026, 12, 1, tzinfo=timezone.utc), saver=_saver)
    old = {c: FL.cell_digest(e) for c, e in FL.read_manifest(cfg).items()}
    json.dump({'created_at': '2026-11-04T10:00:00+00:00', 'cap_years': 3.0},
              open(os.path.join(cfg['paths']['forward'], 'PRE_REGISTRATION.json'), 'w'))
    out = V.promote('20261201', 'the refit beat the frozen model on the leaderboard', cfg, cfg2, now='2030-01-01')
    new = {c: FL.cell_digest(e) for c, e in FL.read_manifest(cfg).items()}
    assert new != old and out['new'] == new and out['old'] == old
    for cell in CELLS:
        FL.verify_cell(cell, FL.read_manifest(cfg))                # the new files verify in place
    md = open(cfg['paths']['results_md'], encoding='utf-8').read()
    assert 'version 20261201 promoted' in md and 'the refit beat the frozen model' in md
    assert not os.path.exists(os.path.join(cfg['paths']['forward'], 'PRE_REGISTRATION.json'))
    with pytest.raises(V.VersionRefused, match='needs a reason'):
        V.promote('20261201', ' ', cfg, cfg2, now='2030-01-01')


def test_the_logger_forecasts_the_shadow_into_its_own_file(env, tmp_path):
    cfg, cfg2 = env
    V.refit(cfg, cfg2, frames={}, today=datetime(2026, 12, 1, tzinfo=timezone.utc), saver=_saver)
    h1 = _h1_weeks()
    fr = _frames(h1, 400)
    status = FL.log_shadow(cfg, fr, {'H1': True, 'D1': True}, 'dry_run', 'now', cfg2,
                           loader=lambda cell, vcfg: FakeModel())
    rows = FL.read_rows(FL.out_path(cfg, 'predictions_v2'))
    assert status.startswith('20261201: 6 forecast') and len(rows) == 6
    assert all(r['shadow'] == '1' and r['version'] == '20261201' for r in rows)
    assert not os.path.exists(FL.out_path(cfg, 'predictions'))       # nothing reaches the registered log
    FL.log_shadow(cfg, _frames(h1, 401), {'H1': True, 'D1': True}, 'dry_run', 'later', cfg2,
                  loader=lambda cell, vcfg: FakeModel())
    assert FL.read_rows(FL.out_path(cfg, 'settlements_v2'))


def test_no_active_version_means_no_shadow(env):
    cfg, cfg2 = env
    assert FL.log_shadow(cfg, {}, {}, 'dry_run', 'now', cfg2) == 'no shadow version'
