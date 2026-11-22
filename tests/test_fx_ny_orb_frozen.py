"""fx-ny-orb-hot, group 6.1: the frozen OLS HAR-X and HAR."""
import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import REPO, load_config
from src.fx_ny_orb import frozen as FZ

CFG = load_config()
REAL = os.path.exists(os.path.join(REPO, 'results', 'fx_session', 'data', 'EURUSD_h1.csv')) and \
    os.path.exists(os.path.join(REPO, 'results', 'fx_nyvol', 'oos_lgbm.csv.gz'))


@pytest.fixture(scope='module')
def data():
    if not REAL:
        pytest.skip('real data not present')
    return FZ.nyvol_data()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    c['paths']['artifacts'] = str(tmp_path / 'artifacts')
    return c


@pytest.mark.skipif(not REAL, reason='real data not present')
def test_refit_is_byte_identical_and_an_edited_file_is_refused(cfg, data):
    FZ.fit(cfg, data)
    first = {p: open(FZ.frozen_path(cfg, p), 'rb').read() for p in cfg['pairs']}
    FZ.fit(cfg, data)
    assert {p: open(FZ.frozen_path(cfg, p), 'rb').read() for p in cfg['pairs']} == first
    rec = FZ.load(cfg, 'EURUSD')
    assert len(rec['coef_harx']) == 15 and len(rec['coef_har']) == 4 and rec['last_day'] <= '2026-11-20'
    with open(FZ.frozen_path(cfg, 'GBPUSD'), 'a', encoding='utf-8') as fh:
        fh.write(' ')
    with pytest.raises(FZ.FrozenChanged):
        FZ.load(cfg, 'GBPUSD')


@pytest.mark.skipif(not REAL, reason='real data not present')
def test_our_ols_reproduces_fx_nyvol_last_fold_harx(data):
    from src.fx_nyvol.walk_forward import load_oos, usable
    from src.fx_nyvol.splits import make_folds
    by_pair, ny = data
    oos = load_oos(ny, 'lgbm')
    rows_all = usable(by_pair)
    last = max(d.index.max() for d in rows_all.values())
    cols = ny['features']['columns']
    for pair in ('EURUSD', 'GBPUSD'):
        rows = rows_all[pair]
        f = make_folds(rows.index, ny, last)[-1]
        train = np.r_[f.fit, f.val]
        coef = FZ.ols(rows[cols].to_numpy(float)[train], rows['y'].to_numpy(float)[train])
        mine = np.array([FZ.predict(coef, x) for x in rows[cols].to_numpy(float)[f.eval]])
        theirs = oos[(oos['pair'] == pair) & (oos['fold'] == f.k)].sort_values('date')['p_harx'].to_numpy()
        np.testing.assert_allclose(mine, theirs, atol=1e-6)
