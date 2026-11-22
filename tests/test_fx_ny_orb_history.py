"""fx-ny-orb-hot, group 3.2: the historical days on the real M15 bars and fx_nyvol's walk-forward HAR-X."""
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import REPO, load_config
from src.fx_ny_orb import history as H
from src.fx_ny_orb import rule as R

CFG = load_config()
REAL = all(os.path.exists(os.path.join(REPO, *CFG['paths']['m1'][p].split('/'))) for p in CFG['pairs']) and \
    os.path.exists(os.path.join(REPO, 'results', 'fx_nyvol', 'oos_lgbm.csv.gz'))


@pytest.fixture(scope='module')
def real(request):
    if not REAL:
        pytest.skip('real data not present')
    return H.assemble(CFG)


@pytest.mark.skipif(not REAL, reason='real data not present')
def test_hot_counts_reproduce_the_probe(real):
    days, _ = real
    hot = days[days['hot']].groupby('pair').size().to_dict()
    assert hot == {'EURUSD': 230, 'GBPUSD': 185}
    assert set(days['day_reason']) <= {'', 'no M15 data', 'corrupt spread'}


@pytest.mark.skipif(not REAL, reason='real data not present')
def test_close14_matches_fx_nyvol_h1_close_within_one_pip(real):
    from src.fx_nyvol import load_config as nyvol_config
    from src.fx_nyvol.target import dataset, load_frames
    days, _ = real
    ny = dataset(*load_frames(nyvol_config()), nyvol_config())
    ok = []
    for pair, g in days[days['day_reason'] == ''].groupby('pair'):
        h1 = ny[pair]['close14'].reindex(pd.DatetimeIndex(g['date']))
        diff = np.abs(g['close14'].to_numpy() - h1.to_numpy()) / CFG['pip'][pair]
        ok.append(np.nanmean(diff <= 1.0))
        assert np.nanmean(diff <= 1.0) >= 0.99, (pair, np.nanmean(diff <= 1.0))
    assert len(ok) == 2


def test_assembly_never_computes_an_exit(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('the assembly must not compute an exit')
    monkeypatch.setattr(R, 'scan_exit', boom)
    monkeypatch.setattr(R, 'simulate_day', boom)
    idx = pd.date_range('2024-01-10 14:00', '2024-01-10 20:45', freq='15min', tz='UTC')
    bars = pd.DataFrame({'mo': 1.1, 'mh': 1.1002, 'ml': 1.0998, 'mc': 1.1, 'spread_price': 0.0, 'corrupt': False},
                        index=idx)
    oos = pd.DataFrame({'pair': ['EURUSD', 'GBPUSD'], 'date': [pd.Timestamp('2024-01-10', tz='UTC')] * 2,
                        'p_harx': [-5.4, -5.6], 'hot_threshold': [-5.5, -5.5], 'y': [-5.3, -5.7],
                        'dst_mismatch': 0.0, 'is_nfp_day': 0.0, 'is_fomc_day': 0.0})
    days, day_bars = H.assemble(CFG, bars={'EURUSD': bars, 'GBPUSD': bars.iloc[:0]}, oos=oos)
    e = days.set_index('pair')
    assert e.loc['EURUSD', 'hot'] and not e.loc['GBPUSD', 'hot']
    assert e.loc['EURUSD', 'F'] == pytest.approx(np.exp(-5.4) * 1.1) and e.loc['GBPUSD', 'day_reason'] == 'no M15 data'
