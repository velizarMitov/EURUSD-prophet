"""fx-bb-reversion, group 4.2: cutting every pair's bars after t leaves the event's feature row unchanged."""
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import daily as D
from src.fx_bb import data_loader as DL
from src.fx_bb import events as E
from src.fx_bb import features as F
from src.fx_bb import load_config
from tests.test_fx_bb_features import six

CFG = load_config()
COLS = CFG['features']['columns']


def _row(ev_row, d1_cut):
    one = pd.DataFrame([ev_row]).reset_index(drop=True)
    return F.build_features(one, d1_cut, CFG).iloc[0]


def test_truncation_on_synthetic_d1():
    d1 = six(400)
    ev, _ = E.build_events(d1, CFG)
    full = F.build_features(ev, d1, CFG)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ev), size=min(20, len(ev)), replace=False):
        t = ev.loc[i, 'date']
        cut = {p: d[d.index <= t] for p, d in d1.items()}
        got = _row(ev.loc[i], cut)
        np.testing.assert_array_equal(got[COLS].to_numpy(), full.loc[i, COLS].to_numpy())
        ev_cut, _ = E.detect(cut[ev.loc[i, 'pair']], ev.loc[i, 'pair'], CFG)
        assert t in set(ev_cut['date'])


REAL = all(os.path.exists(DL.cache_path(p, CFG)) for p in CFG['pairs'])


@pytest.mark.skipif(not REAL, reason='real H1 caches are not present')
def test_truncation_on_the_real_h1_caches_for_20_random_events():
    frames, points = DL.load_all(CFG)
    d1 = D.build_all(frames, points, CFG)
    ev, _ = E.build_events(d1, CFG)
    feats = F.with_features(ev, d1, CFG)
    pool = feats.index[feats['reason2'] == '']
    rng = np.random.default_rng(20261122)
    tds = {p: D.trading_dates(f.index, CFG) for p, f in frames.items()}
    for i in rng.choice(pool, size=20, replace=False):
        t = ev.loc[i, 'date']
        cut_h1 = {p: f[tds[p] <= t] for p, f in frames.items()}        # every bar after t's 17:00 NY close
        cut = D.build_all(cut_h1, points, CFG)
        got = _row(ev.loc[i], cut)
        np.testing.assert_allclose(got[COLS].to_numpy(float), feats.loc[i, COLS].to_numpy(float), rtol=0, atol=0)
        ev_cut, _ = E.detect(cut[ev.loc[i, 'pair']], ev.loc[i, 'pair'], CFG)
        assert t in set(ev_cut['date'])
