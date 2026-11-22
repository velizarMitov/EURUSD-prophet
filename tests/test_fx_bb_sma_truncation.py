"""fx-bb-sma-exit, group 2.3: cutting every bar after the entry bar leaves the entry, levels and dist unchanged."""
import os

import numpy as np
import pytest

from src.fx_bb.daily import build_all
from src.fx_bb.data_loader import cache_path, load_all
from src.fx_bb_sma import bb_view, load_config
from src.fx_bb_sma import trades as T

CFG = load_config()
REAL = all(os.path.exists(cache_path(p, bb_view(CFG))) for p in CFG['pairs'])


@pytest.mark.skipif(not REAL, reason='real H1 caches are not present')
def test_entry_levels_and_dist_unchanged_when_cut_after_the_entry_bar():
    bb = bb_view(CFG)
    frames, points = load_all(bb)
    d1 = build_all(frames, points, bb)
    ctxs = T.contexts(frames, points, d1, CFG)
    tr, _ = T.build_stream(ctxs, d1, CFG)
    pool = tr.index[tr['reason'] == '']
    rng = np.random.default_rng(20261122)
    for i in rng.choice(pool, size=20, replace=False):
        t = tr.loc[i]
        p = t['pair']
        cut_h1 = frames[p][frames[p].index <= t['entry_time']]
        cut_d1 = dict(d1)
        from src.fx_bb.daily import d1_ny
        cut_d1[p] = d1_ny(cut_h1, points[p], CFG['pip'][p], bb)
        x = T.context(p, cut_h1, points[p], cut_d1[p], CFG)
        r = x.ind.loc[t['date']]
        assert x.ind['first_breach'].loc[t['date']]
        assert int(r['dir']) == t['dir'] and r['close'] == t['close'] and r['sma'] == t['sma']
        assert abs(r['close'] - r['sma']) == t['dist'] and r['sma'] == t['tp']
        assert r['close'] - t['dir'] * abs(r['close'] - r['sma']) == t['sl']
        assert int(x.close_pos[t['date']]) == t['entry_pos'] and x.index[-1] == t['entry_time']
