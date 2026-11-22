"""fx-ny-volatility, groups 2.2 and 3.2: the power report and the truncation test on the real caches."""
import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_nyvol import labels as LB
from src.fx_nyvol import target as TG
from src.fx_nyvol.registry import ConfigChanged
from src.fx_session.data_loader import cache_path
from tests.fx_nyvol_synth import CFG, POINTS, tmp_cfg, walk_frames


def test_3470_days_give_about_005_sd():
    assert LB.mde_sd(3470, 0.0125, 0.8) == pytest.approx(0.0523, abs=5e-4)


def test_report_uses_no_model_output_and_refuses_a_changed_config(tmp_path, monkeypatch):
    import src.fx_nyvol as FN
    cfg = tmp_cfg(tmp_path)
    data = TG.dataset(walk_frames(n_weeks=60), POINTS, cfg)
    text = open(LB.write_report(cfg, 'T', data), encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Gross of spread and slippage')
    assert '## Power — before any model' in text and 'Detectable difference' in text
    assert not any(n.startswith(('oos', 'run')) for n in os.listdir(cfg['paths']['results']))
    for mod in ('models', 'walk_forward', 'score'):
        assert f'src.fx_nyvol.{mod}' not in LB.__dict__
    other = copy.deepcopy(cfg)
    other['session']['target_hours'] = [15, 22]
    with pytest.raises(ConfigChanged, match='labels.json'):
        LB.write_report(other, 'T2', data)
    assert FN


REAL = all(os.path.exists(cache_path(p, {'paths': {'data': CFG['paths']['fx_session_data']}})) for p in CFG['factor_pairs'])


@pytest.mark.skipif(not REAL, reason='real H1 caches are not present')
def test_truncation_after_the_14_bar_leaves_every_feature_unchanged_on_the_real_caches():
    frames, points = TG.load_frames(CFG)
    full = TG.dataset(frames, points, CFG)
    pool = full['EURUSD'].index[full['EURUSD']['reason'] == '']
    rng = np.random.default_rng(20261122)
    cols = CFG['features']['columns']
    for d in sorted(rng.choice(pool, size=20, replace=False)):
        cut_at = pd.Timestamp(d) + pd.Timedelta(hours=14)
        cut = {p: f[f.index <= cut_at] for p, f in frames.items()}
        cal = TG.calendar(cut, CFG)
        reason = TG.eligibility(cut, points, CFG, cal)
        tg = TG.targets(cut, points, CFG, cal.index)
        avail = (reason == '').to_numpy()
        from src.fx_nyvol.features import build_features
        feats = build_features(cut, points, CFG, cal.index, {p: tg[p].loc[avail, 'y'] for p in CFG['pairs']})
        for p in CFG['pairs']:
            np.testing.assert_array_equal(feats[p].loc[d, cols].to_numpy(float), full[p].loc[d, cols].to_numpy(float))
