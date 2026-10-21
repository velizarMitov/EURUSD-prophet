"""Gold stage splits (task 3.3; spec "Unified chronological split")."""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.gold import features as F
from src.gold import load_config
from src.gold import splits as S
from src.gold import targets as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config()


def test_validation_never_enters_training_and_the_boundary_row_is_purged():
    p = S.stage_positions(1000, 'validation', CFG)
    assert p['fit_end'] == 700 and p['score_end'] == 800
    assert not np.intersect1d(p['fit'], p['score']).size
    assert p['fit'].max() == 698                      # row 699's target is bar 700


def test_test_and_final_stages():
    t = S.stage_positions(1000, 'test', CFG)
    assert t['score'][0] == 800 and t['score'][-1] == 999 and t['fit'].max() == 798
    f = S.stage_positions(1000, 'final', CFG)
    assert f['score'].size == 0 and f['fit'].max() == 998


def test_training_preprocessing_ignores_validation_rows():
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.date_range('2018-01-01', periods=n, freq='D', tz='UTC')
    c = 1300 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({'open': c, 'high': c * 1.005, 'low': c * 0.995, 'close': c}, index=idx)
    p = S.stage_positions(n, 'validation', CFG)

    def fitted(frame):
        X = F.build(frame, 'D1').to_numpy()[p['fit']]
        y = G.build_targets(frame['close'])['ret'][p['fit']]
        keep = ~np.isnan(X).any(axis=1)
        sc = StandardScaler().fit(X[keep])
        return sc.mean_, sc.scale_, y

    altered = df.copy()
    altered.iloc[p['fit_end']:p['score_end']] *= 1.8
    for a, b in zip(fitted(df), fitted(altered)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, 'results', 'gold', 'gold_d1.csv')),
                    reason='no live gold D1 cache')
def test_live_d1_split_sizes():
    from src.gold import data as D
    n = len(D.read_cache(os.path.join(REPO, 'results', 'gold', 'gold_d1.csv')))
    p = S.stage_positions(n, 'validation', CFG)
    assert abs(p['fit_end'] - 2425) < 30 and abs(len(p['score']) - 347) < 10
