"""fx-ny-volatility, group 6.1: the descriptive report."""
import numpy as np
import pandas as pd
import pytest

from src.fx_nyvol import labels as LB
from src.fx_nyvol import report as RP
from src.fx_nyvol import score as SC
from src.fx_nyvol import target as TG
from src.fx_nyvol import walk_forward as WF
from tests.fx_nyvol_synth import CFG, POINTS, tmp_cfg, walk_frames


def test_q75_threshold_uses_only_the_previous_60_days():
    y = pd.Series(np.random.default_rng(0).normal(size=200))
    thr = WF.hot_threshold(y, CFG)
    assert thr.iloc[:60].isna().all()
    assert thr.iloc[100] == pytest.approx(np.quantile(y.iloc[40:100], 0.75))
    y2 = y.copy()
    y2.iloc[100] = 1e6                                    # today's value never enters today's threshold
    assert WF.hot_threshold(y2, CFG).iloc[100] == thr.iloc[100]


def test_hot_day_precision_recall_and_accuracy():
    y = np.array([1.0, 2, 3, 4])
    p = np.array([2.5, 1, 3.5, 3.5])
    h = RP.hot(y, p, np.full(4, 2.5))
    assert h['precision'] == pytest.approx(1.0) and h['recall'] == pytest.approx(1.0) and h['accuracy'] == 1.0
    assert RP.metrics(y, y, p)['r2'] == pytest.approx(1.0)


def test_report_before_scoring_and_two_identical_builds(tmp_path):
    cfg = tmp_cfg(tmp_path)
    cfg['scoring']['resamples'] = 500
    text = open(RP.build(cfg, 'T'), encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Gross of spread and slippage') and 'Not scored yet' in text
    data = TG.dataset(walk_frames(n_weeks=75), POINTS, cfg)
    LB.write_report(cfg, 'T0', data)
    WF.run_lgbm(cfg, 'T1', data, log=lambda *a: None)
    SC.score_lgbm(cfg, 'T2')
    a = open(RP.build(cfg, 'first'), encoding='utf-8').read().splitlines()
    b = open(RP.build(cfg, 'second'), encoding='utf-8').read().splitlines()
    assert [(x, y) for x, y in zip(a, b) if x != y] == [('_Generated first._', '_Generated second._')]
    text = '\n'.join(a)
    for needle in ('HAR-X', 'Spearman', 'Hot sessions', 'Precision', 'Bid parity', 'MSE per fold', 'NFP', 'FOMC',
                   'DST mismatch', 'Relative MSE reduction'):
        assert needle in text, needle
