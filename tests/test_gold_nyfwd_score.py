"""gold-nyrange-forward, group 5: one-shot scoring and the report."""
import numpy as np
import pandas as pd
import pytest

from src.gold_nyfwd import live as LV
from src.gold_nyfwd import prereg as PR
from src.gold_nyfwd import report as RP
from src.gold_nyfwd import score as SC
from src.gold_nyfwd.registry import REGISTRY, AlreadyScored, StageOrderError, path
from tests._gold_nyfwd_fakes import cfg_in, common_files_in_tmp  # noqa: F401

COMMIT = '2026-12-01T10:00:00+02:00'          # 09:00 Berlin: 2026-12-01 is not a forward day


def gate(_c):
    return COMMIT


@pytest.fixture
def cfg(tmp_path):
    c = cfg_in(tmp_path)
    PR.write(c)
    return c


def plant(cfg, n, harx_better=True, seed=0, start='2026-12-01'):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n, tz='UTC')
    fc, st = [], []
    for d in days:
        y = rng.normal(-5.0, 0.4)
        p_har = y + rng.normal(0, 0.45)
        p_harx = y + rng.normal(0, 0.25 if harx_better else 0.45)
        fc.append({'date': d, 'status': 'ok', 'contract': 'USDIndMar27', 'close14': 2600.0, 'p_harx': p_harx,
                   'p_har': p_har, 'F': float(np.exp(p_harx) * 2600), 'logged_at': d + pd.Timedelta(hours=15)})
        st.append({'date': d, 'status': 'ok', 'y': y, 'p_harx': p_harx, 'p_har': p_har,
                   'se_harx': (y - p_harx) ** 2, 'se_har': (y - p_har) ** 2, 'realised_high': 2610.0,
                   'realised_low': 2590.0, 'settled_at': d + pd.Timedelta(hours=22)})
    LV.append_log(cfg, 'forecasts', fc)
    LV.append_log(cfg, 'settlements', st)
    return days


def test_the_commit_day_is_not_a_forward_day():
    assert SC.committed_label_day(COMMIT) == pd.Timestamp('2026-12-01', tz='UTC')


def test_planted_improvement_is_keep_on_the_first_60_forward_days(cfg):
    days = plant(cfg, 75)
    row = SC.score(cfg, today='2027-03-20', gate=gate)
    assert row['verdict'] == 'KEEP' and row['n_days'] == 60 and row['ci_upper'] < 0
    assert row['first_day'] == str(days[1].date())              # the commit day 2026-12-01 is excluded
    assert row['last_day'] == str(days[60].date()) and row['mse_ratio'] < 1


def test_no_difference_is_drop(cfg):
    plant(cfg, 61, harx_better=False, seed=3)
    assert SC.score(cfg, today='2027-03-20', gate=gate)['verdict'] == 'DROP'


def test_40_days_before_the_deadline_refuse_and_write_nothing(cfg):
    plant(cfg, 41)
    with pytest.raises(StageOrderError):
        SC.score(cfg, today='2027-02-15', gate=gate)
    import os
    assert not os.path.exists(path(cfg, REGISTRY))


def test_40_days_at_the_deadline_are_underpowered(cfg):
    plant(cfg, 41)
    row = SC.score(cfg, today='2027-04-30', gate=gate)
    assert row['verdict'] == 'UNDERPOWERED — NO DECISION' and row['n_days'] == 40


def test_rescoring_refuses_and_leaves_the_registry_byte_identical(cfg):
    plant(cfg, 61)
    SC.score(cfg, today='2027-03-20', gate=gate)
    before = open(path(cfg, REGISTRY), 'rb').read()
    with pytest.raises(AlreadyScored):
        SC.score(cfg, today='2027-03-21', gate=gate)
    assert open(path(cfg, REGISTRY), 'rb').read() == before


# -- report ------------------------------------------------------------------------------------------

def committed():
    return SC.committed_label_day(COMMIT)


def test_pre_score_report_shows_counts_and_no_error_figure(cfg):
    plant(cfg, 31)
    LV.append_log(cfg, 'forecasts', [{'date': pd.Timestamp('2027-01-20', tz='UTC'), 'status': 'no dollar data'}])
    text = RP.build(cfg, committed(), today='2027-01-20', generated='T')
    assert '**30 / 60**' in text and 'not scored yet' in text and 'no dollar data: 1' in text
    assert 'USDIndMar27: 31 days' in text
    assert 'se_' not in text and 'MSE' not in text and 'mse' not in text


def test_report_is_deterministic_apart_from_the_generated_line(cfg):
    plant(cfg, 61)
    SC.score(cfg, today='2027-03-20', gate=gate)
    a = RP.build(cfg, committed(), today='2027-03-20').splitlines()
    b = RP.build(cfg, committed(), today='2027-03-20').splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and all(a[i].startswith('Generated') for i in diff)
    text = '\n'.join(a)
    assert '**KEEP**' in text and 'HAR-X MSE / HAR MSE' in text and '| 2026-12 |' in text
