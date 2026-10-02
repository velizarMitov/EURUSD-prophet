"""forecast_eval splitting (tasks 3.1-3.3)."""
import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import splits as S


def _windows_overlap(train, test, h):
    """Closed windows [t, t+h] intersect iff |i - j| <= h."""
    test = np.asarray(test)
    for i in train:
        if np.any(np.abs(test - i) <= h):
            return True
    return False


# ── 3.1 purged walk-forward ──────────────────────────────────────────────

def test_walk_forward_purges_overlapping_labels_at_h24():
    for sp in S.walk_forward(2000, h=24, n_splits=5, min_train=500):
        assert not _windows_overlap(sp.train, sp.test, 24)


def test_walk_forward_training_precedes_test():
    for sp in S.walk_forward(1000, h=4, n_splits=4, min_train=200):
        assert sp.train.max() < sp.test.min()


def test_walk_forward_is_expanding():
    sps = S.walk_forward(1000, h=4, n_splits=4, min_train=200)
    assert all(len(b.train) > len(a.train) for a, b in zip(sps, sps[1:]))


@pytest.mark.parametrize('h', [1, 48])
def test_embargo_gap_after_every_test_run_is_at_least_h(h):
    sps = S.walk_forward(3000, h=h, n_splits=4, min_train=600) + S.cpcv(3000, h=h).splits
    for sp in sps:
        assert sp.embargo >= h
        for a, b in S._runs(np.sort(sp.test)):
            after = sp.train[(sp.train > b)]
            if after.size:
                assert after.min() - b > h, f'train resumes {after.min() - b} rows after test at h={h}'


def test_purge_removes_exactly_the_leaking_rows():
    tr = S.purged_train(100, np.arange(40, 50), h=3)
    assert set(range(37, 53)).isdisjoint(tr)
    assert {36, 53} <= set(tr)


# ── 3.2 CPCV ─────────────────────────────────────────────────────────────

def test_cpcv_counts():
    cv = S.cpcv(600, h=2, n_groups=6, k=2)
    assert len(cv.splits) == 15 and cv.n_paths == 5


def test_cpcv_each_path_covers_every_observation_once():
    cv = S.cpcv(600, h=2)
    preds = [sp.test.astype(float) for sp in cv.splits]     # predict = own position
    paths = cv.assemble(preds)
    assert paths.shape == (5, 600)
    for row in paths:
        assert not np.isnan(row).any()
        assert np.array_equal(row, np.arange(600))


def test_cpcv_splits_are_purged():
    cv = S.cpcv(1200, h=12)
    for sp in cv.splits:
        assert not _windows_overlap(sp.train, sp.test, 12)


# ── 3.3 uniqueness ───────────────────────────────────────────────────────

def test_uniqueness_is_one_at_h1():
    w = S.uniqueness_weights(np.arange(500), h=1, grid_len=500)
    assert np.all(w == 1.0)
    assert S.effective_n(np.arange(500), 1, 500) == 500


def test_overlap_shrinks_effective_n():
    n = 5000
    eff = S.effective_n(np.arange(n), 24, n)
    assert eff < n / 20                     # ~n/24
    assert eff == pytest.approx(n / 24, rel=0.02)


def test_parity_with_existing_uniqueness_estimators():
    from src.h1_direction_model import mean_label_uniqueness
    from src.h1_horizon_feasibility import uniqueness_from_spans
    idx = pd.date_range('2020-01-01', periods=300, freq='h')
    ctx = pd.DataFrame(index=idx)
    assert np.mean(S.uniqueness_weights(np.arange(300), 1, 300)) == mean_label_uniqueness(ctx, idx) == 1.0
    starts = np.arange(0, 900, 3)
    ref = uniqueness_from_spans(starts, starts + 23, grid_len=950)
    assert np.mean(S.uniqueness_weights(starts, 24, 950)) == pytest.approx(ref, abs=1e-12)


# ── 14.5 the M15 cadence on the same arithmetic ──────────────────────────

def test_m15_block_length_is_one_session():
    """The scored rows are session-restricted, so the session -- not the
    calendar day -- is the unit of dependence (design D4)."""
    from src.forecast_eval import stats as ST
    assert ST.block_length(1, 'M15') == 26
    assert ST.block_length(26, 'M15') == 26
    assert ST.block_length(40, 'M15') == 40
    assert ST.block_length(1, 'H1') == 24 and ST.block_length(1, 'D1') == 5


def test_m15_purge_and_embargo_at_the_longest_horizon():
    n, h = 4000, 26
    wf = S.walk_forward(n, h=h, n_splits=5, min_train=2000)
    for sp in wf:
        assert not _windows_overlap(sp.train, sp.test, h)
        assert sp.train.max() < sp.test.min()
        assert sp.embargo >= h
    cv = S.cpcv(n, h=h, n_groups=6, k=2)
    assert len(cv.splits) == 15 and cv.n_paths == 5
    for sp in cv.splits:
        assert not _windows_overlap(sp.train, sp.test, h)


def test_m15_eligible_positions_keep_the_split_arithmetic_leakage_free():
    """Training is restricted to eligible session rows; the purge still has to
    hold against the test block on the FULL bar grid, because a label spans real
    bars whether or not they are eligible."""
    from src.forecast_eval import m15_data as M
    idx = pd.date_range(pd.Timestamp('2026-06-01', tz='UTC'), periods=96 * 40, freq='15min')
    h = 4
    elig = M.eligible_positions(idx, h)
    n = len(idx)
    wf = S.walk_forward(n, h=h, n_splits=4, min_train=n // 2)
    for sp in wf:
        train = np.intersect1d(sp.train, elig)
        test = np.intersect1d(sp.test, elig)
        if not train.size or not test.size:
            continue
        assert not _windows_overlap(train, test, h)
        assert train.max() < test.min()


def test_m15_uniqueness_is_one_at_h1_and_shrinks_with_the_horizon():
    starts = np.arange(2000)
    assert np.all(S.uniqueness_weights(starts, h=1, grid_len=2000) == 1.0)
    assert S.effective_n(starts, 4, 2000) == pytest.approx(2000 / 4, rel=0.02)
