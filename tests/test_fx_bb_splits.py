"""fx-bb-reversion, group 5.1: expanding 12-month folds over event dates with a 10-business-day gap."""
import numpy as np
import pandas as pd

from src.fx_bb import load_config
from src.fx_bb import splits as S

CFG = load_config()


def events(seed=0, start='2012-03-01', end='2026-11-13', n=1200):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    d = pd.DatetimeIndex(np.sort(rng.choice(days, size=n, replace=True)))
    t3 = pd.DatetimeIndex([x + pd.offsets.BDay(3) for x in d])
    return d, t3


def test_no_training_outcome_reaches_the_next_slice():
    d, t3 = events()
    folds = S.make_folds(d, t3, CFG)
    assert folds
    for f in folds:
        v0 = d[f.val].min()
        assert (S.bdays_between(t3[f.fit], v0) > 10).all()
        assert (S.bdays_between(t3[f.val], f.start) > 10).all()
        assert (S.bdays_between(t3[f.fit], f.start) > 10).all()
        assert d[f.fit].max() < v0 and d[f.val].max() < f.start
        assert set(d[f.val]).isdisjoint(set(d[f.fit]))


def test_fold_starting_2020_trains_only_on_t3_more_than_10_business_days_before():
    d, t3 = events(seed=1)
    f = [x for x in S.make_folds(d, t3, CFG) if x.start == pd.Timestamp('2020-01-01')][0]
    train = np.r_[f.fit, f.val]
    assert (np.busday_count(t3[train].to_numpy().astype('datetime64[D]'), np.datetime64('2020-01-01')) > 10).all()


def test_every_out_of_sample_event_is_predicted_once_and_folds_are_contiguous():
    d, t3 = events(seed=2)
    folds = S.make_folds(d, t3, CFG)
    tested = np.concatenate([f.eval for f in folds])
    assert len(tested) == len(set(tested)) == int((d >= pd.Timestamp('2015-01-01')).sum())
    assert [f.start for f in folds][0] == pd.Timestamp('2015-01-01')
    assert all(a.end == b.start for a, b in zip(folds, folds[1:]))
    assert all(f.end - f.start >= pd.Timedelta(days=365) for f in folds)


def test_validation_is_about_the_last_15_percent_and_keeps_whole_dates():
    d, t3 = events(seed=3)
    for f in S.make_folds(d, t3, CFG):
        n = len(f.fit) + len(f.val)
        assert 0.10 < len(f.val) / n < 0.25
        assert not set(d[f.val]) & set(d[f.fit])


def test_unsorted_events_are_refused():
    d, t3 = events()
    try:
        S.make_folds(d[::-1], t3[::-1], CFG)
    except ValueError as e:
        assert 'sorted' in str(e)
    else:
        raise AssertionError('expected a refusal')
