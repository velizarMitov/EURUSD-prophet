"""Gold bar-label clock guard (task 2.2; spec "Runs refuse an unverified bar clock")."""
import os

import pandas as pd
import pytest

from src.gold import clock as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _weeks(n_weeks, n_wrong, start='2023-01-02'):
    """n_weeks of Mon-Fri H1 bars opening on the predicted label; the first
    n_wrong weeks open one hour late (a shifted broker clock)."""
    stamps = []
    for w in range(n_weeks):
        monday = pd.Timestamp(start, tz='UTC') + pd.Timedelta(weeks=w)
        open_ = monday - pd.Timedelta(days=1) + pd.Timedelta(hours=C.expected_open_hour(monday)) \
            if C.expected_open_hour(monday) == 23 else monday
        if w < n_wrong:
            open_ = open_ + pd.Timedelta(hours=1)
        stamps.extend(pd.date_range(open_, monday + pd.Timedelta(days=4, hours=22), freq='h'))
    return pd.DatetimeIndex(stamps)


def test_expected_hour_follows_dst_mismatch():
    assert C.expected_open_hour(pd.Timestamp('2024-01-15')) == 0      # aligned week
    assert C.expected_open_hour(pd.Timestamp('2024-03-11')) == 23     # US on DST, EU not yet
    assert C.expected_open_hour(pd.Timestamp('2024-10-28')) == 23     # EU off DST, US not yet
    assert C.expected_open_hour(pd.Timestamp('2024-11-04')) == 0      # both off DST again


def test_passes_at_99_percent():
    ev = C.verify(_weeks(100, 1))
    assert ev['match_rate'] >= 0.98


def test_fails_at_90_percent():
    with pytest.raises(C.GoldClockMismatch):
        C.verify(_weeks(100, 10))


def test_no_weekend_cannot_establish_clock():
    with pytest.raises(C.GoldClockMismatch):
        C.verify(pd.date_range('2024-01-02', periods=20, freq='h', tz='UTC'))


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, 'results', 'gold', 'gold_h1.csv')),
                    reason='no live gold H1 cache')
def test_live_cache_matches_the_measured_clock():
    from src.gold import data as D
    h1 = D.read_cache(os.path.join(REPO, 'results', 'gold', 'gold_h1.csv'))
    ev = C.verify(h1.index)
    assert ev['match_rate'] > 0.97
