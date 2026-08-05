"""
Tests for the MT5 coverage guard (src/mt5_coverage.py).

The guard exists because ``results/pooled_h1/USDCHF_h1.csv`` was found missing
42 days while reporting 70,000 bars and a correct last timestamp. The two
properties that matter are symmetrical and both are tested here:

  * it FIRES on an injected hole, naming the missing range
  * it stays QUIET on a normal weekend and on the year-end closure

The second is not a formality. Measured on this broker's own history the
year-end closures run 73.0-77.0 hours -- ABOVE the 72h bound -- so a naive flat
threshold would raise every Christmas and the guard would be switched off within
a week.
"""

import numpy as np
import pandas as pd
import pytest

from src.mt5_coverage import (BARS_PER_DAY, CoverageError, MAX_GAP_HOURS,
                              YEAR_END_ALLOWANCE_HOURS, assert_coverage,
                              find_gaps, find_thin_months, sync_symbol)


def _fx_hours(start, end):
    """Hourly index with FX weekends removed (Sat all day, Sun before 22:00)."""
    idx = pd.date_range(start, end, freq='h')
    keep = ~((idx.dayofweek == 5) | ((idx.dayofweek == 6) & (idx.hour < 22)))
    return idx[keep]


def _frame(index):
    n = len(index)
    return pd.DataFrame({'open': np.ones(n), 'high': np.ones(n),
                         'low': np.ones(n), 'close': np.ones(n)}, index=index)


# --------------------------------------------------------------------------
# it must stay QUIET on normal data
# --------------------------------------------------------------------------

def test_clean_year_of_fx_hours_passes():
    df = _frame(_fx_hours('2025-02-01', '2025-11-30'))
    assert assert_coverage(df, 'H1', label='clean') is df


def test_normal_weekend_is_not_a_gap():
    """The Fri 21:00 -> Sun 22:00 weekend is ~49h and must never fire."""
    idx = _fx_hours('2025-03-03', '2025-03-31')
    gaps = find_gaps(idx)
    assert gaps == [], 'weekend falsely reported as a hole: %s' % (gaps,)
    biggest = pd.Series(idx).diff().dt.total_seconds().max() / 3600.0
    assert 40 < biggest < MAX_GAP_HOURS


def test_year_end_closure_does_not_fire():
    """The real closure this broker emits: 2023-12-22 22:00 -> 2023-12-26 00:00,
    74h, which EXCEEDS the 72h bound and must still be tolerated."""
    idx = _fx_hours('2023-11-15', '2024-02-15')
    hole = (idx > pd.Timestamp('2023-12-22 22:00')) & (idx < pd.Timestamp('2023-12-26 00:00'))
    idx = idx[~hole]
    span = pd.Series(idx).diff().dt.total_seconds().max() / 3600.0
    assert span == pytest.approx(74.0, abs=1.0), 'fixture is not the 74h closure'
    assert span > MAX_GAP_HOURS, 'fixture would pass even without the allowance'
    assert find_gaps(idx) == []
    assert_coverage(_frame(idx), 'H1', label='christmas')


def test_a_gap_just_over_the_allowance_still_fires_at_year_end():
    """The year-end allowance widens the bound; it does not remove it."""
    idx = _fx_hours('2023-11-15', '2024-02-15')
    hole = ((idx > pd.Timestamp('2023-12-20 00:00'))
            & (idx < pd.Timestamp('2023-12-26 00:00')))       # 144h
    idx = idx[~hole]
    gaps = find_gaps(idx)
    assert gaps and gaps[0][2] > YEAR_END_ALLOWANCE_HOURS


# --------------------------------------------------------------------------
# it must FIRE on a hole
# --------------------------------------------------------------------------

def test_injected_hole_fires_and_names_the_range():
    idx = _fx_hours('2025-02-01', '2025-11-30')
    lo, hi = pd.Timestamp('2025-06-15 06:00'), pd.Timestamp('2025-07-28 00:00')
    holed = idx[~((idx > lo) & (idx < hi))]
    # The named boundary is the last bar PRESENT before the hole and the first
    # one after it -- derived from the fixture, not assumed: `lo` here lands on a
    # Sunday, so the last real bar is the preceding Friday close.
    before = holed[holed <= lo].max()
    after = holed[holed >= hi].min()
    with pytest.raises(CoverageError) as e:
        assert_coverage(_frame(holed), 'H1', label='injected')
    msg = str(e.value)
    assert str(before) in msg and str(after) in msg
    assert 'REFUSED' in msg and 'Market Watch' in msg
    assert '%.1f days' % ((after - before).total_seconds() / 86400.0) in msg


def test_the_real_usdchf_signature_is_caught():
    """The exact shape of the file that motivated this module: a 42-day hole in
    the middle, while the bar COUNT stays plausible because the frame simply
    reaches further back."""
    idx = _fx_hours('2025-01-01', '2026-07-28')
    lo, hi = pd.Timestamp('2026-06-15 06:00'), pd.Timestamp('2026-07-28 00:00')
    holed = idx[~((idx > lo) & (idx < hi))]
    assert len(holed) > 9000, 'count alone still looks healthy -- that is the point'
    assert holed.max() >= pd.Timestamp('2026-07-28 00:00'), 'last timestamp looks correct too'
    with pytest.raises(CoverageError, match='2026-06-15'):
        assert_coverage(_frame(holed), 'H1', label='usdchf-signature')


def test_thin_month_fires_even_without_a_contiguous_gap():
    """Scattered loss: a month thinned to 40% with no single long gap."""
    idx = _fx_hours('2025-02-01', '2025-08-31')
    june = idx.month == 6
    rng = np.random.default_rng(0)
    drop = np.zeros(len(idx), bool)
    j = np.flatnonzero(june)
    drop[rng.choice(j, int(len(j) * 0.62), replace=False)] = True
    thinned = idx[~drop]
    assert find_gaps(thinned) == [], 'fixture should have no long contiguous gap'
    thin = find_thin_months(thinned, 'H1')
    assert thin and thin[0][0].month == 6
    with pytest.raises(CoverageError, match='thin month 2025-06'):
        assert_coverage(_frame(thinned), 'H1', label='thinned')


def test_partial_first_and_last_months_are_exempt():
    """A fetch almost always starts and ends mid-month; that is not a hole."""
    idx = _fx_hours('2025-02-25', '2025-06-03')
    assert find_thin_months(idx, 'H1') == []
    assert_coverage(_frame(idx), 'H1', label='partial-ends')


def test_m15_uses_its_own_expected_rate():
    idx = pd.date_range('2025-03-01', '2025-06-30', freq='15min')
    keep = ~((idx.dayofweek == 5) | ((idx.dayofweek == 6) & (idx.hour < 22)))
    assert find_thin_months(idx[keep], 'M15') == []
    assert BARS_PER_DAY['M15'] == pytest.approx(4 * BARS_PER_DAY['H1'])


def test_empty_and_tiny_frames_are_not_flagged():
    assert assert_coverage(None, 'H1') is None
    empty = _frame(pd.DatetimeIndex([]))
    assert assert_coverage(empty, 'H1') is empty


# --------------------------------------------------------------------------
# prevention helper
# --------------------------------------------------------------------------

class _FakeMT5:
    TIMEFRAME_H1 = 16385

    def __init__(self, ready_after=0):
        self.selected, self.calls, self._ready_after = [], 0, ready_after

    def symbol_select(self, symbol, enable):
        self.selected.append((symbol, enable))
        return True

    def copy_rates_range(self, symbol, tf, start, end):
        self.calls += 1
        return np.zeros(3) if self.calls > self._ready_after else None


def test_sync_symbol_selects_into_market_watch_and_probes():
    m = _FakeMT5()
    assert sync_symbol(m, 'EURUSD') is True
    assert m.selected == [('EURUSD', True)]
    assert m.calls == 1


def test_sync_symbol_retries_then_gives_up_without_raising():
    m = _FakeMT5(ready_after=99)
    assert sync_symbol(m, 'CHFJPY', attempts=2, sleep_seconds=0) is False
    assert m.calls == 2


# --------------------------------------------------------------------------
# the guard against the project's own live files
# --------------------------------------------------------------------------

@pytest.mark.parametrize('rel,tf', [('results/eurusd_h1.csv', 'H1'),
                                    ('results/eurusd_m15.csv', 'M15')])
def test_live_production_caches_are_clean(rel, tf):
    """The live H1 cache feeds the production ensemble; it must stay hole-free."""
    import os
    if not os.path.exists(rel):
        pytest.skip('%s not present' % rel)
    d = pd.read_csv(rel, index_col=0)
    d.index = pd.to_datetime(d.index, utc=True, format='mixed').tz_convert('UTC').tz_localize(None)
    assert_coverage(d.sort_index(), tf, label=rel)
