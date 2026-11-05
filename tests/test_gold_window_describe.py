"""Descriptive report (openspec change gold-window-strategy, task 1.2)."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.gold_window import describe as D

# The scratch run of 2026-11-04 read the H1 cache up to this bar.
CUTOFF = pd.Timestamp('2026-11-04 07:00', tz='UTC')


@pytest.fixture(scope='module')
def caches():
    h1, d1 = D.load_caches()
    return h1[h1.time <= CUTOFF].reset_index(drop=True), d1[d1.time <= CUTOFF].reset_index(drop=True)


def test_session_table_matches_the_2026_11_04_figures(caches):
    h = D.hourly_returns(caches[0])
    s = D.by_session(h, D.span_years(h)).round(1)
    assert s.loc['Asia 01-09'].tolist() == pytest.approx([15.4, 6.4, 13.3], abs=0.1)
    assert s.loc['window 16-23'].tolist() == pytest.approx([-3.6, 1.1, -0.4], abs=0.1)


def test_two_runs_differ_only_in_the_generation_line(caches):
    a = D.render(*caches, now=datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)).splitlines()
    b = D.render(*caches, now=datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc)).splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(diff) == 1 and a[diff[0]].startswith('Generated ')


def test_every_section_carries_the_descriptive_label(caches):
    text = D.render(*caches)
    sections = text.split('\n## ')[1:]
    assert len(sections) == 8
    for sec in sections:
        assert D.LABEL in sec, sec.splitlines()[0]
    assert 'Bid-price limitation' in text


def test_tables_have_no_stray_pipes(caches):
    """A '|' inside a header or cell would break the markdown table."""
    for line in D.render(*caches).splitlines():
        if line.startswith('| '):
            header_cells = line.count('|')
            assert header_cells >= 3
    hours = [l for l in D.render(*caches).splitlines() if l.startswith('| 0') or l.startswith('| 1')]
    assert hours and len({l.count('|') for l in hours}) == 1


def test_weekend_gap_is_kept_out_of_the_hour_table(caches):
    h = D.hourly_returns(caches[0])
    assert h.week_open.sum() > 500
    hours = D.by_hour(h, D.span_years(h))
    assert hours[('bars', '')].sum() == (~h.week_open).sum()
