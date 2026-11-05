"""Clocks and session intervals (openspec change gold-market-analytics, task 2.1)."""
from datetime import date

import pandas as pd

from src.gold_analytics import load_config
from src.gold_analytics.clock import display_hhmm, instant_to_label, label_to_instant
from src.gold_analytics.sessions import intervals_for, session_interval, session_intervals

CFG = load_config()
NY = next(s for s in CFG['sessions'] if s['key'] == 'new_york')


def _sofia(label):
    return display_hhmm([label])[0]


def test_label_14_displays_as_15_sofia_winter_and_summer():
    assert display_hhmm(['2026-01-15 14:00', '2026-07-15 14:00']) == ['15:00', '15:00']


def test_the_doubled_fall_back_hour_is_read_in_order():
    labels = pd.date_range('2026-10-25 01:00', periods=4, freq='h', tz='UTC')  # 01, 02, 02(again), 03 in MT5
    labels = labels.delete(3).insert(2, pd.Timestamp('2026-10-25 02:00', tz='UTC'))
    inst = label_to_instant(labels)
    assert [str(t.utcoffset()) for t in inst] == ['2:00:00', '2:00:00', '1:00:00', '1:00:00']
    assert (instant_to_label(inst) == labels).all()


def test_new_york_open_in_january_and_the_march_mismatch_week():
    jan, _ = session_interval(date(2026, 1, 14), NY)
    mar, _ = session_interval(date(2026, 3, 11), NY)  # US on summer time since 8 March, Europe from 29 March
    jul, _ = session_interval(date(2026, 7, 15), NY)
    assert (_sofia(jan), _sofia(mar), _sofia(jul)) == ('15:20', '14:20', '15:20')


def test_a_custom_london_session():
    custom = {'key': 'early', 'start': '07:00', 'end': '10:00', 'zone': 'Europe/London'}
    iv = session_intervals([date(2026, 1, 14), date(2026, 7, 15)], custom)
    assert [(_sofia(a), _sofia(b)) for a, b in zip(iv['start'], iv['end'])] == [('09:00', '12:00'), ('09:00', '12:00')]
    cfg = {**CFG, 'sessions': CFG['sessions'] + [custom]}
    assert len(intervals_for('early', [date(2026, 1, 14)], cfg)) == 1


def test_overlap_is_the_intersection_on_each_day():
    days = [date(2026, 1, 14), date(2026, 3, 11), date(2026, 10, 28)]
    ov = intervals_for('overlap', days, CFG)
    got = [(_sofia(a), _sofia(b)) for a, b in zip(ov['start'], ov['end'])]
    # January: NY 08:20 = 15:20 Sofia, London 16:30 = 18:30 Sofia.
    # March mismatch: NY opens 14:20 Sofia, London still closes 18:30.
    # Late October mismatch: Europe back on winter time, the US not yet: NY 14:20, London 18:30.
    assert got == [('15:20', '18:30'), ('14:20', '18:30'), ('14:20', '18:30')]
