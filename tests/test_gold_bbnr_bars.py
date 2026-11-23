"""gold-bb-nyrange, task 2.1: gold H1 input and NY-close D1 bars."""
import numpy as np
import pandas as pd
import pytest

from src.gold.clock import GoldClockMismatch, expected_open_hour
from src.gold_bbnr import bars as B
from src.gold_bbnr import load_config

CFG = load_config()


def week_bars(monday: str, shift_hours: int = 0) -> pd.DataFrame:
    """Five trading days of hourly bars on the Berlin label clock, the weekly open where gold opens."""
    mon = pd.Timestamp(monday)
    open_h = expected_open_hour(mon)                      # 0 or 23 on the label clock
    first = (mon - pd.Timedelta(days=1) + pd.Timedelta(hours=23)) if open_h == 23 else mon
    idx = []
    t = first
    end = mon + pd.Timedelta(days=4, hours=22)
    while t <= end:
        if t.hour != 23 or t == first:                     # the daily break, except the weekly open
            idx.append(t)
        t += pd.Timedelta(hours=1)
    idx = pd.DatetimeIndex(idx) + pd.Timedelta(hours=shift_hours)
    v = np.linspace(2000, 2010, len(idx))
    return pd.DataFrame({'open': v, 'high': v + 1, 'low': v - 1, 'close': v}, index=idx.tz_localize('UTC'))


def frame(weeks, shift=0):
    return pd.concat([week_bars(w, shift) for w in weeks])


def write(tmp_path, df):
    p = tmp_path / 'gold_h1.csv'
    out = df.copy()
    out.index.name = 'time'
    out.reset_index().assign(time=lambda d: d['time'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')).to_csv(p, index=False)
    return str(p)


WEEKS = [str(d.date()) for d in pd.date_range('2019-01-07', periods=40, freq='W-MON')]


def test_load_clips_the_window_and_checks_the_clock(tmp_path):
    df = frame(WEEKS)
    late = week_bars('2026-11-23')
    p = write(tmp_path, pd.concat([df, late]))
    cfg = {**CFG, 'data': {**CFG['data'], 'window_end': '2019-12-31'}}
    h1 = B.load_h1(cfg, p)
    assert h1.index.max() < pd.Timestamp('2020-01-01', tz='UTC') and list(h1.columns) == B.OHLC


def test_a_shifted_clock_is_refused(tmp_path):
    p = write(tmp_path, frame(WEEKS, shift=2))
    with pytest.raises(GoldClockMismatch):
        B.load_h1(CFG, p)


def test_dst_mismatch_23h_bar_opens_the_next_day():
    # 2019-03-11 .. 03-29: the US is on summer time, Europe is not -> the weekly open is labelled 23:00 Sunday.
    wk = week_bars('2019-03-18')
    assert (wk.index.hour == 23).sum() == 1
    d1 = B.d1_ny(wk, CFG)
    sunday_bar = wk.index[0]
    assert sunday_bar.day == 17 and sunday_bar.hour == 23
    td = B.h1_day_positions(wk, CFG)
    assert pd.Timestamp(td[0]) == pd.Timestamp('2019-03-18')
    assert pd.Timestamp('2019-03-17') not in d1.index


def test_bar_labelled_22_berlin_closes_its_own_day():
    wk = week_bars('2019-06-03')
    td = pd.DatetimeIndex(B.h1_day_positions(wk, CFG))
    at22 = wk.index.hour == 22
    assert (td[at22] == wk.index[at22].tz_localize(None).normalize()).all()


def test_short_day_is_invalid():
    wk = week_bars('2019-06-03')
    tue = wk.index.tz_localize(None).normalize() == pd.Timestamp('2019-06-04')
    keep = ~tue | (wk.index.hour < 8)
    d1 = B.d1_ny(wk[keep], CFG)
    assert not d1.loc['2019-06-04', 'valid'] and d1.loc['2019-06-05', 'valid']
    assert pd.Timestamp('2019-06-04') not in B.valid(d1).index
