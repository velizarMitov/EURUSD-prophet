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


def test_pinned_prefix_ignores_appended_rows_but_not_edits(tmp_path):
    head = 'time,open,high,low,close\r\n2026-11-20T20:00:00+00:00,1,2,0,1\r\n'
    last = f'{B.PINNED_LAST},1,2,0,1.5\r\n'
    p = tmp_path / 'g.csv'
    p.write_bytes((head + last).encode())
    pinned = B.pinned_bytes(str(p))
    p.write_bytes((head + last + '2026-11-20T22:00:00+00:00,1,2,0,9\r\n').encode())
    assert B.pinned_bytes(str(p)) == pinned
    p.write_bytes((head.replace(',1,2,0,1', ',1,2,0,7') + last).encode())
    assert B.pinned_bytes(str(p)) != pinned
    p.write_bytes(head.encode())                               # the pinned row is gone: the whole file
    assert B.pinned_bytes(str(p)) == head.encode()


def test_loader_drops_rows_after_the_pinned_label(tmp_path, monkeypatch):
    monkeypatch.setattr(B, 'verify', lambda idx, thr: None)
    idx = pd.date_range('2026-11-20 18:00', periods=6, freq='h', tz='UTC')
    df = pd.DataFrame({'time': idx.strftime('%Y-%m-%dT%H:%M:%S+00:00'), 'open': 1.0, 'high': 2.0, 'low': 0.0, 'close': 1.0})
    p = tmp_path / 'g.csv'
    df.to_csv(p, index=False)
    got = B.load_h1(CFG, str(p))
    assert got.index.max() == pd.Timestamp(B.PINNED_LAST)


def test_gold_digest_ignores_line_endings(tmp_path):
    import copy
    from src.gold_bbnr import inputs as I
    text = f'time,open,high,low,close\n{B.PINNED_LAST},1,2,0,1.5\n'
    c = copy.deepcopy(CFG)
    p = tmp_path / 'g.csv'
    c['paths_read']['gold_h1'] = str(p)
    p.write_bytes(text.encode())
    lf = I.cache_digests(c)['gold_h1']
    p.write_bytes(text.replace('\n', '\r\n').encode())
    assert I.cache_digests(c)['gold_h1'] == lf
    p.write_bytes(text.replace('1.5', '1.6').encode())
    assert I.cache_digests(c)['gold_h1'] != lf
