"""The four window rules, eligibility and gates (openspec change gold-window-strategy,
task 3.1). Labels are Berlin wall time; Sofia = Berlin + 1 h."""
import numpy as np
import pandas as pd
import pytest

from src import gold_window as W
from src.gold_window import rules as R

CFG = W.load_config()
LOOK = CFG['rules']['gate_lookback_days']


def _days(n, start='2021-01-04'):
    return pd.bdate_range(start, periods=n, tz='UTC')


def _h1(days, close=lambda day, h: 2000.0, spread=0.25, drop=()):
    rows = []
    for d in days:
        for h in range(24):
            t = d + pd.Timedelta(hours=h)
            if t in drop:
                continue
            rows.append({'time': t, 'close': close(d, h), 'spread_usd': spread})
    return pd.DataFrame(rows)


def _d1_pred(days, p_up=0.6, vol=lambda i: 1.0):
    return pd.DataFrame({'time': pd.DatetimeIndex(days), 'p_up': p_up,
                         'vol_pct': [vol(i) for i in range(len(days))]})


# ── eligibility and gates ──────────────────────────────────────────────────

def test_missing_1800_h1_bar_excludes_the_day():
    days = _days(3)
    h1 = _h1(days, drop={days[1] + pd.Timedelta(hours=18)})
    assert list(R.eligible_days_h1(h1)) == [days[0], days[2]]


def test_weekend_days_are_never_eligible():
    days = pd.date_range('2021-01-01', periods=4, freq='D', tz='UTC')    # Fri .. Mon
    assert [d.dayofweek for d in R.eligible_days_h1(_h1(days))] == [4, 0]


def test_m15_eligibility_needs_every_window_bar():
    day = _days(1)[0]
    t = pd.date_range(day + pd.Timedelta(hours=14, minutes=30), day + pd.Timedelta(hours=21, minutes=45), freq='15min')
    m = pd.DataFrame({'time': t, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'spread_usd': 0.25})
    assert list(R.eligible_days_m15(m)) == [day]
    assert list(R.eligible_days_m15(m.drop(index=5))) == []


def test_trailing_threshold_warm_up_and_strictly_prior():
    v = pd.Series(np.arange(LOOK + 3, dtype=float), index=_days(LOOK + 3))
    thr = R.trailing_threshold(v, 0.5, LOOK)
    assert thr.iloc[:LOOK].isna().all()
    # the first threshold uses values 0..249 only, never the day's own value 250
    assert thr.iloc[LOOK] == pytest.approx(np.median(np.arange(LOOK)))


def test_daily_forecast_takes_the_last_bar_strictly_before_the_day():
    mon = pd.Timestamp('2021-01-11', tz='UTC')
    fri, sun = mon - pd.Timedelta(days=3), mon - pd.Timedelta(days=1)
    pred = pd.DataFrame({'time': [fri, sun, mon], 'p_up': [0.6, 0.4, 0.9], 'vol_pct': [1.0, 2.0, 3.0]})
    assert R.daily_forecast(pred, pd.DatetimeIndex([mon]), 'p_up').iloc[0] == 0.4
    no_sun = pred.drop(index=1)
    assert R.daily_forecast(no_sun, pd.DatetimeIndex([mon]), 'p_up').iloc[0] == 0.6


# ── h1_gated_direction ─────────────────────────────────────────────────────

def _h1_setup(last_vol, last_pup):
    days = _days(LOOK + 1)
    h1 = _h1(days)
    rows = []
    rng = np.random.default_rng(0)
    for i, d in enumerate(days):
        for h in range(14, 21):
            last = i == len(days) - 1
            rows.append({'time': d + pd.Timedelta(hours=h), 'p_up': last_pup if last else 0.6,
                         'vol_pct': last_vol if last else float(rng.uniform())})
    return h1, pd.DataFrame(rows), days


def test_h1_gate_closed_below_the_trailing_80th_percentile():
    h1, pred, days = _h1_setup(last_vol=0.5, last_pup=0.6)
    t = R.h1_gated_direction(h1, pred, days, CFG)
    last = t[t.day == days[-1]]
    assert len(last) == 7 and not last.gate_open.any()
    assert not t[t.day < days[LOOK]].gate_open.any()            # warm-up: no threshold yet


def test_h1_gate_open_at_the_85th_percentile_shorts_when_p_up_is_047():
    h1, pred, days = _h1_setup(last_vol=0.85, last_pup=0.47)
    t = R.h1_gated_direction(h1, pred, days, CFG)
    last = t[t.day == days[-1]]
    assert last.gate_open.all() and (last.side == -1).all()
    first = last.iloc[0]
    assert first.entry_time == days[-1] + pd.Timedelta(hours=15)     # close of 14:00 = 16:00 Sofia
    assert first.exit_time == days[-1] + pd.Timedelta(hours=16)      # one hour later


# ── continuation ───────────────────────────────────────────────────────────

def _gate(days, value=2.0, thr=1.0):
    return pd.DataFrame({'gate_value': value, 'threshold': thr}, index=pd.DatetimeIndex(days))


def test_continuation_goes_long_after_a_rise_into_the_window():
    days = _days(2)
    h1 = _h1(days, close=lambda d, h: 2000.0 + (5.0 if (d == days[1] and h >= 14) else 0.0))
    t = R.continuation(h1, _gate(days), days[1:], CFG)
    assert len(t) == 1 and t.side.iloc[0] == 1 and t.gate_open.iloc[0]
    assert t.entry_time.iloc[0] == days[1] + pd.Timedelta(hours=15)   # 16:00 Sofia
    assert t.exit_time.iloc[0] == days[1] + pd.Timedelta(hours=22)    # 23:00 Sofia


def test_continuation_zero_move_means_no_trade():
    days = _days(2)
    assert R.continuation(_h1(days), _gate(days), days[1:], CFG).empty


def test_continuation_respects_a_closed_gate():
    days = _days(2)
    h1 = _h1(days, close=lambda d, h: 2000.0 - (5.0 if (d == days[1] and h >= 14) else 0.0))
    t = R.continuation(h1, _gate(days, value=0.5, thr=1.0), days[1:], CFG)
    assert t.side.iloc[0] == -1 and not t.gate_open.iloc[0]


# ── us_open_breakout ───────────────────────────────────────────────────────

def _m15(day, path):
    """path(minutes_since_14:30_label) -> close; high = low = close."""
    t = pd.date_range(day + pd.Timedelta(hours=14, minutes=30), day + pd.Timedelta(hours=21, minutes=45), freq='15min')
    c = [path(int((x - t[0]) / pd.Timedelta(minutes=1))) for x in t]
    return pd.DataFrame({'time': t, 'high': c, 'low': c, 'close': c, 'spread_usd': 0.25})


def test_breakout_no_break_means_no_trade():
    day = _days(1)[0]
    t = R.us_open_breakout(_m15(day, lambda m: 2000.0), _gate([day]), pd.DatetimeIndex([day]), CFG)
    assert t.empty


def test_breakout_first_break_down_at_1715_sofia_wins():
    day = _days(1)[0]
    # range bars 14:30-15:15 labels sit in [1999, 2001]; label 16:00 Berlin (closes 17:15 Sofia)
    # breaks down, label 17:00 breaks up later
    def path(m):
        if m < 60:
            return 1999.0 if m % 30 else 2001.0
        if m == 90:
            return 1995.0
        if m >= 150:
            return 2010.0
        return 2000.0
    t = R.us_open_breakout(_m15(day, path), _gate([day]), pd.DatetimeIndex([day]), CFG)
    assert len(t) == 1
    r = t.iloc[0]
    assert r.side == -1 and r.entry_price == 1995.0
    assert r.entry_time == day + pd.Timedelta(hours=16, minutes=15)   # 17:15 Sofia
    assert r.exit_time == day + pd.Timedelta(hours=22)                # 23:00 Sofia


def test_breakout_ignores_breaks_after_2000_sofia():
    day = _days(1)[0]
    # first outside close on label 19:00 Berlin, closing 19:15 Berlin = 20:15 Sofia
    t = R.us_open_breakout(_m15(day, lambda m: 2010.0 if m >= 270 else 2000.0), _gate([day]),
                           pd.DatetimeIndex([day]), CFG)
    assert t.empty


# ── d1_gated_direction ─────────────────────────────────────────────────────

def test_d1_monday_call_comes_from_the_last_bar_before_monday():
    days = _days(LOOK + 6)                                   # business days only
    pred = _d1_pred(days, p_up=0.7)
    mon = [d for d in days[LOOK + 1:] if d.dayofweek == 0][0]
    fri = mon - pd.Timedelta(days=3)
    pred.loc[pred.time == fri, 'p_up'] = 0.3
    t = R.d1_gated_direction(_h1(days), pred, days, CFG)
    assert t[t.day == mon].side.iloc[0] == -1


def test_d1_gate_warm_up_then_median():
    days = _days(LOOK + 20)
    pred = _d1_pred(days, vol=lambda i: float(i % 7))
    t = R.d1_gated_direction(_h1(days), pred, days, CFG)
    assert not t[t.day <= days[LOOK]].gate_open.any()        # day LOOK has only LOOK-1 prior forecasts
    later = t[t.day > days[LOOK]]
    assert later.gate_open.any() and (~later.gate_open).any()


# ── window boundary for every rule ─────────────────────────────────────────

def test_no_rule_holds_past_2300_sofia_or_enters_before_1530():
    days = _days(LOOK + 2)
    h1 = _h1(days, close=lambda d, h: 2000.0 + (h - 14) * (1 if d.day % 2 else -1))
    rows = [{'time': d + pd.Timedelta(hours=h), 'p_up': 0.6, 'vol_pct': 1.0} for d in days for h in range(14, 21)]
    pred_h1 = pd.DataFrame(rows)
    gate = _gate(days)
    m15 = pd.concat([_m15(d, lambda m: 2000.0 + (0 if m < 60 else 5)) for d in days], ignore_index=True)
    frames = [R.h1_gated_direction(h1, pred_h1, days, CFG), R.continuation(h1, gate, days, CFG),
              R.us_open_breakout(m15, gate, days, CFG), R.d1_gated_direction(h1, _d1_pred(days), days, CFG)]
    for t in frames:
        assert not t.empty
        assert ((t.exit_time - t.day) <= pd.Timedelta(hours=22)).all()
        assert ((t.entry_time - t.day) >= pd.Timedelta(hours=14, minutes=30)).all()
        assert (t.entry_time < t.exit_time).all()
