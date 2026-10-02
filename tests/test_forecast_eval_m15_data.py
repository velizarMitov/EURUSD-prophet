"""M15 session data layer (tasks 14.1-14.4).

The expected figures are MEASURED on results/curl/raw/EURUSD_M1.parquet, which is
excluded from git (DATA.md section 7). Every test that needs it skips when it is
absent; the clock and session logic is also tested on synthetic frames, which run
everywhere.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import m15_data as M
from src.forecast_eval.costs import MissingSpreadError

HAVE_M1 = os.path.exists(M.M1_PATH)
needs_m1 = pytest.mark.skipif(not HAVE_M1, reason='M1 parquet is not in git (DATA.md 7)')


# ── 14.3 the clock, from first principles and from the file ────────────────

def test_berlin_predicts_the_observed_open_label_in_all_four_regimes():
    """Only Europe/Berlin reproduces 23:00 normally and 22:00 in the US/EU
    daylight-saving mismatch weeks. UTC and New York do not."""
    cases = {'2026-06-28': 23, '2026-01-04': 23,      # normal summer / winter
             '2026-03-08': 22, '2025-10-26': 22}      # US on DST first / EU off first
    for day, expected in cases.items():
        d = pd.Timestamp(day).date()
        assert M.expected_open_label(d, 'Europe/Berlin') == expected, day
    assert M.dst_aligned(pd.Timestamp('2026-06-28').date())
    assert not M.dst_aligned(pd.Timestamp('2026-03-08').date())
    # the rival hypotheses are constant where Berlin moves, so they are excluded
    assert len({M.expected_open_label(pd.Timestamp(d).date(), 'UTC') for d in cases}) > 1
    assert {M.expected_open_label(pd.Timestamp(d).date(), 'America/New_York')
            for d in cases} == {17}


def _weekly_frame(tz: str, weeks=60, start='2025-09-07'):
    """Bars stamped on `tz`'s wall clock, one per hour of a Sunday-open week."""
    opens = pd.date_range(start, periods=weeks, freq='7D', tz='UTC')
    stamps = []
    for sunday in opens:
        ny = datetime(sunday.year, sunday.month, sunday.day, M.MARKET_OPEN_HOUR,
                      tzinfo=ZoneInfo(M.MARKET_OPEN_TZ))
        label = pd.Timestamp(ny.astimezone(ZoneInfo(tz)).replace(tzinfo=None), tz='UTC')
        stamps.append(pd.date_range(label, periods=24 * 5, freq='1h'))
    return pd.DatetimeIndex(np.concatenate([s.asi8 for s in stamps])).tz_localize('UTC')


def test_clock_is_accepted_for_berlin_and_refused_for_a_shifted_frame():
    idx = _weekly_frame('Europe/Berlin')
    ev = M.verify_clock(idx)
    assert ev['label_tz'] == 'Europe/Berlin' and ev['match_rate'] == 1.0
    assert ev['aligned']['n'] and ev['dst_mismatch']['n'], 'both regimes must be covered'
    with pytest.raises(M.ClockMismatch):
        M.verify_clock(idx + pd.Timedelta(hours=1))


def test_clock_needs_a_weekend_boundary():
    with pytest.raises(M.ClockMismatch):
        M.verify_clock(pd.date_range('2026-01-05', periods=50, freq='1h', tz='UTC'))


@needs_m1
def test_measured_clock_on_the_real_file():
    idx = M.m15_bars(verify=False).index
    ev = M.verify_clock(idx)
    assert ev['n_weeks'] > 400
    assert ev['match_rate'] > 0.95
    assert ev['aligned']['modal_label_hour'] == 23
    assert ev['dst_mismatch']['modal_label_hour'] == 22


# ── 14.4 the session window ────────────────────────────────────────────────

def _grid(day='2026-06-01', days=1):
    """M15 bars across whole days, so session edges are exercised."""
    return pd.date_range(pd.Timestamp(day, tz='UTC'), periods=96 * days, freq='15min')


def test_session_window_is_the_owners_window():
    idx = _grid()
    sess = M.session_mask(idx)
    assert sess.sum() == M.BARS_PER_SESSION == 30
    inside = idx[sess]
    assert str(inside[0].time()) == '14:30:00' and str(inside[-1].time()) == '21:45:00'
    assert str(M.owner_clock(inside)[0].time()) == '15:30:00'
    assert str(M.owner_clock(inside)[-1].time()) == '22:45:00'


def test_session_is_the_same_label_window_in_january_and_july():
    for day in ('2026-01-07', '2026-07-08'):
        inside = _grid(day)[M.session_mask(_grid(day))]
        assert str(inside[0].time()) == '14:30:00'
        assert str(inside[-1].time()) == '21:45:00'


def test_weekend_bars_are_never_in_session():
    sat_sun = _grid('2026-06-06', days=2)          # Saturday + Sunday
    assert not M.session_mask(sat_sun).any()


def test_eligibility_needs_the_target_inside_the_session_too():
    idx = _grid()
    for h, last_as_of in ((1, '21:30:00'), (2, '21:15:00'), (4, '20:45:00')):
        pos = M.eligible_positions(idx, h)
        assert str(idx[pos[0]].time()) == '14:30:00'
        assert str(idx[pos[-1]].time()) == last_as_of, h
        assert len(pos) == M.BARS_PER_SESSION - h
        # the target of every eligible trade closes at or before the session end
        tgt = idx[pos + h]
        assert ((tgt.hour * 60 + tgt.minute + M.BAR_MINUTES) <= M.SESSION_END_MIN).all()
        assert (tgt.normalize() == idx[pos].normalize()).all()


def test_a_trade_may_not_span_two_session_days():
    idx = _grid('2026-06-01', days=2)
    pos = M.eligible_positions(idx, 4)
    assert (idx[pos + 4].normalize() == idx[pos].normalize()).all()


def test_the_longest_horizon_still_fits_inside_one_session():
    idx = _grid()
    assert len(M.eligible_positions(idx, M.BARS_PER_SESSION - 4)) == 4
    assert len(M.eligible_positions(idx, M.BARS_PER_SESSION)) == 0


def test_session_position_runs_from_zero_to_one_and_is_nan_outside():
    idx = _grid()
    sp = M.session_position(idx)
    inside = sp[M.session_mask(idx)]
    assert np.isclose(inside[0], 0.0) and inside[-1] < 1.0 and (np.diff(inside) > 0).all()
    assert np.isnan(sp[~M.session_mask(idx)]).all()


def test_eligible_refuses_a_zero_horizon():
    with pytest.raises(ValueError):
        M.eligible_mask(_grid(), 0)


# ── 14.1 the mid-price aggregation ─────────────────────────────────────────

def _m1_probe(minutes=120, spread_points=5.0):
    idx = pd.date_range('2026-06-01 14:30', periods=minutes, freq='1min', tz='UTC')
    close = 1.1 + np.arange(minutes) * 1e-5
    return pd.DataFrame({'open': close, 'high': close + 2e-5, 'low': close - 2e-5,
                         'close': close, 'tick_volume': 10.0,
                         'spread_points': spread_points, 'point': 1e-5}, index=idx)


def test_aggregation_builds_mid_from_the_bars_own_spread():
    m15 = M.aggregate_m15(_m1_probe())
    assert len(m15) == 8 and list(m15.index.minute) == [30, 45, 0, 15, 30, 45, 0, 15]
    assert np.allclose(m15['mid'] - m15['close'], 0.5 * 5 * 1e-5)
    assert np.allclose(m15['spread_price'], 5e-5)
    assert (m15['tick_volume'] == 150.0).all(), 'tick volume is summed over the minutes'
    assert m15['high'].iloc[0] >= m15['close'].iloc[0] >= m15['low'].iloc[0]


def test_aggregation_is_refused_without_a_spread():
    probe = _m1_probe().drop(columns=['spread_points'])
    with pytest.raises(MissingSpreadError, match='spread'):
        M.aggregate_m15(probe)


def test_a_widening_spread_moves_bid_and_mid_apart():
    """The artifact the parity check exists for, in miniature: a flat market, a
    spread that opens from 5 to 25 points, and a bid that drops only 5 points.
    Half the spread widened by 10 points, so the MID actually rose while the bid
    fell -- a bid-only series records a down move that never happened."""
    probe = _m1_probe(minutes=60)
    probe[['open', 'high', 'low', 'close']] = 1.1                  # flat, no drift
    probe.loc[probe.index[30:], 'spread_points'] = 25.0
    probe.loc[probe.index[30:], ['open', 'high', 'low', 'close']] = 1.1 - 5 * 1e-5
    m15 = M.aggregate_m15(probe)
    bid_move = m15['close'].iloc[-1] - m15['close'].iloc[0]
    mid_move = m15['mid'].iloc[-1] - m15['mid'].iloc[0]
    assert np.isclose(bid_move, -5e-5) and np.isclose(mid_move, +5e-5)
    assert bid_move < 0 < mid_move, 'bid falls while mid rises: a pure quote artifact'


@needs_m1
def test_measured_m15_series_and_session_counts():
    m15 = M.m15_bars()
    assert len(m15) == 199_097
    assert str(m15.index[0]) == '2018-08-08 17:30:00+00:00'
    assert str(m15.index[-1]) == '2026-08-07 22:45:00+00:00'
    s = M.session_summary(m15.index)
    assert s['in_session'] == 62_238
    assert s['session_days'] == 2_075
    assert [len(M.eligible_positions(m15.index, h)) for h in (1, 2, 4)] == [60_163, 58_088, 53_938]
    inside = M.session_mask(m15.index)
    assert abs(float(m15.loc[inside, 'spread_price'].median()) - 5e-5) < 1e-9


@needs_m1
def test_the_pinned_m15_csv_is_not_the_source():
    """results/eurusd_m15.csv has no spread column, so it cannot be used."""
    pinned = pd.read_csv(os.path.join(M.REPO, 'results', 'eurusd_m15.csv'), nrows=5)
    assert 'spread' not in ' '.join(pinned.columns)
    src = open(M.__file__, encoding='utf-8').read()
    assert 'eurusd_m15.csv' not in src.split('"""', 2)[2], 'never read for targets'


# ── 14.2 source provenance ─────────────────────────────────────────────────

@needs_m1
def test_source_fingerprint_matches_the_coverage_report():
    fp = M.source_fingerprint()
    assert fp['rows'] == 2_980_060
    assert len(fp['sha256']) == 64
    assert M.verify_source(fp)['sha256'] == fp['sha256']


def test_verify_source_detects_a_one_row_difference(tmp_path):
    probe = tmp_path / 'probe.parquet'
    _m1_probe(minutes=120).to_parquet(probe)
    fp = M.source_fingerprint(str(probe))
    assert fp['rows'] == 120
    _m1_probe(minutes=119).to_parquet(probe)
    with pytest.raises(M.SourceChanged, match='rows'):
        M.verify_source(fp, str(probe))


def test_missing_source_names_how_to_regenerate(tmp_path):
    with pytest.raises(FileNotFoundError, match='curl_mt5_fetch'):
        M.source_fingerprint(str(tmp_path / 'absent.parquet'))
