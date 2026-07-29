"""
BROKER TIME -> NEW YORK TIME conversion, and the H_dir.1 diagnostics redone on
the true FX trading clock.

WHY
---
The server-time diagnostics' `hour_utc` column is mislabelled. src/live_data.py
tags raw MetaQuotes wall-clock values as UTC, but that tag is a LABEL, not a
conversion -- the numbers are broker SERVER hours. The FX trading day is
anchored to 17:00 New York, where the value date rolls, so calendar-UTC hours
are the wrong frame for reading an intraday effect.

TWO CORRECTIONS TO THE BRIEF, both established from the data (STEP 0) rather
than assumed. They matter, so they are stated up front:

  1. The broker is NOT UTC+2/UTC+3 (EET/EEST, "Athens style") and the
     server-to-NY offset is NOT +7. This feed's broker runs CET/CEST --
     UTC+1 in winter, UTC+2 in summer (Europe/Berlin) -- so the server-to-NY
     offset is +6 normally and +5 during US/EU DST mismatch windows.
     Evidence: converting with Europe/Berlin puts 586 of 596 week openings at
     exactly 17:00 NY and 586 week closings at 16:00 NY (the last bar before
     the roll). Europe/Athens misses by exactly one hour on every week.
     This is consistent with the project's earlier live finding that the
     broker sat at UTC+2 in JULY -- which is CEST, not EET.

  2. Therefore SERVER HOUR 23 IS NOT "the 16:00-17:00 NY hour, the last hour
     before rollover". Server midnight is 18:00 NY, so the 23:00 server bar
     covers 17:00-18:00 NY: it is the ROLLOVER HOUR ITSELF -- fx_day_hour 0,
     the FIRST bar of the new FX day, not the last of the old one. The bar
     before the roll is server hour 22. The rollover hypothesis has to be read
     against that placement, not the one the brief assumed.

RULE (a) vs RULE (b)
--------------------
  (a) server follows EU DST  -> server-to-NY offset shifts by one hour during
      the US/EU mismatch windows (US switches 2nd Sunday March / 1st Sunday
      November; EU switches last Sunday March / last Sunday October).
  (b) server locked to New York -> constant offset, no shift ever.

The week-boundary evidence establishes (a): 34 of 596 week openings sit one
server hour earlier than the other 562, and every one of those 34 falls inside
a US/EU mismatch window. DST transitions themselves always occur while the FX
market is closed (EU at 01:00 UTC Sunday, US at 07:00 UTC Sunday), so the
offset is constant WITHIN any given FX week.

BOUNDARIES
----------
Writes ONLY results/pooled_h1/EURUSD_h1_newyork.csv and
results/h1_direction_diagnostics_newyork.csv. Never modifies models/,
_train_pipeline.py, src/inference.py, src/features.py, src/paper_trading.py,
config.json, results/eurusd_h1.csv, the pre-existing results/pooled_h1/*.csv,
src/h1_direction_model.py, src/pooled_h1_*.py, or any
results/*hypothesis_log.csv. Unit tests sha256 the whole set.

The reserved test block [85:100%] is NEVER indexed (the guard from
src/h1_direction_diagnostics.py is reused unchanged).

NO alpha, NO hypothesis log, NO verdict -- descriptive only, on the SAME
already-spent validation slice, using the SAME committed H_dir.1 model. Pips
appear ONLY as a descriptive measure of move size; there is no P&L, no spread,
no cost and no trading simulation anywhere in this module.
"""

import os

import numpy as np
import pandas as pd

from src import h1_direction_model as h1d
from src import h1_direction_diagnostics as diag
from src.pooled_h1_data import POOLED_DIR, OHLC_COLUMNS

# ── Broker zone: ESTABLISHED EMPIRICALLY in STEP 0, not assumed ──
BROKER_TZ = 'Europe/Berlin'          # CET/CEST = UTC+1 winter, UTC+2 summer
NY_TZ = 'America/New_York'
NORMAL_SERVER_TO_NY_OFFSET_H = 6     # server hour - NY hour, outside mismatch
FX_ROLL_HOUR_NY = 17                 # the FX day opens at 17:00 New York
PIP = 0.0001                         # EURUSD pip, for DESCRIPTIVE move size only

NY_HISTORY_CSV = os.path.join(POOLED_DIR, 'EURUSD_h1_newyork.csv')
NY_DIAGNOSTICS_CSV = 'results/h1_direction_diagnostics_newyork.csv'

# Pre-registered NY session buckets. Half-open [start, end) on the NY clock, so
# the five buckets are disjoint and cover all 24 hours exactly once
# (8 + 5 + 3 + 5 + 3 = 24). The brief's "Asia 19:00-02:00" is read as inclusive
# of hour 02, which is what makes the cover exhaustive.
NY_SESSIONS = {
    'Asia':     (19, 3),    # 19,20,21,22,23,0,1,2   -- wraps midnight
    'London':   (3, 8),     # 3,4,5,6,7
    'overlap':  (8, 11),    # 8,9,10   London/NY simultaneous, deepest liquidity
    'NY-only':  (11, 16),   # 11,12,13,14,15
    'rollover': (16, 19),   # 16,17,18 -- the pre- and post-roll window
}
SESSION_ORDER = ('Asia', 'London', 'overlap', 'NY-only', 'rollover')


class BrokerRuleAmbiguousError(RuntimeError):
    """Raised when the week-boundary evidence does not clearly establish which
    DST rule the broker follows. Every number downstream depends on the rule,
    so an ambiguous read STOPS the program instead of guessing."""


class ConversionVerificationError(RuntimeError):
    """Raised when a STEP 2 hard gate fails -- the conversion must relabel time
    only, never touch, reorder, resample or drop a price."""


# ───────────────────── STEP 0: establish the DST rule ─────────────────────────

def load_server_frame(out_dir: str = POOLED_DIR) -> pd.DataFrame:
    """
    Read the cached EURUSD H1 frame and strip the FAKE UTC tag, returning a
    tz-NAIVE index of raw broker wall-clock values -- which is what those
    numbers actually are. The source file is never modified.
    """
    path = os.path.join(out_dir, 'EURUSD_h1.csv')
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    out = df[list(OHLC_COLUMNS)].copy()
    out.index = idx
    out.index.name = 'server_time'
    return out


def week_boundaries(index, gap_hours: int = 12):
    """
    Positions of the first and last bar of every FX week, found from the weekend
    gap in the series itself (any break longer than `gap_hours`) rather than
    from a calendar assumption. Returns (start_positions, end_positions).
    """
    idx = pd.DatetimeIndex(index)
    is_open = np.zeros(len(idx), dtype=bool)
    if len(idx):
        is_open[0] = True
        is_open[1:] = np.diff(idx.to_numpy()) > np.timedelta64(gap_hours, 'h')
    starts = np.flatnonzero(is_open)
    ends = np.append(starts[1:] - 1, len(idx) - 1) if len(starts) else starts
    return starts, ends


def determine_dst_rule(server_df: pd.DataFrame, verbose: bool = True):
    """
    STEP 0 -- decide between rule (a) EU-DST server and rule (b) NY-locked
    server, from a market fact rather than an assumption: the FX week opens
    Sunday 17:00 NY and closes Friday 17:00 NY.

    Under (b) the week-opening SERVER hour is identical all year. Under (a) it
    shifts by exactly one hour during the US/EU DST mismatch windows.

    Raises BrokerRuleAmbiguousError if the modal week-opening hour does not
    dominate, or if the shifted weeks are not confined to mismatch windows.
    """
    idx = pd.DatetimeIndex(server_df.index)
    starts, ends = week_boundaries(idx)
    opens, closes = idx[starts], idx[ends]

    open_hours = pd.Series(opens.hour).value_counts().sort_index()
    close_hours = pd.Series(closes.hour).value_counts().sort_index()
    n_weeks = len(opens)

    # Split the weeks by whether the US and the EU DISAGREE about DST on that
    # date, read straight from the tz database. This is a property of the two
    # DST calendars, not of the broker, so using it to partition the weeks does
    # not assume the answer -- the question is whether the broker's opening hour
    # actually shifts on exactly those weeks.
    in_window = np.array([eu_us_dst_mismatch(t) for t in opens])

    # A week's opening bar is USABLE as a DST anchor only if it is a genuine
    # Sunday-evening open at one of the two plausible hours. Holiday-shortened
    # weeks (Christmas, New Year) open on a Monday or later, for reasons that
    # have nothing to do with DST -- including one Boxing-Day week that happens
    # to open at a plausible-looking hour. They are excluded from the inference
    # and reported explicitly rather than silently absorbed.
    usable = (np.isin(opens.hour, [FX_ROLL_HOUR_NY + NORMAL_SERVER_TO_NY_OFFSET_H,
                                   FX_ROLL_HOUR_NY + NORMAL_SERVER_TO_NY_OFFSET_H - 1])
              & (opens.dayofweek == 6))
    holiday_like = [t.isoformat() for t in opens[~usable]]

    normal = opens[usable & ~in_window]
    window = opens[usable & in_window]
    if not len(window) or not len(normal):
        raise BrokerRuleAmbiguousError(
            'one DST class has no usable week anchors -- the rule cannot be established.')

    # Non-mismatch weeks must be unanimous; they are the calibration baseline.
    baseline_hour = FX_ROLL_HOUR_NY + NORMAL_SERVER_TO_NY_OFFSET_H
    if not (normal.hour == baseline_hour).all():
        raise BrokerRuleAmbiguousError(
            f"weeks with US/EU DST agreeing do not all open at server hour "
            f"{baseline_hour}: {pd.Series(normal.hour).value_counts().to_dict()}"
        )

    shifted = window.hour == baseline_hour - 1     # broker followed the EU switch
    n_shift, n_noshift = int(shifted.sum()), int((~shifted).sum())

    if n_shift and n_noshift:
        # The broker's behaviour CHANGED. Only a clean chronological split is
        # interpretable -- interleaved behaviour would mean neither rule holds.
        first_shift = window[shifted].min()
        last_noshift = window[~shifted].max()
        if last_noshift > first_shift:
            raise BrokerRuleAmbiguousError(
                "mismatch weeks where the broker shifted and did not shift are "
                "INTERLEAVED in time -- neither rule (a) nor rule (b) holds, and no "
                "single era boundary explains it. STOP and report."
            )
        rule, era_boundary = 'a-after-b', first_shift
    elif n_shift:
        rule, era_boundary = 'a', window.min()
    else:
        rule, era_boundary = 'b', None

    if verbose:
        print('=' * 78)
        print('STEP 0 — BROKER DST RULE, FROM THE WEEK BOUNDARIES')
        print('=' * 78)
        print(f"  weeks detected                    : {n_weeks}")
        print(f"  week-OPEN server hours (all)      : {open_hours.to_dict()}")
        print(f"  week-CLOSE server hours (all)     : {close_hours.to_dict()}")
        print(f"  holiday-shortened weeks excluded  : {len(holiday_like)}")
        print(f"  weeks with US/EU DST AGREEING     : {len(normal)}  "
              f"-> ALL open at server hour {baseline_hour}")
        print(f"  weeks in a US/EU MISMATCH window  : {len(window)}  -> "
              f"{n_shift} shifted to {baseline_hour - 1}, {n_noshift} did NOT shift")
        if rule == 'a-after-b':
            print(f"  broker behaviour CHANGED, cleanly  : rule (b) through "
                  f"{last_noshift.date()}, rule (a) from {first_shift.date()}")
        print(f"  => RULE ({rule}): " + {
            'a': 'server follows EU DST throughout',
            'b': 'server locked to New York throughout; constant offset',
            'a-after-b': 'server followed US DST (constant NY offset) in the early '
                         'era, then switched to EU DST',
        }[rule])

    return {
        'rule': rule, 'n_weeks': n_weeks,
        'week_open_server_hours': open_hours.to_dict(),
        'week_close_server_hours': close_hours.to_dict(),
        'baseline_open_hour': int(baseline_hour),
        'n_weeks_normal': int(len(normal)),
        'n_weeks_mismatch': int(len(window)),
        'n_mismatch_shifted': n_shift, 'n_mismatch_not_shifted': n_noshift,
        'eu_rule_era_start': era_boundary.isoformat() if era_boundary is not None else None,
        'shifted_week_dates': [t.isoformat() for t in window[shifted]],
        'unshifted_mismatch_week_dates': [t.isoformat() for t in window[~shifted]],
        'holiday_like_weeks': holiday_like,
        'broker_tz': BROKER_TZ,
    }


def eu_us_dst_mismatch(ts) -> bool:
    """
    True when the EU and the US disagree about daylight saving on this date --
    a property of the two DST calendars alone, read from the tz database, with
    no assumption about the broker.
    """
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert('UTC').tz_localize(None)
    eu = t.tz_localize(BROKER_TZ)
    eu_dst = eu.dst() != pd.Timedelta(0)
    us_dst = eu.tz_convert(NY_TZ).dst() != pd.Timedelta(0)
    return bool(eu_dst != us_dst)


def week_offsets(server_index, era_start):
    """
    Server-to-NY offset in hours for every bar, assigned PER FX WEEK.

    Offsets can only change at a DST transition, and every EU/US transition
    happens while the FX market is closed (EU 01:00 UTC Sunday, US 07:00 UTC
    Sunday), so the offset is constant within a week by construction.

    The rule, established empirically in STEP 0:
      * a week in a US/EU mismatch window, at or after `era_start`, runs at
        offset 5 -- the broker followed the EU switch and New York did not;
      * every other week runs at the normal offset 6.
    `era_start` of None means the broker never followed the EU switch.
    """
    idx = pd.DatetimeIndex(server_index)
    starts, _ends = week_boundaries(idx)
    week_id = np.zeros(len(idx), dtype=np.int64)
    week_id[starts] = 1
    week_id = np.cumsum(week_id) - 1

    opens = idx[starts]
    boundary = pd.Timestamp(era_start) if era_start is not None else None
    per_week = np.full(len(opens), NORMAL_SERVER_TO_NY_OFFSET_H, dtype=np.int64)
    for i, t in enumerate(opens):
        if eu_us_dst_mismatch(t) and boundary is not None and t >= boundary:
            per_week[i] = NORMAL_SERVER_TO_NY_OFFSET_H - 1
    return per_week[week_id], per_week, opens


def offset_hours_server_to_ny(server_ts) -> int:
    """
    Server hour minus New York hour for one raw broker timestamp: +6 when the US
    and EU agree about DST, +5 inside a mismatch window. Read from the tz
    database, never hard-coded.
    """
    ts = pd.Timestamp(server_ts)
    if ts.tz is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    local = ts.tz_localize(BROKER_TZ)
    return int(round((local.utcoffset() - local.tz_convert(NY_TZ).utcoffset())
                     .total_seconds() / 3600.0))


# ───────────────────── STEP 1: build the New York history ─────────────────────

def to_new_york(server_df: pd.DataFrame, era_start=None) -> pd.DataFrame:
    """
    Relabel the raw broker frame onto the New York clock and the FX trading day.
    PRICES ARE NEVER TOUCHED -- open/high/low/close pass through unchanged, in
    the original row order.

    Columns added:
      ny_timestamp      bar open time in America/New_York (tz-aware, DST-correct)
      server_timestamp  the original raw broker timestamp, preserved for audit
      ny_hour           0-23 New York local hour
      fx_day_hour       hours since the FX day opened at 17:00 NY. 0 = the
                        17:00-18:00 NY bar (the ROLLOVER hour), 23 = the
                        16:00-17:00 NY bar (the last hour before the roll)
      fx_week_day       0 = the session opening Sunday 17:00 NY, 1..4 = the
                        following sessions. An FX day spans two calendar dates
      is_dst_mismatch   True where the US/EU DST disagreement makes the
                        server-to-NY offset +5 instead of the usual +6

    `ambiguous`/`nonexistent` are set to 'raise': a DST fold or gap landing on a
    real bar would be a silent corruption, and it must stop the run instead.
    Transitions occur while the market is closed, so this never fires in
    practice -- which is itself worth having asserted.
    """
    naive = pd.DatetimeIndex(server_df.index)
    if naive.tz is not None:
        naive = naive.tz_convert('UTC').tz_localize(None)

    offsets, _per_week, _opens = week_offsets(naive, era_start)
    ny_naive = naive - pd.to_timedelta(offsets, unit='h')
    # 'raise' on both: a bar landing in a New York DST gap or fold would be a
    # silent corruption. Transitions happen while the market is closed, so this
    # never fires -- which is worth having asserted rather than assumed.
    ny = pd.DatetimeIndex(ny_naive).tz_localize(NY_TZ, ambiguous='raise',
                                                nonexistent='raise')

    out = server_df.copy()
    out.index = ny
    out.index.name = 'ny_timestamp'
    out['server_timestamp'] = [t.isoformat() for t in naive]
    out['ny_hour'] = ny.hour

    # The FX clock: shifting the NY time forward past the 17:00 roll turns the
    # trading day into an ordinary calendar day, which makes both the
    # hours-since-roll and the session index fall out directly.
    shifted = pd.DatetimeIndex(ny_naive) + pd.Timedelta(hours=24 - FX_ROLL_HOUR_NY)
    out['fx_day_hour'] = (ny.hour - FX_ROLL_HOUR_NY) % 24
    out['fx_week_day'] = shifted.weekday

    out['is_dst_mismatch'] = offsets != NORMAL_SERVER_TO_NY_OFFSET_H
    out['server_to_ny_offset_h'] = offsets
    # The broker's implied offset from true UTC. A sanity anchor: it must always
    # be +1 (CET) or +2 (CEST) -- the conversion is not free to invent a zone.
    # server = UTC + broker_offset, and ny = server - offsets, so
    # broker_offset = offsets - (utc - ny).
    utc_minus_ny_h = ((ny.tz_convert('UTC').tz_localize(None)
                       - pd.DatetimeIndex(ny_naive)).total_seconds() / 3600.0)
    out['broker_utc_offset_h'] = (offsets - utc_minus_ny_h).round().astype(int)

    cols = list(OHLC_COLUMNS) + ['server_timestamp', 'ny_hour', 'fx_day_hour',
                                 'fx_week_day', 'is_dst_mismatch',
                                 'server_to_ny_offset_h', 'broker_utc_offset_h']
    return out[cols]


# ───────────────────── STEP 2: conversion verification gates ──────────────────

def verify_conversion(server_df: pd.DataFrame, ny_df: pd.DataFrame,
                      verbose: bool = True):
    """
    The six STEP 2 hard gates. Every one raises ConversionVerificationError on
    failure -- the conversion must relabel time and nothing else.
    """
    report = {}

    # 1. row count + OHLC values identical, in the same order
    if len(server_df) != len(ny_df):
        raise ConversionVerificationError(
            f"row count changed: {len(server_df)} -> {len(ny_df)}")
    for col in OHLC_COLUMNS:
        if not np.array_equal(server_df[col].to_numpy(), ny_df[col].to_numpy()):
            raise ConversionVerificationError(
                f"price column {col!r} changed -- the conversion must relabel time only")
    report['gate_1_prices_identical'] = True

    # 2. monotonic increasing, unique, even where the local clock repeats
    ny_idx = pd.DatetimeIndex(ny_df.index)
    if not ny_idx.is_monotonic_increasing:
        raise ConversionVerificationError("ny_timestamp is not monotonic increasing")
    if ny_idx.has_duplicates:
        raise ConversionVerificationError("ny_timestamp contains duplicates")
    local_wall = ny_idx.tz_localize(None)
    report['gate_2_monotonic_unique'] = True
    report['repeated_local_wall_clock_values'] = int(
        len(local_wall) - len(local_wall.unique()))

    # 3. round-trip back to the broker's clock reproduces server_timestamp
    #    exactly, going through REAL UTC rather than back down the same
    #    subtraction (which would be trivially true).
    original = pd.DatetimeIndex(pd.to_datetime(ny_df['server_timestamp']))
    utc = ny_idx.tz_convert('UTC').tz_localize(None)
    back = utc + pd.to_timedelta(ny_df['broker_utc_offset_h'].to_numpy(), unit='h')
    if not np.array_equal(pd.DatetimeIndex(back).to_numpy(), original.to_numpy()):
        n_bad = int((pd.DatetimeIndex(back) != original).sum())
        raise ConversionVerificationError(
            f"round-trip failed on {n_bad} rows -- NY -> UTC -> broker does not "
            "reproduce the original server timestamp")
    # the implied broker zone must be CET/CEST, not something invented
    bad_offsets = set(np.unique(ny_df['broker_utc_offset_h'])) - {1, 2}
    if bad_offsets:
        raise ConversionVerificationError(
            f"implied broker UTC offsets {sorted(bad_offsets)} are outside "
            "{+1, +2} (CET/CEST) -- the conversion invented a zone")
    report['gate_3_roundtrip_exact'] = True
    report['broker_utc_offset_counts'] = (
        pd.Series(ny_df['broker_utc_offset_h']).value_counts().sort_index().to_dict())

    # 4. fx_day_hour == 0  <=>  ny_hour == 17
    roll = ny_df['fx_day_hour'] == 0
    if not (ny_df.loc[roll, 'ny_hour'] == FX_ROLL_HOUR_NY).all():
        raise ConversionVerificationError("fx_day_hour==0 bars are not all at ny_hour 17")
    if not (ny_df.loc[ny_df['ny_hour'] == FX_ROLL_HOUR_NY, 'fx_day_hour'] == 0).all():
        raise ConversionVerificationError("ny_hour==17 bars are not all fx_day_hour 0")
    report['gate_4_roll_anchor'] = True

    # 5. the first bar of each FX week opens at 17:00 NY (or the first traded
    #    hour after it on a holiday-shortened week -- reported, never absorbed)
    starts, _ends = week_boundaries(pd.DatetimeIndex(ny_df['server_timestamp'].values
                                                     .astype('datetime64[ns]')))
    opens = ny_df.iloc[starts]
    at_17 = int((opens['ny_hour'] == FX_ROLL_HOUR_NY).sum())
    late = opens[opens['ny_hour'] != FX_ROLL_HOUR_NY]
    if at_17 < 0.9 * len(opens):
        raise ConversionVerificationError(
            f"only {at_17}/{len(opens)} week openings land on 17:00 NY")
    report['gate_5_week_anchor'] = True
    report['n_weeks'] = int(len(opens))
    report['n_weeks_opening_at_1700_ny'] = at_17
    report['late_week_openings'] = [
        {'ny': str(t), 'ny_hour': int(r.ny_hour), 'fx_day_hour': int(r.fx_day_hour),
         'fx_week_day': int(r.fx_week_day)}
        for t, r in late.iterrows()]
    # week CLOSE should be the 16:00-17:00 NY bar, i.e. fx_day_hour 23
    _s, ends = week_boundaries(pd.DatetimeIndex(ny_df['server_timestamp'].values
                                                .astype('datetime64[ns]')))
    closes = ny_df.iloc[ends]
    report['n_weeks_closing_at_1600_ny'] = int((closes['ny_hour'] == 16).sum())

    # 6. three hand-checked spot dates, printed in full
    report['spot_checks'] = spot_check_rows(ny_df)

    if verbose:
        print('\n' + '=' * 78)
        print('STEP 2 — CONVERSION VERIFICATION')
        print('=' * 78)
        print(f"  1. prices + row count identical      : PASS ({len(ny_df)} rows)")
        print(f"  2. ny_timestamp monotonic + unique   : PASS "
              f"({report['repeated_local_wall_clock_values']} repeated local wall-clock values)")
        print(f"  3. round-trip to broker zone exact   : PASS (all {len(ny_df)} rows)")
        print(f"  4. fx_day_hour 0 <=> ny_hour 17      : PASS")
        print(f"  5. week opens at 17:00 NY            : "
              f"{at_17}/{report['n_weeks']}   closes at 16:00 NY: "
              f"{report['n_weeks_closing_at_1600_ny']}/{report['n_weeks']}")
        for l in report['late_week_openings']:
            print(f"       later opening (holiday week): {l['ny']}  ny_hour={l['ny_hour']}")
        print('  6. spot checks:')
        for s in report['spot_checks']:
            print(f"       {s['label']:<22} server {s['server']}  ->  NY {s['ny']}"
                  f"   offset +{s['offset_h']}h  ny_hour={s['ny_hour']}"
                  f"  fx_day_hour={s['fx_day_hour']}  mismatch={s['is_dst_mismatch']}")
    return report


def spot_check_rows(ny_df: pd.DataFrame):
    """Three hand-checkable bars: mid-January (both zones standard), mid-July
    (both on DST), and one inside a March US/EU mismatch window."""
    picks = [('mid-January (std/std)', '2024-01-17 12:00'),
             ('mid-July (DST/DST)', '2024-07-17 12:00'),
             ('March mismatch window', '2024-03-13 12:00')]
    server_ts = pd.DatetimeIndex(pd.to_datetime(ny_df['server_timestamp']))
    out = []
    for label, wanted in picks:
        target = pd.Timestamp(wanted)
        pos = int(np.argmin(np.abs(server_ts - target)))
        row = ny_df.iloc[pos]
        out.append({
            'label': label,
            'server': str(server_ts[pos]),
            'ny': str(ny_df.index[pos]),
            'ny_hour': int(row['ny_hour']),
            'fx_day_hour': int(row['fx_day_hour']),
            'fx_week_day': int(row['fx_week_day']),
            'offset_h': int(row['server_to_ny_offset_h']),
            'is_dst_mismatch': bool(row['is_dst_mismatch']),
            'close': float(row['close']),
        })
    return out


def build_newyork_history(out_dir: str = POOLED_DIR, write: bool = True,
                          out_path: str = NY_HISTORY_CSV, verbose: bool = True):
    """STEP 0 + STEP 1 + STEP 2 end to end. Writes the NEW file only; the raw
    server-time CSVs are never touched."""
    server_df = load_server_frame(out_dir=out_dir)
    rule = determine_dst_rule(server_df, verbose=verbose)
    ny_df = to_new_york(server_df, era_start=rule['eu_rule_era_start'])
    report = verify_conversion(server_df, ny_df, verbose=verbose)

    n_mismatch = int(ny_df['is_dst_mismatch'].sum())
    report['n_bars_in_mismatch_windows'] = n_mismatch
    report['pct_bars_in_mismatch_windows'] = 100.0 * n_mismatch / len(ny_df)
    if verbose:
        print(f"\n  bars inside US/EU DST mismatch windows : {n_mismatch} "
              f"({report['pct_bars_in_mismatch_windows']:.2f}% of {len(ny_df)})")

    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        ny_df.to_csv(out_path)
    return ny_df, rule, report


# ───────────────────── STEP 3: diagnostics on the NY / FX clock ───────────────

def _pip_moves(raw_close: pd.Series) -> pd.Series:
    """Next-bar move in pips. DESCRIPTIVE move size only -- never a P&L, never
    net of spread or any cost."""
    return (raw_close.shift(-1) - raw_close) / PIP


def ny_session(hour: int) -> str:
    """Pre-registered NY session bucket, half-open [start, end) on the NY clock."""
    h = int(hour)
    for name, (start, end) in NY_SESSIONS.items():
        if start <= end:
            if start <= h < end:
                return name
        elif h >= start or h < end:        # wraps midnight
            return name
    raise ValueError(f'hour {hour} matched no session bucket')


def _group_rows(labels_train, labels_val, key_train, key_val, y_train, y_val,
                pips_train, pips_val, pred_gbm, pred_major, scope):
    """One descriptive row per group value: counts, up-rates and mean pip move
    in train and validation separately, then GBM vs majority on validation."""
    total_edge = float(diag.edge_vector(pred_gbm, pred_major, y_val).sum())
    rows = []
    for value in labels_val:
        mt = np.asarray(key_train) == value
        mv = np.asarray(key_val) == value
        st = diag._block_stats(mv, y_val, pred_gbm, pred_major)
        rows.append({
            'scope': scope, 'group': value,
            'n_train': int(mt.sum()), 'n_val': int(mv.sum()),
            'up_rate_train': float(np.asarray(y_train)[mt].mean()) if mt.any() else np.nan,
            'up_rate_val': float(np.asarray(y_val)[mv].mean()) if mv.any() else np.nan,
            'mean_pips_train': float(np.asarray(pips_train)[mt].mean()) if mt.any() else np.nan,
            'mean_pips_val': float(np.asarray(pips_val)[mv].mean()) if mv.any() else np.nan,
            'acc_gbm': st['acc_gbm'], 'acc_majority': st['acc_majority'],
            'delta': st['delta'], 'se_acc': st['se_acc'],
            'edge_share': (st['edge_sum'] / total_edge) if total_edge else np.nan,
        })
    return rows


def hour_majority_predictor(key_train, y_train, key_val):
    """
    The trivial hour-of-day benchmark: learn the majority direction of each
    clock bucket on the TRAIN slice only, then apply it to validation by
    bucket. Buckets unseen in train fall back to the global train majority.
    """
    key_train = np.asarray(key_train)
    key_val = np.asarray(key_val)
    y_train = np.asarray(y_train).astype(int)
    global_majority = int((y_train == 1).sum() >= (y_train == 0).sum())
    table = {}
    for value in np.unique(key_train):
        yk = y_train[key_train == value]
        table[value] = int((yk == 1).sum() >= (yk == 0).sum())
    return np.array([table.get(v, global_majority) for v in key_val], dtype=int), table


def run_newyork_diagnostics(out_dir: str = POOLED_DIR, write: bool = True,
                            out_path: str = NY_DIAGNOSTICS_CSV,
                            ny_df: pd.DataFrame = None, verbose: bool = True):
    """
    STEP 3 -- the H_dir.1 decomposition redone on the New York clock and the FX
    trading day. Same already-spent validation slice, same committed model, same
    reproduction gate. No alpha, no hypothesis log, no verdict.
    """
    rebuilt = diag.rebuild_h_dir_1(out_dir=out_dir, verbose=verbose)
    diag.assert_no_test_block(rebuilt['train_idx'], rebuilt['val_end_ts'])
    diag.assert_no_test_block(rebuilt['val_idx'], rebuilt['val_end_ts'])

    if ny_df is None:
        ny_df, _rule, _rep = build_newyork_history(out_dir=out_dir, write=False,
                                                   verbose=False)
    # Map every model row (keyed by raw server timestamp) onto its NY labels.
    server_key = pd.DatetimeIndex(pd.to_datetime(ny_df['server_timestamp']))
    clock = pd.DataFrame({
        'ny_hour': ny_df['ny_hour'].to_numpy(),
        'fx_day_hour': ny_df['fx_day_hour'].to_numpy(),
        'fx_week_day': ny_df['fx_week_day'].to_numpy(),
        'is_dst_mismatch': ny_df['is_dst_mismatch'].to_numpy(),
        'ny_dow': pd.DatetimeIndex(ny_df.index).dayofweek,
    }, index=server_key.tz_localize('UTC'))

    raw = h1d.load_eurusd_h1(out_dir=out_dir)
    pips_all = _pip_moves(raw['close'])

    tr, va = rebuilt['train_idx'], rebuilt['val_idx']
    ct, cv = clock.loc[tr], clock.loc[va]
    y_tr = rebuilt['context'].loc[tr, 'label'].to_numpy().astype(int)
    y_va = rebuilt['yval']
    pips_tr = pips_all.loc[tr].to_numpy()
    pips_va = pips_all.loc[va].to_numpy()
    pg, pm = rebuilt['pred_gbm'], rebuilt['pred_major']

    rows = []
    rows += _group_rows(range(24), range(24), ct['ny_hour'], cv['ny_hour'],
                        y_tr, y_va, pips_tr, pips_va, pg, pm, 'ny_hour')
    rows += _group_rows(range(24), range(24), ct['fx_day_hour'], cv['fx_day_hour'],
                        y_tr, y_va, pips_tr, pips_va, pg, pm, 'fx_day_hour')
    week_days = sorted(set(cv['fx_week_day']) | set(ct['fx_week_day']))
    rows += _group_rows(week_days, week_days, ct['fx_week_day'], cv['fx_week_day'],
                        y_tr, y_va, pips_tr, pips_va, pg, pm, 'fx_week_day')

    sess_tr = np.array([ny_session(h) for h in ct['ny_hour']])
    sess_va = np.array([ny_session(h) for h in cv['ny_hour']])
    rows += _group_rows(SESSION_ORDER, SESSION_ORDER, sess_tr, sess_va,
                        y_tr, y_va, pips_tr, pips_va, pg, pm, 'ny_session')

    # The Sunday-EVENING open hours specifically. fx_week_day 0 is a FULL
    # session (Sun 17:00 -> Mon 17:00), so the brief's "thin and structurally
    # different" applies to the Sunday-evening HOURS inside it, not to the whole
    # session. Both are reported so the distinction is not blurred.
    sun_tr = (ct['ny_dow'] == 6).to_numpy()
    sun_va = (cv['ny_dow'] == 6).to_numpy()
    rows += _group_rows([True], [True], sun_tr, sun_va, y_tr, y_va,
                        pips_tr, pips_va, pg, pm, 'sunday_evening_hours_only')

    # The trivial hour-of-day-only benchmarks (train-fit, validation-scored).
    trivial = {}
    for name, kt, kv in (('fx_day_hour', ct['fx_day_hour'], cv['fx_day_hour']),
                         ('ny_hour', ct['ny_hour'], cv['ny_hour']),
                         ('server_hour', pd.DatetimeIndex(tr).hour,
                          pd.DatetimeIndex(va).hour)):
        pred, table = hour_majority_predictor(kt, y_tr, kv)
        trivial[name] = {'acc': float((pred == y_va).mean()), 'table': table}
        rows.append({'scope': 'trivial_predictor', 'group': name,
                     'n_train': len(y_tr), 'n_val': len(y_va),
                     'acc_gbm': trivial[name]['acc'],
                     'acc_majority': rebuilt['acc_majority'],
                     'delta': trivial[name]['acc'] - rebuilt['acc_majority'],
                     'se_acc': diag.accuracy_se(len(y_va))})

    rows.append({'scope': 'slice_total', 'group': 'validation[70:85]',
                 'n_train': len(y_tr), 'n_val': len(y_va),
                 'up_rate_train': float(y_tr.mean()), 'up_rate_val': float(y_va.mean()),
                 'mean_pips_train': float(np.nanmean(pips_tr)),
                 'mean_pips_val': float(np.nanmean(pips_va)),
                 'acc_gbm': rebuilt['acc_gbm'], 'acc_majority': rebuilt['acc_majority'],
                 'delta': rebuilt['delta_acc'], 'se_acc': diag.accuracy_se(len(y_va)),
                 'edge_share': 1.0})

    cols = ['scope', 'group', 'n_train', 'n_val', 'up_rate_train', 'up_rate_val',
            'mean_pips_train', 'mean_pips_val', 'acc_gbm', 'acc_majority',
            'delta', 'se_acc', 'edge_share']
    table = pd.DataFrame(rows).reindex(columns=cols)
    if write:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        table.to_csv(out_path, index=False)

    return {'table': table, 'rebuilt': rebuilt, 'trivial': trivial,
            'clock_val': cv, 'n_val': len(y_va)}


def run(out_dir: str = POOLED_DIR, write: bool = True, verbose: bool = True,
        history_path: str = NY_HISTORY_CSV, diagnostics_path: str = NY_DIAGNOSTICS_CSV):
    """Whole program: STEP 0 -> STEP 1 -> STEP 2 -> STEP 3."""
    ny_df, rule, report = build_newyork_history(
        out_dir=out_dir, write=write, out_path=history_path, verbose=verbose)
    diagnostics = run_newyork_diagnostics(
        out_dir=out_dir, write=write, out_path=diagnostics_path,
        ny_df=ny_df, verbose=verbose)
    return {'ny_df': ny_df, 'rule': rule, 'verification': report,
            'diagnostics': diagnostics}


def _print_report(r):
    """STEP 3 tables, in the brief's reporting order."""
    t = r['diagnostics']['table']
    reb = r['diagnostics']['rebuilt']

    def _show(scope, title, keyfmt=str):
        print(f"\n{title}")
        print(f"   {'group':<14}{'n_val':>7}{'up_tr':>8}{'up_val':>8}{'pips_tr':>9}"
              f"{'pips_val':>10}{'acc':>9}{'base':>8}{'delta':>10}{'SE':>8}{'edge%':>8}")
        for _, row in t[t['scope'] == scope].iterrows():
            print(f"   {keyfmt(row['group']):<14}{int(row['n_val']):>7}"
                  f"{row['up_rate_train']:>8.4f}{row['up_rate_val']:>8.4f}"
                  f"{row['mean_pips_train']:>+9.3f}{row['mean_pips_val']:>+10.3f}"
                  f"{row['acc_gbm']:>9.4f}{row['acc_majority']:>8.4f}"
                  f"{row['delta']:>+10.4f}{row['se_acc'] * 100:>7.2f}pp"
                  f"{row['edge_share'] * 100:>7.1f}%")

    print('\n' + '=' * 78)
    print('STEP 3 — H_dir.1 DIAGNOSTICS ON THE NEW YORK / FX CLOCK')
    print('descriptive only — no alpha, no hypothesis log, no verdict')
    print('=' * 78)
    print(f"   reproduced H_dir.1 accuracy {reb['acc_gbm']:.6f} vs majority "
          f"{reb['acc_majority']:.6f}  (n_val={r['diagnostics']['n_val']})")

    _show('ny_hour', '(a) PER NEW YORK HOUR', lambda v: f"{int(v):02d}:00 NY")
    _show('fx_day_hour', '(b) PER FX_DAY_HOUR (hours since the 17:00 NY roll)',
          lambda v: f"+{int(v):02d}h")
    _show('fx_week_day', '(c) PER FX WEEK DAY (0 = session opening Sunday 17:00 NY)',
          lambda v: f"day {int(v)}")
    _show('sunday_evening_hours_only', '    SUNDAY-EVENING HOURS ONLY (subset of day 0)',
          lambda v: 'Sun evening')
    _show('ny_session', '(d) NY SESSION BUCKETS')

    print('\n   TRIVIAL HOUR-OF-DAY-ONLY BENCHMARKS (train-fit, validation-scored)')
    for _, row in t[t['scope'] == 'trivial_predictor'].iterrows():
        print(f"     {row['group']:<14} acc = {row['acc_gbm']:.4f}   "
              f"vs majority {row['acc_majority']:.4f}  "
              f"(delta {row['delta']:+.4f})")
    print(f"     {'GBM (H_dir.1)':<14} acc = {reb['acc_gbm']:.4f}")


if __name__ == '__main__':
    _print_report(run())
