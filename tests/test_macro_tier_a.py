"""
Unit tests for the TIER-A MACRO FX DIRECTION program (`src/macro_tier_a.py`).

Every guard is tested BOTH ways: that it passes on correct data AND that it
FIRES on deliberately broken data. A guard that has never been shown to bite is
not evidence of anything.

The panel-level tests read the artifacts written by `python -m src.macro_tier_a`
and skip if they are absent, so the suite stays offline and fast.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import macro_tier_a as ta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_CSV = os.path.join(REPO, ta.OUT_DIR, 'panel_G10_primary.csv')
LOG_CSV = os.path.join(REPO, ta.HYPOTHESIS_LOG)
SHA_FIXTURE = os.path.join(REPO, ta.PROTECTED_SHA_FIXTURE)


def _panel():
    if not os.path.exists(PANEL_CSV):
        pytest.skip('panel artifact missing; run `python -m src.macro_tier_a` first')
    return pd.read_csv(PANEL_CSV, parse_dates=['month_end', 'asof'])


def _windows(panel):
    return ta.walk_forward_windows(panel, 'expanding')


def _synthetic_monthly(series_id='SYNTH', start='2000-01-01', periods=48, value=1.0):
    ref = pd.date_range(start, periods=periods, freq='MS')
    return ta._tier_a_publication_dates(
        pd.DataFrame({'ref_date': ref, 'value': np.arange(periods, dtype=float) + value,
                      'publication_date': pd.NaT}))


# ── 1. availability dating ────────────────────────────────────────────────────

def test_availability_dating_never_uses_a_value_published_after_month_end():
    """A feature for month M+1 may use only values published on or before the
    LAST BUSINESS DAY of month M -- and the guard fires on a mis-shifted series."""
    panel = _panel()
    ok = ta.complete_rows(panel)
    cutoff = pd.DatetimeIndex(ok['month_end']).map(ta._bmonth_end)

    # asof is exactly the last business day of the row's own month: lag zero.
    assert (pd.DatetimeIndex(ok['asof']) == pd.DatetimeIndex(cutoff)).all()
    assert (pd.DatetimeIndex(ok['asof']) <= pd.DatetimeIndex(ok['month_end'])).all()

    # The cutoff never rolls forward into the next month -- the BMonthEnd(0) trap.
    same_month = (pd.DatetimeIndex(ok['asof']).to_period('M') ==
                  pd.DatetimeIndex(ok['month_end']).to_period('M'))
    assert same_month.all()

    # GUARD BITES: a series published one month LATE must not be readable at asof.
    s = _synthetic_monthly()
    honest, _r, _p = ta.as_of(s, ta._bmonth_end(pd.Timestamp('2001-06-30')), 'M')
    late = s.copy()
    late['publication_date'] = pd.DatetimeIndex(late['publication_date']) + pd.DateOffset(months=1)
    delayed, _r, _p = ta.as_of(late, ta._bmonth_end(pd.Timestamp('2001-06-30')), 'M')
    assert honest != delayed, 'availability rule is not binding on publication date'
    assert delayed < honest, 'a later-published series must yield an OLDER value'


def test_bmonth_end_never_rolls_into_the_next_month():
    """The specific bug this program was born with: BMonthEnd(0) rolls FORWARD, so
    a month ending on a weekend leaked a full month of future data."""
    for day in pd.date_range('1976-01-31', '2026-06-30', freq='ME'):
        got = ta._bmonth_end(day)
        assert got.to_period('M') == day.to_period('M')
        assert got <= day
        assert got.weekday() < 5


# ── 2. no Tier-B series ───────────────────────────────────────────────────────

def test_no_tier_b_series_reaches_the_feature_matrix():
    for spec in ta.TIER_A_COUNTRY_SERIES.values():
        for sid, _freq in spec.values():
            ta.assert_tier_a_only([sid])
    ta.assert_tier_a_only([ta.US_EQUITY_SERIES[0], ta.EUR_PREDECESSOR_SERIES])
    ta.assert_tier_a_only([s['series'] for s in ta.PAIR_SPECS.values()])

    for col in ta.FEATURE_COLUMNS:
        assert not any(tok in col.upper() for tok in
                       ('CPI', 'INFLATION', 'UNEMPLOY', 'GDP', 'INDUSTRIAL'))

    # GUARD BITES on every deny-listed identifier family.
    for banned in ('CPIAUCSL', 'LRHUTTTTUSM156S', 'CP0000EZ19M086NEST',
                   'INDPRO', 'PRINTO01USM661N', 'GDPC1', 'AUSCPIALLQINMEI'):
        with pytest.raises(ta.TierBLeakError):
            ta.assert_tier_a_only([banned])


# ── 3. quote inversion ────────────────────────────────────────────────────────

def test_quote_inversion_round_trip_and_high_low_swap():
    frame = pd.DataFrame({'price': [1.25, 2.0], 'high': [1.30, 2.10],
                          'low': [1.20, 1.90]})
    inv = ta.invert_quote(frame)
    np.testing.assert_allclose(inv['price'], 1.0 / frame['price'])
    # On inversion high and low SWAP: 1/low is the new high.
    np.testing.assert_allclose(inv['high'], 1.0 / frame['low'])
    np.testing.assert_allclose(inv['low'], 1.0 / frame['high'])
    assert (inv['high'] > inv['low']).all()
    np.testing.assert_allclose(ta.invert_quote(inv)['price'], frame['price'])
    np.testing.assert_allclose(ta.invert_quote(inv)['high'], frame['high'])


def test_invert_flags_pinned_by_name():
    """Which pairs are quoted USD/XXX by market convention is a FACT, pinned so a
    later edit cannot silently flip a currency upside down."""
    expected = {'EURUSD': False, 'GBPUSD': False, 'AUDUSD': False, 'NZDUSD': False,
                'JPYUSD': True, 'CHFUSD': True, 'CADUSD': True, 'SEKUSD': True,
                'NOKUSD': True, 'MXNUSD': True, 'ZARUSD': True}
    assert {p: s['invert'] for p, s in ta.PAIR_SPECS.items()} == expected
    assert set(ta.G10_PANEL) == set(expected) - {'MXNUSD', 'ZARUSD'}


# ── 4. label uniqueness ───────────────────────────────────────────────────────

def test_label_uniqueness_is_exactly_one_and_no_duplicate_pair_month():
    ok = ta.complete_rows(_panel())
    assert ta.label_uniqueness(ok) == 1.0
    assert not ok.duplicated(subset=['pair', 'month_end']).any()
    # Monthly labels are non-overlapping by construction.
    for _pair, sub in ok.groupby('pair'):
        months = pd.DatetimeIndex(sub['month_end'])
        assert months.is_unique and months.is_monotonic_increasing


# ── 5. walk-forward causality, across ALL windows ─────────────────────────────

def test_every_window_trains_strictly_before_it_predicts():
    panel = _panel()
    windows = _windows(panel)
    assert len(windows) > 1
    for w in windows:
        assert max(w['train_months']) < min(w['oos_months']), f"window {w['year']} leaks"
        assert len(w['train_months']) >= ta.MIN_TRAIN_MONTHS
        assert not set(w['train_months']) & set(w['oos_months'])


# ── 6. per-window refitting ───────────────────────────────────────────────────

def test_standardiser_and_majority_are_refit_per_window():
    """Perturbing a LATER window's data must not change an EARLIER window's
    predictions. A single global standardiser or baseline would fail this."""
    ok = ta.complete_rows(_panel())
    windows = _windows(ok)[:6]
    base, base_profile, _c = ta.run_walk_forward(ok, windows)

    cut = max(windows[2]['oos_months'])
    poisoned = ok.copy()
    late = pd.DatetimeIndex(poisoned['month_end']) > cut
    assert late.any(), 'perturbation must actually touch later rows'
    for col in ta.FEATURE_COLUMNS:
        poisoned.loc[late, col] = poisoned.loc[late, col] * 1000.0 + 500.0
    poisoned.loc[late, 'y'] = 1 - poisoned.loc[late, 'y']

    after, after_profile, _c2 = ta.run_walk_forward(poisoned, windows)
    early = base[base['month_end'] <= cut]
    early_after = after[after['month_end'] <= cut]
    pd.testing.assert_series_equal(early['pred_logit'].reset_index(drop=True),
                                   early_after['pred_logit'].reset_index(drop=True))
    pd.testing.assert_series_equal(early['pred_majority'].reset_index(drop=True),
                                   early_after['pred_majority'].reset_index(drop=True))
    # ...and the perturbation is not inert: later windows DO change.
    assert not base['pred_logit'].equals(after['pred_logit'])


# ── 7. the reserved holdout ───────────────────────────────────────────────────

def test_reserved_final_36_months_are_never_indexed_and_the_guard_bites():
    panel = _panel()
    held = ta.reserved_months(panel)
    assert len(held) == ta.RESERVED_TAIL_MONTHS

    touched = set()
    for w in _windows(panel):
        touched |= set(w['train_months']) | set(w['oos_months'])
    assert not (touched & held), 'a window indexed the reserved holdout'
    assert max(touched) < min(held)

    # NON-VACUOUS: without the reservation those months WOULD have been used.
    unheld = set()
    for w in ta.walk_forward_windows(panel, 'expanding', tail=0):
        unheld |= set(w['train_months']) | set(w['oos_months'])
    assert unheld & held, 'the reservation guard is vacuous -- it removed nothing'


# ── 8. the carry rule ─────────────────────────────────────────────────────────

def test_carry_rule_is_exactly_the_stated_sign_comparison_including_ties():
    df = pd.DataFrame({'for_policy_rate': [5.0, 1.0, 2.0, 2.0, -0.5],
                       'us_policy_rate': [1.0, 5.0, 2.0, 2.0, 0.5]})
    np.testing.assert_array_equal(ta.carry_predictions(df, majority=1),
                                  [1, 0, 1, 1, 0])
    np.testing.assert_array_equal(ta.carry_predictions(df, majority=0),
                                  [1, 0, 0, 0, 0])   # ties follow the majority

    # It bets WITH the differential (the pre-registered sign), not against it.
    wide = pd.DataFrame({'for_policy_rate': [9.0], 'us_policy_rate': [0.0]})
    assert ta.carry_predictions(wide, majority=0)[0] == 1


# ── 9. no time / trend / pair-identity feature ────────────────────────────────

def test_no_time_trend_or_pair_identity_feature():
    ta.assert_no_forbidden_features(ta.FEATURE_COLUMNS)
    for banned in ('month_end', 'pair', 'country', 'asof'):
        assert banned not in ta.FEATURE_COLUMNS
    # GUARD BITES.
    for bad in ('month_index', 'time_trend', 'pair_id', 'country_dummy',
                'year_ordinal', 'ccy_onehot'):
        with pytest.raises(ta.ForbiddenFeatureError):
            ta.assert_no_forbidden_features(list(ta.FEATURE_COLUMNS) + [bad])


# ── 10. the staleness guard ───────────────────────────────────────────────────

def test_discontinued_series_goes_missing_rather_than_forward_filling():
    """Sweden's policy rate stops in 2020-10. Past the staleness limit the feature
    must be MISSING, not a five-year-old print pretending to be today's rate."""
    s = _synthetic_monthly(start='2000-01-01', periods=24)     # last ref 2001-12
    last_pub = pd.Timestamp(s['publication_date'].max())

    fresh, _r, _p = ta.as_of(s, last_pub, 'M')
    assert np.isfinite(fresh)
    inside, _r, _p = ta.as_of(s, last_pub + pd.Timedelta(days=100), 'M')
    assert np.isfinite(inside)
    beyond, _r, _p = ta.as_of(s, last_pub + pd.Timedelta(
        days=ta.MAX_STALENESS_DAYS['M'] + 1), 'M')
    assert np.isnan(beyond), 'a discontinued series forward-filled past its last print'

    # And it bites on the real panel: SEK stops well before the panel end.
    ok = ta.complete_rows(_panel())
    sek = ok[ok['pair'] == 'SEKUSD']['month_end']
    if len(sek):
        assert sek.max() < ok['month_end'].max(), 'SEK should die with its policy rate'


# ── 11. the protected set ─────────────────────────────────────────────────────

def test_protected_set_is_sha256_identical():
    """
    The macro_tier_a program must not modify any of these.

    RE-BASELINE NOTE: this fixture pins repo state, so a DIFFERENT, separately
    authorised program legitimately moves an entry. Twice so far, each time a
    single key, each time only after asserting no OTHER entry had drifted:
      * results/h1_direction_hypothesis_log.csv -- the H_dir one-shot test-block
        program spending that family's reserved block.
      * src/inference.py -- the H_dir.1 production integration, whose brief
        permits ADDITIVE modification of the serving module (verified: 173
        insertions, 0 deletions; not one existing line changed).
    One entry was UNPINNED rather than re-baselined: results/eurusd_h1.csv is an
    operational cache the DAILY predictor rewrites whenever its staleness gate
    fires, so pinning it asserted only that nobody had run a prediction. The
    invariant that actually matters -- that the H1-direction path never writes it
    -- is asserted directly in tests/test_h1_production.py.
    Any other file moving here is a boundary violation, not maintenance.
    """
    if not os.path.exists(SHA_FIXTURE):
        pytest.skip('protected-set fixture missing')
    with open(SHA_FIXTURE) as fh:
        expected = json.load(fh)
    assert len(expected) > 50
    for rel, digest in expected.items():
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'protected file vanished: {rel}'
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        assert got == digest, f'PROTECTED FILE MODIFIED: {rel}'


def test_macro_panel_log_still_carries_two_unspent_rows_at_its_original_alpha():
    path = os.path.join(REPO, 'results', 'macro_panel_hypothesis_log.csv')
    df = pd.read_csv(path)
    assert len(df) == 2
    assert (df['verdict'] == 'REGISTERED-UNSPENT').all()
    assert (df['alpha'] == 0.025).all()
    assert df['acc_challenger'].isna().all(), 'an UNSPENT row acquired a result'


# ── the amendment disclosure (condition 3 of the owner decision) ──────────────

def test_amendment_disclosure_is_present_in_the_hypothesis_log_notes():
    """The feature list was amended after the power gate fired. That fact must be
    readable in the log itself, not only in a chat transcript."""
    if not os.path.exists(LOG_CSV):
        pytest.skip('hypothesis log missing; run `python -m src.macro_tier_a` first')
    df = pd.read_csv(LOG_CSV)
    assert len(df) == 2
    for notes in df['notes']:
        assert ta.AMENDMENT_NOTE in notes, 'amendment disclosure missing from log notes'
        for phrase in ('vix_level was REMOVED', 'BEFORE any model was fitted',
                       'DATA COVERAGE ONLY', 'H_tA.1 (the carry rule) is UNAFFECTED',
                       'one column smaller than pre-registered'):
            assert phrase in notes
    assert 'vix_level' in ta.PRE_REGISTERED_FEATURE_COLUMNS
    assert 'vix_level' not in ta.FEATURE_COLUMNS
    assert 'us_equity_12m_return' in ta.FEATURE_COLUMNS, 'equity control must be RETAINED'
    assert len(ta.FEATURE_COLUMNS) == len(ta.PRE_REGISTERED_FEATURE_COLUMNS) - 1


def test_hypothesis_log_family_is_size_two_and_touches_no_other_family():
    if not os.path.exists(LOG_CSV):
        pytest.skip('hypothesis log missing')
    df = pd.read_csv(LOG_CSV)
    assert (df['alpha'] == 0.025).all()
    assert ta.ALPHA == 0.05 / ta.FAMILY_SIZE == 0.025
    assert set(df['hypothesis']) == {'H_tA.1_carry_vs_train_majority',
                                     'H_tA.2_logistic_vs_carry'}
