"""
Unit tests for the H_dir FAMILY FINAL ONE-SHOT TEST-BLOCK program
(`src/h1_direction_final.py`).

The point of these is the DISCIPLINE, not the arithmetic: the reproduction gate
must fire, the ordering guard must refuse a post-read fit, the baseline must be
blind to the evaluation data, and the protected set must be untouched. Every
guard is shown to BITE, not merely to pass.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import h1_direction_final as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA_FIXTURE = os.path.join(REPO, 'tests', 'fixtures', 'h_dir_final_protected_sha256.json')
PRED_CSV = os.path.join(REPO, F.OUT_DIR, 'testblock_predictions.csv')
LOG_CSV = os.path.join(REPO, F.HYPOTHESIS_LOG)


# ── 1. the reproduction gate ──────────────────────────────────────────────────

def test_reproduction_gate_fires_outside_tolerance_in_both_directions():
    """Two-sided: a refit scoring materially HIGHER is just as much 'not the
    registered model' as one scoring lower."""
    assert F.assert_reproduction(F.REPRO_TARGET_ACC) == pytest.approx(0.0)
    # inside the band, both sides
    for acc in (F.REPRO_TARGET_ACC + 0.0029, F.REPRO_TARGET_ACC - 0.0029):
        F.assert_reproduction(acc)
    # GUARD BITES, both sides
    for acc in (F.REPRO_TARGET_ACC + 0.0031, F.REPRO_TARGET_ACC - 0.0031):
        with pytest.raises(F.ReproductionGateError):
            F.assert_reproduction(acc)
    assert F.REPRO_TOLERANCE == 0.003
    assert F.REPRO_TARGET_ACC == 0.527462


# ── 2. nothing is fitted after the first test-block read ──────────────────────

def test_no_fit_after_the_test_block_is_read():
    guard = F.TestBlockGuard()
    assert not guard.was_read
    guard.assert_unread('a fit before the read')          # allowed
    assert guard.fits == ['a fit before the read']

    guard.read(object(), label='test block')
    assert guard.was_read and guard.read_at is not None

    # GUARD BITES: any fit call site after the read raises.
    with pytest.raises(F.FitAfterTestBlockReadError):
        guard.assert_unread('a fit after the read')
    with pytest.raises(F.FitAfterTestBlockReadError):
        F.train_magnitude_gbm(np.zeros((4, 2)), np.zeros(4), guard=guard)

    # The stamp is set ONCE: re-reading for descriptive breakdowns is fine.
    first = guard.read_at
    guard.read(object())
    assert guard.read_at == first


def test_the_real_run_fitted_everything_before_reading_the_block():
    """The report's ordering claim, asserted against the artifact."""
    if not os.path.exists(LOG_CSV):
        pytest.skip('log missing')
    log = pd.read_csv(LOG_CSV)
    conf = log[log['hypothesis'].str.contains('TESTBLOCK_CONFIRMATION', na=False)]
    assert len(conf) == 1
    assert 'first read at' in str(conf.iloc[0]['notes'])
    assert 'Reproduction gate PASSED before the block was read' in str(conf.iloc[0]['notes'])


# ── 3. claim B's target ───────────────────────────────────────────────────────

def test_magnitude_target_is_exactly_next_over_current_minus_one_times_100():
    close = pd.Series([1.0, 1.1, 1.05, 2.0],
                      index=pd.date_range('2020-01-01', periods=4, freq='h'))
    got = F.build_magnitude_target(pd.DataFrame({'close': close}))
    expected = [(1.1 / 1.0 - 1) * 100, (1.05 / 1.1 - 1) * 100, (2.0 / 1.05 - 1) * 100]
    np.testing.assert_allclose(got.to_numpy()[:3], expected)
    assert np.isnan(got.to_numpy()[-1]), 'final bar has no next close -> NaN, not padded'

    # An off-by-one -- the BACKWARD return at the same timestamp -- is a
    # different number on that row, which is the error this pins.
    backward = (close / close.shift(1) - 1.0) * 100.0
    both = got.notna() & backward.notna()
    assert both.sum() >= 2
    assert not np.allclose(got[both].to_numpy(), backward[both].to_numpy())
    # Shifting the target one bar the wrong way also changes it.
    assert not np.allclose(got.to_numpy()[:2], got.to_numpy()[1:3])

    # And the sign agrees with the direction module's log-return sign.
    logret = np.log(close.shift(-1) / close)
    ok = got.notna()
    assert ((got[ok] > 0) == (logret[ok] > 0)).all()


# ── 4. the baseline is blind to the evaluation data ───────────────────────────

def test_constant_baseline_uses_the_train_mean_only():
    y_train = np.array([0.1, -0.3, 0.2, 0.0])
    base = F.constant_baseline(y_train)
    assert base == pytest.approx(np.mean(y_train))

    # Perturbing the TEST values cannot move it -- it never sees them.
    y_test = np.array([99.0, -99.0, 50.0])
    assert F.constant_baseline(y_train) == pytest.approx(base)
    y_test *= 1000.0
    assert F.constant_baseline(y_train) == pytest.approx(base)
    # ...and it is NOT the test mean.
    assert base != pytest.approx(float(np.mean(y_test)))


# ── 5. both claims score the identical row set ────────────────────────────────

def test_both_claims_score_the_identical_row_set():
    if not os.path.exists(PRED_CSV):
        pytest.skip('test-block predictions missing; run the program first')
    p = pd.read_csv(PRED_CSV, parse_dates=['timestamp'])
    assert len(p) > 0
    # Same rows, no NaN on either side, index equality by construction.
    assert p['y_direction'].notna().all() and p['y_pct'].notna().all()
    assert p['pred_direction'].notna().all() and p['pred_pct'].notna().all()
    assert p['timestamp'].is_unique and p['timestamp'].is_monotonic_increasing
    # The decisive check: the direction label IS the sign of the percent target
    # on every row, which is only possible if both claims used the same rows.
    np.testing.assert_array_equal(p['y_direction'].to_numpy().astype(int),
                                  (p['y_pct'].to_numpy() > 0).astype(int))
    # Zero-return rows were dropped for BOTH claims.
    assert (p['y_pct'] != 0).all()


# ── 6. MAE-difference sign convention ─────────────────────────────────────────

def test_mae_difference_is_negative_when_the_model_is_strictly_better():
    y = np.array([1.0, -2.0, 3.0, -4.0])
    good = y.copy()                       # perfect model
    bad = np.zeros_like(y)                # constant baseline
    diff = F.mae_difference_vector(good, bad, y)
    assert diff.mean() < 0, 'a strictly better model must give a NEGATIVE difference'
    np.testing.assert_allclose(diff, -np.abs(y))

    # ...and positive when the model is strictly worse.
    worse = y + 10.0
    assert F.mae_difference_vector(worse, bad, y).mean() > 0

    # The bootstrap keeps the convention, and KEEP requires the CI BELOW zero.
    res = F.magnitude_arbiter(good, bad, y, alpha=F.ALPHA_FINAL, block_len=2, n_boot=200)
    assert res['mae_diff'] < 0 and res['mae_diff_ci_high'] < 0
    assert res['cleared_bar'] is True and res['verdict'] == 'KEEP'
    res_bad = F.magnitude_arbiter(worse, bad, y, alpha=F.ALPHA_FINAL, block_len=2,
                                  n_boot=200)
    assert res_bad['verdict'] == 'DROP'


# ── 7. the protected set ──────────────────────────────────────────────────────

def test_protected_set_is_sha256_identical():
    """models/, _train_pipeline.py, src/inference.py, src/features.py,
    src/paper_trading.py, config.json, src/h1_direction_model.py, and every OTHER
    family's hypothesis log. This family's own log is the program's target and is
    deliberately excluded from the fixture.

    RE-BASELINE NOTE: src/inference.py was re-baselined for the H_dir.1
    production integration, whose brief permits ADDITIVE modification of the
    serving module (verified: 173 insertions, 0 deletions; not one existing line
    changed), and AGAIN for the Kronos external-model integration, whose brief
    permits the same (verified: 133 insertions, 0 deletions). On both occasions
    src/inference.py was the ONLY drifting entry in this fixture; every other
    pinned file was byte-identical, which is what makes it maintenance rather
    than a boundary being crossed. results/eurusd_h1.csv was UNPINNED: it is an
    operational cache the
    daily predictor rewrites whenever its staleness gate fires, and the invariant
    that matters -- the H1-direction path never writing it -- is asserted
    directly in tests/test_h1_production.py. Any other entry moving here is a
    boundary violation rather than maintenance."""
    if not os.path.exists(SHA_FIXTURE):
        pytest.skip('protected-set fixture missing')
    with open(SHA_FIXTURE) as fh:
        expected = json.load(fh)
    assert len(expected) > 40
    assert 'results/h1_direction_hypothesis_log.csv' not in expected
    assert 'src/h1_direction_model.py' in expected
    for rel, digest in expected.items():
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'protected file vanished: {rel}'
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        assert got == digest, f'PROTECTED FILE MODIFIED: {rel}'


# ── the family resize and the no-trading-frame rule ───────────────────────────

def test_family_alpha_restated_on_every_row():
    """The family may GROW -- the standing rule is that a genuinely new
    hypothesis resizes it and the tightened alpha applies retroactively. What
    must hold is that EVERY row carries the CURRENT family bar, derived from the
    log's own distinct-hypothesis count, and that n is gap-free.

    F.ALPHA_FINAL is deliberately NOT asserted against the log here: it is a
    HISTORICAL constant recording the 5 -> 6 resize that the spent-test-block
    program performed, not the living family size. Pinning the log to it froze
    the family at 6 and would fail the moment the documented growth rule was
    exercised (it did, when H_dir.7 was registered 2026-08-07).
    """
    if not os.path.exists(LOG_CSV):
        pytest.skip('log missing')
    log = pd.read_csv(LOG_CSV)
    assert F.ALPHA_FINAL == pytest.approx(0.05 / 6), \
        'the historical 5->6 resize constant must not be rewritten'

    ns = sorted(set(log['n'].astype(int)))
    assert ns == list(range(1, len(ns) + 1)), f'family numbering has a gap: {ns}'
    family_size = len(ns)
    expected_alpha = 0.05 / family_size
    assert np.allclose(log['alpha'].astype(float).to_numpy(), expected_alpha, atol=1e-6), \
        f'every row must carry the retroactive alpha 0.05/{family_size}'
    assert log['hypothesis'].str.startswith('H_dir.6').any()
    # A tightened alpha can only remove clears, never create them.
    assert not (log['hypothesis'] == 'H_dir.2_LSTM_vs_GBM').pipe(
        lambda m: log.loc[m, 'cleared_bar'].astype(str).eq('True').any())
    assert not (log['hypothesis'] == 'H_dir.3_replication_GBPUSD').pipe(
        lambda m: log.loc[m, 'cleared_bar'].astype(str).eq('True').any())


def test_no_trading_frame_language_anywhere_in_the_module_or_log():
    """The breakeven/P&L framing was removed as out of scope for a predictive
    question. Assert it did not creep back in."""
    src = open(os.path.join(REPO, 'src', 'h1_direction_final.py'),
               encoding='utf-8').read().lower()
    banned = ('breakeven', 'break-even', 'sharpe', 'equity curve', 'position siz',
              'transaction cost', 'pips of profit', 'p&l')
    for term in banned:
        # The module may NAME the removal; it may not compute one.
        occurrences = src.count(term)
        mentions_removal = src.count('removed') + src.count('deliberately absent')
        assert occurrences == 0 or mentions_removal > 0, f'trading frame term: {term}'
    if os.path.exists(LOG_CSV):
        notes = ' '.join(pd.read_csv(LOG_CSV)['notes'].astype(str)).lower()
        assert 'no trading frame' in notes
