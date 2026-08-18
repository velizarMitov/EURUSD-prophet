"""Unit tests for the Yıldırım/Toroslu/Fiore (2021) replication study.

Covers the four things the study's conclusions actually rest on:
  1. the paper's histogram upper-bound rule (Algorithm 1),
  2. its entropy threshold search (Algorithm 2),
  3. profit_accuracy (Eq. 25) against a hand-computed confusion table,
  4. the arm-A0 discipline: no selection step may read the test block.

None of these import keras/torch — every function under test is pure
numpy/pandas, which is why the labelling and metric layer was kept free of the
model layer in the first place.
"""
import numpy as np
import pytest

from src.yildirim_replication import (
    ARMS, DEC, INC, NOACT, Block, entropy_of_counts, entropy_threshold_search,
    histogram_threshold_upper_bound, hybrid_decide, label_three_class,
    make_sequences, profit_accuracy, profit_accuracy_counts,
    profit_accuracy_from_counts, run_arm, select_hybrid, select_iterations,
    ITERATION_GRID, MODEL_TYPES,
)


# ---------------------------------------------------------------------------
# 1. Algorithm 1 — histogram upper bound
# ---------------------------------------------------------------------------

def _hand_built_diffs():
    """100 absolute differences with a hand-known 10-bin histogram.

    max = 10.0 -> bin width 1.0, edges 0,1,...,10.
        bin0 [0,1):  50 values (0.5)
        bin1 [1,2):  30 values (1.5)
        bin2 [2,3):  15 values (2.5 x14, 2.75 x1)   <- largest observed 2.75
        bin9 [9,10]:  5 values (9.5 x4, 10.0 x1)
    Descending counts: 50, 30, 15, 5 -> cumulative 50, 80, 95. 85% of 100 is
    85.0, first exceeded at the third bin (bin2), so bin2 is 'the last bin
    added'.
    """
    return np.array([0.5] * 50 + [1.5] * 30 + [2.5] * 14 + [2.75]
                    + [9.5] * 4 + [10.0])


def test_histogram_upper_bound_max_observed_in_last_bin():
    d = _hand_built_diffs()
    assert histogram_threshold_upper_bound(d) == pytest.approx(2.75)


def test_histogram_upper_bound_bin_edge_reading():
    """The alternative reading of 'the maximum difference value of the last bin
    added' (UNDERSPECIFIED[#1]) is the bin's upper edge, here 3.0."""
    d = _hand_built_diffs()
    assert histogram_threshold_upper_bound(d, use_bin_edge=True) == pytest.approx(3.0)


def test_histogram_upper_bound_sign_is_ignored():
    """'the minimum (which is 0)' — the histogram is over ABSOLUTE differences,
    so flipping signs cannot move the bound."""
    d = _hand_built_diffs()
    signed = d * np.where(np.arange(len(d)) % 2 == 0, 1.0, -1.0)
    assert (histogram_threshold_upper_bound(signed)
            == pytest.approx(histogram_threshold_upper_bound(d)))


def test_histogram_upper_bound_coverage_is_strict_exceedance():
    """'summed until the sum exceeded 85%': a bin carrying exactly 85 of 100
    does NOT terminate the accumulation — the next bin must be added too."""
    d = np.array([0.5] * 85 + [5.4] * 15)   # max 5.4 -> bin0 holds 85, bin9 holds 15
    # cumulative 85 is not > 85.0, so bin9 is added and becomes the last bin.
    assert histogram_threshold_upper_bound(d) == pytest.approx(5.4)
    # Sanity: had the rule been >=, accumulation would have stopped at bin0 and
    # the answer would have been 0.5.
    assert histogram_threshold_upper_bound(d) != pytest.approx(0.5)


def test_histogram_upper_bound_ties_take_the_lower_bin():
    """Equal counts are resolved by the stable descending sort, so the lower
    (denser-by-convention) bin is consumed first — UNDERSPECIFIED[#2]."""
    # bin0 and bin1 both hold 5; 85% of 11 is 9.35, exceeded only at the second
    # bin added, which must be bin1 (not bin9) if the sort is stable.
    d = np.array([0.5] * 5 + [1.5] * 5 + [10.0])
    assert histogram_threshold_upper_bound(d) == pytest.approx(1.5)


def test_histogram_upper_bound_degenerate_all_zero():
    assert histogram_threshold_upper_bound(np.zeros(20)) == 0.0


# ---------------------------------------------------------------------------
# 2. Algorithm 2 — entropy threshold search
# ---------------------------------------------------------------------------

def test_entropy_of_counts_matches_closed_form():
    assert entropy_of_counts([1, 1, 1]) == pytest.approx(np.log(3))
    assert entropy_of_counts([1, 1]) == pytest.approx(np.log(2))
    assert entropy_of_counts([5, 0, 0]) == pytest.approx(0.0)
    assert entropy_of_counts([0, 0, 0]) == 0.0


def test_entropy_threshold_search_finds_the_balancing_tau():
    """Six symmetric differences at +/-1, +/-2, +/-3 (x1e-4).

    tau=0.0000 -> (noact, dec, inc) = (0, 3, 3), entropy ln2   = 0.6931
    tau=0.0001 -> (2, 2, 2),            entropy ln3            = 1.0986  <- max
    tau=0.0002 -> (4, 1, 1),            entropy                = 0.8676
    The perfectly balanced split is the unique maximum, and 0.0001 is the
    smallest tau achieving it (0.00015 ties; the search must prefer the
    smaller, i.e. the more-transactions, reading).
    """
    d = np.array([-3e-4, -2e-4, -1e-4, 1e-4, 2e-4, 3e-4])
    ub = histogram_threshold_upper_bound(d)
    assert ub == pytest.approx(3e-4)

    tau, ent = entropy_threshold_search(d, ub)
    assert tau == pytest.approx(1e-4)
    assert ent == pytest.approx(np.log(3))

    counts = np.bincount(label_three_class(d, tau), minlength=3)
    assert counts.tolist() == [2, 2, 2]


def test_entropy_threshold_search_grid_lands_exactly_on_step_multiples():
    """A difference sitting exactly on a grid point must be classified by the
    paper's strict inequality, not by float drift in the tau grid."""
    d = np.array([-1e-4, 1e-4] * 10)
    tau, _ = entropy_threshold_search(d, 5e-4)
    # Every tau >= 1e-4 makes all 20 rows no_action (entropy 0); only tau < 1e-4
    # splits them, so the maximum is at tau = 0 (or any tau below 1e-4).
    assert tau < 1e-4
    assert np.all(label_three_class(d, 1e-4) == NOACT)      # strict `>` / `<`
    assert not np.any(label_three_class(d, 0.99e-4) == NOACT)


def test_label_three_class_uses_the_papers_encoding():
    d = np.array([0.01, -0.01, 0.0, 0.001, -0.001])
    y = label_three_class(d, 0.005)
    assert y.tolist() == [INC, DEC, NOACT, NOACT, NOACT]
    assert (NOACT, DEC, INC) == (0, 1, 2)      # "labeled as 2 / 1 / 0"


def test_entropy_threshold_search_rejects_empty_input():
    with pytest.raises(ValueError):
        entropy_threshold_search(np.array([]), 0.01)


# ---------------------------------------------------------------------------
# 3. Eq. 25 — profit_accuracy against a hand-computed confusion table
# ---------------------------------------------------------------------------

# (y_true, y_pred, raw diff, expected Table-2 cell)
_HAND_TABLE = [
    (INC,   INC,    +0.005, 'True_inc'),
    (INC,   INC,    +0.004, 'True_inc'),
    (DEC,   DEC,    -0.005, 'True_dec'),
    # Paper's conversion rule: predicted a direction on a true no_act row, and
    # the actual movement went that way -> counts as a TRUE prediction.
    (NOACT, INC,    +0.001, 'True_inc'),
    (NOACT, DEC,    -0.001, 'True_dec'),
    # Same situation, actual movement the other way -> the false-noact cells.
    (NOACT, INC,    -0.001, 'False_inc_noact'),
    (NOACT, DEC,    +0.001, 'False_dec_noact'),
    # An exactly flat move has no direction: it can never be 'in the same
    # direction with the prediction' (UNDERSPECIFIED[#5]).
    (NOACT, INC,     0.000, 'False_inc_noact'),
    # Outright direction errors.
    (DEC,   INC,    -0.004, 'False_inc_dec'),
    (INC,   DEC,    +0.004, 'False_dec_inc'),
    # Predicted no_act -> not a transaction, appears in no cell at all.
    (INC,   NOACT,  +0.006, None),
    (NOACT, NOACT,   0.000, None),
]


def _hand_table_arrays():
    y_true = np.array([r[0] for r in _HAND_TABLE])
    y_pred = np.array([r[1] for r in _HAND_TABLE])
    diff = np.array([r[2] for r in _HAND_TABLE])
    return y_true, y_pred, diff


def test_profit_accuracy_counts_match_the_hand_table():
    y_true, y_pred, diff = _hand_table_arrays()
    counts = profit_accuracy_counts(y_true, y_pred, diff)

    expected = {'True_inc': 3, 'True_dec': 2, 'False_inc_noact': 2,
                'False_dec_noact': 1, 'False_inc_dec': 1, 'False_dec_inc': 1}
    assert counts == expected

    # Every cell is accounted for exactly once, and the two Pred(no_act) rows
    # are in none of them.
    assert sum(counts.values()) == sum(1 for r in _HAND_TABLE if r[3] is not None)
    assert sum(counts.values()) == 10


def test_profit_accuracy_formula_is_eq_25_verbatim():
    y_true, y_pred, diff = _hand_table_arrays()
    counts = profit_accuracy_counts(y_true, y_pred, diff)

    num = counts['True_dec'] + counts['True_inc']
    den = (counts['False_dec_noact'] + counts['False_inc_noact'] + counts['True_dec']
           + counts['False_inc_dec'] + counts['False_dec_inc'] + counts['True_inc'])
    assert (num, den) == (5, 10)
    assert profit_accuracy_from_counts(counts) == pytest.approx(0.5)


def test_profit_accuracy_vector_form_agrees_with_the_confusion_table():
    """The run loop and the bootstrap use the vector form; it must be the same
    number as the six-cell table, on every input."""
    y_true, y_pred, diff = _hand_table_arrays()
    table_pa = profit_accuracy_from_counts(profit_accuracy_counts(y_true, y_pred, diff))
    assert profit_accuracy(y_pred, diff) == pytest.approx(table_pa)

    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(5, 200))
        d = rng.normal(0, 0.004, n)
        yt = label_three_class(d, 0.002)
        yp = rng.integers(0, 3, n)
        ref = profit_accuracy_from_counts(profit_accuracy_counts(yt, yp, d))
        got = profit_accuracy(yp, d)
        assert (np.isnan(ref) and np.isnan(got)) or got == pytest.approx(ref)


def test_profit_accuracy_is_nan_when_no_transaction_is_made():
    """The paper prints 'Nan' for the 0/242 cell of Table 15."""
    counts = profit_accuracy_counts(np.array([INC]), np.array([NOACT]), np.array([0.01]))
    assert np.isnan(profit_accuracy_from_counts(counts))
    assert np.isnan(profit_accuracy(np.array([NOACT]), np.array([0.01])))


def test_profit_accuracy_counts_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        profit_accuracy_counts(np.zeros(3), np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------------------
# 4. Arm A0 never reads the test block during selection
# ---------------------------------------------------------------------------

class _Tripwire:
    """Stands in for the test block. Any read at all raises."""

    def __getattr__(self, name):
        raise AssertionError(f"A0 selection touched the test block (.{name})")

    def __getitem__(self, key):
        raise AssertionError(f"A0 selection touched the test block ([{key!r}])")

    def __len__(self):
        raise AssertionError("A0 selection touched the test block (len)")


def _synthetic_blocks(seed=0, n_val=120, n_test=243):
    """Val and test blocks whose best iteration count DISAGREES, so that a leak
    is observable at all: on val, iteration 50 is the strongest; on test it is
    the weakest. Without that the A2/A3 half of these tests would be vacuous.
    """
    rng = np.random.default_rng(seed)

    def _mk(n, quality_by_iter):
        diff = rng.normal(0, 0.004, n)
        y = label_three_class(diff, 0.002)
        blk = Block('b', y, diff)
        for mt in MODEL_TYPES:
            for it in ITERATION_GRID:
                q = quality_by_iter[it]
                # A fraction q of rows carry the true label, the rest a random
                # one — so profit_accuracy really does track q instead of
                # saturating at 1.0 the moment the label is right.
                target = np.where(rng.random(n) < q, y, rng.integers(0, 3, n))
                p = rng.random((n, 3)) * 0.15 + 0.05
                p[np.arange(n), target] += 1.0
                blk.proba[(mt, it)] = p / p.sum(axis=1, keepdims=True)
        return blk

    val = _mk(n_val, {50: 0.90, 100: 0.70, 150: 0.50, 200: 0.30})
    test = _mk(n_test, {50: 0.30, 100: 0.50, 150: 0.70, 200: 0.90})
    return val, test


def test_a0_iteration_selection_never_reads_the_test_block():
    val, _ = _synthetic_blocks()
    blocks = {'val': val, 'test': _Tripwire()}
    for mt in MODEL_TYPES:
        chosen = select_iterations(ARMS['A0'], blocks, mt)
        assert chosen in ITERATION_GRID


def test_a0_hybrid_tuning_never_reads_the_test_block():
    val, _ = _synthetic_blocks()
    blocks = {'val': val, 'test': _Tripwire()}
    mc, tb = select_hybrid(ARMS['A0'], blocks, iters_me=100, iters_ti=100)
    assert tb in ('me', 'ti')
    assert 0.0 <= mc <= 1.0


def test_the_tripwire_actually_fires_for_the_leaking_arms():
    """Guards against a vacuous test: A2 and A3 must trip the same wire that A0
    walks past, otherwise the two tests above prove nothing."""
    val, _ = _synthetic_blocks()
    blocks = {'val': val, 'test': _Tripwire()}
    with pytest.raises(AssertionError, match="touched the test block"):
        select_iterations(ARMS['A2'], blocks, 'ME_LSTM')
    with pytest.raises(AssertionError, match="touched the test block"):
        select_hybrid(ARMS['A3'], blocks, iters_me=100, iters_ti=100)


def _scramble(blk, seed=99):
    """Same shapes, entirely different content."""
    rng = np.random.default_rng(seed)
    out = Block(blk.name, rng.permutation(blk.y), rng.permutation(blk.diff))
    for k, p in blk.proba.items():
        q = rng.random(p.shape)
        out.proba[k] = q / q.sum(axis=1, keepdims=True)
    return out


def test_a0_selections_are_invariant_to_the_test_blocks_contents():
    """End-to-end statement of the same discipline through run_arm: replacing
    the test block's labels and probabilities with noise cannot change ANY
    choice A0 makes. Only the scores it then reports may change."""
    val, test = _synthetic_blocks()
    real = {'train_only': {'val': val, 'test': test},
            'full_series': {'val': val, 'test': test}}
    fake = {'train_only': {'val': val, 'test': _scramble(test)},
            'full_series': {'val': val, 'test': _scramble(test)}}

    a0_real = {m: meta for m, (_, meta) in run_arm(ARMS['A0'], real).items()}
    a0_fake = {m: meta for m, (_, meta) in run_arm(ARMS['A0'], fake).items()}
    assert a0_real == a0_fake

    # Non-vacuity: the leaking arms DO change their choices when the test block
    # changes, so the invariance above is a property of A0, not of the fixture.
    a2_real = {m: meta for m, (_, meta) in run_arm(ARMS['A2'], real).items()}
    a2_fake = {m: meta for m, (_, meta) in run_arm(ARMS['A2'], fake).items()}
    a3_real = {m: meta for m, (_, meta) in run_arm(ARMS['A3'], real).items()}
    a3_fake = {m: meta for m, (_, meta) in run_arm(ARMS['A3'], fake).items()}
    assert a2_real != a2_fake
    assert a3_real != a3_fake


def test_a0_picks_the_validation_optimum_not_the_test_optimum():
    """The fixture is rigged so val prefers 50 iterations and test prefers 200.
    A0 must land on val's answer and A2 on test's — that is the whole of
    Leak 2."""
    val, test = _synthetic_blocks()
    blocks = {'val': val, 'test': test}
    assert select_iterations(ARMS['A0'], blocks, 'TI_LSTM') == 50
    assert select_iterations(ARMS['A2'], blocks, 'TI_LSTM') == 200


def test_arm_specs_encode_the_intended_leak_matrix():
    assert (ARMS['A0'].threshold_scope, ARMS['A0'].iter_block, ARMS['A0'].hybrid_block) \
        == ('train_only', 'val', 'val')
    assert (ARMS['A1'].threshold_scope, ARMS['A1'].iter_block, ARMS['A1'].hybrid_block) \
        == ('full_series', 'val', 'val')
    assert (ARMS['A2'].threshold_scope, ARMS['A2'].iter_block, ARMS['A2'].hybrid_block) \
        == ('train_only', 'test', 'val')
    assert (ARMS['A3'].threshold_scope, ARMS['A3'].iter_block, ARMS['A3'].hybrid_block) \
        == ('train_only', 'val', 'test')
    assert (ARMS['A4'].threshold_scope, ARMS['A4'].iter_block, ARMS['A4'].hybrid_block) \
        == ('full_series', 'test', 'test')


# ---------------------------------------------------------------------------
# Supporting invariants
# ---------------------------------------------------------------------------

def test_hybrid_at_zero_confidence_is_the_papers_published_rule():
    """min_confidence=0.0 must be an exact no-op, so the paper's own combiner is
    a member of every arm's tuning grid rather than a competitor to it."""
    p_me = np.array([
        [0.6, 0.2, 0.2],   # noact from ME     -> noact
        [0.1, 0.7, 0.2],   # both say dec      -> dec
        [0.1, 0.2, 0.7],   # disagree, ME more confident (0.7 > 0.6) -> ME's inc
        [0.2, 0.5, 0.3],   # disagree, equal confidence -> tie_break
    ])
    p_ti = np.array([
        [0.1, 0.2, 0.7],
        [0.2, 0.6, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.3, 0.5],
    ])
    out_ti = hybrid_decide(p_me, p_ti, min_confidence=0.0, tie_break='ti')
    assert out_ti.tolist() == [NOACT, DEC, INC, INC]        # tie -> TI's inc
    out_me = hybrid_decide(p_me, p_ti, min_confidence=0.0, tie_break='me')
    assert out_me.tolist() == [NOACT, DEC, INC, DEC]        # tie -> ME's dec


def test_hybrid_confidence_floor_only_ever_removes_transactions():
    rng = np.random.default_rng(3)
    p_me = rng.random((200, 3))
    p_me /= p_me.sum(axis=1, keepdims=True)
    p_ti = rng.random((200, 3))
    p_ti /= p_ti.sum(axis=1, keepdims=True)
    base = hybrid_decide(p_me, p_ti, 0.0, 'ti')
    prev = (base != NOACT).sum()
    for mc in (0.40, 0.50, 0.60, 0.70):
        cur = (hybrid_decide(p_me, p_ti, mc, 'ti') != NOACT).sum()
        assert cur <= prev
        prev = cur
    # and a transaction that survives must keep the label it had at mc=0
    kept = hybrid_decide(p_me, p_ti, 0.55, 'ti') != NOACT
    assert np.array_equal(hybrid_decide(p_me, p_ti, 0.55, 'ti')[kept], base[kept])


def test_hybrid_rejects_an_unknown_tie_break():
    p = np.full((3, 3), 1 / 3)
    with pytest.raises(ValueError):
        hybrid_decide(p, p, 0.0, 'coin')


def test_make_sequences_is_trailing_only():
    """Sequence i must contain rows [end-T+1, end] and nothing after end — the
    project's no-look-ahead invariant, restated for this module's own windowing."""
    x = np.arange(60, dtype=float).reshape(30, 2)
    seq, ends = make_sequences(x, 5)
    assert seq.shape == (26, 5, 2)
    assert ends[0] == 4 and ends[-1] == 29
    for i, e in enumerate(ends):
        assert np.array_equal(seq[i], x[e - 4:e + 1])
        assert seq[i][-1][0] == x[e][0]


def test_make_sequences_handles_a_too_short_input():
    seq, ends = make_sequences(np.zeros((3, 2)), 5)
    assert seq.shape[0] == 0 and ends.size == 0
