"""Invariant tests for ``src/calibration_audit.py`` (research-only diagnostic).

The load-bearing test is ``test_vectorised_consensus_matches_production_row_by_row``.
``consensus_frame`` is a hand-vectorised COPY of ``PredictionService.compute_consensus``
(the production method takes one dict per call and the audit needs ~8.5k rows). A copy
of production logic that silently drifts would make every gate number in the audit a
measurement of something production does not do -- which is worse than not measuring
at all, because it reads as evidence.

Nothing here asserts predictive power, and nothing here loads a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.calibration_audit import (
    CONFIDENCE_THRESHOLD,
    brier_decomposition,
    confidence_correctness_spearman,
    consensus_frame,
    reliability_table,
)
from src.inference import PredictionService


# ---------------------------------------------------------------------------
# The production-parity control
# ---------------------------------------------------------------------------
def test_vectorised_consensus_matches_production_row_by_row() -> None:
    """Every branch of compute_consensus, on a grid that deliberately straddles
    0.5 (the direction flip) and 0.52 (the guard), plus exact ties on both."""
    grid = np.array([0.30, 0.45, 0.48, 0.4999, 0.50, 0.5001, 0.51, 0.52, 0.5201, 0.60, 0.72])
    p_gbm, p_lstm = (a.ravel() for a in np.meshgrid(grid, grid, indexing="ij"))

    got = consensus_frame(p_gbm, p_lstm)

    for i, (pg, pl) in enumerate(zip(p_gbm, p_lstm)):
        expected = PredictionService.compute_consensus(
            {
                "gbm": {
                    "direction": "UP" if pg >= 0.5 else "DOWN",
                    "confidence": pg if pg >= 0.5 else 1 - pg,
                    "predicted_return_pct": 0.0,
                },
                "lstm": {
                    "direction": "UP" if pl >= 0.5 else "DOWN",
                    "confidence": pl if pl >= 0.5 else 1 - pl,
                    "predicted_return_pct": 0.0,
                },
            }
        )
        row = got.iloc[i]
        assert row["confidence"] == pytest.approx(expected["confidence"]), (pg, pl)

        if expected["direction"] == "MIXED / LOW CONFIDENCE":
            assert row["downgraded"], (pg, pl)
            assert not row["gated"], (pg, pl)
        else:
            assert not row["downgraded"], (pg, pl)
            assert ("UP" if row["up"] else "DOWN") == expected["direction"], (pg, pl)
            assert row["gated"] == expected["agreement"], (pg, pl)


def test_the_guard_is_not_applied_when_the_heads_disagree() -> None:
    """Documents a real asymmetry in production rather than a property of the audit:
    the 0.52 downgrade lives only in compute_consensus's `if agreement:` branch. On
    disagreement the service emits the more confident head's direction with NO
    threshold check. The audit must reproduce that, not a tidier version of it."""
    p_gbm = np.array([0.505])   # UP, confidence 0.505 -- below the bar
    p_lstm = np.array([0.499])  # DOWN, confidence 0.501
    row = consensus_frame(p_gbm, p_lstm).iloc[0]

    assert not row["agree"]
    assert row["confidence"] < CONFIDENCE_THRESHOLD
    assert not row["downgraded"], "a sub-threshold disagreement is still emitted as a direction"
    assert row["emitted_direction"]
    assert not row["gated"], "but it must not count as having cleared the guard"


def test_consensus_probability_signs_back_to_p_up() -> None:
    """`p` must be a genuine P(up) so the reliability table and Brier score mean
    what they say -- confidence alone is folded at 0.5 and would fake calibration."""
    p_gbm = np.array([0.7, 0.2])
    p_lstm = np.array([0.6, 0.3])
    got = consensus_frame(p_gbm, p_lstm)
    np.testing.assert_allclose(got["p"], [0.65, 0.25])
    np.testing.assert_allclose(got["confidence"], [0.65, 0.75])


# ---------------------------------------------------------------------------
# Murphy decomposition
# ---------------------------------------------------------------------------
def test_decomposition_is_exact_when_predictions_are_constant_within_bins() -> None:
    """Brier = calibration - resolution + uncertainty is an identity only when each
    bin holds one distinct prediction. With continuous probabilities qcut binning
    makes it approximate, which is why `residual` is REPORTED rather than assumed --
    this test pins the exact case so a real algebra error cannot hide in the slack.

    The counts must be EXACTLY equal. With random counts the quantile edges land on
    a prediction value instead of between two, qcut folds two distinct predictions
    into one bin, and the identity stops holding -- which is the approximation the
    real audit lives with, not an algebra error."""
    rng = np.random.default_rng(0)
    p = np.repeat([0.2, 0.4, 0.6, 0.8], 1000)
    y = (rng.random(4000) < p).astype(int)

    out = brier_decomposition(p, y, n_bins=4)
    assert out["residual"] == pytest.approx(0.0, abs=1e-12)
    assert out["brier"] == pytest.approx(
        out["calibration"] - out["resolution"] + out["uncertainty"], abs=1e-12
    )
    assert out["refinement"] == pytest.approx(out["uncertainty"] - out["resolution"])


def test_a_perfectly_calibrated_predictor_has_near_zero_calibration_term() -> None:
    rng = np.random.default_rng(1)
    p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=40000)
    y = (rng.random(40000) < p).astype(int)

    out = brier_decomposition(p, y, n_bins=5)
    assert out["calibration"] < 1e-3
    assert out["resolution"] > 0.05, "a predictor this separating must show resolution"


def test_a_constant_predictor_has_zero_resolution_and_scores_the_uncertainty() -> None:
    """The floor every head in this project is really competing against."""
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.49).astype(int)
    p = np.full(5000, y.mean())

    out = brier_decomposition(p, y, n_bins=10)
    assert out["resolution"] == pytest.approx(0.0, abs=1e-12)
    assert out["calibration"] == pytest.approx(0.0, abs=1e-12)
    assert out["brier"] == pytest.approx(out["uncertainty"], abs=1e-12)


def test_uncertainty_depends_only_on_the_target() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(2000) < 0.4).astype(int)
    a = brier_decomposition(rng.random(2000) * 0.2 + 0.4, y, n_bins=10)
    b = brier_decomposition(rng.random(2000), y, n_bins=10)
    assert a["uncertainty"] == pytest.approx(b["uncertainty"])


# ---------------------------------------------------------------------------
# Reliability table + Spearman
# ---------------------------------------------------------------------------
def test_reliability_table_survives_a_near_constant_head() -> None:
    """A direction head whose output collapses onto a handful of values produces
    duplicate quantile edges. Fewer bins is the honest response; a crash here would
    make the audit unable to report exactly the pathology worth reporting."""
    p = np.repeat([0.5, 0.5001], 500)
    y = np.tile([0, 1], 500)
    table = reliability_table(p, y, n_bins=10)
    assert 1 <= len(table) <= 10
    assert table["n"].sum() == len(p)


def test_reliability_bins_partition_the_sample_and_gap_is_signed() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    table = reliability_table(p, y, n_bins=10)

    assert table["n"].sum() == 1000
    assert len(table) == 10
    np.testing.assert_allclose(table["gap"], table["mean_predicted"] - table["realised"])
    # bins are ordered, so their prediction ranges must not overlap
    assert (table["p_low"].to_numpy()[1:] >= table["p_high"].to_numpy()[:-1]).all()


def test_spearman_recovers_the_sign_of_the_confidence_correctness_link() -> None:
    """rho > 0 means confidence ranks correctness -- the assumption the 0.52 guard
    is built on. The inverted case must come back negative, or the statistic could
    not detect an anti-informative confidence."""
    rng = np.random.default_rng(5)
    n = 4000
    conf = rng.random(n) * 0.4                       # |p - 0.5| in [0, 0.4]
    correct = rng.random(n) < (0.5 + conf)           # more confident -> more often right
    p = np.where(rng.random(n) < 0.5, 0.5 + conf, 0.5 - conf)
    y = np.where(p >= 0.5, correct.astype(int), 1 - correct.astype(int))

    good = confidence_correctness_spearman(p, y)
    assert good["rho"] > 0.1 and good["pvalue"] < 1e-6

    flipped = confidence_correctness_spearman(p, 1 - y)
    assert flipped["rho"] < -0.1


def test_spearman_returns_nan_rather_than_raising_on_a_degenerate_slice() -> None:
    """The >=0.52 subsets can collapse to all-correct or single-valued confidence.
    NaN is information; an exception would abort the whole audit over one cell."""
    out = confidence_correctness_spearman(np.full(50, 0.6), np.ones(50, int))
    assert np.isnan(out["rho"])
    assert out["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------
def test_module_declares_the_live_threshold_by_import_not_by_copy() -> None:
    """If production moves the guard, the audit must follow it automatically."""
    assert CONFIDENCE_THRESHOLD is PredictionService.CONFIDENCE_THRESHOLD


def test_audit_module_never_fits_an_estimator() -> None:
    """A calibration audit that retrained anything would be measuring a model that
    is not the one being served. Guard the source text: no .fit( anywhere."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "calibration_audit.py"
    text = src.read_text(encoding="utf-8")
    assert ".fit(" not in text
    assert "fit_transform" not in text
    assert "fit_lag_pca" not in text


def test_dataframe_helpers_do_not_mutate_their_inputs() -> None:
    p = np.linspace(0.4, 0.6, 500)
    y = (np.arange(500) % 2).astype(int)
    p_before, y_before = p.copy(), y.copy()
    reliability_table(p, y)
    brier_decomposition(p, y)
    confidence_correctness_spearman(p, y)
    consensus_frame(p, y.astype(float))
    np.testing.assert_array_equal(p, p_before)
    np.testing.assert_array_equal(y, y_before)


def test_pandas_groupby_observed_default_does_not_drop_bins() -> None:
    """qcut returns integer codes (labels=False), not a Categorical -- so groupby
    cannot silently reintroduce empty categories. Pin it: a future switch to
    labelled bins would change the decomposition weights without any error."""
    p = np.linspace(0, 1, 1000)
    y = (np.arange(1000) % 2).astype(int)
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    assert bins.dtype.kind in "iu"
    assert reliability_table(p, y, 10)["n"].sum() == 1000
