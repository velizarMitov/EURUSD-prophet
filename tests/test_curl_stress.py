"""Invariant tests for ``src/curl_stress.py`` (Idea-2 step-1 curl measurement).

These are the curl family's equivalent of the project's no-look-ahead tests: they pin the
properties that make the metric mean what the plan doc says it means. Every one of them
corresponds to a bug that was actually hit while building the module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import curl_stress as cs
from src import curl_null_simulation as sim


@pytest.fixture(scope="module")
def async_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    df, _ = sim.simulate_bars(1_500, 900, rng=rng)
    return df


def test_graph_shape() -> None:
    assert len(cs.PAIRS) == 6
    assert len(cs.TRIANGLES) == 4
    # K4 cycle space has dimension E - V + 1 = 3: the 4 triangles are DEPENDENT.
    assert len(cs.CYCLE_BASIS) == 6 - 4 + 1 == 3


def test_simultaneous_prices_give_zero_curl() -> None:
    """Orientation / inversion algebra. A sign error here fabricates enormous curl."""
    frame = sim.simultaneous_frame(500, rng=np.random.default_rng(3), consistent=True)
    assert cs.curl_frame(frame).abs().to_numpy().max() < 1e-12


def test_inconsistent_quote_levels_produce_constant_curl() -> None:
    """A level/convention mismatch is a CONSTANT offset, not microstructure stress."""
    frame = sim.simultaneous_frame(500, rng=np.random.default_rng(3), consistent=False)
    curls = cs.curl_frame(frame)
    assert curls.abs().to_numpy().max() > 1e-4  # tens of basis points
    # near-zero variance is the signature that separates it from staleness noise
    assert curls.std().max() < 1e-8
    assert cs.convention_check(frame)["t_stat"].abs().max() > 50.0


def test_cycle_rank_is_three(async_frame: pd.DataFrame) -> None:
    """Guards against inverting a singular 4x4 covariance of the triangle curls."""
    assert cs.cycle_rank(async_frame) == 3


def test_curl_is_mean_zero_under_asynchronicity(async_frame: pd.DataFrame) -> None:
    check = cs.convention_check(async_frame)
    assert check["mean_curl_bp"].abs().max() < 0.2
    assert check["rms_curl_bp"].min() > 0.0


def test_alignment_shift_inflates_curl(async_frame: pd.DataFrame) -> None:
    """One bar of misalignment must dominate a few seconds of staleness."""
    assert cs.shift_placebo(async_frame, ("EUR", "USD"), shift=1) > 3.0


def test_align_bars_rejects_missing_pair() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "tick_volume": 5},
        index=idx,
    )
    with pytest.raises(ValueError):
        cs.align_bars({("EUR", "USD"): frame})


def test_align_bars_intersects_timestamps() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    frames = {}
    for i, pair in enumerate(cs.PAIRS):
        sub = idx[:-1] if i == 0 else idx  # one pair is missing its final bar
        frames[pair] = pd.DataFrame(
            {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "tick_volume": 5.0},
            index=sub,
        )
    out = cs.align_bars(frames)
    assert len(out) == 19  # the unmatched bar is DROPPED, never forward-filled


def test_truncation_factor_bounds() -> None:
    n = np.array([1.0, 2.0, 10.0, 100.0, 1e4])
    f = cs._truncation_factor(n)
    assert np.all((f > 0) & (f <= 1.0))
    assert f[0] < 0.6  # a one-tick bar cannot be stale by more than the bar length
    assert f[-1] == pytest.approx(1.0)
    assert np.all(np.diff(f) >= 0)  # non-decreasing (clamped to 1 above 30 ticks)
    assert np.all(np.diff(cs._truncation_factor(np.arange(1.0, 25.0))) > 0)


def test_infeasibility_is_nonnegative_and_quiet_on_clean_data(
    async_frame: pd.DataFrame,
) -> None:
    g = cs.triangle_infeasibility(async_frame, cs.CYCLE_BASIS[0], buffer_log=1e-5)
    assert np.all(g >= 0.0)
    # no genuine dislocation was injected, so the certificate must essentially never fire
    assert (g > 0).mean() < 0.02


def test_infeasibility_fires_on_large_dislocation() -> None:
    mask = np.ones(400, dtype=bool)
    df, _ = sim.simulate_bars(
        400,
        900,
        rng=np.random.default_rng(5),
        dislocated_pair=("EUR", "GBP"),
        dislocation_log=50e-4,
        dislocation_mask=mask,
    )
    g = np.column_stack(
        [cs.triangle_infeasibility(df, t, buffer_log=1e-5) for t in cs.CYCLE_BASIS]
    ).max(axis=1)
    assert (g > 0).mean() > 0.8


def test_null_calibration_slope_matches_theory(async_frame: pd.DataFrame) -> None:
    """Binned slope of c^2 on P must land near the theoretical 0.5."""
    tri = ("EUR", "USD", "JPY")
    c = cs.triangle_curl(async_frame, tri)
    p = cs.staleness_null(async_frame, tri)
    train = np.ones(len(async_frame), dtype=bool)
    _, slope = cs.calibrate_null(c, p, train_mask=train)
    assert 0.25 < slope < 0.9


def test_excess_curl_is_calibrated_in_sample(async_frame: pd.DataFrame) -> None:
    """E[X] = 1 under the null on the rows the null was fitted to."""
    tri = ("EUR", "USD", "JPY")
    c = cs.triangle_curl(async_frame, tri)
    p = cs.staleness_null(async_frame, tri)
    train = np.ones(len(async_frame), dtype=bool)
    a, b = cs.calibrate_null(c, p, train_mask=train)
    x = cs.excess_curl(c, p, a, b)
    assert 0.8 < np.nanmean(x[train]) < 1.25


def test_static_calibration_drifts_out_of_sample(async_frame: pd.DataFrame) -> None:
    """Documents a REAL limitation rather than papering over it.

    Under a true null, the out-of-sample mean of X is still allowed to wander a long way
    from 1.0 because volatility/liquidity regimes are persistent. Anyone reading a static
    excess-curl level as "stress" without a causal refit is reading regime drift.
    """
    tri = ("EUR", "USD", "JPY")
    c = cs.triangle_curl(async_frame, tri)
    p = cs.staleness_null(async_frame, tri)
    train = np.zeros(len(async_frame), dtype=bool)
    train[: int(0.7 * len(async_frame))] = True
    a, b = cs.calibrate_null(c, p, train_mask=train)
    x = cs.excess_curl(c, p, a, b)
    assert 0.2 < np.nanmean(x[~train]) < 5.0  # deliberately wide: drift is expected


def test_causal_excess_curl_has_no_lookahead(async_frame: pd.DataFrame) -> None:
    """Poisoning FUTURE curl must not change any earlier causal value."""
    tri = ("EUR", "USD", "JPY")
    c = cs.triangle_curl(async_frame, tri)
    p = cs.staleness_null(async_frame, tri)
    kw = {"min_train": 600, "refit_every": 200}
    base = cs.causal_excess_curl(c, p, **kw)
    poisoned = c.copy()
    poisoned[1_000:] *= 50.0
    after = cs.causal_excess_curl(poisoned, p, **kw)
    np.testing.assert_allclose(base[:1_000], after[:1_000], rtol=1e-10)
    assert np.isnan(base[: kw["min_train"]]).all()


def test_calibration_uses_train_rows_only() -> None:
    """Poisoning the post-train rows must not move the fitted coefficients."""
    df, _ = sim.simulate_bars(1_000, 900, rng=np.random.default_rng(9))
    tri = ("EUR", "USD", "JPY")
    train = np.zeros(len(df), dtype=bool)
    train[:700] = True
    c = cs.triangle_curl(df, tri)
    p = cs.staleness_null(df, tri)
    base = cs.calibrate_null(c, p, train_mask=train)
    poisoned = c.copy()
    poisoned[700:] *= 100.0
    assert cs.calibrate_null(poisoned, p, train_mask=train) == base


def test_stress_index_separates_injected_dislocation() -> None:
    mask = np.zeros(2_000, dtype=bool)
    mask[1_600:1_900] = True  # episode sits OUTSIDE the calibration slice
    df, _ = sim.simulate_bars(
        2_000,
        900,
        rng=np.random.default_rng(13),
        dislocated_pair=("EUR", "GBP"),
        dislocation_log=10e-4,
        dislocation_mask=mask,
    )
    train = np.zeros(2_000, dtype=bool)
    train[:1_400] = True
    st = cs.stress_index(df, train_mask=train, smooth=32)["stress_smooth"].to_numpy()
    fin = np.isfinite(st)
    assert np.nanmean(st[fin & mask]) > np.nanmean(st[fin & ~mask]) + 1.0
