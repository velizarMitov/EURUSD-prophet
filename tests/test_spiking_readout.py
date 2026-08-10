"""Invariant tests for ``src/spiking_readout.py`` (H_spk.1 readout-ablation family).

The load-bearing one is ``test_readout_training_leaves_the_substrate_bit_identical``: the
entire design claims S1 and S2 differ ONLY in their readout, and that claim is what makes
delta_AURC attributable to the readout. If a gradient ever leaks into the substrate the
comparison silently becomes "two different models", which is the thing this family exists
to avoid.

RESEARCH ONLY. Nothing here asserts predictive power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

eqx = pytest.importorskip("equinox", reason="requires requirements-ltc.txt")
jax = pytest.importorskip("jax", reason="requires requirements-ltc.txt")
pytest.importorskip("optax", reason="requires requirements-ltc.txt")
import jax.numpy as jnp  # noqa: E402

from src import spiking_readout as S  # noqa: E402

SEQ, HID = 16, 8


def _ohlc_csv(tmp_path, n=400, with_gaps=True):
    """H1-shaped OHLC with weekend gaps and NO tick_volume column."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    if with_gaps:
        keep = ~((idx.dayofweek == 5) | ((idx.dayofweek == 6) & (idx.hour < 22)))
        idx = idx[keep]
    rng = np.random.default_rng(0)
    close = 1.2 + np.cumsum(rng.normal(0, 1e-4, len(idx)))
    frame = pd.DataFrame(
        {
            "time": idx,
            "open": close,
            "high": close + 2e-4,
            "low": close - 2e-4,
            "close": close,
        }
    )
    path = tmp_path / "inst_h1.csv"
    frame.to_csv(path, index=False)
    return path


def test_dataset_builds_without_tick_volume(tmp_path) -> None:
    """The arbiter instruments carry OHLC only. ltc_data.load_bars REJECTS such a file by
    design (its clock hypothesis needs ticks); this family must not, because its hypothesis
    is about the readout and price-only covariates are the honest input."""
    from src import ltc_data as L

    path = _ohlc_csv(tmp_path)
    with pytest.raises(ValueError, match="tick_volume"):
        L.load_bars(path)

    data = S.build_price_dataset(path)
    assert set(S.COVARIATES).issubset(data.columns)
    assert not any("tick" in c for c in data.columns)
    assert np.isfinite(data[list(S.COVARIATES)].to_numpy()).all()


def test_gap_indicator_marks_weekends_and_log_dt_is_finite(tmp_path) -> None:
    data = S.build_price_dataset(_ohlc_csv(tmp_path))
    gaps = data["is_gap"].to_numpy()
    assert set(np.unique(gaps)) <= {0.0, 1.0}
    assert 0 < gaps.mean() < 0.10, "gap flag should be rare but present"
    # every flagged bar really does follow a large jump
    assert (data.loc[gaps > 0, "dt"] > S.GAP_THRESHOLD).all()
    assert np.isfinite(data["log_dt"].to_numpy()).all()


def test_target_is_next_bar_and_has_no_lookahead(tmp_path) -> None:
    """Target is shift(-1) of the percent log return -- the project-wide convention."""
    data = S.build_price_dataset(_ohlc_csv(tmp_path))
    r = data["log_return_pct"].to_numpy()
    t = data["target_return_pct"].to_numpy()
    np.testing.assert_allclose(t[:-1], r[1:], rtol=1e-9)


def test_windows_end_inside_the_mask(tmp_path) -> None:
    data = S.build_price_dataset(_ohlc_csv(tmp_path))
    n = len(data)
    x = data[list(S.COVARIATES)].to_numpy()
    y = data["target_return_pct"].to_numpy()
    mask = np.zeros(n, bool)
    mask[n // 2 :] = True
    xw, yw, idx = S.make_windows(x, y, mask, seq_len=SEQ)
    assert mask[idx].all(), "a window ended outside its split"
    assert (idx >= SEQ - 1).all()
    np.testing.assert_allclose(xw[0, -1], x[idx[0]])


def test_readout_training_leaves_the_substrate_bit_identical(tmp_path) -> None:
    """THE control that makes this family's delta interpretable.

    S1 and S2 must sit on numerically identical mu/sigma. If any gradient reaches the GRU
    or the Gaussian head, delta_AURC stops measuring the readout and starts measuring two
    different models -- exactly the confound the fixed-substrate ladder exists to remove.
    """
    data = S.build_price_dataset(_ohlc_csv(tmp_path, n=900))
    x = data[list(S.COVARIATES)].to_numpy()
    y = data["target_return_pct"].to_numpy()
    dt = data["dt"].to_numpy()
    mask = np.ones(len(data), bool)
    xw, yw, _ = S.make_windows(x, y, mask, seq_len=SEQ)
    dw, _, _ = S.make_windows(dt.reshape(-1, 1), y, mask, seq_len=SEQ)
    dw = dw[..., 0]

    substrate = S.build_substrate(input_size=x.shape[-1], hidden=HID, seed=0)
    before = jax.vmap(substrate)(jnp.asarray(xw[:32]))

    _, _, _ = S.train_readout(
        substrate, xw, yw, dw, hidden=HID, epochs=2, batch=16, lr=1e-2,
        coverage_floor=0.1, seed=0, stop_windows=(xw[:32], yw[:32], dw[:32]),
        verbose=False,
    )

    after = jax.vmap(substrate)(jnp.asarray(xw[:32]))
    for b, a, nm in zip(before, after, ("mu", "sigma", "hidden")):
        np.testing.assert_array_equal(
            np.asarray(b), np.asarray(a), err_msg=f"substrate {nm} changed during S2"
        )


def test_substrate_has_no_continuous_time_machinery() -> None:
    """S0's substrate must be a plain GRU: no warp, no tau, no gauge freedom. If any of
    those reappear, this family stops being an isolation of the readout."""
    substrate = S.build_substrate(input_size=len(S.COVARIATES), hidden=HID, seed=0)
    names = {f for f in dir(substrate) if not f.startswith("_")}
    assert "warp" not in names
    assert "cell" in names and isinstance(substrate.cell, eqx.nn.GRUCell)
    leaves = jax.tree_util.tree_leaves(eqx.filter(substrate, eqx.is_inexact_array))
    assert leaves, "substrate has no trainable parameters"


def test_prereg_matches_the_registered_row() -> None:
    """The log row quotes the hyperparameters. If PREREG drifts from what was registered,
    the run stops being the one that was pre-registered."""
    S.init_hypothesis_log()
    row = pd.read_csv(S.SPK_LOG).iloc[0]
    notes = str(row["notes"])
    assert row["verdict"] in ("PENDING",) or row["date"]
    for token in ("seq_len=64", "hidden=32", "lr=0.003", "batch=64",
                  "coverage_floor=0.10", "block_len=24", "n_boot=2000", "seed=20260810"):
        assert token in notes, f"{token} missing from the registered notes"
    assert float(row["alpha_bonferroni"]) == 0.05
    assert S.PREREG["seq_len"] == 64 and S.PREREG["hidden"] == 32
    assert S.PREREG["block_len"] == 24 and S.PREREG["n_boot"] == 2000


def test_observation_row_spends_no_alpha() -> None:
    """Row 2 records a defect; it must not enter the alpha ladder, or it would silently
    tighten the bar for the one hypothesis actually being tested."""
    S.init_hypothesis_log()
    row = pd.read_csv(S.SPK_LOG).iloc[1]
    assert "OBSERVATION" in str(row["alpha_bonferroni"])
    assert "hypothesis test" not in str(row["verdict"]).lower() or "not a" in str(row["verdict"]).lower()
