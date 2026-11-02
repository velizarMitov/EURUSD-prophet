"""Deciles -> size and P(up) (openspec change tsfm-second-opinion, spec
"A forecast is a size and a probability read from quantiles", design D3)."""
import numpy as np
import pytest

from src.external.tsfm import forecast as Q

LEVELS = Q.LEVELS


def _uniform(lo, hi):
    """Deciles of a uniform distribution on [lo, hi]: its CDF is exactly linear."""
    return lo + LEVELS * (hi - lo)


def test_symmetric_forecast_scenario():
    """q25 / q75 at -0.10 / +0.10, symmetric around 0 -> size 0.10, P(up) 0.50."""
    d = _uniform(-0.2, 0.2)
    assert Q.cdf(d, -0.10) == pytest.approx(0.25)
    assert Q.cdf(d, 0.10) == pytest.approx(0.75)
    assert Q.size(d) == pytest.approx(0.10, abs=1e-9)
    assert Q.p_up(d) == pytest.approx(0.50)


def test_shifted_forecast_scenario():
    """Zero at the 40th percentile -> P(up) 0.60."""
    d = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]) + 0.1
    assert d[3] == pytest.approx(0.0)
    assert Q.p_up(d) == pytest.approx(0.60)


def test_heavily_skewed_case():
    """Almost all mass above zero: P(up) is clipped at 0.9, and the size is the
    median of |R|, which for an all-positive distribution is its median."""
    d = np.array([0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30])
    assert Q.p_up(d) == pytest.approx(0.9)
    assert Q.size(d) == pytest.approx(0.10, abs=1e-6)
    neg = -d[::-1]
    assert Q.p_up(neg) == pytest.approx(0.1)
    assert Q.size(neg) == pytest.approx(0.10, abs=1e-6)


def test_size_grows_as_the_deciles_widen():
    sizes = [Q.size(_uniform(-w, w)) for w in (0.05, 0.1, 0.2, 0.4)]
    assert all(b > a for a, b in zip(sizes, sizes[1:]))
    assert sizes[-1] == pytest.approx(0.2, abs=1e-9)


def test_cdf_extends_linearly_and_clips():
    d = _uniform(-0.2, 0.2)
    assert Q.cdf(d, -0.18) == pytest.approx(0.05)
    assert Q.cdf(d, -1.0) == 0.0
    assert Q.cdf(d, 1.0) == 1.0


def test_tied_deciles_do_not_break_the_cdf():
    d = np.array([-0.1, -0.1, -0.05, 0.0, 0.0, 0.0, 0.05, 0.1, 0.1])
    assert 0.0 <= Q.size(d) < 0.2
    assert 0.1 <= Q.p_up(d) <= 0.9


def test_closes_mode_matches_the_returns_mode():
    rng = np.random.default_rng(1)
    r = np.sort(rng.normal(0.01, 0.15, 9))
    last = 3897.45
    prices = last * np.exp(r / 100.0)
    back = Q.to_return_deciles(prices, 'closes', last)
    assert np.allclose(back, r, atol=1e-12)
    assert Q.summarise(back)['size_pct'] == pytest.approx(Q.summarise(r)['size_pct'], abs=1e-12)
    assert Q.summarise(back)['p_up'] == pytest.approx(Q.summarise(r)['p_up'], abs=1e-12)


def test_summarise_names_every_decile():
    s = Q.summarise(_uniform(-0.2, 0.2))
    assert [k for k in s if k.startswith('q')] == [f'q{i}' for i in range(10, 100, 10)]


def test_unknown_mode_and_missing_close_are_refused():
    with pytest.raises(ValueError):
        Q.to_return_deciles(_uniform(-1, 1), 'ticks')
    with pytest.raises(ValueError):
        Q.to_return_deciles(_uniform(-1, 1), 'closes')
