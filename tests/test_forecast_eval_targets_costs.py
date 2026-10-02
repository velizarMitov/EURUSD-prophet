"""forecast_eval targets, costs and breakeven (tasks 2.1-2.3)."""
import ast
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import costs as C
from src.forecast_eval import targets as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _close(values):
    idx = pd.date_range('2024-01-01', periods=len(values), freq='h', tz='UTC')
    return pd.Series(values, index=idx, dtype=float)


# ── 2.1 targets ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('h', [1, 24])
def test_direction_target_reads_only_close_t_plus_h(h):
    rng = np.random.default_rng(0)
    close = _close(1.1 + np.cumsum(rng.normal(0, 1e-3, 200)))
    y, _ = T.direction_target(close, h)
    expected = np.sign(np.log(close.shift(-h)) - np.log(close))
    mask = y.notna()
    assert ((y[mask] == 1) == (expected[mask] > 0)).all()
    assert y.iloc[-h:].isna().all() and y.iloc[:-h].notna().all()


def test_target_at_t_is_unchanged_by_data_after_t_plus_h():
    close = _close(np.linspace(1.0, 1.2, 50))
    y1, _ = T.direction_target(close, 4)
    tampered = close.copy()
    tampered.iloc[30:] = 9.99
    y2, _ = T.direction_target(tampered, 4)
    pd.testing.assert_series_equal(y1.iloc[:26], y2.iloc[:26])


def test_zero_moves_are_excluded_and_counted():
    close = _close([1.0, 1.0, 1.1, 1.1, 1.0, 1.0])
    y, n_zero = T.direction_target(close, 1)
    assert n_zero == 3
    assert y.isna().sum() == 4          # 3 zero moves + the last bar
    assert list(y.dropna()) == [1.0, 0.0]


def test_return_and_volatility_targets_are_percent_log_returns():
    close = _close([1.0, 1.01, 0.99])
    r = T.return_pct_target(close, 1)
    assert r.iloc[0] == pytest.approx(100 * np.log(1.01))
    assert T.volatility_pct_target(close, 1).iloc[1] == pytest.approx(abs(100 * np.log(0.99 / 1.01)))


def test_horizon_must_be_positive_integer():
    with pytest.raises(ValueError):
        T.log_return(_close([1, 2, 3]), 0)


def _multiplications_by_100(source):
    hits = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            for side in (n.left, n.right):
                if isinstance(side, ast.Constant) and side.value in (100, 100.0):
                    hits.append(n.lineno)
    return hits


def test_package_scales_by_100_only_inside_targets():
    for path in glob.glob(os.path.join(REPO, 'src', 'forecast_eval', '**', '*.py'), recursive=True):
        hits = _multiplications_by_100(open(path, encoding='utf-8').read())
        if os.path.basename(path) == 'targets.py':
            assert len(hits) == 1, f'targets.py must scale exactly once, found lines {hits}'
        else:
            assert not hits, f'{os.path.relpath(path, REPO)} scales by 100 at lines {hits}'


def test_scaling_guard_bites():
    assert _multiplications_by_100('y = r * 100\n') and _multiplications_by_100('y = 100.0 * r\n')


# ── 2.2 costs ────────────────────────────────────────────────────────────

def test_measured_spread_eurusd_is_half_a_pip():
    assert C.measured_spread_price('EURUSD') == pytest.approx(5e-5)


def test_config_round_trip_is_read_from_config():
    pips = json.load(open(os.path.join(REPO, 'config.json')))['paper_trading']['spread_pips']
    assert C.config_round_trip_price('EURUSD') == pytest.approx(pips * 1e-4)
    assert [lv.label for lv in C.cost_levels('EURUSD')] == ['measured', 'config_round_trip']


def test_cost_only_on_position_change():
    pos = [1, 1, 1, -1, -1, 0, 0, 1]
    assert list(C.position_changes(pos)) == [True, False, False, True, False, False, False, True]
    r = C.net_pnl(pos, [1.0] * 8, cost_price=0.1)
    assert r['n_trades'] == 3
    assert r['cost_total'] == pytest.approx(0.3)
    assert r['net_total'] == pytest.approx(r['gross_total'] - 0.3)


def test_gross_and_net_both_reported():
    r = C.net_pnl([1, -1], [2.0, -1.0], cost_price=0.5)
    assert {'gross_total', 'net_total', 'gross_per_trade', 'net_per_trade'} <= set(r)
    assert r['gross_total'] == pytest.approx(3.0) and r['net_total'] == pytest.approx(2.0)


def test_missing_spread_refuses(tmp_path):
    t = tmp_path / 'cov.csv'
    t.write_text('symbol,median_spread_pts\nEURUSD,5\n', encoding='utf-8')
    with pytest.raises(C.MissingSpreadError, match='XAUUSD'):
        C.measured_spread_price('XAUUSD', str(t))
    with pytest.raises(C.MissingSpreadError):
        C.measured_spread_price('EURUSD', str(tmp_path / 'absent.csv'))


def test_rollovers_and_swap():
    ts = pd.Timestamp
    assert C.rollovers_crossed(ts('2026-01-05 20:00', tz='UTC'), ts('2026-01-05 21:30', tz='UTC'), 22) == 0
    assert C.rollovers_crossed(ts('2026-01-05 20:00', tz='UTC'), ts('2026-01-06 01:00', tz='UTC'), 22) == 1
    assert C.rollovers_crossed(ts('2026-01-05 20:00', tz='UTC'), ts('2026-01-08 23:00', tz='UTC'), 22) == 4
    assert C.swap_charge(+1, 2, swap_long=-0.3, swap_short=0.1) == pytest.approx(0.6)
    assert C.swap_charge(-1, 1, swap_long=-0.3, swap_short=0.1) == pytest.approx(-0.1)
    assert C.swap_charge(0, 3, -0.3, 0.1) == 0.0


# ── 2.3 breakeven: reproduces the design calculation ──────────────────────

def test_breakeven_formula():
    assert T.breakeven_accuracy(10.0, 0.0) == 0.5
    assert T.breakeven_accuracy(10.0, 2.0) == pytest.approx(0.6)


def test_breakeven_reproduces_the_design_calculation():
    """design.md / proposal.md: on the EURUSD train slice [0:70%] of the
    pooled H1 cache, h = 1 breaks even at 53.32 % (0.5 pip) and 59.97 % (1.5 pip)."""
    path = os.path.join(REPO, 'results', 'pooled_h1', 'EURUSD_h1.csv')
    if not os.path.exists(path):
        pytest.skip('pooled H1 cache absent')
    df = pd.read_csv(path)
    close = df['close'].iloc[:int(len(df) * 0.70)].reset_index(drop=True)
    assert 100 * T.breakeven_for_horizon(close, 1, 0.5e-4) == pytest.approx(53.32, abs=0.01)
    assert 100 * T.breakeven_for_horizon(close, 1, 1.5e-4) == pytest.approx(59.97, abs=0.01)
    assert 100 * T.breakeven_for_horizon(close, 24, 0.5e-4) == pytest.approx(50.61, abs=0.01)
