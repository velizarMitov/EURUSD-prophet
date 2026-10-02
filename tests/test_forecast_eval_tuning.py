"""Tuning rule (task 7.9)."""
import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import overfit as O
from src.forecast_eval import record as R
from src.forecast_eval import targets as T
from src.forecast_eval import tuning as TU
from src.forecast_eval.challengers import make

RULE = {'min_net_improvement_pct': 10.0, 'max_pbo': 0.5, 'inner_folds': 2}


def test_rule_selects_exactly_one():
    assert TU.select_variant(1.0, 1.2, 0.2, RULE)[0] == 'tuned'
    assert TU.select_variant(1.0, 1.05, 0.2, RULE)[0] == 'frozen'      # < 10 %
    assert TU.select_variant(1.0, 1.5, 0.7, RULE)[0] == 'frozen'       # PBO too high
    assert TU.select_variant(-1.0, -0.85, 0.1, RULE)[0] == 'tuned'     # improves a loss
    assert TU.select_variant(1.0, np.nan, 0.1, RULE)[0] == 'frozen'


def test_undeclared_grid_value_is_refused():
    grid = {'max_depth': [3, 4]}
    with pytest.raises(TU.UndeclaredGridValue):
        TU.with_params({'gbm': {'max_depth': 3}}, {'max_depth': 9}, grid)
    with pytest.raises(TU.UndeclaredGridValue):
        TU.with_params({'gbm': {}}, {'n_estimators': 50}, grid)


def test_inner_tune_and_both_variants_logged(tmp_path):
    rec = R.new_record('t')
    frozen = rec['challengers']['h1_gbm']
    grid = {'max_depth': [3, 4], 'learning_rate': [0.05, 0.1]}
    h1 = F.load_h1().iloc[-24 * 60:]
    m = make('h1_gbm', frozen)
    inp = m.build_inputs(h1)
    s = pd.Series(inp.close)
    d, _ = T.direction_target(s, 1)
    tg = {'dir': d.to_numpy()}
    res = TU.inner_tune('h1_gbm', frozen, grid, inp, np.arange(900), tg, h=1, seed=1, inner_folds=2)
    assert res['n_candidates'] == 4
    assert res['params']['max_depth'] in grid['max_depth']
    selected, _ = TU.select_variant(0.10, 0.12, 0.3, RULE)
    rows = TU.variant_trials('t', 'h1_gbm', 'H1', 1, frozen, res['config'], 0.01, 0.02, 500, selected)
    log = str(tmp_path / 'trials.csv')
    assert O.append_trials(rows, log) == 2
    logged = O.read_trials(log)
    assert sorted(r['variant'] for r in logged) == ['frozen', 'tuned']
    assert sum(r['notes'] == 'SELECTED' for r in logged) == 1
