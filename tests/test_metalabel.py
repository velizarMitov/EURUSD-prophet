"""
METALABEL family — tests.

The family's registered outcome is "every cell UNDERPOWERED — NO DECISION,
zero comparisons spent" (results/metalabel/PRE_REGISTRATION.md, 6082fe5).
These tests therefore pin three things: that the arithmetic behind that
outcome is exactly what was registered, that the guards which make it safe
actually fire, and that the module loads no model and touches no test block.
"""
import ast
import os
import re

import numpy as np
import pandas as pd
import pytest

from src import metalabel as M

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, 'src', 'metalabel.py')


# ── 1. split arithmetic reproduces the registered boundaries ──────────────────

def test_daily_split_reproduces_ablation_boundaries():
    """src/ablation.py::_canonical_split on the 8,605-row euro-era matrix."""
    s = M.daily_split(8605)
    assert (s['train_end'], s['val_end']) == (6023, 6884)
    assert s['n_val'] == 861
    assert s['n_test_reserved'] == 1721


def test_h1_split_reproduces_registered_counts():
    """split_purge_embargo on the 69,349 labelled bars: the registered
    h1_direction rows say n_train 48543, n_val 10378, test 10403."""
    s = M.h1_split(69349)
    assert (s['train_end'], s['val_end']) == (48544, 58946)
    assert s['n_train'] == 48543
    assert s['n_val'] == 10378
    assert s['n_test_reserved'] == 10403
    assert s['val_start_scored'] == 48544 + 24


# ── 2. the test block is unreachable ──────────────────────────────────────────

def test_guard_refuses_positional_index_into_daily_test_block():
    g = M.TestBlockGuard(last_allowed=6883, label='daily')
    g.check(np.arange(6023, 6884))                       # the whole arbiter is fine
    with pytest.raises(M.TestBlockTouchedError):
        g.check(np.arange(6023, 6885))                   # one past val_end


def test_guard_refuses_h1_timestamp_past_val_end():
    val_end_ts = pd.Timestamp('2024-11-19 04:00', tz='UTC')
    g = M.TestBlockGuard(last_allowed=val_end_ts, label='h1')
    g.check(pd.DatetimeIndex([val_end_ts]))
    with pytest.raises(M.TestBlockTouchedError):
        g.check(pd.DatetimeIndex([val_end_ts + pd.Timedelta(hours=1)]))


def test_validation_positions_never_reach_test_block_on_either_frequency():
    d = M.daily_split(8605)
    gd = M.TestBlockGuard(d['val_end'] - 1)
    idx = M.validation_positions(d, gd)
    assert idx.min() == 6023 and idx.max() == 6883 and len(idx) == 861

    h = M.h1_split(69349)
    gh = M.TestBlockGuard(h['val_end'] - 1)
    idx = M.validation_positions(h, gh)
    assert idx.min() == 48568 and idx.max() == 58945 and len(idx) == 10378
    assert gd.checks == 1 and gh.checks == 1


def test_guard_raises_not_warns_when_split_is_tampered():
    """An off-by-one that extends val_end into the test block must RAISE."""
    d = M.daily_split(8605)
    d['val_end'] += 1
    with pytest.raises(M.TestBlockTouchedError):
        M.validation_positions(d, M.TestBlockGuard(6883))


# ── 3. the module loads no model and fits nothing ─────────────────────────────

BANNED_METHODS = ('fit', 'fit_transform', 'predict', 'predict_proba', 'load_model', 'load')
BANNED_LIBS = ('joblib', 'xgboost', 'keras', 'tensorflow', 'torch', 'sklearn')
BANNED_MODULES = ('src.inference', 'src.paper_trading', 'src.h1_direction_model',
                  'src.volatility', 'src.features', 'src.live_data', 'api')


def assert_no_model_access(source, label='source'):
    """Raise AssertionError if `source` calls a fitting/loading method, calls
    load_model(), or imports a model library or a serving module.

    Checked on the AST, not the raw text, so a docstring that SAYS 'no .fit()
    is called' can neither trip it nor satisfy it. The original text-based
    version of this guard false-positived on exactly that docstring; the AST
    rewrite that replaced it had no test proving it still fires, which is why
    it is a function now -- the negative tests below feed it real violations."""
    tree = ast.parse(source)
    called_attrs, called_names, imported = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called_attrs.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
        elif isinstance(node, ast.Import):
            imported |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for banned in BANNED_METHODS:
        assert banned not in called_attrs, f'{label} calls .{banned}(): it must not touch a model'
    assert 'load_model' not in called_names, f'{label} calls load_model()'
    for lib in BANNED_LIBS:
        assert lib not in imported, f'{label} imports {lib}'
    for forbidden in BANNED_MODULES:
        assert forbidden not in imported, f'{label} imports {forbidden}'


def test_module_source_loads_no_model_and_calls_no_fit():
    assert_no_model_access(open(SRC, encoding='utf-8').read(), 'src/metalabel.py')


@pytest.mark.parametrize('violation, expected', [
    ('model = Thing()\nmodel.fit(X, y)\n', r'calls \.fit\(\)'),
    ('scaler.fit_transform(X)\n', r'calls \.fit_transform\(\)'),
    ('p = clf.predict_proba(X)[:, 1]\n', r'calls \.predict_proba\(\)'),
    ('obj = joblib.load(path)\n', r'calls \.load\(\)'),
    ('from keras.models import load_model\nm = load_model(p)\n', r'load_model|keras'),
    ('import xgboost as xgb\n', r'imports xgboost'),
    ('from src.inference import PredictionService\n', r'imports src\.inference'),
    ('import api\n', r'imports api'),
    ('def f():\n    return getattr(m, "x").fit()\n', r'calls \.fit\(\)'),
], ids=['fit', 'fit_transform', 'predict_proba', 'joblib.load', 'load_model',
        'import-xgboost', 'from-src.inference', 'import-api', 'nested-fit'])
def test_no_model_access_guard_fires_on_a_real_violation(violation, expected):
    """The guard must BITE. Each case is a genuine model touch of the kind the
    family must never make; a guard that only ever passes guards nothing."""
    with pytest.raises(AssertionError, match=expected):
        assert_no_model_access(violation, 'probe')


def test_no_model_access_guard_ignores_the_words_in_a_docstring():
    """The false positive that forced the AST rewrite, pinned: the words
    '.fit(' and 'load_model' in a string literal are not calls."""
    benign = (
        '"""This module never calls .fit() and never calls load_model().\n'
        'It also does not import joblib or src.inference."""\n'
        "BANNED = ('fit', 'load_model', 'joblib')\n"
        "def describe():\n"
        "    return 'no .fit() here'\n"
    )
    assert_no_model_access(benign, 'probe')          # must NOT raise


def test_no_model_access_guard_is_the_one_applied_to_the_module():
    """The negative tests above exercise assert_no_model_access; this pins that
    the positive test on src/metalabel.py goes through the same function, so
    the two cannot drift apart."""
    import inspect
    src = inspect.getsource(test_module_source_loads_no_model_and_calls_no_fit)
    assert 'assert_no_model_access(' in src


def test_module_never_names_a_models_dir_artifact():
    src = open(SRC, encoding='utf-8').read()
    assert not re.search(r"['\"]models/", src), 'a models/ path appears in src/metalabel.py'


# ── 4. power arithmetic is exactly what was registered ────────────────────────

def test_family_alpha_is_0_05_over_12_and_held():
    fam = M.family_arithmetic(base_dir=REPO)
    assert fam['feature_log_rows'] == 9
    assert fam['family_size'] == 12
    assert fam['alpha'] == pytest.approx(0.05 / 12)
    assert M.FAMILY_ALPHA == pytest.approx(0.004167, abs=5e-7)
    assert fam['alpha_if_c1_dropped'] > fam['alpha']          # dropping C1 WOULD loosen it
    assert fam['held_at_registered_size'] is True


@pytest.mark.parametrize('n1,n0,just_sig,p80', [
    (353,  508,  9.93, 12.84),     # C0
    (3585, 6793, 2.96, 3.83),      # C1 proposal (from H_dir.2 McNemar counts)
    (3459, 3459, 3.44, 4.46),      # C2
])
def test_mde_matches_registered_values(n1, n0, just_sig, p80):
    assert 100 * M.mde_two_proportion(n1, n0, power=None) == pytest.approx(just_sig, abs=0.01)
    assert 100 * M.mde_two_proportion(n1, n0) == pytest.approx(p80, abs=0.01)


def test_just_significant_is_smaller_than_80pct_power():
    assert M.mde_two_proportion(1000, 1000, power=None) < M.mde_two_proportion(1000, 1000)


def test_n_required_per_cell_for_three_pp_is_7634():
    assert M.n_required_per_cell(0.03) == 7634


def test_c0_years_to_resolve_match_registration():
    y3 = M.years_of_daily_data(861, 0.1284, 0.03)
    assert y3['n_val_needed'] == pytest.approx(15779, abs=40)
    assert y3['arbiter_years'] == pytest.approx(60, abs=1)
    assert y3['total_history_years'] == pytest.approx(605, abs=5)
    assert y3['shortfall_vs_existing'] == pytest.approx(22, abs=1)
    y2 = M.years_of_daily_data(861, 0.1284, 0.02)
    assert y2['total_history_years'] == pytest.approx(1360, abs=10)


def test_design_effect_is_measured_not_assumed():
    d = M.design_effect_from_registered_runs(base_dir=REPO)
    assert d['n_rows'] == 7                                    # seven accuracy rows carry both CIs
    assert 0.9 <= d['deff_median'] <= 1.2                      # direction has ~no autocorrelation
    assert d['deff_max'] <= 1.3


def test_power_table_marks_every_registered_cell_underpowered():
    t = M.power_table(deff=1.035)
    assert len(t) == 3
    assert (t['verdict'] == 'UNDERPOWERED — NO DECISION').all()
    assert (t['mde_80pct_power_pp'] > M.ECONOMIC_FLOOR_PP).all()
    assert (t['mde_80pct_power_x_deff_pp'] >= t['mde_80pct_power_pp']).all()
    assert (t['n_required_per_cell_for_floor'] == 7634).all()


# ── 5. rules ──────────────────────────────────────────────────────────────────

def test_mixed_sentinels_are_not_trades():
    d = ['UP', 'DOWN', 'MIXED / LOW CONFIDENCE', 'MIXED / TIE', 'UP']
    assert M.is_trade(d).tolist() == [True, True, False, False, True]


def test_variant_agreement_is_string_equality_including_both_mixed():
    a = ['UP', 'UP', 'MIXED / LOW CONFIDENCE', 'MIXED / LOW CONFIDENCE']
    b = ['UP', 'DOWN', 'MIXED / LOW CONFIDENCE', 'UP']
    assert M.variant_agreement(a, b).tolist() == [True, False, True, False]


def test_two_model_direction_disagreement_equals_mcnemar_discordance():
    """With two binary models, opposite calls <=> exactly one is right."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    pa = rng.integers(0, 2, 5000)
    pb = rng.integers(0, 2, 5000)
    n_disagree = int((~M.two_model_agreement(pa, pb)).sum())
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    assert n_disagree == b + c


def test_c1_proposal_cells_read_from_registered_h_dir2_row():
    log = pd.read_csv(os.path.join(REPO, 'results', 'h1_direction_hypothesis_log.csv'))
    r = log[log['hypothesis'].astype(str).str.startswith('H_dir.2')].iloc[0]
    cells = M.disagreement_cells_from_mcnemar(r['mcnemar_b'], r['mcnemar_c'], r['n_val'])
    assert cells['n_disagree'] == 3585 and cells['n_agree'] == 6793
    assert cells['share_disagree'] == pytest.approx(0.3454, abs=1e-3)


def test_tercile_boundaries_come_from_train_only():
    rng = np.random.default_rng(1)
    train = np.abs(rng.normal(0, 0.05, 20000))
    val = np.abs(rng.normal(0, 0.20, 5000))       # a very different distribution
    q33, q67 = M.tercile_boundaries(train)
    buckets = M.assign_terciles(val, q33, q67)
    assert set(np.unique(buckets)) <= {0, 1, 2}
    # boundaries are a function of TRAIN alone: recomputing them ignores val entirely
    assert M.tercile_boundaries(train) == (q33, q67)
    # and applying them to the wider val distribution does NOT give equal thirds
    counts = np.bincount(buckets, minlength=3)
    assert counts[2] > counts[0]


# ── 6. bootstrap refusal and alpha-governed width ─────────────────────────────

def test_block_bootstrap_refuses_when_n_le_block_len():
    cc = np.ones(24); cr = np.zeros(24)
    lo, hi, point = M.block_bootstrap_delta(cc, cr, block_len=24, n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)
    assert point == 1.0                                      # the point is still reported


def test_block_bootstrap_ci_widens_at_tighter_alpha():
    rng = np.random.default_rng(2)
    cc = rng.integers(0, 2, 2000).astype(float)
    cr = rng.integers(0, 2, 2000).astype(float)
    lo1, hi1, _ = M.block_bootstrap_delta(cc, cr, block_len=24, n_boot=300, alpha=0.05)
    lo2, hi2, _ = M.block_bootstrap_delta(cc, cr, block_len=24, n_boot=300, alpha=M.FAMILY_ALPHA)
    assert (hi2 - lo2) > (hi1 - lo1)
    assert lo2 <= 0 <= hi2                                    # random labels: covers zero


# ── 7. join sanity check, guarded ─────────────────────────────────────────────

def _daily_prices():
    p = pd.read_csv(os.path.join(REPO, 'results', 'eurusd_features.csv'), usecols=['time', 'close'])
    p['time'] = pd.to_datetime(p['time'])
    return p.set_index('time')


def test_reconstruct_as_of_close_agrees_sub_pip_on_the_validation_slice():
    prices = _daily_prices()
    val = prices.loc['2018-05-02':'2021-02-01']
    assert len(val) > 800
    guard = M.TestBlockGuard(pd.Timestamp('2021-02-01'), label='daily-date')
    rec = M.reconstruct_as_of_close(prices, val.index, guard=guard)
    worst = M.assert_close_agrees(val['close'].to_numpy(), rec.to_numpy())
    assert worst < M.PIP_TOLERANCE


def test_reconstruct_as_of_close_refuses_a_forward_log_date_in_the_test_block():
    """The forward log's as_of dates (2026) lie inside the daily test block, so
    the join sanity check against prediction_log.csv is REFUSED by the guard."""
    prices = _daily_prices()
    guard = M.TestBlockGuard(pd.Timestamp('2021-02-01'), label='daily-date')
    with pytest.raises(M.TestBlockTouchedError):
        M.reconstruct_as_of_close(prices, pd.DatetimeIndex(['2026-09-17']), guard=guard)


def test_assert_close_rejects_a_one_pip_disagreement():
    with pytest.raises(AssertionError):
        M.assert_close_agrees([1.1000], [1.1001])


# ── 8. registry ───────────────────────────────────────────────────────────────

def test_registry_schema_matches_feature_hypothesis_log():
    a = pd.read_csv(os.path.join(REPO, M.FEATURE_LOG))
    b = M.read_registry(base_dir=REPO)
    assert list(a.columns) == list(b.columns)


def test_registry_has_three_unspent_rows_with_blank_metrics():
    b = M.read_registry(base_dir=REPO)
    assert len(b) == 3
    assert b['notes'].str.contains('ZERO COMPARISONS SPENT').all()
    assert b[['point_delta_acc', 'ci95_dacc_low', 'ci95_dacc_high', 'mcnemar_p', 'cleared_bar']] \
        .isna().all().all()
    assert np.allclose(b['alpha_bonferroni'].to_numpy(dtype=float), 0.05 / 12, atol=1e-6)
    assert b['verdict'].str.contains('UNSPENT').all()
    assert b.loc[1, 'verdict'].startswith('INFEASIBLE-AS-SPECIFIED')


# ── 9. run() evaluates nothing ────────────────────────────────────────────────

def test_run_evaluates_nothing_and_reports_the_registered_verdict():
    res = M.run(base_dir=REPO, write=False)
    m = res['meta']
    assert m['comparisons_spent'] == 0
    assert m['cells_evaluated'] == 0
    assert m['models_loaded'] == 0
    assert m['fits_performed'] == 0
    assert m['registry_all_metrics_blank'] and m['registry_all_rows_unspent']
    assert 'NO DECISION' in m['verdict'] and 'INFEASIBLE' in m['verdict']
    assert m['daily_split']['val_end'] == 6884 and m['h1_split']['val_end'] == 58946


def test_run_refuses_if_feature_log_grew_after_registration(tmp_path):
    """If the standing family changes, alpha must NOT be silently re-derived."""
    (tmp_path / 'results').mkdir()
    src = pd.read_csv(os.path.join(REPO, M.FEATURE_LOG))
    pd.concat([src, src.iloc[[0]]]).to_csv(tmp_path / M.FEATURE_LOG, index=False)
    with pytest.raises(RuntimeError, match='family_size on disk'):
        M.run(base_dir=str(tmp_path), write=False)
