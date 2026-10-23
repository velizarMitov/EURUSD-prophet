"""Gold macro drivers (openspec change gold-forecasting-v2, spec gold-forecasting/macro-drivers)."""
import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.gold import features as F
from src.gold import macro as M
from src.gold import prereg2 as P2

CFG2 = P2.load_config_v2()


def _days(start, n):
    return pd.date_range(start, periods=n, freq='B', tz='UTC')


# ── 4.1 no look-ahead ────────────────────────────────────────────────────────

def test_monday_real_yield_comes_from_friday_or_earlier():
    fred = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.DatetimeIndex(
        ['2026-10-15', '2026-10-16', '2026-10-19', '2026-10-20'], tz='UTC'))       # Thu Fri Mon Tue
    gold_idx = pd.DatetimeIndex(['2026-10-19', '2026-10-20'], tz='UTC')           # Mon Tue
    got = M.align_fred(fred, gold_idx)
    assert got.iloc[0] == 2.0            # Monday sees Friday's value, never Monday's own
    assert got.iloc[1] == 3.0            # Tuesday sees Monday's


def test_a_missing_silver_bar_carries_the_past_value_forward():
    silver = pd.Series([30.0, 31.0, 33.0], index=pd.DatetimeIndex(['2026-10-19', '2026-10-20', '2026-10-22'], tz='UTC'))
    gold_idx = pd.DatetimeIndex(['2026-10-20', '2026-10-21', '2026-10-22'], tz='UTC')
    got = M.align_bars(silver, gold_idx)
    assert list(got) == [31.0, 31.0, 33.0]      # Wednesday: Tuesday's close, never Thursday's


def test_no_value_dated_after_the_gold_bar_is_ever_used():
    idx = _days('2024-01-01', 300)
    fred = pd.Series(np.arange(300, dtype=float), index=idx)
    got = M.align_fred(fred, idx)
    assert (got.dropna().to_numpy() < np.arange(300)[got.notna().to_numpy()]).all()


def test_features_are_scale_free_and_a_raw_level_is_refused():
    idx = _days('2024-01-01', 200)
    rng = np.random.default_rng(2)
    gold = pd.DataFrame({'close': 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))}, index=idx)
    walk = lambda s: pd.Series(s * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
    inputs = {'fred': {'dfii10': pd.Series(rng.normal(1.5, 0.1, 200), index=idx), 'dtwexbgs': walk(120.0),
                       'vixcls': pd.Series(rng.uniform(12, 30, 200), index=idx)},
              'mt5': {'silver': walk(25.0), 'brent': walk(80.0)}}
    feats = M.macro_features(gold, inputs)
    assert list(feats.columns) == M.MACRO_COLUMNS and feats.dropna().shape[0] > 100
    with pytest.raises(F.ScaleDependentFeature):
        F.assert_scale_free(M.MACRO_COLUMNS + ['close'])


def test_an_unreachable_source_names_the_series(tmp_path, monkeypatch):
    cfg2 = copy.deepcopy(CFG2)
    cfg2['paths']['macro_cache'] = str(tmp_path)
    import src.macro_data as MD
    monkeypatch.setattr(MD, '_fetch_raw_via_api', lambda *a, **k: None)
    monkeypatch.setattr(MD, '_fetch_raw_via_public', lambda *a, **k: None)
    with pytest.raises(M.MacroUnavailable, match='DFII10'):
        M.fred_series(cfg2, 'dfii10', 'DFII10', '2024-01-01', '2024-02-01')
    monkeypatch.setattr(M, 'fetch_mt5_d1', lambda *a, **k: None)
    with pytest.raises(M.MacroUnavailable, match='SILVER'):
        M.mt5_close(cfg2, 'silver')


def test_the_pinned_file_is_refused_once_changed(tmp_path):
    cfg2 = copy.deepcopy(CFG2)
    cfg2['paths']['macro_cache'] = str(tmp_path)
    idx = _days('2024-01-01', 10)
    pd.DataFrame({c: np.arange(10.0) for c in M.MACRO_COLUMNS}, index=idx).to_csv(M.pinned_path(cfg2))
    sha = M.features_digest(pd.read_csv(M.pinned_path(cfg2), index_col=0))
    assert M.load_pinned_features(cfg2, sha, idx).shape == (10, len(M.MACRO_COLUMNS))
    with open(M.pinned_path(cfg2), 'a') as fh:
        fh.write('2024-01-15 00:00:00+00:00' + ',1' * len(M.MACRO_COLUMNS) + '\n')
    with pytest.raises(M.MacroUnavailable, match='changed after registration'):
        M.load_pinned_features(cfg2, sha, idx)


# ── 4.2 identical rows, registered cells only ───────────────────────────────

def _fixture(n=900, seed=4):
    from src.gold import challengers as C
    from src.gold import splits as S
    from src.gold import targets as G
    from src.gold import load_config
    cfg = load_config()
    idx = _days('2020-01-01', n)
    rng = np.random.default_rng(seed)
    c = 1500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    d1 = pd.DataFrame({'open': c, 'high': c * 1.005, 'low': c * 0.995, 'close': c}, index=idx)
    inp = C.build_inputs(d1, 'D1')
    mac = rng.normal(0, 1, (n, len(M.MACRO_COLUMNS)))
    mac[:120] = np.nan                                  # macro starts later than gold
    mac[700, 3] = np.nan                                # and has one hole inside validation
    inp_b, inp_m = M.with_macro(inp, mac)
    tg = G.build_targets(d1['close'])
    pos = S.stage_positions(n, 'validation', cfg)
    preds = {}
    for name in ('d1_lstm', 'd1_gbm'):
        for v in ('base', 'macro'):
            preds[(name, v)] = {'p_up': np.full(n, 0.5), 'ret_pct': rng.normal(0, 0.01, n),
                                'vol_pct': np.abs(tg['vol']) * (0.9 if v == 'macro' else 1.1) + 0.05}
    return cfg, inp_b, inp_m, tg, pos, preds


def test_both_variants_are_marked_complete_on_the_same_rows():
    _, inp_b, inp_m, *_ = _fixture()
    assert np.array_equal(inp_b.extra['complete'], inp_m.extra['complete'])
    assert not inp_m.extra['complete'][700] and not inp_m.extra['complete'][:120].any()
    assert inp_m.X.shape[1] == inp_b.X.shape[1] + len(M.MACRO_COLUMNS)


def test_scoring_yields_exactly_the_registered_cells():
    cfg, inp_b, inp_m, tg, pos, preds = _fixture()
    fam = {'alpha': 0.0125, 'cells': list(P2.MACRO_CELLS)}
    rows = M.score_from_predictions(CFG2, fam, preds, inp_m, tg, pos, 200)
    assert sorted(r['cell'] for r in rows) == sorted(P2.MACRO_CELLS)
    vol = [r for r in rows if r['cell'] == 'd1_lstm_macro:vol'][0]
    from src.gold import challengers as C
    assert vol['n_rows'] == len(C.usable(inp_m, pos['score'], tg, ('vol',)))
    assert all(r['family'] == 'macro' and r['alpha_bonferroni'] == 0.0125 for r in rows)
