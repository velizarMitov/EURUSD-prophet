"""forecast_eval read-only feature adapters (tasks 6.1-6.2)."""
import glob
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def h1():
    return F.load_h1().iloc[-24 * 120:]          # ~120 sessions: fast, enough warm-up


@pytest.fixture(scope='module')
def daily_raw():
    return F.load_daily_ohlcv().iloc[-1500:]


def _macro(index):
    """Deterministic synthetic macro levels (no network in tests)."""
    idx = pd.to_datetime(index)
    idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    t = np.arange(len(idx), dtype=float)
    return pd.DataFrame({'yield_differential': 1 + 0.01 * np.sin(t / 7),
                         'usd_index': 100 + 0.1 * t,
                         'policy_rate_differential': 0.5 + 0 * t,
                         'inflation_differential': 0.2 + 0.001 * t}, index=idx)


# ── 6.1 parity with the production feature code ──────────────────────────

@pytest.mark.parametrize('feature_set', ['price', 'macro'])
def test_daily_features_equal_production(daily_raw, feature_set):
    from src.features import FEATURE_COLUMNS, add_advanced_features, merge_macro_features
    macro = _macro(daily_raw.index)
    ours = F.daily_features(daily_raw, macro, feature_set)
    prod = add_advanced_features(merge_macro_features(daily_raw, macro))
    common = ours.index.intersection(prod.index)
    assert len(common) > 1000
    cols = F.daily_columns(feature_set)
    pd.testing.assert_frame_equal(ours.loc[common, cols], prod.loc[common, cols])
    assert ours.index[-1] == daily_raw.index[-1], 'the newest bar must survive (no target drop)'


def test_both_feature_sets_share_one_row_set(daily_raw):
    macro = _macro(daily_raw.index)
    a, b = F.daily_features(daily_raw, macro, 'price'), F.daily_features(daily_raw, macro, 'macro')
    assert a.index.equals(b.index)
    assert len(F.daily_columns('price')) == 23 and len(F.daily_columns('macro')) == 27


def test_h1_bar_features_equal_production(h1):
    from src.h1_features import compute_h1_direction_features
    ours = F.h1_bar_features(h1)
    prod = compute_h1_direction_features(h1).dropna()
    pd.testing.assert_frame_equal(ours.drop(columns='close'), prod)
    assert ours.index[-1] == h1.index[-1]


def test_h1_daily_features_equal_production(h1):
    from src.h1_features import build_h1_datasets
    flat, seq, _close = F.h1_daily_features(h1)
    pflat, pseq, _r, _d, pidx = build_h1_datasets(h1=h1)
    pd.testing.assert_frame_equal(flat.loc[pidx], pflat)
    pos = [flat.index.get_loc(d) for d in pidx]
    assert np.array_equal(seq[pos], pseq)
    assert len(flat) == len(pflat) + 1, 'newest session kept'


def test_ti_sequences_equal_production(h1):
    from src.ti_lstm_h1_experimental import build_ti_datasets
    X, idx, _close = F.ti_daily_sequences(h1)
    pX, _r, _d, pidx = build_ti_datasets(h1=h1)
    pos = [idx.get_loc(d) for d in pidx]
    assert np.array_equal(X[pos], pX)
    from src.h1_features import MIN_HOURS
    counts = pd.Series(1, index=h1.index.normalize()).groupby(level=0).size()
    newest_complete = counts[counts >= MIN_HOURS].index[-1]
    assert idx[-1] == newest_complete, 'newest COMPLETE session kept'
    assert len(idx) == len(pidx) + 1


def test_unknown_feature_set_refused(daily_raw):
    with pytest.raises(KeyError):
        F.daily_features(daily_raw, None, 'everything')


# ── 6.2 digest ───────────────────────────────────────────────────────────

def test_digest_changes_when_a_feature_module_changes(tmp_path):
    base = tmp_path / 'repo'
    for rel in F.FEATURE_MODULES:
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(REPO, rel), dst)
    d1 = F.feature_code_digest(base=str(base))
    assert d1 == {**F.feature_code_digest(), '_combined': d1['_combined']} or \
        d1['_combined'] == F.feature_code_digest()['_combined']
    with open(base / 'src' / 'features.py', 'a', encoding='utf-8') as fh:
        fh.write('\n# probe\n')
    d2 = F.feature_code_digest(base=str(base))
    assert d2['src/features.py'] != d1['src/features.py']
    assert d2['_combined'] != d1['_combined']
    assert d2['src/h1_features.py'] == d1['src/h1_features.py']


def test_no_pinned_file_modified():
    for f in glob.glob(os.path.join(REPO, 'tests', 'fixtures', '*_protected_sha256.json')):
        for rel, dig in json.load(open(f)).items():
            if isinstance(dig, str):
                got = hashlib.sha256(open(os.path.join(REPO, rel), 'rb').read()).hexdigest()
                assert got == dig, f'pinned file modified: {rel}'
