"""Round-two gold registration (openspec change gold-forecasting-v2, spec
gold-forecasting/registration-amendments)."""
import copy
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.gold import direction2 as D2
from src.gold import load_config, repo_path
from src.gold import prereg as P
from src.gold import prereg2 as P2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config()
CFG2 = P2.load_config_v2()
PW = CFG['power']


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ── the power gate ──────────────────────────────────────────────────────────

def test_underpowered_cells_are_recorded_and_alpha_follows_the_admitted_size():
    counts = {'n_h1_dir': 6376, 'n_barrier': 4487, 'barrier_timeouts': 685, 'n_h4': 1219, 'n_d1_dir': 346}
    admitted, alpha, table, history, floor = P2.direction2_family(counts, CFG2, PW['edge'], PW['power'], PW['family_alpha'])
    status = {r['cell']: r['status'] for r in table}
    assert admitted == ['h1_selective', 'h1_barrier']
    assert status['h4_dir'] == status['d1_gbm_macro:dir'] == P2.UNDERPOWERED
    assert alpha == pytest.approx(PW['family_alpha'] / 2)
    assert floor >= CFG2['study']['direction2']['selective']['floor_min']
    assert history[0]['family_size'] == len(P2.DIRECTION2_CELLS)


def test_the_power_rule_is_round_ones_and_never_loosened():
    """A family where nothing is admitted spends no alpha and gains none."""
    counts = {'n_h1_dir': 100, 'n_barrier': 100, 'barrier_timeouts': 0, 'n_h4': 100, 'n_d1_dir': 100}
    admitted, alpha, table, *_ = P2.direction2_family(counts, CFG2, PW['edge'], PW['power'], PW['family_alpha'])
    assert admitted == [] and all(r['status'] == P2.UNDERPOWERED for r in table)
    assert table[0]['n_required'] >= 2640


def test_the_selective_floor_rises_only_to_meet_the_requirement():
    counts = {'n_h1_dir': 4000, 'n_barrier': 0, 'barrier_timeouts': 0, 'n_h4': 0, 'n_d1_dir': 0}
    admitted, alpha, table, _, floor = P2.direction2_family(counts, CFG2, PW['edge'], PW['power'], PW['family_alpha'])
    assert admitted == ['h1_selective'] and alpha == PW['family_alpha']
    row = [r for r in table if r['cell'] == 'h1_selective'][0]
    assert row['n_validation'] >= row['n_required'] - 1 and floor == pytest.approx(row['n_validation'] / 4000, abs=1e-3)


# ── the test block stays spent ──────────────────────────────────────────────

def test_a_test_block_request_is_refused():
    with pytest.raises(P2.StageRefused, match='spent in round one'):
        P2.check_stage_allowed('test')


# ── generate: once, round one untouched ─────────────────────────────────────

@pytest.fixture
def cfg2(tmp_path):
    c = copy.deepcopy(CFG2)
    c['paths']['prereg_json'] = str(tmp_path / 'PRE_REGISTRATION_V2.json')
    c['paths']['prereg_md'] = str(tmp_path / 'PRE_REGISTRATION_V2.md')
    c['paths']['registries'] = {k: str(tmp_path / os.path.basename(v)) for k, v in c['paths']['registries'].items()}
    return c


def _round_one_files():
    paths = [repo_path(CFG['paths']['prereg_json']), repo_path(CFG['paths']['prereg_md']),
             os.path.join(REPO, 'src', 'gold', 'gold_config.json')]
    paths += [repo_path(p) for p in CFG['paths']['registries'].values()]
    return {p: _sha(p) for p in paths if os.path.exists(p)}


def _tiny_frames():
    idx = pd.date_range('2024-01-01', periods=3000, freq='h', tz='UTC')
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, len(idx))))
    h1 = pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close': c}, index=idx)
    d = h1['close'].resample('D').last()
    d1 = pd.DataFrame({'open': d, 'high': d * 1.002, 'low': d * 0.998, 'close': d})
    return {'H1': h1, 'D1': d1}


def test_generate_writes_once_and_leaves_round_one_byte_identical(cfg2):
    reg1 = P.registration(CFG)
    if reg1 is None:
        pytest.skip('no round-one registration')
    before = _round_one_files()
    reg = P2.generate(CFG, cfg2, frames=_tiny_frames(), macro_pin={'features_sha256': 'x'})
    assert _round_one_files() == before
    assert reg['disclosure'] == P2.DISCLOSURE and 'seen' in reg['disclosure']
    assert reg['round_one']['families_untouched'] == ['direction', 'return', 'volatility']
    assert set(reg['families']) == {'range', 'macro', 'direction2'}
    assert reg['families']['range']['alpha'] == pytest.approx(0.025)
    assert reg['families']['macro']['alpha'] == pytest.approx(0.0125)
    for path in cfg2['paths']['registries'].values():
        assert open(path, encoding='utf-8').read().startswith(','.join(P.REGISTRY_FIELDS))
    md = open(cfg2['paths']['prereg_md'], encoding='utf-8').read()
    assert 'needs forward confirmation' in md and 'test block [80:100] was spent' in md
    with pytest.raises(P2.StageRefused, match='written once'):
        P2.generate(CFG, cfg2, frames=_tiny_frames(), macro_pin={'features_sha256': 'x'})


# ── the scoring gate ────────────────────────────────────────────────────────

def test_scoring_refuses_an_uncommitted_registration(cfg2):
    with open(cfg2['paths']['prereg_json'], 'w') as fh:
        json.dump({'config_v2_study_sha256': P2.study_digest(cfg2)}, fh)
    assert P2.scoring_allowed_v2(cfg2) is False
    with pytest.raises(P2.StageRefused, match='uncommitted or modified'):
        P2.check_stage_allowed('validation', cfg2)


def test_scoring_refuses_changed_study_settings(cfg2, monkeypatch):
    with open(cfg2['paths']['prereg_json'], 'w') as fh:
        json.dump({'config_v2_study_sha256': P2.study_digest(cfg2)}, fh)
    monkeypatch.setattr(P2.FP, 'committed_and_clean', lambda path, repo=None: 'abc123')
    assert P2.check_stage_allowed('validation', cfg2)['config_v2_study_sha256'] == P2.study_digest(cfg2)
    changed = copy.deepcopy(cfg2)
    changed['study']['range']['gamma'] = 0.01
    with pytest.raises(P2.StageRefused, match='changed after registration'):
        P2.check_stage_allowed('validation', changed)


def test_the_alert_threshold_is_outside_the_pinned_settings():
    c = copy.deepcopy(CFG2)
    c['alerts']['threshold_vol_pct'] = 0.3
    assert P2.study_digest(c) == P2.study_digest(CFG2)


# ── direction2 label builders (used by the registration counts) ─────────────

def test_barrier_labels_use_only_the_next_k_consecutive_hours():
    idx = pd.date_range('2024-01-01', periods=40, freq='h', tz='UTC')
    close = pd.Series(100.0, index=idx)
    close.iloc[25] = 103.0                                  # a jump 5 hours after bar 20
    df = pd.DataFrame({'close': close})
    lab = D2.barrier_labels(df.assign(close=close * np.exp(np.linspace(0, 1e-4, 40))), k=6, m=1.0, span=5)
    assert lab['label'].iloc[20] == 1.0                     # upper barrier first
    assert np.isnan(lab['label'].iloc[38])                  # no full window ahead
    gap = df.drop(idx[30])
    lab2 = D2.barrier_labels(gap.assign(close=gap['close'] * 1.0), k=6, m=1.0, span=5)
    assert not lab2['eligible'].iloc[26]                    # its window spans the missing hour


def test_barrier_purge_drops_labels_reaching_past_the_training_end():
    kept = D2.barrier_purge(100, 6, np.arange(100))
    assert kept.max() == 93


def test_h4_windows_need_every_hour():
    idx = pd.date_range('2024-01-01 00:00', periods=48, freq='h', tz='UTC')
    df = pd.DataFrame({'close': np.linspace(100, 101, 48)}, index=idx).drop(idx[5])
    w = D2.h4_windows(df)
    assert pd.Timestamp('2024-01-01 04:00', tz='UTC') not in set(w['start'])
    assert bool(w.loc[w['start'] == pd.Timestamp('2024-01-01 00:00', tz='UTC'), 'eligible'].iloc[0]) is False


def test_tau_keeps_the_floor_on_training_predictions():
    p = np.random.default_rng(3).uniform(0.3, 0.7, 1000)
    tau = D2.choose_tau(p, 0.45)
    covered = np.isfinite(D2.selective(p, tau))
    assert covered.mean() >= 0.45 and covered.mean() < 0.47
