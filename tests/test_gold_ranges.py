"""Gold price ranges (openspec change gold-forecasting-v2, spec gold-forecasting/price-ranges)."""
import math
import os

import numpy as np
import pandas as pd
import pytest

from src.gold import ranges as R
from src.gold import prereg2 as P2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG2 = P2.load_config_v2()


def test_sigma_from_abs_inverts_the_normal_mean_absolute_value():
    assert R.sigma_from_abs(math.sqrt(2 / math.pi)) == pytest.approx(1.0)


def test_winkler_matches_a_hand_computed_case():
    # 80 % interval [-1, 1]: inside costs the width; 0.5 outside costs 2 + (2 / 0.2) x 0.5
    assert R.winkler(-1, 1, 0.3, 0.2) == pytest.approx(2.0)
    assert R.winkler(-1, 1, 1.5, 0.2) == pytest.approx(2.0 + 10 * 0.5)


def test_no_settled_bar_at_or_after_t_enters_bar_ts_quantile():
    """Change only the outcome of bar 250: every range up to and including bar 250 is identical."""
    y = np.random.default_rng(0).normal(0, 1, 400)
    s = np.ones(400)
    y2 = y.copy()
    y2[250] = 50.0
    lo_a, hi_a, *_ = R.run_aci(y, s, 0.8, 0.005, 100)
    lo_b, hi_b, *_ = R.run_aci(y2, s, 0.8, 0.005, 100)
    assert np.array_equal(lo_a[:251], lo_b[:251], equal_nan=True)
    assert np.array_equal(hi_a[:251], hi_b[:251], equal_nan=True)
    assert not np.array_equal(hi_a[251:260], hi_b[251:260])


def test_coverage_converges_on_iid_normal_data():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 0.2, 20000)
    s = np.full(20000, 0.2)
    for level in (0.8, 0.95):
        _, _, cov, _ = R.run_aci(y, s, level, 0.005, 500)
        assert abs(np.nanmean(cov[1000:]) - level) <= 0.02, level


def test_aci_widens_after_misses():
    aci = R.ACI(0.8, 0.05, 100, seed_z=np.linspace(0, 2, 100))
    q0 = aci.q()
    for _ in range(20):
        aci.step(5.0, covered=False)
    assert aci.q() > q0 and aci.alpha_t < 0.2


def test_95_range_is_wider_than_80():
    seed = np.abs(np.random.default_rng(1).normal(0, 1, 500))
    a80, a95 = R.ACI(0.8, 0.005, 500, seed), R.ACI(0.95, 0.005, 500, seed)
    lo80, hi80 = a80.bounds_pct(0.11)
    lo95, hi95 = a95.bounds_pct(0.11)
    assert lo95 < lo80 < 0 < hi80 < hi95
    p_lo, p_hi = R.price_bounds(4100.0, lo80, hi80)
    assert p_lo < 4100.0 < p_hi


def _synthetic(n=3000, seed=5):
    """A saved-predictions-shaped frame. Round-two tests never score the real
    validation file: the registration must precede the first score."""
    rng = np.random.default_rng(seed)
    vol = np.abs(rng.normal(0.1, 0.03, n)) + 0.02          # true E|r|, varying by bar
    y = rng.normal(0, 1, n) * vol / np.sqrt(2 / np.pi)
    return pd.DataFrame({
        'time': pd.date_range('2023-01-02', periods=n, freq='h', tz='UTC').astype(str),
        'close': 2000.0, 'eligible': True, 'y_dir': (y > 0).astype(float), 'y_ret': y, 'y_vol': np.abs(y),
        'h1_lstm__vol_pct': vol, 'h1_garch_hod__vol_pct': np.full(n, vol.mean())})


def test_range_cell_scores_a_saved_predictions_frame_without_refitting():
    pred = _synthetic()
    st = CFG2['study']['range']
    r = R.score_range_cell('h1_range', 'H1', pred, 'h1_lstm', 'h1_garch_hod', st, 0.025, 200)
    assert r['n_rows'] == 3000 - st['window']['H1'] and r['family'] == 'range'
    assert set(r['_coverage']) == {0.8, 0.95}
    assert 'coverage' in r['notes'] and 'warm-up 500' in r['notes']
    assert r['cleared_bar'] is True                   # the informative sigma beats a flat one
    again = R.score_range_cell('h1_range', 'H1', pred, 'h1_lstm', 'h1_garch_hod', st, 0.025, 200)
    assert again['metric_model'] == r['metric_model']


def test_sharp_but_under_covering_range_does_not_clear():
    st = dict(CFG2['study']['range'], coverage_tolerance=0.0)
    r = R.score_range_cell('h1_range', 'H1', _synthetic(), 'h1_lstm', 'h1_garch_hod', st, 0.025, 200)
    assert r['cleared_bar'] is False and 'coverage' in r['verdict']


# ── 3.2 the round-two stage, range family ───────────────────────────────────

@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    import copy
    import json
    from src.gold import load_config
    from src.gold import prereg as P
    cfg = copy.deepcopy(load_config())
    cfg['paths']['results'] = str(tmp_path)
    cfg2 = copy.deepcopy(CFG2)
    cfg2['paths']['prereg_json'] = str(tmp_path / 'PRE_REGISTRATION_V2.json')
    cfg2['paths']['validation_report'] = str(tmp_path / 'validation_report_v2.md')
    cfg2['paths']['registries'] = {k: str(tmp_path / os.path.basename(v)) for k, v in cfg2['paths']['registries'].items()}
    for path in cfg2['paths']['registries'].values():
        open(path, 'w', encoding='utf-8').write(','.join(P.REGISTRY_FIELDS) + '\n')
    _synthetic().to_csv(tmp_path / 'validation_predictions_h1.csv', index=False)
    d1 = _synthetic(800, seed=9).rename(columns={'h1_lstm__vol_pct': 'd1_lstm__vol_pct',
                                                 'h1_garch_hod__vol_pct': 'd1_garch__vol_pct'})
    d1.to_csv(tmp_path / 'validation_predictions_d1.csv', index=False)
    reg2 = {'config_v2_study_sha256': P2.study_digest(cfg2),
            'families': {'range': {'alpha': 0.025}}}
    json.dump(reg2, open(cfg2['paths']['prereg_json'], 'w'))
    monkeypatch.setattr(P2.FP, 'committed_and_clean', lambda path, repo=None: 'abc123')
    return cfg, cfg2


def test_the_range_stage_runs_once_and_writes_only_round_two_files(stage_env):
    from src.gold import study2 as S2
    cfg, cfg2 = stage_env
    out = S2.run('validation', cfg, cfg2, frames={}, families=('range',))
    assert [r['cell'] for r in out['results']] == ['h1_range', 'd1_range'] and not out['failures']
    rows = list(__import__('csv').DictReader(open(cfg2['paths']['registries']['range'], encoding='utf-8')))
    assert len(rows) == 2 and all('needs forward confirmation' in r['verdict'] for r in rows)
    assert all(r['arbiter'] == S2.ARBITER_V2 for r in rows)
    report = open(cfg2['paths']['validation_report'], encoding='utf-8').read()
    assert 'coverage' in report and 'seen' in report
    with pytest.raises(P2.StageRefused, match='runs once'):
        S2.run('validation', cfg, cfg2, frames={}, families=('range',))


def test_the_stage_refuses_the_test_block(stage_env):
    from src.gold import study2 as S2
    cfg, cfg2 = stage_env
    with pytest.raises(P2.StageRefused, match='spent in round one'):
        S2.run('test', cfg, cfg2, frames={}, families=('range',))


# ── 3.3 live ranges in the logger ───────────────────────────────────────────

def test_the_logger_writes_ranges_for_new_forecasts_only_and_settles_them(tmp_path):
    import copy
    import csv
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_gold_forward_logger import FakeModel, _frames, _h1_weeks
    from src.gold import forward_logger as FL
    from src.gold import load_config
    import hashlib
    cfg = copy.deepcopy(load_config())
    cfg['paths']['forward'] = str(tmp_path / 'forward')
    cfg['paths']['artifacts'] = str(tmp_path / 'art')
    cfg['paths']['manifest'] = str(tmp_path / 'manifest.csv')
    rows = []
    for cells in FL.FORWARD_CELLS.values():
        for cell in cells:
            d = tmp_path / 'art' / cell
            d.mkdir(parents=True)
            (d / 'model.bin').write_bytes(cell.encode())
            rows.append({'cell': cell, 'path': str(d / 'model.bin'), 'sha256': hashlib.sha256(cell.encode()).hexdigest()})
    with open(cfg['paths']['manifest'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['cell', 'path', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    h1 = _h1_weeks()
    run = lambda upto: FL.run_once(cfg, frames=_frames(h1, upto), loader=lambda c, k: FakeModel(), ph='dry_run')
    before = {n: None for n in ('predictions', 'settlements')}
    run(400)
    r1 = FL.read_rows(FL.out_path(cfg, 'ranges'))
    assert {r['cell'] for r in r1} <= {'h1_lstm', 'd1_lstm'} and len(r1) in (2, 4)
    assert all(float(r['lo']) < float(r['hi']) for r in r1)
    lv = {r['key']: {} for r in r1}
    for r in r1:
        lv[r['key']][r['level']] = (float(r['lo']), float(r['hi']))
    for k, v in lv.items():
        assert v['0.95'][0] < v['0.8'][0] and v['0.95'][1] > v['0.8'][1]
    run(401)
    run(402)
    rs = FL.read_rows(FL.out_path(cfg, 'range_settlements'))
    assert rs and {r['inside'] for r in rs} <= {'True', 'False'}
    keys = {r['key'] for r in FL.read_rows(FL.out_path(cfg, 'ranges'))}
    preds = {r['key'] for r in FL.read_rows(FL.out_path(cfg, 'predictions')) if r['cell'] in ('h1_lstm', 'd1_lstm')}
    assert keys == preds                           # one range set per LSTM forecast, none back-filled
    for name, fields in (('predictions', FL.PRED_FIELDS), ('settlements', FL.SETTLE_FIELDS)):
        with open(FL.out_path(cfg, name), encoding='utf-8') as fh:
            assert next(csv.reader(fh)) == fields


def test_existing_forecasts_never_get_a_range_back_filled(tmp_path):
    import copy
    from src.gold import forward_logger as FL
    from src.gold import load_config
    cfg = copy.deepcopy(load_config())
    cfg['paths']['forward'] = str(tmp_path)
    FL.append_rows(FL.out_path(cfg, 'predictions'), [{'key': 'h1_lstm|H1|2026-10-21 16:00:00+00:00', 'cell': 'h1_lstm',
                                                      'cadence': 'H1', 'as_of_bar': '2026-10-21 16:00:00+00:00',
                                                      'as_of_close': 4100, 'vol_pct': 0.1}], FL.PRED_FIELDS)
    assert FL.log_ranges(cfg, '2026-10-21T18:03:00+00:00', []).startswith('0 range')
    assert FL.read_rows(FL.out_path(cfg, 'ranges')) == []


# ── 3.4 the page ────────────────────────────────────────────────────────────

def test_range_card_words_are_bilingual_and_never_trading_levels():
    src = open(os.path.join(REPO, 'static', 'gold.html'), encoding='utf-8').read()
    block = src.split('range_title:', 1)[1].split('stall:', 1)[0].lower()
    for banned in ('stop', 'target', 'entry', 'take-profit', 'take profit', 'exit', 'стоп', 'цел', 'вход', 'изход'):
        assert banned not in block, banned
    for key in ('range_title', 'range80', 'range95', 'range_note_h1', 'range_note_d1', 'h_range', 'in_range', 'out_range'):
        assert f'{key}: {{bg: ' in src, key
