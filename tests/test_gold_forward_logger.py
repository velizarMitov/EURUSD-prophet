"""Gold forward logger (task 8.1; spec gold-forecasting/forward-arbiter)."""
import copy
import csv
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.gold import clock as K
from src.gold import forward_logger as FL
from src.gold import load_config

CFG = load_config()


def _h1_weeks(n_weeks=30, start='2024-01-08'):
    stamps = []
    for w in range(n_weeks):
        monday = pd.Timestamp(start, tz='UTC') + pd.Timedelta(weeks=w)
        first = monday - pd.Timedelta(hours=1) if K.expected_open_hour(monday) == 23 else monday
        day = pd.date_range(first, monday + pd.Timedelta(days=4, hours=22), freq='h')
        stamps.extend(day[day.hour != 23] if K.expected_open_hour(monday) == 0 else day)
    idx = pd.DatetimeIndex(stamps)
    c = 2000 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.001, len(idx))))
    return pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close': c}, index=idx)


def _d1(h1):
    d = h1['close'].resample('D').last().dropna()
    return pd.DataFrame({'open': d, 'high': d * 1.002, 'low': d * 0.998, 'close': d})


class FakeModel:
    def __init__(self, p=0.6):
        self.p = p

    def predict(self, inp):
        n = len(inp)
        return {'p_up': np.full(n, self.p), 'ret_pct': np.full(n, 0.01), 'vol_pct': np.full(n, 0.1)}


@pytest.fixture
def env(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['forward'] = str(tmp_path / 'forward')
    cfg['paths']['artifacts'] = str(tmp_path / 'art')
    cfg['paths']['manifest'] = str(tmp_path / 'manifest.csv')
    rows = []
    for cells in FL.FORWARD_CELLS.values():
        for cell in cells:
            d = tmp_path / 'art' / cell
            d.mkdir(parents=True)
            f = d / 'model.bin'
            f.write_bytes(cell.encode())
            rows.append({'cell': cell, 'path': str(f), 'sha256': hashlib.sha256(cell.encode()).hexdigest()})
    with open(cfg['paths']['manifest'], 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['cell', 'path', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    h1 = _h1_weeks()
    return cfg, tmp_path, h1


def _frames(h1, upto):
    part = h1.iloc[:upto]
    return {'H1': part, 'D1': _d1(part)}


def _run(cfg, frames):
    return FL.run_once(cfg, frames=frames, loader=lambda cell, c: FakeModel(), ph='dry_run')


def _rows(cfg, name):
    return FL.read_rows(FL.out_path(cfg, name))


def test_first_pass_forecasts_only_the_newest_bar_then_nothing_new(env):
    cfg, _, h1 = env
    st = _run(cfg, _frames(h1, 3000))
    preds = _rows(cfg, 'predictions')
    assert {r['as_of_bar'] for r in preds if r['cadence'] == 'H1'} == {str(h1.index[2999])}
    assert len(preds) == 6 and all(r['phase'] == 'dry_run' for r in preds)
    assert _run(cfg, _frames(h1, 3000))['H1'] == 'nothing new'
    assert len(_rows(cfg, 'predictions')) == 6


def test_missed_bars_become_gaps_and_are_never_back_filled(env):
    cfg, _, h1 = env
    _run(cfg, _frames(h1, 3000))
    _run(cfg, _frames(h1, 3004))
    gaps = [r for r in _rows(cfg, 'gaps') if r['cadence'] == 'H1']
    assert [r['as_of_bar'] for r in gaps] == [str(t) for t in h1.index[3000:3003]]
    h1_preds = {r['as_of_bar'] for r in _rows(cfg, 'predictions') if r['cadence'] == 'H1'}
    assert h1_preds == {str(h1.index[2999]), str(h1.index[3003])}


def test_a_digest_mismatch_fails_only_that_cell(env):
    cfg, tmp, h1 = env
    (tmp / 'art' / 'h1_lstm' / 'model.bin').write_bytes(b'retrained quietly')
    _run(cfg, _frames(h1, 3000))
    cells = {r['cell'] for r in _rows(cfg, 'predictions') if r['cadence'] == 'H1'}
    assert cells == {'h1_gbm', 'h1_garch_hod'}
    fails = _rows(cfg, 'failures')
    assert any(r['cell'] == 'h1_lstm' and 'manifest' in r['error'] for r in fails)


def test_settlement_marks_break_bordering_rows_ineligible(env):
    cfg, _, h1 = env
    fridays_last = [i for i in range(1, len(h1) - 1) if (h1.index[i + 1] - h1.index[i]) > pd.Timedelta(hours=20)]
    i = fridays_last[5]
    _run(cfg, _frames(h1, i + 1))                  # as-of = the week's last bar
    _run(cfg, _frames(h1, i + 3))                  # Monday bars arrive
    s = [r for r in _rows(cfg, 'settlements') if r['cadence'] == 'H1' and r['as_of_bar'] == str(h1.index[i])]
    assert s and all(r['eligible'] == 'False' for r in s)
    assert s[0]['exit_bar'] == str(h1.index[i + 1])


def test_ordinary_hour_settles_eligible_with_the_realised_return(env):
    cfg, _, h1 = env
    _run(cfg, _frames(h1, 3000))
    _run(cfg, _frames(h1, 3001))
    s = [r for r in _rows(cfg, 'settlements') if r['cadence'] == 'H1' and r['as_of_bar'] == str(h1.index[2999])]
    assert s and s[0]['eligible'] == 'True'
    expect = 100 * np.log(h1['close'].iloc[3000] / h1['close'].iloc[2999])
    assert float(s[0]['realised_ret_pct']) == pytest.approx(expect)


def test_shifted_clock_stops_logging(env):
    cfg, _, h1 = env
    shifted = h1.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    assert _run(cfg, {'H1': shifted.iloc[:3000], 'D1': _d1(shifted.iloc[:3000])})['status'] == 'clock mismatch'
    assert not _rows(cfg, 'predictions')


def test_concurrent_pass_is_refused(env):
    cfg, _, h1 = env
    lock = os.path.join(cfg['paths']['artifacts'], 'forward.lock')
    with FL.RunLock(lock):
        with pytest.raises(FL.Locked):
            _run(cfg, _frames(h1, 3000))


# ── GVZ step (change gold-gvz-morning-check, task 3.1) ──────────────────────

def _gvz_cfg2(tmp_path):
    from src.gold import prereg2 as P2
    c = copy.deepcopy(P2.load_config_v2())
    c['paths']['macro_cache'] = str(tmp_path / 'macro')
    c['gvz']['log'] = str(tmp_path / 'gvz_log.csv')
    c['gvz']['settlements'] = str(tmp_path / 'gvz_settlements.csv')
    return c


def _gvz_boom(*a, **k):
    raise ConnectionError('offline')


def test_gvz_fresh_cache_contacts_no_source(env):
    cfg, tmp, h1 = env
    c2 = _gvz_cfg2(tmp)
    os.makedirs(c2['paths']['macro_cache'])
    pd.DataFrame({'date': ['2026-10-26'], 'gvz': [23.55]}).to_csv(
        os.path.join(c2['paths']['macro_cache'], c2['gvz']['yahoo_cache']), index=False)
    calls = []

    def spy(*a, **k):
        calls.append(a)
        raise AssertionError('no source may be contacted')
    tue_1403_sofia = pd.Timestamp('2026-10-27 12:03', tz='UTC')
    note = FL.log_gvz(cfg, 'now', {'H1': h1}, cfg2=c2, now=tue_1403_sofia, fred_fetch=spy, yahoo_fetch=spy)
    assert calls == [] and note.startswith('up to date')


def _strip(rows, *drop):
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def test_gvz_failure_is_recorded_and_leaves_the_pass_unchanged(env):
    cfg, tmp, h1 = env
    c2 = _gvz_cfg2(tmp)
    step = lambda c, now_iso, frames: FL.log_gvz(c, now_iso, frames, cfg2=c2,      # noqa: E731
                                                 fred_fetch=_gvz_boom, yahoo_fetch=_gvz_boom)
    plain = copy.deepcopy(cfg)
    plain['paths']['forward'] = str(tmp / 'forward_plain')
    plain['paths']['artifacts'] = str(tmp / 'art')
    for upto in (2000, 2003):
        st = FL.run_once(cfg, frames=_frames(h1, upto), loader=lambda cell, c: FakeModel(), ph='dry_run', gvz=step)
        FL.run_once(plain, frames=_frames(h1, upto), loader=lambda cell, c: FakeModel(), ph='dry_run')
    assert st['gvz'].startswith('failed GvzUnavailable')
    fails = [r for r in _rows(cfg, 'failures') if r['cell'] == 'gvz']
    assert fails and 'GVZ' in fails[0]['error']
    assert _strip(_rows(cfg, 'predictions'), 'logged_at') == _strip(_rows(plain, 'predictions'), 'logged_at')
    assert _strip(_rows(cfg, 'settlements'), 'settled_at') == _strip(_rows(plain, 'settlements'), 'settled_at')
    assert len(_rows(cfg, 'settlements')) > 0


def test_injected_frames_skip_gvz_unless_a_step_is_passed(env):
    cfg, _, h1 = env
    assert 'gvz' not in _run(cfg, _frames(h1, 2000))


def test_health_note_keeps_alerts_and_gvz_within_its_limit(env):
    from src.gold import health as HL
    cfg, _, _ = env
    st = {'D1': 'forecast 3 cell(s)', 'H1': 'forecast 3 cell(s)', 'ranges': '2 range(s), 1 settled',
          'alerts': '0 alert(s), 0 settled', 'gvz': 'fresh 2026-10-26 yfinance, logged, settled 1',
          'shadow': 's' * 90, 'v2': 'v' * 90, 'mt5': True, 'newest_h1': 'x'}
    FL.run_with_health(cfg, runner=lambda: st)
    with open(HL.health_path(cfg), newline='', encoding='utf-8') as fh:
        note = list(csv.DictReader(fh))[-1]['note']
    assert len(note) <= 300 and 'alerts: 0 alert(s)' in note and 'gvz: fresh 2026-10-26 yfinance' in note
