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
