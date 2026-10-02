"""Forward logger (tasks 9.1-9.5). A fake MT5 module and real H1 bars; no
terminal, no network, nothing written outside tmp_path."""
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import forward_logger as FL
from src.forecast_eval import record as R
from src.forecast_eval import study as ST
from src.forecast_eval import targets as T
from src.forecast_eval.challengers import make
from src.forecast_eval.mt5_reader import MT5Reader, MT5Unavailable

LOGIN = 1007437


class FakeMT5:
    TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_D1 = 15, 16385, 16408

    def __init__(self, h1, d1, ok=True, m15=None):
        self.h1, self.d1, self.ok = h1, d1, ok
        self.m15 = m15 if m15 is not None else h1

    def initialize(self):
        return self.ok

    def last_error(self):
        return (-10003, 'IPC initialize failed')

    def shutdown(self):
        pass

    def account_info(self):
        return namedtuple('A', 'login server balance')(LOGIN, 'Fake-Server', 1.0)

    def symbol_select(self, s, v):
        return True

    def copy_rates_range(self, *a):
        return self._rates(self.h1)[-5:]

    def symbol_info(self, s):
        return namedtuple('S', 'point digits swap_long swap_short swap_mode')(1e-5, 5, -7.1, 2.3, 1)

    def copy_rates_from_pos(self, symbol, tf, start, count):
        frame = {self.TIMEFRAME_M15: self.m15, self.TIMEFRAME_H1: self.h1,
                 self.TIMEFRAME_D1: self.d1}[tf]
        return self._rates(frame)[-count:]

    @staticmethod
    def _rates(df):
        out = np.zeros(len(df), dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
                                       ('close', 'f8'), ('tick_volume', 'f8'), ('spread', 'i4'),
                                       ('real_volume', 'i8')])
        out['time'] = (pd.DatetimeIndex(df.index).tz_localize(None).asi8 // 10 ** 9)
        for c in ('open', 'high', 'low', 'close', 'tick_volume'):
            out[c] = df[c].to_numpy()
        out['spread'] = 6
        return out


@pytest.fixture(scope='module')
def h1_all():
    return F.load_h1().iloc[-1600:]


def _snap(h1, server='Fake-Server', source='MT5', m15=None):
    h1 = h1.assign(spread=6.0)
    d1 = h1.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                'tick_volume': 'sum', 'spread': 'mean'}).dropna()
    return FL.Snapshot(h1, d1, server, {'swap_long': -7.1, 'swap_short': 2.3, 'point': 1e-5},
                       source, m15=m15)


@pytest.fixture
def env(tmp_path, h1_all, monkeypatch):
    """A manifest with one h1_gbm version per horizon {1, 4}, trained on bars
    before the forward window."""
    monkeypatch.setattr(ST, 'ART', str(tmp_path / 'art'))
    rec = R.new_record('fw')
    cfg = rec['challengers']['h1_gbm']
    train = h1_all.iloc[:1200]
    digest = F.feature_code_digest()['_combined']
    man = str(tmp_path / 'manifest.csv')
    for h in (1, 4):
        m = make('h1_gbm', cfg)
        inp = m.build_inputs(train)
        d, _ = T.direction_target(pd.Series(inp.close), h)
        pos = np.arange(len(inp) - h)
        m.fit(inp, pos, {'dir': d.to_numpy()}, 42, h)
        row = ST.save_artifact(m, 'fw', 'h1_gbm', 'H1', h, 'frozen', inp, pos, digest)
        ST._append_csv(man, row, ST.MANIFEST_FIELDS)
    return {'out': str(tmp_path / 'fwd'), 'manifest': man, 'record': str(tmp_path / 'none.json'),
            'h1': h1_all}


def _run(env, h1, **kw):
    return FL.run_once(snapshot=_snap(h1, **kw), phase='dry_run', out=env['out'],
                       manifest_path=env['manifest'], record_path=env['record'], macro_df=None)


# ── 9.1 read-only reader ─────────────────────────────────────────────────

def test_reader_returns_server_never_login(h1_all, tmp_path):
    d1 = h1_all.resample('1D').last().dropna()
    fake = FakeMT5(h1_all, d1)
    snap = FL.fetch_snapshot(MT5Reader(fake), now_utc=h1_all.index[-1] + pd.Timedelta(minutes=70),
                             state_path=str(tmp_path / 'offset.json'))
    assert snap.server == 'Fake-Server' and snap.source == 'MT5'
    assert 'spread' in snap.h1.columns and snap.meta['swap_long'] == -7.1
    assert snap.h1.index[-1] <= h1_all.index[-1]


def test_reader_unreachable_terminal_raises():
    with pytest.raises(MT5Unavailable):
        MT5Reader(FakeMT5(None, None, ok=False)).connect()


def test_login_never_reaches_any_output(env):
    _run(env, env['h1'].iloc[:1300])
    for f in os.listdir(env['out']):
        assert str(LOGIN) not in open(os.path.join(env['out'], f), encoding='utf-8').read()


# ── 9.2 predict step ─────────────────────────────────────────────────────

def test_predict_step_writes_every_field_and_is_idempotent(env):
    h1 = env['h1'].iloc[:1300]
    _run(env, h1)
    p = pd.read_csv(os.path.join(env['out'], 'predictions.csv'))
    assert set(p['horizon']) == {1, 4} and len(p) == 2
    assert set(FL.PRED_FIELDS) == set(p.columns)
    assert (p['phase'] == 'dry_run').all() and (p['price_source'] == 'MT5').all()
    assert p['as_of_bar'].nunique() == 1 and pd.Timestamp(p['as_of_bar'].iloc[0]) == h1.index[-1]
    assert p['p_up'].between(0, 1).all()
    before = open(os.path.join(env['out'], 'predictions.csv'), 'rb').read()
    _run(env, h1)
    assert open(os.path.join(env['out'], 'predictions.csv'), 'rb').read() == before


def test_version_used_was_trained_before_the_as_of_bar(env):
    _run(env, env['h1'].iloc[:1300])
    p = pd.read_csv(os.path.join(env['out'], 'predictions.csv'))
    man = pd.read_csv(env['manifest']).set_index('version_id')
    for _, r in p.iterrows():
        assert pd.Timestamp(man.loc[r['version_id'], 'train_end']) < pd.Timestamp(r['as_of_bar'])


# ── 9.3 gaps, never backfilled ───────────────────────────────────────────

def test_missed_bars_become_gaps_and_are_never_predicted(env):
    h1 = env['h1']
    _run(env, h1.iloc[:1300])
    _run(env, h1.iloc[:1304])                      # three closes missed, the 4th is newest
    gaps = pd.read_csv(os.path.join(env['out'], 'gaps.csv'))
    assert len(gaps[gaps['cadence'] == 'H1']) == 3
    missed = set(pd.to_datetime(gaps['bar'], utc=True))
    _run(env, h1.iloc[:1304])
    _run(env, h1.iloc[:1305])
    p = pd.read_csv(os.path.join(env['out'], 'predictions.csv'))
    assert missed.isdisjoint(set(pd.to_datetime(p['as_of_bar'], utc=True)))
    assert p['as_of_bar'].nunique() == 3          # bars 1300, 1304, 1305 only


# ── 9.4 settlement ───────────────────────────────────────────────────────

def test_settlement_appends_and_excludes_non_mt5(env):
    h1 = env['h1']
    _run(env, h1.iloc[:1300])
    path = os.path.join(env['out'], 'predictions.csv')
    p = pd.read_csv(path)
    fake = p.iloc[[0]].copy()
    fake['key'] = 'yf|h1|x'
    fake['price_source'] = 'yfinance'
    FL.append_rows(path, fake.to_dict('records'), FL.PRED_FIELDS)
    before = open(path, 'rb').read()
    _run(env, h1.iloc[:1310])
    assert open(path, 'rb').read().startswith(before)
    s = pd.read_csv(os.path.join(env['out'], 'settlements.csv')).set_index('key')
    assert not s.loc['yf|h1|x', 'scorable'] and 'yfinance' in s.loc['yf|h1|x', 'exclusion_reason']
    first = p.iloc[0]
    assert 'phase dry_run' in s.loc[first['key'], 'exclusion_reason']   # dry-run never scorable
    exit_close = float(h1['close'].iloc[1299 + int(first['target_offset_bars'])])
    assert s.loc[first['key'], 'exit_close'] == pytest.approx(exit_close)
    n = len(s)
    _run(env, h1.iloc[:1310])
    assert len(pd.read_csv(os.path.join(env['out'], 'settlements.csv'))) == n   # settled once


# ── 9.5 failure isolation and the feature-code guard ─────────────────────

def test_one_missing_artifact_does_not_stop_the_others(env):
    man = pd.read_csv(env['manifest'])
    victim = man[man['horizon'] == 4].iloc[0]
    import json
    first_file = sorted(json.loads(victim['files_sha256']))[0]
    os.remove(os.path.join(FL.REPO, first_file))
    _run(env, env['h1'].iloc[:1300])
    p = pd.read_csv(os.path.join(env['out'], 'predictions.csv'))
    f = pd.read_csv(os.path.join(env['out'], 'failures.csv'))
    assert list(p['horizon']) == [1]
    assert list(f['horizon']) == [4] and 'missing or altered' in f['error'].iloc[0]


def test_changed_feature_code_blocks_prediction(env):
    man = pd.read_csv(env['manifest'])
    man['feature_digest'] = 'stale'
    man.to_csv(env['manifest'], index=False)
    _run(env, env['h1'].iloc[:1300])
    f = pd.read_csv(os.path.join(env['out'], 'failures.csv'))
    assert len(f) == 2 and f['error'].str.contains('feature code changed').all()
    assert not os.path.exists(os.path.join(env['out'], 'predictions.csv'))


def test_concurrent_run_is_locked_out(env):
    lock = os.path.join(env['out'], 'logger.lock')
    with FL.RunLock(lock):
        with pytest.raises(FL.Locked):
            with FL.RunLock(lock):
                pass


def test_manifest_with_mixed_timestamp_formats(tmp_path):
    """Regression: daily versions record a naive train_end, H1 versions a
    tz-aware one. The first live run crashed on exactly this mix."""
    man = tmp_path / 'm.csv'
    pd.DataFrame([
        {'version_id': 'd', 'model': 'daily_gbm_price', 'cadence': 'D1', 'horizon': 1,
         'train_end': '2026-08-07 00:00:00'},
        {'version_id': 'h', 'model': 'h1_gbm', 'cadence': 'H1', 'horizon': 1,
         'train_end': '2026-09-30 05:00:00+00:00'},
    ]).to_csv(man, index=False)
    as_of = pd.Timestamp('2026-10-01', tz='UTC')
    assert [v['version_id'] for v in FL.latest_versions(as_of, 'D1', str(man))] == ['d']
    assert [v['version_id'] for v in FL.latest_versions(as_of, 'H1', str(man))] == ['h']
