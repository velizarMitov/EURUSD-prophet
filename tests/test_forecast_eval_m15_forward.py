"""The M15 cadence in the forward logger and the refit (task 14.9).

A fake MT5 module and synthetic M15 bars; no terminal, no network, nothing
written outside tmp_path.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import forward_logger as FL
from src.forecast_eval import m15_data as MD
from src.forecast_eval import record as R
from src.forecast_eval import refit as RF
from src.forecast_eval import study as ST
from src.forecast_eval import targets as T
from src.forecast_eval.challengers import make
from src.forecast_eval.mt5_reader import drop_incomplete_m15_bars

from .test_forecast_eval_forward_logger import FakeMT5, LOGIN            # noqa: F401

POINT = 1e-5


def _trading_week_index(start='2026-06-01', days=60):
    """A realistic forex grid: the week opens Sunday at the label hour
    Europe/Berlin predicts (23:00, or 22:00 in a daylight-saving mismatch week)
    and closes Friday 23:00. `m15_data.verify_clock` is satisfied by this and
    refuses a naive Monday-to-Friday grid, which is the point."""
    idx = pd.date_range(pd.Timestamp(start, tz='UTC'), periods=96 * days, freq='15min')
    dow, hour = idx.dayofweek, idx.hour
    keep = np.zeros(len(idx), dtype=bool)
    keep |= np.isin(dow, [0, 1, 2, 3])                       # Mon-Thu, all hours
    keep |= (dow == 4) & (hour < 23)                         # Friday, until the close
    sunday_open = np.array([MD.expected_open_label(d) for d in idx.date])
    keep |= (dow == 6) & (hour >= sunday_open)               # Sunday, from the open
    return idx[keep]


def _m15_frame(days=60, seed=3, spread_points=5.0, rollover_points=None, start='2026-06-01'):
    rng = np.random.default_rng(seed)
    idx = _trading_week_index(start, days)
    close = 1.10 + np.cumsum(rng.normal(0, 2e-5, len(idx)))
    spread = np.full(len(idx), float(spread_points))
    if rollover_points:
        spread[idx.hour == 23] = rollover_points
    return pd.DataFrame({'open': close, 'high': close + 3e-5, 'low': close - 3e-5,
                         'close': close, 'tick_volume': rng.integers(20, 200, len(idx)).astype(float),
                         'spread': spread}, index=idx)


def test_the_synthetic_feed_satisfies_the_clock_guard():
    ev = MD.verify_clock(_m15_frame(days=120).index)
    assert ev['label_tz'] == 'Europe/Berlin' and ev['match_rate'] >= 0.9
    with pytest.raises(MD.ClockMismatch):
        naive = pd.date_range('2026-06-01', periods=96 * 60, freq='15min', tz='UTC')
        MD.verify_clock(naive[naive.dayofweek < 5])


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A manifest with one m15_session_gbm version per horizon {1, 4}."""
    monkeypatch.setattr(ST, 'ART', str(tmp_path / 'art'))
    rec = R.new_record('fw15')
    cfg = rec['challengers']['m15_session_gbm']
    raw = _m15_frame(days=120)
    frame = MD.from_mt5_bars(raw, POINT)
    train_frame = frame.iloc[:int(len(frame) * 0.6)]
    man = str(tmp_path / 'manifest.csv')
    for h in (1, 4):
        m = make('m15_session_gbm', cfg)
        inp = m.build_inputs(train_frame)
        d, _ = T.direction_target(pd.Series(inp.close), h)
        pos = MD.eligible_positions(inp.index, h)
        m.fit(inp, pos, {'dir': d.to_numpy()}, 42, h)
        ST._append_csv(man, ST.save_artifact(m, 'fw15', 'm15_session_gbm', 'M15', h, 'frozen',
                                             inp, pos, 'dig'), ST.MANIFEST_FIELDS)
    return {'out': str(tmp_path / 'fwd'), 'manifest': man,
            'record': str(tmp_path / 'none.json'), 'raw': raw}


def _snap15(raw, **kw):
    h1 = raw.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                 'tick_volume': 'sum', 'spread': 'mean'}).dropna()
    d1 = raw.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                 'tick_volume': 'sum', 'spread': 'mean'}).dropna()
    return FL.Snapshot(h1, d1, kw.get('server', 'Fake-Server'),
                       {'swap_long': -7.1, 'swap_short': 2.3, 'point': POINT},
                       kw.get('source', 'MT5'), m15=raw)


def _run(env, raw, phase='dry_run'):
    return FL.predict_cadence(_snap15(raw), 'M15', phase, env['out'], env['manifest'],
                              env['record'], 'dig')


def _last_in_session(raw, offset=0):
    """Truncate the frame so its newest bar is an in-session one."""
    sess = np.flatnonzero(MD.session_mask(raw.index))
    return raw.iloc[:sess[-1 - offset] + 1]


# ── closed-bar rule ────────────────────────────────────────────────────────

def test_only_elapsed_m15_bars_count_as_closed():
    idx = pd.date_range('2026-06-01 14:00', periods=6, freq='15min', tz='UTC')
    df = pd.DataFrame({'close': 1.0}, index=idx)
    kept = drop_incomplete_m15_bars(df, pd.Timestamp('2026-06-01 15:00', tz='UTC'))
    assert list(kept.index) == list(idx[:4]), 'the 15:00 bar is still forming'
    assert len(drop_incomplete_m15_bars(df, pd.Timestamp('2026-06-01 14:10', tz='UTC'))) == 0


def test_saturday_m15_bars_are_dropped():
    idx = pd.date_range('2026-06-06 10:00', periods=4, freq='15min', tz='UTC')   # Saturday
    df = pd.DataFrame({'close': 1.0}, index=idx)
    assert len(drop_incomplete_m15_bars(df, pd.Timestamp('2026-06-08', tz='UTC'))) == 0


# ── session-restricted logging ─────────────────────────────────────────────

def test_an_in_session_close_is_logged_for_every_horizon(env):
    raw = _last_in_session(env['raw'])
    res = _run(env, raw)
    assert res['predicted'] == 2 and res['failed'] == 0
    p = pd.read_csv(os.path.join(env['out'], 'predictions.csv'))
    assert set(p['horizon']) == {1, 4} and (p['cadence'] == 'M15').all()
    assert p['p_up'].between(0, 1).all()
    assert (p['spread_points'] == 5.0).all(), "the bar's own spread is recorded"
    assert pd.Timestamp(p['as_of_bar'].iloc[0]) == raw.index[-1]
    assert MD.session_mask(pd.DatetimeIndex([pd.Timestamp(p['as_of_bar'].iloc[0])]))[0]


def test_an_out_of_session_close_writes_neither_a_prediction_nor_a_gap(env):
    """Spec forward-arbiter: outside the session there is no row and no gap."""
    sess = np.flatnonzero(MD.session_mask(env['raw'].index))
    out_of_session = env['raw'].iloc[:sess[-1] + 6]          # past the session end
    assert not MD.session_mask(out_of_session.index)[-1]
    res = _run(env, out_of_session)
    assert res['status'] == 'outside the session'
    assert not os.path.exists(os.path.join(env['out'], 'predictions.csv'))
    assert not os.path.exists(os.path.join(env['out'], 'gaps.csv'))


def test_only_in_session_misses_become_gaps(env):
    """A bar the logger was never going to forecast is not a gap in coverage."""
    raw = env['raw']
    sess = np.flatnonzero(MD.session_mask(raw.index))
    _run(env, raw.iloc[:sess[-40] + 1])                      # establish a last-logged bar
    _run(env, raw.iloc[:sess[-1] + 1])                       # jump forward over a weekend
    gaps = pd.read_csv(os.path.join(env['out'], 'gaps.csv'))
    bars = pd.to_datetime(gaps['bar'], utc=True)
    assert len(bars) and MD.session_mask(pd.DatetimeIndex(bars)).all()


def test_a_repeat_run_is_a_no_op(env):
    raw = _last_in_session(env['raw'])
    _run(env, raw)
    path = os.path.join(env['out'], 'predictions.csv')
    before = open(path, 'rb').read()
    _run(env, raw)
    assert open(path, 'rb').read() == before


# ── settlement on the mid price ────────────────────────────────────────────

def _settle_one(env, h=1, **snap_kw):
    raw = _last_in_session(env['raw'], offset=6)
    _run(env, raw)
    later = _last_in_session(env['raw'])
    FL.settle(_snap15(later, **snap_kw), env['out'])
    return pd.read_csv(os.path.join(env['out'], 'settlements.csv')).set_index('key')


def test_settlement_uses_the_mid_price_at_both_ends(env):
    s = _settle_one(env)
    raw = env['raw']
    row = s.iloc[0]
    exit_bar = pd.Timestamp(row['exit_bar'])
    bid = float(raw.loc[exit_bar, 'close'])
    assert row['exit_close'] == pytest.approx(bid + 5.0 * POINT / 2)


def test_a_bar_without_a_spread_is_excluded_from_scoring(env):
    """No spread means no mid price, and a mid price is never guessed."""
    _run(env, _last_in_session(env['raw'], offset=6))
    snap = _snap15(_last_in_session(env['raw']))
    snap.m15 = snap.m15.drop(columns=['spread'])          # the bars lost their spread
    assert FL.settle(snap, env['out']) > 0
    s = pd.read_csv(os.path.join(env['out'], 'settlements.csv'))
    assert not s['scorable'].astype(bool).any()
    assert s['exclusion_reason'].str.contains('missing spread').all()


def test_a_trade_ending_outside_the_session_is_excluded(env):
    """The study's eligibility rule, enforced again at settlement."""
    raw = env['raw']
    sess = np.flatnonzero(MD.session_mask(raw.index))
    # an as-of bar in session whose 4-bar target falls past the session end
    late = [i for i in sess if MD.session_mask(raw.index)[i]
            and not MD.eligible_mask(raw.index, 4)[i]]
    assert late, 'the frame must contain a late-session bar'
    at = late[-1]
    _run(env, raw.iloc[:at + 1])
    FL.settle(_snap15(raw.iloc[:at + 10]), env['out'])
    s = pd.read_csv(os.path.join(env['out'], 'settlements.csv'))
    h4 = s[s['key'].str.contains(r'\|h4\|')]
    assert len(h4) and not h4['scorable'].astype(bool).any()
    assert h4['exclusion_reason'].str.contains('outside the session').all()


def test_the_login_never_reaches_an_m15_output(env):
    _run(env, _last_in_session(env['raw']))
    for f in os.listdir(env['out']):
        assert str(LOGIN) not in open(os.path.join(env['out'], f), encoding='utf-8').read()


# ── the snapshot and the point size ────────────────────────────────────────

def test_fetch_snapshot_reads_m15_with_its_spread(tmp_path):
    raw = _m15_frame(days=20)
    h1 = raw.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                 'tick_volume': 'sum', 'spread': 'mean'}).dropna()
    from src.forecast_eval.mt5_reader import MT5Reader
    fake = FakeMT5(h1, h1.resample('1D').last().dropna(), m15=raw)
    snap = FL.fetch_snapshot(MT5Reader(fake), now_utc=h1.index[-1] + pd.Timedelta(minutes=70),
                             state_path=str(tmp_path / 'offset.json'))
    assert snap.m15 is not None and 'spread' in snap.m15.columns
    assert snap.meta['point'] == POINT
    assert snap.m15.index[-1] <= raw.index[-1]


def test_source_frame_refuses_to_guess_a_point_size(env):
    snap = _snap15(_last_in_session(env['raw']))
    snap.meta = dict(snap.meta)
    snap.meta.pop('point')
    with pytest.raises(ValueError, match='refusing to guess'):
        FL.source_frame('m15_session_gbm', 'M15', snap, snap.m15.index[-1])


def test_from_mt5_bars_matches_the_aggregation_shape():
    raw = _m15_frame(days=5)
    frame = MD.from_mt5_bars(raw, POINT)
    assert {'open', 'high', 'low', 'close', 'mid', 'tick_volume', 'spread_price',
            'point'} <= set(frame.columns)
    assert np.allclose(frame['spread_price'], 5.0 * POINT)
    assert np.allclose(frame['mid'] - frame['close'], 5.0 * POINT / 2)
    from src.forecast_eval.costs import MissingSpreadError
    with pytest.raises(MissingSpreadError):
        MD.from_mt5_bars(raw.drop(columns=['spread']), POINT)
    with pytest.raises(ValueError, match='point size'):
        MD.from_mt5_bars(raw, 0.0)


# ── refit ──────────────────────────────────────────────────────────────────

def test_refit_history_extends_m15_with_newer_live_bars():
    frame = MD.m15_bars() if os.path.exists(MD.M1_PATH) else None
    if frame is None:
        pytest.skip('M1 parquet is not in git (DATA.md 7)')
    newer = _m15_frame(days=2)
    newer.index = frame.index[-1] + pd.Timedelta(minutes=15) + \
        (newer.index - newer.index[0])
    src = RF.history_sources(newer.index[-1], {'m15': newer, 'point': POINT}, macro=False)
    assert src['m15'] is not None
    assert src['m15'].index[-1] > frame.index[-1]
    cut = RF.history_sources(frame.index[-1], {'m15': newer, 'point': POINT}, macro=False)
    assert cut['m15'].index[-1] <= frame.index[-1], 'nothing after the cutoff'


def test_refit_trains_an_m15_cell_on_eligible_rows_only(env, tmp_path, monkeypatch):
    monkeypatch.setattr(ST, 'ART', str(tmp_path / 'art2'))
    seen = {}
    real_fit = make('m15_session_gbm',
                    R.new_record('x')['challengers']['m15_session_gbm']).__class__.fit

    def spy(self, inputs, train_pos, targets, seed, h):
        seen['pos'] = np.asarray(train_pos)
        seen['index'] = inputs.index
        return real_fit(self, inputs, train_pos, targets, seed, h)

    monkeypatch.setattr(type(make('m15_session_gbm',
                                  R.new_record('x')['challengers']['m15_session_gbm'])),
                        'fit', spy)
    man = pd.read_csv(env['manifest'])
    row = man[man['horizon'] == 4].iloc[0].to_dict()
    sources = {'m15': MD.from_mt5_bars(env['raw'], POINT), 'h1': None, 'daily': {}}
    RF.refit_cell(row, sources, pd.Timestamp('2026-09-05 06:00', tz='UTC'))
    assert seen['pos'].size
    assert MD.session_mask(seen['index'])[seen['pos']].all(), 'session rows only'
    assert set(seen['pos'].tolist()) <= set(MD.eligible_positions(seen['index'], 4).tolist())
