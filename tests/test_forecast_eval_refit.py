"""Monthly walk-forward refits (task 10.1)."""
import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import features as F
from src.forecast_eval import forward_logger as FL
from src.forecast_eval import record as R
from src.forecast_eval import refit as RF
from src.forecast_eval import study as ST
from src.forecast_eval import targets as T
from src.forecast_eval.challengers import make

UTC = 'UTC'


def test_first_weekend_only():
    assert RF.first_weekend(pd.Timestamp('2026-11-07 06:00', tz=UTC))       # Sat the 7th
    assert RF.first_weekend(pd.Timestamp('2026-11-01 06:00', tz=UTC))       # Sun the 1st
    assert not RF.first_weekend(pd.Timestamp('2026-11-14 06:00', tz=UTC))   # 2nd Saturday
    assert not RF.first_weekend(pd.Timestamp('2026-11-05 06:00', tz=UTC))   # Thursday


def test_cutoff_is_the_friday_before():
    assert RF.last_friday_close(pd.Timestamp('2026-11-07 06:00', tz=UTC)) == \
        pd.Timestamp('2026-11-06 23:59:59', tz=UTC)
    assert RF.last_friday_close(pd.Timestamp('2026-11-01 06:00', tz=UTC)) == \
        pd.Timestamp('2026-10-30 23:59:59', tz=UTC)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, 'ART', str(tmp_path / 'art'))
    h1 = F.load_h1()
    sat = pd.Timestamp('2026-09-05 06:00', tz=UTC)                 # first Saturday of Sept 2026
    hist = h1[h1.index <= pd.Timestamp('2026-08-28 23:59', tz=UTC)].iloc[-1500:]
    cfg = R.new_record('rf')['challengers']['h1_gbm']
    m = make('h1_gbm', cfg)
    inp = m.build_inputs(hist)
    d, _ = T.direction_target(pd.Series(inp.close), 1)
    pos = np.arange(len(inp) - 1)
    m.fit(inp, pos, {'dir': d.to_numpy()}, 42, 1)
    man = str(tmp_path / 'manifest.csv')
    ST._append_csv(man, ST.save_artifact(m, 'rf', 'h1_gbm', 'H1', 1, 'frozen', inp, pos,
                                         F.feature_code_digest()['_combined']), ST.MANIFEST_FIELDS)
    window = h1[h1.index <= pd.Timestamp('2026-09-30', tz=UTC)].iloc[-2000:]
    sources = {'h1': window[window.index <= RF.last_friday_close(sat)], 'daily': {}}
    return {'man': man, 'sat': sat, 'sources': sources, 'out': str(tmp_path / 'fwd'), 'h1': window}


def test_refit_appends_a_version_trained_before_its_cutoff(env):
    res = RF.run_refit(env['sat'], env['man'], env['out'], env['sources'])
    assert res['refitted'] == 1 and res['failed'] == 0
    man = pd.read_csv(env['man'])
    assert len(man) == 2
    new = man.iloc[-1]
    assert 'refit202609' in new['version_id']
    assert pd.Timestamp(new['train_end']) <= RF.last_friday_close(env['sat'])


def test_one_refit_per_month(env):
    RF.run_refit(env['sat'], env['man'], env['out'], env['sources'])
    again = RF.run_refit(env['sat'] + pd.Timedelta(hours=20), env['man'], env['out'], env['sources'])
    assert again['status'] == 'not due'
    assert len(pd.read_csv(env['man'])) == 2
    assert RF.run_refit(pd.Timestamp('2026-09-12 06:00', tz=UTC), env['man'], env['out'],
                        env['sources'])['status'] == 'not due'


def test_logger_uses_a_version_only_after_its_training_end(env):
    RF.run_refit(env['sat'], env['man'], env['out'], env['sources'])
    man = pd.read_csv(env['man'])
    refit_end = pd.Timestamp(man.iloc[-1]['train_end'])
    before = FL.latest_versions(refit_end, 'H1', env['man'])
    after = FL.latest_versions(refit_end + pd.Timedelta(hours=60), 'H1', env['man'])
    assert 'refit' not in before[0]['version_id']
    assert 'refit' in after[0]['version_id']
    for v in (before[0], after[0]):
        assert pd.Timestamp(v['train_end']) < refit_end + pd.Timedelta(hours=60)


def test_failed_refit_keeps_the_previous_version(env, monkeypatch):
    monkeypatch.setattr(RF, 'refit_cell', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    res = RF.run_refit(env['sat'], env['man'], env['out'], env['sources'])
    assert res['failed'] == 1
    assert len(pd.read_csv(env['man'])) == 1
    f = pd.read_csv(f"{env['out']}/refit_failures.csv")
    assert 'boom' in f['error'].iloc[0]


def test_history_is_extended_with_newer_mt5_bars():
    """Without the extension a refit retrains on the same frozen history."""
    h1 = F.load_h1()
    last = h1.index[-1]
    newer = h1.iloc[-30:].copy()
    newer.index = newer.index + pd.Timedelta(days=7)
    d1_new = pd.DataFrame({'open': [1.1], 'high': [1.2], 'low': [1.0], 'close': [1.15],
                           'tick_volume': [1.0]},
                          index=pd.DatetimeIndex([F.load_daily_ohlcv().index[-1] + pd.Timedelta(days=3)], tz='UTC'))
    src = RF.history_sources(newer.index[-1] + pd.Timedelta(hours=1), {'h1': newer, 'd1': d1_new}, macro=False)
    assert src['h1'].index[-1] > last
    assert src['daily']['price'].index[-1] == d1_new.index[0].tz_localize(None)
    cut = RF.history_sources(last, {'h1': newer}, macro=False)
    assert cut['h1'].index[-1] <= last, 'nothing after the cutoff'
