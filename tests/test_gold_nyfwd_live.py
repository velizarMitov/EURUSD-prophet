"""gold-nyrange-forward, group 4: the forecast and settlement passes on fake feeds."""
import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import load_config as gb_config
from src.gold_bbnr import ny_features as NF
from src.gold_bbnr.bars import d1_ny
from src.gold_nyfwd import contracts as K
from src.gold_nyfwd import live as LV
from src.gold_nyfwd import prereg as PR
from tests._gold_nyfwd_fakes import GOLD, FakeFeed, cfg_in, common_files_in_tmp, listed, market, no_gate  # noqa: F401

DAY = pd.Timestamp('2026-07-15', tz='UTC')                 # a Wednesday
AT_1608_SOFIA = DAY + pd.Timedelta(hours=15, minutes=8)    # label clock = Berlin = Sofia - 1 h


@pytest.fixture
def cfg(tmp_path):
    c = cfg_in(tmp_path)
    PR.write(c)
    return c


@pytest.fixture(scope='module')
def bars():
    return market()


def forecast(cfg, bars, now, **kw):
    return LV.run_forecast(cfg, FakeFeed(bars, listed(), now), gate=no_gate, **kw)


def settle(cfg, bars, now):
    return LV.run_settle(cfg, FakeFeed(bars, listed(), now), gate=no_gate)


def strip(row):
    return {k: v for k, v in row.items() if k != 'logged_at'}


def test_a_start_at_14_50_logs_nothing(cfg, bars):
    out = forecast(cfg, bars, DAY + pd.Timedelta(hours=14, minutes=50))
    assert out['status'].startswith('too early')
    assert not LV.read_log(cfg, 'forecasts').shape[0]


def test_normal_day_logs_one_ok_row_and_a_second_run_changes_nothing(cfg, bars):
    out = forecast(cfg, bars, AT_1608_SOFIA)
    assert out['status'] == 'logged' and out['row']['status'] == 'ok', out
    r = out['row']
    assert r['contract'] == 'USDIndSep26' and r['F'] == pytest.approx(np.exp(r['p_harx']) * r['close14'])
    assert r['close14'] == pytest.approx(bars[GOLD].loc[DAY + pd.Timedelta(hours=14), 'close'], rel=1e-14)
    before = open(LV.log_path(cfg, 'forecasts'), 'rb').read()
    assert forecast(cfg, bars, AT_1608_SOFIA + pd.Timedelta(minutes=30)) == {'status': 'already logged'}
    assert open(LV.log_path(cfg, 'forecasts'), 'rb').read() == before
    assert LV.read_log(cfg, 'forecasts').shape[0] == 1


def test_a_pass_at_18_00_gives_the_same_row_as_one_at_16_08(tmp_path, bars):
    a, b = cfg_in(tmp_path / 'a'), cfg_in(tmp_path / 'b')
    PR.write(a)
    PR.write(b)
    ra = forecast(a, bars, AT_1608_SOFIA)['row']
    rb = forecast(b, bars, DAY + pd.Timedelta(hours=17))['row']
    assert strip(ra) == strip(rb)


def test_features_equal_gold_bbnr_on_the_full_history_and_use_no_later_bar(cfg, bars):
    row = forecast(cfg, bars, AT_1608_SOFIA)['row']
    gb = gb_config()
    model = PR.coefficients(cfg)
    gold = bars[GOLD]
    dates = pd.DatetimeIndex(sorted(set(gold.index.normalize())))
    mp = K.day_map(dates, K.read(cfg), 2)
    ds = NF.dataset(gold, d1_ny(gold, gb), K.dollar_frame(dates, mp, bars.__getitem__), gb)   # every bar, all day
    x = ds.loc[DAY, model['columns']].to_numpy(float)
    assert not np.isnan(x).any()
    assert row['p_harx'] == pytest.approx(model['coef_harx'][0] + x @ np.array(model['coef_harx'][1:]), rel=1e-12)
    xh = ds.loc[DAY, model['har_columns']].to_numpy(float)
    assert row['p_har'] == pytest.approx(model['coef_har'][0] + xh @ np.array(model['coef_har'][1:]), rel=1e-12)


def test_a_missing_13_00_bar_logs_its_reason_without_a_forecast(cfg, bars):
    holed = dict(bars)
    holed[GOLD] = bars[GOLD].drop(DAY + pd.Timedelta(hours=13))
    out = forecast(cfg, holed, AT_1608_SOFIA)
    assert out['row']['status'] == 'missing pre-session bar' and 'p_harx' not in out['row']
    assert LV.read_log(cfg, 'forecasts')['p_harx'].isna().all()


def test_early_history_is_warm_up(cfg, bars):
    out = forecast(cfg, bars, pd.Timestamp('2026-04-20 15:08', tz='UTC'))
    assert out['row']['status'] == 'warm-up'


def test_no_listed_contract_after_the_day_is_no_dollar_data(cfg, bars):
    out = LV.run_forecast(cfg, FakeFeed(bars, listed(expiry='2026-07-16'), AT_1608_SOFIA), gate=no_gate)
    assert out['row']['status'] == 'no dollar data'


def test_a_shifted_gold_week_is_unverified(cfg, bars):
    shifted = dict(bars)
    g = bars[GOLD]
    monday = pd.Timestamp('2026-07-13', tz='UTC')
    g = g.drop(monday)                                         # this week's open now lands at 01:00
    shifted[GOLD] = g
    out = forecast(cfg, shifted, AT_1608_SOFIA)
    assert out['row']['status'] == 'unverified clock week'


def test_an_uncommitted_prereg_refuses_and_logs_nothing(cfg, bars):
    def gate(_c):
        raise PR.PreregNotCommitted('PRE_REGISTRATION.json is not committed')
    with pytest.raises(PR.PreregNotCommitted):
        LV.run_forecast(cfg, FakeFeed(bars, listed(), AT_1608_SOFIA), gate=gate)
    assert not LV.read_log(cfg, 'forecasts').shape[0]


def test_an_altered_ols_json_makes_the_pass_refuse(cfg, bars, tmp_path):
    fake = tmp_path / 'ols.json'
    fake.write_text('{"edited": true}', encoding='utf-8')
    with pytest.raises(PR.FrozenChanged):
        forecast(cfg, bars, AT_1608_SOFIA, ols_path=str(fake))


# -- settlement ------------------------------------------------------------------------------------

def test_settlement_once_per_day_with_gold_bbnr_target(cfg, bars):
    f = forecast(cfg, bars, AT_1608_SOFIA)['row']
    assert settle(cfg, bars, DAY + pd.Timedelta(hours=21, minutes=30))['status'].startswith('too early')
    out = settle(cfg, bars, DAY + pd.Timedelta(hours=22, minutes=8))
    assert out['status'] == 'settled' and len(out['rows']) == 1
    s = out['rows'][0]
    w = bars[GOLD][(bars[GOLD].index >= DAY + pd.Timedelta(hours=15)) & (bars[GOLD].index <= DAY + pd.Timedelta(hours=21))]
    y = np.log((w['high'].max() - w['low'].min()) / f['close14'])
    assert s['status'] == 'ok' and s['y'] == pytest.approx(y, rel=1e-12)
    assert s['se_harx'] == pytest.approx((y - f['p_harx']) ** 2) and s['se_har'] == pytest.approx((y - f['p_har']) ** 2)
    assert s['realised_high'] == w['high'].max() and s['realised_low'] == w['low'].min()
    before = open(LV.log_path(cfg, 'settlements'), 'rb').read()
    assert settle(cfg, bars, DAY + pd.Timedelta(hours=22, minutes=40)) == {'status': 'nothing to settle'}
    assert open(LV.log_path(cfg, 'settlements'), 'rb').read() == before


def test_an_early_close_is_incomplete_session_without_squared_errors(cfg, bars):
    forecast(cfg, bars, AT_1608_SOFIA)
    early = dict(bars)
    early[GOLD] = bars[GOLD].drop([DAY + pd.Timedelta(hours=h) for h in (19, 20, 21, 22)])
    assert settle(cfg, early, DAY + pd.Timedelta(hours=22, minutes=8))['status'].startswith('waiting')
    out = settle(cfg, early, DAY + pd.Timedelta(days=1, hours=22, minutes=8))
    s = out['rows'][0]
    assert s['status'] == 'incomplete session' and 'se_harx' not in s and 'y' not in s
    assert LV.read_log(cfg, 'settlements')['se_harx'].isna().all()


def test_a_weekday_with_no_forecast_is_never_settled(cfg, bars):
    forecast(cfg, bars, AT_1608_SOFIA)
    nxt = DAY + pd.Timedelta(days=1)                     # PC off at 16:08 on Thursday: no forecast row
    out = settle(cfg, bars, nxt + pd.Timedelta(hours=22, minutes=8))
    assert [r['date'] for r in out['rows']] == [DAY]
    assert settle(cfg, bars, nxt + pd.Timedelta(days=1, hours=22, minutes=8)) == {'status': 'nothing to settle'}
    assert set(LV.read_log(cfg, 'settlements')['date']) == {DAY}
