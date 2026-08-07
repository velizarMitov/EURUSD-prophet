"""
Tests for the Kronos VOLATILITY serving channel.

The direction channel was retired here after being measured to carry no
information three separate ways. These tests exist to stop it coming back by
accident, and to stop the served configuration drifting away from the one the
reported performance was measured at -- with a sampler, quietly changing a
parameter is easy and would silently invalidate every number in the response.
"""

import hashlib
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL_PATH = os.path.join(REPO, 'models', 'external_kronos', 'vol_calibration.json')


# ---------------------------------------------------------------------------
# 1. Serving constants == evaluation constants
# ---------------------------------------------------------------------------

def test_serving_constants_match_the_evaluation():
    """If any of these drifts, the reported performance describes something the
    endpoint no longer does. The values are the ones the clean-window run used."""
    from src.external.kronos import loader as kl
    assert kl.DEFAULT_MODEL == 'mini'
    assert kl.PRED_LEN == 24
    assert kl.SAMPLE_COUNT == 30
    assert kl.T == 1.0
    assert kl.TOP_P == 0.9
    assert kl.CONTEXT_BARS == 512
    assert kl.TRAILING_BARS == 24
    assert 'base' in kl.MODELS, 'base must stay loadable for comparison'


def test_calibration_was_fitted_at_the_serving_configuration():
    from src.external.kronos import loader as kl
    cal = json.load(open(CAL_PATH))
    assert cal['model'] == 'kronos-mini'
    assert cal['pred_len'] == kl.PRED_LEN
    assert cal['sample_count'] == kl.SAMPLE_COUNT
    assert cal['T'] == kl.T and cal['top_p'] == kl.TOP_P
    assert cal['context_bars'] == kl.CONTEXT_BARS


def test_calibration_from_a_different_configuration_is_refused(tmp_path):
    """A mapping fitted at another horizon must not be applied silently."""
    from src.external.kronos import loader as kl
    from src.external.kronos.vendor import KronosUnavailable
    bad = json.load(open(CAL_PATH))
    bad['pred_len'] = 1
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(bad))
    kl.reset_cache()
    with pytest.raises(KronosUnavailable, match='different configuration'):
        kl.load_calibration(str(p))


# ---------------------------------------------------------------------------
# 2. The calibration is LOADED, never refitted at request time
# ---------------------------------------------------------------------------

def test_calibration_is_loaded_not_refitted():
    from src.external.kronos import loader as kl
    from src.external.kronos.predict import apply_isotonic
    kl.reset_cache()
    cal = kl.load_calibration(CAL_PATH)
    disk = json.load(open(CAL_PATH))
    assert cal['isotonic']['x'] == disk['isotonic']['x']
    assert cal['isotonic']['y'] == disk['isotonic']['y']
    assert cal['scale']['constant'] == disk['scale']['constant']
    # The served mapping must equal the persisted one at every knot, up to the
    # persisted support clip.
    lo, hi = cal['output_clip']['lo'], cal['output_clip']['hi']
    for x, y in zip(disk['isotonic']['x'], disk['isotonic']['y']):
        assert apply_isotonic(x, cal) == pytest.approx(min(max(y, lo), hi), abs=1e-12)


def test_calibrated_probability_never_claims_certainty():
    """The top isotonic knot maps to exactly 1.0 on the strength of ONE
    fit-window row. Serving that verbatim would claim certainty from n=1."""
    from src.external.kronos import loader as kl
    from src.external.kronos.predict import apply_isotonic
    kl.reset_cache()
    cal = kl.load_calibration(CAL_PATH)
    n = cal['fit_window']['n']
    assert cal['output_clip']['lo'] == pytest.approx(1.0 / (n + 1))
    assert cal['output_clip']['hi'] == pytest.approx(1.0 - 1.0 / (n + 1))
    assert cal['output_clip']['knots_unchanged'] is True
    for p in (0.0, 0.5, 0.9667, 1.0):
        q = apply_isotonic(p, cal)
        assert 0.0 < q < 1.0, f'p_raw={p} produced a degenerate probability {q}'
    # Monotone, so the ranking the AUC was measured on is untouched.
    grid = np.linspace(0, 1, 101)
    vals = [apply_isotonic(p, cal) for p in grid]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_serving_path_never_imports_a_fitter():
    """Refitting at request time would re-tune the served numbers against
    whatever happened to be in the log. Neither serving module may reach for
    IsotonicRegression at all."""
    for rel in ('src/external/kronos/predict.py', 'src/external/kronos/vol_serving.py',
                'src/external/kronos/loader.py'):
        src = open(os.path.join(REPO, rel), encoding='utf-8').read()
        assert 'IsotonicRegression' not in src, f'{rel} can refit at request time'
        assert 'sklearn' not in src, f'{rel} imports sklearn'


def test_calibration_is_versioned_and_records_its_fit_window():
    cal = json.load(open(CAL_PATH))
    assert cal['fit_window']['n'] == 519
    for k in ('start', 'end', 'source', 'sampling'):
        assert cal['fit_window'][k]
    assert cal['fitted_utc'].endswith('Z')
    # Our corrections must be labelled as ours wherever they are described.
    assert 'OUR correction' in cal['scale']['meaning']
    assert 'OUR correction' in cal['isotonic']['meaning']


# ---------------------------------------------------------------------------
# 3 & 4. The direction channel is retired and nothing serves a direction
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
    sys.path.insert(0, REPO)
    import api
    return TestClient(api.app)


def test_direction_endpoint_returns_the_retirement_notice(client):
    r = client.get('/api/kronos-direction')
    assert r.status_code == 200
    d = r.json()
    assert d['available'] is False
    assert d['retired'] is True
    assert 'no information' in d['reason'] or 'retired' in d['reason']
    assert d['replacement'] == '/api/kronos-volatility'
    for banned in ('direction', 'p_up'):
        assert banned not in d or banned == 'direction'
    assert d.get('direction') is None, 'retired endpoint must not return a call'


def test_direction_view_is_reachable_and_carries_the_note(client):
    r = client.get('/kronos-direction')
    assert r.status_code == 200
    body = r.text
    assert 'retired' in body.lower()
    assert '/kronos-volatility' in body


def test_no_served_field_presents_a_kronos_direction_call(client):
    """The whole point of the retirement: no direction number anywhere."""
    r = client.get('/api/kronos-volatility')
    d = r.json()
    banned = {'direction', 'p_up', 'p_up_24', 'predicted_direction', 'signal'}
    assert not (banned & set(d)), f'direction leaked into the response: {banned & set(d)}'
    if d.get('available'):
        for k in d:
            assert 'direction' not in k.lower() or k == 'clean_window_result'


def test_dashboard_shows_volatility_not_direction():
    html = open(os.path.join(REPO, 'static', 'index.html'), encoding='utf-8').read()
    body = '\n'.join(l for l in html.splitlines()
                     if not l.strip().startswith(('//', '<!--')))
    fn = body.split('async function fetchKronos')[1].split('\n        }')[0]
    assert "'/api/kronos-volatility'" in fn
    assert "'/api/kronos-direction'" not in fn
    assert 'd.p_up' not in fn, 'dashboard still renders a Kronos direction probability'
    assert 'p_vol_amp_calibrated' in fn and 'pred_vol_pct_24h_scaled' in fn
    # Printed, never drawn -- the numbers are too coarse for a gauge.
    assert 'gauge' not in fn.lower()


def test_volatility_view_is_linked_from_every_other_view(client):
    for path in ('/', '/history', '/paper-trading', '/h1-direction'):
        r = client.get(path)
        assert r.status_code == 200, path
        assert '/kronos-volatility' in r.text, f'{path} does not link the volatility view'
    r = client.get('/kronos-volatility')
    assert r.status_code == 200
    for back in ('/', '/history', '/paper-trading', '/h1-direction'):
        assert 'href="%s"' % back in r.text, f'volatility view does not link back to {back}'


# ---------------------------------------------------------------------------
# 5. Graceful degradation
# ---------------------------------------------------------------------------

def test_degrades_when_the_calibration_is_missing(monkeypatch):
    from src.external.kronos import loader as kl
    kl.reset_cache()
    monkeypatch.setattr(kl, '_CAL_PATH', 'models/external_kronos/__absent__.json')
    ready, reason = kl.probe()
    assert ready is False
    assert 'calibration missing' in reason
    kl.reset_cache()


def test_degrades_when_the_package_is_uninstalled(client, monkeypatch):
    """Checkpoint removed AND package uninstalled: a clear unavailable, never a
    500, and never any effect on the other endpoints."""
    import api
    monkeypatch.setattr(api.service, 'kronos_ready', False, raising=False)
    monkeypatch.setattr(api.service, 'kronos_error', 'torch not installed', raising=False)
    r = client.get('/api/kronos-volatility')
    assert r.status_code == 200
    d = r.json()
    assert d['available'] is False and 'torch' in d['reason']
    # /api/predict is POST-only; the rest are views.
    assert client.post('/api/predict').status_code == 200, 'predict affected by Kronos being down'
    for path in ('/history', '/paper-trading', '/h1-direction'):
        assert client.get(path).status_code == 200, f'{path} affected by Kronos being down'


def test_prediction_failure_never_500s(client, monkeypatch):
    import api

    def boom(*a, **k):
        raise RuntimeError('checkpoint unavailable')
    monkeypatch.setattr(api.service, 'kronos_ready', True, raising=False)
    monkeypatch.setattr(api.service, 'predict_kronos_volatility', boom, raising=False)
    r = client.get('/api/kronos-volatility')
    assert r.status_code == 200
    assert r.json()['available'] is False
    assert 'checkpoint unavailable' in r.json()['reason']


# ---------------------------------------------------------------------------
# 6. Completed-bar rule
# ---------------------------------------------------------------------------

def _synthetic_h1(n=700, end='2025-03-14 12:00'):
    idx = pd.date_range(end=pd.Timestamp(end), periods=n * 2, freq='h', tz='UTC')
    keep = ~((idx.dayofweek == 5) | ((idx.dayofweek == 6) & (idx.hour < 22)))
    idx = idx[keep][-n:]
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 2e-4, len(idx)))
    return pd.DataFrame({'open': close, 'high': close + 1e-4,
                         'low': close - 1e-4, 'close': close,
                         'tick_volume': 100.0}, index=idx)


def test_completed_bar_rule_excludes_the_forming_hour():
    """The last context bar must never be the hour still forming."""
    from src.live_data import drop_incomplete_h1_bars
    frame = _synthetic_h1()
    now = pd.Timestamp('2025-03-14 12:37')          # 12:00 bar is still forming
    closed = drop_incomplete_h1_bars(frame.tz_localize(None) if frame.index.tz is None
                                     else frame.tz_convert('UTC').tz_localize(None), now=now)
    assert pd.Timestamp(closed.index[-1]) == pd.Timestamp('2025-03-14 11:00')


def test_forward_window_skips_non_trading_hours():
    """A 24-bar horizon spans a full trading day, so the window must be built on
    real bars -- date_range(freq='h') would hand the model 48 weekend hours."""
    from src.external.kronos.predict import forward_bar_index
    w = forward_bar_index(pd.Timestamp('2025-03-14 12:00'), 24)   # Friday
    assert len(w) == 24
    assert not any(t.dayofweek == 5 for t in w), 'Saturday bar in the forecast window'
    assert not any(t.dayofweek == 6 and t.hour < 22 for t in w)
    assert w[0] == pd.Timestamp('2025-03-14 12:00')


def test_volatility_object_labels_our_corrections_and_counts_24_returns():
    from src.external.kronos.predict import realised_vol, volatility_from_paths
    cal = json.load(open(CAL_PATH))
    rng = np.random.default_rng(1)
    last = 1.10
    paths = last * (1 + np.cumsum(rng.normal(0, 3e-4, (30, 24)), axis=1))
    trailing = last * (1 + np.cumsum(rng.normal(0, 3e-4, 25)))
    trailing[-1] = last
    out = volatility_from_paths(paths, last, trailing, cal)
    assert 0.0 <= out['p_vol_amp_raw'] <= 1.0
    assert 0.0 <= out['p_vol_amp_calibrated'] <= 1.0
    assert out['pred_vol_pct_24h'] > 0 and out['trailing_rv_24h_pct'] > 0
    # The scaled field is the raw one divided by the frozen constant.
    assert out['pred_vol_pct_24h_scaled'] == pytest.approx(
        out['pred_vol_pct_24h'] / cal['scale']['constant'], rel=1e-9)
    assert 'OUR correction' in out['pred_vol_scale_note']
    # eq.12 anchoring: 25 closes -> 24 returns on every side.
    assert realised_vol(np.arange(1.0, 1.25, 0.01)) > 0
    with pytest.raises(RuntimeError, match='trailing closes'):
        volatility_from_paths(paths, last, trailing[:5], cal)


# ---------------------------------------------------------------------------
# 7. Protected set
# ---------------------------------------------------------------------------

def test_protected_set_sha256_identical():
    for name in ('h_dir_final_protected_sha256.json', 'macro_tier_a_protected_sha256.json',
                 'h1_production_protected_sha256.json', 'kronos_protected_sha256.json'):
        p = os.path.join(REPO, 'tests', 'fixtures', name)
        if not os.path.exists(p):
            continue
        for rel, digest in json.load(open(p)).items():
            fp = os.path.join(REPO, rel)
            assert os.path.isfile(fp), f'{name}: protected file vanished: {rel}'
            got = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
            assert got == digest, f'{name}: PROTECTED FILE MODIFIED: {rel}'


def test_forbidden_files_were_not_touched():
    """The boundary list for this program, asserted by name."""
    import subprocess
    forbidden = ['_train_pipeline.py', 'src/features.py', 'src/volatility.py',
                 'src/paper_trading.py', 'config.json']
    out = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--'] + forbidden,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip('git unavailable')
    assert out.stdout.strip() == '', f'forbidden files modified:\n{out.stdout}'


# ---------------------------------------------------------------------------
# ledger settlement
# ---------------------------------------------------------------------------

def test_ledger_settles_only_fully_closed_windows(tmp_path):
    from src.external.kronos import vol_serving as vs
    frame = _synthetic_h1(n=200, end='2025-03-14 12:00')
    closes = frame['close']
    closes.index = closes.index.tz_localize(None)

    # anchor_full has 99 real bars of runway behind it in this series, plenty
    # for a 24-bar window to close. anchor_full + 24 real ticks == the CLOSE
    # of the 24th forecast bar -- the window is (anchor, end], inclusive of
    # end, matching forward_bar_index / forecast_window_end in
    # src/inference.py, so this is exactly what a real served response would
    # log as forecast_window_end for a 24-bar horizon.
    anchor_full = closes.index[100]
    full_end = closes.index[124]
    # anchor_partial sits 9 bars from the end of the series, so a 24-bar
    # window from it genuinely has not printed yet -- distinct from anchor_full
    # so it cannot collide with it under any dedup key.
    anchor_partial = closes.index[190]
    partial_end = anchor_partial + pd.Timedelta(hours=24)

    scenarios = [
        (anchor_full, full_end, 'open', 'full-open'),
        (anchor_partial, partial_end, 'open', 'partial-open'),
        (anchor_full, full_end, 'clock_unconfirmed', 'full-unconfirmed'),
    ]
    rows = []
    for anchor, end, status, tag in scenarios:
        r = {c: 0 for c in vs.LOG_COLUMNS}
        r.update({'as_of_bar_close': anchor.isoformat(),
                  'forecast_window_end': pd.Timestamp(end).isoformat(),
                  'forecast_bar_status': status, 'trailing_rv_24h_pct': 0.30,
                  'pred_vol_pct_24h': 0.25, 'pred_vol_pct_24h_scaled': 0.38,
                  'p_vol_amp_raw': 0.10, 'p_vol_amp_calibrated': 0.40,
                  'model_version': 'kronos-mini@test-' + tag})
        rows.append(r)
    log = tmp_path / 'log.csv'
    pd.DataFrame(rows, columns=vs.LOG_COLUMNS).to_csv(log, index=False)

    led = vs.build_ledger(log_path=str(log), base_dir='', closes=closes)
    assert len(led) == 1, 'only the fully-closed, clock-confirmed window may settle'
    row = led.iloc[0]
    assert row['model_version'] == 'kronos-mini@test-full-open'
    assert row['settled_bars'] == 24
    assert row['amplified'] in (0, 1)
    assert row['brier_raw'] == pytest.approx((0.10 - row['amplified']) ** 2)
    assert row['brier_calibrated'] == pytest.approx((0.40 - row['amplified']) ** 2)
    assert row['persistence_abs_err_pct'] == pytest.approx(abs(0.30 - row['realised_rv_24h_pct']))


def _vol_log_row(anchor, end, status='open', model_version='kronos-mini@test'):
    """One synthetic LOG_COLUMNS-shaped row, matching the style build_ledger
    actually receives (real fields we score against, zero-filled elsewhere)."""
    from src.external.kronos import vol_serving as vs
    r = {c: 0 for c in vs.LOG_COLUMNS}
    r.update({'as_of_bar_close': pd.Timestamp(anchor).isoformat(),
              'forecast_window_end': pd.Timestamp(end).isoformat(),
              'forecast_bar_status': status, 'trailing_rv_24h_pct': 0.30,
              'pred_vol_pct_24h': 0.25, 'pred_vol_pct_24h_scaled': 0.38,
              'p_vol_amp_raw': 0.10, 'p_vol_amp_calibrated': 0.40,
              'model_version': model_version})
    return r


def _hourly_closes(n=40, start='2025-06-02 00:00'):
    """Monday 00:00 UTC, n contiguous real hourly bars -- no weekend inside a
    40h span, so every position is exactly 1 real tick from its neighbour."""
    idx = pd.date_range(start=start, periods=n, freq='h')
    return pd.Series(1.10 + np.arange(n) * 1e-4, index=idx)


def test_fully_closed_window_settles_with_exactly_horizon_bars(tmp_path):
    """A hand-built row whose 24-bar window has fully closed settles, and the
    settled bar count equals horizon_bars=24 under the (anchor, end] convention
    -- forecast_window_end is the CLOSE of the 24th forecast bar (established
    by src/inference.py::predict_kronos_volatility, which sets it to
    forward_bar_index(anchor, horizon_bars)[-1] + 1h), so end itself is one of
    the 24 bars the forecast covers, not one bar past it."""
    from src.external.kronos import vol_serving as vs
    closes = _hourly_closes()
    anchor = pd.Timestamp('2025-06-02 05:00')
    end = anchor + pd.Timedelta(hours=24)

    log = tmp_path / 'log.csv'
    pd.DataFrame([_vol_log_row(anchor, end)], columns=vs.LOG_COLUMNS).to_csv(log, index=False)

    led = vs.build_ledger(log_path=str(log), base_dir='', closes=closes)
    assert len(led) == 1
    assert led.iloc[0]['settled_bars'] == 24
    assert led.iloc[0]['realised_rv_24h_pct'] > 0


def test_still_open_window_does_not_settle(tmp_path):
    """A window whose end has not yet printed in the closes series must not
    settle -- there is no realised volatility for a horizon still in the
    future, whatever `forecast_bar_status` claims."""
    from src.external.kronos import vol_serving as vs
    closes = _hourly_closes(n=20)                   # only 19 real ticks after 00:00
    anchor = pd.Timestamp('2025-06-02 05:00')
    end = anchor + pd.Timedelta(hours=24)            # far beyond the last close

    log = tmp_path / 'log.csv'
    pd.DataFrame([_vol_log_row(anchor, end)], columns=vs.LOG_COLUMNS).to_csv(log, index=False)

    led = vs.build_ledger(log_path=str(log), base_dir='', closes=closes)
    assert len(led) == 0


def test_closed_window_with_missing_bar_does_not_settle(tmp_path):
    """A window whose end has printed but whose price series has a hole inside
    it must not settle -- a partial window would understate realised
    volatility and flatter the forecast, which is worse than not settling."""
    from src.external.kronos import vol_serving as vs
    closes = _hourly_closes()
    anchor = pd.Timestamp('2025-06-02 05:00')
    end = anchor + pd.Timedelta(hours=24)
    gap_closes = closes.drop(pd.Timestamp('2025-06-02 15:00'))   # inside the window

    log = tmp_path / 'log.csv'
    pd.DataFrame([_vol_log_row(anchor, end)], columns=vs.LOG_COLUMNS).to_csv(log, index=False)

    led = vs.build_ledger(log_path=str(log), base_dir='', closes=gap_closes)
    assert len(led) == 0


def test_multiple_calls_inside_one_window_settle_once(tmp_path):
    """Refreshing the page during a still-open window logs several rows with
    the same anchor; once the window closes they must collapse to exactly one
    settled observation, not one per call."""
    from src.external.kronos import vol_serving as vs
    closes = _hourly_closes()
    anchor = pd.Timestamp('2025-06-02 05:00')
    end = anchor + pd.Timedelta(hours=24)

    rows = [_vol_log_row(anchor, end) for _ in range(4)]
    log = tmp_path / 'log.csv'
    pd.DataFrame(rows, columns=vs.LOG_COLUMNS).to_csv(log, index=False)

    led = vs.build_ledger(log_path=str(log), base_dir='', closes=closes)
    assert len(led) == 1


def test_regression_pins_the_settlement_off_by_one(tmp_path):
    """Pins the exact reported bug: a 24-hour window anchored on the hour must
    settle under the fixed (inclusive) convention, and the same window would
    NOT have settled under the strict-exclusive interval this project shipped
    with -- (anchor, end) instead of (anchor, end], which undercounts every
    window by exactly one bar and can never reach horizon_bars=24."""
    from src.external.kronos import vol_serving as vs
    closes = _hourly_closes()
    anchor = pd.Timestamp('2025-06-02 03:00')
    end = anchor + pd.Timedelta(hours=24)

    rv, n = vs._realised(closes, anchor, end)
    assert n == 24, 'fixed convention must settle the full 24-bar window'
    assert rv is not None

    from src.external.kronos import loader as kloader
    didx = pd.DatetimeIndex(closes.index)
    a = didx[didx <= anchor][-1]
    strict_exclusive_window = closes[(didx > a) & (didx < end)]
    assert len(strict_exclusive_window) == 23, (
        'sanity check on the bug as reported: the old exclusive bound gives 23')
    assert len(strict_exclusive_window) < kloader.PRED_LEN, (
        'the strict interval must never reach horizon_bars, which is exactly why '
        'the ledger never settled anything before this fix')


def test_target_settled_is_sized_to_this_channels_effect():
    """500 was carried over from the DIRECTION channel, whose effect was ~20x
    smaller. The target must be sized to the AUC this channel actually measured,
    and se_above_half must reproduce the derivation recorded beside it."""
    from src.external.kronos import vol_serving as vs
    assert vs.TARGET_SETTLED == 100
    assert vs.se_above_half(50) == pytest.approx(2.04, abs=0.05)
    assert vs.se_above_half(100) == pytest.approx(2.89, abs=0.05)
    assert vs.se_above_half(150) == pytest.approx(3.53, abs=0.05)
    # Undefined, never a crash or a fake zero, below n=2.
    assert vs.se_above_half(0) is None and vs.se_above_half(1) is None
    # Monotone in n: more evidence never reads as weaker.
    vals = [vs.se_above_half(n) for n in range(2, 200)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_page_reports_se_for_the_current_n_not_only_the_fraction():
    from src.external.kronos import vol_serving as vs
    anchor = pd.Timestamp('2025-06-02 05:00')
    # >= 2 rows: below that se_above_half is undefined and the page says so
    # instead, which is a different branch (covered by the None assertions above).
    rows = [_vol_log_row(anchor, anchor + pd.Timedelta(hours=24)) for _ in range(4)]
    led = pd.DataFrame(rows)
    led['amplified'] = [1, 0, 1, 0]
    led['realised_rv_24h_pct'] = 0.4
    led = led.reindex(columns=vs.LEDGER_COLUMNS)
    html = vs.render_html(led, pd.DataFrame(rows, columns=vs.LOG_COLUMNS))
    assert 'SE</strong> above' in html, 'page does not show SE above 0.5 for the current n'
    assert 'about POWER, not a forward result' in html, 'SE line must not read as a result'
    assert '~%d needed' % vs.TARGET_SETTLED in html


def test_constant_brier_caution_shows_only_while_outcomes_are_uniform():
    """A constant forecast is trivially perfect on a one-outcome sample, so
    'Brier const. 0.0000' must carry a caution until both outcomes appear."""
    from src.external.kronos import vol_serving as vs
    base = _vol_log_row(pd.Timestamp('2025-06-02 05:00'), pd.Timestamp('2025-06-03 05:00'))

    def _led(amps):
        d = pd.DataFrame([dict(base) for _ in amps])
        d['amplified'] = amps
        d['realised_rv_24h_pct'] = 0.4
        return d.reindex(columns=vs.LEDGER_COLUMNS)

    uniform = vs.render_html(_led([1, 1, 1, 1]), pd.DataFrame([base], columns=vs.LOG_COLUMNS))
    assert 'by construction' in uniform
    assert '4 of 4 amplified' in uniform

    mixed = vs.render_html(_led([1, 0, 1, 1]), pd.DataFrame([base], columns=vs.LOG_COLUMNS))
    assert 'by construction' not in mixed, 'caution must disappear once outcomes vary'

    empty = vs.render_html(pd.DataFrame(columns=vs.LEDGER_COLUMNS),
                           pd.DataFrame(columns=vs.LOG_COLUMNS))
    assert 'by construction' not in empty


def test_view_states_the_gate_and_never_shows_a_direction():
    from src.external.kronos import vol_serving as vs
    html = vs.render_html(pd.DataFrame(columns=vs.LEDGER_COLUMNS),
                          pd.DataFrame(columns=vs.LOG_COLUMNS))
    assert 'OBSERVATIONAL' in html
    assert 'volatility ensemble' in html and 'never been made' in html
    assert 'post-hoc' in html.lower()
    assert 'p_up' not in html
