"""
Tests for the H_dir.1 PRODUCTION INTEGRATION (observational serving).

The load-bearing ones are the CONTRACT tests: feature parity between the
canonical module and the research module it was extracted from, and the
completed-bar rule that keeps a still-forming hour from ever becoming the
prediction base. Both guards are shown to BITE, not merely to pass.
"""

import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, 'models', 'h1_direction')
BACKUP_DIR = MODEL_DIR + '.bak_test'
SHA_FIXTURE = os.path.join(REPO, 'tests', 'fixtures',
                           'h1_production_protected_sha256.json')


def _restore_artifacts_if_a_previous_run_died():
    """
    SELF-HEALING. The degradation test moves models/h1_direction/ aside and puts
    it back in a `finally`. If that process is ever KILLED mid-test the artifacts
    would be left renamed and serving would stay down, so restore them at import
    time rather than requiring a human to notice.
    """
    if os.path.isdir(BACKUP_DIR) and not os.path.isdir(MODEL_DIR):
        shutil.move(BACKUP_DIR, MODEL_DIR)


_restore_artifacts_if_a_previous_run_died()


def _h1_frame(n=400, start='2024-01-01 00:00', freq='h', seed=0):
    """A deterministic synthetic H1 OHLC frame long enough for SMA200 warm-up."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    return pd.DataFrame({'open': close - 0.0001, 'high': close + 0.0003,
                         'low': close - 0.0003, 'close': close}, index=idx)


# ── 1. THE FEATURE-PARITY GATE ────────────────────────────────────────────────

def test_h1_features_are_byte_identical_to_the_research_module():
    """The single-source-of-truth contract. If this fails, the extraction is
    unfaithful and nothing downstream of it is valid."""
    from src.h1_features import (assert_feature_parity, compute_h1_direction_features,
                                 DIRECTION_FEATURE_COLUMNS)
    from src.pooled_h1_model import compute_pooled_features, FEATURE_COLUMNS

    df = _h1_frame(600)
    a = compute_h1_direction_features(df)
    b = compute_pooled_features(df)

    assert list(a.columns) == list(b.columns) == FEATURE_COLUMNS
    assert DIRECTION_FEATURE_COLUMNS == FEATURE_COLUMNS
    assert a.to_numpy().tobytes() == b.to_numpy().tobytes()
    assert (a.isna().to_numpy() == b.isna().to_numpy()).all()
    assert_feature_parity(a, b)          # does not raise


def test_parity_guard_fires_on_a_deliberately_altered_copy():
    """NON-VACUOUS: the gate must detect a drifted extraction, including one
    that differs by a single float and one that only reorders columns."""
    from src.h1_features import assert_feature_parity, compute_h1_direction_features

    df = _h1_frame(400)
    good = compute_h1_direction_features(df)

    tweaked = good.copy()
    tweaked.iloc[-1, 0] = float(tweaked.iloc[-1, 0]) + 1e-12
    with pytest.raises(AssertionError, match='VALUE mismatch'):
        assert_feature_parity(tweaked, good)

    reordered = good[list(good.columns[::-1])]
    with pytest.raises(AssertionError, match='COLUMN mismatch'):
        assert_feature_parity(reordered, good)

    with pytest.raises(AssertionError, match='SHAPE mismatch'):
        assert_feature_parity(good.iloc[:-1], good)


# ── 2 / a. THE BUILD-TIME PIPELINE CHECK ──────────────────────────────────────

def test_build_time_check_refuses_to_write_artifacts_on_drift(tmp_path, monkeypatch):
    """The shipped full-history model has no held-out data, so this check is the
    only thing between a broken pipeline and a silently wrong model."""
    from src import train_h1_direction as T

    out = tmp_path / 'h1_direction'
    monkeypatch.setattr(T, 'REPRO_TARGET_ACC', 0.90)      # force drift
    with pytest.raises(T.PipelineCheckError):
        T.train(model_dir=str(out), write=True, verbose=False)
    assert not out.exists() or not any(out.iterdir()), 'artifacts written despite drift'


def test_reproduction_gate_is_two_sided_and_pinned():
    from src import train_h1_direction as T
    assert T.REPRO_TARGET_ACC == 0.527462 and T.REPRO_TOLERANCE == 0.003
    meta = _meta()
    chk = meta['build_time_pipeline_check']
    assert chk['passed'] is True
    assert abs(chk['check_val_accuracy'] - T.REPRO_TARGET_ACC) <= T.REPRO_TOLERANCE


# ── 3. THE COMPLETED-BAR RULE ─────────────────────────────────────────────────

def test_drop_incomplete_h1_bars_removes_forming_hour_and_weekend_bars():
    from src.live_data import drop_incomplete_h1_bars, H1_WEEKLY_OPEN_HOUR

    # 2024-01-05 is a Friday, 06 Saturday, 07 Sunday, 08 Monday.
    idx = pd.DatetimeIndex([
        '2024-01-05 21:00', '2024-01-05 22:00',        # Fri, keep
        '2024-01-06 03:00', '2024-01-06 23:00',        # SAT, drop both
        '2024-01-07 10:00', '2024-01-07 21:00',        # Sun pre-open, drop
        '2024-01-07 22:00', '2024-01-07 23:00',        # Sun weekly open, KEEP
        '2024-01-08 00:00', '2024-01-08 01:00',        # Mon, keep / forming
    ])
    df = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0}, index=idx)

    kept = drop_incomplete_h1_bars(df, now=pd.Timestamp('2024-01-08 01:30'))
    got = list(kept.index)
    assert pd.Timestamp('2024-01-08 01:00') not in got, 'forming hour survived'
    assert pd.Timestamp('2024-01-08 00:00') in got
    assert not any(t.weekday() == 5 for t in got), 'a Saturday bar survived'
    assert pd.Timestamp('2024-01-07 10:00') not in got
    assert pd.Timestamp('2024-01-07 21:00') not in got
    # The genuine weekly-open bars are IN the training distribution and stay.
    assert pd.Timestamp('2024-01-07 22:00') in got
    assert pd.Timestamp('2024-01-07 23:00') in got
    assert H1_WEEKLY_OPEN_HOUR == 22

    # Both directions: one hour later the previously-forming bar is complete.
    later = drop_incomplete_h1_bars(df, now=pd.Timestamp('2024-01-08 02:00'))
    assert pd.Timestamp('2024-01-08 01:00') in list(later.index)


def test_feed_clock_is_inferred_from_the_feed_not_the_wall_clock():
    """Bar labels carry BROKER SERVER time. Comparing them against utcnow is
    hours too conservative and against local time hours too permissive; both
    were observed live before this inference existed."""
    from src.live_data import infer_h1_feed_now

    # state_path=None makes this stateless: it neither reads nor WRITES the
    # production offset file, so the raw inference is what is under test here
    # and a unit test can never pollute live serving state.
    def infer(idx, now):
        return infer_h1_feed_now(idx, now_utc=now, state_path=None)

    utcnow = pd.Timestamp('2026-07-30 18:16:00')
    # Broker at UTC+2: the forming bar is labelled 20:00.
    idx = pd.date_range('2026-07-30 10:00', '2026-07-30 20:00', freq='h', tz='UTC')
    assert infer(idx, utcnow) == pd.Timestamp('2026-07-30 20:16:00')

    # Holds anywhere inside the forming bar, including its very end.
    assert infer(idx, pd.Timestamp('2026-07-30 18:59:59')) == \
        pd.Timestamp('2026-07-30 20:59:59')

    # A STALE feed falls back to plain UTC rather than inventing an offset.
    stale = pd.date_range('2026-07-20 00:00', '2026-07-20 10:00', freq='h', tz='UTC')
    assert infer(stale, utcnow) == utcnow


def test_unit_tests_never_write_the_production_offset_state(tmp_path):
    """Guard the guard: a stateless call must leave the real state file alone."""
    from src.live_data import H1_OFFSET_STATE_PATH, infer_h1_feed_now

    before = (open(os.path.join(REPO, H1_OFFSET_STATE_PATH)).read()
              if os.path.exists(os.path.join(REPO, H1_OFFSET_STATE_PATH)) else None)
    idx = pd.date_range('2026-07-30 10:00', '2026-07-30 20:00', freq='h', tz='UTC')
    infer_h1_feed_now(idx, now_utc=pd.Timestamp('2026-07-30 18:16:00'),
                      state_path=None)
    after = (open(os.path.join(REPO, H1_OFFSET_STATE_PATH)).read()
             if os.path.exists(os.path.join(REPO, H1_OFFSET_STATE_PATH)) else None)
    assert before == after


# ── the STICKY offset: the top-of-hour emit-lag artifact ──────────────────────

def _state_file(tmp_path, offset=None, accepted_at='2026-07-30T17:30:00'):
    """A persisted-offset state file, optionally pre-seeded."""
    p = tmp_path / 'h1_feed_offset.json'
    if offset is not None:
        p.write_text(json.dumps({'offset_hours': offset,
                                 'accepted_at_utc': accepted_at,
                                 'pending': None, 'history': []}))
    return str(p)


def test_top_of_hour_emit_lag_does_not_shift_the_offset(tmp_path):
    """
    THE BUG THIS FIXES. At utcnow 18:00:05 the true server clock is 20:00:05
    (offset +2), but MT5 has not emitted the 20:00 bar yet, so the newest label
    is still 19:00. diff = 0.9986h, ceil = 1 -> the raw inference says +1 and the
    service would treat an ALREADY CLOSED hour as the open forecast bar.
    """
    from src.live_data import _raw_feed_offset, infer_h1_feed_now

    utcnow = pd.Timestamp('2026-07-30 18:00:05')
    idx = pd.date_range('2026-07-30 10:00', '2026-07-30 19:00', freq='h', tz='UTC')

    # FIRST: drive the UNFIXED path and show the wrong value it produces.
    raw_offset, diff_hours, margin_min = _raw_feed_offset(idx, utcnow)
    assert raw_offset == 1, 'the artifact must actually reproduce'
    assert diff_hours == pytest.approx(0.99861, abs=1e-4)
    assert margin_min < 2.0, 'the artifact sits within minutes of the boundary'
    naive_now = utcnow + pd.Timedelta(hours=raw_offset)
    assert naive_now == pd.Timestamp('2026-07-30 19:00:05'), 'unfixed value'

    # THEN: with the offset persisted, stickiness must reject it.
    state = _state_file(tmp_path, offset=2)
    got = infer_h1_feed_now(idx, now_utc=utcnow, state_path=state)
    assert got == pd.Timestamp('2026-07-30 20:00:05'), 'emit lag shifted the clock'
    assert got != naive_now

    # The provisional value is recorded but NOT accepted.
    saved = json.loads(open(state).read())
    assert saved['offset_hours'] == 2
    assert saved['pending']['offset_hours'] == 1
    assert saved['history'] == [], 'no change may be logged for an unconfirmed flip'

    # And the moment the bar IS emitted, the pending flip is discarded.
    idx2 = pd.date_range('2026-07-30 10:00', '2026-07-30 20:00', freq='h', tz='UTC')
    later = infer_h1_feed_now(idx2, now_utc=pd.Timestamp('2026-07-30 18:00:20'),
                              state_path=state)
    assert later == pd.Timestamp('2026-07-30 20:00:20')
    assert json.loads(open(state).read())['pending'] is None


def test_a_genuine_dst_change_is_accepted(tmp_path):
    """Two fetches ten minutes apart with diff_hours well clear of the boundary.
    A real transition confirms; the artifact never does."""
    from src.live_data import _raw_feed_offset, infer_h1_feed_now

    state = _state_file(tmp_path, offset=2)
    # Broker falls back to +1 (CET). At 18:25 the forming bar is labelled 19:00.
    utcnow1 = pd.Timestamp('2026-10-25 18:25:00')
    idx = pd.date_range('2026-10-25 10:00', '2026-10-25 19:00', freq='h', tz='UTC')
    _o, _d, margin = _raw_feed_offset(idx, utcnow1)
    assert margin > 2.0, 'this fixture must sit clear of the boundary'

    got1 = infer_h1_feed_now(idx, now_utc=utcnow1, state_path=state)
    assert got1 == pd.Timestamp('2026-10-25 19:25:00')          # offset +1 accepted

    utcnow2 = utcnow1 + pd.Timedelta(minutes=10)
    got2 = infer_h1_feed_now(idx, now_utc=utcnow2, state_path=state)
    assert got2 == pd.Timestamp('2026-10-25 19:35:00')          # and it sticks

    saved = json.loads(open(state).read())
    assert saved['offset_hours'] == 1
    assert len(saved['history']) == 1
    entry = saved['history'][0]
    assert entry['previous_offset_hours'] == 2 and entry['offset_hours'] == 1
    assert entry['accepted_at_utc'] == utcnow1.isoformat()      # logged with its time


def test_boundary_adjacent_change_is_accepted_only_after_confirmation(tmp_path):
    """The other acceptance route: a change that DOES sit near the boundary must
    repeat at least 5 minutes after it was first seen."""
    from src.live_data import H1_OFFSET_CONFIRM_MINUTES, resolve_h1_feed_offset

    state = _state_file(tmp_path, offset=2)
    idx = pd.date_range('2026-10-25 10:00', '2026-10-25 19:00', freq='h', tz='UTC')
    t0 = pd.Timestamp('2026-10-25 18:00:05')       # margin ~5 seconds

    off, _s, _ok = resolve_h1_feed_offset(idx, t0, state_path=state)
    assert off == 2, 'a boundary-adjacent change must not be accepted on sight'

    # Too soon: still provisional, still serving the persisted offset.
    off, _s, _ok = resolve_h1_feed_offset(idx, t0 + pd.Timedelta(minutes=1),
                                          state_path=state)
    assert off == 2

    # The SAME boundary-adjacent value an hour later, at the next top of hour
    # (the bar has been emitted, so the newest label advanced with it). This is
    # how a real change repeats near the boundary; the emit-lag artifact does
    # not, because once bars emit normally the raw inference returns to +2.
    idx_next = pd.date_range('2026-10-25 10:00', '2026-10-25 20:00', freq='h', tz='UTC')
    t2 = pd.Timestamp('2026-10-25 19:00:30')
    assert (t2 - t0).total_seconds() / 60.0 >= H1_OFFSET_CONFIRM_MINUTES
    off, saved, _ok = resolve_h1_feed_offset(idx_next, t2, state_path=state)
    assert off == 1
    assert saved['history'][-1]['reason'] == 'confirmed_after_delay'
    assert saved['history'][-1]['previous_offset_hours'] == 2


def test_cold_start_records_the_current_inference(tmp_path):
    from src.live_data import infer_h1_feed_now

    state = _state_file(tmp_path)                  # no persisted value
    assert not os.path.exists(state)
    utcnow = pd.Timestamp('2026-07-30 18:16:00')
    idx = pd.date_range('2026-07-30 10:00', '2026-07-30 20:00', freq='h', tz='UTC')

    got = infer_h1_feed_now(idx, now_utc=utcnow, state_path=state)
    assert got == pd.Timestamp('2026-07-30 20:16:00')

    saved = json.loads(open(state).read())
    assert saved['offset_hours'] == 2
    assert saved['history'][-1]['reason'] == 'cold_start'
    assert saved['history'][-1]['previous_offset_hours'] is None


def test_cold_start_inside_the_emit_lag_window_persists_no_baseline(tmp_path, service):
    """
    A first reading has nothing to defend it, so it is held to the SAME boundary
    bar as a change. Otherwise the artifact becomes the baseline and stickiness
    then protects the WRONG offset -- self-healing on the next clean fetch, but
    only after bad rows have reached the ledger.
    """
    from src.live_data import h1_feed_now_with_status, resolve_h1_feed_offset

    state = _state_file(tmp_path)                  # deleted / absent
    assert not os.path.exists(state)
    utcnow = pd.Timestamp('2026-07-30 18:00:05')
    idx = pd.date_range('2026-07-30 10:00', '2026-07-30 19:00', freq='h', tz='UTC')

    offset, _s, confirmed = resolve_h1_feed_offset(idx, utcnow, state_path=state)
    # (a) NO baseline is persisted.
    assert not os.path.exists(state), 'a boundary-adjacent first reading was persisted'
    assert confirmed is False
    assert offset == 1                                  # usable for this call only

    feed_now, ok = h1_feed_now_with_status(idx, now_utc=utcnow, state_path=state)
    assert ok is False and feed_now == pd.Timestamp('2026-07-30 19:00:05')
    assert not os.path.exists(state)

    # (c) the next CLEAN observation persists the CORRECT offset, as a cold start.
    clean = pd.Timestamp('2026-07-30 18:16:00')
    idx2 = pd.date_range('2026-07-30 10:00', '2026-07-30 20:00', freq='h', tz='UTC')
    offset2, saved, confirmed2 = resolve_h1_feed_offset(idx2, clean, state_path=state)
    assert confirmed2 is True and offset2 == 2
    on_disk = json.loads(open(state).read())
    assert on_disk['offset_hours'] == 2
    assert on_disk['history'][-1]['reason'] == 'cold_start'


def test_unconfirmed_clock_is_not_presented_as_actionable_and_never_settles(tmp_path):
    """(b) The response must not present an actionable direction, and the row
    must not reach the ledger -- ledger rows are this model's only evidence."""
    from src.h1_direction_serving import LOG_COLUMNS, build_ledger
    from src.inference import PredictionService

    # A direction is actionable ONLY when the forecast bar is open.
    for status in ('clock_unconfirmed', 'already_closed', 'market_closed'):
        assert status != 'open'

    bar = '2024-01-17T14:00:00'
    rows = [{'called_at_utc': '2024-01-17T14:05:00', 'as_of_bar_close': bar,
             'as_of_close': 1.1000, 'forecast_bar_start': bar,
             'forecast_bar_end': '2024-01-17T15:00:00', 'direction': 'UP',
             'probability': 0.55, 'minutes_remaining_at_call': 0,
             'forecast_bar_status': 'clock_unconfirmed', 'data_source': 'MT5',
             'model_version': 'v1'}]
    log = tmp_path / 'log.csv'
    pd.DataFrame(rows, columns=LOG_COLUMNS).to_csv(log, index=False)

    ledger = build_ledger(log_path=str(log),
                          closes=pd.Series({pd.Timestamp(bar): 1.1010}))
    assert len(ledger) == 0, 'an unconfirmed-clock prediction reached the ledger'


# ── THE UI ────────────────────────────────────────────────────────────────────

def _read(path):
    return open(os.path.join(REPO, path), encoding='utf-8').read()


def _strip_comments(text: str) -> str:
    """
    Drop HTML and JS comments before scanning markup for banned constructs.

    Necessary because the card's own comments EXPLAIN why there is no gauge; a
    raw substring scan fires on the explanation rather than on any real element.
    """
    import re
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return '\n'.join(l for l in text.splitlines() if not l.strip().startswith('//'))


def test_every_page_links_to_the_h1_direction_page():
    """
    Asserted on the RENDERED pages, not on the page modules' source.
    src/paper_trading.py and src/tracking.py are byte-pinned (the former by a
    pre-existing test that diffs it against git HEAD), so the links are injected
    at the API render layer -- which is what a reader actually sees anyway.
    """
    import api
    from src.paper_trading import LEDGER_COLUMNS, summarize
    from src.paper_trading import render_html as pt_html
    from src.tracking import build_history_html

    assert '/h1-direction' in _read('static/index.html'), 'dashboard has no link'

    # Composed exactly as the routes compose them, but WITHOUT calling the
    # endpoints: /paper-trading rebuilds and rewrites the daily ledger CSVs on
    # every request, and a link test has no business mutating them.
    empty = pd.DataFrame(columns=LEDGER_COLUMNS)
    pt = api._with_h1_nav(
        pt_html({'baseline': {'ledger': empty, 'summary': summarize(empty)}}, 1.5))
    assert '/h1-direction' in pt, '/paper-trading has no link'

    hist = api._with_h1_nav(build_history_html('does_not_exist.csv', {}))
    assert '/h1-direction' in hist, '/history has no link'

    # ...and both routes really do apply that wrapper.
    api_src = _read('api.py')
    assert api_src.count('_with_h1_nav(') >= 3, 'a route is missing the nav wrapper'

    # ...and the pinned page modules were NOT edited to achieve it.
    import subprocess
    for mod in ('src/paper_trading.py', 'src/tracking.py'):
        rc = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', mod],
                            cwd=REPO).returncode
        assert rc == 0, f'{mod} was modified; it must stay byte-identical'


def test_h1_direction_page_links_back_to_the_other_pages():
    from src.h1_direction_serving import render_html
    html = render_html(pd.DataFrame(), pd.DataFrame(), base_dir=REPO)
    for target in ('href="/"', 'href="/history"', 'href="/paper-trading"'):
        assert target in html, f'no back-link {target}'


def test_h1_direction_page_shows_served_version_progress_and_provenance():
    from src.h1_direction_serving import render_html, TARGET_OBSERVATIONS
    html = render_html(pd.DataFrame(), pd.DataFrame(), base_dir=REPO)

    meta = _meta()
    assert 'Currently served' in html
    assert meta['model_version'] in html, 'the live version is not identifiable'
    assert f'of ~{TARGET_OBSERVATIONS} settled' in html

    # The provenance distinction must be VISIBLE page text, not only in a JSON
    # file nobody opens.
    assert '[0:70%] model' in html
    assert 'not validated out of sample' in html
    assert 'seen that test block during training' in html


def test_probability_is_printed_not_drawn_as_a_bar_or_gauge():
    """These probabilities sit within a couple of points of 0.50 by
    construction; a bar would make 0.506 read as a strong call."""
    html = _read('static/index.html')
    card = _strip_comments(html[html.index('h1dirCard'):]).lower()

    assert 'tofixed(3)' in card, 'probability must be printed as a number'
    # No meter/progress/gauge element, and nothing width-driven off the
    # probability -- comments explaining the choice are stripped first.
    for element in ('<meter', '<progress', 'gauge', 'confidence-bar',
                    'width:${d.probability', 'width: ${d.probability'):
        assert element not in card, f'probability rendered as {element}'
    assert 'probability' in card


def test_dashboard_card_handles_every_degraded_state():
    card = _read('static/index.html')
    for status in ('already_closed', 'market_closed', 'clock_unconfirmed'):
        assert status in card, f'card does not handle {status}'
    assert 'available === false' in card or 'available == false' in card
    assert 'This forecast bar has closed' in card
    assert 'fetched at' in card
    # Auto-refresh must be opt-in and must say what enabling it costs.
    assert 'type="checkbox" id="h1dirAuto"' in card
    assert 'checked' not in card.split('h1dirAuto')[1][:80], 'auto-refresh is ON by default'
    assert 'writes a log row every hour' in card


def test_dashboard_still_renders_with_h1_direction_artifacts_removed():
    """The card must degrade, not break the page."""
    if not os.path.isdir(MODEL_DIR):
        pytest.skip('artifacts missing')
    shutil.move(MODEL_DIR, BACKUP_DIR)
    try:
        import api
        from fastapi.testclient import TestClient
        original = api.service
        try:
            api.service = _service()
            client = TestClient(api.app)
            page = client.get('/')
            assert page.status_code == 200
            r = client.get('/api/h1-direction')
            assert r.status_code == 200 and r.json()['available'] is False
            # And the ledger page still renders without a served model.
            v = client.get('/h1-direction')
            assert v.status_code == 200
            assert 'none loaded' in v.text or 'Currently served' in v.text
        finally:
            api.service = original
    finally:
        shutil.move(BACKUP_DIR, MODEL_DIR)


def test_stale_feed_still_degrades_to_utc_and_reports_already_closed(tmp_path, service):
    """Unchanged by the stickiness: a stale feed never borrows the persisted
    offset, and the forecast bar is correctly reported as already closed."""
    from src.live_data import infer_h1_feed_now, resolve_h1_feed_offset

    state = _state_file(tmp_path, offset=2)
    utcnow = pd.Timestamp('2026-07-30 18:16:00')
    stale = pd.date_range('2026-07-20 00:00', '2026-07-20 10:00', freq='h', tz='UTC')

    assert infer_h1_feed_now(stale, now_utc=utcnow, state_path=state) == utcnow
    # The sticky value is preserved, not consumed and not corrupted.
    assert json.loads(open(state).read())['offset_hours'] == 2

    frame = _h1_frame(400, start='2023-12-31 22:00')
    frame = frame[frame.index <= pd.Timestamp('2024-01-17 13:00', tz='UTC')]
    r = service.predict_h1_direction(now=pd.Timestamp('2024-01-20 09:00'), frame=frame)
    assert r['forecast_bar_status'] == 'already_closed'
    assert r['minutes_remaining'] == 0


# ── 4/5/6. THE SERVING CONTRACT ───────────────────────────────────────────────

def _service():
    from src.inference import PredictionService
    with open(os.path.join(REPO, 'config.json')) as f:
        cfg = json.load(f)
    return PredictionService(REPO, cfg)


@pytest.fixture(scope='module')
def service():
    svc = _service()
    if not svc.h1_dir_ready:
        pytest.skip('models/h1_direction artifacts missing')
    return svc


def test_base_bar_is_never_the_currently_forming_hour(service):
    """Mock `now` mid-hour: as_of_bar_close must be the PREVIOUS hour's close."""
    frame = _h1_frame(400, start='2024-01-01 00:00')       # last label 2024-01-17 15:00
    last = frame.index[-1]
    now = last.tz_localize(None) + pd.Timedelta(minutes=37)  # mid-way through it
    r = service.predict_h1_direction(now=now, frame=frame)

    # The last label is still forming, so the base bar is the one BEFORE it and
    # the reported close instant is that bar's end == the forming bar's start.
    assert pd.Timestamp(r['as_of_bar_close']) == last.tz_localize(None)
    assert r['forecast_bar_start'] == r['as_of_bar_close']
    assert pd.Timestamp(r['forecast_bar_end']) == last.tz_localize(None) + pd.Timedelta(hours=1)
    assert r['direction'] in ('UP', 'DOWN') and 0.0 <= r['probability'] <= 1.0


def test_minutes_remaining_arithmetic_and_status_flip(service):
    """14:05 -> 55, 14:50 -> 10, exactly 15:00 -> 0 and the status flips."""
    # Frame ends at 13:00, so the forecast bar is fixed at 14:00-15:00 throughout.
    frame = _h1_frame(400, start='2023-12-31 22:00')
    frame = frame[frame.index <= pd.Timestamp('2024-01-17 13:00', tz='UTC')]
    assert frame.index[-1] == pd.Timestamp('2024-01-17 13:00', tz='UTC')

    def call(hhmm):
        return service.predict_h1_direction(
            now=pd.Timestamp(f'2024-01-17 {hhmm}'), frame=frame)

    r5 = call('14:05')
    assert r5['as_of_bar_close'] == '2024-01-17T14:00:00'
    assert r5['forecast_bar_end'] == '2024-01-17T15:00:00'
    assert r5['minutes_remaining'] == 55 and r5['forecast_bar_status'] == 'open'

    r50 = call('14:50')
    assert r50['minutes_remaining'] == 10 and r50['forecast_bar_status'] == 'open'

    r00 = call('15:00')
    assert r00['minutes_remaining'] == 0
    assert r00['forecast_bar_status'] == 'already_closed'


def test_already_closed_when_more_than_an_hour_past_the_base_bar(service):
    frame = _h1_frame(400, start='2023-12-31 22:00')
    frame = frame[frame.index <= pd.Timestamp('2024-01-17 13:00', tz='UTC')]
    r = service.predict_h1_direction(now=pd.Timestamp('2024-01-17 19:30'), frame=frame)

    assert r['forecast_bar_status'] == 'already_closed'
    assert r['minutes_remaining'] == 0
    # Never presented as actionable.
    assert r['validated_out_of_sample'] is False
    assert 'not a trading instruction' in r['disclaimer'].lower()


# ── 7. GRACEFUL DEGRADATION ───────────────────────────────────────────────────

def test_daily_service_unaffected_when_h1_direction_artifacts_are_deleted():
    """With models/h1_direction/ gone the H1 endpoint must degrade cleanly and
    the DAILY service must be untouched — never a 500, never a cross-family
    failure."""
    if not os.path.isdir(MODEL_DIR):
        pytest.skip('artifacts missing')
    backup = BACKUP_DIR
    shutil.move(MODEL_DIR, backup)
    try:
        svc = _service()
        assert svc.h1_dir_ready is False
        assert any('H1 direction' in e for e in svc.load_errors)
        # The daily families are entirely unaffected.
        assert svc.models_ready is True
        assert svc.baseline_ready is True and svc.macro_ready is True

        import api
        original = api.service
        try:
            api.service = svc
            from fastapi.testclient import TestClient
            client = TestClient(api.app)
            r = client.get('/api/h1-direction')
            assert r.status_code == 200, 'degradation must not raise a 500'
            assert r.json()['available'] is False
            assert 'missing' in r.json()['reason'].lower()

            d = client.post('/api/predict')
            assert d.status_code == 200, 'daily service broke when H1 artifacts vanished'
            assert 'consensus' in d.json() or 'baseline' in d.json()
        finally:
            api.service = original
    finally:
        shutil.move(backup, MODEL_DIR)


# ── 8 / e. THE LEDGER ─────────────────────────────────────────────────────────

def _log_frame(rows):
    from src.h1_direction_serving import LOG_COLUMNS
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def test_three_calls_inside_one_forecast_bar_settle_exactly_one_position(tmp_path):
    from src.h1_direction_serving import LOG_COLUMNS, build_ledger

    bar = '2024-01-17T14:00:00'
    rows = [{
        'called_at_utc': f'2024-01-17T14:{m:02d}:00', 'as_of_bar_close': bar,
        'as_of_close': 1.1000, 'forecast_bar_start': bar,
        'forecast_bar_end': '2024-01-17T15:00:00', 'direction': 'UP',
        'probability': 0.55, 'minutes_remaining_at_call': 60 - m,
        'forecast_bar_status': 'open', 'data_source': 'MT5',
        'model_version': 'v1',
    } for m in (5, 20, 50)]
    log = tmp_path / 'log.csv'
    _log_frame(rows).to_csv(log, index=False)

    closes = pd.Series({pd.Timestamp(bar): 1.1010})
    ledger = build_ledger(log_path=str(log), closes=closes)

    assert len(ledger) == 1, 'repeated calls inflated the sample'
    r = ledger.iloc[0]
    assert r['direction'] == 'UP' and r['entry'] == pytest.approx(1.1000)
    assert r['exit'] == pytest.approx(1.1010)
    assert r['gross_pips'] == pytest.approx(10.0)
    assert r['net_pips'] == pytest.approx(8.5)          # 1.5-pip round trip
    assert r['outcome'] == 'WIN'
    assert r['model_version'] == 'v1'


def test_ledger_carries_model_version_and_a_retrain_is_reported_separately(tmp_path):
    from src.h1_direction_serving import (LOG_COLUMNS, build_ledger, render_html,
                                          read_log, summarize_by_version)

    def row(bar, version, direction, called):
        return {'called_at_utc': called, 'as_of_bar_close': bar, 'as_of_close': 1.1000,
                'forecast_bar_start': bar, 'forecast_bar_end': bar,
                'direction': direction, 'probability': 0.55,
                'minutes_remaining_at_call': 30, 'forecast_bar_status': 'open',
                'data_source': 'MT5', 'model_version': version}

    b1, b2 = '2024-01-17T14:00:00', '2024-01-17T15:00:00'
    b3, b4 = '2024-02-01T10:00:00', '2024-02-01T11:00:00'
    # v1 goes 1W/1L (0.5); v2 goes 2W (1.0) -- deliberately DIFFERENT rates, so a
    # blended headline (0.75) would be visibly wrong for both.
    rows = [row(b1, 'v1', 'UP', 'a'), row(b2, 'v1', 'DOWN', 'b'),
            row(b3, 'v2', 'UP', 'c'), row(b4, 'v2', 'UP', 'd')]
    log = tmp_path / 'log.csv'
    _log_frame(rows).to_csv(log, index=False)

    closes = pd.Series({pd.Timestamp(b1): 1.1010, pd.Timestamp(b2): 1.1010,
                        pd.Timestamp(b3): 1.1010, pd.Timestamp(b4): 1.1010})
    ledger = build_ledger(log_path=str(log), closes=closes)

    assert len(ledger) == 4
    assert 'model_version' in ledger.columns
    assert set(ledger['model_version']) == {'v1', 'v2'}

    summaries = summarize_by_version(ledger, read_log(str(log)))
    assert len(summaries) == 2, 'versions must be reported separately, never blended'
    by = {s['model_version']: s for s in summaries}
    assert by['v1']['settled'] == 2 and by['v2']['settled'] == 2
    assert by['v1']['hit_rate'] == pytest.approx(0.5)
    assert by['v2']['hit_rate'] == pytest.approx(1.0)
    # A blended rate across the retrain would be 0.75 -- wrong for BOTH versions.
    assert by['v1']['hit_rate'] != by['v2']['hit_rate']

    # Cumulative pips restart per version rather than drawing one path through two models.
    v2 = ledger[ledger['model_version'] == 'v2']
    assert v2.iloc[0]['cum_net_pips'] == pytest.approx(v2.iloc[0]['net_pips'])

    html = render_html(ledger, read_log(str(log)), base_dir=REPO)
    assert '2 model versions in this ledger' in html
    assert 'not comparable across a retrain' in html
    assert 'v1' in html and 'v2' in html


# ── 9. SMOKE ──────────────────────────────────────────────────────────────────

def _meta():
    with open(os.path.join(MODEL_DIR, 'h1_direction_meta.json')) as f:
        return json.load(f)


def test_every_h1_direction_artifact_exists():
    for name in ('h1_direction_gbm.json', 'h1_direction_scaler.pkl',
                 'h1_direction_meta.json'):
        p = os.path.join(MODEL_DIR, name)
        assert os.path.isfile(p), f'missing artifact {name}'
        assert os.path.getsize(p) > 0


# ── b / c. THE FULL-HISTORY MODEL'S PROVENANCE ────────────────────────────────

def test_full_history_model_used_more_rows_and_ends_at_the_last_labelled_bar():
    from src import train_h1_direction as T

    meta = _meta()
    raw = T.load_h1_cache(os.path.join(REPO, T.H1_CACHE))
    _frame, target_index = T.build_direction_dataset(raw)
    n70 = len(T.split_purge_embargo(target_index)['train'])

    assert meta['n_train_rows'] == len(target_index)
    assert meta['n_train_rows'] > n70, 'full-history model must use MORE rows'
    assert pd.Timestamp(meta['train_end']).tz_localize(None) == \
        pd.Timestamp(target_index[-1]).tz_localize(None)
    assert pd.Timestamp(meta['train_start']).tz_localize(None) == \
        pd.Timestamp(target_index[0]).tz_localize(None)


def test_meta_declares_it_is_not_validated_out_of_sample():
    from src.train_h1_direction import PROVENANCE_NOTE
    meta = _meta()
    assert meta['validated_out_of_sample'] is False
    assert meta['provenance_note'] == PROVENANCE_NOTE
    for phrase in ('belongs to the [0:70%] model', 'does NOT transfer',
                   'only honest evidence is its own forward ledger'):
        assert phrase in meta['provenance_note']
    assert meta['model_version'].startswith('h1dir-full-')
    assert meta['feature_columns']
    assert meta['seed'] == 42


# ── d. THE RESPONSE MUST NOT BORROW ANOTHER MODEL'S EVIDENCE ──────────────────

def test_response_never_echoes_the_test_block_accuracy_or_ci(service):
    frame = _h1_frame(400)
    r = service.predict_h1_direction(
        now=frame.index[-1].tz_localize(None) + pd.Timedelta(minutes=10), frame=frame)
    blob = json.dumps(r)

    for banned in ('52.96', '0.529559', '49.85', '0.0114', '0.0497', '1.7e-05',
                   '3.10pp', 'McNemar'):
        assert banned not in blob, f'test-block evidence leaked into the response: {banned}'
    assert r['validated_out_of_sample'] is False
    assert 'validated' not in r or r.get('validated') is not True
    assert r['model_version'] and r['trained_through']


# ── 10. THE PROTECTED SET ─────────────────────────────────────────────────────

def test_h1_direction_serving_never_writes_the_daily_h1_cache():
    """
    results/eurusd_h1.csv belongs to the DAILY auxiliary predictor, which
    refreshes it whenever its staleness gate fires -- so byte-freezing it would
    only assert that nobody ran a daily prediction. The real invariant is that
    the H1-DIRECTION path never writes it, and that is asserted here directly.
    """
    import ast
    for mod in ('src/h1_direction_serving.py', 'src/inference.py'):
        tree = ast.parse(_read(mod))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, 'id', '') == 'fetch_h1_market_data'):
                continue
            kw = {k.arg: k.value for k in node.keywords}
            assert 'cache_path' in kw, 'fetch_h1_market_data called without cache_path'
            assert isinstance(kw['cache_path'], ast.Constant) and kw['cache_path'].value is None, \
                'the H1-direction path must pass cache_path=None and never write the daily cache'


def test_h1_direction_ledger_writes_only_its_own_file():
    """
    The daily ledgers are DERIVED artifacts that /paper-trading regenerates on
    every request, so byte-pinning them only asserts nobody opened the page. The
    invariant that matters is that the H1-direction builder writes its OWN file
    and nothing else -- asserted here directly.
    """
    import ast

    from src.h1_direction_serving import LEDGER_PATH, LOG_PATH
    assert LEDGER_PATH == 'results/paper_trading_log_h1_direction.csv'
    assert LOG_PATH == 'results/h1_direction_log.csv'

    # Scan STRING LITERALS excluding docstrings: the module's own docstring
    # explains that its log is separate from the daily prediction_log.csv, and a
    # raw text scan would fire on that explanation rather than on a real path.
    tree = ast.parse(_read('src/h1_direction_serving.py'))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and id(n) not in docstrings]

    for foreign in ('paper_trading_log_baseline', 'paper_trading_log_macro',
                    'paper_trading_log_ti_h1', 'prediction_log.csv'):
        hits = [s for s in literals if foreign in s]
        assert not hits, f'H1 serving references the daily ledger {foreign}: {hits}'


def test_protected_set_is_sha256_identical():
    """
    PINNING POLICY. This fixture pins CODE and MODEL ARTIFACTS -- things that
    must not change. Three classes of file were deliberately UNPINNED because the
    application rewrites them by design, so pinning them asserted only that
    nobody had run the app:
      * results/prediction_log.csv -- appended by every /api/predict.
      * results/eurusd_h1.csv -- refreshed by the daily predictor's staleness gate.
      * results/paper_trading_log_{baseline,macro,ti_h1}.csv -- rebuilt and
        rewritten by every /paper-trading request.
    Each has a direct invariant asserted instead: the H1-direction path never
    writes the daily H1 cache, and its ledger builder writes only its own file.
    src/paper_trading.py and src/tracking.py remain byte-identical to git HEAD.
    """
    if not os.path.exists(SHA_FIXTURE):
        pytest.skip('protected-set fixture missing')
    with open(SHA_FIXTURE) as fh:
        expected = json.load(fh)
    assert len(expected) > 40
    assert not any('h1_direction' in k for k in expected if k.startswith('models/'))
    for rel, digest in expected.items():
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'protected file vanished: {rel}'
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        assert got == digest, f'PROTECTED FILE MODIFIED: {rel}'


def test_no_execution_or_sizing_capability_was_added():
    """
    The prohibition, asserted rather than promised.

    Scans CODE IDENTIFIERS via the AST, not raw text: the modules legitimately
    say "applies no leverage" and "sets no stops" in their docstrings, and a
    substring scan would fire on those denials. What must not exist is a
    callable, attribute or name that DOES any of it.
    """
    import ast

    banned = {'order_send', 'place_order', 'send_order', 'position_size',
              'stop_loss', 'take_profit', 'leverage', 'lot_size', 'set_sl',
              'set_tp', 'open_position', 'close_position'}
    for mod in ('src/h1_direction_serving.py', 'src/train_h1_direction.py',
                'src/h1_features.py'):
        tree = ast.parse(open(os.path.join(REPO, mod), encoding='utf-8').read())
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                found.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name.lower())
            elif isinstance(node, ast.arg):
                found.add(node.arg.lower())
        leaked = found & banned
        assert not leaked, f'{mod} gained an execution capability: {sorted(leaked)}'
        # And nothing may reach for the broker's trading API at all.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, 'module', '') or '']
                assert not any('metatrader5' in str(n).lower() for n in names), \
                    f'{mod} imported the broker trading API'
