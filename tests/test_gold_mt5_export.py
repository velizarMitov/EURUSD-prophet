"""Gold signals on the MT5 chart (openspec change gold-mt5-chart-signals, spec
gold-forecasting/mt5-chart-signals): the export the indicator reads, and a
scan of the indicator source."""
import copy
import os
import re
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.gold import forward_logger as FL
from src.gold import load_config
from src.gold import mt5_export as X
from src.gold import summary as S

CFG = load_config()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MQ5 = os.path.join(REPO, 'mql5', 'Indicators', 'ProphetGoldSignals.mq5')
NOW = datetime(2026, 11, 2, 9, 21, tzinfo=timezone.utc)

# No chart text may name a trade, in either language.
TRADE_EN = re.compile(r'\b(buy|sell|long|short|entry|entries|stop|target|take[- ]?profit)\b', re.I)
TRADE_BG = re.compile(r'(?<![а-яё])(куп|покуп|прода|вход|стоп|цел|тейк|лонг|шорт)', re.I)


def _trade_words(text):
    return TRADE_EN.findall(text) + TRADE_BG.findall(text)


def _p(as_of, p_up=0.6, vol=0.1, close=4000.0, cell='h1_lstm', cad='H1'):
    return {'key': f'{cell}|{cad}|{as_of}', 'cell': cell, 'cadence': cad, 'as_of_bar': as_of, 'as_of_close': close,
            'p_up': p_up, 'ret_pct': 0.01, 'vol_pct': vol, 'implied_price': '', 'model_digest': 'd',
            'phase': 'dry_run', 'logged_at': 'x'}


def _s(as_of, realised_dir, ret, eligible=True, cell='h1_lstm', cad='H1', exit_bar=''):
    return {'key': f'{cell}|{cad}|{as_of}', 'cell': cell, 'cadence': cad, 'as_of_bar': as_of, 'exit_bar': exit_bar,
            'exit_close': '', 'realised_dir': realised_dir, 'realised_ret_pct': ret, 'realised_vol_pct': abs(ret),
            'eligible': eligible, 'phase': 'dry_run', 'settled_at': 'x'}


def _a(as_of, vol):
    return {'key': f'h1_lstm|H1|{as_of}', 'cell': 'h1_lstm', 'as_of_bar': as_of, 'vol_pct': vol,
            'threshold_vol_pct': 0.151, 'percentile': 90.0, 'notified': 'yes', 'logged_at': 'x'}


def _bar(day, hour):
    return f'{day} {hour:02d}:00:00+00:00'


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['forward'] = str(tmp_path / 'forward')
    return c


def _write(cfg, preds=(), setts=(), alerts=(), a_setts=(), v2=()):
    FL.append_rows(FL.out_path(cfg, 'predictions'), list(preds), FL.PRED_FIELDS)
    FL.append_rows(FL.out_path(cfg, 'settlements'), list(setts), FL.SETTLE_FIELDS)
    FL.append_rows(FL.out_path(cfg, 'predictions_v2'), list(v2), FL.V2_PRED_FIELDS)
    from src.gold import alerts as A
    FL.append_rows(FL.out_path(cfg, 'alerts'), list(alerts), A.ALERT_FIELDS)
    FL.append_rows(FL.out_path(cfg, 'alert_settlements'), list(a_setts), A.ALERT_SETTLE_FIELDS)


def _by_key(data, kind='forecast'):
    return {r['key']: r for r in data['rows'] if r['kind'] == kind}


# ── 1.1 selection and scoring ───────────────────────────────────────────────

def test_only_the_chosen_cells_are_drawn(cfg):
    bar, day = _bar('2026-11-02', 9), '2026-10-30 00:00:00+00:00'
    v2 = dict(_p(bar), key=f'h1_lstm|H1|{bar}|20261023', version='20261023', shadow=1)
    _write(cfg, preds=[_p(bar, cell='h1_gbm'), _p(bar), _p(bar, cell='h1_garch_hod'),
                       _p(day, cell='d1_lstm', cad='D1'), _p(day, cell='d1_gbm', cad='D1'),
                       _p(day, cell='d1_garch', cad='D1')], v2=[v2])
    data = X.build(cfg, NOW)
    assert sorted(r['key'] for r in data['rows']) == [f'd1_gbm|D1|{day}', f'd1_lstm|D1|{day}', f'h1_lstm|H1|{bar}']
    assert {(r['cell'], r['cadence']) for r in data['rows']} == {('h1_lstm', 'H1'), ('d1_lstm', 'D1'),
                                                                 ('d1_gbm', 'D1')}


def test_the_window_is_the_last_five_trading_days_by_sofia_date(cfg):
    days = ['2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29', '2026-10-30']
    preds = [_p(_bar(d, h)) for d in days for h in (9, 15)]
    preds.append(_p('2026-10-25 23:00:00+00:00'))           # Sunday's weekly-open bar: its hour is Monday
    old_alert = _a(_bar('2026-10-23', 9), 0.2)
    _write(cfg, preds=preds, alerts=[old_alert, _a(_bar('2026-10-30', 15), 0.2)])
    data = X.build(cfg, NOW)
    assert [str(d) for d in data['days']] == days[2:]
    assert all(r['hour_label'] >= '2026.10.26' for r in data['rows'])
    assert old_alert['key'] not in _by_key(data, 'alert') and len(_by_key(data, 'alert')) == 1


def test_sunday_open_bar_makes_no_phantom_sunday(cfg):
    _write(cfg, preds=[_p(_bar('2026-10-30', 21)), _p('2026-11-01 23:00:00+00:00'), _p(_bar('2026-11-02', 0))])
    data = X.build(cfg, NOW)
    assert [str(d) for d in data['days']] == ['2026-10-30', '2026-11-02']


def test_outcomes_and_the_count(cfg):
    b = [_bar('2026-11-02', h) for h in (5, 6, 7, 8, 9)]
    _write(cfg, preds=[_p(b[0], 0.6), _p(b[1], 0.6), _p(b[2], 0.4), _p(b[3], 0.6), _p(b[4], 0.6)],
           setts=[_s(b[0], 1, 0.12), _s(b[1], 0, -0.08), _s(b[2], 0, -0.05),
                  _s(b[3], 1, 0.2, eligible=False)])
    data = X.build(cfg, NOW)
    got = [r['outcome'] for r in data['rows']]
    assert got == ['right', 'wrong', 'right', 'void', 'open']
    assert (data['right'], data['wrong']) == (2, 1)
    assert data['meta']['count_h1_en'] == ('1 days, arrows ≥{min} %: right {r_h1_lstm} of {t_h1_lstm} '
                                           '(descriptive, not a test)')


def test_an_empty_realised_direction_is_not_counted(cfg):
    bar = _bar('2026-11-02', 9)
    _write(cfg, preds=[_p(bar)], setts=[_s(bar, '', 0.0)])
    [row] = X.build(cfg, NOW)['rows']
    assert row['outcome'] == 'void' and row['text_bg'].endswith('не се брои')


def test_an_alert_without_its_prediction_row_is_shown_in_percent_only(cfg):
    _write(cfg, preds=[_p(_bar('2026-11-02', 8))], alerts=[_a(_bar('2026-11-02', 9), 0.1954)])
    alert = _by_key(X.build(cfg, NOW), 'alert')[f"h1_lstm|H1|{_bar('2026-11-02', 9)}"]
    assert alert['text_en'] == '±0.20 %' and '$' not in alert['tip_en']


# ── 5.1 daily cells ─────────────────────────────────────────────────────────

def _day(d):
    return f'{d} 00:00:00+00:00'


def test_spec_daily_scenario(cfg):
    as_of = _day('2026-10-29')
    _write(cfg, preds=[_p(as_of, 0.4777, 0.517, 4023.57, cell='d1_lstm', cad='D1'),
                       _p(as_of, 0.7782, '', 4023.57, cell='d1_gbm', cad='D1')])
    rows = _by_key(X.build(cfg, NOW))
    lstm, gbm = rows[f'd1_lstm|D1|{as_of}'], rows[f'd1_gbm|D1|{as_of}']
    assert (lstm['direction'], lstm['prob_pct'], lstm['text_en'], lstm['text_bg']) == ('down', 52, 'lstm 52 %', 'lstm 52 %')
    assert (gbm['direction'], gbm['prob_pct'], gbm['text_en']) == ('up', 78, 'gbm 78 %')
    assert lstm['hour_label'] == gbm['hour_label'] == '2026.10.30 00:00' and lstm['hour_end_label'] == '2026.10.31 00:00'
    assert lstm['tip_en'].split('\n')[0] == 'Gold · d1_lstm · day Oct 30' and lstm['hour_sofia_bg'] == '30.10'
    assert 'Expected move: about ±0.52 % (~±20.80 $ at 4023.57)' in lstm['tip_en']
    assert 'Expected move' not in gbm['tip_en'] and 'Очаквано' not in gbm['tip_bg']
    assert lstm['panel_en'] == 'd1_lstm · Oct 30: down 52 %, expected ±0.52 %'
    assert gbm['panel_bg'] == 'd1_gbm · 30.10: нагоре 78 %'
    assert 'денят е отворен' in lstm['tip_bg']


def test_the_settled_exit_bar_dates_a_daily_mark(cfg):
    fri, thu = _day('2026-10-30'), _day('2026-10-29')
    _write(cfg, preds=[_p(fri, 0.6, cell='d1_lstm', cad='D1'), _p(thu, 0.6, cell='d1_lstm', cad='D1')],
           setts=[_s(thu, 0, -0.45, cell='d1_lstm', cad='D1', exit_bar=_day('2026-11-01'))])
    rows = _by_key(X.build(cfg, NOW))
    assert rows[f'd1_lstm|D1|{thu}']['hour_label'] == '2026.11.01 00:00'           # where it was scored
    assert rows[f'd1_lstm|D1|{thu}']['text_en'] == 'lstm 60 % · wrong −0.45 %'
    assert rows[f'd1_lstm|D1|{fri}']['hour_label'] == '2026.11.02 00:00'           # next business day


def test_the_daily_window_is_the_last_twenty_forecast_days(cfg):
    days = pd.bdate_range('2026-09-25', periods=26)
    _write(cfg, preds=[_p(_day(f'{d:%Y-%m-%d}'), cell=c, cad='D1') for d in days for c in ('d1_lstm', 'd1_gbm')])
    data = X.build(cfg, NOW)
    marks = sorted({r['hour_label'] for r in data['rows']})
    assert len(data['d1_days']) == 20 and len(marks) == 20 and len(data['rows']) == 40
    assert marks[-1] == f"{days[-1] + pd.offsets.BDay(1):%Y.%m.%d} 00:00"
    assert data['meta']['count_d1_en'].startswith('20 days, arrows ≥{min} %: lstm right {r_d1_lstm} of {t_d1_lstm}')


@pytest.mark.parametrize('p_up', [0.4777, 0.7782, 0.5, 0.5001])
def test_daily_direction_matches_the_gold_page(cfg, p_up):
    as_of = _day('2026-10-29')
    pred = _p(as_of, p_up, cell='d1_gbm', cad='D1')
    _write(cfg, preds=[pred])
    [row] = X.build(cfg, NOW)['rows']
    cur, _ = S.current_forecast({k: str(v) for k, v in pred.items()}, 'D1', pd.Timestamp(as_of))
    assert row['direction'] == cur['direction'] and row['prob_pct'] == round(cur['probability'] * 100)


def test_the_filter_line_says_it_is_a_display_choice(cfg):
    meta = X.build(cfg, NOW)['meta']
    assert meta['filter_en'] == 'Filter ≥{min} %: a display choice, not a tested rule'
    assert meta['filter_bg'] == 'Филтър ≥{min} %: избор за показване, не е тестван праг'


# ── 1.2 texts ───────────────────────────────────────────────────────────────

def test_spec_examples(cfg):
    fc, al = _bar('2026-11-02', 9), _bar('2026-10-27', 16)
    _write(cfg, preds=[_p(fc, 0.5317, 0.1446, 3897.45), _p(al, 0.5, 0.1954, 4063.36)],
           alerts=[_a(al, 0.1954)])
    data = X.build(cfg, NOW)
    row = _by_key(data)[f'h1_lstm|H1|{fc}']
    assert (row['direction'], row['prob_pct'], row['text_en'], row['text_bg']) == ('up', 53, '53 %', '53 %')
    assert row['hour_label'] == '2026.11.02 10:00' and row['hour_end_label'] == '2026.11.02 11:00'
    assert row['hour_sofia_bg'] == '02.11, 11:00–12:00' and row['hour_sofia_en'] == 'Nov 02, 11:00–12:00'
    assert 'about ±0.14 % (~±5.64 $ at 3897.45)' in row['tip_en'] and 'Direction: up, 53 %' in row['tip_en']
    assert 'около ±0,14 % (~±5,64 $ при цена 3897,45)' in row['tip_bg']
    assert row['tip_en'].endswith('research — no proven edge, gross of cost')
    assert row['panel_bg'] == '11:00–12:00 София: нагоре 53 %, очаквано ±0,14 %'
    alert = _by_key(data, 'alert')[f'h1_lstm|H1|{al}']
    assert alert['text_en'] == '±0.20 % (~±7.94 $)' and alert['text_bg'] == '±0,20 % (~±7,94 $)'
    assert alert['hour_label'] == '2026.10.27 17:00' and '18:00–19:00' in alert['tip_en']


def test_a_near_even_forecast_is_still_drawn(cfg):
    bar = _bar('2026-11-02', 9)
    _write(cfg, preds=[_p(bar, 0.4994)])
    [row] = X.build(cfg, NOW)['rows']
    assert (row['direction'], row['prob_pct']) == ('down', 50)


def test_a_wrong_arrow_shows_wrong_and_the_realised_move(cfg):
    bar = _bar('2026-11-02', 9)
    _write(cfg, preds=[_p(bar, 0.6)], setts=[_s(bar, 0, -0.08)])
    [row] = X.build(cfg, NOW)['rows']
    assert row['text_en'] == '60 % · wrong −0.08 %' and row['text_bg'] == '60 % · сбърка −0,08 %'


def test_sofia_hours_and_the_label_offset_follow_dst(cfg):
    before, after = _bar('2026-10-23', 9), _bar('2026-10-26', 9)   # EU summer time ended 2026-10-25
    _write(cfg, preds=[_p(before), _p(after)])
    rows = _by_key(X.build(cfg, NOW))
    assert rows[f'h1_lstm|H1|{before}']['hour_sofia_en'] == 'Oct 23, 11:00–12:00'
    assert rows[f'h1_lstm|H1|{after}']['hour_sofia_en'] == 'Oct 26, 11:00–12:00'
    assert X.build(cfg, datetime(2026, 10, 23, 9, tzinfo=timezone.utc))['meta']['label_utc_offset_min'] == 120
    assert X.build(cfg, NOW)['meta']['label_utc_offset_min'] == 60
    assert X.build(cfg, NOW)['meta']['written_bg'] == 'данни от 02.11 11:21 (София)'


def test_alert_texts_say_the_direction_is_unknown(cfg):
    bar = _bar('2026-11-02', 9)
    _write(cfg, preds=[_p(bar)], alerts=[_a(bar, 0.2)], a_setts=[{'key': f'h1_lstm|H1|{bar}',
                                                                 'realised_vol_pct': 0.42, 'settled_at': 'x'}])
    alert = _by_key(X.build(cfg, NOW), 'alert')[f'h1_lstm|H1|{bar}']
    assert 'Посоката е неизвестна.' in alert['tip_bg'] and 'Direction unknown.' in alert['tip_en']
    assert 'посоката е неизвестна' in alert['panel_bg'] and 'direction unknown' in alert['panel_en']
    assert alert['outcome'] == 'sized' and alert['text_en'].endswith('actual 0.42 %')


def test_no_chart_text_names_a_trade(cfg):
    b = [_bar('2026-11-02', h) for h in (5, 6, 7, 8)]
    _write(cfg, preds=[_p(b[0], 0.7), _p(b[1], 0.3), _p(b[2], 0.55), _p(b[3], 0.45)],
           setts=[_s(b[0], 1, 0.1), _s(b[1], 1, 0.1), _s(b[2], '', 0.0, eligible=False)],
           alerts=[_a(b[0], 0.3), _a(b[3], 0.2)],
           a_setts=[{'key': f'h1_lstm|H1|{b[0]}', 'realised_vol_pct': 0.5, 'settled_at': 'x'}])
    data = X.build(cfg, NOW)
    texts = [str(v) for r in data['rows'] for k, v in r.items() if k.endswith(('_bg', '_en'))]
    texts += [str(v) for v in data['meta'].values()]
    texts += [t for entry in X.TEXT.values() for t in entry.values()]
    assert texts and not [(t, _trade_words(t)) for t in texts if _trade_words(t)]


def test_the_trade_word_check_catches_both_languages():
    assert _trade_words('Buy now') and _trade_words('take-profit') and _trade_words('Купи злато')
    assert _trade_words('стоп лос') and _trade_words('целта е 4100')
    assert not _trade_words('при цена 3897,45 · позна · сбърка · нагоре · надолу · Посоката е неизвестна')


# ── 1.3 the same direction as /gold ─────────────────────────────────────────

@pytest.mark.parametrize('p_up', [0.5317, 0.5, 0.4994, 0.5001, 0.71, 0.2])
def test_direction_and_probability_match_the_gold_page(cfg, p_up):
    bar = _bar('2026-11-02', 9)
    pred = _p(bar, p_up)
    _write(cfg, preds=[pred])
    [row] = X.build(cfg, NOW)['rows']
    cur, _ = S.current_forecast({k: str(v) for k, v in pred.items()}, 'H1', pd.Timestamp(bar))
    assert row['direction'] == cur['direction'] and row['prob_pct'] == round(cur['probability'] * 100)


# ── 1.4 the file ────────────────────────────────────────────────────────────

def _parse(path):
    with open(path, encoding='utf-8', newline='') as fh:
        lines = fh.read().split('\n')
    assert lines[-1] == '' and '\r' not in ''.join(lines)
    meta = dict(f.split('=', 1) for f in lines[0].split('\t')[1:])
    header = lines[1].split('\t')
    return lines[0].split('\t')[0], meta, [dict(zip(header, l.split('\t'))) for l in lines[2:-1]]


def test_the_file_round_trips(cfg, tmp_path):
    bar = _bar('2026-11-02', 9)
    _write(cfg, preds=[_p(bar, 0.5317, 0.1446, 3897.45)], alerts=[_a(bar, 0.1954)])
    path = str(tmp_path / 'common' / X.SUBDIR / X.FILE_NAME)
    status = X.export(cfg, NOW, path=path)
    tag, meta, rows = _parse(path)
    assert tag == '#meta' and meta['version'] == '2' and meta['label_tz'] == 'Europe/Berlin'
    assert meta['generated_utc'] == '2026.11.02 09:21:00' and meta['label_utc_offset_min'] == '60'
    assert [r['kind'] for r in rows] == ['forecast', 'alert'] and len(rows[0]) == len(X.ROW_FIELDS)
    assert rows[0]['tip_en'].split('\\n')[0] == 'Gold · h1_lstm · Nov 02, 11:00–12:00 (Sofia time)'
    assert status == '2 mark(s), 1 day(s) 2026-11-02..2026-11-02, 0 daily day(s)'
    assert not os.path.exists(path + '.tmp')


def test_fields_never_carry_a_tab_or_a_raw_line_break():
    assert X._field('a\tb\r\nc') == 'a b\\nc'


def test_a_refused_replace_is_retried_then_succeeds(tmp_path):
    calls, waits = [], []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('in use')
        os.replace(src, dst)

    path = str(tmp_path / 'f.tsv')
    X.write_atomic(path, 'x\n', replace=flaky, sleep=waits.append)
    assert len(calls) == 3 and waits == [X.REPLACE_WAIT_S] * 2
    assert open(path, encoding='utf-8').read() == 'x\n'


def test_a_replace_that_always_fails_raises(tmp_path):
    def never(src, dst):
        raise PermissionError('in use')

    with pytest.raises(PermissionError):
        X.write_atomic(str(tmp_path / 'f.tsv'), 'x\n', replace=never, sleep=lambda s: None)


def test_export_path_uses_the_override_then_appdata():
    assert X.export_path({'MT5_COMMON_FILES': r'C:\x'}) == os.path.join(r'C:\x', 'EURUSDProphet', 'gold_signals.tsv')
    assert X.export_path({'APPDATA': r'C:\Users\u\AppData\Roaming'}) == os.path.join(
        r'C:\Users\u\AppData\Roaming', 'MetaQuotes', 'Terminal', 'Common', 'Files', 'EURUSDProphet', 'gold_signals.tsv')
    with pytest.raises(RuntimeError):
        X.export_path({})


# ── 3.4 the indicator never trades ──────────────────────────────────────────

FORBIDDEN_CALLS = re.compile(r'OrderSend|OrderSendAsync|CTrade|#include\s*<Trade|PositionOpen|WebRequest|SocketCreate'
                             r'|#import')
MQL_LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"')
# Inputs and enum members show their trailing comment as a label in MT5's settings dialog.
MQL_SHOWN_COMMENT = re.compile(r'^\s*(?:input\b.*|PGS_\w+\s*=.*)//(.*)$', re.M)


def _mq5():
    with open(MQ5, encoding='utf-8-sig') as fh:
        return fh.read()


def _forbidden(src):
    """(calls, trade words in shown text) of an MQL5 source."""
    shown = MQL_LITERAL.findall(src) + MQL_SHOWN_COMMENT.findall(src)
    return FORBIDDEN_CALLS.findall(src), [w for s in shown for w in _trade_words(s)]


def test_the_indicator_has_no_trading_network_or_dll_call_and_no_trade_words():
    src = _mq5()
    assert '#property indicator_chart_window' in src and 'OnTick' not in src
    assert MQL_LITERAL.findall(src) and MQL_SHOWN_COMMENT.findall(src)
    assert _forbidden(src) == ([], [])


@pytest.mark.parametrize('bad', ['   OrderSend(req, res);', '#include <Trade\\Trade.mqh>', 'CTrade t;',
                                 '#import "kernel32.dll"', 'WebRequest("GET", u, h, 5000, d, r, rh);',
                                 'return bg ? "Купи" : "Buy";', 'input int InpX = 1; // Stop distance'])
def test_the_source_scan_catches_a_violation(bad):
    calls, words = _forbidden(_mq5() + '\n' + bad + '\n')
    assert calls or words


def test_the_indicator_reads_the_file_the_export_writes():
    src = _mq5()
    assert f'"{X.SUBDIR}\\\\{X.FILE_NAME}"' in src and f'#define PGS_VERSION    "{X.VERSION}"' in src
    read = re.findall(r'Col\(head, nh, "(\w+)"( \+ g_lang)?\)', src)
    assert len(read) == 14
    for name, per_lang in read:
        for lang in (X.LANGS if per_lang else ('',)):
            assert name + lang in X.ROW_FIELDS, name + lang
    meta = X.build({'paths': {'forward': os.path.join(REPO, 'no-such-folder')}}, NOW)['meta']
    for key, per_lang in re.findall(r'k == "(\w+)"( \+ g_lang)?', src):
        for lang in (X.LANGS if per_lang else ('',)):
            assert key + lang in meta, key + lang


# ── the second-opinion line (change tsfm-second-opinion, task 8.1) ─────────

SECOND_BG = ('Второ мнение: Chronos ±0,18 %, TimesFM ±0,15 % · размер: '
             'Chronos не е съгласен, TimesFM е съгласен с h1_lstm')


def test_the_second_opinion_line_comes_from_the_external_view(cfg, monkeypatch):
    import src.external.tsfm.view as VW
    seen = []

    def line(lang, repo=None, now=None):
        seen.append((lang, now))
        return SECOND_BG if lang == 'bg' else 'Second opinion: Chronos ±0.18 %, TimesFM ±0.15 % · size: ...'
    monkeypatch.setattr(VW, 'gold_line', line)
    meta = X.build(cfg, NOW)['meta']
    assert meta['second_bg'] == SECOND_BG
    assert meta['second_tip_en'].startswith('external model — research, no proven edge')
    assert {lang for lang, _ in seen} == {'bg', 'en'}
    # the view is asked on the label clock: Berlin wall time, stamped UTC
    assert seen[0][1] == pd.Timestamp(NOW.astimezone(X.LABEL_TZ).replace(tzinfo=None), tz='UTC')


def test_the_second_opinion_line_is_empty_when_nothing_is_current(cfg, monkeypatch):
    import src.external.tsfm.view as VW
    monkeypatch.setattr(VW, 'gold_line', lambda lang, repo=None, now=None: '')
    assert X.build(cfg, NOW)['meta']['second_bg'] == ''


def test_a_broken_external_view_costs_the_chart_nothing(cfg, monkeypatch):
    import src.external.tsfm.view as VW

    def boom(*a, **k):
        raise RuntimeError('external files unreadable')
    monkeypatch.setattr(VW, 'gold_line', boom)
    data = X.build(cfg, NOW)
    assert data['meta']['second_en'] == '' and 'written_en' in data['meta']


def test_second_opinion_texts_carry_no_trade_words():
    for lang in X.LANGS:
        for s in (SECOND_BG, X.TEXT['second_tip'][lang]):
            assert not TRADE_EN.search(s) and not TRADE_BG.search(s), s


def test_the_export_reads_the_external_files_only_through_the_view():
    src = open(os.path.join(REPO, 'src', 'gold', 'mt5_export.py'), encoding='utf-8').read()
    assert 'from src.external.tsfm.view import gold_line' in src
    code = re.sub(r'(\'\'\'|""").*?\1', '', src, flags=re.S)
    assert 'external_tsfm' not in code.replace('src.external.tsfm.view', '')   # no path of its own to those files


def test_the_indicator_shows_the_line_with_its_label_as_tooltip():
    src = _mq5()
    assert 'k == "second_" + g_lang' in src and 'k == "second_tip_" + g_lang' in src
    assert 'ObjectSetString(0, name, OBJPROP_TOOLTIP, g_second_tip)' in src


# ── 1.5 the command ─────────────────────────────────────────────────────────

def test_the_command_rebuilds_the_file(cfg, tmp_path, monkeypatch, capsys):
    _write(cfg, preds=[_p(_bar('2026-11-02', 9))])
    monkeypatch.setenv('MT5_COMMON_FILES', str(tmp_path / 'common'))
    monkeypatch.setattr(X, 'load_config', lambda: cfg)
    X.main([])
    out = capsys.readouterr().out
    assert out.startswith('1 mark(s), 1 day(s) 2026-11-02..2026-11-02, 0 daily day(s) -> ')
    assert os.path.exists(tmp_path / 'common' / 'EURUSDProphet' / 'gold_signals.tsv')
