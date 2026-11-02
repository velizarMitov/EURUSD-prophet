"""Gold signals on the MT5 chart (openspec change gold-mt5-chart-signals, spec
gold-forecasting/mt5-chart-signals).

Rebuilds ONE file that the indicator mql5/Indicators/ProphetGoldSignals.mq5
reads from the MT5 terminals' shared folder:

    %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\EURUSDProphet\\gold_signals.tsv

It holds the h1_lstm forecasts and the large-move alerts of the last 5 trading
days and the d1_lstm / d1_gbm forecasts of the last 20, each with its outcome
once settled, and every text the chart shows, in BG and EN. Python writes all
the words so the wording rules are tested here; the indicator only places them
(design D2). The right/wrong count is a template: only the indicator knows the
probability filter the owner picked, so it fills in the numbers (design D12).

It is the one gold write outside results/gold/ and research_models/gold/, and
nothing in the repo reads it back. It reads only the forward logs: no model,
no market data. h1_gbm, the GARCH baselines and predictions_v2.csv are never
read -- the owner chose h1_lstm for H1 and d1_lstm + d1_gbm for D1. The one
extra line, the external models' second opinion (change tsfm-second-opinion),
comes from the read-only src.external.tsfm.view over results/external_tsfm/.

    python -m src.gold.mt5_export     rebuild the file now
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import load_config, repo_path

CELL = 'h1_lstm'
DRAWN = {'H1': ('h1_lstm',), 'D1': ('d1_lstm', 'd1_gbm')}
DAYS = 5
DAYS_D1 = 20
VERSION = 2
COMMON_ENV = 'MT5_COMMON_FILES'
SUBDIR = 'EURUSDProphet'
FILE_NAME = 'gold_signals.tsv'
REPLACE_TRIES = 5
REPLACE_WAIT_S = 0.2
LABEL_TZ = ZoneInfo('Europe/Berlin')
SOFIA = ZoneInfo('Europe/Sofia')
MQL_TIME = '%Y.%m.%d %H:%M'
LANGS = ('bg', 'en')
MONTHS_EN = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

ROW_FIELDS = ['kind', 'key', 'cell', 'cadence', 'hour_label', 'hour_end_label', 'direction', 'prob_pct', 'vol_pct',
              'close', 'outcome', 'text_bg', 'text_en', 'tip_bg', 'tip_en', 'panel_bg', 'panel_en',
              'hour_sofia_bg', 'hour_sofia_en']

# Size and direction only: never buy, sell, long, short, entry, stop, target or take-profit.
TEXT = {
    'up': {'bg': 'нагоре', 'en': 'up'},
    'down': {'bg': 'надолу', 'en': 'down'},
    'right': {'bg': 'позна', 'en': 'right'},
    'wrong': {'bg': 'сбърка', 'en': 'wrong'},
    'void': {'bg': 'не се брои', 'en': 'not counted'},
    'open': {'bg': 'часът е отворен', 'en': 'hour still open'},
    'open_day': {'bg': 'денят е отворен', 'en': 'day still open'},
    'actual': {'bg': 'реално {v} %', 'en': 'actual {v} %'},
    'title_forecast': {'bg': 'Злато · h1_lstm · {h} (София)', 'en': 'Gold · h1_lstm · {h} (Sofia time)'},
    'title_day': {'bg': 'Злато · {c} · ден {d}', 'en': 'Gold · {c} · day {d}'},
    'title_alert': {'bg': 'Злато: очаква се голямо движение · {h} (София)',
                    'en': 'Gold: a large move is expected · {h} (Sofia time)'},
    'direction': {'bg': 'Посока: {d}, {p} %', 'en': 'Direction: {d}, {p} %'},
    'move': {'bg': 'Очаквано движение: около ±{v} %', 'en': 'Expected move: about ±{v} %'},
    'usd_at': {'bg': ' (~±{u} $ при цена {c})', 'en': ' (~±{u} $ at {c})'},
    'usd': {'bg': ' (~±{u} $)', 'en': ' (~±{u} $)'},
    'outcome': {'bg': 'Резултат: {o}', 'en': 'Outcome: {o}'},
    'unknown': {'bg': 'Посоката е неизвестна.', 'en': 'Direction unknown.'},
    'research': {'bg': 'Изследване — без доказан edge, без разходи', 'en': 'research — no proven edge, gross of cost'},
    'panel_forecast': {'bg': '{h} София: {d} {p} %, очаквано ±{v} %', 'en': '{h} Sofia: {d} {p} %, expected ±{v} %'},
    'panel_day': {'bg': '{c} · {d}: {dir} {p} %', 'en': '{c} · {d}: {dir} {p} %'},
    'panel_move': {'bg': ', очаквано ±{v} %', 'en': ', expected ±{v} %'},
    'panel_alert': {'bg': '⚠ Голямо движение ±{v} %{usd} · посоката е неизвестна',
                    'en': '⚠ Large move ±{v} %{usd} · direction unknown'},
    'written': {'bg': 'данни от {t} (София)', 'en': 'data from {t} Sofia time'},
    # {n} is filled here; {min}, {r_<cell>} and {t_<cell>} by the indicator for the arrows it shows.
    'count_h1': {'bg': '{n} дни, стрелки ≥{min} %: позна {r_h1_lstm} от {t_h1_lstm} (описателно, не е тест)',
                 'en': '{n} days, arrows ≥{min} %: right {r_h1_lstm} of {t_h1_lstm} (descriptive, not a test)'},
    'count_d1': {'bg': '{n} дни, стрелки ≥{min} %: lstm позна {r_d1_lstm} от {t_d1_lstm}, '
                       'gbm {r_d1_gbm} от {t_d1_gbm} (описателно, не е тест)',
                 'en': '{n} days, arrows ≥{min} %: lstm right {r_d1_lstm} of {t_d1_lstm}, '
                       'gbm {r_d1_gbm} of {t_d1_gbm} (descriptive, not a test)'},
    'filter': {'bg': 'Филтър ≥{min} %: избор за показване, не е тестван праг',
               'en': 'Filter ≥{min} %: a display choice, not a tested rule'},
    'second_tip': {'bg': 'външен модел — изследване, без доказано предимство; съгласието е описателно',
                   'en': 'external model — research, no proven edge; agreement is descriptive'},
}


def second_opinion(now: datetime, lang: str) -> str:
    """The external models' panel line (change tsfm-second-opinion, design D9):
    each model's current gold size and its agreement with h1_lstm, read from
    results/external_tsfm/forward/ by the read-only second-opinion view. ''
    when no external forecast is current or the view cannot be read: the chart
    then simply shows no line."""
    try:
        from src.external.tsfm.view import gold_line
        wall = now.astimezone(LABEL_TZ).replace(tzinfo=None)
        return gold_line(lang, now=pd.Timestamp(wall, tz='UTC'))
    except Exception:                                    # noqa: BLE001 -- the chart never depends on it
        return ''


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float('inf') else None


def _wall(label: str) -> datetime:
    """A bar label as naive Berlin wall time (stored with a '+00:00' that is not UTC)."""
    return pd.Timestamp(label).replace(tzinfo=None).to_pydatetime()


def _sofia(wall: datetime) -> datetime:
    return wall.replace(tzinfo=LABEL_TZ).astimezone(SOFIA)


def fmt(v: float, nd: int, lang: str, signed: bool = False) -> str:
    """BG writes a decimal comma; a negative number takes a true minus sign."""
    r = round(float(v), nd)
    s = f'{abs(r):.{nd}f}'
    if lang == 'bg':
        s = s.replace('.', ',')
    if r < 0:
        return '−' + s
    return ('+' + s) if signed and r > 0 else s


def day_text(d: date, lang: str) -> str:
    return f'{d:%d.%m}' if lang == 'bg' else f'{MONTHS_EN[d.month - 1]} {d:%d}'


def hour_span(start: datetime) -> str:
    return f'{start:%H:%M}–{start + timedelta(hours=1):%H:%M}'


def hour_sofia(wall_hour: datetime, lang: str) -> str:
    s = _sofia(wall_hour)
    return f'{day_text(s, lang)}, {hour_span(s)}'


def direction_of(p_up: float) -> tuple[str, float]:
    """The /gold rule (summary.current_forecast): up when P(up) > 0.5, and the
    probability of the side shown."""
    return ('up', p_up) if p_up > 0.5 else ('down', 1.0 - p_up)


def _outcome(direction: str, sett: dict | None) -> tuple[str, float | None]:
    if sett is None:
        return 'open', None
    ret = _float(sett.get('realised_ret_pct'))
    if str(sett.get('eligible')).strip().lower() != 'true' or sett.get('realised_dir') in ('', None):
        return 'void', ret
    hit = (int(float(sett['realised_dir'])) == 1) == (direction == 'up')
    return ('right' if hit else 'wrong'), ret


def _outcome_text(outcome: str, ret: float | None, lang: str, daily: bool = False) -> str:
    if outcome == 'open':
        return TEXT['open_day' if daily else 'open'][lang]
    o = TEXT[outcome][lang]
    return o + (f' {fmt(ret, 2, lang, True)} %' if outcome in ('right', 'wrong') and ret is not None else '')


def _usd(vol, close, lang, at=False):
    if vol is None or close is None:
        return ''
    u = fmt(vol / 100.0 * close, 2, lang)
    return TEXT['usd_at'][lang].format(u=u, c=fmt(close, 2, lang)) if at else TEXT['usd'][lang].format(u=u)


def _base(p: dict, sett: dict | None, wall: datetime, cadence: str) -> tuple[dict, float | None, float | None, int]:
    vol, close = _float(p.get('vol_pct')), _float(p.get('as_of_close'))
    d, prob = direction_of(_float(p.get('p_up')))
    pct = int(round(prob * 100))
    outcome, _ = _outcome(d, sett)
    end = wall + (timedelta(days=1) if cadence == 'D1' else timedelta(hours=1))
    row = {'kind': 'forecast', 'key': p['key'], 'cell': p['cell'], 'cadence': cadence,
           'hour_label': wall.strftime(MQL_TIME), 'hour_end_label': end.strftime(MQL_TIME), 'direction': d,
           'prob_pct': pct, 'vol_pct': '' if vol is None else round(vol, 6),
           'close': '' if close is None else close, 'outcome': outcome}
    return row, vol, close, pct


def _forecast_row(p: dict, sett: dict | None) -> dict:
    wall = _wall(p['as_of_bar']) + timedelta(hours=1)
    row, vol, close, pct = _base(p, sett, wall, 'H1')
    d, ret = row['direction'], _outcome(row['direction'], sett)[1]
    for lang in LANGS:
        o = _outcome_text(row['outcome'], ret, lang)
        move = (TEXT['move'][lang].format(v=fmt(vol, 2, lang)) + _usd(vol, close, lang, at=True)) if vol is not None else ''
        tip = [TEXT['title_forecast'][lang].format(h=hour_sofia(wall, lang)),
               TEXT['direction'][lang].format(d=TEXT[d][lang], p=pct), move,
               TEXT['outcome'][lang].format(o=o), TEXT['research'][lang]]
        row[f'text_{lang}'] = f'{pct} %' if row['outcome'] == 'open' else f'{pct} % · {o}'
        row[f'tip_{lang}'] = '\n'.join(t for t in tip if t)
        row[f'panel_{lang}'] = TEXT['panel_forecast'][lang].format(
            h=hour_span(_sofia(wall)), d=TEXT[d][lang], p=pct, v=fmt(vol, 2, lang) if vol is not None else '?')
        row[f'hour_sofia_{lang}'] = hour_sofia(wall, lang)
    return row


def mark_day(p: dict, sett: dict | None) -> date:
    """The day a daily forecast is for: the settled exit bar once known, before
    that the next business day, as /gold dates it (summary.target_window_end)."""
    if sett and sett.get('exit_bar'):
        return _wall(sett['exit_bar']).date()
    return (pd.Timestamp(_wall(p['as_of_bar'])) + pd.offsets.BDay(1)).date()


def _daily_row(p: dict, sett: dict | None, day: date) -> dict:
    wall = datetime(day.year, day.month, day.day)
    row, vol, close, pct = _base(p, sett, wall, 'D1')
    d, ret, cell = row['direction'], _outcome(row['direction'], sett)[1], p['cell']
    tag = cell.split('_', 1)[1]
    for lang in LANGS:
        o = _outcome_text(row['outcome'], ret, lang, daily=True)
        move = (TEXT['move'][lang].format(v=fmt(vol, 2, lang)) + _usd(vol, close, lang, at=True)) if vol is not None else ''
        tip = [TEXT['title_day'][lang].format(c=cell, d=day_text(day, lang)),
               TEXT['direction'][lang].format(d=TEXT[d][lang], p=pct), move,
               TEXT['outcome'][lang].format(o=o), TEXT['research'][lang]]
        row[f'text_{lang}'] = f'{tag} {pct} %' if row['outcome'] == 'open' else f'{tag} {pct} % · {o}'
        row[f'tip_{lang}'] = '\n'.join(t for t in tip if t)
        row[f'panel_{lang}'] = (TEXT['panel_day'][lang].format(c=cell, d=day_text(day, lang), dir=TEXT[d][lang], p=pct)
                                + (TEXT['panel_move'][lang].format(v=fmt(vol, 2, lang)) if vol is not None else ''))
        row[f'hour_sofia_{lang}'] = day_text(day, lang)
    return row


def _alert_row(a: dict, pred: dict | None, sett: dict | None) -> dict:
    wall = _wall(a['as_of_bar']) + timedelta(hours=1)
    vol = _float(a.get('vol_pct'))
    close = _float(pred.get('as_of_close')) if pred else None
    real = _float(sett.get('realised_vol_pct')) if sett else None
    row = {'kind': 'alert', 'key': a['key'], 'cell': CELL, 'cadence': 'H1', 'hour_label': wall.strftime(MQL_TIME),
           'hour_end_label': (wall + timedelta(hours=1)).strftime(MQL_TIME), 'direction': '',
           'prob_pct': '', 'vol_pct': round(vol, 6), 'close': '' if close is None else close,
           'outcome': 'open' if sett is None else 'sized'}
    for lang in LANGS:
        size = f'±{fmt(vol, 2, lang)} %' + _usd(vol, close, lang)
        actual = TEXT['actual'][lang].format(v=fmt(real, 2, lang)) if real is not None else None
        o = TEXT['open'][lang] if sett is None else (actual or TEXT['void'][lang])
        tip = [TEXT['title_alert'][lang].format(h=hour_sofia(wall, lang)),
               TEXT['move'][lang].format(v=fmt(vol, 2, lang)) + _usd(vol, close, lang, at=True),
               TEXT['unknown'][lang], TEXT['outcome'][lang].format(o=o), TEXT['research'][lang]]
        row[f'text_{lang}'] = size if sett is None else f'{size} · {o}'
        row[f'tip_{lang}'] = '\n'.join(tip)
        row[f'panel_{lang}'] = TEXT['panel_alert'][lang].format(v=fmt(vol, 2, lang), usd=_usd(vol, close, lang))
        row[f'hour_sofia_{lang}'] = hour_sofia(wall, lang)
    return row


def build(cfg: dict | None = None, now: datetime | None = None) -> dict:
    """{'meta', 'rows', 'days', 'd1_days', 'right', 'wrong'}. Hourly marks keep the
    last DAYS trading days -- the Sofia dates of the forecast hours, so the
    Sunday 23:00 weekly-open bar makes no phantom Sunday; daily marks the last
    DAYS_D1 forecast days. right/wrong count the hourly arrows, unfiltered."""
    cfg = cfg or load_config()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fwd = repo_path(cfg['paths']['forward'])
    preds = {}
    for r in _rows(os.path.join(fwd, 'predictions.csv')):
        if r.get('cell') in DRAWN.get(r.get('cadence'), ()) and _float(r.get('p_up')) is not None:
            preds[r['key']] = r
    setts = {r['key']: r for r in _rows(os.path.join(fwd, 'settlements.csv'))}
    h1 = {k: p for k, p in preds.items() if p['cadence'] == 'H1'}
    day = lambda as_of: _sofia(_wall(as_of) + timedelta(hours=1)).date()  # noqa: E731
    days = sorted({day(p['as_of_bar']) for p in h1.values()})[-DAYS:]
    rows = [_forecast_row(p, setts.get(k)) for k, p in h1.items() if day(p['as_of_bar']) >= days[0]]
    d1 = {k: (p, mark_day(p, setts.get(k))) for k, p in preds.items() if p['cadence'] == 'D1'}
    d1_days = sorted({m for _, m in d1.values()})[-DAYS_D1:]
    rows += [_daily_row(p, setts.get(k), m) for k, (p, m) in d1.items() if m >= d1_days[0]]
    a_setts = {r['key']: r for r in _rows(os.path.join(fwd, 'alert_settlements.csv'))}
    alerts = {}
    for a in _rows(os.path.join(fwd, 'alerts.csv')):
        if a.get('cell') == CELL and days and day(a['as_of_bar']) >= days[0]:
            alerts[a['key']] = a
    rows += [_alert_row(a, h1.get(k), a_setts.get(k)) for k, a in alerts.items()]
    rows.sort(key=lambda r: (r['hour_label'], r['kind'] != 'forecast', r['cell']))
    hourly = [r for r in rows if r['kind'] == 'forecast' and r['cadence'] == 'H1']
    sofia_now = now.astimezone(SOFIA)
    meta = {'version': VERSION, 'generated_utc': now.strftime('%Y.%m.%d %H:%M:%S'), 'label_tz': 'Europe/Berlin',
            'label_utc_offset_min': int(now.astimezone(LABEL_TZ).utcoffset().total_seconds() // 60)}
    for lang in LANGS:
        meta[f'written_{lang}'] = TEXT['written'][lang].format(t=f'{day_text(sofia_now, lang)} {sofia_now:%H:%M}')
        meta[f'count_h1_{lang}'] = TEXT['count_h1'][lang].replace('{n}', str(len(days)))
        meta[f'count_d1_{lang}'] = TEXT['count_d1'][lang].replace('{n}', str(len(d1_days)))
        meta[f'filter_{lang}'] = TEXT['filter'][lang]
        meta[f'research_{lang}'] = TEXT['research'][lang]
        meta[f'second_{lang}'] = second_opinion(now, lang)
        meta[f'second_tip_{lang}'] = TEXT['second_tip'][lang]
    return {'meta': meta, 'rows': rows, 'days': days, 'd1_days': d1_days,
            'right': sum(r['outcome'] == 'right' for r in hourly), 'wrong': sum(r['outcome'] == 'wrong' for r in hourly)}


# ── the file ────────────────────────────────────────────────────────────────

def export_path(env=None) -> str:
    env = os.environ if env is None else env
    base = env.get(COMMON_ENV)
    if not base:
        appdata = env.get('APPDATA')
        if not appdata:
            raise RuntimeError(f'no APPDATA and no {COMMON_ENV}: cannot find the MT5 Common\\Files folder')
        base = os.path.join(appdata, 'MetaQuotes', 'Terminal', 'Common', 'Files')
    return os.path.join(base, SUBDIR, FILE_NAME)


def _field(v) -> str:
    """One TSV field: no tab, no CR, and a line break written as the two characters \\n."""
    return str(v).replace('\t', ' ').replace('\r', '').replace('\n', '\\n')


def render(data: dict) -> str:
    meta = '\t'.join(['#meta'] + [f'{k}={_field(v)}' for k, v in data['meta'].items()])
    lines = [meta, '\t'.join(ROW_FIELDS)]
    lines += ['\t'.join(_field(r.get(f, '')) for f in ROW_FIELDS) for r in data['rows']]
    return '\n'.join(lines) + '\n'


def write_atomic(path: str, text: str, replace=os.replace, sleep=time.sleep) -> None:
    """Write a .tmp beside the file, then replace it. The indicator may hold the
    file open for a few milliseconds while it reads, so a refused replace is
    retried; the last refusal is raised for the logger to record."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    for attempt in range(REPLACE_TRIES):
        try:
            replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            sleep(REPLACE_WAIT_S)


def export(cfg: dict | None = None, now: datetime | None = None, path: str | None = None, **write_kw) -> str:
    """Rebuild the chart file; a one-line summary."""
    data = build(cfg, now)
    path = path or export_path()
    write_atomic(path, render(data), **write_kw)
    span = f"{data['days'][0]}..{data['days'][-1]}" if data['days'] else 'no forecasts'
    return f"{len(data['rows'])} mark(s), {len(data['days'])} day(s) {span}, {len(data['d1_days'])} daily day(s)"


def main(argv=None):
    argparse.ArgumentParser(description='rebuild the gold signals file the MT5 indicator reads').parse_args(argv)
    path = export_path()
    print(f'{export(path=path)} -> {path}')


if __name__ == '__main__':
    main()
