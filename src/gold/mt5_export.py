"""Gold signals on the MT5 chart (openspec change gold-mt5-chart-signals, spec
gold-forecasting/mt5-chart-signals).

Rebuilds ONE file that the indicator mql5/Indicators/ProphetGoldSignals.mq5
reads from the MT5 terminals' shared folder:

    %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\EURUSDProphet\\gold_signals.tsv

It holds the h1_lstm forecasts and the large-move alerts of the last 5 trading
days, each with its outcome once settled, and every text the chart shows, in BG
and EN. Python writes all the words so the wording rules are tested here; the
indicator only places them (design D2).

It is the one gold write outside results/gold/ and research_models/gold/, and
nothing in the repo reads it back. It reads only the forward logs: no model,
no market data. h1_gbm, the D1 cells, the baselines and predictions_v2.csv are
never read -- the owner chose h1_lstm only.

    python -m src.gold.mt5_export     rebuild the file now
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import load_config, repo_path

CELL = 'h1_lstm'
DAYS = 5
VERSION = 1
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

ROW_FIELDS = ['kind', 'key', 'hour_label', 'hour_end_label', 'direction', 'prob_pct', 'vol_pct', 'close',
              'outcome', 'text_bg', 'text_en', 'tip_bg', 'tip_en', 'panel_bg', 'panel_en',
              'hour_sofia_bg', 'hour_sofia_en']

# Size and direction only: never buy, sell, long, short, entry, stop, target or take-profit.
TEXT = {
    'up': {'bg': 'нагоре', 'en': 'up'},
    'down': {'bg': 'надолу', 'en': 'down'},
    'right': {'bg': 'позна', 'en': 'right'},
    'wrong': {'bg': 'сбърка', 'en': 'wrong'},
    'void': {'bg': 'не се брои', 'en': 'not counted'},
    'open': {'bg': 'часът е отворен', 'en': 'hour still open'},
    'actual': {'bg': 'реално {v} %', 'en': 'actual {v} %'},
    'title_forecast': {'bg': 'Злато · h1_lstm · {h} (София)', 'en': 'Gold · h1_lstm · {h} (Sofia time)'},
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
    'panel_alert': {'bg': '⚠ Голямо движение ±{v} %{usd} · посоката е неизвестна',
                    'en': '⚠ Large move ±{v} %{usd} · direction unknown'},
    'written': {'bg': 'данни от {t} (София)', 'en': 'data from {t} Sofia time'},
    'count': {'bg': '{n} дни: позна {r} от {t} (описателно, не е тест)',
              'en': '{n} days: right {r} of {t} (descriptive, not a test)'},
    'count_none': {'bg': '{n} дни: още няма приключили часове', 'en': '{n} days: no settled hours yet'},
}


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


def hour_span(start: datetime) -> str:
    return f'{start:%H:%M}–{start + timedelta(hours=1):%H:%M}'


def hour_sofia(wall_hour: datetime, lang: str) -> str:
    s = _sofia(wall_hour)
    day = f'{s:%d.%m}' if lang == 'bg' else f'{MONTHS_EN[s.month - 1]} {s:%d}'
    return f'{day}, {hour_span(s)}'


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


def _usd(vol, close, lang, at=False):
    if vol is None or close is None:
        return ''
    u = fmt(vol / 100.0 * close, 2, lang)
    return TEXT['usd_at'][lang].format(u=u, c=fmt(close, 2, lang)) if at else TEXT['usd'][lang].format(u=u)


def _forecast_row(p: dict, sett: dict | None) -> dict:
    wall = _wall(p['as_of_bar']) + timedelta(hours=1)
    p_up, vol, close = _float(p.get('p_up')), _float(p.get('vol_pct')), _float(p.get('as_of_close'))
    d, prob = direction_of(p_up)
    pct = int(round(prob * 100))
    outcome, ret = _outcome(d, sett)
    row = {'kind': 'forecast', 'key': p['key'], 'hour_label': wall.strftime(MQL_TIME),
           'hour_end_label': (wall + timedelta(hours=1)).strftime(MQL_TIME), 'direction': d,
           'prob_pct': pct, 'vol_pct': '' if vol is None else round(vol, 6),
           'close': '' if close is None else close, 'outcome': outcome}
    for lang in LANGS:
        o = TEXT[outcome][lang]
        if outcome in ('right', 'wrong') and ret is not None:
            o += f' {fmt(ret, 2, lang, True)} %'
        short = f'{pct} %' if outcome == 'open' else f'{pct} % · {o}'
        move = (TEXT['move'][lang].format(v=fmt(vol, 2, lang)) + _usd(vol, close, lang, at=True)) if vol is not None else ''
        tip = [TEXT['title_forecast'][lang].format(h=hour_sofia(wall, lang)),
               TEXT['direction'][lang].format(d=TEXT[d][lang], p=pct), move,
               TEXT['outcome'][lang].format(o=o), TEXT['research'][lang]]
        row[f'text_{lang}'] = short
        row[f'tip_{lang}'] = '\n'.join(t for t in tip if t)
        row[f'panel_{lang}'] = TEXT['panel_forecast'][lang].format(
            h=hour_span(_sofia(wall)), d=TEXT[d][lang], p=pct, v=fmt(vol, 2, lang) if vol is not None else '?')
        row[f'hour_sofia_{lang}'] = hour_sofia(wall, lang)
    return row


def _alert_row(a: dict, pred: dict | None, sett: dict | None) -> dict:
    wall = _wall(a['as_of_bar']) + timedelta(hours=1)
    vol = _float(a.get('vol_pct'))
    close = _float(pred.get('as_of_close')) if pred else None
    real = _float(sett.get('realised_vol_pct')) if sett else None
    row = {'kind': 'alert', 'key': a['key'], 'hour_label': wall.strftime(MQL_TIME),
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
    """{'meta', 'rows', 'days', 'right', 'wrong'} for the last DAYS trading days:
    the Sofia dates of the forecast hours, so the Sunday 23:00 weekly-open bar
    makes no phantom Sunday."""
    cfg = cfg or load_config()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fwd = repo_path(cfg['paths']['forward'])
    preds = {}
    for r in _rows(os.path.join(fwd, 'predictions.csv')):
        if r.get('cell') == CELL and r.get('cadence') == 'H1' and _float(r.get('p_up')) is not None:
            preds[r['key']] = r
    day = lambda as_of: _sofia(_wall(as_of) + timedelta(hours=1)).date()  # noqa: E731
    days = sorted({day(p['as_of_bar']) for p in preds.values()})[-DAYS:]
    first = days[0] if days else None
    setts = {r['key']: r for r in _rows(os.path.join(fwd, 'settlements.csv'))}
    rows = [_forecast_row(p, setts.get(k)) for k, p in preds.items() if day(p['as_of_bar']) >= first]
    a_setts = {r['key']: r for r in _rows(os.path.join(fwd, 'alert_settlements.csv'))}
    alerts = {}
    for a in _rows(os.path.join(fwd, 'alerts.csv')):
        if a.get('cell') == CELL and first is not None and day(a['as_of_bar']) >= first:
            alerts[a['key']] = a
    rows += [_alert_row(a, preds.get(k), a_setts.get(k)) for k, a in alerts.items()]
    rows.sort(key=lambda r: (r['hour_label'], r['kind'] != 'forecast'))
    right = sum(r['outcome'] == 'right' for r in rows)
    wrong = sum(r['outcome'] == 'wrong' for r in rows)
    sofia_now = now.astimezone(SOFIA)
    meta = {'version': VERSION, 'generated_utc': now.strftime('%Y.%m.%d %H:%M:%S'), 'label_tz': 'Europe/Berlin',
            'label_utc_offset_min': int(now.astimezone(LABEL_TZ).utcoffset().total_seconds() // 60)}
    for lang in LANGS:
        day_s = f'{sofia_now:%d.%m}' if lang == 'bg' else f'{MONTHS_EN[sofia_now.month - 1]} {sofia_now:%d}'
        meta[f'written_{lang}'] = TEXT['written'][lang].format(t=f'{day_s} {sofia_now:%H:%M}')
        meta[f'count_{lang}'] = (TEXT['count'][lang].format(n=len(days), r=right, t=right + wrong) if right + wrong
                                 else TEXT['count_none'][lang].format(n=len(days)))
        meta[f'research_{lang}'] = TEXT['research'][lang]
    return {'meta': meta, 'rows': rows, 'days': days, 'right': right, 'wrong': wrong}


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
    return f"{len(data['rows'])} mark(s), {len(data['days'])} day(s) {span}"


def main(argv=None):
    argparse.ArgumentParser(description='rebuild the gold signals file the MT5 indicator reads').parse_args(argv)
    path = export_path()
    print(f'{export(path=path)} -> {path}')


if __name__ == '__main__':
    main()
