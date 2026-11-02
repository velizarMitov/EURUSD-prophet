"""Read-only payload for GET /api/second-opinion and /second-opinion (design D8;
spec external-forecasts/second-opinion).

It reads recorded files only -- the external forward logs, gold's h1_lstm
predictions and alert threshold, EUR/USD's h1_gbm one-hour predictions, the
replay and the verdicts -- and runs no model, fetches no market data, writes
nothing and imports no model library.

Agreement rules (descriptive, never a signal):
    gold size        the same side of the large-move alert threshold as h1_lstm
    gold direction   P(up) on the same side of 0.5 as h1_lstm
    EUR/USD size     above or below the frozen GARCH x hour-of-day baseline
    EUR/USD dir      P(up) on the same side of 0.5 as h1_gbm's one-hour forecast
P(up) is shown, marked "not registered, not judged". The forward record is
descriptive until the registered score is written; then the verdict is shown
exactly as recorded.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import INSTRUMENTS, MODELS, REPO, VERDICTS, repo_path
from . import sources as S

MIN_RECORD = 30
NAMES = {'chronos2': 'Chronos-2', 'timesfm25': 'TimesFM 2.5'}
SHORT = {'chronos2': 'Chronos', 'timesfm25': 'TimesFM'}
INHOUSE = {'gold': 'h1_lstm', 'eurusd': 'h1_gbm'}

TEXTS = {
    'label': {'bg': 'външен модел, zero-shot — изследване, без доказано предимство',
              'en': 'external model, zero-shot — research, no proven edge'},
    'title': {'bg': 'Второ мнение', 'en': 'Second opinion'},
    'subtitle': {'bg': 'Какво очакват два външни модела за размера на следващия час, до нашите модели.',
                 'en': 'What two external models expect for the size of the next hour, beside our own models.'},
    'gold': {'bg': 'Злато', 'en': 'Gold'},
    'eurusd': {'bg': 'EUR/USD', 'en': 'EUR/USD'},
    'size': {'bg': 'Очакван размер на движението', 'en': 'Expected size of the move'},
    'hour': {'bg': 'Час (София)', 'en': 'Hour (Sofia)'},
    'p_up': {'bg': 'Вероятност за покачване', 'en': 'Probability of a rise'},
    'not_registered': {'bg': 'не е регистрирана, не се оценява', 'en': 'not registered, not judged'},
    'no_current': {'bg': 'няма текуща прогноза', 'en': 'no current forecast'},
    'last_forecast': {'bg': 'Последна прогноза', 'en': 'Last forecast'},
    'never': {'bg': 'Все още няма прогнози от външните модели.', 'en': 'No external forecasts yet.'},
    'agree': {'bg': 'съгласен', 'en': 'agrees'},
    'disagree': {'bg': 'не е съгласен', 'en': 'disagrees'},
    'nothing': {'bg': 'няма с какво да се сравни', 'en': 'nothing to compare'},
    'above': {'bg': 'над базата', 'en': 'above the baseline'},
    'below': {'bg': 'под базата', 'en': 'below the baseline'},
    'agreement': {'bg': 'Съгласие (описателно)', 'en': 'Agreement (descriptive)'},
    'agree_size': {'bg': 'по размер', 'en': 'on size'},
    'agree_dir': {'bg': 'по посока', 'en': 'on direction'},
    'threshold': {'bg': 'праг за голямо движение', 'en': 'large-move threshold'},
    'baseline': {'bg': 'база GARCH × час', 'en': 'GARCH × hour baseline'},
    'record': {'bg': 'Резултати досега', 'en': 'Record so far'},
    'descriptive': {'bg': 'описателно — не е присъда', 'en': 'descriptive — not a verdict'},
    'too_few': {'bg': 'твърде малко, за да се каже', 'en': 'too few to say'},
    'settled': {'bg': 'приключени часове', 'en': 'settled hours'},
    'mae': {'bg': 'средна абсолютна грешка', 'en': 'mean absolute error'},
    'vs_baseline': {'bg': 'спрямо базата', 'en': 'vs the baseline'},
    'vs_inhouse': {'bg': 'спрямо h1_lstm', 'en': 'vs h1_lstm'},
    'verdict': {'bg': 'Регистрирана присъда', 'en': 'Registered verdict'},
    'second_link': {'bg': 'Цялото второ мнение', 'en': 'The full second opinion'},
}


LINE_WORDS = {'agree': {'bg': 'е съгласен', 'en': 'agrees'},
              'disagree': {'bg': 'не е съгласен', 'en': 'disagrees'},
              'nothing': {'bg': 'няма с какво да се сравни', 'en': 'nothing to compare'}}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def label_now() -> pd.Timestamp:
    """The broker label clock now: Berlin wall time stamped UTC (CLAUDE.md invariant 6)."""
    return pd.Timestamp.now(tz='Europe/Berlin').tz_localize(None).tz_localize('UTC')


def sofia_span(as_of) -> str:
    """The forecast hour in Sofia time: the bar after the as-of bar, + 1 h (Sofia
    is Berlin + 1 h all year)."""
    start = pd.Timestamp(as_of) + pd.Timedelta(hours=2)
    return f'{start:%H:%M}–{start + pd.Timedelta(hours=1):%H:%M}'


def is_current(as_of, now) -> bool:
    """Current while the forecast hour (as-of + 1 h .. as-of + 2 h) is open."""
    return pd.Timestamp(now) < pd.Timestamp(as_of) + pd.Timedelta(hours=2)


def _read(name, repo):
    from .forward_logger import out_path, read_rows
    return read_rows(out_path(name, repo))


def latest(preds: list[dict], model: str, instrument: str) -> dict | None:
    rows = [r for r in preds if r['model'] == model and r['instrument'] == instrument]
    return max(rows, key=lambda r: r['as_of_bar']) if rows else None


def agreement(instrument: str, ext: dict | None, inhouse: dict | None, threshold: float | None,
              baseline: float | None) -> dict:
    """{'size': ..., 'direction': ...} with keys of TEXTS. `ext` and `inhouse`
    are for the same as-of bar or None."""
    out = {'size': 'nothing', 'direction': 'nothing'}
    if ext is None:
        return out
    size, p = _num(ext.get('size_pct')), _num(ext.get('p_up'))
    if instrument == 'gold':
        hs, hp = (None, None) if inhouse is None else (_num(inhouse.get('vol_pct')), _num(inhouse.get('p_up')))
        if size is not None and hs is not None and threshold is not None:
            out['size'] = 'agree' if (size >= threshold) == (hs >= threshold) else 'disagree'
    else:
        hp = None if inhouse is None else _num(inhouse.get('p_up'))
        if size is not None and baseline is not None:
            out['size'] = 'above' if size > baseline else 'below'
    if p is not None and hp is not None:
        out['direction'] = 'agree' if (p > 0.5) == (hp > 0.5) else 'disagree'
    return out


def record(setts: list[dict], model: str, instrument: str, verdicts: dict) -> dict:
    """The descriptive forward record of one cell; the verdict exactly as written."""
    cell = f'external_tsfm:{model}_{instrument}'
    rows = [r for r in setts if r['model'] == model and r['instrument'] == instrument
            and str(r.get('eligible')) == 'True' and _num(r.get('size_pct')) is not None]
    out = {'cell': cell, 'n': len(rows), 'verdict': verdicts.get(cell)}
    if len(rows) < MIN_RECORD:
        return {**out, 'state': 'too_few'}
    y = [_num(r['realised_abs_pct']) for r in rows]
    m = [abs(_num(r['size_pct']) - yy) for r, yy in zip(rows, y)]
    out.update(state='descriptive', mae=sum(m) / len(m))
    for key, col in (('vs_baseline', 'baseline_size_pct'), ('vs_inhouse', 'inhouse_size_pct')):
        pair = [(abs(_num(r[col]) - yy), mm) for r, yy, mm in zip(rows, y, m) if _num(r.get(col)) is not None]
        if len(pair) >= MIN_RECORD:
            out[key] = {'n': len(pair), 'other_mae': sum(a for a, _ in pair) / len(pair),
                        'model_mae': sum(b for _, b in pair) / len(pair),
                        'difference': sum(a - b for a, b in pair) / len(pair)}
    return out


def build(repo: str = REPO, now=None) -> dict:
    now = pd.Timestamp(now) if now is not None else label_now()
    preds, setts = _read('predictions', repo), _read('settlements', repo)
    path = repo_path(VERDICTS, repo)
    verdicts = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            verdicts = {v['cell']: v for v in json.load(fh)}
    threshold = S.gold_alert_threshold(repo)
    lstm, gbm = S.gold_rows('h1_lstm', repo), S.eurusd_h1_gbm(repo)
    out = {'label': TEXTS['label'], 'texts': TEXTS, 'threshold_pct': threshold, 'instruments': {},
           'last_forecast_at': max((r['logged_at'] for r in preds), default=None)}
    for inst in INSTRUMENTS:
        block = {'models': []}
        for model in MODELS:
            r = latest(preds, model, inst)
            cur = r if r is not None and is_current(r['as_of_bar'], now) else None
            inhouse = None
            if cur is not None:
                inhouse = (lstm if inst == 'gold' else gbm).get(cur['as_of_bar'])
            base = _num(cur.get('baseline_size_pct')) if cur is not None else None
            entry = {'model': model, 'name': NAMES[model], 'short': SHORT[model],
                     'last_as_of': r['as_of_bar'] if r else None, 'last_logged_at': r['logged_at'] if r else None,
                     'config_id': r['config_id'] if r else None, 'current': None,
                     'agreement': agreement(inst, cur, inhouse, threshold, base),
                     'record': record(setts, model, inst, verdicts)}
            if cur is not None:
                size, close = _num(cur['size_pct']), _num(cur['as_of_close'])
                entry['current'] = {
                    'as_of_bar': cur['as_of_bar'], 'hour_sofia': sofia_span(cur['as_of_bar']), 'size_pct': size,
                    'size_usd': size * close / 100 if (size is not None and close is not None and inst == 'gold') else None,
                    'as_of_close': close, 'p_up': _num(cur['p_up']), 'p_up_status': 'not_registered',
                    'baseline_size_pct': base,
                    'inhouse': None if inhouse is None else {
                        'name': INHOUSE[inst], 'size_pct': _num(inhouse.get('vol_pct')), 'p_up': _num(inhouse.get('p_up'))}}
            block['models'].append(entry)
        out['instruments'][inst] = block
    return out


def gold_line(lang: str = 'bg', repo: str = REPO, now=None) -> str:
    """The MT5 panel's second-opinion line (spec gold-forecasting/mt5-chart-signals,
    "The panel shows the second opinion"); '' when no external gold forecast is current."""
    view = build(repo, now)
    cur = [m for m in view['instruments']['gold']['models'] if m['current'] and m['current']['size_pct'] is not None]
    if not cur:
        return ''

    def pct(x):
        s = f'{x:.2f}'
        return s.replace('.', ',') if lang == 'bg' else s
    head = {'bg': 'Второ мнение', 'en': 'Second opinion'}[lang]
    sizes = ', '.join(f'{m["short"]} ±{pct(m["current"]["size_pct"])} %' for m in cur)
    if all(m['agreement']['size'] == 'nothing' for m in cur):
        return f'{head}: {sizes}'
    word = {k: v[lang] for k, v in LINE_WORDS.items()}
    tail = {'bg': 'с h1_lstm', 'en': 'with h1_lstm'}[lang]
    parts = ', '.join(f'{m["short"]} {word[m["agreement"]["size"]]}' for m in cur)
    size_word = {'bg': 'размер', 'en': 'size'}[lang]
    return f'{head}: {sizes} · {size_word}: {parts} {tail}'
