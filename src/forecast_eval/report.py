"""The operator view (tasks 15.1, 15.2; spec forward-arbiter "Operator view of
the forward log", "The operator view shows direction, never a cost verdict").

Writes ONE self-contained HTML file, `results/forward_eval/dashboard.html`, from
the forward logs alone. It is regenerated at the end of every logging run, so it
is never more than one cadence old.

  * It reads only `results/forward_eval/*.csv`, the study record and the power
    table. It is NOT part of the serving application: no route, template or
    module of `api.py` / `src/inference.py` / `src/paper_trading.py` is touched,
    which the change's Non-Goals require.
  * Times are shown in the OWNER's clock (Europe/Sofia). The logs carry bar
    labels, which are Europe/Berlin wall clock (m15_data), so every displayed
    time is shifted by the fixed one-hour difference between the two zones.
  * **No spread, breakeven or net-profit figure reaches the screen.** The owner
    accounts for trading cost themselves (2026-10-02). Those columns stay in the
    logs and in the pre-registration, where the project's methodology requires a
    measured cost to exist -- they simply decide nothing and are not displayed.
  * Running accuracy is always labelled "interim" until the registered sample
    size is reached, with the remaining count beside it, because a verdict before
    that n is exactly the early-stopping error the pre-registration exists to
    prevent.
"""

from __future__ import annotations

import argparse
import html
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import m15_data as MD
from . import prereg as P
from . import record as R
from . import targets as T

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FWD = os.path.join(REPO, 'results', 'forward_eval')
DASHBOARD = os.path.join(FWD, 'dashboard.html')
RECORD = os.path.join(REPO, 'results', 'horizon_study', 'study_record.json')

BAR = {'M15': pd.Timedelta(minutes=15), 'H1': pd.Timedelta(hours=1), 'D1': pd.Timedelta(days=1)}
HORIZON_LABEL = {
    'M15': lambda h: f'{h * 15} мин',
    'H1': lambda h: '1 час' if h == 1 else f'{h} часа',
    'D1': lambda h: '1 ден' if h == 1 else f'{h} дни',
}
EDGE, FAMILY_ALPHA = 0.03, 0.05
CURVES = os.path.join(REPO, 'results', 'horizon_study', 'horizon_curves.csv')
# Below this many settled forecasts an accuracy is noise, not a number worth
# reading: the study itself refuses to score a cell with fewer.
MIN_SETTLED_TO_SHOW = 30
# Columns that must never be rendered (spec: no cost arithmetic on the screen).
COST_WORDS = ('spread', 'breakeven', 'net_per_trade', 'swap', 'cost', 'pip')


def _read(name: str, out: str = FWD) -> pd.DataFrame:
    p = os.path.join(out, f'{name}.csv')
    if not os.path.exists(p) or not os.path.getsize(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def owner_time(stamp) -> pd.Timestamp:
    """A bar label on the owner's clock. Sofia is Berlin + 1 h all year."""
    return MD.owner_clock(pd.DatetimeIndex([pd.Timestamp(stamp)]))[0]


def _fmt(stamp) -> str:
    return '—' if stamp is None or pd.isna(stamp) else owner_time(stamp).strftime('%d.%m %H:%M')


def _pct(fraction, digits: int = 1) -> str:
    """A fraction as a percentage string. Scaling goes through
    `targets._to_pct`, the package's single `* 100` site -- a guarded invariant,
    so this module must not multiply by 100 itself."""
    if fraction is None or not np.isfinite(fraction):
        return '—'
    return f'{T._to_pct(float(fraction)):.{digits}f}%'


def _num(x) -> str:
    """Thousands separated by a space. Applied per number -- a replace over the
    whole document would mangle the CSS. Rounded, not truncated: the registered
    n is 3,817, and int() would print 3,816."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return f'{round(float(x)):,}'.replace(',', ' ')


def direction_models(record_path: str = RECORD) -> set:
    """Models that forecast a direction at all. The volatility challengers have
    a direction head as a by-product of their multi-task architecture, but their
    declared target is the size of the move, so showing their `p_up` as a call
    would be inventing a signal nobody validated."""
    rec = R.load(record_path)
    if rec is None:
        return set()
    return {name for name, cfg in rec['challengers'].items() if cfg.get('kind') == 'direction'}


def history_evidence(curves_path: str = CURVES) -> dict:
    """{(model, horizon): label} from the historical study.

    This is the column that stops a confident-looking number from being read as
    a signal: Kronos routinely reports 100 % of its sampled paths going one way,
    and the study found no interval of it excluding 50 %."""
    if not os.path.exists(curves_path) or not os.path.getsize(curves_path):
        return {}
    d = pd.read_csv(curves_path)
    out = {}
    for _, r in d.iterrows():
        lo = pd.to_numeric(r.get('acc_ci_low'), errors='coerce')
        if not np.isfinite(lo):
            continue
        out[(r['model'], int(r['horizon']))] = (
            'над монета на историята' if lo > 0.5 else 'монета на историята')
    return out


def admitted_cells(record_path: str = RECORD):
    """The registered family, so the view marks exactly the cells that can ever
    reach a verdict."""
    rec = R.load(record_path)
    if rec is None:
        return set(), float('nan'), float('nan')
    cells = P.direction_cells(rec)
    adm, alpha, _table, _hist = P.fixed_point_family(cells, edge=EDGE,
                                                     cap_years=rec.get('cap_years', 3.0),
                                                     family_alpha=FAMILY_ALPHA)
    return set(adm), alpha, P.n_required(alpha, EDGE)


def latest_forecasts(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, horizon): the newest forecast, with the window the
    move is measured over -- from the as-of bar's CLOSE to the target bar's."""
    if preds.empty:
        return preds
    d = preds.copy()
    d['as_of'] = pd.to_datetime(d['as_of_bar'], utc=True, format='mixed')
    d['horizon'] = d['horizon'].astype(int)
    d = d.sort_values('as_of').groupby(['model', 'horizon'], as_index=False).last()
    dur = d['cadence'].map(BAR)
    d['from'] = d['as_of'] + dur
    d['until'] = d['as_of'] + (d['horizon'] + 1) * dur
    d['p_up'] = pd.to_numeric(d['p_up'], errors='coerce')
    d['direction'] = np.where(d['p_up'].isna(), '—', np.where(d['p_up'] >= 0.5, '+', '−'))
    return d


def running_accuracy(preds: pd.DataFrame, settles: pd.DataFrame) -> pd.DataFrame:
    """Settled forecasts per cell: how many and what share were right.

    Dry-run rows are included and labelled, because the point of showing them is
    to see the chain working -- never to decide anything."""
    if preds.empty or settles.empty:
        return pd.DataFrame()
    s = settles[['key', 'correct', 'scorable']].copy()
    s['correct'] = pd.to_numeric(s['correct'], errors='coerce')
    j = preds[['key', 'model', 'cadence', 'horizon', 'phase']].merge(s, on='key', how='inner')
    j = j[j['correct'].notna()]
    if j.empty:
        return pd.DataFrame()
    j['horizon'] = j['horizon'].astype(int)
    g = j.groupby(['model', 'cadence', 'horizon'], as_index=False).agg(
        n=('correct', 'size'), hits=('correct', 'sum'),
        scorable=('scorable', lambda x: int(x.astype(str).str.lower().eq('true').sum())))
    g['accuracy'] = g['hits'] / g['n']
    return g.sort_values(['cadence', 'horizon', 'model'])


def session_state(now=None) -> dict:
    """Where the owner's trading session is right now, on their own clock.

    `now` is real time; it is converted to the BAR-LABEL clock first, because the
    session window is expressed in labels. Testing a true UTC "now" against the
    window would open the session two hours late in summer."""
    label = MD.label_now(now)
    inside = bool(MD.session_mask(pd.DatetimeIndex([label]))[0])
    o = MD.owner_clock(pd.DatetimeIndex([label]))[0]
    return {'in_session': inside, 'owner_now': o.strftime('%d.%m %H:%M'),
            'window': '15:30 – 23:00', 'weekday': bool(o.dayofweek < 5)}


# ── rendering ───────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171a;--mut:#636c76;--line:#e3e6ea;
--up:#0a7d46;--dn:#b3261e;--warn:#8a5a00;--accent:#1f4e9c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#14171a;--card:#1c2024;--ink:#e8eaed;--mut:#9aa4ae;--line:#2b3136;
--up:#4ec98a;--dn:#ff7a70;--warn:#e0b150;--accent:#7aa7ef}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,
Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:16px}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:21px;margin:0 0 2px}
h2{font-size:16px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:12px}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;
font-weight:600;border:1px solid var(--line)}
.on{color:var(--up)}.off{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.cell{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.cell .h{font-size:12px;color:var(--mut);margin-bottom:4px}
.arrow{font-size:30px;font-weight:700;line-height:1}
.up{color:var(--up)}.dn{color:var(--dn)}
.pct{font-size:13px;color:var(--mut)}
.win{font-size:12px;color:var(--mut);margin-top:6px}
.ev{font-size:11px;margin-top:6px;padding-top:6px;border-top:1px solid var(--line)}
.ev.ok{color:var(--up)}.ev.no{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--mut);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;min-width:70px}
.bar>i{display:block;height:100%;background:var(--accent)}
.note{color:var(--mut);font-size:13px}
.star{color:var(--accent);font-weight:700}
ul{margin:8px 0 0;padding-left:20px}li{margin:4px 0}
.warn{color:var(--warn)}
.empty{color:var(--mut);font-style:italic}
"""


def _cell_card(r, admitted: set, evidence: dict) -> str:
    h = int(r['horizon'])
    key = (r['model'], r['cadence'], h)
    star = ' <span class="star" title="решава се — в регистрираното семейство">★</span>' \
        if key in admitted else ''
    cls = 'up' if r['direction'] == '+' else ('dn' if r['direction'] == '−' else 'off')
    pct = '—' if pd.isna(r['p_up']) else _pct(r['p_up'])
    ev = evidence.get((r['model'], h))
    ev_html = ''
    if ev:
        good = ev.startswith('над')
        ev_html = (f'<div class="ev {"ok" if good else "no"}">{ev}</div>')
    return (f'<div class="cell"><div class="h">{html.escape(str(r["model"]))}'
            f' · {HORIZON_LABEL[r["cadence"]](h)}{star}</div>'
            f'<div class="arrow {cls}">{r["direction"]}</div>'
            f'<div class="pct">{pct} нагоре</div>'
            f'<div class="win">{_fmt(r["from"])} → {_fmt(r["until"])}</div>'
            f'{ev_html}</div>')


def _accuracy_table(acc: pd.DataFrame, admitted: set, n_req: float) -> str:
    if acc.empty:
        return '<p class="empty">Още никоя прогноза не е приключила.</p>'
    rows = []
    # Rounded once, so the remaining count and the stated total agree: taking
    # int() here while the total prints round() left them one apart.
    target = round(float(n_req)) if np.isfinite(n_req) else None
    for _, r in acc.iterrows():
        key = (r['model'], r['cadence'], int(r['horizon']))
        decides = key in admitted
        left = max(target - int(r['n']), 0) if decides and target is not None else None
        pctbar = min(r['n'] / n_req, 1.0) if decides and np.isfinite(n_req) else 0.0
        enough = int(r['n']) >= MIN_SETTLED_TO_SHOW
        shown = _pct(r['accuracy']) if enough else '—'
        status = (f'твърде малко прогнози (под {MIN_SETTLED_TO_SHOW})' if not enough
                  else f'междинно — остават {_num(left)}' if left is not None
                  else 'без решение (извън семейството)')
        rows.append(
            f'<tr><td>{html.escape(str(r["model"]))}{" ★" if decides else ""}</td>'
            f'<td>{HORIZON_LABEL[r["cadence"]](int(r["horizon"]))}</td>'
            f'<td class="num">{_num(r["n"])}</td>'
            f'<td class="num">{shown}</td>'
            f'<td><div class="bar"><i style="width:{_pct(pctbar)}"></i></div></td>'
            f'<td class="note">{status}</td></tr>')
    return ('<table><tr><th>модел</th><th>хоризонт</th><th>приключили</th>'
            '<th>познати</th><th>до решение</th><th></th></tr>'
            + ''.join(rows) + '</table>')


def _problems(gaps: pd.DataFrame, fails: pd.DataFrame) -> str:
    out = []
    if gaps.empty and fails.empty:
        return '<p class="note">Няма пропуснати барове и няма грешки.</p>'
    if not gaps.empty:
        by = gaps.groupby('cadence').size().to_dict()
        out.append('<p class="warn">Пропуснати барове: ' +
                   ', '.join(f'{k} — {v}' for k, v in sorted(by.items())) +
                   '. Това е безвъзвратно: пропуснат бар никога не се прогнозира после.</p>')
    if not fails.empty:
        last = fails.tail(5)
        items = ''.join(f'<li>{html.escape(str(r["model"]))} '
                        f'{html.escape(str(r["cadence"]))} — '
                        f'{html.escape(str(r["error"])[:140])}</li>'
                        for _, r in last.iterrows())
        out.append(f'<p class="warn">Грешки: {len(fails)}. Последните:</p><ul>{items}</ul>')
    return ''.join(out)


EXPLAINER = """
<h2>Как се гледа</h2>
<div class="card">
<ul>
<li><b>+ или −</b> е посоката, която моделът очаква. Процентът под нея е неговата
вероятност за „нагоре“: над 50% значи +, под 50% значи −.</li>
<li><b>Двата часа под картата</b> са прозорецът, в който важи прогнозата, по твоето
време. Движението се мери от затварянето на единия бар до затварянето на другия.</li>
<li><b>★</b> маркира клетките от регистрираното семейство — единствените, които
някога могат да получат присъда. Останалите се записват само описателно.</li>
<li><b>Долният ред на картата е най-важният.</b> „над монета на историята“ значи, че
този модел на този хоризонт е показал предимство на 8–11 години данни.
„монета на историята“ значи, че не е — каквото и число да показва днес.</li>
<li><b>Високият процент не е сила на сигнала.</b> Kronos редовно показва 90–100%,
защото брои колко от симулираните му пътища свършват нагоре; на историята нито един
негов интервал не изключва 50%. Числото е увереност на модела, не доказателство.</li>
<li><b>Една отделна прогноза е почти монета.</b> Предимството, ако го има, е около
един процент и се вижда само в хиляди прогнози. Нормално е прогнозите за различни
хоризонти да си противоречат.</li>
<li><b>Моделите за волатилност не са тук.</b> Те прогнозират колко голямо ще е
движението, не в коя посока, затова не им се показва посока.</li>
<li><b>„Познати“ не е присъда.</b> Докато не се съберат регистрираните прогнози,
числото е междинно и не решава нищо — точно затова до него стои колко остават.</li>
<li><b>Разходите не са тук.</b> Спредът и нетната печалба ги сметкаш ти; остават
записани в CSV файловете, но не се показват и не решават нищо.</li>
</ul>
</div>
"""


def render(preds, settles, gaps, fails, admitted, alpha, n_req, state, generated,
           direction_only=None, evidence=None) -> str:
    evidence = evidence or {}
    latest = latest_forecasts(preds)
    acc = running_accuracy(preds, settles)
    if direction_only:
        if not latest.empty:
            latest = latest[latest['model'].isin(direction_only)]
        if not acc.empty:
            acc = acc[acc['model'].isin(direction_only)]
    sess = ('<span class="badge on">в сесия</span>' if state['in_session']
            else '<span class="badge off">извън сесия</span>')
    phase = 'dry_run'
    if not preds.empty and 'phase' in preds:
        phase = str(preds['phase'].iloc[-1])
    phase_note = ('пробен запис — нищо не се зачита' if phase == 'dry_run'
                  else 'зачита се')

    if latest.empty:
        cards = '<p class="empty">Още няма записана прогноза.</p>'
    else:
        # The cells that can reach a verdict come first inside each cadence.
        latest = latest.assign(
            _rank=[0 if (r['model'], r['cadence'], int(r['horizon'])) in admitted else 1
                   for _, r in latest.iterrows()]
        ).sort_values(['_rank', 'horizon', 'model'])
        blocks = []
        for cad in ('M15', 'H1', 'D1'):
            part = latest[latest['cadence'] == cad]
            if part.empty:
                continue
            title = {'M15': '15-минутни барове — твоята сесия',
                     'H1': 'Часови барове', 'D1': 'Дневни барове'}[cad]
            blocks.append(f'<h2>{title}</h2>\n<div class="grid">' +
                          ''.join(_cell_card(r, admitted, evidence) for _, r in part.iterrows()) +
                          '</div>')
        cards = ''.join(blocks)

    return f"""<!doctype html>
<html lang="bg"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>EUR/USD прогнози</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>EUR/USD — какво казват моделите</h1>
<div class="sub">Обновено {generated} твое време · {sess} ·
сесия {state['window']} · режим <b>{html.escape(phase)}</b> ({phase_note})</div>

{cards}

<h2>Точност досега</h2>
<div class="card">{_accuracy_table(acc, admitted, n_req)}
<p class="note">Регистрирани прогнози за решение: <b>{_num(n_req)}</b> на клетка,
при alpha {alpha:.6f}.</p></div>

<h2>Състояние на записа</h2>
<div class="card">{_problems(gaps, fails)}
<p class="note">Записани прогнози: {_num(len(preds))} ·
приключили: {_num(len(settles))}</p></div>

{EXPLAINER}
<div class="sub" style="margin-top:22px">Описателно. Присъда се издава само при
регистрирания брой прогнози, след commit на предварителната регистрация.</div>
</div></body></html>"""


def write_dashboard(out: str = FWD, path: str | None = None, record_path: str = RECORD,
                    now=None) -> str:
    path = path or os.path.join(out, 'dashboard.html')
    admitted, alpha, n_req = admitted_cells(record_path)
    generated = owner_time(MD.label_now(now)).strftime('%d.%m.%Y %H:%M')
    page = render(_read('predictions', out), _read('settlements', out), _read('gaps', out),
                  _read('failures', out), admitted, alpha, n_req,
                  session_state(now), generated,
                  direction_only=direction_models(record_path), evidence=history_evidence())
    low = page.lower()
    leaked = [w for w in COST_WORDS if w in low]
    if leaked:
        raise AssertionError(f'cost arithmetic reached the operator view: {leaked}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write the forward-log operator view.')
    ap.add_argument('--out', default=FWD)
    a = ap.parse_args(argv)
    print(write_dashboard(a.out))


if __name__ == '__main__':
    main()
