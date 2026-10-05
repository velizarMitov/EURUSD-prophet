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
# The headline shows only cells that (a) cleared a coin on history and (b) end
# within this many minutes, which is what the owner can act on in one sitting.
# Everything else is real data but not a decision aid, so it is collapsed.
HEADLINE_MAX_MINUTES = 240
EVIDENCE_OK = 'над монета на историята'
EVIDENCE_NO = 'монета на историята'
AHEAD = {15: 'след 15 минути', 30: 'след 30 минути', 60: 'след 1 час',
         120: 'след 2 часа', 240: 'след 4 часа'}
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


def minutes_ahead(cadence: str, h: int) -> int:
    """How far ahead a cell forecasts, in minutes -- the only unit the owner
    needs, so horizons from different cadences line up on one scale."""
    return int(BAR[cadence].total_seconds() // 60) * int(h)


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
        out[(r['model'], int(r['horizon']))] = EVIDENCE_OK if lo > 0.5 else EVIDENCE_NO
    return out


def study_accuracy(curves_path: str = CURVES) -> dict:
    """{(model, horizon): accuracy on history} -- the number behind the tag."""
    if not os.path.exists(curves_path) or not os.path.getsize(curves_path):
        return {}
    d = pd.read_csv(curves_path)
    out = {}
    for _, r in d.iterrows():
        acc = pd.to_numeric(r.get('accuracy'), errors='coerce')
        if np.isfinite(acc):
            out[(r['model'], int(r['horizon']))] = float(acc)
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
h1{font-size:26px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:16px;margin:26px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:13px;color:var(--mut);margin:16px 0 8px;font-weight:600}
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
/* the headline: one block per horizon */
.hl{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px 18px;margin-bottom:12px}
.when{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;
margin-bottom:10px}
.calls{display:flex;flex-wrap:wrap;gap:22px}
.call{display:flex;align-items:baseline;gap:9px}
.call .ar{font-size:34px;line-height:1}
.call .wd{font-size:19px;font-weight:700;letter-spacing:.02em}
.call .pq{font-size:13px;color:var(--mut)}
.call.up .ar,.call.up .wd{color:var(--up)}
.call.dn .ar,.call.dn .wd{color:var(--dn)}
.call.off .ar,.call.off .wd{color:var(--mut)}
.agree{font-size:12px;color:var(--mut);margin-top:10px}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;margin-bottom:10px}
details>summary{cursor:pointer;font-size:14px;font-weight:600;color:var(--mut)}
details[open]>summary{margin-bottom:12px}
.trust p{margin:0 0 10px}
.trust p:last-child{margin-bottom:0;color:var(--mut);font-size:13px}
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


def _headline(latest: pd.DataFrame, evidence: dict) -> tuple:
    """The whole point of the page: for each horizon the owner can act on, what
    the models that actually cleared a coin on history are saying right now.

    Grouped by HOW FAR AHEAD, not by model: "what happens in the next hour" is
    the question, and a model's name is not an answer. Models with no historical
    evidence are left out entirely -- they are logged, but putting them here
    would present noise as a forecast.

    The calls are listed side by side and never combined into one arrow: no
    ensemble of these models was ever fitted or validated, so inventing a
    majority vote here would be a forecast nobody measured.
    """
    if latest.empty:
        return '', []
    rows = []
    for _, r in latest.iterrows():
        h = int(r['horizon'])
        if evidence.get((r['model'], h)) != EVIDENCE_OK:
            continue
        mins = minutes_ahead(r['cadence'], h)
        if mins > HEADLINE_MAX_MINUTES or mins not in AHEAD:
            continue
        rows.append({'mins': mins, **r})
    if not rows:
        return '', []
    blocks = []
    for mins in sorted({x['mins'] for x in rows}):
        group = [x for x in rows if x['mins'] == mins]
        calls = []
        for x in group:
            up = x['direction'] == '+'
            word = 'НАГОРЕ' if up else ('НАДОЛУ' if x['direction'] == '−' else 'НЯМА ОТГОВОР')
            cls = 'up' if up else ('dn' if x['direction'] == '−' else 'off')
            pct = '' if pd.isna(x['p_up']) else f'<span class="pq">{_pct(x["p_up"], 0)} за нагоре</span>'
            calls.append(f'<div class="call {cls}"><span class="ar">'
                         f'{"↑" if up else "↓" if x["direction"] == "−" else "·"}</span>'
                         f'<span class="wd">{word}</span>{pct}</div>')
        agree = ''
        if len(group) > 1:
            same = len({x['direction'] for x in group}) == 1
            agree = (f'<div class="agree">{"и двата модела казват едно и също" if same else "моделите не са съгласни"}'
                     f' — съгласието не е допълнително доказателство</div>')
        win = f'{_fmt(group[0]["from"])} → {_fmt(group[0]["until"])}'
        blocks.append(f'<div class="hl"><div class="when">{AHEAD[mins]}</div>'
                      f'<div class="calls">{"".join(calls)}</div>'
                      f'{agree}<div class="win">важи {win}</div></div>')
    return ''.join(blocks), rows


def _trust(headline_rows, acc: pd.DataFrame, study: dict, n_req) -> str:
    """Three plain sentences: how good these models were, how far the live test
    has got, and what that means. No jargon, no table."""
    if not headline_rows:
        return ('<p>Сесията ти още не е отворена или няма записана прогноза от модел '
                'с доказана стойност.</p>')
    # A RANGE, not the best cell: the headline mixes models that scored 51 and
    # 53 on history, and quoting only the best would overstate the weaker ones.
    scores = [study[(x['model'], int(x['horizon']))] for x in headline_rows
              if (x['model'], int(x['horizon'])) in study
              and np.isfinite(study[(x['model'], int(x['horizon']))])]
    if not scores:
        hits = '—'
    else:
        lo, hi = round(T._to_pct(min(scores))), round(T._to_pct(max(scores)))
        hits = f'{lo} от 100' if lo == hi else f'между {lo} и {hi} от 100'
    done = 0
    if not acc.empty:
        keys = {(x['model'], int(x['horizon'])) for x in headline_rows}
        sel = acc[[(m, int(h)) in keys for m, h in zip(acc['model'], acc['horizon'])]]
        done = int(sel['n'].max()) if len(sel) else 0
    target = round(float(n_req)) if np.isfinite(n_req) else None
    left = max(target - done, 0) if target else None
    return (
        f'<p><b>На историята тези модели познаваха {hits} пъти.</b> '
        f'Монета познава 50 от 100. Разликата е малка и се вижда само в хиляди прогнози — '
        f'всяка отделна прогноза горе е почти хвърляне на монета.</p>'
        f'<p><b>На живо още не е доказано.</b> Приключили прогнози: {_num(done)} от '
        f'{_num(target)}. Остават {_num(left)}.</p>'
        f'<p>Докато броячът не се напълни, нищо тук не е присъда — само наблюдение.</p>')


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
<details><summary>Как се чете тази страница</summary>
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
</details>
"""


def render(preds, settles, gaps, fails, admitted, alpha, n_req, state, generated,
           direction_only=None, evidence=None, study=None, back_href=None) -> str:
    """One screen: what the evidenced models say, and how far the live test has
    got. Everything else is real data but not a decision aid, so it is folded
    away behind a summary rather than competing for attention."""
    evidence = evidence or {}
    study = study or {}
    latest = latest_forecasts(preds)
    acc = running_accuracy(preds, settles)
    if direction_only:
        if not latest.empty:
            latest = latest[latest['model'].isin(direction_only)]
        if not acc.empty:
            acc = acc[acc['model'].isin(direction_only)]
    sess = ('<span class="badge on">сесията е отворена</span>' if state['in_session']
            else '<span class="badge off">сесията е затворена</span>')
    phase = 'dry_run'
    if not preds.empty and 'phase' in preds:
        phase = str(preds['phase'].iloc[-1])
    phase_note = ('пробен период — нищо не се зачита още' if phase == 'dry_run'
                  else 'зачита се')

    headline, headline_rows = _headline(latest, evidence)
    if not headline:
        headline = ('<div class="hl"><div class="when">няма прогноза</div>'
                    '<p class="note">Моделите с доказана стойност работят само в сесията '
                    f'({state["window"]} твое време). Извън нея тук е празно.</p></div>')

    rest = '<p class="empty">Още няма записана прогноза.</p>'
    if not latest.empty:
        latest = latest.assign(
            _rank=[0 if (r['model'], r['cadence'], int(r['horizon'])) in admitted else 1
                   for _, r in latest.iterrows()]
        ).sort_values(['_rank', 'horizon', 'model'])
        blocks = []
        for cad in ('M15', 'H1', 'D1'):
            part = latest[latest['cadence'] == cad]
            if part.empty:
                continue
            title = {'M15': '15-минутни барове', 'H1': 'Часови барове',
                     'D1': 'Дневни барове'}[cad]
            blocks.append(f'<h3>{title}</h3>\n<div class="grid">' +
                          ''.join(_cell_card(r, admitted, evidence) for _, r in part.iterrows()) +
                          '</div>')
        rest = ''.join(blocks)

    return f"""<!doctype html>
<html lang="bg"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>EUR/USD прогнози</title><style>{CSS}</style></head>
<body><div class="wrap">
{f'<p class="note"><a href="{html.escape(back_href)}">← към таблото</a></p>' if back_href else ''}
<h1>EUR/USD</h1>
<div class="sub">{generated} твое време · {sess} · {phase_note}</div>

{headline}

<h2>Колко да вярваш</h2>
<div class="card trust">{_trust(headline_rows, acc, study, n_req)}</div>

<details><summary>Всички останали модели и хоризонти</summary>
<p class="note">Записват се, но нито един от тях не е показал предимство над монета
на историята на този хоризонт, или е прекалено дълъг, за да го изтъргуваш в една
сесия. Тук са за пълнота.</p>
{rest}</details>

<details><summary>Точност по клетки — числата</summary>
{_accuracy_table(acc, admitted, n_req)}
<p class="note">За решение трябват <b>{_num(n_req)}</b> приключили прогнози на клетка.</p>
</details>

<details><summary>Състояние на записа</summary>
{_problems(gaps, fails)}
<p class="note">Записани прогнози: {_num(len(preds))} ·
приключили: {_num(len(settles))}</p></details>

{EXPLAINER}
<div class="sub" style="margin-top:22px">Описателно. Присъда се издава само при
регистрирания брой прогнози, след commit на предварителната регистрация.</div>
</div></body></html>"""


def build_page(out: str = FWD, record_path: str = RECORD, now=None, back_href=None) -> str:
    """The rendered page, with the no-cost guard applied. Shared by the file
    writer, the local viewer and the /forecasts route in api.py, so none of them
    can drift apart. `back_href` adds a link home when it is served inside the
    application; a standalone file or the :8001 viewer passes none."""
    admitted, alpha, n_req = admitted_cells(record_path)
    generated = owner_time(MD.label_now(now)).strftime('%d.%m.%Y %H:%M')
    page = render(_read('predictions', out), _read('settlements', out), _read('gaps', out),
                  _read('failures', out), admitted, alpha, n_req,
                  session_state(now), generated,
                  direction_only=direction_models(record_path), evidence=history_evidence(),
                  study=study_accuracy(), back_href=back_href)
    leaked = [w for w in COST_WORDS if w in page.lower()]
    if leaked:
        raise AssertionError(f'cost arithmetic reached the operator view: {leaked}')
    return page


def write_dashboard(out: str = FWD, path: str | None = None, record_path: str = RECORD,
                    now=None) -> str:
    path = path or os.path.join(out, 'dashboard.html')
    page = build_page(out, record_path, now)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(page)
    return path


# ── the local viewer (task 15.3) ────────────────────────────────────────────

DEFAULT_PORT = 8001


def make_handler(out: str = FWD, record_path: str = RECORD):
    """A request handler that re-renders the page on every GET, so a reload
    always shows the current logs rather than the last written file."""
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = 'EURUSDProphetForecasts/1.0'

        def do_GET(self):                                # noqa: N802 -- stdlib name
            if self.path.split('?')[0] not in ('/', '/index.html'):
                self.send_error(404, 'nothing here but /')
                return
            try:
                body = build_page(out, record_path).encode('utf-8')
            except Exception as e:                       # noqa: BLE001 -- report, never crash
                self.send_error(500, f'{type(e).__name__}: {e}')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):                               # noqa: N802 -- read-only viewer
            # Drain the body first: replying before the client has finished
            # sending makes Windows abort the connection instead of delivering
            # the 405.
            length = int(self.headers.get('Content-Length') or 0)
            while length > 0:
                length -= len(self.rfile.read(min(length, 65536)))
            self.send_error(405, 'this viewer only reads')

        def log_message(self, fmt, *args):
            pass                                         # keep the console readable

    return Handler


def serve(port: int = DEFAULT_PORT, out: str = FWD, record_path: str = RECORD,
          host: str = '127.0.0.1', serve_forever: bool = True):
    """A tiny read-only viewer, separate from the served application.

    Bound to 127.0.0.1 only: these are the owner's private forecasts and nothing
    here should be reachable from the network. It imports no part of `api.py`,
    `src/inference.py` or `src/paper_trading.py`, so the production dashboard
    stays exactly as it was (design Non-Goals).
    """
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer((host, int(port)), make_handler(out, record_path))
    url = f'http://{host}:{httpd.server_address[1]}/'
    print(f'прогнозите са на {url}   (Ctrl+C спира)', flush=True)
    if not serve_forever:
        return httpd, url
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('спрях')
    finally:
        httpd.server_close()
    return httpd, url


def main(argv=None):
    ap = argparse.ArgumentParser(description='The forward-log operator view.')
    ap.add_argument('--out', default=FWD)
    ap.add_argument('--serve', action='store_true',
                    help=f'run the local viewer on 127.0.0.1:{DEFAULT_PORT} instead of writing the file')
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--open', action='store_true', help='open a browser at the viewer')
    a = ap.parse_args(argv)
    if a.serve:
        if a.open:
            import threading
            import webbrowser
            threading.Timer(0.7, webbrowser.open,
                            [f'http://127.0.0.1:{a.port}/']).start()
        serve(a.port, a.out)
        return
    print(write_dashboard(a.out))


if __name__ == '__main__':
    main()
