"""RTS's rule book as data, its clock and its plain-language README (spec rts-signals/rule-book; design D6).

    python -m src.rts_signals.rules --readme     rebuild results/rts_signals/README.md

Every number the simulation uses comes from rules.json. Its digest goes into the pre-registration, so an
edited number makes every result built under the old digest stale.

Clock: RTS speaks New York time (close 16:00, broker open 18:00, Friday spot close 12:00). Bars in this
repo carry Berlin wall-clock labels stored as if UTC, so a New York time is converted per date with
zoneinfo. That handles the US/EU daylight-saving mismatch weeks: 18:00 New York is the label 00:00 of the
next day in a normal week, and 23:00 in the late-October and March mismatch weeks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import pandas as pd

from . import PKG, load_config, results_dir

RULES_PATH = os.path.join(PKG, 'rules.json')
NY = 'America/New_York'
LABEL = 'Europe/Berlin'


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def rules_digest(rules: dict | None = None) -> str:
    rules = rules if rules is not None else load_rules()
    blob = json.dumps(rules, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def ny_to_label(day, hhmm: str) -> pd.Timestamp:
    """New York wall time `hhmm` on calendar `day` as a Berlin bar label (naive Berlin, stored as UTC)."""
    h, m = (int(x) for x in hhmm.split(':'))
    t = pd.Timestamp(pd.Timestamp(day).date()).replace(hour=h, minute=m).tz_localize(NY)
    return pd.Timestamp(t.tz_convert(LABEL).replace(tzinfo=None)).tz_localize('UTC')


def next_weekday(day) -> pd.Timestamp:
    d = pd.Timestamp(pd.Timestamp(day).date()) + pd.Timedelta(days=1)
    while d.dayofweek >= 5:
        d += pd.Timedelta(days=1)
    return d


def session(report_date, rules: dict | None = None, spot: bool = True) -> dict:
    """The session a report dated `report_date` (New York calendar day) is for.

    open  = 18:00 New York on the report date (over a weekend the first bar after it is Sunday's open);
    end   = 16:00 New York on the next weekday, or 12:00 when that weekday is a Friday in spot;
    trade_date = that next weekday, which is how RTS dates the trade in its track record."""
    rules = rules or load_rules()
    ck = rules['clock']
    trade = next_weekday(report_date)
    friday = spot and trade.dayofweek == 4
    return {'report_date': pd.Timestamp(pd.Timestamp(report_date).date()), 'trade_date': trade,
            'open': ny_to_label(report_date, ck['open_ny']),
            'end': ny_to_label(trade, ck['friday_spot_close_ny'] if friday else ck['close_ny']),
            'close_4pm': ny_to_label(report_date, ck['close_ny']), 'friday_close': friday}


# -- README ----------------------------------------------------------------------------------------

def readme(rules: dict | None = None) -> str:
    r = rules or load_rules()
    r0, r2, r3, r4, r5, ck = r['rule0'], r['rule2'], r['rule3'], r['rule4'], r['rule5'], r['clock']
    p = r0['proportionality_points']
    sections = [
        ('Правило 0 — вход на отварянето / Rule 0 — enter at the open',
         f"Ако новият вход е равен на цената в {ck['close_ny']} Ню Йорк, влизаш при отварянето на брокера "
         f"({ck['open_ny']} Ню Йорк). Стопът и целта са тези от таблицата.",
         f"When the new entry equals the {ck['close_ny']} New York price, enter at the broker open "
         f"({ck['open_ny']} New York) with the stop and target from the table.",
         'Пример / Example: CAD BUY 14374, стоп/stop 14339, цел/target 14402 → купува се USDCAD при отварянето, '
         'стоп 1.4339, цел 1.4402.'),
        ('Пропорционалност / Proportionality',
         f"Ако отварянето е на {p} или повече точки от затварянето, стопът и целта запазват разстоянието си в "
         'точки, но от реалното отваряне.',
         f"If the open is {p} points or more from the close, keep the stop and target distances in points, measured "
         'from the actual open.',
         'Пример / Example: EURUSD затваря/closes 1.0900, продажба/sell, стоп/stop 1.0930 (30 т.), отваря/opens '
         '1.0920 → вход/entry 1.0920, стоп/stop 1.0950.'),
        ('Правило 1 — партизански вход / Rule 1 — guerilla entry',
         'Входът е различен от затварянето. Поръчката се чака след отварянето; ако цената не стигне нивото до '
         'края на сесията, сделка няма.',
         'The entry differs from the close. It is filled only if the price touches it after the open; if it is '
         'never touched before the session ends, there is no trade.',
         'Пример / Example: EURO SELL, затваряне/close 10975, вход/entry 11009 → продажба само ако EURUSD стигне '
         '1.1009.'),
        ('Правило 2 — бягащ ход / Rule 2 — runaway move',
         f"Ако след печалба (или от затварянето, когато партизанският вход не е достигнат) цената продължи още "
         f"{r2['trigger_points']} точки в същата посока, влизаш отново в същата посока със стоп {r2['stop_points']} "
         f"и цел {r2['target_points']} точки. Никога в петък при спот.",
         f"If after a profit (or from the close when the guerilla is not reached) the price runs another "
         f"{r2['trigger_points']} points the same way, re-enter the same way with stop {r2['stop_points']} and "
         f"target {r2['target_points']} points. Never on a Friday in spot.",
         'Пример / Example: продажба с цел 1.0866 е изпълнена; цената пада до 1.0826 → нова продажба 1.0826, '
         'стоп 1.0866, цел 1.0766.'),
        ('Правило 3 — възстановяване / Rule 3 — recovery',
         f"След стоп, ако цената се върне до входа, влизаш пак със същия стоп и цел. Ако някое разстояние е "
         f"отрицателно или до {r3['min_distance_points']} точки, ползваш стоп {r3['fallback_stop_points']} и цел "
         f"{r3['fallback_target_points']}. Най-много {r3['max_losses_spot']} загуби на една сделка. Не важи при NPR.",
         f"After a stop, if the price returns to the entry, re-enter with the same stop and target. If a distance is "
         f"negative or {r3['min_distance_points']} points or less, use stop {r3['fallback_stop_points']} and target "
         f"{r3['fallback_target_points']}. At most {r3['max_losses_spot']} losses on one trade. Not under NPR.",
         'Пример / Example: покупка 1.3090 спряна на 1.3048; цената се връща на 1.3090 → нова покупка 1.3090 '
         'със същите нива.'),
        ('Правило 4 — обръщане / Rule 4 — reversal',
         f"Ако цената тръгне {r4['trigger_points']} точки срещу посоката (след стоп/цел или от затварянето), "
         f"обръщаш със стоп {r4['stop_points']} и цел {r4['target_points']} точки.",
         f"If the price moves {r4['trigger_points']} points against the direction (after a stop or target, or off "
         f"the close), reverse with stop {r4['stop_points']} and target {r4['target_points']} points.",
         'Пример / Example: посока BUY, затваряне 1.3100; цената пада до 1.3020 → продажба 1.3020, стоп 1.3064, '
         'цел 1.2972.'),
        ('Правило 5 — обратен гап / Rule 5 — opposite gap',
         f"Ако отварянето е на {r5['gap_points']} точки срещу посоката, съветът не се изпълнява. Ако цената "
         f"продължи още {r5['extra_points']} точки, може да обърнеш.",
         f"If the open gaps {r5['gap_points']} points against the direction, do not follow the advice. If the price "
         f"goes another {r5['extra_points']} points, you may reverse.",
         'Пример / Example: BUY, затваряне 1.1000, отваряне 1.0960 → без покупка; при 1.0940 → продажба.'),
        ('NPR — без препоръчана позиция / No position recommended',
         'NPR в колоната за вход: само сделки по Правила 2 и 4. NPR и в посоката: никакви сделки.',
         'NPR in the entry column: only Rule 2 and Rule 4 trades. NPR in the direction too: no trades at all.',
         'Пример / Example: 11/27 всички двойки NPR → базова сделка няма.'),
        ('Затваряне в петък / Friday close',
         f"При спот всички позиции се затварят в петък в {ck['friday_spot_close_ny']} Ню Йорк. Правило 2 не се "
         'ползва в петък.',
         f"In spot every position is closed on Friday at {ck['friday_spot_close_ny']} New York. Rule 2 is not used "
         'on a Friday.',
         'Пример / Example: отворена в четвъртък сделка, не достигнала нито стоп, нито цел, излиза в петък 12:00 '
         'Ню Йорк = 18:00 Берлин (19:00 София).'),
    ]
    L = ['# RTS сигнали — правилата / RTS signals — the rules', '',
         '> **Резултатите на RTS са хипотетични и брутни (без спред). / RTS\'s results are hypothetical and gross '
         'of cost.**', '',
         'Генерирано от `src/rts_signals/rules.json` (не редактирай на ръка). / Generated from '
         '`src/rts_signals/rules.json` (do not edit by hand).', '',
         f"Версия на правилата / Rules version: {r['version']}. Digest: `{rules_digest(r)}`.", '',
         '## Как се четат отчетите / How the reports read', '',
         '- **Morning**: само информация, дългосрочни позиции; не са сделки. / information only, long-term '
         'positions; not trades.',
         '- **Spot Afternoon** (4 pm Ню Йорк): посока, вход, стоп и цел за следващата сесия. / direction, entry, '
         'stop and target for the next session.',
         '- **Futures** (CME): не се четат — търгува се само спот. / not read — spot only.',
         '- Цени / Prices: 13134 = 1.3134; YEN 15898 = 158.98. YEN, SWISS и CAD са USDJPY, USDCHF, USDCAD.', '',
         '## Часовник / Clock', '',
         f"Затваряне {ck['close_ny']}, отваряне {ck['open_ny']} Ню Йорк. В нормална седмица 18:00 Ню Йорк = 00:00 "
         'Берлин (01:00 София); в седмиците с разминаване на лятното часово време = 23:00 Берлин.', '']
    for title, bg, en, ex in sections:
        L += [f'## {title}', '', f'- {bg}', f'- {en}', f'- {ex}', '']
    L += ['## Какво проверява този проект / What this project checks', '',
          '- **Базова сделка / Base trade**: само Правило 0 (с пропорционалност) или Правило 1, стоп/цел, иначе '
          'изход в 16:00 Ню Йорк (петък 12:00). Брутно и нето от записания спред. / Rule 0 (with proportionality) '
          'or Rule 1 only; stop/target, else exit at 16:00 New York (Friday 12:00). Gross and net of the recorded '
          'spread.',
          '- Една свещ, докоснала и стопа, и целта, се брои за стоп. / A bar touching both counts as the stop.',
          '- **Пълни правила / Full rules** (2–5): само описателно сравнение с отчета на RTS. / descriptive '
          'reconciliation with RTS\'s own record only.', '']
    return '\n'.join(L)


def write_readme(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    p = os.path.join(results_dir(cfg), 'README.md')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(readme())
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description='RTS rule book')
    ap.add_argument('--readme', action='store_true', help='rebuild results/rts_signals/README.md')
    ap.add_argument('--digest', action='store_true', help='print the rules digest')
    a = ap.parse_args(argv)
    if a.readme:
        print(write_readme())
    if a.digest or not a.readme:
        print(rules_digest())


if __name__ == '__main__':
    main()
