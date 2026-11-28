# RTS сигнали — правилата / RTS signals — the rules

> **Резултатите на RTS са хипотетични и брутни (без спред). / RTS's results are hypothetical and gross of cost.**

Генерирано от `src/rts_signals/rules.json` (не редактирай на ръка). / Generated from `src/rts_signals/rules.json` (do not edit by hand).

Версия на правилата / Rules version: 2017-01-30. Digest: `8e3226661f386a9f578b92ff19c428d4ad20432dbb4c9b4d7bff62cabf4c5daa`.

## Как се четат отчетите / How the reports read

- **Morning**: само информация, дългосрочни позиции; не са сделки. / information only, long-term positions; not trades.
- **Spot Afternoon** (4 pm Ню Йорк): посока, вход, стоп и цел за следващата сесия. / direction, entry, stop and target for the next session.
- **Futures** (CME): не се четат — търгува се само спот. / not read — spot only.
- Цени / Prices: 13134 = 1.3134; YEN 15898 = 158.98. YEN, SWISS и CAD са USDJPY, USDCHF, USDCAD.

## Часовник / Clock

Затваряне 16:00, отваряне 18:00 Ню Йорк. В нормална седмица 18:00 Ню Йорк = 00:00 Берлин (01:00 София); в седмиците с разминаване на лятното часово време = 23:00 Берлин.

## Правило 0 — вход на отварянето / Rule 0 — enter at the open

- Ако новият вход е равен на цената в 16:00 Ню Йорк, влизаш при отварянето на брокера (18:00 Ню Йорк). Стопът и целта са тези от таблицата.
- When the new entry equals the 16:00 New York price, enter at the broker open (18:00 New York) with the stop and target from the table.
- Пример / Example: CAD BUY 14374, стоп/stop 14339, цел/target 14402 → купува се USDCAD при отварянето, стоп 1.4339, цел 1.4402.

## Пропорционалност / Proportionality

- Ако отварянето е на 15 или повече точки от затварянето, стопът и целта запазват разстоянието си в точки, но от реалното отваряне.
- If the open is 15 points or more from the close, keep the stop and target distances in points, measured from the actual open.
- Пример / Example: EURUSD затваря/closes 1.0900, продажба/sell, стоп/stop 1.0930 (30 т.), отваря/opens 1.0920 → вход/entry 1.0920, стоп/stop 1.0950.

## Правило 1 — партизански вход / Rule 1 — guerilla entry

- Входът е различен от затварянето. Поръчката се чака след отварянето; ако цената не стигне нивото до края на сесията, сделка няма.
- The entry differs from the close. It is filled only if the price touches it after the open; if it is never touched before the session ends, there is no trade.
- Пример / Example: EURO SELL, затваряне/close 10975, вход/entry 11009 → продажба само ако EURUSD стигне 1.1009.

## Правило 2 — бягащ ход / Rule 2 — runaway move

- Ако след печалба (или от затварянето, когато партизанският вход не е достигнат) цената продължи още 40 точки в същата посока, влизаш отново в същата посока със стоп 40 и цел 60 точки. Никога в петък при спот.
- If after a profit (or from the close when the guerilla is not reached) the price runs another 40 points the same way, re-enter the same way with stop 40 and target 60 points. Never on a Friday in spot.
- Пример / Example: продажба с цел 1.0866 е изпълнена; цената пада до 1.0826 → нова продажба 1.0826, стоп 1.0866, цел 1.0766.

## Правило 3 — възстановяване / Rule 3 — recovery

- След стоп, ако цената се върне до входа, влизаш пак със същия стоп и цел. Ако някое разстояние е отрицателно или до 15 точки, ползваш стоп 38 и цел 44. Най-много 3 загуби на една сделка. Не важи при NPR.
- After a stop, if the price returns to the entry, re-enter with the same stop and target. If a distance is negative or 15 points or less, use stop 38 and target 44. At most 3 losses on one trade. Not under NPR.
- Пример / Example: покупка 1.3090 спряна на 1.3048; цената се връща на 1.3090 → нова покупка 1.3090 със същите нива.

## Правило 4 — обръщане / Rule 4 — reversal

- Ако цената тръгне 80 точки срещу посоката (след стоп/цел или от затварянето), обръщаш със стоп 44 и цел 48 точки.
- If the price moves 80 points against the direction (after a stop or target, or off the close), reverse with stop 44 and target 48 points.
- Пример / Example: посока BUY, затваряне 1.3100; цената пада до 1.3020 → продажба 1.3020, стоп 1.3064, цел 1.2972.

## Правило 5 — обратен гап / Rule 5 — opposite gap

- Ако отварянето е на 40 точки срещу посоката, съветът не се изпълнява. Ако цената продължи още 20 точки, може да обърнеш.
- If the open gaps 40 points against the direction, do not follow the advice. If the price goes another 20 points, you may reverse.
- Пример / Example: BUY, затваряне 1.1000, отваряне 1.0960 → без покупка; при 1.0940 → продажба.

## NPR — без препоръчана позиция / No position recommended

- NPR в колоната за вход: само сделки по Правила 2 и 4. NPR и в посоката: никакви сделки.
- NPR in the entry column: only Rule 2 and Rule 4 trades. NPR in the direction too: no trades at all.
- Пример / Example: 11/27 всички двойки NPR → базова сделка няма.

## Затваряне в петък / Friday close

- При спот всички позиции се затварят в петък в 12:00 Ню Йорк. Правило 2 не се ползва в петък.
- In spot every position is closed on Friday at 12:00 New York. Rule 2 is not used on a Friday.
- Пример / Example: отворена в четвъртък сделка, не достигнала нито стоп, нито цел, излиза в петък 12:00 Ню Йорк = 18:00 Берлин (19:00 София).

## Какво проверява този проект / What this project checks

- **Базова сделка / Base trade**: само Правило 0 (с пропорционалност) или Правило 1, стоп/цел, иначе изход в 16:00 Ню Йорк (петък 12:00). Брутно и нето от записания спред. / Rule 0 (with proportionality) or Rule 1 only; stop/target, else exit at 16:00 New York (Friday 12:00). Gross and net of the recorded spread.
- Една свещ, докоснала и стопа, и целта, се брои за стоп. / A bar touching both counts as the stop.
- **Пълни правила / Full rules** (2–5): само описателно сравнение с отчета на RTS. / descriptive reconciliation with RTS's own record only.
