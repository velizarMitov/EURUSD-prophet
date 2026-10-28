# Spec Delta

## MODIFIED Requirements

### Requirement: An alert is logged, shown and notified once
Each alert SHALL be appended to `results/gold/forward/alerts.csv`, with the as-of bar,
the expected move and the threshold. The logger SHALL raise one Windows notification
per alert, and that notification SHALL stay on screen until the owner dismisses it.
`/gold` SHALL show a banner while the alert's hour is open, including on a page that was
opened before the alert was written. If the notification fails, the alert SHALL still
be logged.

#### Scenario: Two passes on the same bar
- **WHEN** a second pass sees the same as-of bar
- **THEN** no second alert or notification is produced

#### Scenario: Notifications unavailable
- **WHEN** the Windows notification call fails
- **THEN** the alert row is still written and the failure is recorded in the health row

#### Scenario: Notification is not dismissed
- **WHEN** an alert's notification is raised and the owner does not touch it
- **THEN** it stays on screen with its dismiss button and does not disappear after a few seconds

## ADDED Requirements

### Requirement: An open page picks up a new alert
An open `/gold` page SHALL re-read the alert at least once a minute, whether or not its
tab is in view, and SHALL re-read it at once when its tab comes back into view. A banner
SHALL disappear the same way once its hour has closed.

#### Scenario: Page opened before the alert
- **WHEN** the owner opens `/gold` at 13:00 Sofia and the logger writes an alert at 15:03 Sofia
- **THEN** the banner appears on that page by 15:04 Sofia without a reload

#### Scenario: Owner returns to the tab
- **WHEN** the `/gold` tab has been in the background and the owner switches back to it
- **THEN** the page re-reads the alert immediately

#### Scenario: Hour closes
- **WHEN** the alerted hour has closed and no newer alert is open
- **THEN** the banner is hidden within a minute

### Requirement: The banner names the hour and the size
The banner SHALL stand apart from the page's standing notices. It SHALL give the
expected size in percent and, when the close the alert was raised on is known, in
dollars at that close. It SHALL name the alerted hour in Sofia time, never on the
bar-label clock, and SHALL say that the direction is unknown, in BG and EN.

#### Scenario: Alert on the 16:00 bar
- **WHEN** an alert of ±0.1954 % is open for as-of bar 2026-10-27 16:00 (label clock) at close 4063.36
- **THEN** the banner shows ±0.20 %, about ±7.94 $, the hour 18:00–19:00 Sofia time, and that the direction is unknown

#### Scenario: Close not known
- **WHEN** the newest h1_lstm row shown is not the alert's as-of bar
- **THEN** the banner shows the size in percent only

#### Scenario: Banner text
- **WHEN** the banner is shown in either language
- **THEN** it contains no buy, sell, entry, stop or target wording

### Requirement: The tab title carries the alert
While a banner is shown, the page's browser tab title SHALL begin with a warning sign and
the expected size. The title SHALL return to its plain form when the banner is hidden.

#### Scenario: Alert open
- **WHEN** an alert of ±0.20 % is open and the owner looks at another tab
- **THEN** the `/gold` tab title begins "⚠ Large move ±0.20 %" (BG: "⚠ Голямо движение ±0,20 %")

#### Scenario: Alert closed
- **WHEN** the banner is hidden
- **THEN** the tab title no longer carries the warning sign
