# Spec Delta

## MODIFIED Requirements

### Requirement: Navigation shows where the owner is
The navigation SHALL link to Home, Gold, Gold analytics, Forecasts, History, Paper
trading and the research console. It SHALL mark the current page as current, and the
less-used views (H1 direction, Kronos volatility, Second opinion) SHALL be under one
"More" menu. A retired view SHALL not be listed.

#### Scenario: On the gold page
- **WHEN** `/gold` is open
- **THEN** the Gold item is marked `aria-current="page"` and every other item is a plain link

#### Scenario: Retired view
- **WHEN** the navigation is rendered
- **THEN** it has no link to `/kronos-direction`

#### Scenario: Second opinion
- **WHEN** `/second-opinion` is open
- **THEN** the "More" menu lists it and marks it `aria-current="page"`

#### Scenario: Gold analytics
- **WHEN** `/gold-analytics` is open
- **THEN** the main navigation lists it after Gold and marks it `aria-current="page"`
