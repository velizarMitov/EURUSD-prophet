# Gold descriptive statistics

Generated 2026-11-04 09:22 UTC by `python -m src.gold_window.describe` from `results/gold/gold_h1.csv` (2015-01-02 .. 2026-11-04 09:00) and `results/gold/gold_d1.csv`.

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

Hours are **Sofia** time, at which the H1 bar closes (a bar labelled h Berlin closes at h + 2 Sofia). Returns are log % per year of each span, on bid closes.

## Total by span

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | all hours %/yr | weekend gap %/yr |
|---|---|---|
| 2015-18 | +1.9 | +0.8 |
| 2019-22.10 | +6.5 | +3.0 |
| 2022.11-26 | +22.4 | +1.8 |

## By session

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | 2015-18 | 2019-22.10 | 2022.11-26 |
|---|---|---|---|
| Asia 01-09 | +15.4 | +6.4 | +13.3 |
| London 09-16 | -6.5 | -1.4 | +7.9 |
| window 16-23 | -3.6 | +1.1 | -0.4 |
| late 23-01 | -4.1 | -2.6 | -0.2 |

## By hour

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

**Bid-price limitation.** The cache holds bid closes only, and MT5 keeps one spread value per bar (0.00 for 2015-2017, a flat 0.25 or 0.50 later), which cannot show the widening at the daily break. The fall into 00:00 and the rise into 02:00 Sofia may therefore be partly a bid artifact; only bid/ask data can separate the two.

| | return %/yr 2015-18 | return %/yr 2019-22.10 | return %/yr 2022.11-26 | mean abs move % 2015-18 | mean abs move % 2019-22.10 | mean abs move % 2022.11-26 | up % 2015-18 | up % 2019-22.10 | up % 2022.11-26 | bars |
|---|---|---|---|---|---|---|---|---|---|---|
| 00:00 | -3.9 | -2.8 | -0.5 | 0.065 | 0.075 | 0.085 | 40 | 41 | 46 | 2779 |
| 01:00 | -0.2 | +0.2 | +0.3 | 0.032 | 0.114 | 0.131 | 27 | 52 | 54 | 372 |
| 02:00 | +8.2 | +5.0 | +8.6 | 0.078 | 0.078 | 0.113 | 70 | 56 | 63 | 2484 |
| 03:00 | +2.0 | +1.3 | +1.1 | 0.076 | 0.083 | 0.119 | 52 | 51 | 53 | 3053 |
| 04:00 | +0.5 | -1.1 | +5.2 | 0.104 | 0.116 | 0.159 | 49 | 48 | 52 | 3056 |
| 05:00 | +0.7 | +1.5 | -0.5 | 0.109 | 0.130 | 0.185 | 51 | 52 | 50 | 3055 |
| 06:00 | +0.5 | -0.6 | -0.2 | 0.072 | 0.085 | 0.134 | 50 | 47 | 48 | 3055 |
| 07:00 | +0.2 | -1.1 | +0.8 | 0.060 | 0.071 | 0.101 | 49 | 48 | 49 | 3057 |
| 08:00 | +1.2 | -0.6 | +0.3 | 0.066 | 0.083 | 0.117 | 52 | 49 | 50 | 3057 |
| 09:00 | +2.0 | +2.0 | -2.0 | 0.087 | 0.105 | 0.149 | 53 | 51 | 51 | 3057 |
| 10:00 | -4.6 | +0.8 | +3.7 | 0.108 | 0.130 | 0.147 | 44 | 48 | 51 | 3057 |
| 11:00 | +0.6 | +1.7 | -0.0 | 0.120 | 0.140 | 0.154 | 51 | 53 | 52 | 3056 |
| 12:00 | -0.6 | +0.4 | +0.5 | 0.091 | 0.123 | 0.140 | 47 | 53 | 53 | 3055 |
| 13:00 | -0.6 | -1.0 | -0.8 | 0.091 | 0.112 | 0.118 | 50 | 50 | 48 | 3055 |
| 14:00 | -0.8 | -2.1 | +3.1 | 0.087 | 0.113 | 0.128 | 48 | 50 | 55 | 3055 |
| 15:00 | +0.0 | -1.6 | +2.6 | 0.113 | 0.131 | 0.148 | 50 | 51 | 51 | 3055 |
| 16:00 | -0.5 | +0.4 | -1.3 | 0.200 | 0.238 | 0.258 | 50 | 52 | 50 | 3055 |
| 17:00 | -3.0 | -0.9 | -1.4 | 0.196 | 0.227 | 0.265 | 49 | 50 | 51 | 3055 |
| 18:00 | +1.4 | +1.3 | -2.1 | 0.167 | 0.225 | 0.253 | 50 | 52 | 50 | 3056 |
| 19:00 | +0.7 | +3.1 | +1.1 | 0.127 | 0.161 | 0.185 | 51 | 53 | 51 | 3055 |
| 20:00 | -1.6 | -3.0 | -1.5 | 0.111 | 0.119 | 0.147 | 49 | 50 | 51 | 3055 |
| 21:00 | +0.2 | -0.1 | +1.8 | 0.117 | 0.119 | 0.139 | 49 | 49 | 51 | 3019 |
| 22:00 | +0.6 | +0.7 | -0.5 | 0.107 | 0.118 | 0.129 | 51 | 51 | 51 | 2993 |
| 23:00 | -1.7 | +0.0 | +2.3 | 0.092 | 0.106 | 0.131 | 48 | 51 | 50 | 2965 |

## By weekday (D1 close to close, mean % per day)

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | 2015-18 | 2019-22.10 | 2022.11-26 |
|---|---|---|---|
| Mon | -0.067 | +0.028 | +0.026 |
| Tue | -0.050 | +0.012 | +0.116 |
| Wed | +0.031 | +0.052 | +0.075 |
| Thu | +0.017 | +0.028 | +0.106 |
| Fri | +0.109 | +0.006 | +0.098 |

## By month (mean monthly log return %)

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean % | +3.62 | +0.76 | +0.68 | +1.72 | +0.02 | -0.98 | +1.17 | +1.79 | -1.17 | +0.87 | -0.82 | +2.15 |
| up years | 8/11 | 6/12 | 5/12 | 8/12 | 7/12 | 4/12 | 8/12 | 7/12 | 3/12 | 7/12 | 6/12 | 8/11 |

## By calendar year (%)

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | return % |
|---|---|
| 2015 | -10.5 |
| 2016 | +7.1 |
| 2017 | +12.5 |
| 2018 | -2.7 |
| 2019 | +18.2 |
| 2020 | +24.1 |
| 2021 | -5.8 |
| 2022 | +1.2 |
| 2023 | +12.1 |
| 2024 | +27.4 |
| 2025 | +62.5 |
| 2026 | -9.3 |

## Concentration in the best days (D1, log %)

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | value |
|---|---|
| total | +120.1 |
| without best 10 | +76.8 |
| without best 25 | +30.0 |
| without best 50 | -31.2 |
| without best 100 | -130.6 |
| share of days with abs r > 2 % | +4.5 |
| big days up % | +49.3 |

## Size clustering: today by quintile of yesterday's absolute move

_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._

| | Q1 calm | Q2 | Q3 | Q4 | Q5 stormy |
|---|---|---|---|---|---|
| mean abs r today % | +0.669 | +0.698 | +0.685 | +0.685 | +0.813 |
| mean r today % | +0.045 | +0.013 | +0.044 | +0.013 | +0.081 |
