"""The label and power report, before any outcome (spec gold-bb-nyrange/evaluation; design D6).

    python -m src.gold_bbnr.labels      write results/gold_bbnr/labels.md and labels.json

Uses NO event outcome, NO exit scan, NO target of any day and NO model output:

  Part 1a  events after cooldown; exclusions `no outcome` (no t+3), `warm-up`, `before first fold`,
           `thin fold` (from event and t+3 DATES only); dispersion = SD of the 3-day ATR move over ALL
           valid days (a property of the price series, never of the events).
  Part 1b  candidate entries = every first breach. The one-open-trade rule depends on exit times, which
           are only known after the gate, so the counts are UPPER BOUNDS; thin folds use the t+10 date in
           place of the exit date (the latest possible exit). Declared SD 1.0 R.
  Part 2   eligible days and `warm-up` from bar presence alone (the HAR lags use a placeholder), days
           from the first fold, the first training size; declared SD 1.0 of the squared-error difference.

    rule cells   MDE = (z(1 - alpha) + z(power)) * SD / sqrt(n)
    model cells  MDE accuracy = 0.5 + (z(1 - alpha) + z(power)) * 0.5 / sqrt(n)
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import GROSS_HEADER, config_digest, load_config
from . import bb_events as E
from . import bb_features as F
from . import bb_trades as T
from . import ny_features as NF
from .bars import valid
from .inputs import Inputs, cache_digests, dxy_gaps, load
from .registry import LABELS_JSON, LABELS_MD, guard, path, write_json
from .splits import day_folds, event_folds


def z_sum(alpha: float, power: float) -> float:
    z = NormalDist().inv_cdf
    return z(1 - alpha) + z(power)


def mde_mean(sd: float, n: int, alpha: float, power: float) -> float:
    return z_sum(alpha, power) * sd / math.sqrt(n) if n else float('nan')


def mde_accuracy(n: int, alpha: float, power: float) -> float:
    return 0.5 + z_sum(alpha, power) * 0.5 / math.sqrt(n) if n else float('nan')


def sd_3d_move(d1: pd.DataFrame, cfg: dict) -> float:
    ind = E.indicators(valid(d1), cfg)
    h = int(cfg['horizon_days'])
    move = (ind['close'].shift(-h) - ind['close']) / ind['atr']
    return float(move.std(ddof=1))


def _years(dates) -> dict:
    s = pd.Series(1, index=pd.DatetimeIndex(dates))
    return {int(k): int(v) for k, v in s.groupby(s.index.year).sum().items()}


def part_1a(inp: Inputs, cfg: dict) -> dict:
    ev, counts = E.detect(inp.d1, cfg)
    ev = F.with_features(ev, inp.d1, inp.dxy_d1, cfg)
    use = ev[ev['reason2'] == ''].sort_values('date').reset_index(drop=True)
    folds, fr = event_folds(use['date'], use['date_t3'], cfg)
    model = use[fr == '']
    rule = ev[ev['reason'] == '']
    sc = cfg['scoring']['bb']
    pw = cfg['scoring']['power']
    sd = sd_3d_move(inp.d1, cfg)
    return {'counts': counts, 'events': len(ev), 'rule_events': len(rule), 'rule_dates': int(rule['date'].nunique()),
            'no_outcome': int((ev['reason'] == 'no outcome').sum()), 'warm_up': int((ev['reason2'] == 'warm-up').sum()),
            'before_first_fold': int((fr == 'before first fold').sum()), 'thin_fold': int((fr == 'thin fold').sum()),
            'model_events': len(model), 'n_folds': len(folds),
            'folds': [{'k': f.k, 'start': str(f.start.date()), 'train': len(f.fit) + len(f.val), 'test': len(f.eval)}
                      for f in folds],
            'per_year_rule': _years(rule['date']), 'per_year_model': _years(model['date']), 'sd_3d_atr': sd,
            'mde_rule': mde_mean(sd, len(rule), sc['alpha'], pw),
            'mde_model_accuracy': mde_accuracy(len(model), sc['alpha'], pw)}


def part_1b(inp: Inputs, cfg: dict) -> dict:
    ev = T.entries(inp.d1, cfg)
    ev = F.with_features(ev.assign(date_t3=ev['date_t10']), inp.d1, inp.dxy_d1, cfg)
    cand = ev[ev['reason'] == '']
    use = ev[ev['reason2'] == ''].sort_values('date').reset_index(drop=True)
    folds, fr = event_folds(use['date'], use['date_t10'], cfg)
    model = use[fr == '']
    sc = cfg['scoring']['bb_sma']
    pw = cfg['scoring']['power']
    return {'candidates': len(ev), 'no_exit_data': int((ev['reason'] == 'no exit data').sum()),
            'warm_up': int((ev['reason2'] == 'warm-up').sum()), 'rule_upper_bound': len(cand),
            'before_first_fold': int((fr == 'before first fold').sum()), 'thin_fold': int((fr == 'thin fold').sum()),
            'model_upper_bound': len(model), 'n_folds': len(folds),
            'folds': [{'k': f.k, 'start': str(f.start.date()), 'train': len(f.fit) + len(f.val), 'test': len(f.eval)}
                      for f in folds],
            'per_year_candidates': _years(cand['date']), 'declared_sd_r': sc['declared_sd_r'],
            'mde_rule_at_upper_bound': mde_mean(sc['declared_sd_r'], len(cand), sc['alpha'], pw),
            'mde_model_accuracy_at_upper_bound': mde_accuracy(len(model), sc['alpha'], pw)}


def part_2(inp: Inputs, cfg: dict) -> dict:
    ds = NF.dataset(inp.h1, inp.d1, inp.dxy_h1, cfg, read_target=False)
    use = ds[ds['reason'] == '']
    folds = day_folds(use.index, cfg)
    scored = sum(len(f.eval) for f in folds)
    sc = cfg['scoring']['nyvol']
    rc = ds['reason'].value_counts().to_dict()
    return {'dates': len(ds), 'reasons': {k or 'usable': int(v) for k, v in rc.items()}, 'usable': len(use),
            'first_usable': str(use.index.min().date()) if len(use) else None, 'scored_days': scored,
            'n_folds': len(folds), 'first_train': (len(folds[0].fit) + len(folds[0].val)) if folds else 0,
            'per_year_scored': _years(use.index[use.index >= pd.Timestamp(cfg['folds_ny']['first_eval_start'], tz='UTC')]
                                      .tz_localize(None)),
            'data_hour_floored': int(ds.attrs.get('data_hour_floored', 0)),
            'mde_sd_units': mde_mean(sc['declared_sd_se_diff'], scored, sc['alpha'], cfg['scoring']['power'])}


def build(cfg: dict, inp: Inputs) -> dict:
    return {'config_digest': config_digest(cfg), 'cache_digests': cache_digests(cfg),
            'gold': {'h1_bars': len(inp.h1), 'first': str(inp.h1.index.min()), 'last': str(inp.h1.index.max()),
                     'd1_days': len(inp.d1), 'valid_days': int(inp.d1['valid'].sum())},
            'dxy': {'h1_bars': len(inp.dxy_h1), 'h1_first': str(inp.dxy_h1.index.min()), 'd1_days': len(inp.dxy_d1),
                    'gap_days': dxy_gaps(inp.dxy_h1, float(cfg['dxy']['gap_flag']))},
            'part_1a': part_1a(inp, cfg), 'part_1b': part_1b(inp, cfg), 'part_2': part_2(inp, cfg)}


def _f(x, n=3):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def render(rec: dict, cfg: dict, now_iso: str) -> str:
    a, b, c = rec['part_1a'], rec['part_1b'], rec['part_2']
    sc = cfg['scoring']
    L = ['# gold-bb-nyrange — labels and power, before any outcome', '', f'> **{GROSS_HEADER}**', '',
         f"Config digest `{rec['config_digest']}`. No outcome, exit, target or model output was read.", '',
         '## Data', '',
         f"- Gold H1: {rec['gold']['h1_bars']:,} bars, {rec['gold']['first']} .. {rec['gold']['last']}; "
         f"{rec['gold']['valid_days']:,} valid NY-close days.",
         f"- DXY H1 (MT5): {rec['dxy']['h1_bars']:,} bars from {rec['dxy']['h1_first']}; DXY daily (Yahoo): "
         f"{rec['dxy']['d1_days']:,} days.",
         f"- DXY days opening > {cfg['dxy']['gap_flag']:.0%} from the previous close: {len(rec['dxy']['gap_days'])} "
         f"({', '.join(d for d, _ in rec['dxy']['gap_days'][:12]) or 'none'}).", '',
         '## Part 1a — Bollinger reversion (alpha 0.05/2 per cell)', '',
         f"- First breaches {a['counts']['first_breaches']} ({a['counts']['up']} up, {a['counts']['down']} down); "
         f"removed by the 3-day cooldown {a['counts']['cooldown_removed']}; events {a['events']}.",
         f"- Rule cell: {a['rule_events']} events on {a['rule_dates']} dates (no outcome {a['no_outcome']}). "
         f"SD of the 3-day ATR move over all valid days {_f(a['sd_3d_atr'])}. "
         f"**Minimum detectable mean r: {_f(a['mde_rule'])} ATR.**",
         f"- Model cell: {a['model_events']} events (warm-up {a['warm_up']}, before 2018 {a['before_first_fold']}, "
         f"thin fold {a['thin_fold']}) in {a['n_folds']} folds. **Minimum detectable accuracy "
         f"{_f(a['mde_model_accuracy'])}.**",
         '', '| Fold start | Train | Test |', '|---|---|---|'] + \
        [f"| {f['start']} | {f['train']} | {f['test']} |" for f in a['folds']] + \
        ['', '## Part 1b — the SMA20 trade (alpha 0.05/2 per cell)', '',
         f"- Candidate entries (every first breach) {b['candidates']}; no exit data {b['no_exit_data']}. The "
         'one-open-trade rule removes more once exits are known, so these counts are **upper bounds**.',
         f"- Rule cell, at most {b['rule_upper_bound']} trades: **minimum detectable mean R ≥ "
         f"{_f(b['mde_rule_at_upper_bound'])} R** (declared SD {b['declared_sd_r']} R).",
         f"- Model cell, at most {b['model_upper_bound']} trades (warm-up {b['warm_up']}, before 2018 "
         f"{b['before_first_fold']}, thin fold {b['thin_fold']}, folds from the latest possible exit date): "
         f"**minimum detectable accuracy ≥ {_f(b['mde_model_accuracy_at_upper_bound'])}**.", '',
         '## Part 2 — the NY session range (alpha 0.05/3 per cell)', '',
         f"- Weekdays {c['dates']:,}; usable {c['usable']:,} from {c['first_usable']}; reasons {c['reasons']}.",
         f"- Scored days from 2018-01-01: {c['scored_days']:,} in {c['n_folds']} folds; first training set "
         f"{c['first_train']} days. **Minimum detectable mean squared-error difference: {_f(c['mde_sd_units'], 4)} "
         'SD of that difference.**',
         f"- Scored days per year: {c['per_year_scored']}.", '',
         f"One-sided alphas {sc['bb']['alpha']}, {sc['bb_sma']['alpha']}, {_f(sc['nyvol']['alpha'], 4)}; power "
         f"{sc['power']:.0%}.", '', f'_Generated {now_iso}._', '']
    return '\n'.join(L)


def write_report(cfg: dict | None = None, now_iso: str | None = None, inp: Inputs | None = None) -> str:
    cfg = cfg or load_config()
    guard(cfg)
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    inp = inp or load(cfg)
    rec = build(cfg, inp)
    rec['written_at'] = now_iso
    write_json(cfg, LABELS_JSON, rec)
    p = path(cfg, LABELS_MD)
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(render(rec, cfg, now_iso))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='write the gold-bb-nyrange label and power report').parse_args(argv)
    print(write_report())


if __name__ == '__main__':
    main()
