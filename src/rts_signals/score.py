"""One-shot scoring of the forward cell `rts_spot_net_r` (spec rts-signals/evaluation; design D8).

    python -m src.rts_signals.score

Uses the FIRST `min_trades` (300) eligible forward trades in report-date order, as pre-registered, so the
result does not depend on when the scoring runs. Fewer than 300 before the deadline (2027-06-30): it
refuses and writes nothing. On or after the deadline with fewer: `UNDERPOWERED — NO DECISION`. A second
scoring refuses and leaves the registry unchanged.

Statistic: mean net R. Test: fx_bb.score.test_ratio over business days keyed by report date (5-day
blocks, 10 000 resamples, seed 20261202), one-sided 'greater': KEEP only if the lower alpha-quantile > 0.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import CELL, COST_HEADER, config_digest, load_config, sha256_file
from .ledger import committed_label_day, forward_eligible
from .registry import PREREG_JSON, AlreadyScored, StageOrderError, append, guard, path, read_rows  # noqa: F401
from .rules import rules_digest

UNDER = 'UNDERPOWERED — NO DECISION'


def score(cfg: dict | None = None, today=None, gate=None) -> dict:
    from ..fx_bb.score import day_sums, test_ratio
    from .prereg import gate as prereg_gate
    cfg = cfg or load_config()
    guard(cfg)
    committed = committed_label_day((gate or prereg_gate)(cfg))
    if any(r['hypothesis'] == CELL for r in read_rows(cfg)):
        raise AlreadyScored(f'{CELL} is already scored; the cell is scored once')
    from .bars import label_now
    today = pd.Timestamp(today) if today is not None else label_now()
    if today.tzinfo is None:
        today = today.tz_localize('UTC')
    today = today.normalize()
    fw, sc = cfg['forward'], cfg['scoring']
    need = int(fw['min_trades'])
    deadline = pd.Timestamp(fw['deadline'], tz='UTC')
    d = forward_eligible(cfg, committed)
    n = len(d)
    if n < need and today < deadline:
        raise StageOrderError(f'scoring waits for {need} forward trades (have {n}) or the deadline {fw["deadline"]}')
    base = {'n': 1, 'date': str(today.date()), 'hypothesis': CELL, 'alpha': sc['alpha'],
            'family_size': sc['family_size'], 'block_days': sc['block_days'], 'resamples': sc['resamples'],
            'seed': sc['seed'], 'config_digest': config_digest(cfg), 'rules_digest': rules_digest(),
            'prereg_sha256': sha256_file(path(cfg, PREREG_JSON))}
    if n < need:
        row = {**base, 'n_trades': n, 'verdict': UNDER,
               'notes': f'{n} forward trades at the deadline, fewer than {need}. {COST_HEADER}'}
    else:
        d = d.iloc[:need]
        dates = pd.to_datetime(d['report_date'])
        S, N = day_sums(dates, d['net_r'].astype(float).to_numpy())
        res = test_ratio(S, N, float(sc['alpha']), int(sc['block_days']), int(sc['resamples']), int(sc['seed']),
                         'greater')
        row = {**base, 'n_trades': need, 'first_day': str(dates.min().date()), 'last_day': str(dates.max().date()),
               'mean_net_r': res['statistic'], 'bound': res['bound'], 'boot_se': res['boot_se'],
               'mde_80': res['mde_80'], 'verdict': res['verdict'],
               'notes': f'First {need} forward RTS spot base trades after the pre-registration commit. A DROP means '
                        f'no detectable edge net of spread, not proof of none. {COST_HEADER}'}
    append(cfg, [row])
    return row


def main(argv=None):
    argparse.ArgumentParser(description='score the rts_spot_net_r cell, once').parse_args(argv)
    r = score()
    print({k: r.get(k) for k in ('hypothesis', 'n_trades', 'mean_net_r', 'bound', 'verdict')})


if __name__ == '__main__':
    main()
