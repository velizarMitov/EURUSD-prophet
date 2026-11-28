"""One-shot scoring of the forward cell (spec gold-nyrange-forward/forward-evaluation; design D7).

    python -m src.gold_nyfwd.score

Forward days: settled `ok` days whose label date is after the pre-registration's commit date on the
Berlin label clock. The cell uses the FIRST `min_days` (60) of them, as pre-registered, so the result does
not depend on when the scoring is run. With fewer than 60 before the deadline (2027-04-30), the scoring
refuses and writes nothing; on or after the deadline with fewer than 60, the cell is recorded
`UNDERPOWERED — NO DECISION`. A second scoring refuses (AlreadyScored) and leaves the registry unchanged.

Statistic: per day se_harx - se_har; fx_fib's test_cell (circular 5-day block bootstrap, 10 000 resamples,
seed 20261201), KEEP when the upper (1 - alpha) bound is below 0. Gross, not a trading result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from . import CELL, GROSS_HEADER, config_digest, load_config, sha256_file
from .registry import PREREG_JSON, AlreadyScored, StageOrderError, append, guard, path, read_rows  # noqa: F401

UNDER = 'UNDERPOWERED — NO DECISION'


def committed_label_day(iso: str) -> pd.Timestamp:
    from .feed import label_now
    return label_now(pd.Timestamp(iso).tz_convert('UTC')).normalize()


def forward_days(cfg: dict, committed: pd.Timestamp) -> pd.DataFrame:
    from .live import read_log
    st = read_log(cfg, 'settlements')
    if not len(st):
        return st
    st = st[(st['status'] == 'ok') & (st['date'] > committed)].sort_values('date')
    return st


def score(cfg: dict | None = None, today=None, gate=None) -> dict:
    from ..fx_fib.score import test_cell
    from .prereg import gate as prereg_gate
    cfg = cfg or load_config()
    guard(cfg)
    committed = committed_label_day((gate or prereg_gate)(cfg))
    if any(r['hypothesis'] == CELL for r in read_rows(cfg)):
        raise AlreadyScored(f'{CELL} is already scored; the cell is scored once')
    from .feed import label_now
    today = pd.Timestamp(today) if today is not None else label_now()
    if today.tzinfo is None:
        today = today.tz_localize('UTC')
    today = today.normalize()
    fw, sc = cfg['forward'], cfg['scoring']
    need = int(fw['min_days'])
    deadline = pd.Timestamp(fw['deadline'], tz='UTC')
    d = forward_days(cfg, committed)
    n = len(d)
    if n < need and today < deadline:
        raise StageOrderError(f'scoring waits for {need} forward days (have {n}) or the deadline {fw["deadline"]}')
    base = {'n': 1, 'date': str(today.date()), 'hypothesis': CELL, 'alpha': sc['alpha'],
            'family_size': sc['family_size'], 'block_days': sc['block_days'], 'resamples': sc['resamples'],
            'seed': sc['seed'], 'config_digest': config_digest(cfg), 'prereg_sha256': sha256_file(path(cfg, PREREG_JSON))}
    if n < need:
        row = {**base, 'n_days': n, 'verdict': UNDER,
               'notes': f'{n} forward days at the deadline, fewer than {need}. {GROSS_HEADER}'}
    else:
        d = d.iloc[:need]
        diff = (d['se_harx'].astype(float) - d['se_har'].astype(float)).to_numpy()
        res = test_cell(diff, float(sc['alpha']), int(sc['block_days']), int(sc['resamples']), int(sc['seed']))
        mx, mh = float(d['se_harx'].astype(float).mean()), float(d['se_har'].astype(float).mean())
        row = {**base, 'n_days': need, 'first_day': str(d['date'].min().date()), 'last_day': str(d['date'].max().date()),
               'mse_harx': mx, 'mse_har': mh, 'mse_ratio': mx / mh,
               **{k: res[k] for k in ('mean_diff', 'ci_upper', 'boot_se', 'mde_80', 'verdict')},
               'notes': f'Frozen gold_bbnr OLS HAR-X vs HAR on the first {need} live forward days after the '
                        f'pre-registration commit. Expected power about 34 %; a DROP is not evidence that the model '
                        f'fails. {GROSS_HEADER}'}
    append(cfg, [row])
    return row


def main(argv=None):
    argparse.ArgumentParser(description='score the gold-nyrange-forward cell, once').parse_args(argv)
    r = score()
    print({k: r.get(k) for k in ('hypothesis', 'n_days', 'mse_ratio', 'mean_diff', 'ci_upper', 'verdict')})


if __name__ == '__main__':
    main()
